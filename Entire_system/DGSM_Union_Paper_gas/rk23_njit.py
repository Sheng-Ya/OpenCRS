"""njit re-implementation of SciPy's RK23 driver for this model.

`scipy.integrate.solve_ivp(..., method="RK23")` spends ~30% of each simulation
inside its Python step loop, and every RHS evaluation crosses the
Python->numba boundary (66 boxed arguments).  Compiling the Bogacki-Shampine
loop itself removes both costs: the RHS becomes an njit-to-njit call.

The arithmetic here is a line-by-line transcription of scipy 1.14's
``_ivp/rk.py`` (``rk_step``, ``RungeKutta._step_impl``), ``_ivp/common.py``
(``select_initial_step``, ``norm``) and the ``t_eval=None`` path of
``_ivp/ivp.py``, so it reproduces the reference trajectory bit for bit.  Two
places need care:

* ``np.dot`` must be used for the 3- and 4-term stage combinations and for the
  error norm.  OpenBLAS accumulates those with vectorised/FMA kernels, so a
  hand-written loop is *not* bit-identical.  numba routes ``np.dot`` to the
  same BLAS, and was verified to agree with numpy exactly.
* ``np.linalg.norm`` is *not* interchangeable: numba implements it with
  ``dnrm2`` while numpy uses ``sqrt(ddot)``.  ``_norm`` below spells out the
  numpy form.

Only ``status`` and the final state are returned -- the caller uses nothing
else -- which also drops scipy's per-step accumulation of every intermediate
state (~140 MB per 60 s segment).
"""

import numpy as np
from numba import njit

from All_derivatives_njit import (
    model_derivatives_njit,
    _RHS_READ_KEYS,
    _RHS_WRITE_KEYS,
)

# scipy.integrate._ivp.rk module constants
SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0

# RK23.error_estimator_order == 2, and select_initial_step is called with
# order=error_estimator_order.
_ERROR_EXPONENT = -1 / 3
_INITIAL_STEP_EXPONENT = 1 / 3


@njit(error_model="numpy")
def _norm(x):
    """scipy.integrate._ivp.common.norm -- RMS norm, numpy's exact form."""
    return np.sqrt(np.dot(x, x)) / x.size ** 0.5


@njit(error_model="numpy")
def model_rhs(t, y, args):
    """`combined_system` from Samples_for_DGSM_Union, without the dict lookups.

    Keeps the circular-buffer bookkeeping identical: when the solver retries a
    rejected step it walks back in time, and the three samples written by the
    failed attempt are rewound and overwritten.
    """
    (i_array, j_array, all_time, buffer_limit, n_state, Input_Parameters,
     read_arrays, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time,
     write_arrays) = args

    i = np.int64(i_array[0])
    actual_index = i % buffer_limit

    if i > 1:
        latest_nonzero_index = (i - 1) % buffer_limit
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:
            num_removed = 3
            index = (actual_index - 3) % buffer_limit
            for j in range(num_removed):
                all_time[(index + j) % buffer_limit] = 0
        else:
            num_removed = 0
    else:
        num_removed = 0

    derivatives_all = model_derivatives_njit(
        t, y[:n_state], num_removed, i, buffer_limit, all_time,
        Input_Parameters, *read_arrays, cs_t1, cs_t2, knots_1, knots_2,
        exercise_start_time, *write_arrays,
    )

    all_time[(i - num_removed) % buffer_limit] = t
    i_array[0] = i - num_removed + 1
    j_array[0] = np.int64(j_array[0]) - num_removed + 1

    return derivatives_all


def make_rk23_driver(rhs):
    """Compile the RK23 loop around an njit ``rhs(t, y, args)``.

    Written as a factory so the stepper can be verified against scipy on a
    cheap test problem using exactly the code that runs in production.
    """

    @njit(error_model="numpy")
    def rk23_driver(t0, t_bound, y0, max_step, rtol, atol, args):
        n = y0.size

        B = np.empty(3)
        B[0] = 2 / 9
        B[1] = 1 / 3
        B[2] = 4 / 9

        E = np.empty(4)
        E[0] = 5 / 72
        E[1] = -1 / 12
        E[2] = -1 / 9
        E[3] = 1 / 8

        if t_bound != t0:
            direction = np.sign(t_bound - t0)
        else:
            direction = 1.0

        t = t0
        y = y0
        f = rhs(t, y, args)

        # --- common.select_initial_step -------------------------------------
        interval_length = np.abs(t_bound - t0)
        if interval_length == 0.0:
            h_abs = 0.0
        else:
            scale = atol + np.abs(y0) * rtol
            d0 = _norm(y0 / scale)
            d1 = _norm(f / scale)
            if d0 < 1e-5 or d1 < 1e-5:
                h0 = 1e-6
            else:
                h0 = 0.01 * d0 / d1
            h0 = min(h0, interval_length)
            y1 = y0 + h0 * direction * f
            f1 = rhs(t0 + h0 * direction, y1, args)
            d2 = _norm((f1 - f) / scale) / h0

            if d1 <= 1e-15 and d2 <= 1e-15:
                h1 = max(1e-6, h0 * 1e-3)
            else:
                h1 = (0.01 / max(d1, d2)) ** _INITIAL_STEP_EXPONENT

            h_abs = min(min(min(100 * h0, h1), interval_length), max_step)

        K = np.empty((4, n))
        status = 0

        while True:
            # --- base.OdeSolver.step, corner case ---------------------------
            if n == 0 or t == t_bound:
                t = t_bound
                break

            # --- RungeKutta._step_impl --------------------------------------
            min_step = 10 * np.abs(np.nextafter(t, direction * np.inf) - t)

            if h_abs > max_step:
                h_abs = max_step
            elif h_abs < min_step:
                h_abs = min_step

            step_accepted = False
            step_rejected = False
            too_small = False
            t_new = t
            y_new = y
            f_new = f

            while not step_accepted:
                if h_abs < min_step:
                    too_small = True
                    break

                h = h_abs * direction
                t_new = t + h

                if direction * (t_new - t_bound) > 0:
                    t_new = t_bound

                h = t_new - t
                h_abs = np.abs(h)

                # --- rk.rk_step ---
                K[0] = f
                dy = (K[0] * 0.5) * h
                K[1] = rhs(t + 0.5 * h, y + dy, args)
                dy = (K[0] * 0.0 + K[1] * 0.75) * h
                K[2] = rhs(t + 0.75 * h, y + dy, args)
                y_new = y + h * np.dot(K[:3].T, B)
                f_new = rhs(t + h, y_new, args)
                K[3] = f_new

                scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
                error_norm = _norm(np.dot(K.T, E) * h / scale)

                if error_norm < 1:
                    if error_norm == 0:
                        factor = MAX_FACTOR
                    else:
                        factor = min(MAX_FACTOR,
                                     SAFETY * error_norm ** _ERROR_EXPONENT)

                    if step_rejected:
                        factor = min(1.0, factor)

                    h_abs *= factor
                    step_accepted = True
                else:
                    h_abs *= max(MIN_FACTOR,
                                 SAFETY * error_norm ** _ERROR_EXPONENT)
                    step_rejected = True

            if too_small:
                status = -1
                break

            t = t_new
            y = y_new
            f = f_new

            if direction * (t - t_bound) >= 0:
                break

        return status, y

    return rk23_driver


_model_driver = make_rk23_driver(model_rhs)


def solve_rk23(updates, t_span, y0, n_state, buffer_limit, max_step, rtol,
               atol, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2,
               exercise_start_time):
    """Drop-in replacement for the `solve_ivp(..., method="RK23")` call.

    Returns ``(status, y_final)``; ``status`` is 0 on success and -1 when the
    required step size fell below the spacing between numbers, matching
    `OdeResult.status`.
    """
    read_arrays = tuple(updates[key] for key in _RHS_READ_KEYS)
    write_arrays = tuple(updates[key] for key in _RHS_WRITE_KEYS)

    args = (
        updates["i"], updates["j"], updates["all_time"], buffer_limit, n_state,
        Input_Parameters, read_arrays, cs_t1, cs_t2, knots_1, knots_2,
        exercise_start_time, write_arrays,
    )

    return _model_driver(float(t_span[0]), float(t_span[1]), y0, max_step,
                         rtol, atol, args)
