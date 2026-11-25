"""Fully compiled Bogacki-Shampine RK23 driver for the root exercise model.

The step controller mirrors SciPy's RK23 implementation, while the accepted
time/state trajectory is retained because Run_model_all_correct_order.py uses
it to construct restart conditions and returns it for plotting.
"""

from types import SimpleNamespace

import numpy as np
from numba import njit

from All_derivatives import bind_rk23_storage, model_derivatives_njit


SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
_ERROR_EXPONENT = -1 / 3
_INITIAL_STEP_EXPONENT = 1 / 3


@njit(error_model="numpy")
def _norm(x):
    """SciPy's RMS norm, using NumPy's dot-product accumulation order."""
    return np.sqrt(np.dot(x, x)) / x.size ** 0.5


@njit(error_model="numpy")
def model_rhs(t, y, args):
    """Combined model RHS without Python dictionary access or dispatch."""
    (
        i_array, j_array, all_time, buffer_limit, Input_Parameters,
        read_arrays, circular_write_arrays, plot_write_arrays,
        cs_t1, cs_t2, knots_1, knots_2,
    ) = args

    i = np.int64(i_array[0])
    j = np.int64(j_array[0])
    actual_index = i % buffer_limit

    if i > 1:
        latest_nonzero_index = (i - 1) % buffer_limit
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:
            num_removed = 3
            index = (actual_index - 3) % buffer_limit
            for offset in range(num_removed):
                all_time[(index + offset) % buffer_limit] = 0
        else:
            num_removed = 0
    else:
        num_removed = 0

    derivatives = model_derivatives_njit(
        t, y, num_removed, i, j, buffer_limit, all_time, Input_Parameters,
        read_arrays, circular_write_arrays, plot_write_arrays,
        cs_t1, cs_t2, knots_1, knots_2,
    )

    all_time[(i - num_removed) % buffer_limit] = t
    i_array[0] = i - num_removed + 1
    j_array[0] = j - num_removed + 1
    return derivatives


def make_rk23_driver(rhs):
    """Compile the RK23 loop around an ``njit`` RHS."""

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

        # Match solve_ivp's t/y result while keeping trajectory growth inside
        # compiled code. Capacity doubles only occasionally.
        capacity = 1024
        times = np.empty(capacity)
        states = np.empty((n, capacity))
        times[0] = t
        states[:, 0] = y
        count = 1

        K = np.empty((4, n))
        status = 0

        while True:
            if n == 0 or t == t_bound:
                t = t_bound
                break

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
                        factor = min(
                            MAX_FACTOR,
                            SAFETY * error_norm ** _ERROR_EXPONENT,
                        )

                    if step_rejected:
                        factor = min(1.0, factor)

                    h_abs *= factor
                    step_accepted = True
                else:
                    h_abs *= max(
                        MIN_FACTOR,
                        SAFETY * error_norm ** _ERROR_EXPONENT,
                    )
                    step_rejected = True

            if too_small:
                status = -1
                break

            t = t_new
            y = y_new
            f = f_new

            if count == capacity:
                new_capacity = capacity * 2
                new_times = np.empty(new_capacity)
                new_states = np.empty((n, new_capacity))
                new_times[:capacity] = times
                new_states[:, :capacity] = states
                times = new_times
                states = new_states
                capacity = new_capacity

            times[count] = t
            states[:, count] = y
            count += 1

            if direction * (t - t_bound) >= 0:
                break

        return status, times[:count], states[:, :count]

    return rk23_driver


_model_driver = make_rk23_driver(model_rhs)


def solve_rk23(
    updates, t_span, y0, buffer_limit, max_step, rtol, atol,
    Input_Parameters, cs_t1, cs_t2, knots_1, knots_2,
):
    """Run compiled RK23 and return the attributes used from ``OdeResult``."""
    read_arrays, circular_write_arrays, plot_write_arrays = bind_rk23_storage(
        updates
    )
    args = (
        updates["i"], updates["j"], updates["all_time"], buffer_limit,
        Input_Parameters, read_arrays, circular_write_arrays,
        plot_write_arrays,
        cs_t1, cs_t2, knots_1, knots_2,
    )

    status, times, states = _model_driver(
        float(t_span[0]), float(t_span[1]), y0, max_step, rtol, atol, args
    )
    if status == 0:
        message = "The solver successfully reached the end of the integration interval."
    else:
        message = "Required step size is less than spacing between numbers."

    return SimpleNamespace(
        status=int(status),
        message=message,
        t=times,
        y=states,
        success=(status == 0),
    )
