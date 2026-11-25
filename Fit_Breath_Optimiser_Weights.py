"""
Re-estimate the breath-optimiser weights (lambda1, lambda2, n, Pmax, Pmax_dot) after correcting the optimiser.

The corrected optimiser (Resp_Control_Breath_Optimiser_New.py) is fitted so that its optimal inspiratory and
expiratory times reproduce those of the previous optimiser used for the paper results
(Entire_system/DGSM_Union_Paper_gas/Resp_Control_Breath_Optimiser.py, commit 09f25c2) with the previous weights
from fixed_params.py, which were chosen to give literature breathing frequencies at given minute ventilation.
Both are evaluated with the simulator settings (tolerance 1e-4, bounds and VAflow range of minimise_breathing).

Pmax and Pmax_dot are restricted to healthy-adult ranges. lambda1, lambda2 and n have no measured counterpart and
are only bounded broadly.

Run from ODE_Exercise/. Uses all CPU cores; takes a few minutes.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, differential_evolution
from Resp_Control_Breath_Optimiser_New import objective as objective_new

# Previous optimiser, as used for the paper results (the root Resp_Control_Breath_Optimiser.py is now the corrected one)
_path = Path(__file__).resolve().parent / "Entire_system" / "DGSM_Union_Paper_gas" / "Resp_Control_Breath_Optimiser.py"
_spec = importlib.util.spec_from_file_location("Resp_Control_Breath_Optimiser_Paper", _path)
_previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_previous)
objective_old = _previous.objective

E_rs, R_rs, P_ao = 21.9, 3.02, 0.0
GV_dead, V0_dead = 0.1698, 0.1587
VA_rest = 0.0673
tolerance = 1e-4
bounds_t = [(0.5, 6), (0.5, 9)]                  # [t1, t2], as in minimise_breathing
OLD_WEIGHTS = [0.86, 0.489, 0.4, 50.0, 1000.0]   # lambda1, lambda2, n, Pmax, Pmax_dot (fixed_params.py)

# name: (low, high), searched on a log scale
SEARCH = {
    "lambda1": (1e-3, 1e3),
    "lambda2": (1e-3, 1e3),
    "n": (0.02, 20.0),
    "Pmax": (70.0, 150.0),       # maximal inspiratory pressure, healthy adults (cmH2O)
    "Pmax_dot": (400.0, 2000.0), # maximal rate of inspiratory pressure development, sniff (cmH2O/s)
}

# 22 log-spaced alveolar ventilations over the spline range (more points at rest and moderate exercise), plus VA_rest
VA_FIT = np.unique(np.r_[np.geomspace(0.04, 1.6, 22), VA_rest])


def solve(objective_fn, weights, VAflow, x0):
    """Optimal (t1, t2) at one alveolar ventilation"""
    VD = GV_dead * VAflow + V0_dead
    res = minimize(objective_fn, x0=np.array(x0), args=([*weights, E_rs, R_rs, P_ao], VAflow, VD, 0.001, tolerance),
                   method='nelder-mead', bounds=bounds_t)
    return res.x


def old_targets():
    """Previous optimiser's curve, generated as in minimise_breathing, then re-optimised exactly at each fit point"""
    VA_grid = np.linspace(0.04, 1.6, 200)
    curve = np.empty((len(VA_grid), 2))
    guess = (1.5, 1.85)
    for i, VAflow in enumerate(VA_grid):
        guess = curve[i] = solve(objective_old, OLD_WEIGHTS, VAflow, guess)
    return np.array([solve(objective_old, OLD_WEIGHTS, VAflow,
                           (np.interp(VAflow, VA_grid, curve[:, 0]), np.interp(VAflow, VA_grid, curve[:, 1])))
                     for VAflow in VA_FIT])


def mismatch(x, targets):
    """Mean squared log ratio of new to previous optimal (t1, t2); 10% too long and 10% too short count equally"""
    weights = 10.0 ** np.asarray(x)
    try:
        with np.errstate(all='ignore'):
            new = np.array([solve(objective_new, weights, VAflow, tgt) for VAflow, tgt in zip(VA_FIT, targets)])
        value = float(np.mean(np.log(new / targets) ** 2))
        return value if np.isfinite(value) else 1e3
    except Exception:
        return 1e3


if __name__ == "__main__":
    targets = old_targets()
    log_bounds = [(np.log10(lo), np.log10(hi)) for lo, hi in SEARCH.values()]

    # Global search, then local refinement within the same bounds
    res = differential_evolution(mismatch, log_bounds, args=(targets,), popsize=12, maxiter=100, tol=1e-7, seed=2,
                                 workers=-1, updating='deferred', polish=False, disp=True)
    res = minimize(mismatch, res.x, args=(targets,), method='nelder-mead', bounds=log_bounds,
                   options={'xatol': 1e-5, 'fatol': 1e-9, 'maxiter': 800})

    fitted = 10.0 ** res.x
    rounded = np.array([float(f"{w:.2g}") for w in fitted])  # two significant figures
    for label, w in (("fitted", fitted), ("rounded", rounded)):
        rms = np.sqrt(mismatch(np.log10(w), targets))
        print(f"{label:8s}", {k: round(float(v), 4) for k, v in zip(SEARCH, w)}, f"RMS relative mismatch {100 * rms:.1f}%")
