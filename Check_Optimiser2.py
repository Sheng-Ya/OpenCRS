"""
Check the corrected breath optimiser (Resp_Control_Breath_Optimiser_New.py): corrected inspiration,
tau = t2^2 / L brake, return to FRC, stable expiration formula. Cost: inspiratory work + lambda1 smoothness,
plus lambda2 x the resistive work of expiration, with lambda1, lambda2, n, Pmax, Pmax_dot refitted to the
old t1/t2 vs VAflow curve.

Dashed lines are the old optimiser used for the paper results (Entire_system/DGSM_Union_Paper_gas/
Resp_Control_Breath_Optimiser.py) with the fixed_params values at tolerance 1e-3, as run by Check_Optimiser.py
(t1 = t2 at ~56 L/min). The new optimiser uses the pipeline tolerance 1e-4.
"""
import importlib.util
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser_New import objective

# Old optimiser (the root Resp_Control_Breath_Optimiser.py is now the corrected one)
_path = Path(__file__).resolve().parent / "Entire_system" / "DGSM_Union_Paper_gas" / "Resp_Control_Breath_Optimiser.py"
_spec = importlib.util.spec_from_file_location("Resp_Control_Breath_Optimiser_Paper", _path)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
objective_old = _old.objective

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "lines.linewidth": 1.5,
})

# ------------------------------
# PARAMETERS
# ------------------------------
# lambda1, lambda2, n, Pmax, Pmax_dot
NEW_PARAMS = [0.37, 6.5, 0.42, 70.0, 2000.0]   # corrected optimiser, expiratory resistive work cost
OLD_PARAMS = [0.86, 0.489, 0.4, 50.0, 1000.0]  # fixed_params values, old optimiser
P_ao = 0.0

E_rs_nominal = 21.9
R_rs_nominal = 3.02
GV_dead = 0.1698
V0_dead = 0.1587
VA_rest = 0.0673

tolerance = 1e-4               # new optimiser, same as the DGSM/HM pipeline
OLD_TOLERANCE = 1e-3           # old optimiser, as in Check_Optimiser.py
bounds = [(0.5, 6), (0.5, 9)]  # [t1, t2], same as minimise_breathing in the pipeline


def sweep(objective_fn, lam, VA_values, E_values, R_values, initial_guess=(1.5, 1.85), tol=tolerance):
    """
    Optimal t1, t2 along a sequence of (VAflow, E_rs, R_rs), warm-starting each from the previous optimum
    """
    guess = np.array(initial_guess)
    t1 = np.empty(len(VA_values))
    t2 = np.empty(len(VA_values))
    for i, (VAflow, E_rs, R_rs) in enumerate(zip(VA_values, E_values, R_values)):
        required_params = [*lam, E_rs, R_rs, P_ao]
        VD = GV_dead * VAflow + V0_dead
        res = minimize(objective_fn, x0=guess, args=(required_params, VAflow, VD, 0.001, tol),
                       method='nelder-mead', bounds=bounds)
        t1[i], t2[i] = res.x
        guess = res.x
    return t1, t2


def plot_vs(x, new_t1, new_t2, old_t1, old_t2, xlabel, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, new_t1, label='t1 new', color='blue', s=10)
    plt.scatter(x, new_t2, label='t2 new', color='red', s=10)
    plt.plot(x, old_t1, '--', color='blue', alpha=0.6, label='t1 old')
    plt.plot(x, old_t2, '--', color='red', alpha=0.6, label='t2 old')
    plt.xlabel(xlabel)
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


# ------------------------------
# 1. Vary R_rs (VAflow at rest, E_rs nominal)
# ------------------------------
R_rs_values = np.linspace(1.51, 6.04, 120)
VA_values = np.full(len(R_rs_values), VA_rest)
E_values = np.full(len(R_rs_values), E_rs_nominal)

t1_new, t2_new = sweep(objective, NEW_PARAMS, VA_values, E_values, R_rs_values)
t1_old, t2_old = sweep(objective_old, OLD_PARAMS, VA_values, E_values, R_rs_values, tol=OLD_TOLERANCE)
plot_vs(R_rs_values, t1_new, t2_new, t1_old, t2_old, 'R_rs', 'Optimal t1 and t2 vs R_rs (VAflow rest, E_rs nominal)')

# ------------------------------
# 2. Vary E_rs (VAflow at rest, R_rs nominal)
# ------------------------------
E_rs_values = np.linspace(10.95, 43.8, 120)
VA_values = np.full(len(E_rs_values), VA_rest)
R_values = np.full(len(E_rs_values), R_rs_nominal)

t1_new, t2_new = sweep(objective, NEW_PARAMS, VA_values, E_rs_values, R_values)
t1_old, t2_old = sweep(objective_old, OLD_PARAMS, VA_values, E_rs_values, R_values, tol=OLD_TOLERANCE)
plot_vs(E_rs_values, t1_new, t2_new, t1_old, t2_old, 'E_rs', 'Optimal t1 and t2 vs E_rs (VAflow rest, R_rs nominal)')

# ------------------------------
# 3. E_rs x R_rs grid over the DGSM sampling range (VAflow at rest)
# ------------------------------
E_grid, R_grid = np.meshgrid(np.linspace(21.9 * 0.5, 21.9 * 1.2, 40), np.linspace(3.02 * 0.8, 3.02 * 1.5, 40))
t1_grid, t2_grid = sweep(objective, NEW_PARAMS, np.full(E_grid.size, VA_rest), E_grid.flatten(), R_grid.flatten())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Optimal t1 vs E_rs and R_rs")
plt.xlabel("E_rs")
plt.ylabel("R_rs")
plt.contourf(E_grid, R_grid, t1_grid.reshape(E_grid.shape), levels=20)
plt.colorbar(label="t1 (s)")
plt.subplot(1, 2, 2)
plt.title("Optimal t2 vs E_rs and R_rs")
plt.xlabel("E_rs")
plt.ylabel("R_rs")
plt.contourf(E_grid, R_grid, t2_grid.reshape(E_grid.shape), levels=20)
plt.colorbar(label="t2 (s)")
plt.tight_layout()
plt.show()

# ------------------------------
# 4. Vary VAflow (E_rs, R_rs nominal), same grid as minimise_breathing
# ------------------------------
VAflow_vals = np.linspace(0.04, 1.6, 200)
E_values = np.full(len(VAflow_vals), E_rs_nominal)
R_values = np.full(len(VAflow_vals), R_rs_nominal)

t1_new, t2_new = sweep(objective, NEW_PARAMS, VAflow_vals, E_values, R_values)
t1_old, t2_old = sweep(objective_old, OLD_PARAMS, VAflow_vals, E_values, R_values, tol=OLD_TOLERANCE)
VD = GV_dead * VAflow_vals + V0_dead
VEflow_new = 60 * (VAflow_vals + VD / (t1_new + t2_new))
VEflow_old = 60 * (VAflow_vals + VD / (t1_old + t2_old))

plt.figure(figsize=(10, 5))
plt.scatter(VEflow_new, t1_new, label='Inspiration time new', color='blue', s=10)
plt.scatter(VEflow_new, t2_new, label='Expiration time new', color='red', s=10)
plt.plot(VEflow_old, t1_old, '--', color='blue', alpha=0.6, label='Inspiration time old')
plt.plot(VEflow_old, t2_old, '--', color='red', alpha=0.6, label='Expiration time old')
plt.xlabel('Minute Ventilation (L/min)')
plt.ylabel('Time (s)')
plt.title('Optimal t1 and t2 vs VEflow')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(VEflow_new, t1_new / (t1_new + t2_new), color='black', label='new')
plt.plot(VEflow_old, t1_old / (t1_old + t2_old), '--', color='black', alpha=0.6, label='old')
plt.xlabel('Minute Ventilation (L/min)')
plt.ylabel('Ti / Ttot')
plt.title('Inspiratory fraction of the breath')
plt.grid(True)
plt.legend()
plt.show()

cs_t1 = CubicSpline(VAflow_vals, t1_new, bc_type="natural")
cs_t2 = CubicSpline(VAflow_vals, t2_new, bc_type="natural")

plt.figure(figsize=(10, 5))
plt.plot(VAflow_vals, cs_t1(VAflow_vals), label='t1 CubicSpline', color='blue', linewidth=2)
plt.plot(VAflow_vals, cs_t2(VAflow_vals), label='t2 CubicSpline', color='red', linewidth=2)
plt.scatter(VAflow_vals, t1_new, color='blue', s=5, label='Optimal t1')
plt.scatter(VAflow_vals, t2_new, color='red', s=5, label='Optimal t2')
plt.xlabel('VAflow (L/s)')
plt.ylabel('Time (s)')
plt.title('Optimal t1 and t2 vs VAflow (new)')
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------
# Summary at rest, moderate and heavy ventilation
# ------------------------------
print(f"{'VAflow':>7} | {'old t1':>6} {'old t2':>6} {'old Ti/T':>8} | {'new t1':>6} {'new t2':>6} {'new Ti/T':>8}")
for VA in (VA_rest, 0.45, 1.0):
    o1, o2 = np.interp(VA, VAflow_vals, t1_old), np.interp(VA, VAflow_vals, t2_old)
    n1, n2 = np.interp(VA, VAflow_vals, t1_new), np.interp(VA, VAflow_vals, t2_new)
    print(f"{VA:7.4f} | {o1:6.2f} {o2:6.2f} {o1 / (o1 + o2):8.2f} | {n1:6.2f} {n2:6.2f} {n1 / (n1 + n2):8.2f}")
