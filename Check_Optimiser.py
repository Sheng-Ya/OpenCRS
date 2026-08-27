# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.interpolate import CubicSpline
# from scipy.optimize import differential_evolution, minimize
# from Resp_Control_Breath_Optimiser import objective
#
# # ── Parameters ──────────────────────────────────────────────────────
# GV_dead  = 0.1698
# V0_dead  = 0.15131603
#
# bounds = [(0.4, 3), (0.4, 6)]   # [t1, t2]
# tolerance = 0.001
#
# required_params = [
#     0.3,        # lambda1
#     0.289,      # lambda2
#     0.90,       # n
#     1000,       # Pmax
#     10000,      # Pmax_dot
#     21.9 * 1.2, # E_rs  (+20%)
#     3.02 * 1.2, # R_rs  (+20%)  ← the problematic case
#     0.0,        # P_ao
# ]
#
# VAflow_vals = np.linspace(0.03, 1.6, 200)
# VD = GV_dead * VAflow_vals + V0_dead
#
# # ── Helper: objective wrapper for differential_evolution ────────────
# # differential_evolution passes (x,) then *args, so signature matches already.
#
# def obj_wrapper(x, required_params, VAflow, VD_vol, dt, tol):
#     """Wrapper that catches NaN/Inf and returns a large penalty."""
#     try:
#         val = objective(x, required_params, VAflow, VD_vol, dt, tol)
#         if not np.isfinite(val):
#             return 1e12
#         return val
#     except Exception:
#         return 1e12
#
#
# # ── Main optimisation loop ──────────────────────────────────────────
# optimal_t1 = np.full(len(VAflow_vals), np.nan)
# optimal_t2 = np.full(len(VAflow_vals), np.nan)
# optimal_cost = np.full(len(VAflow_vals), np.inf)
#
# prev_x = None  # for warm-starting
#
# for idx, VAflow in enumerate(VAflow_vals):
#     VD_vol = VD[idx]
#     args = (required_params, VAflow, VD_vol, 0.001, tolerance)
#
#     # ── Stage 1: differential_evolution (global) ────────────────────
#     de_kwargs = dict(
#         func=obj_wrapper,
#         bounds=bounds,
#         args=args,
#         strategy='best1bin',
#         maxiter=300,
#         tol=1e-8,
#         seed=42,
#         polish=True,          # Nelder-Mead polish at the end
#         init='latinhypercube',
#         popsize=20,
#     )
#
#     # Inject previous solution as a warm-start hint (SciPy >= 1.7)
#     if prev_x is not None:
#         de_kwargs['x0'] = prev_x
#
#     res = differential_evolution(**de_kwargs)
#
#     if res.success or np.isfinite(res.fun):
#         optimal_t1[idx] = res.x[0]
#         optimal_t2[idx] = res.x[1]
#         optimal_cost[idx] = res.fun
#         prev_x = res.x
#         print(f"[{idx:3d}] VAflow={VAflow:.4f}  t1={res.x[0]:.4f}  t2={res.x[1]:.4f}  cost={res.fun:.6e}")
#     else:
#         print(f"[{idx:3d}] VAflow={VAflow:.4f}  FAILED")
#
#
# # ── Stage 2: detect and repair jumps ────────────────────────────────
# JUMP_THRESHOLD = 0.15  # flag if t1 or t2 changes by more than this between neighbours
#
# dt1 = np.abs(np.diff(optimal_t1))
# dt2 = np.abs(np.diff(optimal_t2))
# jump_mask = (dt1 > JUMP_THRESHOLD) | (dt2 > JUMP_THRESHOLD)
#
# # Indices just after a jump (and one before) need re-checking
# suspect = set()
# for i in np.where(jump_mask)[0]:
#     suspect.update(range(max(0, i-1), min(len(VAflow_vals), i+3)))
#
# if suspect:
#     print(f"\n── Repairing {len(suspect)} suspect points with multi-start ──")
#
# for idx in sorted(suspect):
#     VAflow = VAflow_vals[idx]
#     VD_vol = VD[idx]
#     args = (required_params, VAflow, VD_vol, 0.001, tolerance)
#
#     best_cost = optimal_cost[idx]
#     best_x = np.array([optimal_t1[idx], optimal_t2[idx]])
#
#     # Build candidate seeds: neighbours + grid corners + centre
#     seeds = []
#     for offset in [-2, -1, 1, 2]:
#         j = idx + offset
#         if 0 <= j < len(VAflow_vals) and np.isfinite(optimal_t1[j]):
#             seeds.append([optimal_t1[j], optimal_t2[j]])
#     # Add a coarse grid of starting points
#     for t1_seed in np.linspace(bounds[0][0], bounds[0][1], 4):
#         for t2_seed in np.linspace(bounds[1][0], bounds[1][1], 4):
#             seeds.append([t1_seed, t2_seed])
#
#     for seed in seeds:
#         try:
#             res = minimize(obj_wrapper, x0=seed, args=args,
#                            method='Nelder-Mead',
#                            options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 2000})
#             if np.isfinite(res.fun) and res.fun < best_cost - 1e-12:
#                 best_cost = res.fun
#                 best_x = res.x
#         except Exception:
#             pass
#
#     if best_cost < optimal_cost[idx] - 1e-12:
#         print(f"  [{idx:3d}] REPAIRED: t1 {optimal_t1[idx]:.4f}->{best_x[0]:.4f}, "
#               f"t2 {optimal_t2[idx]:.4f}->{best_x[1]:.4f}, "
#               f"cost {optimal_cost[idx]:.4e}->{best_cost:.4e}")
#         optimal_t1[idx] = best_x[0]
#         optimal_t2[idx] = best_x[1]
#         optimal_cost[idx] = best_cost
#
#
# # ── Compute VEflow and plot ─────────────────────────────────────────
# T_total = optimal_t1 + optimal_t2
# VEflow = 60 * (VAflow_vals + VD / T_total)
#
# plt.figure(figsize=(10, 5))
# plt.scatter(VEflow, optimal_t1, label='Optimal Inspiration Time', color='blue', alpha=0.6, s=25)
# plt.scatter(VEflow, optimal_t2, label='Optimal Expiration Time', color='red', alpha=0.6, s=25)
# plt.xlabel('VE_Flow (L/min)')
# plt.ylabel('Time (s)')
# plt.title('Optimal t1 and t2 vs VEflow (differential_evolution + repair)')
# plt.legend()
# plt.show()

# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.interpolate import CubicSpline
# from scipy.optimize import minimize, NonlinearConstraint
# from check import Parameters as params
# from Resp_Control_Breath_Optimiser import objective

# bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
# tolerance = 0.001

# # ------------------------------
# # FIXED VALUES
# # ------------------------------
# VAflow = 0.0673               # constant VAflow
# E_rs_const = 21.9             # <- set to whatever E_rs you want to keep fixed
#
# # Compute VD once since VAflow is constant
# VD = params["GV_dead"] * VAflow + params["V0_dead"]
#
# # ------------------------------
# # Vary R_rs only
# # ------------------------------
# R_rs_values = np.linspace(1.51, 6.04, 500)
#
# optimal_t1 = []
# optimal_t2 = []
# initial_guess = [1.5, 1.85]
#
# for R_rs in R_rs_values:
#
#     required_params = [
#         0.3,
#         0.289,
#         0.9,
#         1000,
#         10000,
#         E_rs_const,  # <----- varies
#         R_rs,  # <----- varies
#         0.0,
#     ]
#
#     try:
#         res = minimize(
#             objective,
#             x0=np.array(initial_guess[-2:]),
#             args=(required_params, VAflow, VD, 0.001, tolerance),
#             method='nelder-mead',
#             bounds=bounds
#         )
#
#         if res.success:
#             t1_opt, t2_opt = res.x
#             optimal_t1.append(t1_opt)
#             optimal_t2.append(t2_opt)
#
#             # good new guess for next iteration
#             initial_guess.extend(res.x)
#
#             print(f"R_rs={R_rs:.2f} → t1={t1_opt:.3f}, t2={t2_opt:.3f}")
#
#         else:
#             print(f"R_rs={R_rs:.2f} → optimization failed")
#             optimal_t1.append(np.nan)
#             optimal_t2.append(np.nan)
#
#     except Exception as e:
#         print(f"R_rs={R_rs:.2f} → error: {e}")
#         optimal_t1.append(np.nan)
#         optimal_t2.append(np.nan)
#
# # Convert to arrays
# optimal_t1 = np.array(optimal_t1)
# optimal_t2 = np.array(optimal_t2)
#
# # ------------------------------
# # Plotting
# # ------------------------------
# plt.figure(figsize=(10, 5))
#
# plt.scatter(R_rs_values, optimal_t1, label='Optimal t1', marker='o', alpha=0.7, s=10)
# plt.scatter(R_rs_values, optimal_t2, label='Optimal t2', marker='s', alpha=0.7, s=10)
#
# plt.xlabel("R_rs")
# plt.ylabel("Time (s)")
# plt.title("Optimal t1 and t2 vs R_rs (VAflow & E_rs constant)")
# plt.grid(True)
# plt.legend()
# plt.show()

from matplotlib import pyplot as plt
# Set global style
plt.rcParams.update({
    "font.size": 15,  # Larger font
    # "font.weight": "bold",  # Bold text
    # "axes.labelweight": "bold",
    "axes.titlesize": 15,
    # "axes.titleweight": "bold",
    "legend.fontsize": 15,
    "lines.linewidth": 1,  # Thicker lines
})
#
#
# bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
# tolerance = 0.001
#
# # ------------------------------
# # FIXED VALUES
# # ------------------------------
# R_rs_const = 3.02            # example constant R_rs (set to yours)
#
# # Compute VD once since VAflow is constant
# VD = params["GV_dead"] * VAflow + params["V0_dead"]
#
# # ------------------------------
# # Vary E_rs only
# # ------------------------------
# E_rs_values = np.linspace(10.95, 43.8, 500)    # choose range as needed
#
# optimal_t1 = []
# optimal_t2 = []
# initial_guess = [1.5, 1.85]
#
# for E_rs in E_rs_values:
#
#     required_params = [
#         0.86,
#         0.489,
#         1.101,
#         1000,
#         10000,
#         E_rs,  # <----- varies
#         R_rs_const,  # <----- varies
#         0.0,
#     ]
#
#     try:
#         res = minimize(
#             objective,
#             x0=np.array(initial_guess[-2:]),
#             args=(required_params, VAflow, VD, 0.001, tolerance),
#             method='nelder-mead',
#             bounds=bounds
#         )
#
#         if res.success:
#             t1_opt, t2_opt = res.x
#             optimal_t1.append(t1_opt)
#             optimal_t2.append(t2_opt)
#             initial_guess.extend(res.x)
#             print(f"E_rs={E_rs:.2f} → t1={t1_opt:.3f}, t2={t2_opt:.3f}")
#         else:
#             print(f"E_rs={E_rs:.2f} → optimization failed")
#             optimal_t1.append(np.nan)
#             optimal_t2.append(np.nan)
#
#     except Exception as e:
#         print(f"E_rs={E_rs:.2f} → error: {e}")
#         optimal_t1.append(np.nan)
#         optimal_t2.append(np.nan)
#
#
# # Convert to arrays
# optimal_t1 = np.array(optimal_t1)
# optimal_t2 = np.array(optimal_t2)
#
# # ------------------------------
# # Plotting
# # ------------------------------
# plt.figure(figsize=(10, 5))
#
# plt.scatter(E_rs_values, optimal_t1, label='Optimal t1', marker='o', alpha=0.7, s=10)
# plt.scatter(E_rs_values, optimal_t2, label='Optimal t2', marker='s', alpha=0.7, s=10)
#
# plt.xlabel("E_rs")
# plt.ylabel("Time (s)")
# plt.title("Optimal t1 and t2 vs E_rs (VAflow and R_rs constant)")
# plt.grid(True)
# plt.legend()
# plt.show()







import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective

#
#
# bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
# tolerance = 0.001
#
# # ------------------------------
# # FIXED VAflow
# # ------------------------------
# VAflow = 0.0673   # <-- your preferred constant value
#
# # ------------------------------
# # NEW: combinations of E_rs and R_rs
# # ------------------------------
# E_rs_values = np.linspace(21.9*0.5, 21.9*1.2, 50)   # example range
# R_rs_values = np.linspace(3.02*0.8, 3.02*1.5, 50)   # example range
#
# # create Cartesian product of E_rs × R_rs
# E_grid, R_grid = np.meshgrid(E_rs_values, R_rs_values)
# E_flat = E_grid.flatten()
# R_flat = R_grid.flatten()
#
# # ------------------------------
# # Compute VD only once (VAflow is constant)
# # ------------------------------
# GV_dead = 0.1698
# V0_dead = 0.1587
# VD = GV_dead * VAflow + V0_dead
#
# optimal_t1 = []
# optimal_t2 = []
# failed = []
# initial_guess = [1.5, 1.85]
#
# for idx in range(len(E_flat)):
#     E_rs = E_flat[idx]
#     R_rs = R_flat[idx]
#
#     required_params = [
#         0.86,
#         0.689,
#         0.5,
#         50,
#         1000,
#         E_rs,  # <----- varies
#         R_rs,  # <----- varies
#         0.0,
#     ]
#
#     try:
#         res = minimize(
#             objective,
#             x0=np.array(initial_guess[-2:]),
#             args=(required_params, VAflow, VD, 0.001, tolerance),
#             method='nelder-mead',
#             bounds=bounds
#         )
#
#         if res.success:
#             t1_opt, t2_opt = res.x
#             optimal_t1.append(t1_opt)
#             optimal_t2.append(t2_opt)
#
#             initial_guess.extend(res.x)
#
#             print(f"E_rs={E_rs:.2f}, R_rs={R_rs:.2f} → t1={t1_opt:.3f}, t2={t2_opt:.3f}")
#         else:
#             print(f"E_rs={E_rs:.2f}, R_rs={R_rs:.2f} → optimization failed")
#             optimal_t1.append(np.nan)
#             optimal_t2.append(np.nan)
#
#     except Exception as e:
#         print(f"E_rs={E_rs:.2f}, R_rs={R_rs:.2f} → error: {e}")
#         optimal_t1.append(np.nan)
#         optimal_t2.append(np.nan)
#
# # Convert to arrays
# optimal_t1 = np.array(optimal_t1)
# optimal_t2 = np.array(optimal_t2)
#
# # reshape back to the grid shape for nicer plotting
# T1_grid = optimal_t1.reshape(E_grid.shape)
# T2_grid = optimal_t2.reshape(E_grid.shape)
#
# # ------------------------------
# # Plotting
# # ------------------------------
# plt.figure(figsize=(12,5))
#
# plt.subplot(1,2,1)
# plt.title("Optimal t1 vs E_rs and R_rs")
# plt.xlabel("E_rs")
# plt.ylabel("R_rs")
# plt.contourf(E_grid, R_grid, T1_grid, levels=20)
# plt.colorbar(label="t1 (s)")
#
# plt.subplot(1,2,2)
# plt.title("Optimal t2 vs E_rs and R_rs")
# plt.xlabel("E_rs")
# plt.ylabel("R_rs")
# plt.contourf(E_grid, R_grid, T2_grid, levels=20)
# plt.colorbar(label="t2 (s)")
#
# plt.tight_layout()
# plt.show()













bounds = [(0.5, 6), (0.5, 9)]  # [t1, t2]
tolerance = 0.001

VAflow_vals = np.linspace(0.01, 1.2, 200)
# VAflow_repeated = np.linspace(0.03, 1.6, 200)
# VAflow_repeated = np.repeat(VAflow_vals, 1)
GV_dead = 0.1698
V0_dead = 0.1587
VD = GV_dead * VAflow_vals + V0_dead

optimal_t1 = []
optimal_t2 = []
failed_indices = []
initial_guess = [1.5, 1.85]

for idx, VAflow in enumerate(VAflow_vals):
    VD_volume = VD[idx]
    required_params = [
        0.86,
        0.489,
        0.4,
        50,
        1000,
        21.9,
        3.02,
        0.0,
    ]
    try:
        res = minimize(objective, x0=np.array(initial_guess[-2:]), args=(required_params, VAflow, VD_volume, 0.001, tolerance), method='nelder-mead', bounds=bounds)
        # res = minimize(objective, x0=Next_Conditions["Nd"][-2:], method='nelder-mead', bounds=bounds)
        if res.success:
            t1_opt, t2_opt = res.x
            optimal_t1.append(t1_opt)
            optimal_t2.append(t2_opt)
            initial_guess.extend(res.x)
            print(f"VAflow = {VAflow:.4f} → optimal t1 = {t1_opt:.4f}, t2 = {t2_opt:.4f}")
        else:
            print(f"VAflow = {VAflow:.4f} → optimization failed")
            optimal_t1.append(np.nan)
            optimal_t2.append(np.nan)
    except Exception as e:
        print(f"VAflow = {VAflow:.4f} → exception: {e}")
        optimal_t1.append(np.nan)
        optimal_t2.append(np.nan)


# Convert to arrays for indexing
VAflow_clean = np.array(VAflow_vals)
VD_clean = np.array(VD)
t1_clean = np.array(optimal_t1)
t2_clean = np.array(optimal_t2)

VEflow = 60 * (VAflow_clean + VD_clean / (t1_clean + t2_clean))

t1_mean = np.array([np.nanmean(t1_clean[VAflow_clean == v]) for v in VAflow_vals])
t2_mean = np.array([np.nanmean(t2_clean[VAflow_clean == v]) for v in VAflow_vals])

cs_t1 = CubicSpline(VAflow_vals, t1_mean, bc_type="natural")
cs_t2 = CubicSpline(VAflow_vals, t2_mean, bc_type="natural")

coeffs_t1 = cs_t1.c              # shape (4, N-1)
coeffs_t2 = cs_t2.c              # shape (4, N-1)
knots_1 = cs_t1.x
knots_2 = cs_t2.x

t1_spline = cs_t1(VAflow_vals)
t2_spline = cs_t2(VAflow_vals)

plt.figure(figsize=(10, 5))
plt.scatter(VEflow, t1_clean, label='Optimal Inspiration Time', color='blue', alpha=0.6, s=25)
plt.scatter(VEflow, t2_clean, label='Optimal Expiration Time', color='red', alpha=0.6, s=25)
plt.xlabel('Minute Ventilation (L/min)')
plt.ylabel('Time (s)')
plt.title('Optimal t1 and t2 vs VEflow Using nelder-mead')
plt.legend()
# plt.grid(True)
plt.show()
# np.savez("t1_t2_vs_VAflow.npz", VAflow=VAflow_clean, t1=t1_clean, t2=t2_clean)
#
# plt.savefig("optimal_t1_t2.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(10, 5))
# Spline fits
plt.plot(
    VAflow_vals, t1_spline,
    label='t1 CubicSpline',
    color='blue', linewidth=2
)
plt.plot(
    VAflow_vals, t2_spline,
    label='t2 CubicSpline',
    color='red', linewidth=2
)

plt.scatter(
    VAflow_vals, t1_mean,
    color='blue', s=40, marker='o', label='t1 mean (knots)'
)
plt.scatter(
    VAflow_vals, t2_mean,
    color='red', s=40, marker='o', label='t2 mean (knots)'
)

plt.scatter(VAflow_clean, t1_clean, label='Optimal t1 (Inspiration Time)', color='blue', alpha=0.6, s=5)
plt.scatter(VAflow_clean, t2_clean, label='Optimal t2 (Expiration Time)', color='red', alpha=0.6, s=5)
plt.xlabel('VAflow (L/s)')
plt.ylabel('Time (s)')
plt.title('Optimal t1 and t2 vs VAflow Using nelder-mead')
plt.legend()
plt.grid(True)
plt.show()




# dt = 0.001
# bounds = [(1, 3), (1.5, 6)]  # [t1, t2] bounds
# tolerance = 0.001
# opt = BreathOptimiser(params, VAflow, VD, dt, tolerance)
#
# t1_vals = np.linspace(1, 3, 30)
# t2_vals = np.linspace(1.5, 6, 30)
#
#
# T1, T2 = np.meshgrid(t1_vals, t2_vals)
# Z_inspire = np.zeros_like(T1)
# Z_expire = np.zeros_like(T1)
#
# for i in range(T1.shape[0]):
#     for j in range(T1.shape[1]):
#         t1 = T1[i, j]
#         t2 = T2[i, j]
#         try:
#             I_inspire, I_expire = opt.compute_integral_inspire(t1, t2)
#             Z_inspire[i, j] = I_inspire
#             Z_expire[i, j] = I_expire
#         except Exception:
#             Z_inspire[i, j] = np.nan
#             Z_expire[i, j] = np.nan
#
# # Plot side-by-side
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#
# cs1 = axs[0].contourf(T1, T2, Z_inspire, levels=50, cmap='viridis')
# fig.colorbar(cs1, ax=axs[0], label='Integral Inspire')
# axs[0].set_title('Inspiratory Work')
# axs[0].set_xlabel('t1 (Inspiration time)')
# axs[0].set_ylabel('t2 (Expiration time)')
#
# cs2 = axs[1].contourf(T1, T2, Z_expire, levels=50, cmap='plasma')
# fig.colorbar(cs2, ax=axs[1], label='Integral Expire')
# axs[1].set_title('Expiratory Work')
# axs[1].set_xlabel('t1 (Inspiration time)')
# axs[1].set_ylabel('t2 (Expiration time)')
#
# plt.tight_layout()
# plt.show()





# T1, T2 = np.meshgrid(t1_vals, t2_vals)
# Z = np.zeros_like(T1)
#
# for i in range(T1.shape[0]):
#     for j in range(T1.shape[1]):
#         t1 = T1[i, j]
#         t2 = T2[i, j]
#         try:
#             Z[i, j] = opt.objective([t1, t2])
#         except Exception as e:
#             Z[i, j] = np.nan  # mark failed evaluations
#
# # Plot
# plt.figure(figsize=(10, 6))
# cp = plt.contourf(T1, T2, Z, levels=50, cmap='viridis')
# plt.colorbar(cp, label='Work (J)')
# plt.xlabel('t1 (Inspiration time)')
# plt.ylabel('t2 (Expiration time)')
# plt.title('Work')
# plt.show()



# # Set up lambda1 and lambda2 sweep ranges
# lambda1_vals = np.linspace(0, 2, 30)
# lambda2_vals = np.linspace(0, 2, 30)
# L1, L2 = np.meshgrid(lambda1_vals, lambda2_vals)
# Z = np.zeros_like(L1)
#
# for i in range(L1.shape[0]):
#     for j in range(L1.shape[1]):
#
#         lambda1 = L1[i, j]
#         lambda2 = L2[i, j]
#
#         # Update parameters
#         params["lambda1"] = lambda1
#         params["lambda2"] = lambda2
#         required_params = [params["lambda1"], params["lambda2"], params["n"], params["Pmax"], params["Pmax_dot"]]
#
#         try:
#             res = minimize(objective, x0=np.array(initial_guess[-2:]),
#                            args=(required_params, VAflow, VD, dt, tolerance), method='nelder-mead', bounds=bounds)
#
#             if res.success:
#                 Z[i, j] = res.x[0]
#             else:
#                 Z[i, j] = np.nan
#         except Exception:
#             Z[i, j] = np.nan
#
#         print(f"Processing λ1={lambda1:.2f}, λ2={lambda2:.2f} → t2 = {Z[i, j]:.4f}")
#
# # Plot heatmap of minimum work vs lambda1, lambda2
# plt.figure(figsize=(10, 6))
# cp = plt.contourf(L1, L2, Z, levels=50, cmap='viridis')
# plt.colorbar(cp, label='Inspiration time t1')
# plt.xlabel('lambda1 (smoothness penalty)')
# plt.ylabel('lambda2 (expiration penalty weight)')
# plt.title('Inspiration time t1 across λ1 and λ2 (optimized over t1, t2)')
# plt.tight_layout()
# plt.show()

