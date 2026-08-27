"""
MCMC_Union.py — Post-history-matching calibration pipeline for rest + exercise

Pipeline:
  1. Load NROY results from union history matching
  2. Fit an NROY-informed prior over the calibrated parameter subset
  3. Pyro NUTS MCMC -> posterior distribution targeting rest and exercise data

Why NUTS over emcee / random-walk MH:
  - high-dimensional calibration: gradient-based NUTS scales as O(d^{1/4}),
    whereas emcee's stretch move degrades rapidly past ~30 dims.
  - The GP emulators (autoemulate TransformedEmulator with GPyTorch)
    support with_grad=True, so the full chain
        theta -> x_transform -> GP kernel -> predictive mean/var -> log_prob
    is differentiable through torch autograd.
  - NUTS automatically adapts step size and mass matrix during warmup,
    handling the very different scales across the calibrated parameters.
  - ~100-200x more sample-efficient than emcee for this dimensionality.

Expects in working directory:
  NROY_Points_union_{PERCENT}.npy  -- NROY parameter vectors
  NROY_Params_union_{PERCENT}.npy  -- NROY bounds dict
  {EMULATOR_DIR}/{output_name}/GaussianProcessMatern32_{output_name}_best.joblib

Usage:
  python MCMC_Union.py
  python MCMC_Union.py --sequential
  python MCMC_Union.py --chain-id 0 --n-chains 4
  python MCMC_Union.py --aggregate-only --n-chains 4
"""

import math
# import os
import sys
import json
import argparse
import subprocess
import warnings
import re
import time
import joblib
import numpy as np
import torch
import gpytorch
import multiprocessing
import pyro
# import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC

# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler
from scipy.stats import norm as _scipy_norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(multiprocessing.cpu_count()))   # set before sklearn/joblib uses loky

warnings.filterwarnings("ignore")

# ================================================================
# CLI: HPC array-job orchestration
# ================================================================
# --chain-id c       : run only chain c (0..N_CHAINS-1) and save
#                      chain_z_{c}.npy + chain_diag_{c}.joblib in out_dir,
#                      then exit before diagnostics/plotting.
# --aggregate-only   : skip chain running; load all chain_z_{c}.npy from
#                      out_dir and run diagnostics + plots only.
# --n-chains N       : expected number of chains to run/aggregate.
# (no args)          : run N_CHAINS chain workers concurrently, then aggregate.
# --sequential       : run all chains sequentially in this process.
_parser = argparse.ArgumentParser()
_parser.add_argument("--chain-id", type=int, default=-1)
_parser.add_argument("--aggregate-only", action="store_true")
_parser.add_argument("--n-chains", type=int, default=None)
_parser.add_argument("--sequential", action="store_true")
_parser.add_argument(
    "--hm-artifacts-dir",
    default=None,
    help="Directory containing union HM artifacts; default is the launch directory.",
)
_parser.add_argument(
    "--emulator-dir",
    default=None,
    help="Directory containing union GP emulators; default is HM artifacts dir/Emulator_union_all_wave.",
)
_parser.add_argument(
    "--run-dir",
    default=None,
    help="Directory for MCMC outputs/logs; default is launch directory/MCMC_Union_...",
)
_parser.add_argument(
    "--threads-per-chain",
    type=int,
    default=None,
    help="CPU threads assigned to each chain worker; overrides scheduler-aware default.",
)
_parser.add_argument(
    "--max-threads-per-chain",
    type=int,
    default=int(os.environ.get("MCMC_MAX_THREADS_PER_CHAIN", "64")),
    help="Cap automatic per-chain thread count; set 0 to disable. Default: 16.",
)
_args, _ = _parser.parse_known_args()
CHAIN_ID       = _args.chain_id
AGGREGATE_ONLY = _args.aggregate_only



# ================================================================
# SETTINGS
# ================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)

PERCENT      = 50                        # param range +/-% used in HM # change

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCH_DIR = os.getcwd()


def _resolve_path(path, base_dir=LAUNCH_DIR):
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


HM_ARTIFACTS_DIR = (
    _resolve_path(_args.hm_artifacts_dir)
    if _args.hm_artifacts_dir
    else os.path.abspath(LAUNCH_DIR)
)
root = HM_ARTIFACTS_DIR
EMULATOR_DIR = (
    _resolve_path(_args.emulator_dir, HM_ARTIFACTS_DIR)
    if _args.emulator_dir
    else os.path.join(HM_ARTIFACTS_DIR, "Emulator_union_all_wave")
)     # GP emulators from last refitted wave # change
out_dir = (
    _resolve_path(_args.run_dir, LAUNCH_DIR)
    if _args.run_dir
    else os.path.join(LAUNCH_DIR, f"MCMC_Union_{PERCENT}_28_08_copula_prior")
)
os.makedirs(out_dir, exist_ok=True)


USE_COPULA_PRIOR   = True          # False -> uniform box prior
KDE_SUBSAMPLE      = 5000          # subsample size per-axis KDE eval at MCMC
KDE_BANDWIDTH      = "silverman"   # "silverman" | "scott" | float
CORR_SHRINK        = 0.0           # linear shrinkage of R toward identity
COPULA_F_EPS       = 1e-6          # clamp for Phi^{-1}(F_i) at marginal tails

# ---- Logspline marginals (replaces per-axis Gaussian KDE) ------------------
# The Silverman + Gaussian-KDE marginals produce
#   (a) subsample-noise wiggles on near-uniform axes,
#   (b) inward-shifted peaks on axes that pile up against HM bounds (Gaussian
#       kernels leak mass past the boundary).
# Diagnostic: Plot_Copula_Marginals_vs_NROY.py vs ..._logspline.py.
# Fix: per-axis penalised cubic B-spline log-density on [L_i, U_i]:
#   log f_i(x) = sum_j beta_j * B_j(x)  on [prior_lower[i], prior_upper[i]]
# fit by penalised maximum likelihood (P-spline; Eilers & Marx 1996).
# Bounded support is enforced by the clamped knot vector (non-zero density
# at the HM bound is allowed; density is exactly zero outside).
# MCMC evaluation: precomputed (x_grid, f_grid, cdf_grid) per axis, then
# torch.searchsorted + piecewise-linear interp + analytic piecewise-quadratic
# CDF.  Differentiable; O(log G) per axis per step, cheaper than the KDE loop.
USE_LOGSPLINE_MARGINALS = True
LOGSPLINE_N_INTERIOR    = 14        # interior knots on [L_i, U_i]
LOGSPLINE_LAMBDA        = 5.0       # 2nd-difference smoothness penalty
LOGSPLINE_N_GRID        = 1000      # per-axis fine grid for Z + CDF + MCMC
LOGSPLINE_PLOT_DPI      = 1200      # high-resolution PNG export for copula plot

# NUTS MCMC
N_WARMUP        = 500                    # warmup (step-size + mass-matrix adapt)
N_SAMPLES       = 3000                   # posterior draws per chain
N_CHAINS        = 4                      # independent chains
MAX_TREE_DEPTH  = 7
TARGET_ACCEPT   = 0.8

if _args.n_chains is not None:
    N_CHAINS = _args.n_chains
if N_CHAINS < 1:
    raise ValueError(f"--n-chains must be >= 1, got {N_CHAINS}")
if AGGREGATE_ONLY and CHAIN_ID >= 0:
    raise ValueError("--aggregate-only and --chain-id are mutually exclusive")
if CHAIN_ID < -1 or CHAIN_ID >= N_CHAINS:
    raise ValueError(f"--chain-id={CHAIN_ID} outside [0, {N_CHAINS - 1}]")
if _args.threads_per_chain is not None and _args.threads_per_chain < 1:
    raise ValueError(
        f"--threads-per-chain must be >= 1, got {_args.threads_per_chain}"
    )
if _args.max_threads_per_chain is not None and _args.max_threads_per_chain < 0:
    raise ValueError(
        f"--max-threads-per-chain must be >= 0, got {_args.max_threads_per_chain}"
    )


def _first_int_from_env_value(value):
    if not value:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _available_cpu_count():
    """Best-effort CPU count that respects scheduler/affinity limits."""
    candidates = []
    for name in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_CPUS_PER_NODE",
        "PBS_NP",
        "NSLOTS",
    ):
        value = _first_int_from_env_value(os.environ.get(name))
        if value and value > 0:
            candidates.append(value)
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(len(os.sched_getaffinity(0)))
        except Exception:
            pass
    candidates.append(os.cpu_count() or multiprocessing.cpu_count() or 1)
    return max(1, min(c for c in candidates if c and c > 0))


def _auto_threads_per_worker(active_workers):
    threads = max(1, _available_cpu_count() // max(active_workers, 1))
    max_threads = _args.max_threads_per_chain
    if max_threads:
        threads = min(threads, max_threads)
    return threads


def _apply_local_thread_limits(active_workers):
    threads = _args.threads_per_chain
    if threads is None:
        env_threads = _first_int_from_env_value(os.environ.get("TORCH_NUM_THREADS"))
        threads = env_threads or _auto_threads_per_worker(active_workers)
        if env_threads and _args.max_threads_per_chain:
            threads = min(threads, _args.max_threads_per_chain)

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["TORCH_NUM_THREADS"] = str(threads)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(threads)

    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        torch.set_num_threads(threads)
    return threads


def _parallel_worker_env(n_chains, threads_per_chain):
    """Environment for concurrent chain subprocesses on Linux/HPC nodes."""
    env = os.environ.copy()
    if threads_per_chain is None:
        threads_per_chain = _auto_threads_per_worker(n_chains)

    thread_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    )
    for name in thread_vars:
        env[name] = str(threads_per_chain)
    return env, threads_per_chain


def _mcmc_driver_args():
    args = [
        "--hm-artifacts-dir", HM_ARTIFACTS_DIR,
        "--emulator-dir", EMULATOR_DIR,
        "--run-dir", out_dir,
        "--max-threads-per-chain", str(_args.max_threads_per_chain),
    ]
    if _args.threads_per_chain is not None:
        args.extend(["--threads-per-chain", str(_args.threads_per_chain)])
    return args


def _run_parallel_chains_and_aggregate():
    """Default no-arg driver: run one subprocess per chain, then aggregate."""
    script_path = os.path.abspath(__file__)
    run_cwd = os.getcwd()
    env, threads_per_chain = _parallel_worker_env(
        N_CHAINS, _args.threads_per_chain
    )

    print("  Mode: PARALLEL DRIVER (one subprocess per chain)", flush=True)
    print(
        f"Launching {N_CHAINS} chain workers concurrently "
        f"({threads_per_chain} CPU thread(s) per worker by default).",
        flush=True,
    )
    print(f"Worker logs will be written to {os.path.abspath(out_dir)}", flush=True)

    workers = []
    try:
        for c in range(N_CHAINS):
            log_path = os.path.abspath(os.path.join(out_dir, f"chain_{c}.log"))
            log_file = open(log_path, "w", buffering=1)
            cmd = [
                sys.executable,
                script_path,
                "--chain-id", str(c),
                "--n-chains", str(N_CHAINS),
                *_mcmc_driver_args(),
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=run_cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            workers.append((c, proc, log_path, log_file))
            print(f"  chain {c}: pid={proc.pid}, log={log_path}", flush=True)

        failed = []
        for c, proc, log_path, log_file in workers:
            return_code = proc.wait()
            log_file.close()
            if return_code != 0:
                failed.append((c, return_code, log_path))
            else:
                print(f"  chain {c}: complete", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted; terminating chain workers...", flush=True)
        for _, proc, _, log_file in workers:
            if proc.poll() is None:
                proc.terminate()
            log_file.close()
        raise

    if failed:
        print("\nOne or more chain workers failed:", flush=True)
        for c, return_code, log_path in failed:
            print(
                f"  chain {c}: exit code {return_code}; see {log_path}",
                flush=True,
            )
        sys.exit(1)

    aggregate_log = os.path.abspath(os.path.join(out_dir, "aggregate.log"))
    print(f"\nAll chains complete; aggregating results into {out_dir}/", flush=True)
    print(f"Aggregate log: {aggregate_log}", flush=True)
    with open(aggregate_log, "w", buffering=1) as log_file:
        aggregate_return_code = subprocess.call(
            [
                sys.executable,
                script_path,
                "--aggregate-only",
                "--n-chains", str(N_CHAINS),
                *_mcmc_driver_args(),
            ],
            cwd=run_cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if aggregate_return_code != 0:
        print(
            f"Aggregation failed with exit code {aggregate_return_code}; "
            f"see {aggregate_log}",
            flush=True,
        )
        sys.exit(aggregate_return_code)

    print("Parallel MCMC run complete.", flush=True)
    sys.exit(0)


if (
    N_CHAINS > 1
    and CHAIN_ID < 0
    and not AGGREGATE_ONLY
    and not _args.sequential
):
    _run_parallel_chains_and_aggregate()

CHAIN_WORKER = CHAIN_ID >= 0
WRITE_SHARED_ARTIFACTS = not CHAIN_WORKER
ACTIVE_WORKERS = 1 if (CHAIN_WORKER or _args.sequential or AGGREGATE_ONLY) else N_CHAINS
THREADS_PER_CHAIN_EFFECTIVE = _apply_local_thread_limits(ACTIVE_WORKERS)

# Atrial contraction is enforced via the active-emptying fraction
#   r = (V_pre_contraction - V_min) / (V_max - V_min)
# treated as a derived Gaussian target. The corresponding observation entries
# (Pre-LA/Pre-RA Contraction Volume) hold the ratio mean and variance, NOT the
# raw pre-contraction mL volume, so the per-output Gaussian must skip these
# indices and the ratio is scored separately under the same Gaussian likelihood
# family.
REST_LA_MIN_IDX, REST_LA_MAX_IDX, REST_LA_PRE_IDX = 13, 14, 17
REST_RA_MIN_IDX, REST_RA_MAX_IDX, REST_RA_PRE_IDX = 9, 10, 18
REST_ATRIAL_GAUSSIAN_SKIP = (REST_LA_PRE_IDX, REST_RA_PRE_IDX)
EXERCISE_LA_MIN_IDX, EXERCISE_LA_MAX_IDX, EXERCISE_LA_PRE_IDX = 38, 39, 42
EXERCISE_RA_MIN_IDX, EXERCISE_RA_MAX_IDX, EXERCISE_RA_PRE_IDX = 34, 35, 43

ATRIAL_GAUSSIAN_SKIP = (REST_LA_PRE_IDX, REST_RA_PRE_IDX, EXERCISE_LA_PRE_IDX, EXERCISE_RA_PRE_IDX)
ATRIAL_COV_JITTER = 1e-10
ATRIAL_RATIO_TARGET = (0.25, 0.0025)

# Display-only atrial pre-contraction volume targets used in the final
# posterior-predictive box plot. The likelihood remains defined on the
# active-emptying fraction ratio above. The actual display mean/std are
# derived later from the observation moments for (V_min, V_max, f).

# Posterior predictive
N_PRED_CHECK = 1500                       # posterior samples for predictive check

# Likelihood: Gaussian on direct targets + Gaussian on derived atrial ratios


def _propagated_vpre_display_stats(vmin_mean, vmin_var, vmax_mean, vmax_var,
                                   f_mean, f_var):
    """Display-only mean/std for V_pre = V_min + f * (V_max - V_min).

    This mirrors the zero-cross-covariance assumption used by the MCMC
    observation model: V_min, V_max and the active-emptying fraction f are
    treated as independent when constructing the display normalisation.
    """
    delta_mean = float(vmax_mean) - float(vmin_mean)
    f_mean = float(f_mean)
    f_var = float(f_var)

    vpre_mean = float(vmin_mean) + f_mean * delta_mean
    ef2 = f_mean ** 2 + f_var
    e1mf2 = (1.0 - f_mean) ** 2 + f_var
    vpre_var = (
        e1mf2 * float(vmin_var)
        + ef2 * float(vmax_var)
        + f_var * delta_mean ** 2
    )
    return vpre_mean, math.sqrt(max(vpre_var, 0.0))

def extract_fast_caches(emulators, output_names):
    """Pre-warm GPyTorch prediction caches and extract y-transform params.

    Instead of reimplementing the Matern-3/2 kernel manually (which can diverge
    numerically from GPyTorch's internal noise/jitter handling), this stores a
    reference to each GP model and calls GPyTorch directly during MCMC.  The
    only part of the autoemulate pipeline that is bypassed is the expensive
    delta_method/vmap wrapper, which is unnecessary for affine y-transforms
    (StandardizeTransform: y = y_t * std + mean).

    NOTE: The autoemulate pipeline has TWO y-inverse-transforms:
      1. GP's own y_transform (standardize_y=True by default in the GP class)
      2. TransformedEmulator's y_transforms
    Both are affine (StandardizeTransform), so we pre-combine them into a
    single affine: y_final = y_gp * combined_std + combined_mean.
    """
    caches = {}
    for name in output_names:
        te = emulators[name]
        gp = te.model
        gp.eval()
        gp.likelihood.eval()

        # ---- x-transform (TransformedEmulator's only; GP has standardize_x=False) ----
        x_mean = te.x_transforms[0].mean.detach().squeeze(0)   # (d,)
        x_std  = te.x_transforms[0].std.detach().squeeze(0)    # (d,)

        # ---- y-transforms: combine GP's + TransformedEmulator's ----
        # TransformedEmulator's y-inverse: y_te = y_t * te_std + te_mean
        te_y_mean = te.y_transforms[0].mean.detach().squeeze()  # scalar
        te_y_std  = te.y_transforms[0].std.detach().squeeze()   # scalar

        # GP's own y-inverse: y_gp_out = y_gp * gp_std + gp_mean
        # (applied inside Emulator.predict before TransformedEmulator sees it)
        if gp.y_transform is not None and getattr(gp.y_transform, '_is_fitted', False):
            gp_y_mean = gp.y_transform.mean.detach().squeeze()
            gp_y_std  = gp.y_transform.std.detach().squeeze()
            # Compose: y_final = (y_gp * gp_std + gp_mean) * te_std + te_mean
            combined_y_std  = gp_y_std * te_y_std
            combined_y_mean = gp_y_mean * te_y_std + te_y_mean
        else:
            combined_y_std  = te_y_std
            combined_y_mean = te_y_mean

        # Pre-warm GPyTorch prediction strategy (triggers Cholesky once)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _ = gp(gp.train_inputs[0][:1])

        caches[name] = {
            "gp": gp,
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": combined_y_mean, "y_std": combined_y_std,
        }
    return caches


# ================================================================
# BATCHED / MANUAL MATERN-3/2 PATH  (no gpytorch at inference time)
# ================================================================
# Stacks all GPs into a single set of (n_out, ...) tensors that share
# X_train, precomputes the Cholesky of each training kernel + its alpha,
# and runs a hand-written Matern-3/2 kernel.  Removes GPyTorch's per-call
# MultivariateNormal / LazyTensor / fast_pred_var overhead (which dominates
# at batch size 1) and also removes the per-output Python loop — replaced by
# batched matmul / triangular solve.
#
# Math recap (for one output, written without batch dim):
#   training kernel  K  = outputscale * matern32(X, X; ls) + noise * I
#   Cholesky         L  : L L^T = K
#   alpha               = K^{-1} (y_train - mean_const)
#   at a new point x*:
#       k*  = outputscale * matern32(x*, X; ls)
#       mu_latent  = mean_const + k* . alpha
#       var_latent = outputscale - k* . (K^{-1} k*)
#   then the combined affine y-inverse-transform gives final mu, var.
# This matches what `gp(x)` returns in GPyTorch eval mode (latent, no noise).

SQRT3 = math.sqrt(3.0)


def _matern32_cross(x_t, X_train, lengthscale, outputscale):
    """k(x_t, X_train) for Matern-3/2, batched over outputs.

    x_t:         (d,)        — one test point, shared across outputs
    X_train:     (n, d)      — shared training inputs
    lengthscale: (n_out, d)
    outputscale: (n_out,)
    returns k_star: (n_out, n)
    """
    # scaled per-output:  (x_t - X_train) / lengthscale[i]   →  (n_out, n, d)
    diff = (x_t.unsqueeze(0).unsqueeze(1) - X_train.unsqueeze(0)) \
           / lengthscale.unsqueeze(1)
    # euclidean distance with tiny epsilon for gradient stability at r=0
    r = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-30)              # (n_out, n)
    return outputscale.unsqueeze(-1) * (1.0 + SQRT3 * r) * torch.exp(-SQRT3 * r)


def _matern32_cross_fast(x_t, X_train_scaled, X_train_scaled_norm2,
                         lengthscale, outputscale):
    """Same as _matern32_cross, but uses the identity
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    to avoid materialising an (n_out, n, d) intermediate tensor.

    X_train_scaled:       (n_out, n, d) — X_train / lengthscale[i], precomputed
    X_train_scaled_norm2: (n_out, n)    — ||X_train / lengthscale[i]||^2, precomputed

    The cross term is a BLAS-accelerated batched matvec (einsum), and the
    squared-norm reduction happens on (n_out, n) instead of (n_out, n, d).
    Gradient flows through x_t exactly as before.
    """
    x_scaled = x_t.unsqueeze(0) / lengthscale                       # (n_out, d)
    x_norm2  = (x_scaled ** 2).sum(dim=-1)                          # (n_out,)
    # cross[i, n] = <x_scaled[i], X_train_scaled[i, n]>
    cross = torch.einsum('od,ond->on', x_scaled, X_train_scaled)    # (n_out, n)
    r_sq = (x_norm2.unsqueeze(-1) + X_train_scaled_norm2
            - 2.0 * cross).clamp_min(1e-30)
    r = torch.sqrt(r_sq)                                            # (n_out, n)
    return outputscale.unsqueeze(-1) * (1.0 + SQRT3 * r) * torch.exp(-SQRT3 * r)


def build_batched_fast_caches(emulators, output_names, gp_caches):
    """Extract hyperparameters, stack across outputs, precompute L and alpha.

    Requires all GPs to share the same X_train and the same x-transform.
    That is the autoemulate default when they are trained together; we
    assert it here so a silent mismatch can't poison the posterior.
    """
    first_gp = emulators[output_names[0]].model
    X_train = first_gp.train_inputs[0].detach().to(torch.float32).clone()
    n_train, d = X_train.shape
    n_out = len(output_names)

    y_train   = torch.empty(n_out, n_train, dtype=torch.float32)
    lengthscale = torch.empty(n_out, d, dtype=torch.float32)
    outputscale = torch.empty(n_out, dtype=torch.float32)
    noise       = torch.empty(n_out, dtype=torch.float32)
    mean_const  = torch.empty(n_out, dtype=torch.float32)
    y_mean      = torch.empty(n_out, dtype=torch.float32)
    y_std       = torch.empty(n_out, dtype=torch.float32)

    # x-transform shared across all GPs — taken from first, asserted for rest
    x_mean = gp_caches[output_names[0]]["x_mean"].detach().to(torch.float32).clone()
    x_std  = gp_caches[output_names[0]]["x_std"].detach().to(torch.float32).clone()

    for i, name in enumerate(output_names):
        gp = emulators[name].model
        c  = gp_caches[name]

        Xi = gp.train_inputs[0].detach().to(torch.float32)
        assert Xi.shape == X_train.shape, \
            f"{name}: train_inputs shape {Xi.shape} != {X_train.shape}"
        assert torch.allclose(Xi, X_train, atol=1e-5, rtol=1e-5), \
            f"{name}: train_inputs differ from reference GP — batched stacking requires shared X_train"
        assert torch.allclose(c["x_mean"].to(torch.float32), x_mean, atol=1e-5), \
            f"{name}: x_mean differs from reference"
        assert torch.allclose(c["x_std"].to(torch.float32), x_std, atol=1e-5), \
            f"{name}: x_std differs from reference"

        _yt = gp.train_targets.detach().to(torch.float32)
        if _yt.dim() > 1:
            _yt = _yt.reshape(-1)
        assert _yt.shape == (n_train,), \
            f"{name}: train_targets shape {tuple(_yt.shape)} (expected ({n_train},))"
        y_train[i] = _yt

        ls = gp.covar_module.base_kernel.lengthscale.detach().squeeze().to(torch.float32)
        assert ls.shape == (d,), f"{name}: lengthscale shape {ls.shape} (expected ({d},))"
        lengthscale[i] = ls

        outputscale[i] = gp.covar_module.outputscale.detach().squeeze().to(torch.float32)

        _noise = gp.likelihood.noise.detach().to(torch.float32).reshape(-1)
        assert _noise.numel() in (1, n_train), \
            f"{name}: likelihood.noise has {_noise.numel()} elements (expected 1 or {n_train})"
        noise[i] = _noise[0] if _noise.numel() == 1 else _noise.mean()

        # Evaluate the mean module on X_train directly — works for any mean type,
        # catches cases where `hasattr(..., 'constant')` silently falls through.
        with torch.no_grad():
            _tm = gp.mean_module(X_train).detach().to(torch.float32).reshape(-1)
        assert _tm.shape == (n_train,), \
            f"{name}: mean_module(X_train) has shape {tuple(_tm.shape)} (expected ({n_train},))"
        _tm_const = _tm[0]
        if not torch.allclose(_tm, _tm_const.expand_as(_tm), atol=1e-5, rtol=1e-4):
            raise NotImplementedError(
                f"{name}: mean_module ({type(gp.mean_module).__name__}) is not constant "
                f"over training inputs (range {_tm.min().item():.4g}..{_tm.max().item():.4g}). "
                f"Batched path supports ConstantMean / ZeroMean only."
            )
        mean_const[i] = _tm_const

        y_mean[i] = c["y_mean"].detach().to(torch.float32)
        y_std[i]  = c["y_std"].detach().to(torch.float32)

    # Build training-kernel batch K (n_out, n, n) and factorise once.
    X_scaled = X_train.unsqueeze(0) / lengthscale.unsqueeze(1)           # (n_out, n, d)
    dists    = torch.cdist(X_scaled, X_scaled, p=2.0)                    # (n_out, n, n)
    K_base   = (1.0 + SQRT3 * dists) * torch.exp(-SQRT3 * dists)
    K_latent = outputscale.view(-1, 1, 1) * K_base
    K        = K_latent + noise.view(-1, 1, 1) * torch.eye(
        n_train, dtype=torch.float32
    ).unsqueeze(0)
    L        = torch.linalg.cholesky(K)                                  # (n_out, n, n)

    alpha = torch.cholesky_solve(
        (y_train - mean_const.unsqueeze(-1)).unsqueeze(-1), L
    ).squeeze(-1)                                                        # (n_out, n)

    # Estimate cross-output residual correlation once from the training fit.
    mu_train_latent = mean_const.unsqueeze(-1) + torch.bmm(
        K_latent, alpha.unsqueeze(-1)
    ).squeeze(-1)
    y_train_phys = y_train * y_std.unsqueeze(-1) + y_mean.unsqueeze(-1)
    mu_train_phys = mu_train_latent * y_std.unsqueeze(-1) + y_mean.unsqueeze(-1)
    residual = y_train_phys - mu_train_phys
    residual_centered = residual - residual.mean(dim=1, keepdim=True)
    residual_cov = residual_centered @ residual_centered.T / max(n_train - 1, 1)
    residual_std = torch.sqrt(torch.diagonal(residual_cov).clamp_min(1e-12))
    residual_corr = residual_cov / (
        residual_std.unsqueeze(1) * residual_std.unsqueeze(0)
    ).clamp_min(1e-12)
    residual_corr.fill_diagonal_(1.0)

    # Per-test-step speedup: cache the scaled training inputs and their
    # squared norms so _matern32_cross_fast can compute
    # ||x/ls - X/ls||^2 = ||x/ls||^2 + ||X/ls||^2 - 2 <x/ls, X/ls>
    # without allocating an (n_out, n, d) difference tensor every NUTS step.
    X_train_scaled       = X_scaled.contiguous()                         # (n_out, n, d)
    X_train_scaled_norm2 = (X_train_scaled ** 2).sum(dim=-1)             # (n_out, n)

    return {
        "X_train": X_train, "lengthscale": lengthscale,
        "outputscale": outputscale, "mean_const": mean_const,
        "L": L, "alpha": alpha,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "X_train_scaled": X_train_scaled,
        "X_train_scaled_norm2": X_train_scaled_norm2,
        "residual_corr": residual_corr,
    }


def fit_gaussian_copula(data_np, n_kde=5000, bandwidth="silverman",
                        corr_shrink=0.0, f_eps=1e-6, random_state=0):
    """Fit a Gaussian copula with per-axis Gaussian-KDE marginals.

    Structure (Sklar's theorem):
        p(theta) = c(F_1(theta_1), ..., F_d(theta_d)) * prod_i f_i(theta_i)
    Gaussian copula:
        c(u) = MVN(Phi^{-1}(u); 0, R) / prod_i phi(Phi^{-1}(u_i))
    so
        log p(theta) = -0.5 z^T (R^{-1} - I) z - 0.5 log|R|
                       + sum_i log f_i(theta_i),
    with z_i = Phi^{-1}(F_i(theta_i)).

    Fitting choices:
      - Marginals f_i, F_i: univariate Gaussian KDE on a per-axis subsample
        of NROY points of size `n_kde`. KDE gives a smooth, differentiable
        log-density (important: NUTS needs continuous gradients); the
        subsample keeps per-step MCMC cost at O(d * n_kde) erf calls.
      - Bandwidths: Silverman's rule h_i = 1.06 * sigma_i * N^{-1/5} using
        the full NROY sample std (more accurate than subsample std).
      - Correlation matrix R: estimated in z-space via mid-rank empirical
        CDF on ALL NROY points (z_i = Phi^{-1}((rank_i + 0.5) / N)), then
        normalised to unit diagonal. Mid-rank is the standard, robust
        copula-R estimator; it differs from the KDE CDF by O(h), far
        below the sample-size uncertainty in R.

    data_np: (N, d) NROY points in original parameter units.
    Returns a dict of torch tensors ready for `_copula_log_prob`.
    """
    rng = np.random.default_rng(random_state)
    data_np = np.asarray(data_np, dtype=np.float64)
    N, d = data_np.shape

    # ---- 1. per-axis KDE subsample -----------------------------------
    if N > n_kde:
        idx = rng.choice(N, size=n_kde, replace=False)
        kde_data = data_np[idx]                          # (n_kde, d)
    else:
        kde_data = data_np
    M = kde_data.shape[0]

    # ---- 2. bandwidths (Silverman / Scott / scalar) ------------------
    sigma = data_np.std(axis=0, ddof=1)
    if bandwidth == "silverman":
        h = 1.06 * sigma * N ** (-1.0 / 5.0)
    elif bandwidth == "scott":
        h = 1.059 * sigma * N ** (-1.0 / (d + 4.0))  # d-aware Scott, rarely used here
    else:
        h = np.full(d, float(bandwidth))
    # Guard zero-variance axes (should have been caught upstream)
    h = np.maximum(h, 1e-8)

    # ---- 3. rank-Gaussianise all NROY points for R estimation --------
    # mid-rank empirical CDF: F_i(theta_i) = (rank_i + 0.5) / N
    ranks = np.argsort(np.argsort(data_np, axis=0), axis=0).astype(np.float64)
    F_all = (ranks + 0.5) / N                            # (N, d)
    z_all = _scipy_norm.ppf(np.clip(F_all, f_eps, 1.0 - f_eps))  # (N, d)

    # ---- 4. correlation matrix R + shrinkage -------------------------
    R = np.cov(z_all, rowvar=False, ddof=1)
    sigma_R = np.sqrt(np.diag(R))
    R = R / sigma_R[:, None] / sigma_R[None, :]          # force unit diagonal
    if corr_shrink > 0.0:
        R = (1.0 - corr_shrink) * R + corr_shrink * np.eye(d)

    # ---- 5. Cholesky (R = L L^T) + log|R| ----------------------------
    L_R = np.linalg.cholesky(R)
    log_det_R = 2.0 * np.sum(np.log(np.diag(L_R)))
    eigvals = np.linalg.eigvalsh(R)

    return {
        # torch caches for MCMC
        "kde_data": torch.tensor(kde_data.T, dtype=torch.float32),   # (d, M)
        "kde_h":    torch.tensor(h,          dtype=torch.float32),   # (d,)
        "kde_M":    torch.tensor(float(M),   dtype=torch.float32),
        "L_R":      torch.tensor(L_R,        dtype=torch.float32),   # (d, d) lower
        "half_logdet_R": torch.tensor(0.5 * log_det_R, dtype=torch.float32),
        "F_eps":    torch.tensor(f_eps,      dtype=torch.float32),
        # numpy diagnostics
        "R_eig_min": float(eigvals.min()),
        "R_eig_max": float(eigvals.max()),
        "R_cond":    float(eigvals.max() / max(eigvals.min(), 1e-30)),
        "h_np":      h,
        "M":         int(M),
        "N":         int(N),
    }


def _copula_log_prob(theta, cop):
    """Differentiable log-density of a Gaussian copula with Gaussian-KDE
    marginals.  Returns log p(theta) up to a theta-independent constant.

    Evaluation:
        u_ik = (theta_i - kde_data_ik) / h_i       shape (d, M)
        f_i(theta_i) = (1/(M h_i)) sum_k phi(u_ik)
        F_i(theta_i) = (1/M) sum_k Phi(u_ik)
        z_i          = Phi^{-1}(F_i(theta_i))
        log p(theta) = -0.5 (||L_R^{-1} z||^2 - ||z||^2) - 0.5 log|R|
                       + sum_i log f_i(theta_i)

    Subtraction of ||z||^2 cancels the per-axis standard-normal denominator
    in the copula density (so marginals in p_i carry the skew, not the
    MVN). That matches the Sklar factorisation exactly.
    """
    u = (theta.unsqueeze(-1) - cop["kde_data"]) / cop["kde_h"].unsqueeze(-1)   # (d, M)

    # log f_i: log-sum-exp over M kernels, then 1/(M*h_i) normaliser
    log_phi = -0.5 * u ** 2 - 0.5 * math.log(2.0 * math.pi)                    # (d, M)
    log_fi = (torch.logsumexp(log_phi, dim=-1)
              - torch.log(cop["kde_M"]) - torch.log(cop["kde_h"]))             # (d,)

    # F_i via Phi(u) averaged over kernels, clamped for finite Phi^{-1}
    Fi = 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0))).mean(dim=-1)              # (d,)
    Fi = Fi.clamp(cop["F_eps"], 1.0 - cop["F_eps"])
    z  = math.sqrt(2.0) * torch.erfinv(2.0 * Fi - 1.0)                          # (d,)

    # Copula density in z-space: R^{-1} - I quadratic form via one tri-solve
    w = torch.linalg.solve_triangular(
        cop["L_R"], z.unsqueeze(-1), upper=False
    ).squeeze(-1)                                                              # (d,)
    log_copula = -0.5 * ((w ** 2).sum() - (z ** 2).sum()) - cop["half_logdet_R"]

    return log_copula + log_fi.sum()


# ============================================================
# Logspline (P-spline) copula prior — replacement for the per-axis
# Silverman Gaussian KDE used above.  See the block comment at
# USE_LOGSPLINE_MARGINALS in the settings section for motivation.
# ============================================================
def _fit_logspline_1d(data, lo, hi,
                      n_interior=14, smooth_lambda=5.0, n_grid=1000):
    """Penalised cubic B-spline log-density on the closed interval [lo, hi].

    Parameterisation:
        log f(x) = sum_j beta_j * B_j(x),  x in [lo, hi]
    Clamped cubic B-spline basis (multiplicity (k+1) at each boundary) so
    that density at x = lo and x = hi is a free parameter (captures NROY
    boundary pile-up).  beta is fit by penalised MLE with second-difference
    penalty (P-spline):
        max_beta  sum_i B(x_i) beta - N log Z(beta) - lambda || D^2 beta ||^2
    Z(beta) = integral_lo^hi exp(B(x) beta) dx, computed by Simpson's rule
    on a fine grid.

    Returns (x_grid, f_grid, ok_flag) with f_grid normalised so that
        integral_lo^hi f_grid dx = 1.
    """
    from scipy.interpolate import BSpline as _BSpline_
    from scipy.integrate import simpson as _simpson_
    from scipy.optimize import minimize as _minimize_

    k = 3
    # Clamped knot vector: k+1 repeats at each boundary so design_matrix
    # produces a full-rank basis including boundary support.
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    t = np.concatenate([np.full(k + 1, lo), interior, np.full(k + 1, hi)])
    n_basis = len(t) - k - 1

    data = np.clip(np.asarray(data, dtype=np.float64), lo, hi)
    N = len(data)

    x_grid = np.linspace(lo, hi, n_grid)
    B_grid = np.asarray(
        _BSpline_.design_matrix(x_grid, t, k, extrapolate=False).todense()
    )                                                             # (G, n_basis)
    B_data = np.asarray(
        _BSpline_.design_matrix(data, t, k, extrapolate=False).todense()
    )                                                             # (N, n_basis)

    D = np.diff(np.eye(n_basis), n=2, axis=0)
    P = D.T @ D
    B_data_sum = B_data.sum(axis=0)

    def neg_lp(beta):
        eta_g = B_grid @ beta
        m = eta_g.max()
        Zs = _simpson_(np.exp(eta_g - m), x=x_grid)
        log_Z = np.log(Zs) + m
        return -((B_data @ beta).sum()
                 - N * log_Z
                 - smooth_lambda * (beta @ P @ beta))

    def grad(beta):
        eta_g = B_grid @ beta
        m = eta_g.max()
        fu = np.exp(eta_g - m)
        Zs = _simpson_(fu, x=x_grid)
        p_norm = fu / Zs
        int_Bp = _simpson_(B_grid * p_norm[:, None], x=x_grid, axis=0)
        return -(B_data_sum - N * int_Bp - 2.0 * smooth_lambda * (P @ beta))

    res = _minimize_(
        neg_lp, np.zeros(n_basis), jac=grad, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8},
    )
    beta_hat = res.x
    eta_g = B_grid @ beta_hat
    m = eta_g.max()
    Z = _simpson_(np.exp(eta_g - m), x=x_grid) * np.exp(m)
    f_grid = np.exp(eta_g) / Z
    return x_grid, f_grid, bool(res.success)


def fit_logspline_copula(data_np, prior_lower, prior_upper,
                         n_interior=14, smooth_lambda=5.0, n_grid=1000,
                         corr_shrink=0.0, f_eps=1e-6, axis_names=None):
    """Fit a Gaussian copula with per-axis logspline (P-spline) marginals.

    Unlike `fit_gaussian_copula`:
      - Marginals f_i are bounded to [prior_lower[i], prior_upper[i]] by
        construction (support enforced by the clamped B-spline knot vector).
      - No Silverman boundary bias; no subsample noise.
      - MCMC evaluation is O(log G) per axis via precomputed grids +
        searchsorted + linear interp (see _logspline_copula_log_prob).

    Dependence structure R is estimated identically to `fit_gaussian_copula`:
    mid-rank empirical CDF -> rank-Gaussianisation -> sample correlation.
    (Using the fitted logspline CDF for rank-Gaussianisation instead of
     the mid-rank estimator is possible but differs by O(h_kde) in the
     old method and by O(n_grid^-2) here — negligible vs. sample noise in
     R.  Keeping mid-rank preserves the validated copula-R estimator.)

    Returns a dict of torch tensors ready for `_logspline_copula_log_prob`,
    plus numpy diagnostics.
    """
    data_np = np.asarray(data_np, dtype=np.float64)
    N, d = data_np.shape
    prior_lower = np.asarray(prior_lower, dtype=np.float64)
    prior_upper = np.asarray(prior_upper, dtype=np.float64)

    # ---- 1. per-axis logsplines -----------------------------------------
    x_grids = np.zeros((d, n_grid), dtype=np.float64)
    f_grids = np.zeros((d, n_grid), dtype=np.float64)
    cdf_grids = np.zeros((d, n_grid), dtype=np.float64)
    n_bad_fit = 0
    for i in range(d):
        lo = float(prior_lower[i])
        hi = float(prior_upper[i])
        # Tighten to empirical extent if data is strictly inside bounds
        # (avoids fitting a long flat tail of zero density outside NROY).
        emp_lo = float(data_np[:, i].min())
        emp_hi = float(data_np[:, i].max())
        lo = max(lo, emp_lo - 1e-9)
        hi = min(hi, emp_hi + 1e-9)
        xg, fg, ok = _fit_logspline_1d(
            data_np[:, i], lo, hi,
            n_interior=n_interior,
            smooth_lambda=smooth_lambda,
            n_grid=n_grid,
        )
        x_grids[i] = xg
        f_grids[i] = fg
        # Cumulative trapezoidal -> CDF on grid (monotone, starts at 0)
        dx = np.diff(xg)
        cdf = np.zeros(n_grid)
        cdf[1:] = np.cumsum(0.5 * (fg[:-1] + fg[1:]) * dx)
        cdf /= cdf[-1]                                            # enforce CDF(hi) = 1
        cdf_grids[i] = cdf
        if not ok:
            n_bad_fit += 1
            if axis_names is not None:
                print(f"    WARNING: logspline MLE did not converge cleanly "
                      f"on axis '{axis_names[i]}'")

    # ---- 2. rank-Gaussianise via mid-rank empirical CDF -----------------
    ranks = np.argsort(np.argsort(data_np, axis=0), axis=0).astype(np.float64)
    F_all = (ranks + 0.5) / N
    z_all = _scipy_norm.ppf(np.clip(F_all, f_eps, 1.0 - f_eps))   # (N, d)

    # ---- 3. correlation matrix R + shrinkage ---------------------------
    R = np.cov(z_all, rowvar=False, ddof=1)
    sigma_R = np.sqrt(np.diag(R))
    R = R / sigma_R[:, None] / sigma_R[None, :]
    if corr_shrink > 0.0:
        R = (1.0 - corr_shrink) * R + corr_shrink * np.eye(d)

    # ---- 4. Cholesky + diagnostics --------------------------------------
    try:
        L_R = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        jitter = 1e-8
        while True:
            try:
                L_R = np.linalg.cholesky(R + jitter * np.eye(d))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                if jitter > 1e-2:
                    raise
    half_logdet_R = float(np.log(np.diag(L_R)).sum())
    eigvals = np.linalg.eigvalsh(R)

    return {
        "marginal_type":  "logspline",
        # torch caches for MCMC (float32 matches the GP backend dtype)
        "x_grid":         torch.tensor(x_grids,   dtype=torch.float32),   # (d, G)
        "f_grid":         torch.tensor(f_grids,   dtype=torch.float32),   # (d, G)
        "cdf_grid":       torch.tensor(cdf_grids, dtype=torch.float32),   # (d, G)
        "L_R":            torch.tensor(L_R,        dtype=torch.float32),  # (d, d)
        "half_logdet_R":  torch.tensor(half_logdet_R, dtype=torch.float32),
        "F_eps":          torch.tensor(float(f_eps), dtype=torch.float32),
        # Per-axis gather indices. Cached once so the NUTS inner loop does
        # not recreate it from scratch every call to _logspline_copula_log_prob.
        "arange_d":       torch.arange(d, dtype=torch.long),
        # numpy diagnostics
        "R_eig_min":      float(eigvals.min()),
        "R_eig_max":      float(eigvals.max()),
        "R_cond":         float(eigvals.max() / max(eigvals.min(), 1e-30)),
        "N":              int(N),
        "n_grid":         int(n_grid),
        "n_interior":     int(n_interior),
        "smooth_lambda":  float(smooth_lambda),
        "n_bad_fit":      int(n_bad_fit),
    }


def _logspline_copula_log_prob(theta, cop):
    """Differentiable log-density of a Gaussian copula with per-axis
    logspline marginals; O(log G) per axis via precomputed grids.

    Evaluation:
        bin          = searchsorted(x_grid, theta)              # (d,)
        dx           = x_hi - x_lo
        slope        = (f_hi - f_lo) / dx
        f_i(theta_i) = f_lo + slope * (theta_i - x_lo)          # linear in theta
        F_i(theta_i) = F_lo + f_lo * (theta_i - x_lo)
                             + 0.5 * slope * (theta_i - x_lo)^2 # quadratic
        z_i          = Phi^{-1}(F_i)
        log p(theta) = -0.5 (||L_R^{-1} z||^2 - ||z||^2)
                       - half_logdet_R
                       + sum_i log f_i(theta_i)

    F_i is the analytic integral of the piecewise-linear f_i, so
    dF_i / dtheta_i == f_i(theta_i) at every evaluation — the gradient
    that NUTS sees is consistent between f_i and F_i.
    """
    x_grid   = cop["x_grid"]        # (d, G)
    f_grid   = cop["f_grid"]        # (d, G)
    cdf_grid = cop["cdf_grid"]      # (d, G)
    L_R      = cop["L_R"]
    half_logdet_R = cop["half_logdet_R"]
    F_eps    = cop["F_eps"]
    d, G = x_grid.shape

    theta_c = theta.to(x_grid.dtype)

    # Per-axis bin search (right edge of bracketing interval).  With 2D
    # sorted_sequence, searchsorted operates along the last dim per row.
    idx_hi = torch.searchsorted(
        x_grid, theta_c.unsqueeze(-1).contiguous()
    ).squeeze(-1)                                                 # (d,)
    idx_hi = idx_hi.clamp(min=1, max=G - 1)
    idx_lo = idx_hi - 1

    # Gather per-axis bin edges and grid values.
    arange_d = cop["arange_d"]
    x_lo = x_grid[arange_d, idx_lo]
    x_hi = x_grid[arange_d, idx_hi]
    f_lo = f_grid[arange_d, idx_lo]
    f_hi = f_grid[arange_d, idx_hi]
    F_lo = cdf_grid[arange_d, idx_lo]

    dx     = (x_hi - x_lo).clamp_min(1e-30)
    slope  = (f_hi - f_lo) / dx
    delta  = theta_c - x_lo

    # Piecewise-linear f, piecewise-quadratic F (analytic integral of f).
    f_th   = (f_lo + slope * delta).clamp_min(1e-30)
    F_th   = F_lo + f_lo * delta + 0.5 * slope * delta ** 2
    F_th   = F_th.clamp(F_eps, 1.0 - F_eps)

    # Inverse standard-normal CDF.
    z = math.sqrt(2.0) * torch.erfinv(2.0 * F_th - 1.0)          # (d,)

    # Copula density: -0.5 (w'w - z'z) - log|L_R|  via one tri-solve.
    w = torch.linalg.solve_triangular(
        L_R, z.unsqueeze(-1), upper=False
    ).squeeze(-1)                                                # (d,)
    log_copula = -0.5 * ((w ** 2).sum() - (z ** 2).sum()) - half_logdet_R

    log_f = torch.log(f_th).sum()
    return log_copula + log_f


def _safe_ratio_denominator(denom, eps=1e-8):
    """Keep ratio denominators away from zero without flipping sign."""
    sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
    return torch.where(denom.abs() < eps, sign * eps, denom)


def _local_output_covariance(vars_, residual_corr, idxs):
    """Approximate local joint covariance from per-output vars and global residual corr."""
    idx_t = torch.as_tensor(idxs, dtype=torch.long, device=vars_.device)
    local_var = vars_.index_select(0, idx_t).clamp(min=1e-12)
    local_std = torch.sqrt(local_var)
    corr = residual_corr.to(device=vars_.device, dtype=vars_.dtype)
    corr_sub = corr.index_select(0, idx_t).index_select(1, idx_t)
    cov = corr_sub * torch.outer(local_std, local_std)
    eye = torch.eye(len(idxs), dtype=vars_.dtype, device=vars_.device)
    return 0.5 * (cov + cov.T) + ATRIAL_COV_JITTER * eye


def _ratio_moments(mean_vec, cov):
    """Cubature-propagated mean and variance of the active-emptying fraction.

    `mean_vec` and `cov` describe the joint Gaussian (V_min, V_max, V_pre)
    GP marginals. Returns (ratio_mean, ratio_var) for
        r = (V_pre - V_min) / (V_max - V_min)
    so the caller can fold it into the same Gaussian likelihood machinery as
    the direct targets (total_var = obs_var + ratio_var).
    """
    n_dim = mean_vec.numel()
    chol = torch.linalg.cholesky(cov)
    disp = math.sqrt(float(n_dim)) * chol.T
    sigma_points = torch.cat(
        (mean_vec.unsqueeze(0) + disp, mean_vec.unsqueeze(0) - disp), dim=0
    )
    ratio = (
        sigma_points[:, 2] - sigma_points[:, 0]
    ) / _safe_ratio_denominator(sigma_points[:, 1] - sigma_points[:, 0])
    ratio_mean = ratio.mean()
    ratio_var = ((ratio - ratio_mean) ** 2).mean().clamp_min(1e-12)
    return ratio_mean, ratio_var


def _format_max_sig_figs(value, sig_figs=4, max_decimals=3):
    """Format numeric labels with capped sig figs and decimal places."""
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"

    rounded = float(f"{value:.{sig_figs}g}")
    abs_rounded = abs(rounded)
    magnitude = int(math.floor(math.log10(abs_rounded))) if abs_rounded > 0 else 0
    decimals = min(max(sig_figs - magnitude - 1, 0), max_decimals)

    if decimals > 0:
        label = f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")
    else:
        label = f"{rounded:.0f}"

    # Keep one decimal when rounding collapses a non-integer to an integer
    # label, e.g. 29.999 -> 30.0, while staying within the sig-fig cap.
    if (
        "." not in label
        and not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-12)
        and abs_rounded < 10 ** (sig_figs - 1)
    ):
        label = f"{rounded:.1f}"

    return label


def plot_prior_marginals_vs_nroy(copula_cache, nroy_subset, subset_vars, prior_lower,
                                 prior_upper, out_dir, n_bins=40, n_cols=7):
    """Save per-axis prior marginals overlaid on the empirical NROY histograms.

    For logspline marginals, this reuses the fitted (x_grid, f_grid) already
    stored in `copula_cache`, so no second fitting pass is needed. For KDE
    marginals, it evaluates the Gaussian-KDE density on a plotting grid using
    the cached subsample and bandwidths.
    """
    if copula_cache is None:
        print("  Skipping prior-marginal plot (uniform box prior).")
        return None

    print("  Saving prior marginals vs NROY plot...")

    nroy_subset = np.asarray(nroy_subset, dtype=np.float64)
    prior_lower = np.asarray(prior_lower, dtype=np.float64)
    prior_upper = np.asarray(prior_upper, dtype=np.float64)

    ndim = len(subset_vars)
    n_rows = math.ceil(ndim / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.2 * n_rows))
    axes = np.atleast_2d(axes)

    marginal_type = copula_cache.get("marginal_type", "kde")
    is_logspline_plot = marginal_type == "logspline"
    if is_logspline_plot:
        fig.set_size_inches(2.35 * n_cols, 1.8 * n_rows)
    legend_added = False
    max_sig_fig_formatter = FuncFormatter(
        lambda x, _pos: _format_max_sig_figs(x, sig_figs=4)
    )

    for j, name in enumerate(subset_vars):
        ax = axes[j // n_cols, j % n_cols]
        data_j = nroy_subset[:, j]

        ax.hist(
            data_j, bins=n_bins, density=True,
            color="steelblue",
            alpha=0.2 if is_logspline_plot else 0.25,
            edgecolor="white" if is_logspline_plot else "none",
            linewidth=0.25 if is_logspline_plot else 0.0,
            # label="NROY sample" if is_logspline_plot and not legend_added else None,
        )

        L = float(prior_lower[j])
        U = float(prior_upper[j])
        emp_lo = float(data_j.min())
        emp_hi = float(data_j.max())
        L = max(L, emp_lo - 1e-9)
        U = min(U, emp_hi + 1e-9)

        if marginal_type == "logspline":
            xg = copula_cache["x_grid"][j].detach().cpu().numpy()
            fg = copula_cache["f_grid"][j].detach().cpu().numpy()
            ax.plot(
                xg, fg, color="#0b3d91", lw=1.1,
                # label="Logspline prior" if not legend_added else None,
            )
            ax.set_xlim(L, U)
            ax.set_xticks([L, U])
            ax.xaxis.set_major_formatter(max_sig_fig_formatter)
            ax.set_yticks([])

            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_linewidth(0.6)
                ax.spines[spine].set_color("#4d4d4d")
            ax.tick_params(axis="x", labelsize=12, length=2.5, width=0.6)
            ax.tick_params(axis="y", length=0)
        else:
            xg = np.linspace(L, U, 600)
            kde_data_j = copula_cache["kde_data"][j].detach().cpu().numpy()
            h_j = float(copula_cache["kde_h"][j].item())
            u = (xg[:, None] - kde_data_j[None, :]) / h_j
            fg = np.exp(-0.5 * u ** 2).sum(axis=1)
            fg /= (kde_data_j.size * h_j * math.sqrt(2.0 * math.pi))
            ax.plot(xg, fg, color="crimson", lw=1.3,
                    label="KDE marginal" if not legend_added else None)
            ax.xaxis.set_major_formatter(max_sig_fig_formatter)
            ax.yaxis.set_major_formatter(max_sig_fig_formatter)

        if not is_logspline_plot:
            ax.axvline(L, color="grey", ls=":", lw=0.5, alpha=0.6)
            ax.axvline(U, color="grey", ls=":", lw=0.5, alpha=0.6)
        # ax.set_title(name, fontsize=8)
        if is_logspline_plot:
            ax.set_xlabel(name, fontsize=12, labelpad=2)
            ax.set_ylabel("")
        else:
            ax.tick_params(labelsize=10)
            ax.set_xlabel(name, fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
        legend_added = True

    for k in range(ndim, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12,
                   frameon=False, bbox_to_anchor=(0.5, 1.0))

    if marginal_type == "logspline":
        # title = (
        #     f"Logspline (P-spline) marginals on HM bounds "
        #     f"[n_interior={LOGSPLINE_N_INTERIOR}, lambda={LOGSPLINE_LAMBDA}] "
        #     f"vs NROY empirical distribution"
        # )
        out_name = "copula_marginals_vs_NROY_logspline.png"
    else:
        title = (
            f"Gaussian-KDE copula marginals on HM bounds "
            f"[bandwidth={KDE_BANDWIDTH}] vs NROY empirical distribution"
        )
        out_name = "copula_marginals_vs_NROY_kde.png"

    if is_logspline_plot:
        fig.text(0.006, 0.5, "Density", va="center", rotation="vertical",
                 fontsize=12)
        fig.tight_layout(rect=[0.025, 0, 1, 0.985])
    else:
        # fig.suptitle(title, y=1.005, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_path = os.path.join(out_dir, out_name)
    plot_dpi = LOGSPLINE_PLOT_DPI if is_logspline_plot else 150
    fig.savefig(out_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def _first_existing_path(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    joined = "\n    ".join(candidates)
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n    {joined}"
    )


def make_fast_potential_fn_batched(prior_lo, prior_hi, obs_means_t, obs_vars_t,
                                   bc, copula=None):
    """Batched, manual Matern-3/2 potential.  No gpytorch calls at inference.

    Replaces the per-output Python loop with two batched tensor ops:
      k_star (n_out, n) — one call to _matern32_cross
      variance          — one batched triangular solve via cholesky_solve
    Gradient flows through all of these, so Pyro NUTS autograd works unchanged.

    Prior:
      - copula=None  -> uniform prior on the NROY box [prior_lo, prior_hi]
                        (only the sigmoid reparam Jacobian contributes)
      - copula dict  -> Gaussian copula with per-axis KDE marginals fitted
                        on NROY points, evaluated on theta via
                        _copula_log_prob().  The sigmoid reparam is kept
                        as the unconstrained-space transform; its Jacobian
                        is added so NUTS targets the correct distribution
                        in z.
    """
    log_width = torch.log(prior_hi - prior_lo)

    lengthscale          = bc["lengthscale"]
    outputscale          = bc["outputscale"]
    mean_const           = bc["mean_const"]
    L                    = bc["L"]
    alpha                = bc["alpha"]
    x_mean               = bc["x_mean"]
    x_std                = bc["x_std"]
    y_mean               = bc["y_mean"]
    y_std                = bc["y_std"]
    X_train_scaled       = bc["X_train_scaled"]
    X_train_scaled_norm2 = bc["X_train_scaled_norm2"]
    residual_corr        = bc["residual_corr"]

    def _potential(z_dict):
        z = z_dict["theta"]
        sig_z = torch.sigmoid(z)
        theta = prior_lo + sig_z * (prior_hi - prior_lo)

        log_det_jac = (
            torch.nn.functional.logsigmoid(z)
            + torch.nn.functional.logsigmoid(-z)
            + log_width
        ).sum()

        if copula is not None:
            if copula.get("marginal_type") == "logspline":
                log_prior = _logspline_copula_log_prob(theta, copula)
            else:
                log_prior = _copula_log_prob(theta, copula)
        else:
            log_prior = torch.zeros((), dtype=theta.dtype, device=theta.device)

        x_t = (theta - x_mean) / x_std                                   # (d,)

        k_star = _matern32_cross_fast(
            x_t, X_train_scaled, X_train_scaled_norm2,
            lengthscale, outputscale,
        )                                                                # (n_out, n)

        mu_latent = mean_const + (k_star * alpha).sum(dim=-1)            # (n_out,)

        # var_latent[i] = outputscale[i] - k*_i^T K_i^{-1} k*_i
        #              = outputscale[i] - ||L_i^{-1} k*_i||^2
        # One batched forward-substitution replaces the two tri-solves of
        # cholesky_solve, and we never materialise v = K^{-1} k*.
        w = torch.linalg.solve_triangular(
            L, k_star.unsqueeze(-1), upper=False,
        ).squeeze(-1)                                                    # (n_out, n)
        var_latent = (outputscale - (w ** 2).sum(dim=-1)).clamp(min=1e-10)

        mus   = mu_latent * y_std + y_mean                               # (n_out,)
        vars_ = var_latent * y_std ** 2                                  # (n_out,)

        total_var = (obs_vars_t + vars_).clamp(min=1e-10)
        z_norm = (obs_means_t - mus) / total_var.sqrt()

        gaussian_mask = torch.ones_like(mus, dtype=torch.bool)
        gaussian_mask[list(ATRIAL_GAUSSIAN_SKIP)] = False
        ll = -0.5 * (
            z_norm[gaussian_mask] ** 2 + torch.log(total_var[gaussian_mask])
        ).sum()

        # Atrial active-emptying fraction enters as a Gaussian likelihood term
        # on a derived ratio. The cubature step propagates the joint emulator
        # uncertainty across (V_min, V_max, V_pre) into a single ratio
        # variance, which is then summed with the target variance exactly as
        # for the direct outputs.
        rest_la_mean = torch.stack((mus[REST_LA_MIN_IDX], mus[REST_LA_MAX_IDX], mus[REST_LA_PRE_IDX]))
        rest_ra_mean = torch.stack((mus[REST_RA_MIN_IDX], mus[REST_RA_MAX_IDX], mus[REST_RA_PRE_IDX]))
        rest_la_cov = _local_output_covariance(
            vars_, residual_corr, (REST_LA_MIN_IDX, REST_LA_MAX_IDX, REST_LA_PRE_IDX)
        )
        rest_ra_cov = _local_output_covariance(
            vars_, residual_corr, (REST_RA_MIN_IDX, REST_RA_MAX_IDX, REST_RA_PRE_IDX)
        )
        rest_la_r_mean, rest_la_r_var = _ratio_moments(rest_la_mean, rest_la_cov)
        rest_ra_r_mean, rest_ra_r_var = _ratio_moments(rest_ra_mean, rest_ra_cov)

        rest_la_total_var = (obs_vars_t[REST_LA_PRE_IDX] + rest_la_r_var).clamp(min=1e-10)
        rest_ra_total_var = (obs_vars_t[REST_RA_PRE_IDX] + rest_ra_r_var).clamp(min=1e-10)
        
        exercise_la_mean = torch.stack((mus[EXERCISE_LA_MIN_IDX], mus[EXERCISE_LA_MAX_IDX], mus[EXERCISE_LA_PRE_IDX]))
        exercise_ra_mean = torch.stack((mus[EXERCISE_RA_MIN_IDX], mus[EXERCISE_RA_MAX_IDX], mus[EXERCISE_RA_PRE_IDX]))
        exercise_la_cov = _local_output_covariance(
            vars_, residual_corr, (EXERCISE_LA_MIN_IDX, EXERCISE_LA_MAX_IDX, EXERCISE_LA_PRE_IDX)
        )
        exercise_ra_cov = _local_output_covariance(
            vars_, residual_corr, (EXERCISE_RA_MIN_IDX, EXERCISE_RA_MAX_IDX, EXERCISE_RA_PRE_IDX)
        )
        exercise_la_r_mean, exercise_la_r_var = _ratio_moments(exercise_la_mean, exercise_la_cov)
        exercise_ra_r_mean, exercise_ra_r_var = _ratio_moments(exercise_ra_mean, exercise_ra_cov)

        exercise_la_total_var = (obs_vars_t[EXERCISE_LA_PRE_IDX] + exercise_la_r_var).clamp(min=1e-10)
        exercise_ra_total_var = (obs_vars_t[EXERCISE_RA_PRE_IDX] + exercise_ra_r_var).clamp(min=1e-10)
        
        
        ll = ll - 0.5 * (
            (obs_means_t[REST_LA_PRE_IDX] - rest_la_r_mean) ** 2 / rest_la_total_var
            + torch.log(rest_la_total_var)
            + (obs_means_t[REST_RA_PRE_IDX] - rest_ra_r_mean) ** 2 / rest_ra_total_var
            + torch.log(rest_ra_total_var)
            + (obs_means_t[EXERCISE_LA_PRE_IDX] - exercise_la_r_mean) ** 2 / exercise_la_total_var
            + torch.log(exercise_la_total_var)
            + (obs_means_t[EXERCISE_RA_PRE_IDX] - exercise_ra_r_mean) ** 2 / exercise_ra_total_var
            + torch.log(exercise_ra_total_var)
        )

        ll = torch.nan_to_num(ll, nan=-1e8, posinf=-1e8, neginf=-1e8)
        log_prior = torch.nan_to_num(log_prior, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return -(ll + log_prior + log_det_jac)

    return _potential


def batched_predict_mean(theta_batch, bc):
    """Mean-only batched prediction for N points  × n_out outputs.

    theta_batch: (B, d)    — unstandardised params
    returns:     (B, n_out) of final predictions (post y-inverse-transform)
    """
    X_train     = bc["X_train"]            # (n, d)
    lengthscale = bc["lengthscale"]        # (n_out, d)
    outputscale = bc["outputscale"]        # (n_out,)
    mean_const  = bc["mean_const"]         # (n_out,)
    alpha       = bc["alpha"]              # (n_out, n)
    x_mean      = bc["x_mean"]
    x_std       = bc["x_std"]
    y_mean      = bc["y_mean"]
    y_std       = bc["y_std"]

    x_t = (theta_batch - x_mean) / x_std                                 # (B, d)
    # diff: (B, n_out, n, d)
    diff = (x_t.unsqueeze(1).unsqueeze(2) - X_train.unsqueeze(0).unsqueeze(0)) \
           / lengthscale.unsqueeze(0).unsqueeze(2)
    r = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-30)                      # (B, n_out, n)
    k_star = outputscale.view(1, -1, 1) * (1.0 + SQRT3 * r) * torch.exp(-SQRT3 * r)
    mu_latent = mean_const + (k_star * alpha.unsqueeze(0)).sum(dim=-1)   # (B, n_out)
    return mu_latent * y_std + y_mean


# ================================================================
# OUTPUT NAMES (same order as union emulators / simulator outputs)
# ================================================================
OUTPUT_NAMES_PER_STATE = [
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV",
    "ESV", "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure",
    "Min_RV_Pressure", "Min_RA_Volume", "Max_RA_Volume",
    "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume",
    "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening", "Pre_LA_Contraction_Volume",
    "Pre_RA_Contraction_Volume", "LV_Pressure_Deriv",
    "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation",
    "PaO2", "PaCO2"
]
output_names = (
    [f"Rest_{name}" for name in OUTPUT_NAMES_PER_STATE]
    + [f"Exercise_{name}" for name in OUTPUT_NAMES_PER_STATE]
)

# ================================================================
# UNION TARGETS: {output_name: (population_mean, population_variance)}
# ================================================================
observation = {
# Rest
"Rest_Heart_Rate": (1.23, 0.05), "Rest_Systolic_Pressure": (123, 324), "Rest_Diastolic_Pressure": (76.7, 65.61),
"Rest_EDV": (152.1, 767.29), "Rest_ESV": (62.3, 243.36), "Rest_Max_RV_Volume": (151.9, 1004.89),
"Rest_Min_RV_Volume": (64.4, 299.29), "Rest_Max_RV_Pressure": (22.5, 56.25), "Rest_Min_RV_Pressure": (4.0, 9.0),
"Rest_Min_RA_Volume": (45.7, 125.44), "Rest_Max_RA_Volume": (92.4, 380.25), "Rest_Max_RA_Pressure_Atrial_contraction": (8.0, 9.0),
"Rest_Max_RA_Pressure_Tricuspid_Opening": (5.0, 9.0), "Rest_Min_LA_Volume": (30.6, 84.64), "Rest_Max_LA_Volume": (68.3, 306.25),
"Rest_Max_LA_Pressure_Atrial_contraction": (13.0, 9.0), "Rest_Max_LA_Pressure_Mitral_Opening": (12.0, 9.0),
"Rest_Pre_LA_Contraction_Volume": ATRIAL_RATIO_TARGET, "Rest_Pre_RA_Contraction_Volume": ATRIAL_RATIO_TARGET,
"Rest_LV_Pressure_Deriv": (1461.0, 146689.0), "Rest_RV_Pressure_Deriv": (271.0, 3025.0),
"Rest_Tidal_Volume": (0.850, 0.16), "Rest_Minute_Ventilation": (11.4, 15.21), "Rest_PaO2": (102.3, 125.44),
"Rest_PaCO2": (35.5, 24.01),

# Exercise
"Exercise_Heart_Rate": (2.58, 0.12), "Exercise_Systolic_Pressure": (165, 529), "Exercise_Diastolic_Pressure": (76.4, 82.81),
"Exercise_EDV": (145.5, 681.21), "Exercise_ESV": (45.5, 75.69), "Exercise_Max_RV_Volume": (139.4, 681.21),
"Exercise_Min_RV_Volume": (40.3, 112.36), "Exercise_Max_RV_Pressure": (29.5, 56.25), "Exercise_Min_RV_Pressure": (9.9, 31.36),
"Exercise_Min_RA_Volume": (27.9, 25.0), "Exercise_Max_RA_Volume": (77.3, 342.25), "Exercise_Max_RA_Pressure_Atrial_contraction": (12, 16),
"Exercise_Max_RA_Pressure_Tricuspid_Opening": (11, 16), "Exercise_Min_LA_Volume": (23.0, 94.09), "Exercise_Max_LA_Volume": (66.3, 388.09),
"Exercise_Max_LA_Pressure_Atrial_contraction": (19, 49), "Exercise_Max_LA_Pressure_Mitral_Opening": (19, 64),
"Exercise_Pre_LA_Contraction_Volume": ATRIAL_RATIO_TARGET, "Exercise_Pre_RA_Contraction_Volume": ATRIAL_RATIO_TARGET,
"Exercise_LV_Pressure_Deriv": (1750, 272484), "Exercise_RV_Pressure_Deriv": (713, 12100),
"Exercise_Tidal_Volume": (2.22, 0.4096), "Exercise_Minute_Ventilation": (62.6, 320.41), "Exercise_PaO2": (97.2, 36.0),
"Exercise_PaCO2": (38.4, 6.76)
}

REST_LA_PRE_DISPLAY_MEAN, REST_LA_PRE_DISPLAY_STD = _propagated_vpre_display_stats(
    observation["Rest_Min_LA_Volume"][0],
    observation["Rest_Min_LA_Volume"][1],
    observation["Rest_Max_LA_Volume"][0],
    observation["Rest_Max_LA_Volume"][1],
    observation["Rest_Pre_LA_Contraction_Volume"][0],
    observation["Rest_Pre_LA_Contraction_Volume"][1],
)
REST_RA_PRE_DISPLAY_MEAN, REST_RA_PRE_DISPLAY_STD = _propagated_vpre_display_stats(
    observation["Rest_Min_RA_Volume"][0],
    observation["Rest_Min_RA_Volume"][1],
    observation["Rest_Max_RA_Volume"][0],
    observation["Rest_Max_RA_Volume"][1],
    observation["Rest_Pre_RA_Contraction_Volume"][0],
    observation["Rest_Pre_RA_Contraction_Volume"][1],
)

EXERCISE_LA_PRE_DISPLAY_MEAN, EXERCISE_LA_PRE_DISPLAY_STD = _propagated_vpre_display_stats(
    observation["Exercise_Min_LA_Volume"][0],
    observation["Exercise_Min_LA_Volume"][1],
    observation["Exercise_Max_LA_Volume"][0],
    observation["Exercise_Max_LA_Volume"][1],
    observation["Exercise_Pre_LA_Contraction_Volume"][0],
    observation["Exercise_Pre_LA_Contraction_Volume"][1],
)
EXERCISE_RA_PRE_DISPLAY_MEAN, EXERCISE_RA_PRE_DISPLAY_STD = _propagated_vpre_display_stats(
    observation["Exercise_Min_RA_Volume"][0],
    observation["Exercise_Min_RA_Volume"][1],
    observation["Exercise_Max_RA_Volume"][0],
    observation["Exercise_Max_RA_Volume"][1],
    observation["Exercise_Pre_RA_Contraction_Volume"][0],
    observation["Exercise_Pre_RA_Contraction_Volume"][1],
)

# ================================================================
# CALIBRATION PARAMETER SUBSET (from DGSM sensitivity analysis) # change
# ================================================================
subset_vars_set = {
    # Rest_only
    'ahead1', 'Cvb_O2_n', 'Emax_la', 'f_ab_max', 'fes_inf', 'fes_min', 'fes_o', 'fev_inf', 'Io_sv', 'kcc_sv', 'kes',
    'P_n', 'R_pp', 'Vu_ra',

    # Exercise Only
    'alpha2', 'beta2', 'C_pv', 'G_ap', 'GEmax_lv', 'GEmax_rv', 'GR_amp', 'GV_dead', 'GV_sv', 'K2', 'phi_max', 'R_tr',
    'Rvc_n', 's', 'Wp_v',

    # Overlap
    'a2', 'C2', 'C_O2_param1', 'C_O2_param2', 'C_sv', 'E_rs', 'Emax_lv0', 'Emax_rv0', 'fev_o', 'GT_s', 'GT_v', 'KcCO2',
    'KE_la', 'KE_lv', 'KE_ra', 'KE_rv', 'l', 'MO2_bp', 'P0_la', 'P0_lv', 'P0_rv', 'PaCO2_n', 'PAMO2_nominal', 'r',
    'R_po', 'R_rs', 'R_sa', 'rise_time_ven', 'scale_param1', 'scale_param4', 'T0', 'theta_po_max', 'theta_tr_max',
    'V0_dead', 'V_nominal', 'V_scale', 'VA_rest', 'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_rv', 'Vu_sv0'
}



# ============================================================
# 1. LOAD HISTORY MATCHING RESULTS
# ============================================================
print("=" * 60)
print("STEP 1 -- Loading history matching results")
print("=" * 60)
print(f"  HM artifacts dir: {HM_ARTIFACTS_DIR}")
print(f"  Emulator dir: {EMULATOR_DIR}")
print(f"  Run dir: {out_dir}")
print(f"  CPU threads per active chain/process: {THREADS_PER_CHAIN_EFFECTIVE}")

nroy_points_np = np.load(
    _first_existing_path(
        [
            os.path.join(HM_ARTIFACTS_DIR, f"NROY_Points_union_all_{PERCENT}.npy"),
            # f"NROY_Points_union_{PERCENT}.npy",
            # f"NROY_Points_rest_union_{PERCENT}.npy",
            # f"NROY_Points_union_{PERCENT}.npy",
        ],
        "union NROY points",
    )
)
nroy_params_dict = np.load(
    _first_existing_path(
        [
            os.path.join(HM_ARTIFACTS_DIR, f"NROY_Params_union_all_{PERCENT}.npy"),
            # f"NROY_Params_union_{PERCENT}.npy",
            # f"NROY_Params_rest_union_{PERCENT}.npy",
            # f"NROY_Params_union_{PERCENT}.npy",
        ],
        "union NROY parameter bounds",
    ),
    allow_pickle=True # change
).item()

# Parameter ordering from the HM bounds dict (matches sp["names"])
all_param_names = list(nroy_params_dict.keys())

# Calibration subset, preserved in the order they appear in the full vector
subset_vars = [n for n in all_param_names if n in subset_vars_set]
param_idx = [all_param_names.index(n) for n in subset_vars]
ndim = len(subset_vars)

# Extract calibration columns from NROY points
nroy_subset = nroy_points_np[:, param_idx].astype(np.float32)
n_nroy = nroy_subset.shape[0]

# NROY bounds for priors
prior_lower = np.array(
    [nroy_params_dict[n][0] for n in subset_vars], dtype=np.float32
)
prior_upper = np.array(
    [nroy_params_dict[n][1] for n in subset_vars], dtype=np.float32
)

# Warn about degenerate bounds
range_width = prior_upper - prior_lower
narrow = range_width < 1e-12
if narrow.any():
    names_narrow = [subset_vars[i] for i in np.where(narrow)[0]]
    print(f"  WARNING: {len(names_narrow)} params have degenerate NROY "
          f"range and will be effectively fixed: {names_narrow}")

print(f"  NROY points loaded:    {n_nroy}")
print(f"  Calibration params:    {ndim}")
print(f"  Full parameter vector: {len(all_param_names)}")

# ============================================================
# 1b. FIT GAUSSIAN COPULA PRIOR ON NROY POINTS
# ============================================================
# The HM bounds dict only retains per-axis min/max of surviving points, so a
# uniform box prior throws away all joint-space information. We instead fit a
# Gaussian copula to the NROY cloud:
#     p(theta) = c(F_1..F_d) * prod_i f_i(theta_i)
# with per-axis Gaussian-KDE marginals (f_i, F_i) and a single correlation
# matrix R modelling the dependency in rank-Gaussianised space. Motivation
# (replaces the previous full-cov GMM + BIC approach):
#   1. A full-cov GMM has d + d(d+1)/2 ~= 1891 free params per component at
#      d is large, so BIC's p*log(N) penalty dominates the log-likelihood gain and
#      collapses K to 1, reproducing a plain MVN that cannot represent the
#      visible skew in axes like T0, GT_v, P_n, ahead1.
#   2. Diagnostic hexbins (see NROY_joint_multimodality_check.py) showed
#      each skewed-axis pair as a single connected high-density region, not
#      disjoint modes. Marginal bimodality is a projection artefact of
#      corner-concentrated unimodal density in a ~60D box -- exactly the
#      case the Gaussian copula handles well.
#   3. The copula decouples marginals from dependency: KDE captures skew in
#      each p_i exactly, and R captures the joint correlations. No K-sweep
#      is required, so BIC misbehaviour is sidestepped entirely.
copula_prior_cache = None
if USE_COPULA_PRIOR:
    print("\n" + "=" * 60)
    if USE_LOGSPLINE_MARGINALS:
        print("STEP 1b -- Fitting Gaussian copula prior (logspline marginals)")
    else:
        print("STEP 1b -- Fitting Gaussian copula prior (Gaussian KDE marginals)")
    print("=" * 60)

    # Zero-variance columns (degenerate NROY dims) -> fall back to uniform
    axis_std = nroy_subset.std(axis=0, ddof=1)
    zero_var = axis_std < 1e-12
    if zero_var.any():
        bad = [subset_vars[i] for i in np.where(zero_var)[0]]
        print(f"  WARNING: zero-variance NROY dims {bad}; "
              f"falling back to uniform prior")
        copula_prior_cache = None
    elif USE_LOGSPLINE_MARGINALS:
        print(f"  Fitting on N={n_nroy} points, d={ndim}")
        print(f"  Logspline: n_interior={LOGSPLINE_N_INTERIOR}, "
              f"lambda={LOGSPLINE_LAMBDA}, grid={LOGSPLINE_N_GRID}")
        print(f"  R shrinkage toward I: {CORR_SHRINK}")

        import time as _time_
        _t0 = _time_.time()
        copula_prior_cache = fit_logspline_copula(
            nroy_subset,
            prior_lower=prior_lower,
            prior_upper=prior_upper,
            n_interior=LOGSPLINE_N_INTERIOR,
            smooth_lambda=LOGSPLINE_LAMBDA,
            n_grid=LOGSPLINE_N_GRID,
            corr_shrink=CORR_SHRINK,
            f_eps=COPULA_F_EPS,
            axis_names=subset_vars,
        )
        print(f"  Fit time: {_time_.time() - _t0:.1f} s  "
              f"(non-converged axes: {copula_prior_cache['n_bad_fit']})")
        print(f"  R eigenvalues: min={copula_prior_cache['R_eig_min']:.3e}, "
              f"max={copula_prior_cache['R_eig_max']:.3e}, "
              f"cond={copula_prior_cache['R_cond']:.2e}")

        # Sanity check: log-density finite at a random NROY point
        _test_idx = np.random.default_rng(RANDOM_SEED).integers(0, n_nroy)
        _test_theta = torch.tensor(nroy_subset[_test_idx], dtype=torch.float32)
        with torch.no_grad():
            _lp = _logspline_copula_log_prob(_test_theta, copula_prior_cache)
        print(f"  log p(theta) at NROY[{_test_idx}] = {_lp.item():.4f} "
              f"(should be finite)")
        assert torch.isfinite(_lp), "Logspline-copula log-density non-finite"

        # Sanity check 2: gradient w.r.t. theta is finite (NUTS needs it)
        _grad_theta = torch.tensor(nroy_subset[_test_idx],
                                   dtype=torch.float32, requires_grad=True)
        _lp_g = _logspline_copula_log_prob(_grad_theta, copula_prior_cache)
        _lp_g.backward()
        assert torch.isfinite(_grad_theta.grad).all(), (
            "Logspline-copula gradient non-finite; check grid density / "
            "bounds / F_eps"
        )
        print(f"  gradient check passed: "
              f"max|d log p / d theta| = "
              f"{_grad_theta.grad.abs().max().item():.3e}")

        # Save for reproducibility / diagnostics
        copula_dump = {
            "marginal_type":  "logspline",
            "x_grid":         copula_prior_cache["x_grid"].cpu().numpy(),
            "f_grid":         copula_prior_cache["f_grid"].cpu().numpy(),
            "cdf_grid":       copula_prior_cache["cdf_grid"].cpu().numpy(),
            "L_R":            copula_prior_cache["L_R"].cpu().numpy(),
            "half_logdet_R":  float(copula_prior_cache["half_logdet_R"].item()),
            "F_eps":          float(copula_prior_cache["F_eps"].item()),
            "N":              copula_prior_cache["N"],
            "n_grid":         copula_prior_cache["n_grid"],
            "n_interior":     copula_prior_cache["n_interior"],
            "smooth_lambda":  copula_prior_cache["smooth_lambda"],
            "R_eig_min":      copula_prior_cache["R_eig_min"],
            "R_eig_max":      copula_prior_cache["R_eig_max"],
            "R_cond":         copula_prior_cache["R_cond"],
            "corr_shrink":    CORR_SHRINK,
            "subset_vars":    subset_vars,
            "prior_lower":    np.asarray(prior_lower, dtype=np.float64),
            "prior_upper":    np.asarray(prior_upper, dtype=np.float64),
        }
        if WRITE_SHARED_ARTIFACTS:
            joblib.dump(copula_dump, os.path.join(out_dir, "copula_prior.joblib"))
            print(f"  Saved copula to {out_dir}/copula_prior.joblib")
        else:
            print("  Single-chain mode: skipping shared copula_prior.joblib write")
    else:
        print(f"  Fitting on N={n_nroy} points, d={ndim}")
        print(f"  KDE subsample per axis: {KDE_SUBSAMPLE}")
        print(f"  Bandwidth rule: {KDE_BANDWIDTH}")
        print(f"  R shrinkage toward I: {CORR_SHRINK}")

        copula_prior_cache = fit_gaussian_copula(
            nroy_subset,
            n_kde=KDE_SUBSAMPLE,
            bandwidth=KDE_BANDWIDTH,
            corr_shrink=CORR_SHRINK,
            f_eps=COPULA_F_EPS,
            random_state=RANDOM_SEED,
        )
        print(f"  R eigenvalues: min={copula_prior_cache['R_eig_min']:.3e}, "
              f"max={copula_prior_cache['R_eig_max']:.3e}, "
              f"cond={copula_prior_cache['R_cond']:.2e}")
        print(f"  Median bandwidth (Silverman): "
              f"{np.median(copula_prior_cache['h_np']):.4g}")

        # Sanity check: log-density finite at a random NROY point
        _test_idx = np.random.default_rng(RANDOM_SEED).integers(0, n_nroy)
        _test_theta = torch.tensor(nroy_subset[_test_idx], dtype=torch.float32)
        with torch.no_grad():
            _lp = _copula_log_prob(_test_theta, copula_prior_cache)
        print(f"  log p(theta) at NROY[{_test_idx}] = {_lp.item():.4f} "
              f"(should be finite)")
        assert torch.isfinite(_lp), "Copula log-density non-finite on NROY point"

        # Save for reproducibility / diagnostics
        copula_dump = {
            "marginal_type":  "kde",
            "kde_data":       copula_prior_cache["kde_data"].cpu().numpy(),
            "kde_h":          copula_prior_cache["kde_h"].cpu().numpy(),
            "L_R":            copula_prior_cache["L_R"].cpu().numpy(),
            "half_logdet_R":  float(copula_prior_cache["half_logdet_R"].item()),
            "F_eps":          float(copula_prior_cache["F_eps"].item()),
            "M":              copula_prior_cache["M"],
            "N":              copula_prior_cache["N"],
            "R_eig_min":      copula_prior_cache["R_eig_min"],
            "R_eig_max":      copula_prior_cache["R_eig_max"],
            "bandwidth_rule": KDE_BANDWIDTH,
            "corr_shrink":    CORR_SHRINK,
            "subset_vars":    subset_vars,
        }
        if WRITE_SHARED_ARTIFACTS:
            joblib.dump(copula_dump, os.path.join(out_dir, "copula_prior.joblib"))
            print(f"  Saved copula to {out_dir}/copula_prior.joblib")
        else:
            print("  Single-chain mode: skipping shared copula_prior.joblib write")
else:
    print("\n  USE_COPULA_PRIOR = False -> uniform box prior on NROY bounds")

# Prior bounds as tensors
prior_lo = torch.tensor(prior_lower, dtype=torch.float32)
prior_hi = torch.tensor(prior_upper, dtype=torch.float32)

# Save a diagnostic plot of the fitted prior marginals before MCMC starts.
# Array workers skip this shared output to avoid concurrent writes; the
# aggregate job writes it once.
if WRITE_SHARED_ARTIFACTS and copula_prior_cache is not None:
    _ = plot_prior_marginals_vs_nroy(
        copula_prior_cache,
        nroy_subset=nroy_subset,
        subset_vars=subset_vars,
        prior_lower=prior_lower,
        prior_upper=prior_upper,
        out_dir=out_dir,
    )
elif not WRITE_SHARED_ARTIFACTS:
    print("  Single-chain mode: skipping shared prior marginal plot")

# ============================================================
# 2. LOAD GP EMULATORS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 -- Loading GP emulators")
print("=" * 60)

emulators = {}
for name in output_names:
    path = os.path.join(
        EMULATOR_DIR, name,
        f"GaussianProcessMatern32_{name}_best.joblib",
    )
    emulators[name] = joblib.load(path)
print(f"  Loaded {len(emulators)} emulators from {EMULATOR_DIR}/")

# Pre-compute GP internals (Cholesky, alpha) for fast MCMC evaluation
print("  Pre-warming GPyTorch prediction caches...")
gp_caches = extract_fast_caches(emulators, output_names)
first_gp = list(gp_caches.values())[0]["gp"]
print(f"  Cached {len(gp_caches)} GPs "
      f"(n_train={first_gp.train_inputs[0].shape[-2]})")

# ---- Build batched / manual-Matern-3/2 cache and verify vs GPyTorch ----
print("  Building batched manual-forward cache...")
batched_cache = build_batched_fast_caches(emulators, output_names, gp_caches)
print(f"  Batched cache: L={tuple(batched_cache['L'].shape)}, "
      f"alpha={tuple(batched_cache['alpha'].shape)}")


_n_train_here = batched_cache["L"].shape[-1]

# Clear any cached prediction strategy on each GP so it rebuilds under the
# exact-Cholesky context below (otherwise the cache from the pre-warming step
# sticks around and keeps using CG).
for _name in output_names:
    _g = gp_caches[_name]["gp"]
    if hasattr(_g, "prediction_strategy"):
        _g.prediction_strategy = None

# ============================================================
# 4. PYRO NUTS MCMC  (custom potential_fn — no model tracing)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 -- Pyro NUTS MCMC")
print("=" * 60)

# Observation tensors
obs_means_t = torch.tensor(
    [observation[n][0] for n in output_names],
    dtype=torch.float32,
)
obs_vars_t = torch.tensor(
    [observation[n][1] for n in output_names],
    dtype=torch.float32,
)
obs_stds_t = obs_vars_t.sqrt()

# Build the fast potential energy function — batched manual Matern-3/2 path.
# The old gpytorch-based make_fast_potential_fn() is still used above for the
# verification block; NUTS runs against the batched version below.
potential_fn = make_fast_potential_fn_batched(
    prior_lo, prior_hi, obs_means_t, obs_vars_t, batched_cache,
    copula=copula_prior_cache,
)

# --- Convert initial point to unconstrained space via logit ---
eps = 1e-6 * (prior_hi - prior_lo).clamp(min=1e-20)
# top_chain_idx = np.argsort(densities)[-(N_CHAINS):][::-1]
# init_theta = torch.tensor(
#     nroy_subset[top_chain_idx], dtype=torch.float32,
# )

rng = np.random.default_rng(RANDOM_SEED)
start_idx = rng.choice(n_nroy, size=N_CHAINS, replace=False)
init_theta = torch.tensor(nroy_subset[start_idx], dtype=torch.float32)

init_theta = torch.clamp(init_theta, prior_lo + eps, prior_hi - eps)

# logit: z = log( (theta - lo) / (hi - theta) )
theta_01 = (init_theta - prior_lo) / (prior_hi - prior_lo)
init_z_all = torch.log(theta_01 / (1.0 - theta_01))    # (N_CHAINS, ndim)

# Verify initial potential is finite.  In array mode, check this worker's
# own initial point rather than chain 0's.
initial_check_chain = CHAIN_ID if CHAIN_WORKER else 0
with torch.no_grad():
    z_test = init_z_all[initial_check_chain]
    pe0 = potential_fn({"theta": z_test})
    print(f"  Initial potential energy: {pe0.item():.4f}")
    assert torch.isfinite(pe0), f"Initial PE is {pe0.item()}, check NROY start"

# Verify gradient is finite at the initial point
z_grad = z_test.clone().requires_grad_(True)
grad_t0 = time.perf_counter()
pe_grad = potential_fn({"theta": z_grad})
grad_check = torch.autograd.grad(pe_grad, z_grad)[0]
grad_seconds = time.perf_counter() - grad_t0
print(f"  Initial gradient: finite={grad_check.isfinite().all().item()}, "
      f"norm={grad_check.norm().item():.4f}, "
      f"potential+grad time={grad_seconds:.3f}s")
assert grad_check.isfinite().all(), "Gradient has NaN/Inf at initial point"

# ---------- Multi-chain NUTS (parallel driver / sequential / array-task / aggregate) ----------
# Four execution modes (see CLI block at top):
#   1. Default            : parent process launches one subprocess per chain,
#                           then runs --aggregate-only after all workers finish.
#   2. --sequential       : run all N_CHAINS sequentially in this process.
#   3. --chain-id c       : run only chain c, save per-chain output, exit.
#                           Used by the default driver and HPC array jobs.
#   4. --aggregate-only   : skip chains entirely; load chain_z_{c}.npy from
#                           out_dir for c in 0..N_CHAINS-1.
# Posterior samples, split-R-hat and ESS are identical regardless of mode;
# parallel driver / array-job mode collapses wall time from N_CHAINS x to 1 x per chain.
print(f"  {N_CHAINS} chain(s) x ({N_WARMUP} warmup + {N_SAMPLES} samples)")
print(f"  ndim = {ndim},  max_tree_depth = {MAX_TREE_DEPTH}")
print(f"  target_accept_prob = {TARGET_ACCEPT}")
if AGGREGATE_ONLY:
    print(f"  Mode: AGGREGATE-ONLY (loading per-chain files from {out_dir})")
elif CHAIN_ID >= 0:
    print(f"  Mode: SINGLE-CHAIN (chain-id={CHAIN_ID})")
else:
    print("  Mode: SEQUENTIAL (all chains in this process)")

chain_samples_z  = []   # list of (N_SAMPLES, ndim) tensors
chain_diag_list  = []   # per-chain diagnostics dicts
chain_fell_back  = []   # which chains used HMC fallback
mcmc_c           = None

if AGGREGATE_ONLY:
    for c in range(N_CHAINS):
        z_path = os.path.join(out_dir, f"chain_z_{c}.npy")
        if not os.path.exists(z_path):
            raise FileNotFoundError(
                f"Aggregate mode: missing {z_path}. "
                f"Run all N_CHAINS={N_CHAINS} array tasks first."
            )
        zc = np.load(z_path)
        chain_samples_z.append(torch.tensor(zc, dtype=torch.float32))
        diag_path = os.path.join(out_dir, f"chain_diag_{c}.joblib")
        if os.path.exists(diag_path):
            payload = joblib.load(diag_path)
            chain_diag_list.append(payload.get("diag", {}))
            chain_fell_back.append(bool(payload.get("fell_back", False)))
        else:
            chain_diag_list.append({})
            chain_fell_back.append(False)
    print(f"  Loaded {N_CHAINS} chain samples from {out_dir}/")
else:
    chains_to_run = [CHAIN_ID] if CHAIN_ID >= 0 else list(range(N_CHAINS))
    if CHAIN_ID >= 0 and not (0 <= CHAIN_ID < N_CHAINS):
        raise ValueError(f"--chain-id={CHAIN_ID} outside [0, {N_CHAINS - 1}]")

    for c in chains_to_run:
        print(f"\n  --- Chain {c + 1}/{N_CHAINS} ---")
        pyro.set_rng_seed(RANDOM_SEED + c)
        init_z_c = init_z_all[c].clone()

        nuts_c = NUTS(
            potential_fn=potential_fn,
            step_size=1e-3,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            max_tree_depth=MAX_TREE_DEPTH,
            target_accept_prob=TARGET_ACCEPT,
            jit_compile=False,
        )
        mcmc_c = MCMC(
            nuts_c,
            num_samples=N_SAMPLES,
            warmup_steps=N_WARMUP,
            num_chains=1,
            initial_params={"theta": init_z_c},
        )

        fell_back = False
        try:
            mcmc_c.run()
        except Exception as nuts_err:
            print(f"  chain {c}: NUTS failed ({nuts_err})")
            print(f"  chain {c}: falling back to HMC")
            hmc_c = HMC(
                potential_fn=potential_fn,
                step_size=1e-3,
                adapt_step_size=True,
                num_steps=20,
                jit_compile=False,
            )
            mcmc_c = MCMC(
                hmc_c,
                num_samples=N_SAMPLES,
                warmup_steps=N_WARMUP,
                num_chains=1,
                initial_params={"theta": init_z_c},
            )
            mcmc_c.run()
            fell_back = True

        z_c = mcmc_c.get_samples()["theta"]                  # (N_SAMPLES, ndim)
        chain_samples_z.append(z_c)
        try:
            diag_c = mcmc_c.diagnostics()
        except Exception as diag_err:
            print(f"  chain {c}: diagnostics() failed: {diag_err}")
            diag_c = {}
        chain_diag_list.append(diag_c)
        chain_fell_back.append(fell_back)

        # Single-chain (HPC array task) mode: persist this chain's output
        # so the aggregator can stack it later.
        if CHAIN_ID >= 0:
            np.save(os.path.join(out_dir, f"chain_z_{c}.npy"),
                    z_c.detach().cpu().numpy())
            joblib.dump(
                {"diag": diag_c, "fell_back": fell_back, "chain_id": c},
                os.path.join(out_dir, f"chain_diag_{c}.joblib"),
            )
            print(f"  chain {c}: saved chain_z_{c}.npy "
                  f"+ chain_diag_{c}.joblib to {out_dir}/")

    # Array-task mode: stop here — aggregation is done by a separate job.
    if CHAIN_ID >= 0:
        print(f"\n  Single-chain run complete (chain {CHAIN_ID}). "
              f"Run with --aggregate-only after all {N_CHAINS} chains finish.")
        sys.exit(0)

# Stack into (C, N, d); keep last MCMC object for any residual references.
z_chains = torch.stack(chain_samples_z, dim=0)
mcmc = mcmc_c

# ============================================================
# 5. DIAGNOSTICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5 -- MCMC diagnostics")
print("=" * 60)

# Samples are in unconstrained space — transform back to [lo, hi]
posterior_z = z_chains.reshape(-1, z_chains.shape[-1])          # (C*N, ndim)
posterior = prior_lo + torch.sigmoid(posterior_z) * (prior_hi - prior_lo)
posterior_np = posterior.detach().cpu().numpy()

posterior_chains = prior_lo + torch.sigmoid(z_chains) * (prior_hi - prior_lo)  # (C, N, ndim)

# One-dimensional marginal summaries.  These are useful diagnostics, but they
# should not be treated as a jointly valid posterior parameter vector.
post_mean   = posterior_np.mean(axis=0)
post_std    = posterior_np.std(axis=0)
post_median = np.median(posterior_np, axis=0)
post_q05    = np.percentile(posterior_np, 5, axis=0)
post_q95    = np.percentile(posterior_np, 95, axis=0)

# ============================================================
# 6. POSTERIOR PREDICTIVE CHECK
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 -- Posterior predictive check")
print("=" * 60)

# --- Posterior predictive distribution (N_PRED_CHECK samples) ---
print(f"\n--- Posterior predictive distribution ({N_PRED_CHECK} samples) ---")
check_idx = np.random.choice(
    len(posterior_np), min(N_PRED_CHECK, len(posterior_np)), replace=False
)
n_check = len(check_idx)

# Single batched forward through the manual-Matern-3/2 cache: replaces
# n_check * n_outputs individual gpytorch calls with one tensor op.
with torch.no_grad():
    theta_batch = torch.tensor(posterior_np[check_idx], dtype=torch.float32)
    pred_matrix = batched_predict_mean(theta_batch, batched_cache).cpu().numpy()

# Replace the raw mL predictions at the atrial-diff indices with the derived
# active-emptying fraction so the predictive table, normalised box plot and
# saved pred_check_matrix.npy compare directly against the ratio targets
# stored in obs_means_t / obs_vars_t.
rest_la_denom = pred_matrix[:, REST_LA_MAX_IDX] - pred_matrix[:, REST_LA_MIN_IDX]
rest_ra_denom = pred_matrix[:, REST_RA_MAX_IDX] - pred_matrix[:, REST_RA_MIN_IDX]
rest_la_denom = np.where(np.abs(rest_la_denom) < 1e-8, np.where(rest_la_denom >= 0, 1e-8, -1e-8), rest_la_denom)
rest_ra_denom = np.where(np.abs(rest_ra_denom) < 1e-8, np.where(rest_ra_denom >= 0, 1e-8, -1e-8), rest_ra_denom)
pred_matrix[:, REST_LA_PRE_IDX] = (pred_matrix[:, REST_LA_PRE_IDX] - pred_matrix[:, REST_LA_MIN_IDX]) / rest_la_denom
pred_matrix[:, REST_RA_PRE_IDX] = (pred_matrix[:, REST_RA_PRE_IDX] - pred_matrix[:, REST_RA_MIN_IDX]) / rest_ra_denom

exercise_la_denom = pred_matrix[:, EXERCISE_LA_MAX_IDX] - pred_matrix[:, EXERCISE_LA_MIN_IDX]
exercise_ra_denom = pred_matrix[:, EXERCISE_RA_MAX_IDX] - pred_matrix[:, EXERCISE_RA_MIN_IDX]
exercise_la_denom = np.where(np.abs(exercise_la_denom) < 1e-8, np.where(exercise_la_denom >= 0, 1e-8, -1e-8), exercise_la_denom)
exercise_ra_denom = np.where(np.abs(exercise_ra_denom) < 1e-8, np.where(exercise_ra_denom >= 0, 1e-8, -1e-8), exercise_ra_denom)
pred_matrix[:, EXERCISE_LA_PRE_IDX] = (pred_matrix[:, EXERCISE_LA_PRE_IDX] - pred_matrix[:, EXERCISE_LA_MIN_IDX]) / exercise_la_denom
pred_matrix[:, EXERCISE_RA_PRE_IDX] = (pred_matrix[:, EXERCISE_RA_PRE_IDX] - pred_matrix[:, EXERCISE_RA_MIN_IDX]) / exercise_ra_denom

print(f"{'Output':<45} {'Mean Pred':>10} {'Std Pred':>10} "
      f"{'Target':>10} {'% <=1s':>8}")
print("-" * 90)
for i, name in enumerate(output_names):
    preds  = pred_matrix[:, i]
    tgt    = obs_means_t[i].item()
    std    = obs_stds_t[i].item()
    within = (np.abs(preds - tgt) <= std).mean() * 100
    print(f"{name:<45} {preds.mean():10.3f} {preds.std():10.3f} "
          f"{tgt:10.3f} {within:7.1f}%")

# ============================================================
# 7. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7 -- Saving results")
print("=" * 60)

np.save(os.path.join(out_dir, "posterior_samples.npy"), posterior_np)
np.save(os.path.join(out_dir, "posterior_mean.npy"), post_mean)
np.save(os.path.join(out_dir, "posterior_std.npy"), post_std)
np.save(os.path.join(out_dir, "posterior_median.npy"), post_median)
np.save(os.path.join(out_dir, "subset_vars.npy"),
        np.array(subset_vars, dtype=object))
# np.save(os.path.join(out_dir, "knn_densities.npy"), densities)
# np.save(os.path.join(out_dir, "knn_best_start.npy"), best_start)
np.save(os.path.join(out_dir, "pred_check_matrix.npy"), pred_matrix)

# ---- Additional saves for robust MCMC analysis ----

# 1. Per-chain samples in constrained space (for R-hat / split-R-hat).
posterior_chains_np = posterior_chains.detach().cpu().numpy()          # (C, N, d)
np.save(os.path.join(out_dir, "posterior_chains.npy"), posterior_chains_np)
np.save(os.path.join(out_dir, "posterior_chains_z.npy"),
        z_chains.detach().cpu().numpy())

# 2. Unconstrained samples (for warm-starting or geometry diagnostics).
np.save(os.path.join(out_dir, "posterior_z.npy"),
        posterior_z.detach().cpu().numpy())

# 3. Pyro MCMC diagnostics: per-chain Pyro output + proper split-R-hat / ESS.
#    Split-R-hat needs >=2 chains, so we compute it from z_chains directly
#    rather than relying on the single-chain mcmc_c.diagnostics() which
#    reports r_hat=NaN per chain.
def _jsonify(v):
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    return v

diag_out = {
    "per_chain": [_jsonify(d) for d in chain_diag_list],
    "chain_fell_back_to_hmc": chain_fell_back,
}
try:
    from pyro.ops.stats import split_gelman_rubin, effective_sample_size
    # z_chains: (C, N, d) with chain_dim=0, sample_dim=1
    r_hat_z = split_gelman_rubin(z_chains)        # (d,)
    n_eff_z = effective_sample_size(z_chains)     # (d,)
    diag_out["split_r_hat_z"] = r_hat_z.tolist()
    diag_out["n_eff_z"]       = n_eff_z.tolist()
    diag_out["r_hat_max"]     = float(r_hat_z.max().item())
    diag_out["n_eff_min"]     = float(n_eff_z.min().item())
    print(f"  max split-R-hat (unconstrained): {r_hat_z.max().item():.4f}")
    print(f"  min ESS (unconstrained):         {n_eff_z.min().item():.1f}")
except Exception as e:
    print(f"  WARN: could not compute split-R-hat / ESS: {e}")

try:
    with open(os.path.join(out_dir, "mcmc_diagnostics.json"), "w") as f:
        json.dump(diag_out, f, indent=2, default=str)
except Exception as e:
    print(f"  WARN: could not save mcmc_diagnostics.json: {e}")

# 4. Log-posterior trace (useful for convergence check).
log_post_trace = np.zeros(posterior_z.shape[0])
for k in range(posterior_z.shape[0]):
    with torch.no_grad():
        log_post_trace[k] = -potential_fn({"theta": posterior_z[k]}).item()
np.save(os.path.join(out_dir, "log_posterior_trace.npy"), log_post_trace)

# Joint representative point: use an actual posterior draw, not coordinate-wise
# marginal means/medians.  This is the highest log-posterior draw among the
# sampled states, so it preserves the posterior dependence structure.
best_joint_idx = int(np.argmax(log_post_trace))
post_joint_sample = posterior_np[best_joint_idx]
post_joint_sample_z = posterior_z[best_joint_idx].detach().cpu().numpy()
np.save(os.path.join(out_dir, "posterior_best_joint_sample.npy"), post_joint_sample)
np.save(os.path.join(out_dir, "posterior_best_joint_sample_z.npy"), post_joint_sample_z)
np.save(os.path.join(out_dir, "posterior_best_joint_sample_idx.npy"),
        np.array(best_joint_idx, dtype=np.int64))
print(f"  best sampled joint log-posterior index: {best_joint_idx}")

# 5. Run configuration (for reproducibility).
config = {
    "random_seed":     RANDOM_SEED,
    "n_warmup":        N_WARMUP,
    "n_samples":       N_SAMPLES,
    "n_chains":        N_CHAINS,
    "target_accept":   TARGET_ACCEPT,
    "max_tree_depth":  MAX_TREE_DEPTH,
    "hm_artifacts_dir": HM_ARTIFACTS_DIR,
    "emulator_dir":    EMULATOR_DIR,
    "run_dir":         out_dir,
    "threads_per_chain": THREADS_PER_CHAIN_EFFECTIVE,
    "max_threads_per_chain": _args.max_threads_per_chain,
    "available_cpu_count": _available_cpu_count(),
    "percent":         PERCENT,
    # "knn_k":           KNN_K,
    "n_pred_check":    N_PRED_CHECK,
    "use_copula_prior":     bool(USE_COPULA_PRIOR and copula_prior_cache is not None),
    "marginal_type":        (copula_prior_cache.get("marginal_type")
                             if copula_prior_cache is not None else None),
    "kde_subsample":        (KDE_SUBSAMPLE
                             if USE_COPULA_PRIOR and not USE_LOGSPLINE_MARGINALS
                             else None),
    "kde_bandwidth":        (KDE_BANDWIDTH
                             if USE_COPULA_PRIOR and not USE_LOGSPLINE_MARGINALS
                             else None),
    "logspline_n_interior": (LOGSPLINE_N_INTERIOR
                             if USE_COPULA_PRIOR and USE_LOGSPLINE_MARGINALS
                             else None),
    "logspline_lambda":     (LOGSPLINE_LAMBDA
                             if USE_COPULA_PRIOR and USE_LOGSPLINE_MARGINALS
                             else None),
    "logspline_n_grid":     (LOGSPLINE_N_GRID
                             if USE_COPULA_PRIOR and USE_LOGSPLINE_MARGINALS
                             else None),
    "corr_shrink":          CORR_SHRINK if USE_COPULA_PRIOR else None,
    "copula_f_eps":         COPULA_F_EPS if USE_COPULA_PRIOR else None,
    "copula_R_cond":        (copula_prior_cache["R_cond"]
                             if copula_prior_cache is not None else None),
}
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# 6. Observation targets + NROY bounds (make analysis self-contained).
np.save(os.path.join(out_dir, "obs_means.npy"), obs_means_t.numpy())
np.save(os.path.join(out_dir, "obs_vars.npy"),  obs_vars_t.numpy())
np.save(os.path.join(out_dir, "output_names.npy"),
        np.array(output_names, dtype=object))
np.save(os.path.join(out_dir, "prior_lower.npy"), prior_lower)
np.save(os.path.join(out_dir, "prior_upper.npy"), prior_upper)

# Full 224-dim parameter vector at best sampled joint posterior point.
# Non-calibration params stay at their nominal (fixed) values
nominal = np.array(
    [0.5 * (nroy_params_dict[n][0] + nroy_params_dict[n][1])
     for n in all_param_names], dtype=np.float32,
)
full_joint_sample = nominal.copy()
full_joint_sample[param_idx] = post_joint_sample.astype(np.float32)
np.save(os.path.join(out_dir, "full_param_best_joint_sample.npy"),
        full_joint_sample)

# Also save as {name: value} dict for easy simulator use
posterior_param_dict = {
    name: float(full_joint_sample[i])
    for i, name in enumerate(all_param_names)
}
np.save(os.path.join(out_dir, "posterior_param_dict.npy"),
        posterior_param_dict)

print(f"  Saved to {out_dir}/")

# ============================================================
# 8. PLOTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8 -- Plots")
print("=" * 60)

# 8a. Trace plots + marginal posteriors (first 8 params)
n_plot = ndim
fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3 * n_plot))
if n_plot == 1:
    axes = axes[np.newaxis, :]

for j in range(n_plot):
    # Trace
    for c in range(posterior_chains.shape[0]):
        axes[j, 0].plot(
            posterior_chains[c, :, j].cpu().numpy(),
            alpha=0.5, linewidth=0.5,
        )


    # Marginal posterior
    axes[j, 1].hist(
        posterior_np[:, j], bins=50, density=True, alpha=0.7, color="steelblue"
    )
    axes[j, 1].axvline(
        post_joint_sample[j], color="red", ls="--", lw=1.2,
        label="best joint sample"
    )
    axes[j, 1].axvline(
        prior_lower[j], color="black", ls=":", alpha=0.4, label="NROY bounds"
    )
    axes[j, 1].axvline(prior_upper[j], color="black", ls=":", alpha=0.4)
    axes[j, 1].set_title(f"Marginal: {subset_vars[j]}", fontsize=8)
    axes[j, 1].legend(fontsize=6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "trace_and_marginals.png"), dpi=200)
plt.close()

# 8b. Normalised posterior predictive box plot
fig, ax = plt.subplots(figsize=(16, 5))
plot_matrix = pred_matrix.copy()
plot_tgt_means_np = obs_means_t.numpy().copy()
plot_tgt_stds_np = obs_stds_t.numpy().copy()

# The likelihood for the atrial observables is defined on the active-emptying
# fraction ratio, but for visual interpretation it is often more intuitive to
# display the corresponding pre-atrial-contraction volume:
#   V_pre = V_min + r * (V_max - V_min)
plot_matrix[:, REST_LA_PRE_IDX] = (
    pred_matrix[:, REST_LA_MIN_IDX]
    + pred_matrix[:, REST_LA_PRE_IDX] * (pred_matrix[:, REST_LA_MAX_IDX] - pred_matrix[:, REST_LA_MIN_IDX])
)
plot_matrix[:, REST_RA_PRE_IDX] = (
    pred_matrix[:, REST_RA_MIN_IDX]
    + pred_matrix[:, REST_RA_PRE_IDX] * (pred_matrix[:, REST_RA_MAX_IDX] - pred_matrix[:, REST_RA_MIN_IDX])
)
plot_tgt_means_np[REST_LA_PRE_IDX] = REST_LA_PRE_DISPLAY_MEAN
plot_tgt_means_np[REST_RA_PRE_IDX] = REST_RA_PRE_DISPLAY_MEAN
plot_tgt_stds_np[REST_LA_PRE_IDX] = REST_LA_PRE_DISPLAY_STD
plot_tgt_stds_np[REST_RA_PRE_IDX] = REST_RA_PRE_DISPLAY_STD

plot_matrix[:, EXERCISE_LA_PRE_IDX] = (
    pred_matrix[:, EXERCISE_LA_MIN_IDX]
    + pred_matrix[:, EXERCISE_LA_PRE_IDX] * (pred_matrix[:, EXERCISE_LA_MAX_IDX] - pred_matrix[:, EXERCISE_LA_MIN_IDX])
)
plot_matrix[:, EXERCISE_RA_PRE_IDX] = (
    pred_matrix[:, EXERCISE_RA_MIN_IDX]
    + pred_matrix[:, EXERCISE_RA_PRE_IDX] * (pred_matrix[:, EXERCISE_RA_MAX_IDX] - pred_matrix[:, EXERCISE_RA_MIN_IDX])
)
plot_tgt_means_np[EXERCISE_LA_PRE_IDX] = EXERCISE_LA_PRE_DISPLAY_MEAN
plot_tgt_means_np[EXERCISE_RA_PRE_IDX] = EXERCISE_RA_PRE_DISPLAY_MEAN
plot_tgt_stds_np[EXERCISE_LA_PRE_IDX] = EXERCISE_LA_PRE_DISPLAY_STD
plot_tgt_stds_np[EXERCISE_RA_PRE_IDX] = EXERCISE_RA_PRE_DISPLAY_STD

normalised = (plot_matrix - plot_tgt_means_np) / plot_tgt_stds_np

short_names = [n.replace("_", "\n") for n in output_names]
short_names[REST_LA_PRE_IDX] = "REST LA\nPre-A\nVolume"
short_names[REST_RA_PRE_IDX] = "REST RA\nPre-A\nVolume"
short_names[EXERCISE_LA_PRE_IDX] = "EXERCISE LA\nPre-A\nVolume"
short_names[EXERCISE_RA_PRE_IDX] = "EXERCISE RA\nPre-A\nVolume"


x_pos = np.arange(len(output_names))

ax.boxplot(
    normalised, positions=x_pos, widths=0.6,
    showfliers=False, patch_artist=True,
    boxprops=dict(facecolor="steelblue", alpha=0.5),
    medianprops=dict(color="red", linewidth=1.5),
)
ax.axhline(0, color="black", ls="-", linewidth=0.5)
ax.axhline(1,  color="green", ls="--", alpha=0.6, label="+/- 1 sigma")
ax.axhline(-1, color="green", ls="--", alpha=0.6)
ax.axhline(3,  color="red",   ls=":",  alpha=0.4, label="+/- 3 sigma")
ax.axhline(-3, color="red",   ls=":",  alpha=0.4)
ax.set_xticks(x_pos)
ax.set_xticklabels(short_names, rotation=90, fontsize=6)
ax.set_ylabel("(predicted - target) / sigma")
ax.set_title("Posterior predictive normalised by population sigma (atria shown as pre-A volume)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "posterior_predictive_normalised.png"), dpi=200
)
plt.close()
