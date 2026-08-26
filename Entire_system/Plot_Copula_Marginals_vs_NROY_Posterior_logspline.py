"""Overlay MCMC posterior marginals on the copula-prior-vs-NROY plot.

This mirrors the logspline plotting style used by `KNN_MCMC_Rest.py` for
`copula_marginals_vs_NROY_logspline.png`, then adds one more curve per panel:
the marginal posterior density for that parameter inferred from the saved MCMC
parameter samples.

Important:
    Use `posterior_samples.npy` here, not `pred_check_matrix.npy`.
    `posterior_samples.npy` contains samples of the calibration parameters
    themselves. `pred_check_matrix.npy` contains posterior-predictive emulator
    outputs, which are distributions over observables rather than parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.integrate import simpson
from scipy.interpolate import BSpline
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (
    SCRIPT_DIR
    / "MCMC_HPC"
    / "MCMC_Rest_20_05_05_1500_logspline_copula_prior"
)
DEFAULT_OUT_NAME = "copula_marginals_vs_NROY_logspline_with_posterior.png"

N_BINS = 40
N_COLS = 7
DEFAULT_DPI = 600
X_PAD_FRAC = 0.02
NROY_DISTRIBUTION_COLOR = "#9DB8D8"
POSTERIOR_DISTRIBUTION_COLOR = "#D68484"


def _format_max_sig_figs(value: float, sig_figs: int = 4, max_decimals: int = 3) -> str:
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

    if (
        "." not in label
        and not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-12)
        and abs_rounded < 10 ** (sig_figs - 1)
    ):
        label = f"{rounded:.1f}"

    return label


def _annotate_actual_parameter_value(ax: plt.Axes, value: float) -> None:
    """Add the actual parameter value beside its vertical marker."""
    value = float(value)
    if not np.isfinite(value):
        return

    x_min, x_max = ax.get_xlim()
    if value < x_min or value > x_max:
        return

    place_right = (x_max - value) >= (value - x_min)
    ax.annotate(
        f"{value:.2f}",
        xy=(value, 0.05),
        xycoords=("data", "axes fraction"),
        xytext=(3 if place_right else -3, 0),
        textcoords="offset points",
        ha="left" if place_right else "right",
        va="bottom",
        fontsize=ACTUAL_PARAM_LABEL_FONTSIZE,
        color=ACTUAL_PARAM_LINE_COLOR,
        clip_on=True,
        zorder=5,
    )


def _dense_design(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """Return the dense B-spline design matrix."""
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(
        BSpline.design_matrix(x, knots, degree, extrapolate=False).todense()
    )


def fit_logspline_density_on_grid(
    data: np.ndarray,
    x_grid: np.ndarray,
    n_interior: int,
    smooth_lambda: float,
) -> tuple[np.ndarray, bool]:
    """Fit a bounded-support P-spline log-density and evaluate it on `x_grid`."""
    x_grid = np.asarray(x_grid, dtype=np.float64)
    lo = float(x_grid[0])
    hi = float(x_grid[-1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x_grid), False

    degree = 3
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    knots = np.concatenate(
        [np.full(degree + 1, lo), interior, np.full(degree + 1, hi)]
    )
    n_basis = len(knots) - degree - 1

    data = np.clip(np.asarray(data, dtype=np.float64), lo, hi)
    if data.size == 0:
        return np.zeros_like(x_grid), False

    b_grid = _dense_design(x_grid, knots, degree)
    b_data = _dense_design(data, knots, degree)
    d2 = np.diff(np.eye(n_basis), n=2, axis=0)
    penalty = d2.T @ d2
    b_data_sum = b_data.sum(axis=0)
    n_obs = data.size

    def neg_log_post(beta: np.ndarray) -> float:
        eta_grid = b_grid @ beta
        max_eta = eta_grid.max()
        f_unn = np.exp(eta_grid - max_eta)
        z_shifted = simpson(f_unn, x=x_grid)
        log_z = np.log(z_shifted) + max_eta
        log_lik = (b_data @ beta).sum() - n_obs * log_z
        smooth_penalty = smooth_lambda * (beta @ penalty @ beta)
        return -(log_lik - smooth_penalty)

    def grad_neg_log_post(beta: np.ndarray) -> np.ndarray:
        eta_grid = b_grid @ beta
        max_eta = eta_grid.max()
        f_unn = np.exp(eta_grid - max_eta)
        z_shifted = simpson(f_unn, x=x_grid)
        p_grid = f_unn / z_shifted
        int_bp = simpson(b_grid * p_grid[:, None], x=x_grid, axis=0)
        d_ll = b_data_sum - n_obs * int_bp
        d_pen = 2.0 * smooth_lambda * (penalty @ beta)
        return -(d_ll - d_pen)

    result = minimize(
        neg_log_post,
        np.zeros(n_basis),
        jac=grad_neg_log_post,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8},
    )

    beta_hat = result.x
    eta_grid = b_grid @ beta_hat
    max_eta = eta_grid.max()
    f_grid = np.exp(eta_grid - max_eta)
    norm = simpson(f_grid, x=x_grid)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.zeros_like(x_grid), False
    f_grid /= norm
    return f_grid, bool(result.success)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="MCMC output directory containing copula_prior.joblib and posterior_samples.npy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the overlay figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Figure export DPI.",
    )
    return parser.parse_args()


def main() -> None:
    A = np.load("C:/Users/vanes/Downloads/exercise_model/ODE_Exercise/Entire_system/MCMC_HPC/MCMC_Rest_20_05_05_1500_logspline_copula_prior/refined_map_sample.npy")
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_path = (args.output.resolve() if args.output is not None else run_dir / DEFAULT_OUT_NAME)

    config_path = run_dir / "config.json"
    copula_path = run_dir / "copula_prior.joblib"
    posterior_path = run_dir / "posterior_samples.npy"
    subset_path = run_dir / "subset_vars.npy"
    map_path = run_dir / "refined_map_sample.npy"

    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    copula_cache = joblib.load(copula_path)
    posterior = np.load(posterior_path)
    map_sample = np.load(map_path)
    subset_vars = [str(x) for x in np.load(subset_path, allow_pickle=True).tolist()]

    mcmc_root = run_dir.parent
    nroy_points = np.load(mcmc_root / "nroy_points_wave_3.npy")
    nroy_params = np.load(
        mcmc_root / f"NROY_Params_rest_{config['percent']}_{config['date_suffix']}.npy",
        allow_pickle=True,
    ).item()

    all_param_names = list(nroy_params.keys())
    param_idx = [all_param_names.index(name) for name in subset_vars]
    nroy_subset = nroy_points[:, param_idx].astype(np.float64)

    x_grid = np.asarray(copula_cache["x_grid"], dtype=np.float64)
    prior_f_grid = np.asarray(copula_cache["f_grid"], dtype=np.float64)
    prior_lower = np.asarray(copula_cache["prior_lower"], dtype=np.float64)
    prior_upper = np.asarray(copula_cache["prior_upper"], dtype=np.float64)

    if posterior.shape[1] != len(subset_vars):
        raise ValueError(
            f"Posterior dimension mismatch: expected {len(subset_vars)} columns, "
            f"found {posterior.shape[1]}."
        )
    if map_sample.shape != (len(subset_vars),):
        raise ValueError(
            f"MAP sample shape mismatch: expected {(len(subset_vars),)}, "
            f"found {map_sample.shape}."
        )

    n_interior = int(copula_cache.get("n_interior", config.get("logspline_n_interior", 14)))
    smooth_lambda = float(copula_cache.get("smooth_lambda", config.get("logspline_lambda", 5.0)))

    print("Fitting posterior logspline marginals on saved copula support...")
    posterior_f_grid = np.zeros_like(prior_f_grid)
    bad_axes: list[str] = []
    for j, name in enumerate(subset_vars):
        posterior_f_grid[j], ok = fit_logspline_density_on_grid(
            posterior[:, j], x_grid[j], n_interior=n_interior, smooth_lambda=smooth_lambda
        )
        if not ok:
            bad_axes.append(name)
        if (j + 1) % 10 == 0 or j == len(subset_vars) - 1:
            print(f"  fitted {j + 1}/{len(subset_vars)}")

    n_dim = len(subset_vars)
    n_rows = math.ceil(n_dim / N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(3.0 * N_COLS, 2.2 * n_rows))
    fig.set_size_inches(2.35 * N_COLS, 1.8 * n_rows)
    axes = np.atleast_2d(axes)
    formatter = FuncFormatter(lambda x, _pos: _format_max_sig_figs(x, sig_figs=4))

    for j, name in enumerate(subset_vars):
        ax = axes[j // N_COLS, j % N_COLS]
        data_j = nroy_subset[:, j]

        ax.hist(
            data_j,
            bins=N_BINS,
            density=True,
            color=NROY_DISTRIBUTION_COLOR,
            alpha=0.5,
            edgecolor="white",
            linewidth=0.25,
        )

        ax.plot(x_grid[j], prior_f_grid[j], color=NROY_DISTRIBUTION_COLOR, lw=1.1)
        ax.fill_between(
            x_grid[j], 0.0, posterior_f_grid[j], color=POSTERIOR_DISTRIBUTION_COLOR, alpha=0.07
        )
        ax.plot(x_grid[j], posterior_f_grid[j], color=POSTERIOR_DISTRIBUTION_COLOR, lw=1.1)
        ax.axvline(map_sample[j], color=ACTUAL_PARAM_LINE_COLOR, lw=0.9)

        lo = max(float(prior_lower[j]), float(data_j.min()) - 1e-9)
        hi = min(float(prior_upper[j]), float(data_j.max()) + 1e-9)
        width = max(hi - lo, 1e-12)
        x_pad = X_PAD_FRAC * width
        ax.set_xlim(lo - x_pad, hi + x_pad)
        _annotate_actual_parameter_value(ax, map_sample[j])
        ax.set_xticks([lo, hi])
        ax.xaxis.set_major_formatter(formatter)
        ax.set_yticks([])
        ax.set_xlabel(name, fontsize=12, labelpad=2)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(0.6)
            ax.spines[spine].set_color("#4d4d4d")
        ax.tick_params(axis="x", labelsize=12, length=2.5, width=0.6)
        ax.tick_params(axis="y", length=0)

    for k in range(n_dim, n_rows * N_COLS):
        axes[k // N_COLS, k % N_COLS].axis("off")

    fig.tight_layout(rect=[0.025, 0, 1, 0.985])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved overlay figure: {output_path}")
    if bad_axes:
        print(
            "Warning: posterior logspline fit did not report clean convergence for "
            f"{len(bad_axes)} axes: {', '.join(bad_axes)}"
        )


if __name__ == "__main__":
    main()
