import argparse
import json
import math
import os

import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The GP outputs are trained on raw pre-LA/pre-RA contraction volumes. As in
# MCMC_Union.py, the target is the derived active-emptying fraction:
#   r = (V_pre_contraction - V_min) / (V_max - V_min)
ATRIAL_RATIO_TARGET = (0.25, 0.0025)
ATRIAL_RATIO_DISPLAY_MEAN = ATRIAL_RATIO_TARGET[0]
ATRIAL_RATIO_DISPLAY_VAR = ATRIAL_RATIO_TARGET[1]
ATRIAL_RATIO_DISPLAY_STD = math.sqrt(ATRIAL_RATIO_DISPLAY_VAR)

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

ATRIAL_DISPLAY_STATS = {
    "Rest_LA": (ATRIAL_RATIO_DISPLAY_MEAN, ATRIAL_RATIO_DISPLAY_STD, "REST LA\nPre-A\nFraction"),
    "Rest_RA": (ATRIAL_RATIO_DISPLAY_MEAN, ATRIAL_RATIO_DISPLAY_STD, "REST RA\nPre-A\nFraction"),
    "Exercise_LA": (ATRIAL_RATIO_DISPLAY_MEAN, ATRIAL_RATIO_DISPLAY_STD, "EXERCISE LA\nPre-A\nFraction"),
    "Exercise_RA": (ATRIAL_RATIO_DISPLAY_MEAN, ATRIAL_RATIO_DISPLAY_STD, "EXERCISE RA\nPre-A\nFraction"),
}


def _log_det_jac_np(z, prior_lo, prior_hi):
    log_width = np.log(prior_hi - prior_lo)
    log_sig_pos = -np.logaddexp(0.0, -z)
    log_sig_neg = -np.logaddexp(0.0, z)
    return (log_sig_pos + log_sig_neg + log_width).sum(axis=-1)


def _sigmoid_np(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logit_np(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def compute_map(run_dir, top_k=10):
    posterior = np.load(os.path.join(run_dir, "posterior_samples.npy"))
    posterior_z = np.load(os.path.join(run_dir, "posterior_z.npy"))
    log_post_z = np.load(os.path.join(run_dir, "log_posterior_trace.npy"))
    prior_lo = np.load(os.path.join(run_dir, "prior_lower.npy"))
    prior_hi = np.load(os.path.join(run_dir, "prior_upper.npy"))

    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        n_chains = int(cfg["n_chains"])
        n_samples = int(cfg["n_samples"])
    else:
        chains = np.load(os.path.join(run_dir, "posterior_chains.npy"))
        n_chains, n_samples, _ = chains.shape

    log_det = _log_det_jac_np(posterior_z, prior_lo, prior_hi)
    log_post_theta = log_post_z - log_det

    topk_idx = np.argsort(log_post_theta)[::-1][:top_k]
    top_k_list = [
        {
            "rank": r + 1,
            "flat_idx": int(i),
            "chain": int(i // n_samples),
            "draw": int(i % n_samples),
            "log_post_theta": float(log_post_theta[i]),
            "log_post_z": float(log_post_z[i]),
            "sample": posterior[i],
            "sample_z": posterior_z[i],
        }
        for r, i in enumerate(topk_idx)
    ]

    best = top_k_list[0]
    return {
        "sampled_map": best,
        "top_k": top_k_list,
        "n_chains": n_chains,
        "n_samples": n_samples,
        "prior_lower": prior_lo,
        "prior_upper": prior_hi,
        "posterior_z": posterior_z,
    }


# ---------------------------------------------------------------------
# Emulator loading / differentiable forward pass
# ---------------------------------------------------------------------

def _resolve_emulator_dir(run_dir, cfg):
    # emu_cfg = cfg.get("emulator_dir", "Emulator_union_all_wave")
    emu_cfg = cfg.get("emulator_dir", "Emulator_union_all_wave") # change
    if os.path.isabs(emu_cfg) and os.path.isdir(emu_cfg):
        return emu_cfg
    emu_leaf = os.path.basename(os.path.normpath(emu_cfg))
    candidates = [
        emu_cfg,
        os.path.join(SCRIPT_DIR, emu_cfg),
        os.path.join(os.path.dirname(os.path.abspath(run_dir)), emu_cfg),
        os.path.join(run_dir, emu_cfg),
        os.path.join(SCRIPT_DIR, emu_leaf),
        os.path.join(os.path.dirname(os.path.abspath(run_dir)), emu_leaf),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Could not find emulator directory '{emu_cfg}'. Tried: {candidates}."
    )


def _load_gp_caches(emulator_dir, output_names):
    import torch
    import gpytorch

    caches = {}
    for name in output_names:
        path = os.path.join(
            emulator_dir, name,
            f"GaussianProcessMatern32_{name}_best.joblib",
        )
        te = joblib.load(path)
        gp = te.model
        gp.eval()
        gp.likelihood.eval()

        x_mean = te.x_transforms[0].mean.detach().squeeze(0)
        x_std = te.x_transforms[0].std.detach().squeeze(0)

        te_y_mean = te.y_transforms[0].mean.detach().squeeze()
        te_y_std = te.y_transforms[0].std.detach().squeeze()

        if gp.y_transform is not None and getattr(gp.y_transform, "_is_fitted", False):
            gp_y_mean = gp.y_transform.mean.detach().squeeze()
            gp_y_std = gp.y_transform.std.detach().squeeze()
            y_std = gp_y_std * te_y_std
            y_mean = gp_y_mean * te_y_std + te_y_mean
        else:
            y_std, y_mean = te_y_std, te_y_mean

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _ = gp(gp.train_inputs[0][:1])

        caches[name] = {
            "gp": gp,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        }
    return caches


def _predict_at(theta_np, gp_caches, output_names):
    import torch
    import gpytorch

    theta = torch.tensor(theta_np, dtype=torch.float32)
    mu = np.zeros(len(output_names), dtype=np.float64)
    sd = np.zeros(len(output_names), dtype=np.float64)
    for i, name in enumerate(output_names):
        c = gp_caches[name]
        xt = (theta - c["x_mean"]) / c["x_std"]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out = c["gp"](xt.unsqueeze(0))
            mean_t = out.mean.squeeze()
            var_t = out.variance.squeeze().clamp(min=1e-10)
        mu[i] = (mean_t * c["y_std"] + c["y_mean"]).item()
        sd[i] = (torch.sqrt(var_t) * c["y_std"]).item()
    return mu, sd


def _resolve_output_indices(output_names):
    name_to_idx = {str(name): i for i, name in enumerate(output_names)}

    group_specs = [
        ("Rest_LA", "Rest_Min_LA_Volume", "Rest_Max_LA_Volume", "Rest_Pre_LA_Contraction_Volume"),
        ("Rest_RA", "Rest_Min_RA_Volume", "Rest_Max_RA_Volume", "Rest_Pre_RA_Contraction_Volume"),
        ("Exercise_LA", "Exercise_Min_LA_Volume", "Exercise_Max_LA_Volume", "Exercise_Pre_LA_Contraction_Volume"),
        ("Exercise_RA", "Exercise_Min_RA_Volume", "Exercise_Max_RA_Volume", "Exercise_Pre_RA_Contraction_Volume"),
    ]

    groups = []
    for key, min_name, max_name, pre_name in group_specs:
        if min_name in name_to_idx and max_name in name_to_idx and pre_name in name_to_idx:
            display_mean, display_std, label = ATRIAL_DISPLAY_STATS.get(
                key, (ATRIAL_RATIO_DISPLAY_MEAN,
                      ATRIAL_RATIO_DISPLAY_STD,
                      "LA\nPre-A\nFraction" if key == "LA" else "RA\nPre-A\nFraction")
            )
            groups.append({
                "key": key,
                "min": name_to_idx[min_name],
                "max": name_to_idx[max_name],
                "pre": name_to_idx[pre_name],
                "display_mean": display_mean,
                "display_std": display_std,
                "label": label,
            })

    if not groups:
        return None
    return {"atrial_groups": groups}


def _expected_observation_arrays(output_names):
    missing = [str(name) for name in output_names if str(name) not in observation]
    if missing:
        return None, None, missing
    means = np.array([observation[str(name)][0] for name in output_names], dtype=np.float64)
    vars_ = np.array([observation[str(name)][1] for name in output_names], dtype=np.float64)
    return means, vars_, []


def _safe_ratio_denominator(x, eps=1e-8):
    import torch

    sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return torch.where(x.abs() < eps, sign * eps, x)


def _safe_ratio_denominator_np(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    sign = np.where(x >= 0.0, 1.0, -1.0)
    return np.where(np.abs(x) < eps, sign * eps, x)


def _ratio_mean_sd_from_outputs(mean_vec, sd_vec, idx_min, idx_max, idx_pre):
    """Delta-method ratio mean/sd for plotting raw emulator output columns."""
    vmin = float(mean_vec[idx_min])
    vmax = float(mean_vec[idx_max])
    vpre = float(mean_vec[idx_pre])
    denom = float(_safe_ratio_denominator_np(vmax - vmin))
    ratio = (vpre - vmin) / denom

    d_vpre = 1.0 / denom
    d_vmin = (vpre - vmax) / (denom ** 2)
    d_vmax = -(vpre - vmin) / (denom ** 2)
    var = (
        (d_vmin * float(sd_vec[idx_min])) ** 2
        + (d_vmax * float(sd_vec[idx_max])) ** 2
        + (d_vpre * float(sd_vec[idx_pre])) ** 2
    )
    return ratio, math.sqrt(max(var, 0.0))


def _local_output_covariance(vars_, residual_corr, idxs, jitter=1e-10):
    import torch

    idx_t = torch.as_tensor(idxs, dtype=torch.long, device=vars_.device)
    local_var = vars_.index_select(0, idx_t).clamp(min=1e-12)
    local_std = torch.sqrt(local_var)
    corr = residual_corr.to(device=vars_.device, dtype=vars_.dtype)
    corr_sub = corr.index_select(0, idx_t).index_select(1, idx_t)
    cov = corr_sub * torch.outer(local_std, local_std)
    eye = torch.eye(len(idxs), dtype=vars_.dtype, device=vars_.device)
    return 0.5 * (cov + cov.T) + jitter * eye


def _ratio_moments(mean_vec, cov):
    import torch

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


def _convert_pre_outputs_to_ratios(matrix, output_indices, ratio_value_threshold=2.0):
    if matrix is None or output_indices is None:
        return matrix

    converted = np.asarray(matrix, dtype=np.float64).copy()
    squeeze = False
    if converted.ndim == 1:
        converted = converted.reshape(1, -1)
        squeeze = True
    elif converted.ndim != 2:
        return matrix

    for group in output_indices.get("atrial_groups", []):
        min_idx = group["min"]
        max_idx = group["max"]
        pre_idx = group["pre"]
        pre_values = converted[:, pre_idx]
        finite_pre = pre_values[np.isfinite(pre_values)]
        if finite_pre.size and np.nanmedian(np.abs(finite_pre)) <= ratio_value_threshold:
            continue

        denom = _safe_ratio_denominator_np(converted[:, max_idx] - converted[:, min_idx])
        converted[:, pre_idx] = (converted[:, pre_idx] - converted[:, min_idx]) / denom

    return converted.reshape(-1) if squeeze else converted


def _estimate_residual_corr(gp_caches, output_names):
    import torch
    import gpytorch

    residuals = []
    n_train = None

    for name in output_names:
        c = gp_caches[name]
        gp = c["gp"]
        x_train = gp.train_inputs[0]
        y_train = gp.train_targets.detach().reshape(-1).to(torch.float32)
        if n_train is None:
            n_train = y_train.numel()
        elif y_train.numel() != n_train:
            raise ValueError(
                f"Training target length mismatch while estimating residual correlation for {name}."
            )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mu_latent = gp(x_train).mean.detach().reshape(-1).to(torch.float32)

        y_std = c["y_std"].detach().to(torch.float32)
        y_mean = c["y_mean"].detach().to(torch.float32)
        y_train_phys = y_train * y_std + y_mean
        mu_train_phys = mu_latent * y_std + y_mean
        residuals.append(y_train_phys - mu_train_phys)

    residual = torch.stack(residuals, dim=0)
    residual_centered = residual - residual.mean(dim=1, keepdim=True)
    residual_cov = residual_centered @ residual_centered.T / max(n_train - 1, 1)
    residual_std = torch.sqrt(torch.diagonal(residual_cov).clamp_min(1e-12))
    residual_corr = residual_cov / (
        residual_std.unsqueeze(1) * residual_std.unsqueeze(0)
    ).clamp_min(1e-12)
    residual_corr = torch.nan_to_num(residual_corr, nan=0.0, posinf=0.0, neginf=0.0)
    residual_corr.fill_diagonal_(1.0)
    return residual_corr


# ---------------------------------------------------------------------
# Copula prior loading / evaluation
# ---------------------------------------------------------------------

def _load_copula_prior(run_dir):
    import torch

    path = os.path.join(run_dir, "copula_prior.joblib")
    if not os.path.exists(path):
        return None
    raw = joblib.load(path)
    cop = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray):
            dtype = torch.long if v.dtype.kind in ("i", "u") else torch.float32
            cop[k] = torch.tensor(v, dtype=dtype)
        else:
            cop[k] = v
    if cop.get("marginal_type") == "logspline" and "arange_d" not in cop:
        cop["arange_d"] = torch.arange(cop["x_grid"].shape[0], dtype=torch.long)
    if "F_eps" in cop and not isinstance(cop["F_eps"], torch.Tensor):
        cop["F_eps"] = torch.tensor(float(cop["F_eps"]), dtype=torch.float32)
    if "half_logdet_R" in cop and not isinstance(cop["half_logdet_R"], torch.Tensor):
        cop["half_logdet_R"] = torch.tensor(float(cop["half_logdet_R"]), dtype=torch.float32)
    return cop


def _copula_log_prob(theta, cop):
    import torch

    u = (theta.unsqueeze(-1) - cop["kde_data"]) / cop["kde_h"].unsqueeze(-1)
    log_phi = -0.5 * u ** 2 - 0.5 * math.log(2.0 * math.pi)
    log_fi = (
        torch.logsumexp(log_phi, dim=-1)
        - torch.log(torch.tensor(float(cop["M"]), dtype=theta.dtype, device=theta.device))
        - torch.log(cop["kde_h"].to(theta.device))
    )
    Fi = 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0))).mean(dim=-1)
    Fi = Fi.clamp(cop["F_eps"].to(theta.device), 1.0 - cop["F_eps"].to(theta.device))
    z = math.sqrt(2.0) * torch.erfinv(2.0 * Fi - 1.0)
    w = torch.linalg.solve_triangular(
        cop["L_R"].to(theta.device), z.unsqueeze(-1), upper=False
    ).squeeze(-1)
    log_copula = -0.5 * ((w ** 2).sum() - (z ** 2).sum()) - cop["half_logdet_R"].to(theta.device)
    return log_copula + log_fi.sum()


def _logspline_copula_log_prob(theta, cop):
    import torch

    x_grid = cop["x_grid"]
    f_grid = cop["f_grid"]
    cdf_grid = cop["cdf_grid"]
    L_R = cop["L_R"]
    half_logdet_R = cop["half_logdet_R"]
    F_eps = cop["F_eps"]
    _, G = x_grid.shape

    theta_c = theta.to(x_grid.dtype)
    idx_hi = torch.searchsorted(x_grid, theta_c.unsqueeze(-1).contiguous()).squeeze(-1)
    idx_hi = idx_hi.clamp(min=1, max=G - 1)
    idx_lo = idx_hi - 1

    arange_d = cop["arange_d"]
    x_lo = x_grid[arange_d, idx_lo]
    x_hi = x_grid[arange_d, idx_hi]
    f_lo = f_grid[arange_d, idx_lo]
    f_hi = f_grid[arange_d, idx_hi]
    F_lo = cdf_grid[arange_d, idx_lo]

    dx = (x_hi - x_lo).clamp_min(1e-30)
    slope = (f_hi - f_lo) / dx
    delta = theta_c - x_lo

    f_th = (f_lo + slope * delta).clamp_min(1e-30)
    F_th = F_lo + f_lo * delta + 0.5 * slope * delta ** 2
    F_th = F_th.clamp(F_eps, 1.0 - F_eps)

    z = math.sqrt(2.0) * torch.erfinv(2.0 * F_th - 1.0)
    w = torch.linalg.solve_triangular(L_R, z.unsqueeze(-1), upper=False).squeeze(-1)
    log_copula = -0.5 * ((w ** 2).sum() - (z ** 2).sum()) - half_logdet_R
    return log_copula + torch.log(f_th).sum()


# ---------------------------------------------------------------------
# Theta-space posterior objective, optimized in unconstrained z-space
# ---------------------------------------------------------------------

def make_theta_space_objective(prior_lo, prior_hi, obs_means, obs_vars, gp_caches,
                               output_names, copula=None, residual_corr=None,
                               output_indices=None):
    import torch
    import gpytorch

    prior_lo_t = torch.tensor(prior_lo, dtype=torch.float32)
    prior_hi_t = torch.tensor(prior_hi, dtype=torch.float32)
    obs_means_t = torch.tensor(obs_means, dtype=torch.float32)
    obs_vars_t = torch.tensor(obs_vars, dtype=torch.float32)

    def log_post_theta_from_z(z_t):
        sig_z = torch.sigmoid(z_t)
        theta = prior_lo_t + sig_z * (prior_hi_t - prior_lo_t)

        mus = []
        vars_ = []
        for i, name in enumerate(output_names):
            c = gp_caches[name]
            xt = (theta - c["x_mean"]) / c["x_std"]
            with gpytorch.settings.fast_pred_var():
                out = c["gp"](xt.unsqueeze(0))
            mean_t = out.mean.squeeze()
            var_t = out.variance.squeeze().clamp(min=1e-10)
            mus.append(mean_t * c["y_std"] + c["y_mean"])
            vars_.append(var_t * (c["y_std"] ** 2))

        mus = torch.stack(mus)
        vars_ = torch.stack(vars_)

        total_var = (obs_vars_t + vars_).clamp(min=1e-10)
        z_norm = (obs_means_t - mus) / torch.sqrt(total_var)
        ll = -0.5 * (z_norm ** 2 + torch.log(total_var)).sum()
        target_z_obs = (mus - obs_means_t) / torch.sqrt(obs_vars_t.clamp(min=1e-10))

        atrial_groups = [] if output_indices is None else output_indices.get("atrial_groups", [])
        if atrial_groups:
            ratio_corr = residual_corr
            if ratio_corr is None:
                ratio_corr = torch.eye(mus.numel(), dtype=vars_.dtype, device=vars_.device)

            gaussian_mask = torch.ones_like(mus, dtype=torch.bool)
            atrial_idxs = [group["pre"] for group in atrial_groups]
            gaussian_mask[atrial_idxs] = False
            ll = -0.5 * (
                z_norm[gaussian_mask] ** 2 + torch.log(total_var[gaussian_mask])
            ).sum()

            target_z_obs = target_z_obs.clone()
            for group in atrial_groups:
                idxs = (group["min"], group["max"], group["pre"])
                ratio_mean, ratio_var = _ratio_moments(
                    mus[list(idxs)],
                    _local_output_covariance(vars_, ratio_corr, idxs),
                )
                pre_idx = group["pre"]
                ratio_total_var = (obs_vars_t[pre_idx] + ratio_var).clamp(min=1e-10)
                ll = ll - 0.5 * (
                    (obs_means_t[pre_idx] - ratio_mean) ** 2 / ratio_total_var
                    + torch.log(ratio_total_var)
                )
                target_z_obs[pre_idx] = 0.0

        target_excess = torch.relu(torch.abs(target_z_obs) - 0.9)
        ll = ll - 25 * (target_excess ** 2).sum()

        lp = torch.tensor(0.0, dtype=torch.float32)
        # if copula is None:
        #     lp = torch.tensor(0.0, dtype=torch.float32)
        # elif copula.get("marginal_type") == "logspline":
        #     lp = _logspline_copula_log_prob(theta, copula)
        # else:
        #     lp = _copula_log_prob(theta, copula)

        total = ll + lp
        return torch.nan_to_num(total, nan=-1e8, posinf=-1e8, neginf=-1e8)

    return log_post_theta_from_z


def _boundary_hits(theta, prior_lo, prior_hi, names=None, tol=0.01):
    theta = np.asarray(theta, dtype=np.float64)
    prior_lo = np.asarray(prior_lo, dtype=np.float64)
    prior_hi = np.asarray(prior_hi, dtype=np.float64)
    names = list(names) if names is not None else [f"param_{i}" for i in range(theta.size)]

    width = prior_hi - prior_lo
    frac = np.full(theta.shape, np.nan, dtype=np.float64)
    valid = np.abs(width) > 0.0
    frac[valid] = (theta[valid] - prior_lo[valid]) / width[valid]

    hits = []
    for i, value in enumerate(frac):
        if not np.isfinite(value):
            continue
        if value <= tol or value >= 1.0 - tol:
            hits.append({
                "name": str(names[i]),
                "side": "lower" if value <= tol else "upper",
                "fraction": float(value),
                "value": float(theta[i]),
                "lower": float(prior_lo[i]),
                "upper": float(prior_hi[i]),
            })
    return hits


def _make_z_bounds(ndim, posterior_z=None, z_trust_quantile=0.001,
                   min_bound_fraction=1e-4):
    bounds = None

    if min_bound_fraction is not None and min_bound_fraction > 0.0:
        eps = float(min_bound_fraction)
        if not 0.0 < eps < 0.5:
            raise ValueError("--min-bound-fraction must be in (0, 0.5).")
        lo = np.full(ndim, _logit_np(eps), dtype=np.float64)
        hi = np.full_like(lo, _logit_np(1.0 - eps))
        bounds = (lo, hi)

    if posterior_z is not None and z_trust_quantile is not None and z_trust_quantile > 0.0:
        q = float(z_trust_quantile)
        if not 0.0 < q < 0.5:
            raise ValueError("--z-trust-quantile must be in (0, 0.5), or 0 to disable.")
        z_lo = np.nanquantile(posterior_z, q, axis=0).astype(np.float64)
        z_hi = np.nanquantile(posterior_z, 1.0 - q, axis=0).astype(np.float64)
        if bounds is None:
            bounds = (z_lo, z_hi)
        else:
            lo, hi = bounds
            if lo.size == 1:
                lo = np.full_like(z_lo, lo.item())
                hi = np.full_like(z_hi, hi.item())
            bounds = (np.maximum(lo, z_lo), np.minimum(hi, z_hi))

    if bounds is None:
        return None

    lo, hi = bounds
    bad = lo >= hi
    if np.any(bad):
        eps_lo = _logit_np(float(min_bound_fraction or 1e-8))
        eps_hi = _logit_np(1.0 - float(min_bound_fraction or 1e-8))
        lo[bad] = eps_lo
        hi[bad] = eps_hi
    return list(zip(lo.tolist(), hi.tolist()))


def _run_one_start(s, objective_fn, prior_lo, prior_hi, z_bounds,
                   z_lo, z_hi, maxiter, gtol):
    from scipy.optimize import minimize
    import torch

    # Worker is a fresh loky process; pin to 1 thread so n_starts workers
    # don't oversubscribe (each L-BFGS-B step is dominated by per-GP Python
    # overhead, not BLAS, so 1 thread per worker is fastest).
    torch.set_num_threads(1)

    def fun_and_grad(z_np):
        z_t = torch.tensor(z_np, dtype=torch.float32, requires_grad=True)
        logp = objective_fn(z_t)
        loss = -logp
        loss.backward()
        grad = z_t.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.item()), grad

    z0 = np.asarray(s["sample_z"], dtype=np.float64)
    x0 = z0.copy()
    if z_lo is not None:
        x0 = np.clip(x0, z_lo, z_hi)
    opt = minimize(
        fun=lambda z: fun_and_grad(z)[0],
        x0=x0,
        jac=lambda z: fun_and_grad(z)[1],
        method="L-BFGS-B",
        bounds=z_bounds,
        options={"maxiter": int(maxiter), "gtol": float(gtol), "maxls": 50},
    )
    z_star = opt.x.astype(np.float64)
    theta_star = prior_lo + (prior_hi - prior_lo) * _sigmoid_np(z_star)
    edge_fraction = np.minimum(
        (theta_star - prior_lo) / (prior_hi - prior_lo),
        (prior_hi - theta_star) / (prior_hi - prior_lo),
    )
    logp_star = -float(opt.fun)
    return {
        "start_rank": int(s["rank"]),
        "start_flat_idx": int(s["flat_idx"]),
        "start_log_post_theta": float(s["log_post_theta"]),
        "success": bool(opt.success),
        "message": str(opt.message),
        "nit": int(getattr(opt, "nit", -1)),
        "nfev": int(getattr(opt, "nfev", -1)),
        "final_log_post_theta": logp_star,
        "z": z_star,
        "theta": theta_star,
        "min_edge_fraction": float(np.min(edge_fraction)),
        "n_edge_fraction_lt_1e-3": int(np.sum(edge_fraction < 1e-3)),
        "n_edge_fraction_lt_1e-2": int(np.sum(edge_fraction < 1e-2)),
        "improvement": logp_star - float(s["log_post_theta"]),
    }


def refine_map_multistart(top_k, objective_fn, prior_lo, prior_hi,
                          n_starts=10, maxiter=600, gtol=1e-5,
                          posterior_z=None, z_trust_quantile=0.001,
                          min_bound_fraction=1e-4, n_jobs=None):
    from joblib import Parallel, delayed
    from joblib.externals.loky import get_reusable_executor

    n_starts = min(n_starts, len(top_k))
    starts = top_k[:n_starts]
    z_bounds = _make_z_bounds(
        ndim=len(prior_lo),
        posterior_z=posterior_z,
        z_trust_quantile=z_trust_quantile,
        min_bound_fraction=min_bound_fraction,
    )

    if z_bounds is not None:
        z_lo = np.asarray([b[0] for b in z_bounds], dtype=np.float64)
        z_hi = np.asarray([b[1] for b in z_bounds], dtype=np.float64)
    else:
        z_lo = z_hi = None

    effective_jobs = n_starts if n_jobs is None else int(n_jobs)
    effective_jobs = max(1, min(effective_jobs, n_starts))

    if effective_jobs == 1 or n_starts == 1:
        results = [
            _run_one_start(s, objective_fn, prior_lo, prior_hi,
                           z_bounds, z_lo, z_hi, maxiter, gtol)
            for s in starts
        ]
    else:
        results = Parallel(n_jobs=effective_jobs, backend="loky")(
            delayed(_run_one_start)(
                s, objective_fn, prior_lo, prior_hi,
                z_bounds, z_lo, z_hi, maxiter, gtol,
            )
            for s in starts
        )
        get_reusable_executor().shutdown(wait=True)

    best = max(results, key=lambda r: r["final_log_post_theta"])
    return best, results




def _plot_vs_targets(run_dir, output_names,
                     obs_means, obs_stds,
                     sampled_mu, sampled_sd,
                     refined_mu, refined_sd,
                     pred_matrix=None, output_indices=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_out = len(output_names)
    x_pos = np.arange(n_out)
    short_names = [str(n).replace("_", "\n") for n in output_names]
    plot_obs_means = np.asarray(obs_means, dtype=np.float64).copy()
    plot_obs_stds = np.asarray(obs_stds, dtype=np.float64).copy()
    sampled_plot_mu = np.asarray(sampled_mu, dtype=np.float64).copy()
    sampled_plot_sd = np.asarray(sampled_sd, dtype=np.float64).copy()
    refined_plot_mu = np.asarray(refined_mu, dtype=np.float64).copy()
    refined_plot_sd = np.asarray(refined_sd, dtype=np.float64).copy()
    plot_matrix = None if pred_matrix is None else np.asarray(pred_matrix, dtype=np.float64).copy()

    atrial_groups = [] if output_indices is None else output_indices.get("atrial_groups", [])
    if plot_matrix is not None and atrial_groups:
        plot_matrix = _convert_pre_outputs_to_ratios(plot_matrix, output_indices)

    for group in atrial_groups:
        min_idx = group["min"]
        max_idx = group["max"]
        pre_idx = group["pre"]

        plot_obs_means[pre_idx] = group["display_mean"]
        plot_obs_stds[pre_idx] = group["display_std"]
        short_names[pre_idx] = group["label"]

        sampled_plot_mu[pre_idx], sampled_plot_sd[pre_idx] = _ratio_mean_sd_from_outputs(
            sampled_mu, sampled_sd, min_idx, max_idx, pre_idx
        )
        refined_plot_mu[pre_idx], refined_plot_sd[pre_idx] = _ratio_mean_sd_from_outputs(
            refined_mu, refined_sd, min_idx, max_idx, pre_idx
        )

    fig, ax = plt.subplots(figsize=(max(12, 0.6 * n_out), 6))

    if plot_matrix is not None:
        normalised = (plot_matrix - plot_obs_means) / plot_obs_stds
        ax.boxplot(
            normalised, positions=x_pos, widths=0.5,
            showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.35,
                          edgecolor="steelblue"),
            medianprops=dict(color="steelblue", linewidth=1.2),
            whiskerprops=dict(color="steelblue", alpha=0.6),
            capprops=dict(color="steelblue", alpha=0.6),
        )

    ax.axhline(0, color="black", ls="-", linewidth=0.6)
    ax.axhline(1, color="green", ls="--", alpha=0.6, label=r"$\pm 1\sigma$")
    ax.axhline(-1, color="green", ls="--", alpha=0.6)
    ax.axhline(3, color="red", ls=":", alpha=0.4, label=r"$\pm 3\sigma$")
    ax.axhline(-3, color="red", ls=":", alpha=0.4)

    sampled_res = (sampled_plot_mu - plot_obs_means) / plot_obs_stds
    sampled_err = sampled_plot_sd / plot_obs_stds
    ax.errorbar(
        x_pos, sampled_res, yerr=sampled_err,
        fmt="D", color="navy", markersize=6, markeredgecolor="white",
        markeredgewidth=0.5, elinewidth=0.9, capsize=2.5, zorder=4,
        label="Sampled MAP",
    )

    refined_res = (refined_plot_mu - plot_obs_means) / plot_obs_stds
    refined_err = refined_plot_sd / plot_obs_stds
    ax.errorbar(
        x_pos, refined_res, yerr=refined_err,
        fmt="*", color="crimson", markersize=11, markeredgecolor="black",
        markeredgewidth=0.4, elinewidth=1.0, capsize=2.5, zorder=5,
        label="Refined MAP",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_ylabel(r"(predicted - target) / $\sigma_{\mathrm{obs}}$")
    ax.set_title("Sampled vs refined MAP emulator predictions vs. targets")

    all_low = np.concatenate([sampled_res - sampled_err, refined_res - refined_err])
    all_high = np.concatenate([sampled_res + sampled_err, refined_res + refined_err])
    ymin = min(-3.5, np.nanpercentile(all_low, 5), np.nanmin(all_low) - 0.5)
    ymax = max(3.5, np.nanpercentile(all_high, 95), np.nanmax(all_high) + 0.5)
    ax.set_ylim(ymin, ymax)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "refined_map_vs_targets.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

# ---------------------------------------------------------------------
# Reporting / save
# ---------------------------------------------------------------------

def _save_outputs(run_dir, sampled_map, refined_best, all_runs,
                  sampled_pred=None, refined_pred=None,
                  sampled_sd=None, refined_sd=None):
    np.save(os.path.join(run_dir, "sampled_map_sample.npy"), sampled_map["sample"])
    np.save(os.path.join(run_dir, "sampled_map_sample_z.npy"), sampled_map["sample_z"])
    np.save(os.path.join(run_dir, "refined_map_sample.npy"), refined_best["theta"])
    np.save(os.path.join(run_dir, "refined_map_sample_z.npy"), refined_best["z"])

    if sampled_pred is not None:
        np.save(os.path.join(run_dir, "sampled_map_predictions.npy"), sampled_pred)
        np.save(os.path.join(run_dir, "sampled_map_prediction_stds.npy"), sampled_sd)
    if refined_pred is not None:
        np.save(os.path.join(run_dir, "refined_map_predictions.npy"), refined_pred)
        np.save(os.path.join(run_dir, "refined_map_prediction_stds.npy"), refined_sd)

    info = {
        "sampled_map": {
            "flat_idx": int(sampled_map["flat_idx"]),
            "chain": int(sampled_map["chain"]),
            "draw": int(sampled_map["draw"]),
            "log_post_theta": float(sampled_map["log_post_theta"]),
            "log_post_z": float(sampled_map["log_post_z"]),
        },
        "refined_map": {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else bool(v) if isinstance(v, (np.bool_, bool)) else str(v))
            for k, v in refined_best.items() if k not in ("theta", "z")
        },
        "all_starts": [
            {
                k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else bool(v) if isinstance(v, (np.bool_, bool)) else str(v))
                for k, v in row.items() if k not in ("theta", "z")
            }
            for row in all_runs
        ],
    }
    with open(os.path.join(run_dir, "refined_map_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Refine theta-space MAP by local optimisation from top posterior draws.")
    p.add_argument(
        "run_dir",
        nargs="?",
        # default=os.path.join("MCMC_Union_50_29_05_logspline_copula_prior"),
        default=os.path.join("MCMC_Union_50_28_08_copula_prior"),  # change
        help="Path to a MCMC_Union_* output directory.",
    )
    p.add_argument("--top-k", type=int, default=10, help="How many top posterior draws to rank.")
    p.add_argument("--n-starts", type=int, default=5, help="How many of the top draws to use as optimisation starts.")
    p.add_argument("--maxiter", type=int, default=600, help="Maximum L-BFGS iterations per start.")
    p.add_argument("--gtol", type=float, default=1e-5, help="Projected-gradient tolerance for L-BFGS-B.")
    p.add_argument(
        "--z-trust-quantile",
        type=float,
        default=0.001,
        help=(
            "Constrain refinement to per-parameter posterior_z quantiles "
            "[q, 1-q]. Use 0 to disable this trust region."
        ),
    )
    p.add_argument(
        "--min-bound-fraction",
        type=float,
        default=1e-4,
        help=(
            "Keep refined parameters this fractional distance inside the "
            "prior box. Use 0 to allow exact prior boundaries."
        ),
    )
    p.add_argument("--emulator-dir", default=None, help="Override EMULATOR_DIR from config.json.")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Parallel workers for the multistart loop (loky backend). "
            "Defaults to --n-starts. Pass 1 to disable parallelism."
        ),
    )
    p.add_argument("--no-save", action="store_true", help="Print only; do not save outputs.")
    args = p.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        run_dir_candidates = [
            run_dir,
            os.path.join(SCRIPT_DIR, run_dir),
            os.path.join(os.path.dirname(SCRIPT_DIR), run_dir),
        ]
        run_dir = next((c for c in run_dir_candidates if os.path.isdir(c)), None)
        if run_dir is None:
            raise SystemExit(f"run_dir not found. Tried: {run_dir_candidates}")

    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    output_names = np.load(os.path.join(run_dir, "output_names.npy"), allow_pickle=True).tolist()
    obs_means = np.load(os.path.join(run_dir, "obs_means.npy"))
    obs_vars = np.load(os.path.join(run_dir, "obs_vars.npy"))
    expected_means, expected_vars, missing_obs = _expected_observation_arrays(output_names)
    if not missing_obs:
        if not (np.allclose(obs_means, expected_means) and np.allclose(obs_vars, expected_vars)):
            print(
                "WARNING: saved obs_means/obs_vars differ from the in-script union "
                "targets. Using in-script Rest+Exercise targets for refinement; "
                "rerun MCMC_Union.py if the posterior was generated with stale targets."
            )
        obs_means = expected_means
        obs_vars = expected_vars

    map_info = compute_map(run_dir, top_k=args.top_k)
    prior_lo = map_info["prior_lower"]
    prior_hi = map_info["prior_upper"]
    sampled_map = map_info["sampled_map"]

    if args.emulator_dir is not None:
        emulator_dir = args.emulator_dir
        if not os.path.isdir(emulator_dir):
            raise SystemExit(f"--emulator-dir not found: {emulator_dir}")
    else:
        emulator_dir = _resolve_emulator_dir(run_dir, cfg)

    print(f"Loading emulators from: {emulator_dir}")
    gp_caches = _load_gp_caches(emulator_dir, output_names)
    residual_corr = _estimate_residual_corr(gp_caches, output_names)
    output_indices = _resolve_output_indices(output_names)
    copula = _load_copula_prior(run_dir) if cfg.get("use_copula_prior", False) else None

    objective_fn = make_theta_space_objective(
        prior_lo=prior_lo,
        prior_hi=prior_hi,
        obs_means=obs_means,
        obs_vars=obs_vars,
        gp_caches=gp_caches,
        output_names=output_names,
        copula=copula,
        residual_corr=residual_corr,
        output_indices=output_indices,
    )

    refined_best, all_runs = refine_map_multistart(
        top_k=map_info["top_k"],
        objective_fn=objective_fn,
        prior_lo=prior_lo,
        prior_hi=prior_hi,
        n_starts=args.n_starts,
        maxiter=args.maxiter,
        gtol=args.gtol,
        posterior_z=map_info["posterior_z"],
        z_trust_quantile=args.z_trust_quantile,
        min_bound_fraction=args.min_bound_fraction,
        n_jobs=args.n_jobs,
    )

    print()
    print("Sampled theta-space MAP:")
    print(f"  chain/draw         = {sampled_map['chain']} / {sampled_map['draw']}")
    print(f"  flat index         = {sampled_map['flat_idx']}")
    print(f"  log p(theta|y)     = {sampled_map['log_post_theta']:.6f}")

    print()
    print("Refined theta-space MAP:")
    print(f"  start rank         = {refined_best['start_rank']}")
    print(f"  converged          = {refined_best['success']}")
    print(f"  iterations         = {refined_best['nit']}")
    print(f"  log p(theta|y)     = {refined_best['final_log_post_theta']:.6f}")
    print(f"  improvement        = {refined_best['improvement']:.6f}")
    print(f"  min edge fraction  = {refined_best['min_edge_fraction']:.6g}")
    print(f"  edge count <1e-3   = {refined_best['n_edge_fraction_lt_1e-3']}")
    print(f"  edge count <1e-2   = {refined_best['n_edge_fraction_lt_1e-2']}")
    print(f"  message            = {refined_best['message']}")

    print()
    print("All starts:")
    for row in all_runs:
        print(
            f"  start rank {row['start_rank']:>2d}: "
            f"success={str(row['success']):<5}  "
            f"logp={row['final_log_post_theta']:.6f}  "
            f"delta={row['improvement']:+.6f}  "
            f"nit={row['nit']:>3d}  "
            f"min_edge={row['min_edge_fraction']:.3g}"
        )

    sampled_mu, sampled_sd = _predict_at(sampled_map["sample"], gp_caches, output_names)
    refined_mu, refined_sd = _predict_at(refined_best["theta"], gp_caches, output_names)

    obs_stds = np.sqrt(obs_vars)
    pred_matrix = None
    pred_matrix_path = os.path.join(run_dir, "pred_check_matrix.npy")
    if os.path.exists(pred_matrix_path):
        pred_matrix = np.load(pred_matrix_path)

    plot_path = _plot_vs_targets(
        run_dir=run_dir,
        output_names=output_names,
        obs_means=obs_means,
        obs_stds=obs_stds,
        sampled_mu=sampled_mu,
        sampled_sd=sampled_sd,
        refined_mu=refined_mu,
        refined_sd=refined_sd,
        pred_matrix=pred_matrix,
        output_indices=output_indices,
    )

    if not args.no_save:
        _save_outputs(
            run_dir=run_dir,
            sampled_map=sampled_map,
            refined_best=refined_best,
            all_runs=all_runs,
            sampled_pred=sampled_mu,
            refined_pred=refined_mu,
            sampled_sd=sampled_sd,
            refined_sd=refined_sd,
        )
        print()
        print(f"Saved: {os.path.join(run_dir, 'refined_map_sample.npy')}")
        print(f"Saved: {os.path.join(run_dir, 'refined_map_sample_z.npy')}")
        print(f"Saved: {os.path.join(run_dir, 'refined_map_info.json')}")
        print(f"Saved: {plot_path}")



    from SALib import ProblemSpec

    upper = 1.5
    lower = 0.5

    # change
    sp = ProblemSpec({
    'names': [
        # gas
        "beta2", "C2", "K2", "a2",
        "alpha2", "KCCO2", "GV_dead",
        # resp control
        "KcCO2", "KcMRV", "KpCO2", "KpO2",
        "V0_dead", "VA_rest",
        "E_rs", "R_rs",
        # cardio
        "C_jp", "C_sa", "L_sa", "R_sa",
        "C_amv", "C_bv", "C_ev", "C_hv",
        "C_rmv", "C_sv", "kr_am", "P_0",
        "R_amv_n", "R_bv_n", "R_ev_n", "R_hv_n",
        "R_rmv_n", "R_sv_n", "K1_vc", "D1",
        "Vvc_min", "Kr_vc",
        "Rvc_n", "C_pa", "C_pp",
        "C_pv", "L_pa", "R_pa", "R_pp",
        "R_pv", "Emax_la", "P0_la", "Emax_ra",
        "P0_ra", "KE_la", "KE_ra", "P0_lv",
        "P0_rv",
        "s",
        # cardio control
        "fab_o", "fes_o", "fes_inf", "fes_max",
        "fev_o", "fev_inf", "kes", "kev",
        "Io_sh", "Io_sp", "Io_sv", "Io_v",
        "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v",
        "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
        "Ysv_max", "Ysv_min", "Yv_max", "Yv_min",
        "theta_v", "Wb_sh", "Wb_sp", "Wb_sv",
        "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v",
        "Wp_sp", "Wp_sv", "Wp_v",
        "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
        "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv",
        "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp",
        "GR_sp", "GV_amv", "GV_ev", "GV_rmv",
        "GV_sv", "R_amp0", "R_ep0", "R_rmp0",
        #
        "R_sp0", "g_ccsh", "g_ccsp",
        "kisc_sh", "kisc_sp", "kisc_sv",
        "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn",
        "theta_spn", "theta_svn", "x_sh", "x_sp",
        "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min",
        "k_ab", "P_n", "P_n_max", "f_acCO2_n",
        "f_ac_max", "f_ac_min", "k_ac", "K_H",
        "PaO2_ac_n", "G_ap", "GT_s", "GT_v",
        "T0", "A", "B", "C",
        "D", "Cvb_O2_n", "gb_O2", "MO2_bp",
        "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2",
        "grm_O2", "Kh_CO2", "Krm_CO2", "MO2_hpn",
        "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n",
        "gam_O2", "gM", "Io_met", "kmet",
        "MO2_ampn", "phi_max", "phi_min",
        # added params
        "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max",
        "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max",
        "Kp_po", "Kf_po", "Kb_po", "Kv_po", "theta_po_max",
        "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max",
        "alpha_O2", "R_po", "R_mi", "R_tr",
        "R_ao", "C_O2_param1", "C_O2_param2", "C_O2_param3",
        "PAMO2_nominal", "Vu_bv", "Vu_hv",
        "Vu_jp", "Vu_vc",
        "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv",
        "Vu_ra", "Vu_rv",

        "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
        "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv",
        "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0",
        "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc",
        "tau_isc", "tau_p", "tau_z", "tau_ac",
        "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2",
        "tau_O2", "tau_w", "tau_M", "tau_met",
        "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep",
        "DR_rmp", "DR_sp", "DV_amv", "DV_ev",
        "DV_rmv", "DV_sv", "DT_s", "DT_v",
        "Dmet", "Ta", "KE_lv", "KE_rv",
        "T1", "T2", "VL_CO2", "VL_O2",
        "KCSFCO2", "VB", "tauMR", "VTCO2",
        "VTO2", "tau_MRV",

        # further added
        "scale_param1", "scale_param3", "scale_param4",
        "scale_param6", "Pa_O2_lower",
        "rise_time_atr", "rise_time_ven", "fall_time_ven", "ahead1",
        "theta_min", "r", "l", "V_nominal", "V_scale"
    ],

    'bounds': [
        # gas
        [0.03255 * lower, 0.03255 * upper], [87 * lower, 87 * upper], [194.4 * lower, 194.4 * upper], [1.819 * lower, 1.819 * upper],
        [0.05591 * lower, 0.05591 * upper], [346000 * lower, 346000 * upper], [0.1698 * lower, 0.1698 * upper],
        # resp control
        [0.2332 * lower, 0.2332 * upper], [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
        [0.1587 * lower, 0.1587 * upper], [0.0673 * lower, 0.0673 * upper],
        [21.9 * lower, 21.9 * 1.2], [3.02 * 0.8, 3.02 * upper],
        # cardio
        [3.72 * lower, 3.72 * upper], [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.06 * lower, 0.06 * upper],
        [9.4 * lower, 9.4 * upper], [10.71 * lower, 10.71 * upper], [20 * lower, 20 * upper], [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper], [61.11 * lower, 61.11 * upper], [24.17 * lower, 24.17 * upper], [10 * lower, 10 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper], [0.224 * lower, 0.224 * upper],
        [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper], [0.15 * lower, 0.15 * upper], [0.3855 * lower, 0.3855 * upper],
        [50 * lower, 50 * upper], [10000 * lower, 10000 * upper],
        [0.025 * lower, 0.025 * upper], [5.85 * lower, 5.85 * upper], [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper], [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
        [0.0056 * lower, 0.0056 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper],
        [0.45 * lower, 0.45 * upper], [0.05 * lower, 0.05 * upper], [0.05 * lower, 0.05 * upper], [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper],
        [0.04 * lower, 0.04 * upper],
        # cardio control
        [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper], [80 * lower, 80 * upper],
        [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper], [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper], [0.65 * lower, 0.65 * upper], [0.45 * lower, 0.45 * upper], [0.126 * lower, 0.126 * upper],
        [0.114 * lower, 0.114 * upper], [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
        [9 * lower, 9 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper], [-0.037 * upper, -0.037 * lower],
        [64.9 * lower, 64.9 * upper], [-0.437 * upper, -0.437 * lower], [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower],
        [-0.68 * upper, -0.68 * lower], [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
        [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper], [0.2 * lower, 0.2 * upper],
        [-0.3997 * upper, -0.3997 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower],
        [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
        [2.392 * lower, 2.392 * upper], [1.412 * lower, 1.412 * upper], [2.66 * lower, 2.66 * upper], [0.475 * lower, 0.475 * upper],
        [0.282 * lower, 0.282 * upper], [2.47 * lower, 2.47 * upper], [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper],
        [0.695 * lower, 0.695 * upper], [-58.29 * upper, -58.29 * lower], [-74.21 * upper, -74.21 * lower], [-58.29 * upper, -58.29 * lower],
        [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper], [5.27 * lower, 5.27 * upper],
        #
        [2.49 * lower, 2.49 * upper], [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper],
        [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper], [3.6 * lower, 3.6 * upper],
        [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper], [53 * lower, 53 * upper], [6 * lower, 6 * upper],
        [6 * lower, 6 * upper], [40 * lower, 40 * upper], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
        [11.76 * lower, 11.76 * upper], [92 * lower, 92 * 1.15], [120 * 0.9, 120 * upper], [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper], [3 * lower, 3 * upper],
        [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper], [-0.13 * upper, -0.13 * lower], [0.09 * lower, 0.09 * upper],
        [0.58 * lower, 0.58 * upper], [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper], [0.925 * lower, 0.925 * upper],
        [6.57 * lower, 6.57 * upper], [0.11 * lower, 0.11 * upper], [0.155 * lower, 0.155 * upper], [35 * lower, 35 * upper],
        [30 * lower, 30 * upper], [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper], [0.4 * lower, 0.4 * upper],
        [0.86 * lower, 0.86 * upper], [19.71 * lower, 19.71 * upper], [12660 * lower, 12660 * upper], [0.1555 * lower, 0.1555 * upper],
        [30 * lower, 30 * upper], [40 * lower, 40 * upper], [0.4266 * lower, 0.4266 * upper], [0.18 * lower, 0.18 * upper],
        [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper], [-1.87 * upper, -1.87 * lower],
        # added params
        [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper], [1.309 * lower, 1.309 * upper],
        [1200 * lower, 1200 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [3.5 * lower, 3.5 * upper], [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper], [2000 * lower, 2000 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper], [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [3.5 * lower, 3.5 * upper], [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper], [350 * lower, 350 * upper], [400 * lower, 400 * upper], [400 * lower, 400 * upper],
        [350 * lower, 350 * upper], [0.00134 * lower, 0.00134 * upper], [2.6 * lower, 2.6 * upper], [3.03e-5 * lower, 3.03e-5 * upper],
        [104 * lower, 104 * upper], [279.49 * lower, 279.49 * upper], [93.16 * lower, 93.16 * upper],
        [579.76 * lower, 579.76 * upper], [123 * lower, 123 * upper],
        [116.68 * lower, 116.68 * upper], [114 * lower, 114 * upper], [24 * lower, 24 * upper], [15.908 * lower, 15.908 * upper],
        [27 * lower, 27 * upper], [38.703 * lower, 38.703 * upper],

        [8 * lower, 8 * upper], [8 * lower, 8 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [20 * lower, 20 * upper],
        [20 * lower, 20 * upper], [20 * lower, 20 * upper], [20 * lower, 20 * upper], [286.4 * lower, 286.4 * upper],
        [607.8 * lower, 607.8 * upper], [190.95 * lower, 190.95 * upper], [1361.6 * lower, 1361.6 * upper], [20 * lower, 20 * upper],
        [30 * lower, 30 * upper], [2.076 * lower, 2.076 * upper], [6.37 * lower, 6.37 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper], [20 * lower, 20 * upper],
        [10 * lower, 10 * upper], [5 * lower, 5 * upper], [40 * lower, 40 * upper], [10 * lower, 10 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
        [5 * lower, 5 * upper], [5 * lower, 5 * upper], [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper],
        [4 * lower, 4 * upper], [0.3 * lower, 0.3 * upper], [0.014 * lower, 0.014 * upper], [0.011 * lower, 0.011 * upper],
        [1 * lower, 1 * upper], [2 * lower, 2 * upper], [3 * lower, 3 * upper], [2.5 * lower, 2.5 * upper],
        [320 * lower, 320 * upper], [0.9 * lower, 0.9 * upper], [50 * lower, 50 * upper], [15 * lower, 15 * upper],
        [6 * lower, 6 * upper], [50 * lower, 50 * upper],

        # further added params
        [4.9 * lower, 4.9 * upper], [0.3 * lower, 0.3 * upper], [26.6 * lower, 26.6 * upper],
        [0.04 * lower, 0.04 * upper], [80 * lower, 80 * upper],
        [0.1 * 0.8, 0.1 * 1.2], [0.3 * 0.8, 0.3 * 1.2], [0.45 * 0.85, 0.45 * 1.15], [0.93 * 0.95, 0.93 * 1.05],
        [0.0873 * lower, 0.0873 * upper], [1.2 * 0.85, 1.2 * 1.15], [1.2 * 0.85, 1.2 * 1.15], [181 * lower, 181 * upper], [31 * lower, 31 * upper]]
})

    original_bounds = [b[:] for b in sp["bounds"]]


    def build_full_parameters_from_subset(subset_values, sp, subset_vars, precision=12):
        """
        Build full Parameters dict from a subset array.

        - Parameters in subset_vars are taken from subset_values
        - All other parameters are set to their nominal value
          = midpoint of their bounds

        Parameters
        ----------
        subset_values : array-like
            Values for the retained subset parameters, in the same order as subset_vars.
        sp : ProblemSpec-like dict
            Must contain:
                sp["names"]  : list of parameter names
                sp["bounds"] : list of [lower, upper] bounds
        subset_vars : list[str]
            Retained parameter names in the correct order.
        precision : int
            Decimal rounding precision.

        Returns
        -------
        dict
            Full parameter dictionary.
        """
        subset_values = np.asarray(subset_values, dtype=float)

        if len(subset_values) != len(subset_vars):
            raise ValueError(
                f"Length mismatch: got {len(subset_values)} subset values "
                f"but {len(subset_vars)} subset parameter names."
            )

        # map retained parameter name -> sampled value
        subset_map = {
            name: round(float(val), precision)
            for name, val in zip(subset_vars, subset_values)
        }

        Parameters = {}
        for name, (lo, hi) in zip(sp["names"], sp["bounds"]):
            if name in subset_map:
                Parameters[name] = subset_map[name]
            else:
                nominal = round(0.5 * (float(lo) + float(hi)), precision)
                Parameters[name] = nominal

        return Parameters


    def format_parameters_dict(parameters, precision=12):
        rounded_parameters = {
            k: round(float(v), precision)
            for k, v in parameters.items()
        }
        return "Parameters = { " + ", ".join(f'"{k}": {v}' for k, v in rounded_parameters.items()) + " }"


    # ------------------------------------------------------------
    # Your existing subset_vars definition
    # ------------------------------------------------------------
    subset_vars_path = os.path.join(run_dir, "subset_vars.npy")
    subset_vars = np.load(subset_vars_path, allow_pickle=True).tolist()



    # ------------------------------------------------------------
    # load MAP/refined values
    # ------------------------------------------------------------
    subset_values = refined_best["theta"]

    if len(subset_vars) != len(subset_values):
        raise ValueError(
            f"Length mismatch: {len(subset_vars)=}, {len(subset_values)=}"
        )

    subset_map = dict(zip(subset_vars, subset_values))

    # ------------------------------------------------------------
    # update bounds using MAP as new nominal, preserving shape
    # ------------------------------------------------------------
    new_bounds = []
    for name, (lo, hi), (old_lo, old_hi) in zip(sp["names"], sp["bounds"], original_bounds):
        if name in subset_map:
            nominal = float(subset_map[name])
            old_mid = 0.5 * (old_lo + old_hi)
            lo_factor = old_lo / old_mid
            hi_factor = old_hi / old_mid

            new_lo = nominal * lo_factor
            new_hi = nominal * hi_factor

            if new_lo > new_hi:
                new_lo, new_hi = new_hi, new_lo

            new_bounds.append([new_lo, new_hi])
        else:
            new_bounds.append([old_lo, old_hi])

    sp["bounds"] = new_bounds

    # ------------------------------------------------------------
    # formatting helpers
    # ------------------------------------------------------------
    def fmt(x, dp=12):
        s = f"{x:.{dp}f}".rstrip("0").rstrip(".")
        return "0" if s == "-0" else s

    def approx_equal(a, b, tol=1e-10):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    def recover_pattern(old_lo, old_hi):
        """
        Express original bounds as:
          old_lo = base * m1
          old_hi = base * m2
        returning (base, m1_string, m2_string)
        """
        candidates = []

        # try midpoint-based base first
        mid = 0.5 * (old_lo + old_hi)
        if abs(mid) > 1e-14:
            m1 = old_lo / mid
            m2 = old_hi / mid
            candidates.append((mid, m1, m2))

        # try lower/0.8 or upper/1.2 style
        for mult in [0.8, 1.2, 0.85, 1.15, 0.9, 1.05, 0.92, 1.08]:
            if abs(mult) > 1e-14:
                candidates.append((old_lo / mult, mult, old_hi / (old_lo / mult) if abs(old_lo / mult) > 1e-14 else None))
                candidates.append((old_hi / mult, old_lo / (old_hi / mult) if abs(old_hi / mult) > 1e-14 else None, mult))

        # choose a candidate where both multipliers look like your preferred constants
        allowed = [lower, upper, 0.8, 1.2, 0.85, 1.15, 0.9, 1.05, 0.92, 1.08]

        def mult_to_str(m):
            if m is None:
                return None
            for a, s in [
                (lower, "lower"),
                (upper, "upper"),
                (0.8, "0.8"),
                (1.2, "1.2"),
                (0.85, "0.85"),
                (1.15, "1.15"),
                (0.9, "0.9"),
                (1.05, "1.05"),
                (0.92, "0.92"),
                (1.08, "1.08"),
            ]:
                if approx_equal(m, a):
                    return s
            return fmt(m)

        best = None
        for base, m1, m2 in candidates:
            if m1 is None or m2 is None or abs(base) < 1e-14:
                continue
            if approx_equal(base * m1, old_lo) and approx_equal(base * m2, old_hi):
                s1 = mult_to_str(m1)
                s2 = mult_to_str(m2)
                score = int(s1 in {"lower", "upper", "0.8", "1.2", "0.85", "1.15", "0.9", "1.05", "0.92", "1.08"}) \
                      + int(s2 in {"lower", "upper", "0.8", "1.2", "0.85", "1.15", "0.9", "1.05", "0.92", "1.08"})
                cand = (score, abs(base), base, s1, s2)
                if best is None or cand[0] > best[0]:
                    best = cand

        if best is None:
            # fallback
            base = mid
            m1 = old_lo / base
            m2 = old_hi / base
            return base, fmt(m1), fmt(m2)

        _, _, base, s1, s2 = best
        return base, s1, s2

    parameters = {}
    for name, (old_lo, old_hi) in zip(sp["names"], original_bounds):
        base0, _, _ = recover_pattern(old_lo, old_hi)

        if name in subset_map:
            base = float(subset_map[name])
        else:
            base = base0

        parameters[name] = base

    refined_param_npy = os.path.join(run_dir, "refined_map_parameters.npy")
    refined_param_json = os.path.join(run_dir, "refined_map_parameters.json")
    refined_param_vector = os.path.join(run_dir, "refined_map_parameter_vector.npy")

    np.save(refined_param_npy, parameters)
    np.save(
        refined_param_vector,
        np.asarray([float(parameters[name]) for name in sp["names"]], dtype=np.float64),
    )
    with open(refined_param_json, "w") as f:
        json.dump({k: float(v) for k, v in parameters.items()}, f, indent=2)

    print(format_parameters_dict(parameters))
    print(f"Saved: {refined_param_npy}")
    print(f"Saved: {refined_param_json}")
    print(f"Saved: {refined_param_vector}")

if __name__ == "__main__":
    main()
