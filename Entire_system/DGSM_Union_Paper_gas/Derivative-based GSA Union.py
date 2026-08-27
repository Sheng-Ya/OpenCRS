from pathlib import Path

import numpy as np
from SALib import ProblemSpec

import dgsm_edited as dgsm
from plots.plot_dgsm_convergence_summary_from_cache import (
    compute_convergence_cache,
    save_convergence_summary_from_cache,
)
from plots.plot_dgsm_reduced_tidal_from_cache import save_condition_figures


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PERCENTAGE = 50
COVERAGE = 0.90
MIN_FRAC = 0.02
JACCARD_TOL = 0.95

X_FILENAME = "DGSM_500_X_union_50_27_08_gas.npy"
RESULT_FILENAME = "DGSM_500_Result_union_50_27_08_gas.npy"

OUTPUT_NAMES_BASE = [
    "Heart Rate", "LV Systolic Pressure", "LV Diastolic Pressure", "LV EDV", "LV ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Min RA Pressure A descent", "Max RA Pressure A Wave",
    "Max RA Pressure V Wave", "Min RA Pressure V descent",
    "Min LA Volume", "Max LA Volume", "Min LA Pressure A descent", "Max LA Pressure A Wave",
    "Max LA Pressure V Wave", "Min LA Pressure V descent",
    "LA Pre-Atrial Contraction Volume", "RA Pre-Atrial Contraction Volume",
    "Max LV Pressure Deriv", "Max RV Pressure Deriv", "Tidal Volume",
    "Minute Ventilation", "Cardiac Output", "PaO2", "PaCO2",
    "Percentage Volume Change", "Pericardial Pressure",
]

TARGET_NAMES_BASE = [
    "Heart Rate", "LV Systolic Pressure", "LV Diastolic Pressure", "LV EDV", "LV ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Max RA Pressure A Wave",
    "Max RA Pressure V Wave", "Min LA Volume", "Max LA Volume",
    "Max LA Pressure A Wave", "Max LA Pressure V Wave",
    "LA Pre-Atrial Contraction Volume", "RA Pre-Atrial Contraction Volume",
    "Max LV Pressure Deriv", "Max RV Pressure Deriv", "Tidal Volume",
    "Minute Ventilation", "PaO2", "PaCO2",
]


def build_problem_spec(lower: float, upper: float) -> ProblemSpec:
    return ProblemSpec({
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
            "R_rmv_n", "R_sv_n", "K1_vc","D1",
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


def filter_dgsm_blocks(
    x: np.ndarray,
    result: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the simulation-quality filters used by the DGSM figures."""
    if x.ndim != 2 or result.ndim != 2:
        raise ValueError("X and Result must both be two-dimensional arrays.")
    if len(x) != len(result):
        raise ValueError("X and Result must contain the same number of rows.")

    block_size = x.shape[1] + 1
    n_blocks, remainder = divmod(len(x), block_size)
    if remainder:
        raise ValueError(
            f"X contains {len(x)} rows, which is not divisible by block size "
            f"{block_size}."
        )

    result_blocks = result.reshape(n_blocks, block_size, result.shape[1])
    base_values = result_blocks[:, :1, :]
    perturbed_values = result_blocks[:, 1:, :]

    finite_blocks = np.all(np.isfinite(result_blocks), axis=(1, 2))
    nonzero_rest_hr = result_blocks[:, 0, 0] != 0

    block_std = np.nanstd(result_blocks, axis=1)
    std_threshold = np.nanmean(block_std, axis=0) + 3 * np.nanstd(
        block_std, axis=0
    )
    stable_variability = np.all(block_std <= std_threshold, axis=1)

    converged_rest_hr = np.all(
        np.abs(perturbed_values[:, :, 0] - base_values[:, :, 0]) < 0.03,
        axis=1,
    )
    varying_exercise_hr = np.any(
        np.abs(perturbed_values[:, :, 32] - base_values[:, :, 32]) != 0.0,
        axis=1,
    )
    converged_rest_tidal_volume = np.all(
        np.abs(perturbed_values[:, :, 25] - base_values[:, :, 25]) < 0.03,
        axis=1,
    )

    keep_blocks = (
        finite_blocks
        & nonzero_rest_hr
        & stable_variability
        & converged_rest_hr
        & varying_exercise_hr
        & converged_rest_tidal_volume
    )
    row_mask = np.repeat(keep_blocks, block_size)
    return x[row_mask], result[row_mask], block_size


def format_text_block(
    output_label: str,
    dgsm_values: np.ndarray,
    parameter_names: np.ndarray,
) -> str:
    order = np.argsort(dgsm_values)[::-1]
    sorted_values = dgsm_values[order]
    sorted_names = parameter_names[order]

    total = sorted_values.sum()
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError(
            f"{output_label} has a non-positive or invalid DGSM total: {total}"
        )

    threshold = MIN_FRAC * total
    keep = sorted_values >= threshold
    kept_values = sorted_values[keep]
    kept_names = sorted_names[keep]
    cumulative = np.cumsum(kept_values)

    if cumulative.size == 0:
        n_selected = 0
        selected_values = np.array([], dtype=float)
        selected_names = np.array([], dtype=str)
        reached = 0.0
    else:
        n_selected = min(
            int(np.searchsorted(cumulative, COVERAGE * total) + 1),
            len(cumulative),
        )
        selected_values = kept_values[:n_selected]
        selected_names = kept_names[:n_selected]
        reached = cumulative[n_selected - 1] / total

    lines = [
        "=" * 80,
        f"Output: {output_label}",
        (
            f"Min per-parameter contribution: {MIN_FRAC*100:.1f}% "
            f"(DGSM >= {threshold:.4e})"
        ),
        f"Parameters selected (up to 90% total, with cutoff): {n_selected}",
        f"Fraction of total DGSM reached: {reached*100:.2f}%",
    ]
    if reached < COVERAGE:
        lines.append(
            "NOTE: Could not reach 90% without including parameters < 1.0% each."
        )

    lines.append("-" * 80)
    lines.extend(
        f"{name:25s} : {value:.4e}  ({value/total*100:.3f}%)"
        for name, value in zip(selected_names, selected_values)
    )
    return "\n".join(lines)


def main() -> None:
    lower = 1 - PERCENTAGE / 100
    upper = 1 + PERCENTAGE / 100
    problem = build_problem_spec(lower, upper)

    x = np.load(BASE_DIR / X_FILENAME)
    result = np.load(BASE_DIR / RESULT_FILENAME)
    # Compute variability (std) within each block
    D = x.shape[1]
    block_size = D + 1
    n_blocks = x.shape[0] // block_size
    base_idx = np.arange(0, x.shape[0], block_size)
    block_std = np.zeros((n_blocks, result.shape[1]))

    for b, i in enumerate(base_idx):
        block = result[i:i + block_size]
        block_std[b] = np.nanstd(block, axis=0)
        print(f"Block {b:4d} | std = {block_std[b, 3]:.4g}")


    x, result, block_size = filter_dgsm_blocks(x, result)
    print(int(result.shape[0]/block_size))

    output_names = (
        [f"Rest {name}" for name in OUTPUT_NAMES_BASE]
        + [f"Exercise {name}" for name in OUTPUT_NAMES_BASE]
    )
    target_names = (
        [f"Rest {name}" for name in TARGET_NAMES_BASE]
        + [f"Exercise {name}" for name in TARGET_NAMES_BASE]
    )


    target_indices = np.asarray(
        [output_names.index(name) for name in target_names],
        dtype=int,
    )
    analysis_by_output: dict[str, dict[str, np.ndarray]] = {}
    text_blocks = {"Rest": [], "Exercise": []}
    parameter_names: np.ndarray | None = None

    for result_column, output_label in zip(target_indices, target_names):
        values = result[:, result_column]

        sensitivity = dgsm.analyze(
            problem,
            x,
            values,
            print_to_console=False,
        )
        current_names = np.asarray(sensitivity["names"], dtype=str)
        if parameter_names is None:
            parameter_names = current_names
        elif not np.array_equal(current_names, parameter_names):
            raise RuntimeError(
                f"DGSM parameter names for {output_label} do not match "
                "the other outputs."
            )

        dgsm_values = np.asarray(sensitivity["dgsm"], dtype=float)
        analysis_by_output[output_label] = {
            "dgsm": dgsm_values,
            "conf": np.asarray(sensitivity["dgsm_conf"], dtype=float),
        }
        condition = output_label.split(" ", 1)[0]
        text_blocks[condition].append(
            format_text_block(output_label, dgsm_values, current_names)
        )

    if parameter_names is None:
        raise RuntimeError("No DGSM targets were analysed.")

    for condition, blocks in text_blocks.items():
        output_path = BASE_DIR / f"DGSM_Union_{condition}.txt"
        output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    cache_output_names = np.asarray(target_names, dtype=str)
    dgsm_matrix = np.stack(
        [analysis_by_output[name]["dgsm"] for name in target_names]
    )
    conf_matrix = np.stack(
        [analysis_by_output[name]["conf"] for name in target_names]
    )
    plot_cache_path = BASE_DIR / (
        f"DGSM_{PERCENTAGE}_reduced_tidal_cache.npz"
    )
    np.savez_compressed(
        plot_cache_path,
        output_names=cache_output_names,
        parameter_names=parameter_names,
        dgsm_values=dgsm_matrix,
        conf_values=conf_matrix,
        percentage=np.asarray(PERCENTAGE),
        source_x=np.asarray(X_FILENAME),
        source_result=np.asarray(RESULT_FILENAME),
    )

    save_condition_figures(
        cache_path=plot_cache_path,
        output_dir=PLOTS_DIR,
        percentage=PERCENTAGE,
        coverage=COVERAGE,
        min_frac=MIN_FRAC,
        dpi=300,
    )

    convergence_cache_path = PLOTS_DIR / "convergence_metrics.npz"
    compute_convergence_cache(
        problem=problem,
        x=x,
        result=result,
        output_names=cache_output_names,
        output_indices=target_indices,
        parameter_names=parameter_names,
        block_size=block_size,
        analyze_fn=dgsm.analyze,
        cache_path=convergence_cache_path,
        final_dgsm_values=dgsm_matrix,
        min_blocks=20,
        step=10,
        num_resamples=0,
    )

    convergence_figure_path = (
        PLOTS_DIR / "appendix_convergence_summary.png"
    )
    save_convergence_summary_from_cache(
        cache_path=convergence_cache_path,
        output_path=convergence_figure_path,
        coverage=COVERAGE,
        min_frac=MIN_FRAC,
        jaccard_tol=JACCARD_TOL,
        dpi=250,
    )


if __name__ == "__main__":
    main()
