import math
import warnings
from collections import OrderedDict

from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from scipy.stats import spearmanr
import seaborn as sns

# from SALib.analyze import dgsm
import dgsm_edited as dgsm
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings(
    "ignore",
    message=".*use_inf_as_na option is deprecated.*",
    category=FutureWarning,
)

percentage = 50 # change
lower = 1 - percentage/100
upper = 1 + percentage/100

X = np.load('DGSM_500_X_union_50_27_08.npy')[:35490,:] # change
Result = np.load(r'Y:\project\DGSM_Union_Paper_Longer_Delay\Result_DGSM_union.npy')[:35490,:] # change

D = X.shape[1]
block_size = D + 1
n_blocks = X.shape[0] // block_size
# Find basepoint indices (first row of each block)
base_idx = np.arange(0, X.shape[0], block_size)
# Mask: True if basepoint result != 0
mask_blocks = Result[base_idx, 0] != 0   # check column 0 (e.g. HR); adjust if needed
# OR: drop block if *any* nan appears in that block
mask_blocks_nan = np.array([
    np.all(np.isfinite(Result[i:i+block_size]))   # True if block has no nan
    for i in base_idx
])

# Compute variability (std) within each block
block_std = np.zeros((n_blocks, Result.shape[1]))

for b, i in enumerate(base_idx):
    block = Result[i:i + block_size]
    block_std[b] = np.nanstd(block, axis=0)

for b, i in enumerate(base_idx):
    print(
        f"Block {b:4d} | std = {block_std[b, 3]:.4g}"
        # f"STD: {block_std[b]}"
    )

# Threshold = mean + 3*std of block stds, computed across blocks for each output
std_mean = np.nanmean(block_std, axis=0)
std_std  = np.nanstd(block_std, axis=0)
std_thresh = std_mean + 3 * std_std

# Keep blocks only if ALL output stds are below their respective thresholds
mask_blocks_std = np.all(block_std <= std_thresh, axis=1)

# Keep only blocks where all perturbed HR values are within 0.03 of the base HR (convergence check)
HR_col = 0
mask_blocks_conv = np.array([
    np.all(np.abs(Result[i + 1:i + block_size, HR_col] - Result[i, HR_col]) < 0.03)
    for i in base_idx
])

max_HR_col = 32
mask_blocks_max = np.array([
    np.all(np.abs(Result[i + 1:i + block_size, max_HR_col] - Result[i, max_HR_col]) == 0.0)
    for i in base_idx
])

mask_blocks_max = ~mask_blocks_max

tidal_col = 25
mask_blocks_conv_tidal = np.array([
    np.all(np.abs(Result[i + 1:i + block_size, tidal_col] - Result[i, tidal_col]) < 0.03)
    for i in base_idx
])

mask_blocks = mask_blocks & mask_blocks_nan & mask_blocks_conv & mask_blocks_std & mask_blocks_conv_tidal & mask_blocks_max# & mask_blocks_E_rs & mask_blocks_R_rs # & mask_blocks_std
print(np.count_nonzero(mask_blocks))
# Expand mask to all rows in a block
mask_full = np.repeat(mask_blocks, block_size)


# Filter arrays
X = X[mask_full]
Result = Result[mask_full]

HR = Result[:, 17]

# new +/-50%,90% does not have dc
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
            [21.9 * 0.8, 21.9 * 1.2], [3.02 * 0.8, 3.02 * 1.2],
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
            [30 * lower, 30 * upper], [2.076 * lower, 2.076 * upper], [0.8 * lower, 0.8 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper], [20 * lower, 20 * upper],
            [10 * lower, 10 * upper], [5 * lower, 5 * upper], [40 * lower, 40 * upper], [10 * lower, 10 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
            [5 * lower, 5 * upper], [5 * lower, 5 * upper], [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper],
            [4 * lower, 4 * upper], [0.3 * lower, 0.3 * upper], [0.014 * lower, 0.014 * upper], [0.011 * lower, 0.011 * upper],
            [0.1 * lower, 0.1 * upper], [0.2 * lower, 0.2 * upper], [3 * lower, 3 * upper], [2.5 * lower, 2.5 * upper],
            [20 * lower, 20 * upper], [0.01 * lower, 0.01 * upper], [50 * lower, 50 * upper], [0.25 * lower, 0.25 * upper],
            [0.25 * lower, 0.25 * upper], [50 * lower, 50 * upper],

            # further added params
            [4.9 * lower, 4.9 * upper], [0.3 * lower, 0.3 * upper], [26.6 * lower, 26.6 * upper],
            [0.04 * lower, 0.04 * upper], [80 * lower, 80 * upper],
            [0.1 * 0.8, 0.1 * 1.2], [0.3 * 0.8, 0.3 * 1.2], [0.45 * 0.85, 0.45 * 1.15], [0.93 * 0.95, 0.93 * 1.05],
            [0.0873 * lower, 0.0873 * upper], [1.2 * 0.85, 1.2 * 1.15], [1.2 * 0.85, 1.2 * 1.15], [181 * lower, 181 * upper],
            [31 * lower, 31 * upper]]
        })

output_names = [
    "Heart Rate", "LV Systolic Pressure", "LV Diastolic Pressure", "LV EDV", "LV ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Min RA Pressure A descent", "Max RA Pressure A Wave",
    "Max RA Pressure V Wave", "Min RA Pressure V descent",
    "Min LA Volume", "Max LA Volume", "Min LA Pressure A descent", "Max LA Pressure A Wave",
    "Max LA Pressure V Wave", "Min LA Pressure V descent",
    "LA Pre-Atrial Contraction Volume", "RA Pre-Atrial Contraction Volume", "Max LV Pressure Deriv", "Max RV Pressure Deriv", "Tidal Volume", "Minute Ventilation",
    "Cardiac Output", "PaO2", "PaCO2", "Percentage Volume Change", "Pericardial Pressure"]

output_names_reduced = [
    "Heart Rate", "LV Systolic Pressure", "LV Diastolic Pressure", "LV EDV", "LV ESV",
    "Max RV Volume", "Min RV Volume", "Max RV Pressure", "Min RV Pressure",
    "Min RA Volume", "Max RA Volume", "Max RA Pressure A Wave",
    "Max RA Pressure V Wave",
    "Min LA Volume", "Max LA Volume", "Max LA Pressure A Wave",
    "Max LA Pressure V Wave",
    "LA Pre-Atrial Contraction Volume", "RA Pre-Atrial Contraction Volume", "Max LV Pressure Deriv", "Max RV Pressure Deriv", "Tidal Volume", "Minute Ventilation",
    "PaO2", "PaCO2"]

output_names = (["Rest " + name for name in output_names] + ["Exercise " + name for name in output_names])

output_names_reduced = (["Rest " + name for name in output_names_reduced] + ["Exercise " + name for name in output_names_reduced])



dgsm_summary = OrderedDict()

# min_frac = 0.003  # 0.3%
min_frac = 0.02
# min_frac = 0.0

for col in range(Result.shape[1]):

    Y = Result[:, col]
    output_label = output_names[col]

    # Skip invalid outputs
    if not np.all(np.isfinite(Y)):
        print(f"Skipping {output_label} (non-finite values)")
        continue

    # DGSM analysis
    Si = dgsm.analyze(sp, X, Y, print_to_console=False)

    dgsm_vals = np.array(Si["dgsm"], dtype=float)
    names = np.array(Si["names"])

    # Sort descending
    order = np.argsort(dgsm_vals)[::-1]
    dgsm_sorted = dgsm_vals[order]
    names_sorted = names[order]

    total = dgsm_sorted.sum()
    if total <= 0 or not np.isfinite(total):
        print(f"Skipping {output_label} (non-positive/invalid DGSM total: {total})")
        continue

    # --- NEW: keep only params contributing >= 0.3% of total ---
    thresh = min_frac * total
    keep = dgsm_sorted >= thresh
    dgsm_kept = dgsm_sorted[keep]
    names_kept = names_sorted[keep]

    # Cumulative sensitivity over kept params, but target is 90% of ORIGINAL total
    cumu = np.cumsum(dgsm_kept)

    target = 0.9 * total
    if cumu.size == 0:
        idx_90 = 0
        top_names_90 = np.array([])
        top_dgsm_90 = np.array([])
        reached = 0.0
    else:
        idx_90 = np.searchsorted(cumu, target) + 1
        # if you can't reach 90% without including <0.3% params, cap at all kept
        idx_90 = min(idx_90, len(cumu))
        top_names_90 = names_kept[:idx_90]
        top_dgsm_90 = dgsm_kept[:idx_90]
        reached = cumu[idx_90 - 1] / total

    dgsm_summary[output_label] = {
        "n_params_90": idx_90,
        "param_names": top_names_90,
        "dgsm_values": top_dgsm_90,
        "min_frac": min_frac,
        "fraction_of_total_reached": reached,
        "n_params_passing_0p3pct": int(keep.sum()),
    }

    # ---- Console output ----
    print("\n" + "=" * 80)
    print(f"Output: {output_label}")
    print(f"Min per-parameter contribution: {min_frac*100:.1f}% (DGSM >= {thresh:.4e})")
    print(f"Parameters selected (up to 90% total, with cutoff): {idx_90}")
    print(f"Fraction of total DGSM reached: {reached*100:.2f}%")
    if reached < 0.90:
        print("NOTE: Could not reach 90% without including parameters < 1.0% each.")

    print("-" * 80)
    for name, val in zip(top_names_90, top_dgsm_90):
        print(f"{name:25s} : {val:.4e}  ({val/total*100:.3f}%)")


from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt

dgsm_by_output = {}
for j, out_name in enumerate(output_names):
    Y = Result[:, j]
    Si = dgsm.analyze(sp, X, Y, print_to_console=False)
    dgsm_by_output[out_name] = {
        "dgsm": np.asarray(Si["dgsm"], dtype=float),
        "conf": np.asarray(Si["dgsm_conf"], dtype=float),
        "names": np.asarray(Si["names"]),
    }

coverage = 0.9
n_cols = 4
plot_output_names = output_names_reduced
n_out = len(plot_output_names)
n_rows = math.ceil(n_out / n_cols)

# Match the pastel publication style used by the DGSM convergence and
# target-overlap figures.  The palette cycles by output panel while each
# panel retains the original single-series horizontal bar chart.
panel_colors = ("#CDB8E5", "#AFC8E8", "#8FCAD4", "#E89A9A")
line_dark = "#2F4858"
grid_color = "#E5E7EB"

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.8 * n_cols, 3.6 * n_rows),
    constrained_layout=True,
    facecolor="white",
)
axes = np.atleast_1d(axes).ravel()

dgsm_summary = OrderedDict()

for plot_index, (ax, out_name) in enumerate(zip(axes, plot_output_names)):
    dg = dgsm_by_output[out_name]["dgsm"]
    cf = dgsm_by_output[out_name]["conf"]
    pn = dgsm_by_output[out_name]["names"]

    # sort descending
    order = np.argsort(dg)[::-1]
    dg_sorted = dg[order]
    cf_sorted = cf[order]
    pn_sorted = pn[order]

    # keep only finite, positive DGSM
    good = np.isfinite(dg_sorted) & (dg_sorted > 0) & np.isfinite(cf_sorted)
    dg_sorted = dg_sorted[good]
    cf_sorted = cf_sorted[good]
    pn_sorted = pn_sorted[good]

    total = np.sum(dg_sorted)
    if total <= 0 or not np.isfinite(total):
        ax.text(0.5, 0.5, "No finite DGSM", ha="center", va="center")
        ax.set_title(out_name, fontsize=10)
        ax.axis("off")
        continue

    # enforce minimum 1% of ORIGINAL total
    thresh = min_frac * total
    keep = dg_sorted >= thresh

    dg_kept = dg_sorted[keep]
    cf_kept = cf_sorted[keep]
    pn_kept = pn_sorted[keep]

    if dg_kept.size == 0:
        ax.text(0.5, 0.5, "No params >= 1%", ha="center", va="center")
        ax.set_title(out_name, fontsize=10)
        ax.axis("off")
        dgsm_summary[out_name] = {
            "n_params_90": 0,
            "param_names": np.array([]),
            "dgsm_values": np.array([]),
            "conf_values": np.array([]),
            "fraction_of_total_reached": 0.0,
            "n_params_passing_cutoff": 0,
            "min_frac": min_frac,
        }
        continue

    # cumulative DGSM of kept params, but relative to ORIGINAL total
    cumu = np.cumsum(dg_kept)
    target = coverage * total

    idx_90 = np.searchsorted(cumu, target) + 1
    idx_90 = min(idx_90, len(cumu))   # cap if 90% cannot be reached

    dg_keep = dg_kept[:idx_90]
    cf_keep = cf_kept[:idx_90]
    pn_keep = pn_kept[:idx_90]

    reached = cumu[idx_90 - 1] / total
    note = ""
    if reached < coverage:
        note = f"\nReached {reached*100:.1f}% only"

    # plot (reverse so biggest is at top)
    y = np.arange(len(dg_keep))
    ax.barh(
        y,
        dg_keep[::-1],
        xerr=cf_keep[::-1],
        color=panel_colors[plot_index % len(panel_colors)],
        edgecolor="none",
        error_kw={
            "ecolor": line_dark,
            "elinewidth": 0.9,
            "capsize": 2,
            "capthick": 0.9,
        },
    )
    ax.set_yticks(y)
    ax.set_yticklabels(pn_keep[::-1], fontsize=9)
    ax.set_title(
        f"{out_name}\n{coverage*100:.0f}% DGSM, params >= {min_frac*100:.0f}% ({len(dg_keep)} params){note}",
        fontsize=10
    )
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=grid_color, linewidth=0.8)
    ax.set_axisbelow(True)

    dgsm_summary[out_name] = {
        "n_params_90": len(dg_keep),
        "param_names": pn_keep,
        "dgsm_values": dg_keep,
        "conf_values": cf_keep,
        "fraction_of_total_reached": reached,
        "n_params_passing_cutoff": int(keep.sum()),
        "min_frac": min_frac,
    }

# turn off unused subplot axes
for ax in axes[n_out:]:
    ax.axis("off")

# x-label on bottom row and selected additional outputs
extra_xlabel_outputs = {"Tidal Volume", "Minute Ventilation", "PaO2"}
for r in range(n_rows):
    for c in range(n_cols):
        k_ax = r * n_cols + c
        if k_ax < n_out and (r == n_rows - 1 or plot_output_names[k_ax] in extra_xlabel_outputs):
            axes[k_ax].set_xlabel("DGSM")

# plt.show()
plt.savefig(f"DGSM_{percentage}_reduced_tidal.png", dpi=300, bbox_inches="tight")
plt.close()











Si = dgsm.analyze(sp, X, HR, print_to_console=True)

## Extract and sort
dgsm1 = np.array(Si['dgsm'])
names = np.array(Si['names'])
conf = np.array(Si['dgsm_conf'])

dgsm_sorted = np.argsort(dgsm1)[::-1]  # descending order
top_dgsm = dgsm1[dgsm_sorted]
top_names = names[dgsm_sorted]
top_conf = conf[dgsm_sorted]

# Split the sorted arrays in half
mid = len(top_dgsm) // 2
dgsm_1, dgsm_2 = top_dgsm[:mid], top_dgsm[mid:]
names_1, names_2 = top_names[:mid], top_names[mid:]
conf_1, conf_2 = top_conf[:mid], top_conf[mid:]

# Plot first half
plt.figure(figsize=(10, 8))
plt.bar(names_1, dgsm_1, yerr=conf_1)
plt.xlabel("Sensitivity Index (DGSM)")
plt.title("HR DGSM")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plot second half
plt.figure(figsize=(10, 8))
plt.bar(names_2, dgsm_2, yerr=conf_2)
plt.xlabel("Sensitivity Index (DGSM)")
plt.title("HR DGSM")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Calculate cumulative sum and total sum
cumusum = np.cumsum(top_dgsm)
total = cumusum[-1]

# Find the index where cumulative sum reaches 90% of total
threshold_index = np.searchsorted(cumusum, 0.90 * total) + 1  # +1 to include that index

# Get variables contributing to 90% of sensitivity
vars_90 = top_names[:threshold_index]
sens_90 = top_dgsm[:threshold_index]

print(f"Number of variables contributing 90% sensitivity: {threshold_index}")
# print("Variables:")
# for var, sens in zip(vars_90, sens_90):
#     print(f"{var}: {sens}")

# Optional: Plot these variables only
plt.figure(figsize=(10, 6))
plt.bar(vars_90, sens_90)
plt.xlabel("Parameters")
plt.ylabel("DGSM Sensitivity")
plt.title("Parameters contributing 90% of DGSM Sensitivity")
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# ranking convergence compared to final basepoint
def rank_stability_final(problem, X, Y, max_blocks, step=1, top_k=threshold_index):
    """
    Compute DGSM rankings as number of base points increases,
    comparing each to the final ranking.

    Parameters
    ----------
    problem : dict
        SALib problem definition
    X, Y : arrays
        Inputs and outputs
    max_blocks : int
        Maximum number of base points (blocks)
    step : int
        Increment in blocks
    top_k : int
        Number of top parameters to track for overlap stability
    """
    D = problem["num_vars"]
    block_sizes = range(step, max_blocks + 1, step)
    rankings = []

    # First, compute rankings for all block sizes
    for nb in block_sizes:
        N = (D + 1) * nb
        Si = dgsm.analyze(problem, X[:N, :], Y[:N])
        dgsm_vals = np.array(Si['dgsm'])
        rank = np.argsort(dgsm_vals)[::-1][:top_k]
        rankings.append(rank)

    # Use final ranking as reference
    final_rank = rankings[-1]

    corrs = []
    overlaps = []

    for rank in rankings:
        # Spearman correlation with final ranking
        corr, _ = spearmanr(rank, final_rank)
        corrs.append(corr)

        # Top-k overlap
        overlap = len(set(rank) & set(final_rank)) / top_k
        overlaps.append(overlap)

    return block_sizes, corrs, overlaps, rankings


# Example usage
max_blocks = int(len(HR)/block_size)
block_sizes, corrs, overlaps, rankings = rank_stability_final(sp, X, HR, max_blocks=max_blocks, step=10)

# Plot both metrics
plt.figure(figsize=(8,5))
plt.plot(block_sizes, corrs, marker="o", label="Spearman correlation")
plt.plot(block_sizes, overlaps, marker="s", label="Top-k overlap")
plt.xlabel("Number of base points (blocks)")
plt.ylabel("Stability metric")
plt.title("DGSM Rankings with Increasing Base Points Compared to All Samples")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
