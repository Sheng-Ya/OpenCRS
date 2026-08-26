import os
import copy
import signal
import numpy as np
from SALib import ProblemSpec
from SALib.sample import finite_diff
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective
import argparse
import time

from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from All_derivatives_njit import make_wrapper
from fixed_params import Parameters as Old_Parameters

from Initial_Conditions_after_running_again import Initial_Conditions
from All_Next_Conditions import make_fresh_storage

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

target_values = np.arange(0, 10000, 10)
BUFFER_LIMIT = 80000

max_time = 60 # Maximum time limit to avoid infinite loops
RAW_OUTPUT_DIM = 32
UNION_OUTPUT_DIM = RAW_OUTPUT_DIM * 2
CONVERGENCE_TOLERANCE = 0.03
MAX_CONVERGENCE_ATTEMPTS = 5
MIN_MEASUREMENT_DURATION = 60

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time, fast_model_derivatives):

    i = Initial_Conditions_dict["i"].item()
    actual_index = i % BUFFER_LIMIT

    all_time = Initial_Conditions_dict["all_time"]

    if i > 1:  # t != 0:
        latest_nonzero_index = (i - 1) % BUFFER_LIMIT
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:
            num_removed = 3
            index = (actual_index - 3) % BUFFER_LIMIT
            for j in range(num_removed):
                all_time[(index + j) % BUFFER_LIMIT] = 0

        else:
            num_removed = 0
    else:
        num_removed = 0

    # Indices for slicing
    idx_resp_contr = num_cardio + num_cardio_control + num_gas + num_resp_control

    # Extract each subsystem's state variables
    resp_contr_state = Initial_Conditions_numpy[:idx_resp_contr]

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    derivatives_all = fast_model_derivatives(t, resp_contr_state, Initial_Conditions_dict, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time)
    all_time[(i - num_removed) % BUFFER_LIMIT] = t
    Initial_Conditions_dict["i"][0] = i - num_removed + 1
    Initial_Conditions_dict["j"][0] = Initial_Conditions_dict["j"].item() - num_removed + 1

    return derivatives_all

# gas exchange
required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                     "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                     "PCSFCO2", "MRTO2", "MRTCO2", "CTO2", "CvtCO2", "CBO2", "CvbCO2", "MRV"]
IC_gas = np.array([Initial_Conditions[key] for key in required_gas_keys], dtype=float)
num_gas = len(required_gas_keys)

# cardiovascular system
required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                           "VT_hv", "VT_rmv", "VT_amv", "P_sp", "P_sa", "Q_sa", "VT_vc",
                         "theta_ao", "dtheta_ao_dt", "theta_po", "dtheta_po_dt", "theta_mi", "dtheta_mi_dt", "theta_tr", "dtheta_tr_dt"]
IC_cardio = np.array([Initial_Conditions[key] for key in required_cardio_keys], dtype=float)
num_cardio = len(required_cardio_keys)

# cardiovascular controller
required_cardio_control_keys = ["theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv",
                         "theta_change_O2_sh", "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", "R_ep_change",
                         "R_sp_change", "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change",
                         "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change", "Emax_rv_change", "Ts_change",
                         "Tv_change", "xb_O2", "xb_CO2", "xh_O2", "xh_CO2", "Wh", "xrm_O2", "xrm_CO2", "xam_O2",
                         "xM", "x_met", "P_n_current"]

IC_cardio_contr = np.array([Initial_Conditions[key] for key in required_cardio_control_keys], dtype=float)
num_cardio_control = len(required_cardio_control_keys)

# resp control ventilation
required_resp_control_keys = ["VE_integral"]
IC_resp_contr = np.array([Initial_Conditions[key] for key in required_resp_control_keys], dtype=float)
num_resp_control = len(required_resp_control_keys)

IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_contr))

def minimise_breathing(t1, t2, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao):
    dt = 0.001 # must edit in Resp_Control_Breath_Optimiser too
    bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
    tolerance = 0.0001

    VAflow_vals = np.linspace(0.04, 1.6, 200)
    # VAflow_repeated = np.repeat(VAflow_vals, 3)

    VD = GV_dead * VAflow_vals + V0_dead

    optimal_t1 = []
    optimal_t2 = []
    initial_guess = [t1, t2]
    required_params = [lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao]

    for idx, VAflow in enumerate(VAflow_vals):
        VD_volume = VD[idx]

        res = minimize(objective, x0= np.array(initial_guess[-2:]),
                       args=(required_params, VAflow, VD_volume, dt, tolerance), method='nelder-mead', bounds=bounds)
        t1_opt, t2_opt = res.x
        optimal_t1.append(t1_opt)
        optimal_t2.append(t2_opt)
        initial_guess.extend(res.x)


    # Convert to arrays for indexing
    # VAflow_clean = np.array(VAflow_vals)
    t1_clean = np.array(optimal_t1)
    t2_clean = np.array(optimal_t2)

    # t1_mean = np.array([np.nanmean(t1_clean[VAflow_clean == v]) for v in VAflow_vals])
    # t2_mean = np.array([np.nanmean(t2_clean[VAflow_clean == v]) for v in VAflow_vals])

    cs_t1 = CubicSpline(VAflow_vals, t1_clean, bc_type="natural")
    cs_t2 = CubicSpline(VAflow_vals, t2_clean, bc_type="natural")

    return cs_t1.c, cs_t2.c, cs_t1.x, cs_t2.x


def zero_state_result():
    return [0.0] * RAW_OUTPUT_DIM


def zero_union_result():
    return [0.0] * UNION_OUTPUT_DIM


def ordered_store(storage, key):
    i_buffer = storage["i"].item() % BUFFER_LIMIT
    return np.concatenate((storage[key][i_buffer:], storage[key][:i_buffer]))


def last_distinct_values(values, n=10):
    values = values[np.isfinite(values)]
    distinct = []
    previous = None
    for value in values[::-1]:
        if previous is None or value != previous:
            distinct.append(float(value))
            previous = value
            if len(distinct) == n:
                break
    return np.asarray(distinct, dtype=float)


def find_valve_edge_indices(valve_state, N):
    cumulative = np.concatenate(([0], np.cumsum(valve_state)))
    previous_window = cumulative[N:-1] - cumulative[:-N - 1]
    return np.flatnonzero(valve_state[N:] & (previous_window == 0)) + N


def has_converged(storage):
    hr_segments = last_distinct_values(ordered_store(storage, "HR_store"))
    return hr_segments.size >= 2 and np.ptp(hr_segments) < CONVERGENCE_TOLERANCE


def is_failed_state_result(result):
    result = np.asarray(result, dtype=float)
    return (
        result.size != RAW_OUTPUT_DIM
        or not np.isfinite(result).all()
        or np.all(result == 0.0)
    )


def copy_storage(storage):
    return {key: value.copy() for key, value in storage.items()}


def simulate_cpu(
    Current_Parameters,
    local_updates,
    old_parameters,
    IC_initial=None,
    state="Rest",
    attempt=0,
    breath_coef=None,
    exercise_start_time=None,
):
    # local_updates = {key: copy.deepcopy(value) for key, value in storage.items()}
    i = local_updates["i"].item()
    latest_nonzero_index = (i - 1) % BUFFER_LIMIT
    latest_nonzero_value = local_updates["all_time"][latest_nonzero_index]

    if IC_initial is None:
        IC_current = IC_overall.copy()
        t_span = [0, max_time]
        exercise_start_time = np.inf
    elif state == "Exercise" and attempt == 0:
        IC_current = IC_initial.copy()
        t_span = [latest_nonzero_value, latest_nonzero_value + 180]
        if exercise_start_time is None:
            exercise_start_time = latest_nonzero_value
    else:
        IC_current = IC_initial.copy()
        t_span = [latest_nonzero_value, latest_nonzero_value + 60]
        if state == "Exercise":
            if exercise_start_time is None:
                exercise_start_time = latest_nonzero_value
        else:
            exercise_start_time = np.inf

    # Cardio parameters
    (A_im, T_im, Tc, g_thor, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa,
     R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la,
     Emax_ra, P0_ra, KE_ra, C_sa, L_sa, R_sa, K1_vc, D1, Vvc_min, Kr_vc, Rvc_n,
     C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv,
     kr_am, P_0) = (
    Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
    ["A_im", "T_im", "Tc", "g_thor", "P_thormax_n", "P_thormin_n", "VT_n", "C_pa",
     "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "KE_lv", "KE_rv", "P0_lv", "P0_rv",
     "Emax_la", "P0_la", "KE_la", "Emax_ra", "P0_ra", "KE_ra", "C_sa", "L_sa",
     "R_sa", "K1_vc", "D1", "Vvc_min", "Kr_vc", "Rvc_n", "C_jp",
     "R_ev_n", "R_sv_n", "R_bv_n", "R_hv_n", "R_rmv_n", "R_amv_n", "C_ev", "C_sv", "C_bv", "C_hv", "C_rmv", "C_amv",
     "kr_am", "P_0"])

    # Cardio controller parameters
    (fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv,
    kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh,
    Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0, Emax_rv0, fes_min, GEmax_lv,
    GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv,
    theta_shn, theta_spn, theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, P_n_max,
    f_acCO2_n, f_ac_max, f_ac_min, k_ac, K_H, PaO2_ac_n, G_ap, DT_v, GT_s, GT_v, T0, A, B, C, D,
    Cvb_O2_n, gb_O2, R_bpn, Cvh_O2_n, Cvrm_O2_n, gh_O2, grm_O2, Kh_CO2, Krm_CO2, MO2_hpn,
    MO2_rmp, R_hpn, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max, phi_min) = \
    [Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
     ["fab_o", "fes_o", "fes_inf", "fes_max", "fev_o",
      "fev_inf", "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max",
      "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp",
      "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
      "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
      "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp", "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh",
      "PO2_sp", "PO2_sv", "theta_shn", "theta_spn", "theta_svn", "x_sh", "x_sp", "x_sv",
      "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "P_n_max", "f_acCO2_n", "f_ac_max", "f_ac_min",
      "k_ac", "K_H", "PaO2_ac_n", "G_ap", "DT_v", "GT_s", "GT_v", "T0", "A", "B", "C", "D",
      "Cvb_O2_n", "gb_O2", "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2",
      "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met",
      "kmet", "MO2_ampn", "phi_max", "phi_min"]]

    # Gas exchange and mixing
    (a2_gas, alpha2, beta2, C2, K2, PACO2_Delay_IC, PAO2_Delay_IC, P_atm,
     P_ws, Z, dc, KCCO2, MRBCO2, MO2_bp, MRTCO2_basal, MRTO2_basal,
     MRCO2, MRO2, s) = (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in [
    "a2", "alpha2", "beta2", "C2", "K2", "PACO2_Delay_IC",
    "PAO2_Delay_IC", "P_atm", "P_ws", "Z", "dc", "KCCO2", "MRBCO2",
    "MO2_bp", "MRTCO2_basal", "MRTO2_basal", "MRCO2", "MRO2", "s"])

    # Resp control
    (GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao) = \
    (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in ["GV_dead", "KcCO2", "KcMRV", "KpCO2", "KpO2",
   "V0_dead", "VA_rest", "lambda1", "lambda2", "n", "Pmax", "Pmax_dot", "E_rs", "R_rs", "P_ao"])

    # added params
    (Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
    Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
    R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
    Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vu_pa, Vu_pp,
    Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
    tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
    tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
    DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
    scale_param1, scale_param3, scale_param4, scale_param6,
    Pa_O2_lower, rise_time_atr, rise_time_ven,
    fall_time_ven, ahead1, theta_min, delta_P, r, l, V_nominal, V_scale
     ) = \
    (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in ["Kp_ao", "Kf_ao", "Kb_ao",
    "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max", "Kp_po", "Kf_po", "Kb_po", "Kv_po",
    "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "alpha_O2", "R_po", "R_mi", "R_tr", "R_ao",
    "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal", "Vu_sa", "V_tot", "Vu_jp",
    "Vu_bv", "Vu_hv", "Vu_vc", "Vu_pa", "Vu_pp", "Vu_pv",
    "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp",
    "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc", "tau_isc",
    "tau_p", "tau_z", "tau_ac", "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2", "tau_O2", "tau_w", "tau_M", "tau_met",
    "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv", "DT_s", "DT_v",
    "Dmet", "Fi_CO2", "Fi_O2", "Ta", "T1", "T2", "VL_CO2", "VL_O2", "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2", "tau_MRV",
    "scale_param1", "scale_param3", "scale_param4", "scale_param6",
    "Pa_O2_lower", "rise_time_atr", "rise_time_ven",
     "fall_time_ven", "ahead1", "theta_min", "delta_P", "r", "l", "V_nominal", "V_scale"])

    # determine the correct breathing profile
    if breath_coef is None:
        cs_t1, cs_t2, knots_1, knots_2 = (minimise_breathing(1.5,1.85, GV_dead, V0_dead, lambda1, lambda2, n, Pmax,
                                                             Pmax_dot, E_rs, R_rs, P_ao))
    else:
        cs_t1, cs_t2, knots_1, knots_2 = breath_coef

    Input_Parameters = np.array([A_im, T_im, Tc, g_thor, P_thormax_n, P_thormin_n, VT_n, C_pa,
    C_pp, C_pv, L_pa, R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la, Emax_ra, P0_ra, KE_ra, C_sa,
    L_sa, R_sa, K1_vc, D1, Vvc_min, Kr_vc, Rvc_n, C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv,
    C_bv, C_hv, C_rmv, C_amv, kr_am, P_0, fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv,
    Io_v, kcc_sh, kcc_sp, kcc_sv, kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v,
    Wb_sh, Wb_sp, Wb_sv, Wc_sh, Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0,
    Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0,
    R_rmp0, R_sp0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv, theta_shn, theta_spn,
    theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n,  P_n_max, f_acCO2_n, f_ac_max, f_ac_min,
    k_ac, K_H, PaO2_ac_n, G_ap, DT_v, GT_s, GT_v, T0, A, B, C, D, Cvb_O2_n, gb_O2, R_bpn, Cvh_O2_n, Cvrm_O2_n, gh_O2,
    grm_O2, Kh_CO2, Krm_CO2, MO2_hpn, MO2_rmp, R_hpn, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max,
    phi_min, a2_gas, alpha2, beta2, C2, K2, PACO2_Delay_IC, PAO2_Delay_IC, P_atm, P_ws, Z, dc, KCCO2, MRBCO2, MO2_bp,
    MRTCO2_basal, MRTO2_basal, MRCO2, MRO2, s, GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2,
    n, Pmax, Pmax_dot, E_rs, R_rs, P_ao,
    # added params
    Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
    Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
    R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
    Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vu_pa, Vu_pp,
    Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
    tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
    tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
    DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
    scale_param1, scale_param3, scale_param4, scale_param6,
     Pa_O2_lower, rise_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P, r, l, V_nominal, V_scale])

    # Bind the storage arrays once instead of looking them up by name during
    # every derivative evaluation.
    fast_model_derivatives = make_wrapper(local_updates)

    # Solve ODE in one go
    ODE_solution = solve_ivp(
        combined_system,
        t_span,
        IC_current,
        max_step=0.001,
        method="RK23",
        rtol=1e-3,
        atol=1e-6,
        args=(local_updates, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time, fast_model_derivatives)
    )


    if ODE_solution.status == -1:
        # Integration failed or early termination
        print("fail")
        print(Input_Parameters[10:15])
        return zero_state_result(), None, None, None

    i_buffer = local_updates["i"].item() % BUFFER_LIMIT

    theta_ao = np.concatenate((local_updates["theta_ao_store"][i_buffer:], local_updates["theta_ao_store"][:i_buffer]))
    theta_po = np.concatenate((local_updates["theta_po_store"][i_buffer:], local_updates["theta_po_store"][:i_buffer]))
    theta_mi = np.concatenate((local_updates["theta_mi_store"][i_buffer:], local_updates["theta_mi_store"][:i_buffer]))
    theta_tr = np.concatenate((local_updates["theta_tr_store"][i_buffer:], local_updates["theta_tr_store"][:i_buffer]))

    V_lv = np.concatenate((local_updates["V_lv_store"][i_buffer:], local_updates["V_lv_store"][:i_buffer]))
    V_rv = np.concatenate((local_updates["V_rv_store"][i_buffer:], local_updates["V_rv_store"][:i_buffer]))
    V_ra = np.concatenate((local_updates["V_ra_store"][i_buffer:], local_updates["V_ra_store"][:i_buffer]))
    V_la = np.concatenate((local_updates["V_la_store"][i_buffer:], local_updates["V_la_store"][:i_buffer]))

    N = 50  # number of consecutive closed samples required

    is_open_ao = theta_ao > theta_min
    open_idx1 = find_valve_edge_indices(is_open_ao, N)

    is_closed_ao = theta_ao <= theta_min
    close_idx1 = find_valve_edge_indices(is_closed_ao, N)

    if open_idx1.size == 0 or close_idx1.size == 0:
        print("ao fail")
        return zero_state_result(), None, None, None

    is_open_po = theta_po > theta_min
    open_idx2 = find_valve_edge_indices(is_open_po, N)

    is_closed_po = theta_po <= theta_min
    close_idx2 = find_valve_edge_indices(is_closed_po, N)

    if open_idx2.size == 0 or close_idx2.size == 0:
        print("po fail")
        return zero_state_result(), None, None, None

    is_open_mi = theta_mi > theta_min
    open_idx3 = find_valve_edge_indices(is_open_mi, N)

    is_closed_mi = theta_mi <= theta_min
    close_idx3 = find_valve_edge_indices(is_closed_mi, N)

    if open_idx3.size == 0 or close_idx3.size == 0:
        print("mi fail")
        return zero_state_result(), None, None, None

    is_open_tr = theta_tr > theta_min
    open_idx4 = find_valve_edge_indices(is_open_tr, N)

    is_closed_tr = theta_tr <= theta_min
    close_idx4 = find_valve_edge_indices(is_closed_tr, N)

    if open_idx4.size == 0 or close_idx4.size == 0:
        print("tr fail")
        return zero_state_result(), None, None, None

    pairs_ao = np.array([
        (o, close_idx1[(close_idx1 > o) & (close_idx1 < o_next)][-1])
        for o, o_next in zip(open_idx1[:-1], open_idx1[1:])
        if np.any((close_idx1 > o) & (close_idx1 < o_next))])

    pairs_po = np.array([
        (o, close_idx2[(close_idx2 > o) & (close_idx2 < o_next)][-1])
        for o, o_next in zip(open_idx2[:-1], open_idx2[1:])
        if np.any((close_idx2 > o) & (close_idx2 < o_next))])

    pairs_mi = np.array([
        (o, close_idx3[(close_idx3 > o) & (close_idx3 < o_next)][-1])
        for o, o_next in zip(open_idx3[:-1], open_idx3[1:])
        if np.any((close_idx3 > o) & (close_idx3 < o_next))])

    pairs_tr = np.array([
        (o, close_idx4[(close_idx4 > o) & (close_idx4 < o_next)][-1])
        for o, o_next in zip(open_idx4[:-1], open_idx4[1:])
        if np.any((close_idx4 > o) & (close_idx4 < o_next))])

    pairs_ao = pairs_ao[-10:]
    pairs_po = pairs_po[-10:]
    pairs_mi = pairs_mi[-10:]
    pairs_tr = pairs_tr[-10:]

    # Max pressure during atrial contraction takes the max p between phi_atr = 0 & 1
    phi_atr = np.concatenate((local_updates["phi_atr_store"][i_buffer:], local_updates["phi_atr_store"][:i_buffer]))
    phi = np.concatenate((local_updates["phi_store"][i_buffer:], local_updates["phi_store"][:i_buffer]))

    dphi = np.diff(phi_atr, prepend=phi_atr[0])
    is_rising = dphi > 0
    edges = np.diff(is_rising.astype(int))
    start_idx = np.where(edges == 1)[0] + 1
    end_idx = np.where(edges == -1)[0] + 1

    n_pairs = min(len(start_idx), len(end_idx))
    # If first end comes before first start, skip that end
    if len(end_idx) > 0 and len(start_idx) > 0 and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]
        n_pairs = min(len(start_idx), len(end_idx))

    # Truncate to matching pairs
    start_idx = start_idx[:n_pairs]
    end_idx = end_idx[:n_pairs]

    # systolic pressure
    P_sa = np.concatenate((local_updates["P_sa_store"][i_buffer:], local_updates["P_sa_store"][:i_buffer]))
    P_sa_max_idx = np.array([o + np.argmax(P_sa[o:c]) for o, c in pairs_ao])

    P_la = np.concatenate((local_updates["P_la_store"][i_buffer:], local_updates["P_la_store"][:i_buffer]))
    # max pressure at atrial contraction
    P_la_max_idx = np.array([s + np.argmax(P_la[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    # period of V descent when mitral valve is open -> get second min la P
    P_la_descent2_idx = np.array([o + np.argmin(P_la[o:c]) for o, c in pairs_mi])
    P_la_descent1_idx = np.array([c + np.argmin(P_la[c:o_next]) for (_, c), (o_next, _) in zip(pairs_mi[:-1], pairs_mi[1:])])


    P_ra = np.concatenate((local_updates["P_ra_store"][i_buffer:], local_updates["P_ra_store"][:i_buffer]))
    # max pressure at atrial contraction
    P_ra_max_idx = np.array([s + np.argmax(P_ra[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    # period of V descent when tricuspid valve is open -> get second min la P
    P_ra_descent2_idx = np.array([o + np.argmin(P_ra[o:c]) for o, c in pairs_tr])
    P_ra_descent1_idx = np.array([c + np.argmin(P_ra[c:o_next]) for (_, c), (o_next, _) in zip(pairs_tr[:-1], pairs_tr[1:])])

    P_rv = np.concatenate((local_updates["P_rv_store"][i_buffer:], local_updates["P_rv_store"][:i_buffer]))
    P_rv_max_idx = np.array([o + np.argmax(P_rv[o:c]) for o, c in pairs_po])
    # RVEDP: last sample before ventricular activation rises in the filling window.
    # RVEDP: last sample before ventricular activation rises each beat.
    phi_eps = 1e-8
    phi_rise_idx = np.where((phi[:-1] <= phi_eps) & (phi[1:] > phi_eps))[0] + 1
    P_rv_edp_idx = phi_rise_idx[-10:] - 1

    time_since_beat_store = np.concatenate((local_updates["time_since_beat_store"][i_buffer:], local_updates["time_since_beat_store"][:i_buffer]))
    cardiac_cycle_start_idx = np.where(np.diff(time_since_beat_store) > 0)[0] + 1
    if cardiac_cycle_start_idx.size < 11:
        print("P_peri cycle fail")
        return zero_state_result(), None, None, None

    P_peri = np.concatenate((local_updates["P_peri_store"][i_buffer:], local_updates["P_peri_store"][:i_buffer]))
    P_peri_cycle_idx = cardiac_cycle_start_idx[-11:]
    P_peri_cycle_max_idx = np.array([b0 + np.argmax(P_peri[b0:b1]) for b0, b1 in zip(P_peri_cycle_idx[:-1], P_peri_cycle_idx[1:])])
    mean_max_P_peri = np.mean(P_peri[P_peri_cycle_max_idx])

    # Get past 10 HR
    HR = np.concatenate((local_updates["HR_store"][i_buffer:], local_updates["HR_store"][:i_buffer]))

    past_10_flat_segments = []
    # Start from the end and track the current segment value
    prev_value = None
    for j in range(len(HR) - 1, -1, -1):
        current_value = HR[j]
        if current_value != prev_value:
            # New segment found
            past_10_flat_segments.append(current_value)
            prev_value = current_value
            if len(past_10_flat_segments) == 10:
                break

    # Find transitions: where phi_atr goes from 0 to >0
    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1
    local_mins = starts[-10:]
    last_10_b4_LA_atrial_contract = V_la[local_mins]
    last_10_b4_RA_atrial_contract = V_ra[local_mins]

    tidal = np.concatenate((local_updates["tidal_store"][i_buffer:], local_updates["tidal_store"][:i_buffer]))
    finish_breath_time = np.concatenate((local_updates["finish_breath_time"][i_buffer:], local_updates["finish_breath_time"][:i_buffer]))
    dtr = np.diff(finish_breath_time)

    breath_starts = np.where(dtr > 0)[0] + 1
    if breath_starts.size >= 2:
        max_tidal = np.max(tidal[breath_starts[-2]:breath_starts[-1]])
    else:
        max_tidal = np.max(tidal[tidal > 0]) if np.any(tidal > 0) else 0.0

    VAflow = np.concatenate((local_updates["VAflow_store"][i_buffer:], local_updates["VAflow_store"][:i_buffer]))
    t1 = np.concatenate((local_updates["t1_store"][i_buffer:], local_updates["t1_store"][:i_buffer]))
    t2 = np.concatenate((local_updates["t2_store"][i_buffer:], local_updates["t2_store"][:i_buffer]))
    VD = GV_dead * VAflow[-1] + V0_dead
    VDflow = (1 / (t1[-1] + t2[-1])) * VD
    Minute_Ventilation = (VAflow[-1] + VDflow) * 60

    cardiac_output = np.mean(local_updates["Q_pp_store"])
    Pa_O2 = np.mean(local_updates["Pa_O2_every_store"])
    Pa_CO2 = np.mean(local_updates["Pa_CO2_every_store"])

    Total_Volume = V_ra + V_rv + V_lv + V_la
    is_active = phi_atr > 0.0  # atrial contraction window
    edges = np.diff(is_active.astype(int))

    start_idx = np.where(edges == 1)[0] + 1  # 0 → active
    end_idx = np.where(edges == -1)[0] + 1  # active → 0

    if len(start_idx) and len(end_idx) and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]

    n_pairs = min(len(start_idx), len(end_idx))
    start_idx = start_idx[:n_pairs]
    end_idx = end_idx[:n_pairs]

    Total_Vol_min_idx = np.array([s + np.argmin(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]
    Total_Vol_max_idx = np.array([s + np.argmax(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    mean_min_Total_Volume = np.mean(Total_Volume[Total_Vol_min_idx])
    mean_max_Total_Volume = np.mean(Total_Volume[Total_Vol_max_idx])
    Pericardial_Volume_difference = mean_max_Total_Volume - mean_min_Total_Volume
    Vol_percentage_change = Pericardial_Volume_difference / mean_max_Total_Volume

    dP_lv_dt_store = np.concatenate((local_updates["dP_lv_dt_store"][i_buffer:], local_updates["dP_lv_dt_store"][:i_buffer]))
    dP_lv_dt_idx = np.array([s + np.argmax(dP_lv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    dP_rv_dt_store = np.concatenate((local_updates["dP_rv_dt_store"][i_buffer:], local_updates["dP_rv_dt_store"][:i_buffer]))
    dP_rv_dt_idx = np.array([s + np.argmax(dP_rv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    # print(np.mean(P_sa[open_idx1]), np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_edp_idx]), np.mean(P_la[P_la_descent1_idx]), Vol_percentage_change)

    IC_current = ODE_solution.y[:, -1]

    return ([np.mean(past_10_flat_segments), np.mean(P_sa[P_sa_max_idx]), np.mean(P_sa[open_idx1]),
            np.mean(V_lv[pairs_ao[:, 0]]), np.mean(V_lv[pairs_ao[:, 1]]), np.mean(V_rv[pairs_po[:, 0]]), np.mean(V_rv[pairs_po[:, 1]]),
            np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_edp_idx]),
            np.mean(V_ra[pairs_tr[:, 1]]), np.mean(V_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent1_idx]),
            np.mean(P_ra[P_ra_max_idx]), np.mean(P_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent2_idx]),
            np.mean(V_la[pairs_mi[:, 1]]), np.mean(V_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent1_idx]),
            np.mean(P_la[P_la_max_idx]), np.mean(P_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent2_idx]),
            np.mean(last_10_b4_LA_atrial_contract), np.mean(last_10_b4_RA_atrial_contract),
            np.mean(dP_lv_dt_store[dP_lv_dt_idx]), np.mean(dP_rv_dt_store[dP_rv_dt_idx]), max_tidal, Minute_Ventilation,
            cardiac_output, Pa_O2, Pa_CO2, Vol_percentage_change, mean_max_P_peri],
            IC_current, local_updates,
            [cs_t1, cs_t2, knots_1, knots_2])


def timeout_handler(signum, frame):
    raise TimeoutError("Simulation timeout")

def safe_simulate_cpu(
    params,
    storage,
    old_parameters,
    timeout=200,
    IC_initial=None,
    state="Rest",
    attempt=0,
    breath_coef=None,
    exercise_start_time=None,
):
    try:
        use_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        result = simulate_cpu(
            params, storage, old_parameters, IC_initial, state, attempt, breath_coef, exercise_start_time
        )
        if use_alarm:
            signal.alarm(0)  # Cancel timeout
        return result
    except Exception as exc:
        if hasattr(signal, "alarm"):
            signal.alarm(0)  # Cancel timeout
        print(f"simulation failed: {exc}")
        return zero_state_result(), None, None, None


def run_state_to_convergence(
    params,
    storage,
    old_parameters,
    IC_final=None,
    state="Rest",
    breath_coef=None,
    exercise_start_time=None,
    use_timeout=True,
    min_duration=MIN_MEASUREMENT_DURATION,
):
    result = zero_state_result()
    converged = False
    if IC_final is None:
        state_start_time = 0.0
    else:
        state_start_index = (storage["i"].item() - 1) % BUFFER_LIMIT
        state_start_time = storage["all_time"][state_start_index]

    for attempt in range(MAX_CONVERGENCE_ATTEMPTS):
        if use_timeout:
            result, IC_final, storage, breath_coef = safe_simulate_cpu(
                params,
                storage,
                old_parameters,
                IC_initial=IC_final,
                state=state,
                attempt=attempt,
                breath_coef=breath_coef,
                exercise_start_time=exercise_start_time,
            )
        else:
            result, IC_final, storage, breath_coef = simulate_cpu(
                params,
                storage,
                old_parameters,
                IC_initial=IC_final,
                state=state,
                attempt=attempt,
                breath_coef=breath_coef,
                exercise_start_time=exercise_start_time,
            )

        if storage is None or IC_final is None or is_failed_state_result(result):
            return zero_state_result(), None, None, None, False

        latest_nonzero_index = (storage["i"].item() - 1) % BUFFER_LIMIT
        latest_nonzero_value = storage["all_time"][latest_nonzero_index]
        state_elapsed = latest_nonzero_value - state_start_time

        if has_converged(storage) and state_elapsed >= min_duration:
            converged = True
            break

    return result, IC_final, storage, breath_coef, converged


def run_basepoint(base_sample, old_Parameters):
    storage_copy = make_fresh_storage()

    try:
        # Run basepoints to rest convergence; perturbations warm-start from this state.
        base_result, IC_final, storage_final, breath_coef, rest_converged = run_state_to_convergence(
            base_sample,
            storage_copy,
            old_Parameters,
            state="Rest",
            use_timeout=False,
        )

        minimise_coef = [
            base_sample["GV_dead"],
            base_sample["V0_dead"],
            base_sample["E_rs"],
            base_sample["R_rs"],
        ]

        if is_failed_state_result(base_result):
            print("fail", dict(list(base_sample.items())[:10]))
            return None

        return {
            "result": base_result,
            "IC_final": IC_final,
            "storage_final": storage_final,
            "breath_coef": breath_coef,
            "minimise_coef": minimise_coef,
        }
    except Exception as e:
        print(f"[BASEPOINT EXCEPTION] idx={base_sample}")
        return None

def parallel_simulations(param_samples, n_jobs, save_path='Result_DGSM_union.npy'):
    results_all = []

    if os.path.exists(save_path):
        os.remove(save_path)

    # Break into blocks of block_size (1 base + (block_size - 1) perturbations)
    block_size = len(param_samples[0]) + 1
    param_blocks = [param_samples[i:i + block_size] for i in range(0, len(param_samples), block_size)]

    # run base points first
    base_results = Parallel(n_jobs=n_jobs)(delayed(run_basepoint)(params[0], Old_Parameters)
        for params in param_blocks)


    # go through each base point and perturbation with the corresponding initial conditions
    for i, block in enumerate(param_blocks):
        base = base_results[i]

        if base is None: # base failed → whole block invalid
            results_all.extend(np.zeros((block_size, UNION_OUTPUT_DIM)))
            print("block_failed")
            np.save(save_path, np.array(results_all))
            continue

        # # Otherwise, run full block in parallel
        # with tqdm_joblib.tqdm_joblib(tqdm(desc=f"Sim Block {i}", total=len(block), disable=True)):
        #     results_perturbations = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(params,
        #     base["storage_final"], Old_Parameters, base["IC_final"], base["breath_coef"], base["minimise_coef"]) for params in block)
        t0 = time.time()
        results_perturbations = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(params,
        base["storage_final"], Old_Parameters, base["IC_final"], base["breath_coef"], base["minimise_coef"]) for params in block)
        elapsed = time.time() - t0
        rest_nonconverged = sum(not res["rest_converged"] for res in results_perturbations)
        exercise_nonconverged = sum(not res["exercise_converged"] for res in results_perturbations)
        print(
            f"Block {i}/{len(param_blocks)}: {elapsed:.1f}s | "
            f"Rest non-converged: {rest_nonconverged}/{len(block)} | "
            f"Exercise non-converged: {exercise_nonconverged}/{len(block)}"
        )

        results_block = [res["result"] for res in results_perturbations]
        results_all.extend(results_block)

        # Save chunk incrementally (appending)
        # np.save(f'IC_final_{i:03d}.npy', IC_final)  # individual chunks
        # np.save(f'Next_final_{i:03d}.npy', storage_final)  # individual chunks

        # Save after each block
        np.save(save_path, np.array(results_all))

    return results_all


def run_simulation(params, storage_final, Old_Parameters, IC_final, breath_coef, minimise_coef):
    storage_final = copy_storage(storage_final)
    IC_final = IC_final.copy()

    # If the perturbation changes breathing-optimiser inputs, recompute the
    # breathing spline before reconverging at rest.
    next_minimise_coef = [params["GV_dead"], params["V0_dead"], params["E_rs"], params["R_rs"]]
    local_breath_coef = breath_coef if next_minimise_coef == minimise_coef else None

    result_rest, IC_final, storage_final, local_breath_coef, rest_converged = run_state_to_convergence(
        params,
        storage_final,
        Old_Parameters,
        IC_final=IC_final,
        state="Rest",
        breath_coef=local_breath_coef,
    )

    if storage_final is None or IC_final is None or is_failed_state_result(result_rest):
        return {
            "result": zero_union_result(),
            "rest_converged": False,
            "exercise_converged": False,
        }

    latest_nonzero_index = (storage_final["i"].item() - 1) % BUFFER_LIMIT
    exercise_start_time = storage_final["all_time"][latest_nonzero_index]

    result_exercise, IC_final, storage_final, local_breath_coef, exercise_converged = run_state_to_convergence(
        params,
        storage_final,
        Old_Parameters,
        IC_final=IC_final,
        state="Exercise",
        breath_coef=local_breath_coef,
        exercise_start_time=exercise_start_time,
    )

    if storage_final is None or IC_final is None or is_failed_state_result(result_exercise):
        return {
            "result": zero_union_result(),
            "rest_converged": rest_converged,
            "exercise_converged": False,
        }

    return {
        "result": list(result_rest) + list(result_exercise),
        "rest_converged": rest_converged,
        "exercise_converged": exercise_converged,
    }



if __name__ == "__main__":
    lower = 0.5
    upper = 1.5
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
            [0.0873 * lower, 0.0873 * upper], [1.2 * 0.85, 1.2 * 1.15], [1.2 * 0.85, 1.2 * 1.15], [181 * lower, 181 * upper], [31 * lower, 31 * upper]]
    })

    param_keys = list(sp["names"])

    # DGSM uses finite differences sampling since it is a derivative based method
    # X = finite_diff.sample(sp, 500)
    # np.save("DGSM_500_X_union_50_09_06.npy", X)
    X = np.load("DGSM_500_X_union_50_09_06.npy")
    # A = np.load("X:\home\project\DGSM_Paper_Final_Union\Result_task_00_union_300_400.npy")


    param_samples = [dict(zip(param_keys, row)) for row in X]
    print(f"Number of samples created: {len(X)}")

    Result = parallel_simulations(param_samples, n_jobs=64)
    np.save(f'DGSM_500_Result_union_50_09_06.npy', Result)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--task-id", type=int, required=True)
    # parser.add_argument("--n-jobs", type=int, required=True)
    # parser.add_argument("--basepoints-per-job", type=int, required=True)
    # args = parser.parse_args()
    #
    # task_id = args.task_id
    # block_size = X.shape[1] + 1
    # basepoints_per_job = args.basepoints_per_job
    # rows_per_job = block_size * basepoints_per_job
    # start = task_id * rows_per_job
    # end = (task_id + 1) * rows_per_job
    # X = X[start:end]
    #
    # param_samples = [dict(zip(param_keys, row)) for row in X]
    # print(f"Number of samples created: {len(X)}")
    #
    # result_path = f'Result_task_{task_id:02d}_union.npy'
    # Result = parallel_simulations(param_samples, n_jobs=args.n_jobs, save_path=result_path)
