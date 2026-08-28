import numpy as np
import torch
from SALib import ProblemSpec
from pathlib import Path
import sys

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from Simulator_Union import Simulator
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from check import Parameters as trial
from All_derivatives_njit import (
    model_derivatives,
    make_wrapper,
    co2_content_from_pco2,
    invert_co2_content_to_pco2,
)
from rk23_njit import solve_rk23
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective
import signal

from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from fixed_params import Parameters as Old_Parameters
from Initial_Conditions_after_running_again import Initial_Conditions
from scipy.interpolate import CubicSpline
from All_Next_Conditions import make_fresh_storage

lower = 0.8
upper = 1.2
#
#
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

    'bounds': [[1.0 * lower, 1.0 * upper] for _ in range(272)]
})


target_values = np.arange(0, 10000, 10)
BUFFER_LIMIT = 80000

# Integrate with the njit Bogacki-Shampine loop in rk23_njit rather than
# scipy's Python RK23 driver.  Both produce bit-identical trajectories; set
# this to False to fall back to scipy (used for regression checks).
USE_NJIT_RK23 = True

max_time = 600 # Maximum time limit to avoid infinite loops
RAW_OUTPUT_DIM = 31
CONVERGENCE_TOLERANCE = 0.03
MAX_CONVERGENCE_ATTEMPTS = 5
MIN_MEASUREMENT_DURATION = 600

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


# Indices of the four gas *content* states in IC_overall (CTO2 76, CvtCO2 77, CBO2 78,
# CvbCO2 79).  Contents are parameter-dependent; the tensions they represent are not.
# Carrying tensions rather than contents into a run keeps every sample at the same
# physiological starting state instead of at the same numbers.
GAS_KEYS = ("a2", "alpha2", "beta2", "C2", "K2", "alpha_O2", "Z")


def _gas_params(sample, old_parameters):
    return [sample[k] if k in sample else old_parameters[k] for k in GAS_KEYS]


def gas_tensions(IC, sample, old_parameters):
    a2g, al2, be2, C2, K2, aO2, Z = _gas_params(sample, old_parameters)
    PvtO2 = max(IC[76] / aO2, 1.0)
    PvbO2 = max(IC[78] / aO2, 1.0)
    return (PvtO2,
            invert_co2_content_to_pco2(IC[77], PvtO2, a2g, al2, be2, C2, K2, Z),
            PvbO2,
            invert_co2_content_to_pco2(IC[79], PvbO2, a2g, al2, be2, C2, K2, Z))


def apply_gas_tensions(IC, tensions, sample, old_parameters):
    a2g, al2, be2, C2, K2, aO2, Z = _gas_params(sample, old_parameters)
    PvtO2, PvtCO2, PvbO2, PvbCO2 = tensions
    IC = IC.copy()
    IC[76] = aO2 * PvtO2
    IC[78] = aO2 * PvbO2
    IC[77] = co2_content_from_pco2(PvtCO2, PvtO2, a2g, al2, be2, C2, K2, Z)
    IC[79] = co2_content_from_pco2(PvbCO2, PvbO2, a2g, al2, be2, C2, K2, Z)
    return IC


# The gas entries in Initial_Conditions_after_running_again.py are CONTENTS, which only
# mean something alongside the dissociation-curve parameters they were converged at.
REFERENCE_GAS_PARAMS = {"a2": 1.819, "alpha2": 0.05591, "beta2": 0.03255, "C2": 87.0, "K2": 194.4, "alpha_O2": 3.17e-05,}
REST_GAS_TENSIONS = gas_tensions(IC_overall, REFERENCE_GAS_PARAMS, Old_Parameters)


def find_valve_edge_indices(valve_state, N):
    """Vectorised form of the per-sample "first open after N closed samples" loop."""
    cumulative = np.concatenate(([0], np.cumsum(valve_state)))
    previous_window = cumulative[N:-1] - cumulative[:-N - 1]
    return np.flatnonzero(valve_state[N:] & (previous_window == 0)) + N


class Cardiopulmonary(Simulator):
    def __init__(self, param_ranges, output_names):
        super().__init__(param_ranges, output_names, log_level="debug")

    def _forward(self, X):
        X = X.detach().cpu().numpy().astype(float)
        param_sample = [dict(zip(sp["names"], row)) for row in X]
        storage_local = make_fresh_storage()
        return self.rest_and_exercise_simulation(param_sample, storage_local, Old_Parameters)

    @staticmethod
    def _zero_result():
        return torch.zeros((1, RAW_OUTPUT_DIM), dtype=torch.float32)

    @staticmethod
    def _ordered_store(storage, key):
        i_buffer = storage["i"].item() % BUFFER_LIMIT
        return np.concatenate((storage[key][i_buffer:], storage[key][:i_buffer]))

    @staticmethod
    def _last_distinct_values(values, n=10):
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

    def _has_converged(self, storage):
        hr_segments = self._last_distinct_values(self._ordered_store(storage, "HR_store"))
        return hr_segments.size >= 2 and np.ptp(hr_segments) < CONVERGENCE_TOLERANCE

    @staticmethod
    def _is_failed_result(result):
        return (
            not torch.is_tensor(result)
            or result.numel() == 0
            or not torch.isfinite(result).all()
            or torch.all(result == 0)
        )

    def combined_system(
        self, t, Initial_Conditions_numpy, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control,
        num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time,
        fast_model_derivatives
    ):

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
        derivatives_all = fast_model_derivatives(t, resp_contr_state, Initial_Conditions_dict, num_removed, i,
                                                 BUFFER_LIMIT, all_time, Input_Parameters, cs_t1, cs_t2, knots_1,
                                                 knots_2, exercise_start_time)
        all_time[(i - num_removed) % BUFFER_LIMIT] = t
        Initial_Conditions_dict["i"][0] = i - num_removed + 1
        Initial_Conditions_dict["j"][0] = Initial_Conditions_dict["j"].item() - num_removed + 1

        # # Debugging check for progress
        # if t != 0:
        #     diff = np.abs(t - target_values)
        #     if np.any(diff < 0.0001):
        #         print(t)

        return derivatives_all

    def simulate_cpu(
        self,
        Current_Parameters,
        local_updates,
        old_parameters,
        IC_initial=None,
        state=None,
        attempt=None,
        breath_coef=None,
        exercise_start_time=None,
    ):
        i = local_updates["i"].item()
        latest_nonzero_index = (i - 1) % BUFFER_LIMIT
        latest_nonzero_value = local_updates["all_time"][latest_nonzero_index]

        if IC_initial is None:
            IC_current = IC_overall.copy()
            t_span = [0, max_time]
            exercise_start_time = np.inf
        elif state == "Exercise" and attempt == 0:
            IC_current = IC_initial.copy()
            t_span = [latest_nonzero_value, latest_nonzero_value + 1400]
            if exercise_start_time is None:
                exercise_start_time = latest_nonzero_value
        else:
            IC_current = IC_initial.copy()
            segment = max_time if (state == "Rest" and attempt == 0) else 60
            t_span = [latest_nonzero_value, latest_nonzero_value + segment]
            if state == "Exercise":
                if exercise_start_time is None:
                    exercise_start_time = latest_nonzero_value
            else:
                exercise_start_time = np.inf

        Current_Parameters = Current_Parameters[0]

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
             "R_ev_n", "R_sv_n", "R_bv_n", "R_hv_n", "R_rmv_n", "R_amv_n", "C_ev", "C_sv", "C_bv", "C_hv", "C_rmv",
             "C_amv",
             "kr_am", "P_0"])

        # Cardio controller parameters
        (fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv,
         kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv,
         Wc_sh,
         Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0, Emax_rv0, fes_min,
         GEmax_lv,
         GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, AT, g_ccsh,
         g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv,
         theta_shn, theta_spn, theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, P_n_max,
         f_acCO2_n, f_ac_max, f_ac_min, k_ac, K_H, PaO2_ac_n, G_ap, DT_v, GT_s, GT_v, T0, A, B, C, D,
         Cvb_O2_n, gb_O2, R_bpn, Cvh_O2_n, Cvrm_O2_n, gh_O2, grm_O2, Kh_CO2, Krm_CO2, MO2_hpn,
         MO2_rmp, R_hpn, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max, phi_min) = \
            [Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
             ["fab_o", "fes_o", "fes_inf", "fes_max", "fev_o",
              "fev_inf", "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v",
              "Ysh_max",
              "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp",
              "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv",
              "Wt_v",
              "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
              "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp", "g_ccsv",
              "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh",
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
            (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
             ["GV_dead", "KcCO2", "KcMRV", "KpCO2", "KpO2",
              "V0_dead", "VA_rest", "lambda1", "lambda2", "n", "Pmax", "Pmax_dot", "E_rs", "R_rs", "P_ao"])

        # added params
        (Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
         Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
         R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
         Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vu_pa, Vu_pp,
         Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv,
         tau_Vev,
         tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts,
         tau_Tv,
         tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev,
         DV_rmv,
         DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
         scale_param1, scale_param3, scale_param4, scale_param6,
         Pa_O2_lower, rise_time_atr, rise_time_ven,
         fall_time_ven, ahead1, theta_min, delta_P, r, l, V_nominal, V_scale
         ) = \
            (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
             ["Kp_ao", "Kf_ao", "Kb_ao",
              "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max", "Kp_po", "Kf_po", "Kb_po",
              "Kv_po",
              "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "alpha_O2", "R_po", "R_mi", "R_tr",
              "R_ao",
              "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal", "Vu_sa", "V_tot", "Vu_jp",
              "Vu_bv", "Vu_hv", "Vu_vc", "Vu_pa", "Vu_pp", "Vu_pv",
              "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp",
              "tau_Rsp",
              "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc",
              "tau_isc",
              "tau_p", "tau_z", "tau_ac", "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2", "tau_O2", "tau_w", "tau_M",
              "tau_met",
              "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv",
              "DT_s", "DT_v",
              "Dmet", "Fi_CO2", "Fi_O2", "Ta", "T1", "T2", "VL_CO2", "VL_O2", "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2",
              "tau_MRV",
              "scale_param1", "scale_param3", "scale_param4", "scale_param6",
              "Pa_O2_lower", "rise_time_atr", "rise_time_ven",
              "fall_time_ven", "ahead1", "theta_min", "delta_P", "r", "l", "V_nominal", "V_scale"])

        # Gas stores start from fixed tensions, not fixed contents: the content<->tension
        # mapping depends on sampled parameters (alpha_O2, C2, a2, K2, alpha2, beta2), so
        # freezing contents would start each draw at a different physiological state.
        if IC_initial is None:
            IC_current = apply_gas_tensions(IC_current, REST_GAS_TENSIONS, Current_Parameters, old_parameters)

        # Breathing profile depends on this parameter set, not on the current
        # convergence segment, so reuse it across rest/exercise extensions.
        if breath_coef is None:
            breath_coef = self.minimise_breathing(
                1.5, 1.85, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao
            )
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

        # Solve ODE in one go
        if USE_NJIT_RK23:
            ode_status, IC_final = solve_rk23(
                local_updates,
                t_span,
                IC_current,
                num_cardio + num_cardio_control + num_gas + num_resp_control,
                BUFFER_LIMIT,
                0.001,
                1e-3,
                1e-6,
                Input_Parameters, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time,
            )
        else:
            # Bind the storage arrays once instead of looking them up by name during
            # every derivative evaluation.
            fast_model_derivatives = make_wrapper(local_updates)

            ODE_solution = solve_ivp(
                self.combined_system,
                t_span,
                IC_current,
                max_step=0.001,
                method="RK23",
                rtol=1e-3,
                atol=1e-6,
                args=(
                local_updates, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2,
                knots_1, knots_2, exercise_start_time, fast_model_derivatives)
            )
            ode_status = ODE_solution.status
            IC_final = ODE_solution.y[:, -1].copy()

        if ode_status == -1:
            # Integration failed or early termination
            print("fail")
            print(r, l, V_nominal, V_scale)
            return self._zero_result(), None, None, None

        i_buffer = local_updates["i"].item() % BUFFER_LIMIT

        theta_ao = np.concatenate(
            (local_updates["theta_ao_store"][i_buffer:], local_updates["theta_ao_store"][:i_buffer]))
        theta_po = np.concatenate(
            (local_updates["theta_po_store"][i_buffer:], local_updates["theta_po_store"][:i_buffer]))
        theta_mi = np.concatenate(
            (local_updates["theta_mi_store"][i_buffer:], local_updates["theta_mi_store"][:i_buffer]))
        theta_tr = np.concatenate(
            (local_updates["theta_tr_store"][i_buffer:], local_updates["theta_tr_store"][:i_buffer]))

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
            return self._zero_result(), None, None, None

        is_open_po = theta_po > theta_min
        open_idx2 = find_valve_edge_indices(is_open_po, N)

        is_closed_po = theta_po <= theta_min
        close_idx2 = find_valve_edge_indices(is_closed_po, N)

        if open_idx2.size == 0 or close_idx2.size == 0:
            print("po fail")
            return self._zero_result(), None, None, None

        is_open_mi = theta_mi > theta_min
        open_idx3 = find_valve_edge_indices(is_open_mi, N)

        is_closed_mi = theta_mi <= theta_min
        close_idx3 = find_valve_edge_indices(is_closed_mi, N)

        if open_idx3.size == 0 or close_idx3.size == 0:
            print("mi fail")
            return self._zero_result(), None, None, None

        is_open_tr = theta_tr > theta_min
        open_idx4 = find_valve_edge_indices(is_open_tr, N)

        is_closed_tr = theta_tr <= theta_min
        close_idx4 = find_valve_edge_indices(is_closed_tr, N)

        if open_idx4.size == 0 or close_idx4.size == 0:
            print("tr fail")
            return self._zero_result(), None, None, None

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

        # systolic pressure
        P_sa = np.concatenate((local_updates["P_sa_store"][i_buffer:], local_updates["P_sa_store"][:i_buffer]))
        P_sa_max_idx = np.array([o + np.argmax(P_sa[o:c]) for o, c in pairs_ao])

        n_pairs = min(len(start_idx), len(end_idx))
        # If first end comes before first start, skip that end
        if len(end_idx) > 0 and len(start_idx) > 0 and end_idx[0] < start_idx[0]:
            end_idx = end_idx[1:]
            n_pairs = min(len(start_idx), len(end_idx))

        # Truncate to matching pairs
        start_idx = start_idx[:n_pairs]
        end_idx = end_idx[:n_pairs]

        P_la = np.concatenate((local_updates["P_la_store"][i_buffer:], local_updates["P_la_store"][:i_buffer]))
        # max pressure at atrial contraction
        P_la_max_idx = np.array([s + np.argmax(P_la[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

        # period of V descent when mitral valve is open -> get second min la P
        P_la_descent2_idx = np.array([o + np.argmin(P_la[o:c]) for o, c in pairs_mi])
        P_la_descent1_idx = np.array(
            [c + np.argmin(P_la[c:o_next]) for (_, c), (o_next, _) in zip(pairs_mi[:-1], pairs_mi[1:])])

        P_ra = np.concatenate((local_updates["P_ra_store"][i_buffer:], local_updates["P_ra_store"][:i_buffer]))
        # max pressure at atrial contraction
        P_ra_max_idx = np.array([s + np.argmax(P_ra[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

        # period of V descent when tricuspid valve is open -> get second min la P
        P_ra_descent2_idx = np.array([o + np.argmin(P_ra[o:c]) for o, c in pairs_tr])
        P_ra_descent1_idx = np.array(
            [c + np.argmin(P_ra[c:o_next]) for (_, c), (o_next, _) in zip(pairs_tr[:-1], pairs_tr[1:])])

        P_rv = np.concatenate((local_updates["P_rv_store"][i_buffer:], local_updates["P_rv_store"][:i_buffer]))
        P_rv_max_idx = np.array([o + np.argmax(P_rv[o:c]) for o, c in pairs_po])
        # RVEDP: last sample before ventricular activation rises each beat.
        phi_eps = 1e-8
        phi_rise_idx = np.where((phi[:-1] <= phi_eps) & (phi[1:] > phi_eps))[0] + 1
        P_rv_edp_idx = phi_rise_idx[-10:] - 1

        time_since_beat_store = np.concatenate(
            (local_updates["time_since_beat_store"][i_buffer:], local_updates["time_since_beat_store"][:i_buffer]))
        cardiac_cycle_start_idx = np.where(np.diff(time_since_beat_store) > 0)[0] + 1

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

        dP_lv_dt_store = np.concatenate(
            (local_updates["dP_lv_dt_store"][i_buffer:], local_updates["dP_lv_dt_store"][:i_buffer]))
        dP_lv_dt_idx = np.array([s + np.argmax(dP_lv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

        dP_rv_dt_store = np.concatenate(
            (local_updates["dP_rv_dt_store"][i_buffer:], local_updates["dP_rv_dt_store"][:i_buffer]))
        dP_rv_dt_idx = np.array([s + np.argmax(dP_rv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

        # print(np.mean(P_sa[open_idx1]), np.mean(P_rv[P_rv_max_idx]))
        # LA_Contraction_Volume_diff = np.mean(last_10_b4_LA_atrial_contract) - np.mean(V_la[pairs_mi[:, 1]])
        # RA_Contraction_Volume_diff = np.mean(last_10_b4_RA_atrial_contract) - np.mean(V_ra[pairs_tr[:, 1]])


        result = torch.tensor([np.mean(past_10_flat_segments), np.mean(P_sa[P_sa_max_idx]), np.mean(P_sa[open_idx1]),
            np.mean(V_lv[pairs_ao[:, 0]]), np.mean(V_lv[pairs_ao[:, 1]]), np.mean(V_rv[pairs_po[:, 0]]), np.mean(V_rv[pairs_po[:, 1]]),
            np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_edp_idx]),
            np.mean(V_ra[pairs_tr[:, 1]]), np.mean(V_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent1_idx]),
            np.mean(P_ra[P_ra_max_idx]), np.mean(P_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent2_idx]),
            np.mean(V_la[pairs_mi[:, 1]]), np.mean(V_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent1_idx]),
            np.mean(P_la[P_la_max_idx]), np.mean(P_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent2_idx]),
            np.mean(last_10_b4_LA_atrial_contract), np.mean(last_10_b4_RA_atrial_contract),
            np.mean(dP_lv_dt_store[dP_lv_dt_idx]), np.mean(dP_rv_dt_store[dP_rv_dt_idx]), max_tidal, Minute_Ventilation,
            cardiac_output, Pa_O2, Pa_CO2, Vol_percentage_change], dtype=torch.float32).unsqueeze(0)
        return result, IC_final, local_updates, breath_coef

    def minimise_breathing(self, t1, t2, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao):
        dt = 0.001  # must edit in Resp_Control_Breath_Optimiser too
        bounds = [(0.5, 6), (0.5, 9)]  # [t1, t2]
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

            res = minimize(objective, x0=np.array(initial_guess[-2:]),
                           args=(required_params, VAflow, VD_volume, dt, tolerance), method='nelder-mead',
                           bounds=bounds)
            t1_opt, t2_opt = res.x
            optimal_t1.append(t1_opt)
            optimal_t2.append(t2_opt)
            initial_guess.extend(res.x)

        # Convert to arrays for indexing
        # VAflow_clean = np.array(VAflow_repeated)
        t1_clean = np.array(optimal_t1)
        t2_clean = np.array(optimal_t2)

        # t1_mean = np.array([np.nanmean(t1_clean[VAflow_clean == v]) for v in VAflow_vals])
        # t2_mean = np.array([np.nanmean(t2_clean[VAflow_clean == v]) for v in VAflow_vals])

        cs_t1 = CubicSpline(VAflow_vals, t1_clean, bc_type="natural")
        cs_t2 = CubicSpline(VAflow_vals, t2_clean, bc_type="natural")

        return cs_t1.c, cs_t2.c, cs_t1.x, cs_t2.x

    def safe_simulate_cpu(
        self, params, storage, old_parameters, IC_final, state, attempt, breath_coef=None,
        exercise_start_time=None, timeout=200
    ):
        try:
            use_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, self.timeout_handler)
                signal.alarm(timeout)
            result = self.simulate_cpu(
                params, storage, old_parameters, IC_final, state, attempt, breath_coef, exercise_start_time
            )
            if use_alarm:
                signal.alarm(0)  # Cancel timeout
            return result
        except Exception as exc:
            if hasattr(signal, "alarm"):
                signal.alarm(0)  # Cancel timeout
            print(f"simulation failed: {exc}")
            return self._zero_result(), None, None, None

    def _state_elapsed(self, storage, state_start_time):
        latest_nonzero_index = (storage["i"].item() - 1) % BUFFER_LIMIT
        return storage["all_time"][latest_nonzero_index] - state_start_time

    def rest_and_exercise_simulation(self, params, storage, old_parameters):
        storage_final = {key: value.copy() for key, value in storage.items()}
        IC_final = None
        breath_coef = None
        result_rest = self._zero_result()
        result_exercise = self._zero_result()

        for attempt in range(MAX_CONVERGENCE_ATTEMPTS):
            result_rest, IC_final, storage_final, breath_coef = self.safe_simulate_cpu(
                params, storage_final, old_parameters, IC_final, state="Rest", attempt=attempt, breath_coef=breath_coef
            )

            if storage_final is None or IC_final is None or self._is_failed_result(result_rest):
                return torch.cat([self._zero_result(), self._zero_result()], dim=1)

            # HR converging is not enough on its own: ventilation and the gas stores settle
            # far more slowly, so require the state to have run for MIN_MEASUREMENT_DURATION
            # before its outputs are read.
            if (self._has_converged(storage_final)
                    and self._state_elapsed(storage_final, 0.0) >= MIN_MEASUREMENT_DURATION):
                break

        latest_nonzero_index = (storage_final["i"].item() - 1) % BUFFER_LIMIT
        exercise_start_time = storage_final["all_time"][latest_nonzero_index]

        for attempt in range(MAX_CONVERGENCE_ATTEMPTS):
            result_exercise, IC_final, storage_final, breath_coef = self.safe_simulate_cpu(
                params, storage_final, old_parameters, IC_final, state="Exercise", attempt=attempt,
                breath_coef=breath_coef, exercise_start_time=exercise_start_time
            )

            if storage_final is None or IC_final is None or self._is_failed_result(result_exercise):
                return torch.cat([self._zero_result(), self._zero_result()], dim=1)

            if (self._has_converged(storage_final)
                    and self._state_elapsed(storage_final, exercise_start_time) >= MIN_MEASUREMENT_DURATION):
                return torch.cat([result_rest, result_exercise], dim=1)

        # print("exercise did not meet convergence tolerance before max attempts")
        return torch.cat([result_rest, result_exercise], dim=1)



    def timeout_handler(self, signum, frame):
        raise TimeoutError("Simulation timeout")
