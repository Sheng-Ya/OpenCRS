import csv
import pickle
import sys
from pathlib import Path

local_dependency_directory = Path(__file__).resolve().parent / ".python_deps"
if local_dependency_directory.exists():
    sys.path.append(str(local_dependency_directory))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective
from scipy.signal import find_peaks, savgol_filter
# from line_profiler import LineProfiler
from All_derivatives import model_derivatives
from rk23_njit import solve_rk23
from Entire_system.fixed_params import Parameters
from check import Parameters as new_params

from Initial_Conditions_after_running_again import Initial_Conditions
from Next_Conditions_all_derivatives import Next_Conditions, make_fresh_storage


target_values = np.arange(0, 10000, 10)

time_saved = 0.005
BUFFER_LIMIT = 80000

# Use the fully compiled Bogacki-Shampine loop.  Set to False for a direct
# regression comparison with SciPy's Python RK23 driver.
USE_NJIT_RK23 = True

min_time = 10 # Minimum time in seconds before checking
max_time = 800 # Maximum time limit to avoid infinite loops
time_step = 200  # Chunk size per solve

EXERCISE_START_TIME = 100.0
BAROREFLEX_DIAGNOSTIC_BEATS = 5
BAROREFLEX_SATURATION_LIMIT = 0.05


def get_parameter_value(name):
    return new_params[name] if name in new_params else Parameters[name]


def get_float_option(option):
    if option not in sys.argv:
        return None

    option_index = sys.argv.index(option)
    if option_index + 1 >= len(sys.argv):
        raise ValueError(f"{option} requires a numeric value")
    return float(sys.argv[option_index + 1])


def get_last_complete_beats(time, pressure, start_time, end_time, number_of_beats):
    state_indices = np.flatnonzero((time >= start_time) & (time < end_time))
    if state_indices.size == 0:
        raise ValueError(f"No pressure samples found between {start_time} and {end_time} s")

    state_pressure = pressure[state_indices]
    peak_prominence = max(2.0, 0.10 * np.ptp(state_pressure))
    local_peaks, _ = find_peaks(state_pressure, prominence=peak_prominence)
    peak_indices = state_indices[local_peaks]

    required_peaks = number_of_beats + 1
    if peak_indices.size < required_peaks:
        raise ValueError(
            f"Only {peak_indices.size} arterial pressure peaks were detected between "
            f"{start_time} and {end_time} s; {required_peaks} are required"
        )

    beat_boundaries = peak_indices[-required_peaks:]
    window_indices = np.arange(beat_boundaries[0], beat_boundaries[-1] + 1)
    return window_indices, beat_boundaries


def time_average(values, time):
    duration = time[-1] - time[0]
    if duration <= 0:
        return np.nan
    return np.trapz(values, time) / duration


def calculate_baroreflex_metrics(
        state_name, time, P_sa, dP_sa_dt, P_tilda, P_n_current, sigmoid_argument,
        normalised_f_ab, f_ab, beat_boundaries, saturation_limit, firing_gain_scale):
    beat_rows = []

    for beat_number, (beat_start, beat_end) in enumerate(
            zip(beat_boundaries[:-1], beat_boundaries[1:]), start=1):
        beat_slice = slice(beat_start, beat_end + 1)
        beat_time = time[beat_slice]
        beat_q = normalised_f_ab[beat_slice]
        beat_f_ab = f_ab[beat_slice]
        beat_local_gain = firing_gain_scale * beat_q * (1 - beat_q)

        lower_saturation = time_average((beat_q <= saturation_limit).astype(float), beat_time)
        upper_saturation = time_average((beat_q >= 1 - saturation_limit).astype(float), beat_time)

        beat_rows.append({
            "state": state_name,
            "beat": beat_number,
            "duration_s": beat_time[-1] - beat_time[0],
            "mean_f_ab_spikes_per_s": time_average(beat_f_ab, beat_time),
            "mean_local_firing_gain_spikes_per_s_per_mmHg": time_average(beat_local_gain, beat_time),
            "lower_saturation_fraction": lower_saturation,
            "upper_saturation_fraction": upper_saturation,
            "total_saturation_fraction": lower_saturation + upper_saturation,
        })

    state_slice = slice(beat_boundaries[0], beat_boundaries[-1] + 1)
    state_time = time[state_slice]
    state_q = normalised_f_ab[state_slice]
    state_local_gain = firing_gain_scale * state_q * (1 - state_q)
    lower_saturation = time_average((state_q <= saturation_limit).astype(float), state_time)
    upper_saturation = time_average((state_q >= 1 - saturation_limit).astype(float), state_time)

    state_summary = {
        "state": state_name,
        "mean_P_sa_mmHg": time_average(P_sa[state_slice], state_time),
        "systolic_P_sa_mmHg": np.max(P_sa[state_slice]),
        "diastolic_P_sa_mmHg": np.min(P_sa[state_slice]),
        "pulse_pressure_mmHg": np.ptp(P_sa[state_slice]),
        "maximum_dP_sa_dt_mmHg_per_s": np.max(dP_sa_dt[state_slice]),
        "minimum_dP_sa_dt_mmHg_per_s": np.min(dP_sa_dt[state_slice]),
        "minimum_P_tilda_mmHg": np.min(P_tilda[state_slice]),
        "maximum_P_tilda_mmHg": np.max(P_tilda[state_slice]),
        "mean_P_tilda_mmHg": time_average(P_tilda[state_slice], state_time),
        "mean_P_n_current_mmHg": time_average(P_n_current[state_slice], state_time),
        "minimum_sigmoid_argument": np.min(sigmoid_argument[state_slice]),
        "maximum_sigmoid_argument": np.max(sigmoid_argument[state_slice]),
        "minimum_normalised_f_ab": np.min(state_q),
        "maximum_normalised_f_ab": np.max(state_q),
        "mean_f_ab_spikes_per_s": time_average(f_ab[state_slice], state_time),
        "mean_local_firing_gain_spikes_per_s_per_mmHg": time_average(state_local_gain, state_time),
        "minimum_f_ab_spikes_per_s": np.min(f_ab[state_slice]),
        "maximum_f_ab_spikes_per_s": np.max(f_ab[state_slice]),
        "lower_saturation_fraction": lower_saturation,
        "upper_saturation_fraction": upper_saturation,
        "total_saturation_fraction": lower_saturation + upper_saturation,
    }
    return state_summary, beat_rows


def plot_baroreflex_diagnostics(solution):
    time = solution.t
    state_variables = solution.y

    P_sa_index = required_cardio_keys.index("P_sa")
    P_tilda_index = num_cardio + required_cardio_control_keys.index("P_tilda")
    P_n_current_index = num_cardio + required_cardio_control_keys.index("P_n_current")

    P_sa = state_variables[P_sa_index]
    P_tilda = state_variables[P_tilda_index]
    P_n_current = state_variables[P_n_current_index]

    # The solver is restricted to a 0.001 s maximum step, so differentiating the
    # accepted P_sa state gives a well-resolved visual check of the derivative
    # that enters the baroreceptor lead term.
    dP_sa_dt = np.gradient(P_sa, time, edge_order=2)

    f_ab_min = get_parameter_value("f_ab_min")
    f_ab_max = get_parameter_value("f_ab_max")
    k_ab = get_parameter_value("k_ab")
    tau_p = get_parameter_value("tau_p")
    tau_z = get_parameter_value("tau_z")

    sigmoid_argument = (P_tilda - P_n_current) / k_ab
    normalised_f_ab = np.empty_like(sigmoid_argument)
    positive_argument = sigmoid_argument >= 0
    normalised_f_ab[positive_argument] = 1 / (1 + np.exp(-sigmoid_argument[positive_argument]))
    negative_exponential = np.exp(sigmoid_argument[~positive_argument])
    normalised_f_ab[~positive_argument] = negative_exponential / (1 + negative_exponential)
    f_ab = f_ab_min + (f_ab_max - f_ab_min) * normalised_f_ab

    rest_window, rest_boundaries = get_last_complete_beats(
        time, P_sa, 0.0, EXERCISE_START_TIME, BAROREFLEX_DIAGNOSTIC_BEATS
    )
    exercise_window, exercise_boundaries = get_last_complete_beats(
        time, P_sa, EXERCISE_START_TIME, time[-1] + 1.0, BAROREFLEX_DIAGNOSTIC_BEATS
    )

    diagnostic_states = [
        ("Rest", rest_window, rest_boundaries),
        ("Exercise", exercise_window, exercise_boundaries),
    ]

    state_summaries = []
    beat_rows = []
    for state_name, _, beat_boundaries in diagnostic_states:
        state_summary, state_beat_rows = calculate_baroreflex_metrics(
            state_name, time, P_sa, dP_sa_dt, P_tilda, P_n_current, sigmoid_argument,
            normalised_f_ab, f_ab, beat_boundaries, BAROREFLEX_SATURATION_LIMIT,
            (f_ab_max - f_ab_min) / k_ab
        )
        state_summaries.append(state_summary)
        beat_rows.extend(state_beat_rows)

    fig, axes = plt.subplots(6, 2, figsize=(15, 16), sharex="col")
    saturation_argument = np.log((1 - BAROREFLEX_SATURATION_LIMIT) / BAROREFLEX_SATURATION_LIMIT)
    lower_f_ab_threshold = f_ab_min + BAROREFLEX_SATURATION_LIMIT * (f_ab_max - f_ab_min)
    upper_f_ab_threshold = f_ab_min + (1 - BAROREFLEX_SATURATION_LIMIT) * (f_ab_max - f_ab_min)

    for column, (state_name, window_indices, beat_boundaries) in enumerate(diagnostic_states):
        window_time = time[window_indices]
        relative_time = window_time - window_time[0]
        boundary_times = time[beat_boundaries] - window_time[0]

        axes[0, column].plot(relative_time, P_sa[window_indices], color="tab:blue", label="$P_{sa}$")
        axes[0, column].set_ylabel("$P_{sa}$ (mmHg)")
        axes[0, column].legend(loc="upper right")

        axes[1, column].plot(relative_time, dP_sa_dt[window_indices], color="tab:orange",
                             label="$dP_{sa}/dt$")
        axes[1, column].axhline(0, color="0.5", linewidth=0.8)
        axes[1, column].set_ylabel("$dP_{sa}/dt$\n(mmHg/s)")
        axes[1, column].legend(loc="upper right")

        axes[2, column].plot(relative_time, P_tilda[window_indices], color="tab:purple",
                             label="$\\widetilde{P}$")
        axes[2, column].plot(relative_time, P_n_current[window_indices], color="black", linestyle="--",
                             label="$P_n$")
        axes[2, column].set_ylabel("Pressure (mmHg)")
        axes[2, column].legend(loc="upper right")

        axes[3, column].plot(relative_time, sigmoid_argument[window_indices], color="tab:brown",
                             label="$(\\widetilde{P}-P_n)/k_{ab}$")
        axes[3, column].axhline(-saturation_argument, color="tab:red", linestyle="--",
                                label="5%/95% thresholds")
        axes[3, column].axhline(saturation_argument, color="tab:red", linestyle="--")
        axes[3, column].set_ylabel("Sigmoid argument")
        axes[3, column].legend(loc="upper right")

        axes[4, column].plot(relative_time, normalised_f_ab[window_indices], color="tab:green", label="$q$")
        axes[4, column].axhline(BAROREFLEX_SATURATION_LIMIT, color="tab:red", linestyle="--")
        axes[4, column].axhline(1 - BAROREFLEX_SATURATION_LIMIT, color="tab:red", linestyle="--")
        axes[4, column].set_ylim(-0.03, 1.03)
        axes[4, column].set_ylabel("Normalised $f_{ab}$, $q$")
        axes[4, column].legend(loc="upper right")

        saturated = ((normalised_f_ab[window_indices] <= BAROREFLEX_SATURATION_LIMIT) |
                     (normalised_f_ab[window_indices] >= 1 - BAROREFLEX_SATURATION_LIMIT))
        axes[5, column].plot(relative_time, f_ab[window_indices], color="tab:blue", label="$f_{ab}$")
        axes[5, column].fill_between(
            relative_time, f_ab_min, f_ab_max, where=saturated,
            color="tab:red", alpha=0.15, label="Saturated"
        )
        axes[5, column].axhline(f_ab_min, color="black", linestyle=":", label="$f_{ab}$ limits")
        axes[5, column].axhline(f_ab_max, color="black", linestyle=":")
        axes[5, column].axhline(lower_f_ab_threshold, color="tab:red", linestyle="--", linewidth=0.8)
        axes[5, column].axhline(upper_f_ab_threshold, color="tab:red", linestyle="--", linewidth=0.8)
        axes[5, column].set_ylabel("$f_{ab}$ (spikes/s)")
        axes[5, column].set_xlabel("Time from first displayed systolic peak (s)")
        axes[5, column].legend(loc="upper right")

        for row in range(6):
            axes[row, column].set_title(state_name if row == 0 else "")
            axes[row, column].grid(True, alpha=0.3)
            for boundary_time in boundary_times:
                axes[row, column].axvline(boundary_time, color="0.75", linewidth=0.7, zorder=0)

    summary_lines = []
    for summary in state_summaries:
        summary_lines.append(
            f"{summary['state']}: saturation={100 * summary['total_saturation_fraction']:.1f}% "
            f"(low {100 * summary['lower_saturation_fraction']:.1f}%, "
            f"high {100 * summary['upper_saturation_fraction']:.1f}%), "
            f"mean $f_{{ab}}$={summary['mean_f_ab_spikes_per_s']:.1f} spikes/s, "
            f"$P_{{sa}}$={summary['systolic_P_sa_mmHg']:.1f}/"
            f"{summary['diastolic_P_sa_mmHg']:.1f} mmHg, "
            f"max $dP_{{sa}}/dt$={summary['maximum_dP_sa_dt_mmHg_per_s']:.0f} mmHg/s"
        )

    fig.suptitle(
        f"Baroreflex diagnostics over the final {BAROREFLEX_DIAGNOSTIC_BEATS} complete beats "
        f"($\\tau_z$={tau_z:g} s, $\\tau_p$={tau_p:g} s, $\\tau_z/\\tau_p$={tau_z / tau_p:.2f})",
        fontsize=15
    )
    fig.text(0.5, 0.008, "\n".join(summary_lines), ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))

    output_directory = Path(__file__).resolve().parent / "baroreflex_diagnostics"
    output_directory.mkdir(parents=True, exist_ok=True)
    tau_z_label = f"{tau_z:g}".replace(".", "p")
    figure_path = output_directory / f"baroreflex_final_5_beats_tau_z_{tau_z_label}.png"
    metrics_path = output_directory / f"baroreflex_final_5_beats_tau_z_{tau_z_label}.csv"
    state_summary_path = output_directory / f"baroreflex_state_summary_tau_z_{tau_z_label}.csv"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metric_fields = list(beat_rows[0].keys())
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(beat_rows)

    with state_summary_path.open("w", newline="", encoding="utf-8") as state_summary_file:
        writer = csv.DictWriter(state_summary_file, fieldnames=list(state_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(state_summaries)

    print("\nBaroreflex diagnostic summary")
    print(f"tau_z={tau_z:g} s, tau_p={tau_p:g} s, tau_z/tau_p={tau_z / tau_p:.3f}")
    print(f"f_ab limits={f_ab_min:.3f} to {f_ab_max:.3f} spikes/s; saturation=q<=0.05 or q>=0.95")
    for summary in state_summaries:
        print(
            f"{summary['state']}: P_sa={summary['systolic_P_sa_mmHg']:.2f}/"
            f"{summary['diastolic_P_sa_mmHg']:.2f} mmHg; pulse pressure={summary['pulse_pressure_mmHg']:.2f} mmHg; "
            f"dP_sa/dt={summary['minimum_dP_sa_dt_mmHg_per_s']:.1f} to "
            f"{summary['maximum_dP_sa_dt_mmHg_per_s']:.1f} mmHg/s; "
            f"P_tilda={summary['minimum_P_tilda_mmHg']:.2f} to {summary['maximum_P_tilda_mmHg']:.2f} mmHg; "
            f"sigmoid argument={summary['minimum_sigmoid_argument']:.2f} to "
            f"{summary['maximum_sigmoid_argument']:.2f}; mean f_ab={summary['mean_f_ab_spikes_per_s']:.2f} spikes/s; "
            f"mean local gain={summary['mean_local_firing_gain_spikes_per_s_per_mmHg']:.3f} "
            f"spikes/s/mmHg; "
            f"saturation={100 * summary['total_saturation_fraction']:.1f}% "
            f"(low={100 * summary['lower_saturation_fraction']:.1f}%, "
            f"high={100 * summary['upper_saturation_fraction']:.1f}%)"
        )
    print(f"Saved diagnostic figure: {figure_path}")
    print(f"Saved per-beat metrics: {metrics_path}\n")
    return figure_path, metrics_path


def find_valve_edge_indices(valve_state, N):
    """Find an edge preceded by N samples in the opposite valve state."""
    cumulative = np.concatenate(([0], np.cumsum(valve_state)))
    previous_window = cumulative[N:-1] - cumulative[:-N - 1]
    return np.flatnonzero(valve_state[N:] & (previous_window == 0)) + N

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2):

    i = Initial_Conditions_dict["i"].item()
    actual_index = i % BUFFER_LIMIT

    all_time = Initial_Conditions_dict["all_time"]

    if i > 1: # t != 0:
        latest_nonzero_index = (i - 1) % BUFFER_LIMIT
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:
            # # num_removed = 6
            # index = -1  # Set a default value for safety
            #
            # # Iterating through the buffer in circular order
            # for j in range(BUFFER_LIMIT):
            #     logical_index = (latest_nonzero_index - j - 1) % BUFFER_LIMIT  # Traversing backwards
            #     if all_time[logical_index] < t:
            #         index = (logical_index + 1) % BUFFER_LIMIT
            #         break
            #
            # num_removed = (actual_index - index) if (actual_index - index) >= 0 else BUFFER_LIMIT + (actual_index - index)
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
    derivatives_all = model_derivatives(t, resp_contr_state, Initial_Conditions_dict, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2)

    all_time[(i - num_removed) % BUFFER_LIMIT] = t
    Initial_Conditions_dict["i"][0] = i - num_removed + 1
    Initial_Conditions_dict["j"][0] = Initial_Conditions_dict["j"].item() - num_removed + 1

    # Debugging check for progress
    if t != 0:
        diff = np.abs(t - target_values)
        if np.any(diff < 0.0001):
            print(t)

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
required_resp_control_keys = ["VE_integral"] #, "v_r", "x_r"]
IC_resp_contr = np.array([Initial_Conditions[key] for key in required_resp_control_keys], dtype=float)
num_resp_control = len(required_resp_control_keys)

IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_contr))
# IC_overall = np.concatenate((IC_cardio, IC_cardio_contr))
# IC_overall = IC_cardio


# def minimise_breathing(t1, t2, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao):
#     dt = 0.001 # must edit in Resp_Control_Breath_Optimiser too
#     bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
#     tolerance = 0.0001
#
#     VAflow_vals = np.linspace(0.01, 1.2, 200)
#     # VAflow_vals = np.repeat(VAflow_vals, 3)
#
#     VD = GV_dead * VAflow_vals + V0_dead
#
#     optimal_t1 = np.empty_like(VAflow_vals)
#     optimal_t2 = np.empty_like(VAflow_vals)
#     initial_guess = np.array([t1, t2], dtype=float)
#     required_params = [lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao]
#
#     for i, (VAflow, VD_volume) in enumerate(zip(VAflow_vals, VD)):
#         res = minimize(objective, x0= initial_guess,
#                        args=(required_params, VAflow, VD_volume, dt, tolerance), method='nelder-mead', bounds=bounds)
#
#         initial_guess = res.x
#         optimal_t1[i] = initial_guess[0]
#         optimal_t2[i] = initial_guess[1]
#
#     cs_t1 = CubicSpline(VAflow_vals, optimal_t1, bc_type="natural")
#     cs_t2 = CubicSpline(VAflow_vals, optimal_t2, bc_type="natural")
#
#     # t1_spline = cs_t1(VAflow_vals)
#     # t2_spline = cs_t2(VAflow_vals)
#     # plt.figure(figsize=(10, 5))
#     # # Spline fits
#     # plt.plot(VAflow_vals, t1_spline, label='t1 CubicSpline', color='blue', linewidth=2)
#     # plt.plot(VAflow_vals, t2_spline, label='t2 CubicSpline', color='red', linewidth=2)
#     # plt.scatter(VAflow_vals, t1_mean, color='blue', s=10, marker='o', label='t1 mean (knots)')
#     # plt.scatter(VAflow_vals, t2_mean, color='red', s=10, marker='o', label='t2 mean (knots)')
#     # plt.scatter(VAflow_clean, t1_clean, label='Optimal t1 (Inspiration Time)', color='blue', alpha=0.6, s=5)
#     # plt.scatter(VAflow_clean, t2_clean, label='Optimal t2 (Expiration Time)', color='red', alpha=0.6, s=5)
#     # plt.xlabel('VAflow (L/s)')
#     # plt.ylabel('Time (s)')
#     # plt.title('Optimal t1 and t2 vs VAflow Using nelder-mead')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.show()
#
#     return cs_t1.c, cs_t2.c, cs_t1.x, cs_t2.x


def minimise_breathing(t1, t2, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao):
    dt = 0.001 # must edit in Resp_Control_Breath_Optimiser too
    bounds = [(0.5, 3), (0.5, 6)]  # [t1, t2]
    tolerance = 0.0001

    VAflow_vals = np.linspace(0.01, 1.6, 200)
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


def simulate():
    Next_Conditions.clear()
    Next_Conditions.update(make_fresh_storage())
    # Initial setup
    IC_current = IC_overall.copy()

    (A_im, T_im, Tc, g_thor, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa,
    R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la,
    Emax_ra, P0_ra, KE_ra, C_sa, L_sa, R_sa, K1_vc, D1, Vvc_min, Kr_vc, Rvc_n,
    C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv,
    kr_am, P_0) = (
    new_params[k] if k in new_params else Parameters[k] for k in
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
    [new_params[k] if k in new_params else Parameters[k] for k in
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
     MRCO2, MRO2, s) = (new_params[k] if k in new_params else Parameters[k] for k in [
    "a2", "alpha2", "beta2", "C2", "K2", "PACO2_Delay_IC",
    "PAO2_Delay_IC", "P_atm", "P_ws", "Z", "dc", "KCCO2", "MRBCO2",
    "MO2_bp", "MRTCO2_basal", "MRTO2_basal", "MRCO2", "MRO2", "s"])

    # Resp control
    (GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao) = \
    (new_params[k] if k in new_params else Parameters[k] for k in ["GV_dead", "KcCO2", "KcMRV", "KpCO2", "KpO2",
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
    (new_params[k] if k in new_params else Parameters[k] for k in ["Kp_ao", "Kf_ao", "Kb_ao",
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

    # # determine the correct breathing profile
    cs_t1, cs_t2, knots_1, knots_2 = (minimise_breathing(1.5,
    1.85, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao))

    # data = {
    #     "cs_t1": cs_t1,
    #     "cs_t2": cs_t2,
    #     "knots_1": knots_1,
    #     "knots_2": knots_2,
    # }
    #
    # with open("breathing_splines.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # with open("breathing_splines.pkl", "rb") as f:
    #     data = pickle.load(f)
    #
    # cs_t1 = data["cs_t1"]
    # cs_t2 = data["cs_t2"]
    # knots_1 = data["knots_1"]
    # knots_2 = data["knots_2"]

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
        ODE_solution = solve_rk23(
            Next_Conditions,
            (0, max_time),
            IC_current,
            BUFFER_LIMIT,
            0.001,
            1e-3,
            1e-6,
            Input_Parameters, cs_t1, cs_t2, knots_1, knots_2,
        )
    else:
        ODE_solution = solve_ivp(
            combined_system,
            (0, max_time),
            IC_current,
            max_step=0.001,
            method="RK23",
            rtol=1e-3,
            atol=1e-6,
            args=(Next_Conditions, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2)
        )

    if ODE_solution.status == -1:
        print("ODE solver failed:", ODE_solution.message)
        return ODE_solution

    # Post-processing: use buffer to get recent data
    i_buffer = Next_Conditions["i"].item() % BUFFER_LIMIT

    all_time = np.concatenate((Next_Conditions["all_time"][i_buffer:], Next_Conditions["all_time"][:i_buffer]))
    time_since_beat_store = np.concatenate((Next_Conditions["time_since_beat_store"][i_buffer:], Next_Conditions["time_since_beat_store"][:i_buffer]))
    finish_breath_time = np.concatenate((Next_Conditions["finish_breath_time"][i_buffer:], Next_Conditions["finish_breath_time"][:i_buffer]))

    dtb = np.diff(time_since_beat_store)
    dtr = np.diff(finish_breath_time)
    cardiac_cycle_start_idx = np.where(dtb > 0)[0] + 1
    breath_cycle_start_idx = np.where(dtr > 0)[0] + 1
    beat_idx = cardiac_cycle_start_idx[-1]
    breath_idx = breath_cycle_start_idx[-1]
    last_beat_t = all_time[beat_idx]
    last_breath_t = all_time[breath_idx]

    interp = interp1d(
        ODE_solution.t,
        ODE_solution.y,
        axis=1,
        kind="linear",
        fill_value="extrapolate"
    )

    state_last_beat = interp(last_beat_t)
    state_last_breath = interp(last_breath_t)
    combined = np.concatenate((state_last_beat[:57], state_last_breath[57:]))
    print(combined)
    np.save("combined.npy", combined)

    theta_ao = np.concatenate((Next_Conditions["theta_ao_store"][i_buffer:], Next_Conditions["theta_ao_store"][:i_buffer]))
    theta_po = np.concatenate((Next_Conditions["theta_po_store"][i_buffer:], Next_Conditions["theta_po_store"][:i_buffer]))
    theta_mi = np.concatenate((Next_Conditions["theta_mi_store"][i_buffer:], Next_Conditions["theta_mi_store"][:i_buffer]))
    theta_tr = np.concatenate((Next_Conditions["theta_tr_store"][i_buffer:], Next_Conditions["theta_tr_store"][:i_buffer]))

    V_rv = np.concatenate((Next_Conditions["V_rv_store"][i_buffer:], Next_Conditions["V_rv_store"][:i_buffer]))
    V_ra = np.concatenate((Next_Conditions["V_ra_store"][i_buffer:], Next_Conditions["V_ra_store"][:i_buffer]))
    V_la = np.concatenate((Next_Conditions["V_la_store"][i_buffer:], Next_Conditions["V_la_store"][:i_buffer]))

    N = 50  # number of consecutive closed samples required

    is_open_ao = theta_ao > theta_min
    open_idx1 = find_valve_edge_indices(is_open_ao, N)

    is_closed_ao = theta_ao <= theta_min
    close_idx1 = find_valve_edge_indices(is_closed_ao, N)

    is_open_po = theta_po > theta_min
    open_idx2 = find_valve_edge_indices(is_open_po, N)

    is_closed_po = theta_po <= theta_min
    close_idx2 = find_valve_edge_indices(is_closed_po, N)

    is_open_mi = theta_mi > theta_min
    open_idx3 = find_valve_edge_indices(is_open_mi, N)

    is_closed_mi = theta_mi <= theta_min
    close_idx3 = find_valve_edge_indices(is_closed_mi, N)

    is_open_tr = theta_tr > theta_min
    open_idx4 = find_valve_edge_indices(is_open_tr, N)

    is_closed_tr = theta_tr <= theta_min
    close_idx4 = find_valve_edge_indices(is_closed_tr, N)

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

    # P_peri
    P_peri = np.concatenate((Next_Conditions["P_peri_store"][i_buffer:], Next_Conditions["P_peri_store"][:i_buffer]))
    P_peri_cycle_idx = cardiac_cycle_start_idx[-11:]
    P_peri_cycle_max_idx = np.array([b0 + np.argmax(P_peri[b0:b1]) for b0, b1 in zip(P_peri_cycle_idx[:-1], P_peri_cycle_idx[1:])])
    P_peri_cycle_max = P_peri[P_peri_cycle_max_idx]
    mean_max_P_peri = np.mean(P_peri_cycle_max)

    # Max pressure during atrial contraction takes the max p between phi_atr = 0 & 1
    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i_buffer:], Next_Conditions["phi_atr_store"][:i_buffer]))
    phi = np.concatenate((Next_Conditions["phi_store"][i_buffer:], Next_Conditions["phi_store"][:i_buffer]))

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
    P_sa = np.concatenate((Next_Conditions["P_sa_store"][i_buffer:], Next_Conditions["P_sa_store"][:i_buffer]))
    P_sa_max_idx = np.array([o + np.argmax(P_sa[o:c]) for o, c in pairs_ao])

    # Mean pulmonary artery pressure over the last 10 complete beats.
    last_10_cycle_idx = cardiac_cycle_start_idx[-11:]
    P_pa = np.concatenate((Next_Conditions["P_pa_store"][i_buffer:], Next_Conditions["P_pa_store"][:i_buffer]))
    mean_P_pa = np.mean([np.mean(P_pa[b0:b1]) for b0, b1 in zip(last_10_cycle_idx[:-1], last_10_cycle_idx[1:])])

    P_la = np.concatenate((Next_Conditions["P_la_store"][i_buffer:], Next_Conditions["P_la_store"][:i_buffer]))
    # max pressure at atrial contraction
    P_la_max_idx = np.array([s + np.argmax(P_la[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    # period of V descent when mitral valve is open -> get second min la P
    P_la_descent2_idx = np.array([o + np.argmin(P_la[o:c]) for o, c in pairs_mi])
    P_la_descent1_idx = np.array([c + np.argmin(P_la[c:o_next]) for (_, c), (o_next, _) in zip(pairs_mi[:-1], pairs_mi[1:])])

    P_ra = np.concatenate((Next_Conditions["P_ra_store"][i_buffer:], Next_Conditions["P_ra_store"][:i_buffer]))
    # max pressure at atrial contraction
    P_ra_max_idx = np.array([s + np.argmax(P_ra[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    # period of V descent when tricuspid valve is open -> get second min la P
    P_ra_descent2_idx = np.array([o + np.argmin(P_ra[o:c]) for o, c in pairs_tr])
    P_ra_descent1_idx = np.array([c + np.argmin(P_ra[c:o_next]) for (_, c), (o_next, _) in zip(pairs_tr[:-1], pairs_tr[1:])])

    V_lv = np.concatenate((Next_Conditions["V_lv_store"][i_buffer:], Next_Conditions["V_lv_store"][:i_buffer]))
    peaks, _ = find_peaks(V_lv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-V_lv, distance=int(500), prominence=1)

    last_10_troughs_V_lv = troughs[-10:]
    last_10_min_V_lv = V_lv[last_10_troughs_V_lv]

    last_10_peaks_V_lv = peaks[-10:]
    last_10_max_V_lv = V_lv[last_10_peaks_V_lv]

    P_rv = np.concatenate((Next_Conditions["P_rv_store"][i_buffer:], Next_Conditions["P_rv_store"][:i_buffer]))
    P_rv_max_idx = np.array([o + np.argmax(P_rv[o:c]) for o, c in pairs_po])
    # RVEDP: last sample before ventricular activation rises each beat.
    phi_eps = 1e-8
    phi_rise_idx = np.where((phi[:-1] <= phi_eps) & (phi[1:] > phi_eps))[0] + 1
    P_rv_edp_idx = phi_rise_idx[-10:] - 1

    HR = np.concatenate((Next_Conditions["HR_store"][i_buffer:], Next_Conditions["HR_store"][:i_buffer]))

    past_10_flat_segments = []
    prev_value = None
    for j in range(len(HR) - 1, -1, -1):
        current_value = HR[j]
        if current_value != prev_value:
            past_10_flat_segments.append(current_value)
            prev_value = current_value
            if len(past_10_flat_segments) == 10:
                break

    # Find transitions: where phi_atr goes from 0 to >0
    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1
    local_mins = starts[-10:]
    last_10_b4_LA_atrial_contract = V_la[local_mins]
    last_10_b4_RA_atrial_contract = V_ra[local_mins]

    # maximum ventricular pressure derivative
    is_active = phi_atr > 0.0  # atrial contraction window
    edges = np.diff(is_active.astype(int))

    start_idx = np.where(edges == 1)[0] + 1  # 0 → active
    end_idx = np.where(edges == -1)[0] + 1  # active → 0

    if len(start_idx) and len(end_idx) and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]

    n_pairs = min(len(start_idx), len(end_idx))
    start_idx = start_idx[:n_pairs]
    end_idx = end_idx[:n_pairs]

    dP_lv_dt_store = np.concatenate((Next_Conditions["dP_lv_dt_store"][i_buffer:], Next_Conditions["dP_lv_dt_store"][:i_buffer]))
    dP_lv_dt_idx = np.array([s + np.argmax(dP_lv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    dP_rv_dt_store = np.concatenate((Next_Conditions["dP_rv_dt_store"][i_buffer:], Next_Conditions["dP_rv_dt_store"][:i_buffer]))
    dP_rv_dt_idx = np.array([s + np.argmax(dP_rv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    tidal = np.concatenate((Next_Conditions["tidal_store"][i_buffer:], Next_Conditions["tidal_store"][:i_buffer]))

    breath_starts = np.where(dtr > 0)[0] + 1
    if breath_starts.size >= 2:
        max_tidal = np.max(tidal[breath_starts[-2]:breath_starts[-1]])
    else:
        max_tidal = np.max(tidal[tidal > 0]) if np.any(tidal > 0) else 0.0

    VAflow = np.concatenate((Next_Conditions["VAflow_store"][i_buffer:], Next_Conditions["VAflow_store"][:i_buffer]))
    t1 = np.concatenate((Next_Conditions["t1_store"][i_buffer:], Next_Conditions["t1_store"][:i_buffer]))
    t2 = np.concatenate((Next_Conditions["t2_store"][i_buffer:], Next_Conditions["t2_store"][:i_buffer]))
    VD = GV_dead * VAflow[-1] + V0_dead
    VDflow = (1 / (t1[-1] + t2[-1])) * VD
    Minute_Ventilation = (VAflow[-1] + VDflow) * 60

    Q_pp = np.concatenate((Next_Conditions["Q_pp_store"][i_buffer:], Next_Conditions["Q_pp_store"][:i_buffer]))
    Pa_O2_every = np.concatenate((Next_Conditions["Pa_O2_every_store"][i_buffer:], Next_Conditions["Pa_O2_every_store"][:i_buffer]))
    Pa_CO2_every = np.concatenate((Next_Conditions["Pa_CO2_every_store"][i_buffer:], Next_Conditions["Pa_CO2_every_store"][:i_buffer]))
    cardiac_output = np.mean([np.mean(Q_pp[b0:b1]) for b0, b1 in zip(last_10_cycle_idx[:-1], last_10_cycle_idx[1:])])
    Pa_O2 = np.mean([np.mean(Pa_O2_every[b0:b1]) for b0, b1 in zip(last_10_cycle_idx[:-1], last_10_cycle_idx[1:])])
    Pa_CO2 = np.mean([np.mean(Pa_CO2_every[b0:b1]) for b0, b1 in zip(last_10_cycle_idx[:-1], last_10_cycle_idx[1:])])

    Total_Volume = V_ra + V_rv + V_lv + V_la

    Total_Vol_min_idx = np.array([s + np.argmin(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]
    Total_Vol_max_idx = np.array([s + np.argmax(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    mean_min_Total_Volume = np.mean(Total_Volume[Total_Vol_min_idx])
    mean_max_Total_Volume = np.mean(Total_Volume[Total_Vol_max_idx])
    Pericardial_Volume_difference = mean_max_Total_Volume - mean_min_Total_Volume
    Vol_percentage_change = Pericardial_Volume_difference / mean_max_Total_Volume

    # LA_Contraction_Volume_diff = np.mean(last_10_b4_LA_atrial_contract) - np.mean(V_la[pairs_mi[:, 1]])
    # RA_Contraction_Volume_diff = np.mean(last_10_b4_RA_atrial_contract) - np.mean(V_ra[pairs_tr[:, 1]])

    # np.savez(f'HR_vs_time.npz', HR=Next_Conditions["HR_check"], time=Next_Conditions["time_history"], HR_average = Next_Conditions["HR"])
    print(np.mean(past_10_flat_segments), np.mean(P_sa[P_sa_max_idx]), np.mean(P_sa[open_idx1]),
          np.mean(V_lv[pairs_ao[:, 0]]), np.mean(V_lv[pairs_ao[:, 1]]), np.mean(V_rv[pairs_po[:, 0]]), np.mean(V_rv[pairs_po[:, 1]]),
          np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_edp_idx]),
          np.mean(V_ra[pairs_tr[:, 1]]), np.mean(V_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent1_idx]),
          np.mean(P_ra[P_ra_max_idx]), np.mean(P_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent2_idx]),
          np.mean(V_la[pairs_mi[:, 1]]), np.mean(V_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent1_idx]),
          np.mean(P_la[P_la_max_idx]), np.mean(P_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent2_idx]),
          np.mean(last_10_b4_LA_atrial_contract), np.mean(last_10_b4_RA_atrial_contract),
          np.mean(dP_lv_dt_store[dP_lv_dt_idx]), np.mean(dP_rv_dt_store[dP_rv_dt_idx]), max_tidal,
          Minute_Ventilation, cardiac_output, Pa_O2, Pa_CO2, Vol_percentage_change, mean_max_P_peri, mean_P_pa, sep=", ")


    return (ODE_solution, np.mean(past_10_flat_segments), np.mean(P_sa[P_sa_max_idx]), np.mean(P_sa[open_idx1]),
            np.mean(last_10_max_V_lv), np.mean(last_10_min_V_lv), np.mean(V_rv[pairs_po[:, 0]]),
            np.mean(V_rv[pairs_po[:, 1]]),
            np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_edp_idx]),
            mean_max_P_peri, IC_current, Next_Conditions, ODE_solution.t, ODE_solution.y)


if __name__ == "__main__":

    tau_z_override = get_float_option("--tau-z")
    if tau_z_override is not None:
        new_params["tau_z"] = tau_z_override
        print(f"Using command-line tau_z override: {tau_z_override:g} s")

    p_n_max_override = get_float_option("--p-n-max")
    if p_n_max_override is not None:
        new_params["P_n_max"] = p_n_max_override
        print(f"Using command-line P_n_max override: {p_n_max_override:g} mmHg")

    baroreflex_diagnostics_only = "--baroreflex-diagnostics-only" in sys.argv

    # A = np.load("IC_final.npy", allow_pickle=True)
    # B = np.load("Next_final.npy", allow_pickle=True).item()
    # # #
    # for key, value in B.items():
    #     if isinstance(value, np.ndarray):
    #         print(f"{key}: {value[-1]}")
    #     else:
    #         print(f"{key}: {value}")

    # # Pic
    # state_vars = [
    #     "VT_pa", "VT_pp", "VT_pv", "Q_pa",
    #     "VT_la", "VT_lv", "VT_ra", "VT_rv",
    #     "VT_sv", "VT_bv", "VT_hv", "VT_rmv", "VT_amv", "P_sp", "P_sa", "Q_sa", "VT_vc",
    #     "theta_ao", "dtheta_ao_dt", "theta_po", "dtheta_po_dt", "theta_mi", "dtheta_mi_dt", "theta_tr", "dtheta_tr_dt",
    #
    #     # Cardio controller state variables
    #     "theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv", "theta_change_O2_sh",
    #     "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", "R_ep_change", "R_sp_change",
    #     "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change", "Vu_rmv_change", "Vu_amv_change",
    #     "Emax_lv_change",
    #     "Emax_rv_change", "Ts_change", "Tv_change", "xb_O2", "xb_CO2", "xh_O2", "xh_CO2", "Wh", "xrm_O2", "xrm_CO2",
    #     "xam_O2", "xM", "x_met",
    #     "P_n_current",
    #
    #     # Gas exchange state variables
    #     "Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2", "Pd_5_O2",
    #     "Pd_5_CO2",
    #     "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2", "PCSFCO2", "MRTO2", "MRTCO2", "CTO2",
    #     "CvtCO2", "CBO2", "CvbCO2", "MRV",
    #
    #     # Resp control state variable
    #     "VE_integral"
    # ]
    # Initial_Conditions = dict(zip(state_vars, A))
    #
    # # Example output (pretty-printed)
    # import pprint
    #
    # pprint.pprint(Initial_Conditions)


    # lp = LineProfiler()
    # lp.add_function(Resp_Control_Breath_Optimiser.objective)
    #
    # lp.add_function(model_derivatives)
    # lp.enable()
    # solution1, HR1, Psys1, Pdia1, save_IC1, save_Next1, t_full1, y_full1 = simulate()
    solution, HR, Psys, Pdia, EDV, ESV, V_rv_max, V_rv_min, P_rv_max, P_rv_min, mean_P_peri, save_IC, save_Next, t_full, y_full = simulate()
    print("ODE Status:", solution.status)
    print("ODE Message:", solution.message)

    plot_baroreflex_diagnostics(solution)
    if baroreflex_diagnostics_only:
        raise SystemExit(0)

    # np.save(f'IC_final_resp.npy', solution.y[:, -1])  # individual chunks
    # np.save(f'Next_final_resp.npy', save_Next)  # individual chunks
    # lp.disable()
    # lp.print_stats()

    time = solution.t
    state_variables = solution.y

    state_variable_names = (
            required_cardio_keys +
            required_cardio_control_keys +
            required_gas_keys +
            required_resp_control_keys
    )

    index = np.where(Next_Conditions["time_history"] == 1e6)[0][0] - 1
    # print(HR)
    print(len(Next_Conditions["time_history"][:index]))

    variables_to_plot = [
        # "CvbCO2", "CvbO2", "VAflow", "V",
        "f_ab", # "P_sa", "VT_vc", #"PvtCO2", "dV_dt"
    ]

    for key in variables_to_plot:
        if key in Next_Conditions:  # Check if the key exists in updates
            plt.figure(figsize=(8, 4))  # Create a new figure for each variable
            plt.plot(Next_Conditions["time_history"][:index], Next_Conditions[key][:index], label=key, linewidth=2)
            plt.xlabel("Time (s)")
            plt.ylabel(key)
            plt.title(f"Plot of {key} over Time")
            plt.legend()
            plt.grid(True)
            plt.show()

    # plt.rcParams.update({
    #     "font.size": 15,  # Larger font
    #     # "font.weight": "bold",  # Bold text
    #     # "axes.labelweight": "bold",
    #     "axes.titlesize": 15,
    #     # "axes.titleweight": "bold",
    #     "legend.fontsize": 15,
    #     "lines.linewidth": 3.5,  # Thicker lines
    # })


    i = Next_Conditions["i"].item() % BUFFER_LIMIT
    sorted_times = np.concatenate((Next_Conditions["all_time"][i:], Next_Conditions["all_time"][:i]))

    # Number of state variables
    num_variables = state_variables.shape[0]
    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors

    # Plot all state variables
    plt.figure(figsize=(14, 10))

    for i, label in enumerate(required_gas_keys):
        if label in ["Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PCSFCO2", "MRTO2", "MRTCO2", "CTO2", "CvtCO2", "CBO2", "CvbCO2", "MRV"]:  # Skip "VT_sv"
            continue
        color = colors[
            i % len(colors)]  # Cycle through colors if there are more than 20 variables # Cycle through markers
        plt.plot(time, state_variables[len(required_cardio_keys + required_cardio_control_keys) + i], label=label,
                 color=color, linestyle='-', markersize=4)

    plt.xlabel("Time")
    plt.ylabel("State Variables")
    plt.title("Evolution of State Variables Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    plt.grid()
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pa_O2_art_plot"][:index], label="Pa_O2_art_plot",
             color="m")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["PA_O2"][:index],
             label="PA_O2", color="r")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["PA_CO2"][:index], label="Vagal firing", color="c")

    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pa_O2"][:index],
             label="Pa_O2", color="b")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Tv_change"][:index], label="Tv_change",
    #          color="k")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Tv"][:index], label="sigma_Tv", color="c")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ts_change"][:index], label="Ts_change",
    #          color='g')

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Firing rate (spikes/s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sh"][:index], label="Heart sympathetic", color="m")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sh_delay2"][:index], label="Delay Heart sympathetic", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_v"][:index], label="Vagal firing", color="c")

    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_v_delay02"][:index], label="Delay Vagal firing", color="b")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Tv_change"][:index], label="Tv_change", color="k")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Tv"][:index], label="sigma_Tv", color="c")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ts_change"][:index], label="Ts_change", color='g')

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Firing rate (spikes/s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    # # plt.show()
    ax2 = ax1.twinx()
    #
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ts_change"][:index], label="Ts_change", color='g')
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_sh"][:index], label="theta_sh", color="m")
    ax2.plot(Next_Conditions["time_history"][:index], 60*Next_Conditions["HR_check"][:index], label="HR", color="k")
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    #
    ax2.set_ylabel("Heart Rate (BPM)")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")
    plt.show()

    # theta_ao = np.concatenate((Next_Conditions["theta_ao_store"][i:], Next_Conditions["theta_ao_store"][:i]))
    # theta_po = np.concatenate((Next_Conditions["theta_po_store"][i:], Next_Conditions["theta_po_store"][:i]))
    #
    # theta_min = 0.12037493286132811
    # N = 50
    # is_open_ao = theta_ao > theta_min
    # open_idx1 = []
    # for k in range(N, len(theta_ao)):
    #     if is_open_ao[k] and not np.any(is_open_ao[k - N:k]):
    #         open_idx1.append(k)
    # open_idx1 = np.array(open_idx1)
    #
    # is_closed_ao = theta_ao <= theta_min
    # close_idx1 = []
    # for k in range(N, len(theta_ao)):
    #     if is_closed_ao[k] and not np.any(is_closed_ao[k - N:k]):
    #         close_idx1.append(k)
    # close_idx1 = np.array(close_idx1)
    #
    # is_open_po = theta_po > theta_min
    # open_idx2 = []
    # for k in range(N, len(theta_po)):
    #     if is_open_po[k] and not np.any(is_open_po[k - N:k]):
    #         open_idx2.append(k)
    # open_idx2 = np.array(open_idx2)
    #
    # is_closed_po = theta_po <= theta_min
    # close_idx2 = []
    # for k in range(N, len(theta_po)):
    #     if is_closed_po[k] and not np.any(is_closed_po[k - N:k]):
    #         close_idx2.append(k)
    # close_idx2 = np.array(close_idx2)
    #
    # pairs_ao = np.array([
    #     (o, close_idx1[(close_idx1 > o) & (close_idx1 < o_next)][-1])
    #     for o, o_next in zip(open_idx1[:-1], open_idx1[1:])
    #     if np.any((close_idx1 > o) & (close_idx1 < o_next))])
    #
    # pairs_po = np.array([
    #     (o, close_idx2[(close_idx2 > o) & (close_idx2 < o_next)][-1])
    #     for o, o_next in zip(open_idx2[:-1], open_idx2[1:])
    #     if np.any((close_idx2 > o) & (close_idx2 < o_next))])
    #
    # pairs_ao = pairs_ao[-10:]
    # pairs_po = pairs_po[-10:]
    #
    # V_lv = np.concatenate((Next_Conditions["V_lv_store"][i:], Next_Conditions["V_lv_store"][:i]))
    # # P_sa_max_idx = np.array([o + np.argmax(P_sa[o:c]) for o, c in pairs_ao])
    # # another = np.array([o + np.argmax(P_sa[o:c]) for o, c in pairs_po])
    #
    # print(open_idx1)
    # print(close_idx1)
    # print(open_idx2)
    # print(close_idx2)
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, V_lv, label="V_lv")
    # ax1.scatter(sorted_times[pairs_ao[:, 0]], V_lv[pairs_ao[:, 0]], color='r', marker='o', label="Detected Maxima AO")
    # ax1.scatter(sorted_times[pairs_ao[:, 1]], V_lv[pairs_ao[:, 1]], color='b', marker='o', label="Detected Minima AO")
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # ax2 = ax1.twinx()
    # # ax2.plot(sorted_times, theta_ao, color='b')
    # ax2.plot(sorted_times, theta_po, color='k')
    # ax2.legend(loc="upper right")
    # plt.show()

    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sp"][:index], label="Q_sp")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_ep"][:index], label="Q_ep")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sv"][:index], label="Q_sv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_hv"][:index], label="Q_hv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_rmv"][:index], label="Q_rmv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_amv"][:index], label="Q_amv")
    #
    # # Add labels and legend
    # plt.ylabel("Volume (mL)")
    # plt.xlabel("Time (s)")
    # plt.title("Traces")
    # plt.legend()
    # plt.show()

    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_ev"][:index], label="Extrasplanchnic V$_{Unstressed}$")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_amv"][:index], label="Active Muscle V$_{Unstressed}$")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_rmv"][:index], label="Resting Muscle V$_{Unstressed}$")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_sv"][:index], label="Splanchnic V$_{Unstressed}$")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_bv"][:index], label="Cerebral V$_{Unstressed}$")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["AA"][:index], label="Coronary V$_{Unstressed}$")
    #
    # # Add labels and legend
    # plt.ylabel("Volume (mL)")
    # plt.xlabel("Time (s)")
    # plt.title("Traces")
    # plt.legend()
    # plt.show()

    # fig, ax1 = plt.subplots()
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_la"][:index], label="P_la")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["AA"][:index], label="P_peri")
    # ax1.legend()
    #
    # ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv", color="y")
    # plt.show()




    # LV
    plt.plot(Next_Conditions["VT_lv"][index - 100000:index], Next_Conditions["P_lv"][index - 100000:index],
             label="LV", linewidth=2.5)
    plt.plot(Next_Conditions["VT_rv"][index - 20000:index], Next_Conditions["P_rv"][index - 20000:index],
             label="RV", linewidth=2.5)
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend()
    plt.show()

    # VAflow = np.concatenate((Next_Conditions["VAflow_store"][i:], Next_Conditions["VAflow_store"][:i]))
    # fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, VAflow)
    # ax1.set_xlabel("Time (s)")
    # ax1.legend(loc="upper left")
    # plt.show()

    i = Next_Conditions["i"].item() % BUFFER_LIMIT
    Q_pp = np.concatenate((Next_Conditions["Q_pp_store"][i:], Next_Conditions["Q_pp_store"][:i]))
    time_since_beat = np.concatenate((Next_Conditions["time_since_beat_store"][i:], Next_Conditions["time_since_beat_store"][:i]))

    dtsb = np.diff(time_since_beat)
    beat_idx = np.where(dtsb > 0)[0] + 1
    beat_idx = beat_idx[-11:]

    Q_pp_beat_avg = np.array([np.mean(Q_pp[b0:b1]) for b0, b1 in zip(beat_idx[:-1], beat_idx[1:])])
    beat_mid_idx = [(b0 + b1) // 2 for b0, b1 in zip(beat_idx[:-1], beat_idx[1:])]
    print(np.mean(Q_pp_beat_avg), np.mean(Q_pp))

    # fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, Q_pp, label="Q_pp")
    #
    # # scatter plot all 10 points
    # ax1.scatter(sorted_times[beat_mid_idx], Q_pp_beat_avg, color='g', marker='x', s=100, label="Last 10 flat values")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()

    HR = np.concatenate((Next_Conditions["HR_store"][i:], Next_Conditions["HR_store"][:i]))
    # print(max(HR), min(HR))
    #
    # past_10_flat_segments = []
    # prev_value = None
    # for j in range(len(HR) - 1, -1, -1):
    #     current_value = HR[j]
    #     if current_value != prev_value:
    #         # store both time and HR value as a tuple
    #         past_10_flat_segments.append((sorted_times[j], current_value))
    #         prev_value = current_value
    #         if len(past_10_flat_segments) == 10:
    #             break
    #
    # # unpack times and values
    # avg_times, avg_values = zip(*past_10_flat_segments)
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, HR, label="HR")
    #
    # # scatter plot all 10 points
    # ax1.scatter(avg_times, avg_values, color='g', marker='x', s=100, label="Last 10 flat values")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()

    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_amv"][:index], label="P_amv", color="r")
    # ax1.set_xlabel("Time (s)")
    # ax1.legend(loc="upper left")
    # ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["AA"][:index], label="VT_amv", color="b")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_amv"][:index], label="Vu_amv", color="g")
    # ax2.legend(loc="upper right")
    # plt.show()
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_amp"][:index], label="Q_amp", color="r")
    # ax1.set_xlabel("Time (s)")
    # ax1.legend(loc="upper left")
    # ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_amv"][:index], label="P_amv", color="b")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sp"][:index], label="P_sp", color="g")
    # ax2.legend(loc="upper right")
    # plt.show()

    # RA
    plt.plot(Next_Conditions["VT_ra"][index - 10000:index], Next_Conditions["P_ra"][index - 10000:index],
             label="RA")
    plt.plot(Next_Conditions["VT_la"][index - 10000:index], Next_Conditions["P_la"][index - 10000:index],
             label="LA")
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend()
    plt.show()

    P_sa = np.concatenate((Next_Conditions["P_sa_store"][i:], Next_Conditions["P_sa_store"][:i]))

    theta_ao = np.concatenate((Next_Conditions["theta_ao_store"][i:], Next_Conditions["theta_ao_store"][:i]))
    theta_po = np.concatenate((Next_Conditions["theta_po_store"][i:], Next_Conditions["theta_po_store"][:i]))
    theta_mi = np.concatenate((Next_Conditions["theta_mi_store"][i:], Next_Conditions["theta_mi_store"][:i]))
    theta_tr = np.concatenate((Next_Conditions["theta_tr_store"][i:], Next_Conditions["theta_tr_store"][:i]))
    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i:], Next_Conditions["phi_atr_store"][:i]))
    phi = np.concatenate((Next_Conditions["phi_store"][i:], Next_Conditions["phi_store"][:i]))

    N = 100  # number of consecutive closed samples required

    theta_min = new_params["theta_min"]
    is_open = theta_ao > theta_min
    open_idx1 = []
    for k in range(N, len(theta_ao)):
        if is_open[k] and not np.any(is_open[k - N:k]):
            open_idx1.append(k)
    open_idx1 = np.array(open_idx1)[-10:]

    is_closed_ao = theta_ao <= theta_min
    close_idx1 = []
    for k in range(N, len(theta_ao)):
        if is_closed_ao[k] and not np.any(is_closed_ao[k - N:k]):
            close_idx1.append(k)
    close_idx1 = np.array(close_idx1)[-10:]

    is_open_po = theta_po > theta_min
    open_idx2 = []
    for k in range(N, len(theta_po)):
        if is_open_po[k] and not np.any(is_open_po[k - N:k]):
            open_idx2.append(k)
    open_idx2 = np.array(open_idx2)[-10:]

    is_closed_po = theta_po <= theta_min
    close_idx2 = []
    for k in range(N, len(theta_po)):
        if is_closed_po[k] and not np.any(is_closed_po[k - N:k]):
            close_idx2.append(k)
    close_idx2 = np.array(close_idx2)[-10:]

    is_open_mi = theta_mi > theta_min
    open_idx3 = []
    for k in range(N, len(theta_mi)):
        if is_open_mi[k] and not np.any(is_open_mi[k - N:k]):
            open_idx3.append(k)
    open_idx3 = np.array(open_idx3)[-10:]

    is_closed_mi = theta_mi <= theta_min
    close_idx3 = []
    for k in range(N, len(theta_mi)):
        if is_closed_mi[k] and not np.any(is_closed_mi[k - N:k]):
            close_idx3.append(k)
    close_idx3 = np.array(close_idx3)[-10:]

    is_open_tr = theta_tr > theta_min
    open_idx4 = []
    for k in range(N, len(theta_tr)):
        if is_open_tr[k] and not np.any(is_open_tr[k - N:k]):
            open_idx4.append(k)
    open_idx4 = np.array(open_idx4)[-10:]
    print(open_idx4)

    is_closed_tr = theta_tr <= theta_min
    close_idx4 = []
    for k in range(N, len(theta_tr)):
        if is_closed_tr[k] and not np.any(is_closed_tr[k - N:k]):
            close_idx4.append(k)
    close_idx4 = np.array(close_idx4)[-10:]

    is_active = phi_atr > 0.0  # atrial contraction window
    edges = np.diff(is_active.astype(int))

    start_idx = np.where(edges == 1)[0] + 1  # 0 → active
    end_idx = np.where(edges == -1)[0] + 1  # active → 0

    if len(start_idx) and len(end_idx) and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]

    n_pairs = min(len(start_idx), len(end_idx))
    start_idx = start_idx[:n_pairs]
    end_idx = end_idx[:n_pairs]

    dP_rv_dt_store = np.concatenate((Next_Conditions["dP_rv_dt_store"][i:], Next_Conditions["dP_rv_dt_store"][:i]))
    dP_rv_dt_idx = np.array([s + np.argmax(dP_rv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]

    P_rv = np.concatenate((Next_Conditions["P_rv_store"][i:], Next_Conditions["P_rv_store"][:i]))
    all_time = np.concatenate((Next_Conditions["all_time"][i:], Next_Conditions["all_time"][:i]))


    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["Qi_rv"][index-80000:index], label="Qi_rv")
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["Q_ra"][index-80000:index], label="Q_ra")
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["VT_ra"][index-80000:index], label="VT_ra")
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["VT_la"][index-80000:index], label="VT_la")

    ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["phi_atr"][index-80000:index], label="phi_atr")
    ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["phi"][index-80000:index], label="phi")
    ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["theta_tr"][index - 80000:index],
             label="Tricuspid")  #
    ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["theta_po"][index - 80000:index],
             label="Pulmonary")  #

    # ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["x_r_store"][index - 80000:index], label="x_r")
    # ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["v_r_store"][index - 80000:index], label="v_r", color="grey")

    #
    # ax1.set_xlabel("Time (s)")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["P_ra"][index - 80000:index],
             label="P_ra", color="m")
    ax2.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["AA"][index - 80000:index],
             label="P_peri", color="k")
    ax2.legend(loc="upper right")
    plt.show()


    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["Qi_rv"][index-80000:index], label="Qi_rv")
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["Q_ra"][index-80000:index], label="Q_ra")
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["VT_ra"][index-80000:index], label="VT_ra")
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["VT_la"][index-80000:index], label="VT_la")

    ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["phi_atr"][index-80000:index], label="phi_atr")
    ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["phi"][index-80000:index], label="phi")
    ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["theta_tr"][index - 80000:index],
             label="Tricuspid")  #
    ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["theta_po"][index - 80000:index],
             label="Pulmonary")  #

    # ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["x_r_store"][index - 80000:index], label="x_r")
    # ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["v_r_store"][index - 80000:index], label="v_r", color="grey")

    #
    # ax1.set_xlabel("Time (s)")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["P_rv"][index - 80000:index],
             label="P_rv", color="y")
    ax2.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["P_pa"][index - 80000:index],
             label="P_pa", color="k")
    ax2.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["P_vc"][index - 80000:index],
             label="P_vc", color="c")
    ax2.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["P_ra"][index - 80000:index],
             label="P_ra", color="m")
    ax2.legend(loc="upper right")
    plt.show()



    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["Q_pp"][index - 80000:index], label="Q_pp")
    # ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["Qi_lv"][index - 80000:index], label="Qi_lv")
    # ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["Q_la"][index - 80000:index], label="Q_la")
    # ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["VT_la"][index - 80000:index], label="VT_la")
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["AA"][index-80000:index], label="P_peri")
    #
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["phi_atr"][index-80000:index], label="phi_atr", color="k")
    ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["theta_mi"][index - 80000:index], label="Mitral")  #
    ax1.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["theta_ao"][index - 80000:index], label="Aortic")  #

    # ax1.set_xlabel("Time (s)")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    # # ax2.plot(sorted_times, theta_ao, label="theta_ao", color="k")
    # ax2.plot(sorted_times, theta_mi, label="theta_mi", color="c")
    ax2.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["P_la"][index - 80000:index],
             label="P_la", color="m")
    # ax2.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["P_lv"][index-80000:index], label="P_lv", color="k")
    # ax2.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["P_pv"][index - 80000:index],
    #          label="P_pv", color="r")
    ax2.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["P_lv"][index-80000:index], label="P_lv", color="y")

    # ax2.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["phi_atr"][index-80000:index], label="phi_atr", color="c")
    #
    # ax2.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["VT_rv"][index-80000:index], label="VT_rv", color="y")
    #
    ax2.legend(loc="upper right")
    plt.show()

    # Total_Volume = Next_Conditions["AA"][index-80000:index]
    # time = Next_Conditions["time_history"][index-80000:index]
    #
    # Total_Vol_min_idx = np.array([s + np.argmin(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]
    # Total_Vol_max_idx = np.array([s + np.argmax(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-10:]
    #
    # mean_min_Total_Volume = np.mean(Total_Volume[Total_Vol_min_idx])
    # mean_max_Total_Volume = np.mean(Total_Volume[Total_Vol_max_idx])
    # Pericardial_Volume_difference = mean_max_Total_Volume - mean_min_Total_Volume
    # Vol_percentage_change = Pericardial_Volume_difference / mean_max_Total_Volume
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["AA"][index-80000:index])
    # ax1.scatter(time[Total_Vol_min_idx], Total_Volume[Total_Vol_min_idx], color='r', marker='x', label="Detected Minima")
    # ax1.scatter(time[Total_Vol_max_idx], Total_Volume[Total_Vol_max_idx], color='k', marker='x', label="Detected Maxima")
    # ax1.set_xlabel("Time (s)")
    # ax1.legend(loc="upper left")
    # ax2 = ax1.twinx()
    # ax2.plot(sorted_times, theta_ao, label="theta_ao", color="k")
    # ax2.plot(sorted_times, theta_mi, label="theta_mi", color="c")
    # ax2.plot(sorted_times, theta_po, label="theta_po", color="m")
    # ax2.plot(sorted_times, theta_tr, label="theta_tr", color="tomato")
    # ax2.plot(sorted_times, phi_atr, label="phi_atr", color="blue")
    # ax2.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["phi"][index-80000:index], label="phi", color="r")
    # ax2.legend(loc="upper right")
    # plt.show()

    # V_ra = np.concatenate((Next_Conditions["V_ra_store"][i:], Next_Conditions["V_ra_store"][:i]))
    # fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, V_ra, label="V_ra")
    #
    # ax1.scatter(sorted_times[open_idx4], V_ra[open_idx4], color='b', marker='o', label="Valve Detected Maxima")
    # ax1.scatter(sorted_times[close_idx4], V_ra[close_idx4], color='k', marker='o', label="Valve Detected Minima")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    #
    # ax2 = ax1.twinx()
    # ax2.plot(sorted_times, theta_ao, label="theta_ao", color="k")
    # ax2.plot(sorted_times, theta_mi, label="theta_mi", color="c")
    # ax2.plot(sorted_times, theta_po, label="theta_po", color="m")
    # ax2.plot(sorted_times, theta_tr, label="theta_tr", color="tomato")
    #
    # ax2.legend(loc="upper right")
    # #
    # plt.show()


    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i:], Next_Conditions["phi_atr_store"][:i]))
    phi = np.concatenate((Next_Conditions["phi_store"][i:], Next_Conditions["phi_store"][:i]))

    P_ra = np.concatenate((Next_Conditions["P_ra_store"][i:], Next_Conditions["P_ra_store"][:i]))
    dphi = np.diff(phi_atr, prepend=phi_atr[0])

    # plt.figure(figsize=(12, 6))
    # # LA pressure
    # plt.plot(sorted_times, phi_atr, label="phi_atr")
    # plt.plot(sorted_times, dphi, label="dphi")
    # plt.legend(loc="upper left")
    #
    # plt.show()

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

    P_ra_max_idx = np.array([s + np.argmax(P_ra[s:e]) for s, e in zip(start_idx, end_idx)])
    P_ra_max = P_ra[P_ra_max_idx][-10:]

    P_ra_valve_open = P_ra[open_idx4]

    # j = np.searchsorted(close_idx1, open_idx1, side="right")
    # valid = j < len(close_idx1)
    # pairs = np.column_stack([open_idx1[valid], close_idx1[j[valid]]])
    #
    # P_ra_descent1_idx = np.array([o + np.argmin(P_ra[o:c]) for o, c in pairs])[-10:]
    # P_ra_min_descent1 = P_ra[P_ra_descent1_idx]

    j = np.searchsorted(close_idx4, open_idx4, side="right")
    valid = j < len(close_idx4)
    pairs = np.array([
        (o, close_idx4[(close_idx4 > o) & (close_idx4 < o_next)][-1])
        for o, o_next in zip(open_idx4[:-1], open_idx4[1:])
        if np.any((close_idx4 > o) & (close_idx4 < o_next))])

    P_ra_descent2_idx = np.array([o + np.argmin(P_ra[o:c]) for o, c in pairs])[-10:]
    P_ra_min_descent2 = P_ra[P_ra_descent2_idx]
    P_ra_descent1_idx = np.array([c + np.argmin(P_ra[c:o_next]) for (_, c), (o_next, _) in zip(pairs[:-1], pairs[1:])])
    P_ra_min_descent1 = P_ra[P_ra_descent1_idx]

    # #
    # plt.figure(figsize=(12, 6))
    # # LA pressure
    # plt.plot(sorted_times, P_ra, label="P_ra", linewidth=2)
    # # Mark maxima during atrial upstroke
    # plt.scatter(sorted_times[P_ra_max_idx], P_ra[P_ra_max_idx], zorder=5, label="Max P_ra during φ_atr ↑")
    # plt.scatter(sorted_times[open_idx4], P_ra_valve_open, color='k', s=20, marker='o', label='Valve open')
    # plt.scatter(sorted_times[P_ra_descent2_idx], P_ra_min_descent2, color='c', s=20, marker='o', label='V descent')
    # plt.scatter(sorted_times[P_ra_descent1_idx], P_ra_min_descent1, color='r', s=20, marker='o', label='A descent')
    #
    # # Secondary axis for phi_atr
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # ax2.plot(sorted_times, phi_atr, linestyle="--", alpha=0.7, label="φ_atr")
    # ax2.plot(sorted_times, theta_po, label="theta_po", color="m")
    # ax2.plot(sorted_times, theta_tr, label="theta_tr", color="tomato")
    # ax2.set_ylabel("φ_atr")
    #
    # # Labels & legend
    # ax.set_xlabel("Time index")
    # ax.set_ylabel("P_ra")
    # ax.legend(loc="upper left")
    # ax2.legend(loc="upper right")
    #
    # plt.title("Right Atrial Pressure Peaks During Atrial Activation Upstroke")
    # plt.tight_layout()
    # plt.show()
    #
    # Flows
    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pa_CO2_art_plot"][:index],
             label="Pa_CO2_art_plot")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["PA_CO2"][:index],
             label="PA_CO2")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pa_CO2"][:index],
             label="Pa_CO2")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Flow (mL/s)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["VAflow"][:index],
             label="VAflow", color="k")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")

    plt.show()
    #
    #


    #
    # fig, ax1 = plt.subplots()

    cardiac_output = np.mean(Next_Conditions["Q_pp_store"])

    t1 = Next_Conditions["TI"][:index]
    Breath_time = 1/Next_Conditions["BF"][:index]
    t2 = Breath_time - t1
    # Minute_Ventilation = 60 * (Next_Conditions["VAflow"][index - 80000:index] + t1 / (t1 + t2))
    VD = new_params["GV_dead"] * Next_Conditions["VAflow"][:index] + new_params["V0_dead"]
    VDflow = (1 / (t1 + t2)) * VD
    Minute_Ventilation1 = (Next_Conditions["VAflow"][:index] + VDflow) * 60

    # VAflow = np.concatenate((Next_Conditions["VAflow_store"][i:], Next_Conditions["VAflow_store"][:i]))
    # t1 = np.concatenate((Next_Conditions["t1_store"][i:], Next_Conditions["t1_store"][:i]))
    # t2 = np.concatenate((Next_Conditions["t2_store"][i:], Next_Conditions["t2_store"][:i]))
    # VD = new_params["GV_dead"] * VAflow + new_params["V0_dead"]
    # VDflow = (1 / (t1 + t2)) * VD
    # Minute_Ventilation = (VAflow + VDflow) * 60

    fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, Minute_Ventilation1, label="Minute_Ventilation")
    ax1.plot(Next_Conditions["time_history"][:index], Minute_Ventilation1, label="Minute Ventilation all", color="r")

    # scatter plot all 10 points
    # ax1.scatter(sorted_times[-1], Minute_Ventilation[-1], color='g', marker='x', s=100, label="Last 10 flat values")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()




    tidal = np.concatenate((Next_Conditions["tidal_store"][i:], Next_Conditions["tidal_store"][:i]))
    peaks, _ = find_peaks(tidal, distance=int(1000))
    last_10_peaks_tidal = peaks[-1]
    last_10_max_tidal = tidal[last_10_peaks_tidal]

    Pa_O2 = np.mean(Next_Conditions["Pa_O2_every_store"])
    Pa_CO2 = np.mean(Next_Conditions["Pa_CO2_every_store"])


    # left atria

    V_la = np.concatenate((Next_Conditions["V_la_store"][i:], Next_Conditions["V_la_store"][:i]))
    peaks, _ = find_peaks(V_la, distance=int(1000), prominence=1)
    troughs, _ = find_peaks(-V_la, distance=int(1000), prominence=1)

    last_10_troughs_V_la = troughs[-10:]
    last_10_min_V_la = V_la[last_10_troughs_V_la]

    last_10_peaks_V_la = peaks[-10:]
    last_10_max_V_la = V_la[last_10_peaks_V_la]

    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i:], Next_Conditions["phi_atr_store"][:i]))
    phi = np.concatenate((Next_Conditions["phi_store"][i:], Next_Conditions["phi_store"][:i]))
    # Find transitions: where phi_atr goes from 0 to >0
    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1

    local_mins = starts[-10:]

    # fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, V_la, label="V_la")
    # ax1.scatter(sorted_times[troughs], V_la[troughs], color='r', marker='o',label="Atrial max volume during V-wave")
    # ax1.scatter(sorted_times[open_idx1], V_la[open_idx1], color='b', marker='o', label="theta_ao EDV")
    #
    # ax1.scatter(sorted_times[peaks], V_la[peaks], color='g', marker='x', label="Atrial ESV")
    # ax1.scatter(sorted_times[open_idx3], V_la[open_idx3], color='r', marker='x', label="theta_mi ESV")
    #
    # ax1.scatter(sorted_times[local_mins], V_la[local_mins], color='k', marker='o', label="Atrial EDV")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    #
    # ax2 = ax1.twinx()
    # ax2.plot(sorted_times, theta_ao, label="theta_ao", color="k")
    # ax2.plot(sorted_times, theta_mi, label="theta_mi", color="c")
    # ax2.plot(sorted_times, theta_po, label="theta_po", color="m")
    # ax2.plot(sorted_times, theta_tr, label="theta_tr", color="tomato")
    #
    #
    # ax2.legend(loc="upper right")
    #
    # plt.show()

    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i:], Next_Conditions["phi_atr_store"][:i]))
    phi = np.concatenate((Next_Conditions["phi_store"][i:], Next_Conditions["phi_store"][:i]))

    P_la = np.concatenate((Next_Conditions["P_la_store"][i:], Next_Conditions["P_la_store"][:i]))
    peaks, _ = find_peaks(P_la, distance=int(2000), prominence=1)
    troughs, _ = find_peaks(-P_la, distance=int(2000), prominence=1)

    last_10_troughs_P_la = troughs[-10:]
    last_10_min_P_la = P_la[last_10_troughs_P_la]

    last_10_peaks_P_la = peaks[-10:]
    last_10_max_P_la = P_la[last_10_peaks_P_la]

    # print(np.mean(last_10_min_P_la), np.mean(last_10_max_P_la))

    fig, ax1 = plt.subplots()
    ax1.plot(sorted_times, P_la, label="P_la")

    ax1.scatter(sorted_times[troughs], P_la[troughs], color='r', marker='o', label="Detected Minima")
    ax1.scatter(sorted_times[peaks], P_la[peaks], color='g', marker='x', label="Detected Maxima")
    ax1.plot(sorted_times, V_la, color='b')


    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(sorted_times, phi_atr, label="phi_atr", color="k")

    ax2.legend(loc="upper right")
    plt.show()

    # plt.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["R_ep"][index - 80000:index], label="Extrasplanchnic R$_{Peripheral}$")
    # plt.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["R_amp"][index - 80000:index], label="Active Muscle R$_{Peripheral}$")
    # plt.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["R_rmp"][index - 80000:index], label="Resting Muscle R$_{Peripheral}$")
    # plt.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["R_sp"][index - 80000:index], label="Splanchnic R$_{Peripheral}$")
    # plt.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["R_bp"][index - 80000:index], label="Brain R$_{Peripheral}$")
    # plt.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["R_hp"][index - 80000:index], label="Coronary R$_{Peripheral}$")
    #
    # # Add labels and legend
    # plt.ylabel("Resistance (mmHg·s/ml)")
    # plt.xlabel("Time (s)")
    # plt.title("Traces")
    # plt.legend()
    # plt.show()


    # LA
    fig, ax1 = plt.subplots()

    plt.plot(Next_Conditions["time_history"][index - 80000:index], Next_Conditions["V"][index - 80000:index], linewidth=2.5)
    plt.xlabel("Time (s)")
    plt.xticks(range(int(Next_Conditions["time_history"][index - 80000:index][0]), int(Next_Conditions["time_history"][index - 80000:index][-1]) + 1, 2))  # every 2 s
    plt.ylabel("Inspired/Expired Volume (L)")
    plt.legend()
    plt.show()

    plt.plot(Next_Conditions["time_history"][index - 8000:index], Next_Conditions["phi"][index - 8000:index], label="Ventricle Activation")
    plt.plot(Next_Conditions["time_history"][index - 8000:index], Next_Conditions["phi_atr"][index - 8000:index], label="Atrial Activation")
    # plt.plot(Next_Conditions["time_history"][index - 8000:index], 57.2958 * Next_Conditions["theta_tr"][index - 8000:index], label="Tricuspid")  #
    # plt.plot(Next_Conditions["time_history"][index - 8000:index], 57.2958 * Next_Conditions["theta_mi"][index - 8000:index], label="Mitral")  #
    # plt.plot(Next_Conditions["time_history"][index - 8000:index], 57.2958 * Next_Conditions["theta_ao"][index - 8000:index], label="Aortic")  #
    # plt.plot(Next_Conditions["time_history"][index - 8000:index], 57.2958 * Next_Conditions["theta_po"][index - 8000:index], label="Pulmonary")  #
    # plt.plot(Next_Conditions["time_history"][index - 8000:index], Next_Conditions["VT_ra"][index - 8000:index], label="VT_ra")  #

    # plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # plt.ylabel("Valve Angle (degrees)")
    # # plt.title("Pressure-Volume Traces")
    # # Rotate tick labels
    # # plt.xticks(rotation=45)
    # plt.xlabel("Time (s)")

    # Legend in upper left
    plt.legend(loc="upper right")
    # plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["Pa_O2"][index-80000:index], label="Arterial pO$_{2}$")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Nt"][:index], label="Nt")
    plt.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["PvtO2"][index-80000:index], label="Venous pO$_{2}$")
    plt.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["Pa_CO2"][index-80000:index], label="Arterial pCO$_{2}$")
    plt.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["PvtCO2"][index-80000:index], label="Venous pCO$_{2}$")
    plt.plot(Next_Conditions["time_history"][index-80000:index], Next_Conditions["Pb_CO2"][index-80000:index], label="Cerebral pCO$_{2}$")

    plt.xlabel("Time (s)")
    plt.ylabel("Partial Pressure (mmHg)")
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    plt.legend(loc="center left")
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ca_O2"][:index], label="Arterial c[O$_{2}$]", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ca_CO2"][:index], label="Arterial c[CO$_{2}$]", color="m")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Cv_O2"][:index], label="Venous c[O$_{2}$]", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Cv_CO2"][:index], label="Venous c[CO$_{2}$]", color="g")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["CvbO2"][:index], label="CvbO2", color="y")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Concentration (ml/ml blood)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="center left")
    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_ra"][:index], label="V_ra")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_rv"][:index], label="V_rv")

    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_la"][:index], label="V_la")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_lv"][:index], label="V_lv")

    ax1.set_xlabel("Time (s)")
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend()

    plt.show()


    # Number of state variables
    num_variables = state_variables.shape[0]
    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors

    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["CvtCO2"][:index], label="CvtCO2", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["CvbCO2"][:index], label="CvbCO2", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Cv_CO2"][:index], label="Cv_CO2", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["QT"][:index], label="QT", color="r")


    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()


    # Plot all state variables
    plt.figure(figsize=(14, 10))
    for i, label in enumerate(state_variable_names):
        if label != "P_tilda":
     #    if label in ["VT_pa", "VT_pp", "VT_pv", "Q_pa",
     #    "VT_la", "VT_lv", "VT_ra", "VT_rv",
     #    "VT_sv", "VT_bv", "VT_hv", "VT_rmv", "VT_amv", "P_sp", "P_sa", "Q_sa", 'VT_vc',
     #    "theta_ao", "dtheta_ao_dt", "theta_po", 'dtheta_po_dt', "theta_mi", 'dtheta_mi_dt', "theta_tr", 'dtheta_tr_dt',
     #
     # # Cardio controller state variables
     #    "theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv", "theta_change_O2_sh",
     #    "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", 'R_ep_change', "R_sp_change",
     #    "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change", "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change",
     #    "Emax_rv_change", "Ts_change", "Tv_change", 'xb_O2', "xb_CO2", "xh_O2", 'xh_CO2', "Wh", 'xrm_O2', 'xrm_CO2', 'xam_O2', "xM", "x_met", "P_n_current"]:  # Skip "Wh"
            continue
        color = colors[i % len(colors)]  # Cycle through colors if there are more than 20 variables # Cycle through markers
        plt.plot(solution.t, state_variables[i], label=label, color=color, linestyle='-', markersize=4)

    plt.xlabel("Time")
    plt.ylabel("State Variables")
    plt.title("Evolution of State Variables Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    plt.grid()
    plt.tight_layout()
    plt.show()
