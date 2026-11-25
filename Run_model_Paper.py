import pickle
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective
from scipy.signal import find_peaks, savgol_filter
# from line_profiler import LineProfiler
from All_derivatives import model_derivatives
from Entire_system.fixed_params import Parameters
from check import Parameters as new_params

from Initial_Conditions_after_running_again import Initial_Conditions
from Next_Conditions_all_derivatives import Next_Conditions


target_values = np.arange(0, 10000, 10)

time_saved = 0.005
BUFFER_LIMIT = 80000

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_CACHE_PATH = SCRIPT_DIR / "Run_model_Paper_cache.pkl"
LOAD_RUN_CACHE = True
# The 400 s cache was 3.8 GB; at max_time = 1150 expect ~11 GB on disk, and the write
# competes for RAM with solve_ivp's own accumulated trajectory.
SAVE_RUN_CACHE = True
REFRESH_RUN_CACHE = False  # Set True when model parameters/ICs changed and the ODE must rerun.
SIMULATION_RESULT_NAMES = (
    "solution", "HR", "Psys", "Pdia", "EDV", "ESV", "V_rv_max", "V_rv_min",
    "P_rv_max", "P_rv_min", "mean_P_peri", "save_IC", "save_Next", "t_full", "y_full",
)
PAPER_PLOT_RC = {
    "figure.facecolor": "white",
    "figure.figsize": (6, 5.0),
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#4d4d4d",
    "xtick.color": "#4d4d4d",
    "ytick.color": "#4d4d4d",
    "font.size": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "mathtext.fontset": "dejavusans",
}
PAPER_COLORS = {
    "red": "#E89A9A",
    "pink": "#f2a6a6",
    "orange": "#E8C88F",
    "peach": "#ffbe7d",
    "green": "#88C87E",
    "teal": "#7DB6C0",
    "blue": "#7F9ED6",
    "light_blue": "#a6c8e6",
    "purple": "#BBA3D6",
    "lavender": "#c7a9e3",
    "brown": "#6b5748",
    "tan": "#c0a080",
}

min_time = 10 # Minimum time in seconds before checking
max_time = 1630 # Maximum time limit to avoid infinite loops
time_step = 200  # Chunk size per solve

PLOT_TRIM_SECONDS = 40.0
REGULATION_PANEL_OUTPUT = SCRIPT_DIR / "Run_model_Paper_regulation_six_panel.png"


def style_paper_legend(legend):
    if legend is None:
        return None

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#dddddd")
    legend.get_frame().set_linewidth(1.0)
    return legend


def apply_paper_axis_style(ax, *, show_legend=True, legend_kwargs=None):
    ax.set_facecolor("white")
    ax.grid(False)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#707070")
        ax.spines[side].set_linewidth(1.4)

    ax.tick_params(axis="both", colors="#4d4d4d", labelsize=13, width=1.2, length=4)
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")

    if not show_legend:
        return None

    kwargs = {"loc": "best", "frameon": True}
    if legend_kwargs:
        kwargs.update(legend_kwargs)
    return style_paper_legend(ax.legend(**kwargs))


def _binned_band(ax, t, y, label, color, nbins=1600, lw=2.0):
    """Draw a trace as a per-bin min-max band with its mean on top.

    These traces carry ~4.5M samples over 1150 s. Drawn raw, a pulsatile signal
    renders as a solid block. Binning in time keeps the range of the reference
    figure and puts the mean trajectory back on top of it.

    `t` must be ascending, which time_history is.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    if t.size == 0:
        return
    edges = np.linspace(t[0], t[-1], nbins + 1)
    starts = np.searchsorted(t, edges[:-1], side="left")
    ends = np.searchsorted(t, edges[1:], side="left")
    keep = ends > starts
    if not np.any(keep):
        ax.plot(t, y, label=label, color=color, linewidth=lw)
        return
    idx = starts[keep]
    counts = np.diff(np.append(idx, t.size))
    centres = (0.5 * (edges[:-1] + edges[1:]))[keep]
    lo = np.minimum.reduceat(y, idx)
    hi = np.maximum.reduceat(y, idx)
    mean = np.add.reduceat(y, idx) / counts

    # Only shade where the signal actually oscillates within a bin; for a smooth
    # trace the band collapses onto the mean line and adds nothing.
    if np.nanmax(hi - lo) > 1e-9:
        ax.fill_between(centres, lo, hi, color=color, alpha=0.28, linewidth=0.0)
    ax.plot(centres, mean, label=label, color=color, linewidth=lw)


def _beat_averaged_trace(ax, t, y, ventricular_activation, label, color, lw=2.0):
    """Draw one time-weighted mean per complete cardiac cycle.

    A new beat starts when ventricular activation rises from zero. Integrating
    over each interval avoids bias from the solver's irregular evaluation times.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    ventricular_activation = np.asarray(ventricular_activation)
    if t.size < 2:
        return

    activation_eps = 1e-8
    beat_starts = np.flatnonzero(
        (ventricular_activation[:-1] <= activation_eps)
        & (ventricular_activation[1:] > activation_eps)
    ) + 1
    if beat_starts.size < 2:
        ax.plot(t, y, label=label, color=color, linewidth=lw)
        return

    beat_times = np.empty(beat_starts.size - 1)
    beat_means = np.empty(beat_starts.size - 1)
    for j, (start, end) in enumerate(zip(beat_starts[:-1], beat_starts[1:])):
        cycle_t = t[start:end + 1]
        cycle_y = y[start:end + 1]
        duration = cycle_t[-1] - cycle_t[0]
        beat_times[j] = 0.5 * (cycle_t[0] + cycle_t[-1])
        beat_means[j] = np.sum(
            0.5 * (cycle_y[:-1] + cycle_y[1:]) * np.diff(cycle_t)
        ) / duration

    ax.plot(beat_times, beat_means, label=label, color=color, linewidth=lw, zorder=4)


def _raw_trace(ax, t, y, label, color, max_points=200000, lw=2.0, alpha=1.0, zorder=None):
    """Draw instantaneous signal values without temporal averaging.

    By default, a bounded stride keeps rendering practical while retaining
    actual samples rather than replacing them with per-bin means. Set
    ``max_points=None`` when every stored sample should be drawn.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    if t.size == 0:
        return

    if max_points is None:
        plot_t = t
        plot_y = y
    else:
        step = max(1, int(np.ceil(t.size / max_points)))
        sample_idx = np.arange(0, t.size, step)
        if sample_idx[-1] != t.size - 1:
            sample_idx = np.append(sample_idx, t.size - 1)
        plot_t = t[sample_idx]
        plot_y = y[sample_idx]
    ax.plot(
        plot_t,
        plot_y,
        label=label,
        color=color,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )


def _savefig_unlocked(fig, output_path):
    """Save `fig` as PNG and PDF, falling back if the PNG target is locked.

    On Windows an image viewer holding the PNG open (PyCharm's preview tab,
    Photos) memory-maps it, and the write fails with OSError 22 rather than a
    plain permission error. Losing a figure to that after the whole run is
    wasteful, so retry under a suffixed name instead of raising.
    """
    output_path = Path(output_path)
    pdf_path = output_path.with_suffix(".pdf")
    try:
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
        fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
        print(f"Wrote PDF: {pdf_path}")
        return output_path
    except OSError as exc:
        for n in range(1, 100):
            alt = output_path.with_name(f"{output_path.stem}_{n}{output_path.suffix}")
            if alt.exists():
                continue
            alt_pdf = alt.with_suffix(".pdf")
            fig.savefig(alt, dpi=300, facecolor="white", bbox_inches="tight")
            fig.savefig(alt_pdf, facecolor="white", bbox_inches="tight")
            print(f"{output_path.name} was locked ({exc}); wrote {alt.name} instead.")
            print(f"Wrote PDF: {alt_pdf}")
            return alt
        raise


def _panel_letter(ax, letter):
    ax.text(
        -0.10, 1.05, letter,
        transform=ax.transAxes,
        fontsize=19, fontweight="bold", color="#333333",
        ha="left", va="bottom",
    )


def write_regulation_six_panel(Next_Conditions, full_time, plot_time, plot_slice, index, output_path):
    """The regulation figure: afferents, efferents and the four effector groups.

    These six panels used to be six separate `plt.subplots()` blocks that were
    uncommented one at a time and saved by hand. Same traces, same PAPER_COLORS,
    now drawn onto one 2x3 canvas and written to `output_path`.

    Panels A, B and F run from t = 0 so the start-up transient is visible; C, D
    and E use `plot_time`, which drops the first PLOT_TRIM_SECONDS.
    """
    plt.rcParams.update(PAPER_PLOT_RC)
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9.6))

    # A - afferent inputs
    ax = axes[0, 0]
    _raw_trace(
        ax,
        full_time,
        Next_Conditions["f_ab"][:index],
        "_nolegend_",
        PAPER_COLORS["red"],
        max_points=None,
        lw=0.6,
        alpha=0.28,
        zorder=1,
    )
    _beat_averaged_trace(
        ax,
        full_time,
        Next_Conditions["f_ab"][:index],
        Next_Conditions["phi"][:index],
        "Baroreceptor firing (beat average)",
        PAPER_COLORS["red"],
    )
    _binned_band(ax, full_time, Next_Conditions["f_ap"][:index], "Lung stretch receptor firing", PAPER_COLORS["purple"])
    _binned_band(ax, full_time, Next_Conditions["f_ac"][:index], "Chemoreceptor firing", PAPER_COLORS["teal"])
    _raw_trace(ax, full_time, Next_Conditions["Nt"][:index], "Respiratory neuromuscular drive", PAPER_COLORS["blue"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (spikes/s)")
    # Pinned just below the top-right corner so the peak of the baroreceptor
    # band stays visible behind the legend box.
    apply_paper_axis_style(
        ax,
        legend_kwargs={"loc": "upper right", "bbox_to_anchor": (1.0, 0.88)},
    )
    _panel_letter(ax, "A")

    # B - efferent outputs
    ax = axes[0, 1]
    _binned_band(ax, full_time, Next_Conditions["f_sv"][:index], "Venous Volume\nSympathetic Activity", PAPER_COLORS["red"])
    _binned_band(ax, full_time, Next_Conditions["f_sh"][:index], "HR & Contractility\nSympathetic Activity", PAPER_COLORS["purple"])
    _binned_band(ax, full_time, Next_Conditions["f_sp"][:index], "Peripheral Resistance\nSympathetic Activity", PAPER_COLORS["teal"])
    _binned_band(ax, full_time, Next_Conditions["f_v"][:index], "Vagal Activity", PAPER_COLORS["blue"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (spikes/s)")
    apply_paper_axis_style(ax)
    _panel_letter(ax, "B")

    # C - stressed volumes
    ax = axes[0, 2]
    _binned_band(ax, plot_time, Next_Conditions["V_sv"][plot_slice], "Splanchnic V$_{Stressed}$", PAPER_COLORS["red"])
    _binned_band(ax, plot_time, Next_Conditions["V_ev"][plot_slice], "Extrasplanchnic V$_{Stressed}$", PAPER_COLORS["purple"])
    _binned_band(ax, plot_time, Next_Conditions["V_amv"][plot_slice], "Active Muscle V$_{Stressed}$", PAPER_COLORS["teal"])
    _binned_band(ax, plot_time, Next_Conditions["V_rmv"][plot_slice], "Resting V$_{Stressed}$", PAPER_COLORS["blue"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Volume (mL)")
    apply_paper_axis_style(ax)
    _panel_letter(ax, "C")

    # D - peripheral resistances
    ax = axes[1, 0]
    _binned_band(ax, plot_time, Next_Conditions["R_hp"][plot_slice], "Coronary R$_{Peripheral}$", PAPER_COLORS["red"])
    _binned_band(ax, plot_time, Next_Conditions["R_rmp"][plot_slice], "Resting Muscle R$_{Peripheral}$", PAPER_COLORS["orange"])
    _binned_band(ax, plot_time, Next_Conditions["R_bp"][plot_slice], "Brain R$_{Peripheral}$", PAPER_COLORS["green"])
    _binned_band(ax, plot_time, Next_Conditions["R_sp"][plot_slice], "Splanchnic R$_{Peripheral}$", PAPER_COLORS["teal"])
    _binned_band(ax, plot_time, Next_Conditions["R_amp"][plot_slice], "Active Muscle R$_{Peripheral}$", PAPER_COLORS["blue"])
    _binned_band(ax, plot_time, Next_Conditions["R_ep"][plot_slice], "Extrasplanchnic R$_{Peripheral}$", PAPER_COLORS["purple"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Resistance (mmHg$\cdot$s/ml)")
    apply_paper_axis_style(ax)
    _panel_letter(ax, "D")

    # E - heart rate and max elastances (twin axis)
    ax = axes[1, 1]
    _raw_trace(ax, plot_time, 60 * Next_Conditions["HR"][plot_slice], "Heart Rate", PAPER_COLORS["red"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (bpm)")
    apply_paper_axis_style(ax, show_legend=False)
    _panel_letter(ax, "E")

    ax2 = ax.twinx()
    _binned_band(ax2, plot_time, Next_Conditions["Emax_rv"][plot_slice], "RV Max Elastance", PAPER_COLORS["purple"])
    _binned_band(ax2, plot_time, Next_Conditions["Emax_lv"][plot_slice], "LV Max Elastance", PAPER_COLORS["teal"])
    ax2.set_ylabel("Elastance (mmHg/ml)")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#707070")
    ax2.spines["right"].set_linewidth(1.4)
    ax2.tick_params(axis="y", colors="#4d4d4d", labelsize=13, width=1.2, length=4)
    ax2.yaxis.label.set_color("#333333")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    style_paper_legend(
        ax.legend(handles1 + handles2, labels1 + labels2, loc="best", frameon=True)
    )

    # F - alveolar ventilation
    ax = axes[1, 2]
    _binned_band(ax, full_time, 60 * Next_Conditions["VAflow"][:index], "Alveolar Ventilation", PAPER_COLORS["red"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ventilation (L/min)")
    apply_paper_axis_style(ax)
    _panel_letter(ax, "F")

    fig.tight_layout(w_pad=2.6, h_pad=2.6)
    output_path = _savefig_unlocked(fig, output_path)
    plt.close(fig)
    print(f"Wrote: {output_path}")
    return output_path

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
    bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
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
    open_idx1 = []
    for k in range(N, len(theta_ao)):
        if is_open_ao[k] and not np.any(is_open_ao[k - N:k]):
            open_idx1.append(k)
    open_idx1 = np.array(open_idx1)

    is_closed_ao = theta_ao <= theta_min
    close_idx1 = []
    for k in range(N, len(theta_ao)):
        if is_closed_ao[k] and not np.any(is_closed_ao[k - N:k]):
            close_idx1.append(k)
    close_idx1 = np.array(close_idx1)

    is_open_po = theta_po > theta_min
    open_idx2 = []
    for k in range(N, len(theta_po)):
        if is_open_po[k] and not np.any(is_open_po[k - N:k]):
            open_idx2.append(k)
    open_idx2 = np.array(open_idx2)

    is_closed_po = theta_po <= theta_min
    close_idx2 = []
    for k in range(N, len(theta_po)):
        if is_closed_po[k] and not np.any(is_closed_po[k - N:k]):
            close_idx2.append(k)
    close_idx2 = np.array(close_idx2)

    is_open_mi = theta_mi > theta_min
    open_idx3 = []
    for k in range(N, len(theta_mi)):
        if is_open_mi[k] and not np.any(is_open_mi[k - N:k]):
            open_idx3.append(k)
    open_idx3 = np.array(open_idx3)

    is_closed_mi = theta_mi <= theta_min
    close_idx3 = []
    for k in range(N, len(theta_mi)):
        if is_closed_mi[k] and not np.any(is_closed_mi[k - N:k]):
            close_idx3.append(k)
    close_idx3 = np.array(close_idx3)

    is_open_tr = theta_tr > theta_min
    open_idx4 = []
    for k in range(N, len(theta_tr)):
        if is_open_tr[k] and not np.any(is_open_tr[k - N:k]):
            open_idx4.append(k)
    open_idx4 = np.array(open_idx4)

    is_closed_tr = theta_tr <= theta_min
    close_idx4 = []
    for k in range(N, len(theta_tr)):
        if is_closed_tr[k] and not np.any(is_closed_tr[k - N:k]):
            close_idx4.append(k)
    close_idx4 = np.array(close_idx4)

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


def save_run_cache(results, cache_path=RUN_CACHE_PATH):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "cache_version": 1,
        "result_names": SIMULATION_RESULT_NAMES,
        "results": dict(zip(SIMULATION_RESULT_NAMES, results)),
    }
    with cache_path.open("wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved simulation cache: {cache_path}")


def load_run_cache(cache_path=RUN_CACHE_PATH):
    with cache_path.open("rb") as fh:
        cache = pickle.load(fh)

    results = cache.get("results", {})
    missing = [name for name in SIMULATION_RESULT_NAMES if name not in results]
    if missing:
        raise KeyError(f"Cache is missing keys: {', '.join(missing)}")

    print(f"Loaded simulation cache: {cache_path}")
    return tuple(results[name] for name in SIMULATION_RESULT_NAMES)


def run_or_load_simulation():
    if LOAD_RUN_CACHE and not REFRESH_RUN_CACHE and RUN_CACHE_PATH.exists():
        try:
            return load_run_cache()
        except (OSError, EOFError, pickle.UnpicklingError, KeyError) as exc:
            print(f"Could not load simulation cache ({exc}); rerunning model.")

    results = simulate()
    if SAVE_RUN_CACHE:
        save_run_cache(results)
    return results


if __name__ == "__main__":

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
    solution, HR, Psys, Pdia, EDV, ESV, V_rv_max, V_rv_min, P_rv_max, P_rv_min, mean_P_peri, save_IC, save_Next, t_full, y_full = run_or_load_simulation()
    Next_Conditions = save_Next
    print("ODE Status:", solution.status)
    print("ODE Message:", solution.message)

    # np.save(f'IC_final_resp.npy', solution.y[:, -1])  # individual chunks
    # np.save(f'Next_final_resp.npy', save_Next)  # individual chunks
    # lp.disable()
    # lp.print_stats()

    index = np.where(Next_Conditions["time_history"] == 1e6)[0][0] - 1
    # print(HR)
    print(len(Next_Conditions["time_history"][:index]))

    plt.rcParams.update(PAPER_PLOT_RC)
    full_time = Next_Conditions["time_history"][:index]
    trimmed = np.flatnonzero(full_time >= PLOT_TRIM_SECONDS)
    plot_start = int(trimmed[0]) if trimmed.size else 0
    plot_slice = slice(plot_start, index)
    plot_time = Next_Conditions["time_history"][plot_slice]

    # variables_to_plot = [
    #     # "CvbCO2", "CvbO2", "VAflow", "V",
    #     "AA", # "P_sa", "VT_vc", #"PvtCO2", "dV_dt"
    # ]
    #
    # for key in variables_to_plot:
    #     if key in Next_Conditions:  # Check if the key exists in updates
    #         plt.figure(figsize=(8, 4))  # Create a new figure for each variable
    #         plt.plot(Next_Conditions["time_history"][:index], Next_Conditions[key][:index], label=key, linewidth=2)
    #         plt.xlabel("Time (s)")
    #         plt.ylabel(key)
    #         plt.title(f"Plot of {key} over Time")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()

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

    write_regulation_six_panel(
        Next_Conditions,
        full_time,
        plot_time,
        plot_slice,
        index,
        REGULATION_PANEL_OUTPUT,
    )
