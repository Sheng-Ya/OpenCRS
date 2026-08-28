import argparse
import csv
import re
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "MCMC_Union_50_28_08_copula_prior" # change
DEFAULT_OUTPUT_NAME = "refined_map_vs_targets_with_actual_simulation_28_08.png"

# The union simulator returns rest raw outputs followed by exercise raw outputs.
# Each state has 31 raw outputs, and the history-matching emulator drops these
# six non-calibration columns per state to get the 25 displayed targets.
RESULT_COLS_TO_DROP_PER_STATE = (11, 14, 17, 20, 27, 30)
RAW_OUTPUTS_PER_STATE = 31
ALIGNED_OUTPUTS_PER_STATE = RAW_OUTPUTS_PER_STATE - len(RESULT_COLS_TO_DROP_PER_STATE)

AUTO_ACTUAL_FILENAMES = (
    "Rest_exercise_refined_simulation_HM_28-08.txt",
)

BASE_DISPLAY_LABELS = {
    "Heart_Rate": r"$\overline{\mathrm{HR}}$",
    "Systolic_Pressure": r"$P_{\mathrm{sys,LV}}$",
    "Diastolic_Pressure": r"$P_{\mathrm{dia,LV}}$",
    "EDV": r"$V_{\mathrm{ED,LV}}$",
    "ESV": r"$V_{\mathrm{ES,LV}}$",
    "Max_RV_Volume": r"$V_{\mathrm{ED,RV}}$",
    "Min_RV_Volume": r"$V_{\mathrm{ES,RV}}$",
    "Max_RV_Pressure": r"$P_{\mathrm{sys,RV}}$",
    "Min_RV_Pressure": r"$P_{\mathrm{dia,RV}}$",
    "Min_RA_Volume": r"$V_{\min,\mathrm{RA}}$",
    "Max_RA_Volume": r"$V_{\max,\mathrm{RA}}$",
    "Max_RA_Pressure_Atrial_contraction": r"$P_{\max,A,\mathrm{RA}}$",
    "Max_RA_Pressure_Tricuspid_Opening": r"$P_{\max,V,\mathrm{RA}}$",
    "Min_LA_Volume": r"$V_{\min,\mathrm{LA}}$",
    "Max_LA_Volume": r"$V_{\max,\mathrm{LA}}$",
    "Max_LA_Pressure_Atrial_contraction": r"$P_{\max,A,\mathrm{LA}}$",
    "Max_LA_Pressure_Mitral_Opening": r"$P_{\max,V,\mathrm{LA}}$",
    "Pre_LA_Contraction_Volume": r"$V_{\mathrm{pre-A,LA}}$",
    "Pre_RA_Contraction_Volume": r"$V_{\mathrm{pre-A,RA}}$",
    "LV_Pressure_Deriv": r"$\max\,\mathrm{d}P_{\mathrm{LV}}/\mathrm{d}t$",
    "RV_Pressure_Deriv": r"$\max\,\mathrm{d}P_{\mathrm{RV}}/\mathrm{d}t$",
    "Tidal_Volume": r"$V_T$",
    "Minute_Ventilation": r"$\dot{V}_E$",
    "PaO2": r"$\overline{P_{\mathrm{a}O_2}}$",
    "PaCO2": r"$\overline{P_{\mathrm{a}CO_2}}$",
    "Pericardial_Pressure": r"$P_{\mathrm{peri}}$",
}

TARGET_LINE_COLOR = "#4A4A4A"
TARGET_BAND_COLOR = "#4A4A4A"
POSTERIOR_BOX_COLOR = "#7DB6C0"
POSTERIOR_EDGE_COLOR = "#5D9FA8"
SAMPLED_MAP_COLOR = "#5674B9"
SAMPLED_MAP_EDGE_COLOR = "#253A76"
REFINED_MAP_COLOR = "#D68484"
REFINED_MAP_EDGE_COLOR = "#6F3A3A"
SIMULATION_COLOR = "#F2C14E"
SIMULATION_EDGE_COLOR = "#7A5B00"
AXIS_COLOR = "#555555"
TEXT_COLOR = "#303030"


def _load_required_array(run_dir, file_name):
    path = run_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, allow_pickle=file_name == "output_names.npy")


def _load_plot_data(run_dir):
    output_names = _load_required_array(run_dir, "output_names.npy").tolist()
    obs_means = _load_required_array(run_dir, "obs_means.npy").astype(np.float64)
    obs_vars = _load_required_array(run_dir, "obs_vars.npy").astype(np.float64)
    sampled_mu = _load_required_array(run_dir, "sampled_map_predictions.npy").astype(
        np.float64
    )
    sampled_sd = _load_required_array(
        run_dir, "sampled_map_prediction_stds.npy"
    ).astype(np.float64)
    refined_mu = _load_required_array(run_dir, "refined_map_predictions.npy").astype(
        np.float64
    )
    refined_sd = _load_required_array(
        run_dir, "refined_map_prediction_stds.npy"
    ).astype(np.float64)

    pred_matrix_path = run_dir / "pred_check_matrix.npy"
    pred_matrix = None
    if pred_matrix_path.exists():
        pred_matrix = np.load(pred_matrix_path).astype(np.float64)

    n_out = len(output_names)
    arrays = {
        "obs_means": obs_means,
        "obs_vars": obs_vars,
        "sampled_map_predictions": sampled_mu,
        "sampled_map_prediction_stds": sampled_sd,
        "refined_map_predictions": refined_mu,
        "refined_map_prediction_stds": refined_sd,
    }
    for name, array in arrays.items():
        if array.shape != (n_out,):
            raise ValueError(f"{name} has shape {array.shape}; expected {(n_out,)}.")
    if pred_matrix is not None and (
        pred_matrix.ndim != 2 or pred_matrix.shape[1] != n_out
    ):
        raise ValueError(
            f"pred_check_matrix has shape {pred_matrix.shape}; "
            f"expected (n_samples, {n_out})."
        )

    return {
        "output_names": output_names,
        "obs_means": obs_means,
        "obs_stds": np.sqrt(obs_vars),
        "sampled_mu": sampled_mu,
        "sampled_sd": sampled_sd,
        "refined_mu": refined_mu,
        "refined_sd": refined_sd,
        "pred_matrix": pred_matrix,
    }


def _strip_state_prefix(output_name):
    name = str(output_name)
    for prefix in ("Rest_", "Exercise_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _display_label(output_name):
    name = str(output_name)
    if name.startswith("Rest_"):
        state = "Rest"
    elif name.startswith("Exercise_"):
        state = "Exercise"
    else:
        state = None

    base = _strip_state_prefix(name)
    label = BASE_DISPLAY_LABELS.get(base, base.replace("_", "\n"))
    if state is None:
        return label
    return f"{state}\n{label}"


def _resolve_ratio_groups(output_names):
    name_to_idx = {str(name): i for i, name in enumerate(output_names)}
    specs = [
        (
            "Rest LA",
            "Rest_Min_LA_Volume",
            "Rest_Max_LA_Volume",
            "Rest_Pre_LA_Contraction_Volume",
        ),
        (
            "Rest RA",
            "Rest_Min_RA_Volume",
            "Rest_Max_RA_Volume",
            "Rest_Pre_RA_Contraction_Volume",
        ),
        (
            "Exercise LA",
            "Exercise_Min_LA_Volume",
            "Exercise_Max_LA_Volume",
            "Exercise_Pre_LA_Contraction_Volume",
        ),
        (
            "Exercise RA",
            "Exercise_Min_RA_Volume",
            "Exercise_Max_RA_Volume",
            "Exercise_Pre_RA_Contraction_Volume",
        ),
        ("LA", "Min_LA_Volume", "Max_LA_Volume", "Pre_LA_Contraction_Volume"),
        ("RA", "Min_RA_Volume", "Max_RA_Volume", "Pre_RA_Contraction_Volume"),
    ]

    groups = []
    for label, min_name, max_name, pre_name in specs:
        if min_name in name_to_idx and max_name in name_to_idx and pre_name in name_to_idx:
            groups.append(
                {
                    "label": label,
                    "min": name_to_idx[min_name],
                    "max": name_to_idx[max_name],
                    "pre": name_to_idx[pre_name],
                }
            )
    return groups


def _safe_ratio_denominator(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    sign = np.where(x >= 0.0, 1.0, -1.0)
    return np.where(np.abs(x) < eps, sign * eps, x)


def _pre_a_volume_mean_sd_from_ratio(mean_vec, sd_vec, idx_min, idx_max, idx_pre):
    vmin = float(mean_vec[idx_min])
    vmax = float(mean_vec[idx_max])
    ratio = float(mean_vec[idx_pre])
    delta = vmax - vmin
    volume = vmin + ratio * delta

    d_vmin = 1.0 - ratio
    d_vmax = ratio
    d_ratio = delta
    var = (
        (d_vmin * float(sd_vec[idx_min])) ** 2
        + (d_vmax * float(sd_vec[idx_max])) ** 2
        + (d_ratio * float(sd_vec[idx_pre])) ** 2
    )
    return volume, np.sqrt(max(var, 0.0))


def _pre_a_values_are_ratio(values, ratio_groups, axis="vector"):
    if values is None or not ratio_groups:
        return False

    values = np.asarray(values, dtype=np.float64)
    for group in ratio_groups:
        idx_pre = group["pre"]
        pre_values = values[:, idx_pre] if axis == "matrix" else np.asarray([values[idx_pre]])
        finite_values = pre_values[np.isfinite(pre_values)]
        if finite_values.size == 0:
            continue

        # Atrial volumes in this model are tens of mL, while the ratio target is
        # around 0.25. This catches arrays saved on the ratio scale.
        if abs(float(np.nanmedian(finite_values))) > 2.0:
            return False

    return True


def _convert_pre_a_vector_ratio_to_volume(values, ratio_groups):
    if values is None or not ratio_groups:
        return values

    converted = np.asarray(values, dtype=np.float64).copy()
    for group in ratio_groups:
        idx_min = group["min"]
        idx_max = group["max"]
        idx_pre = group["pre"]
        converted[idx_pre] = converted[idx_min] + converted[idx_pre] * (
            converted[idx_max] - converted[idx_min]
        )
    return converted


def _convert_pre_a_matrix_ratio_to_volume(matrix, ratio_groups):
    if matrix is None or not ratio_groups:
        return matrix

    converted = np.asarray(matrix, dtype=np.float64).copy()
    for group in ratio_groups:
        idx_min = group["min"]
        idx_max = group["max"]
        idx_pre = group["pre"]
        converted[:, idx_pre] = converted[:, idx_min] + converted[:, idx_pre] * (
            converted[:, idx_max] - converted[:, idx_min]
        )
    return converted


def _convert_pre_a_vector_to_volume_if_needed(values, ratio_groups):
    if values is None or not ratio_groups:
        return values, False
    if _pre_a_values_are_ratio(values, ratio_groups, axis="vector"):
        return _convert_pre_a_vector_ratio_to_volume(values, ratio_groups), True
    return np.asarray(values, dtype=np.float64).copy(), False


def _convert_pre_a_matrix_to_volume_if_needed(matrix, ratio_groups):
    if matrix is None or not ratio_groups:
        return matrix, False
    if _pre_a_values_are_ratio(matrix, ratio_groups, axis="matrix"):
        return _convert_pre_a_matrix_ratio_to_volume(matrix, ratio_groups), True
    return np.asarray(matrix, dtype=np.float64).copy(), False


def _convert_pre_a_predictions_to_volume_if_needed(mean_vec, sd_vec, ratio_groups):
    if not ratio_groups:
        return mean_vec, sd_vec, False

    converted_mean = np.asarray(mean_vec, dtype=np.float64).copy()
    converted_sd = np.asarray(sd_vec, dtype=np.float64).copy()
    if not _pre_a_values_are_ratio(converted_mean, ratio_groups, axis="vector"):
        return converted_mean, converted_sd, False

    for group in ratio_groups:
        idx_min = group["min"]
        idx_max = group["max"]
        idx_pre = group["pre"]
        converted_mean[idx_pre], converted_sd[idx_pre] = (
            _pre_a_volume_mean_sd_from_ratio(
            mean_vec, sd_vec, idx_min, idx_max, idx_pre
        )
        )
    return converted_mean, converted_sd, True


def _convert_pre_a_observations_to_volume(obs_means, obs_stds, ratio_groups):
    converted_means = np.asarray(obs_means, dtype=np.float64).copy()
    converted_stds = np.asarray(obs_stds, dtype=np.float64).copy()
    if not ratio_groups:
        return converted_means, converted_stds

    for group in ratio_groups:
        idx_min = group["min"]
        idx_max = group["max"]
        idx_pre = group["pre"]
        if abs(float(converted_means[idx_pre])) > 2.0:
            continue
        converted_means[idx_pre], converted_stds[idx_pre] = (
            _pre_a_volume_mean_sd_from_ratio(
                obs_means, obs_stds, idx_min, idx_max, idx_pre
            )
        )
    return converted_means, converted_stds


def _load_numeric_file(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if len(data.files) != 1:
            raise ValueError(
                f"{path} contains multiple arrays: {data.files}. "
                "Use a .npy file or save the desired vector as the only .npz array."
            )
        return data[data.files[0]]
    if suffix in {".csv", ".txt"}:
        text = path.read_text()
        numeric_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        values = [float(match) for match in re.findall(numeric_pattern, text)]
        if values:
            return np.asarray(values, dtype=np.float64)

        csv_values = []
        with path.open(newline="") as f:
            for row in csv.reader(f):
                for item in row:
                    item = item.strip()
                    if item:
                        csv_values.append(float(item))
        return np.asarray(csv_values, dtype=np.float64)
    raise ValueError(f"Unsupported actual simulation file type: {path}")


def _flatten_one_sample(array, label):
    array = np.asarray(array, dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim == 0:
        raise ValueError(f"{label} contains a scalar; expected an output vector.")
    if array.ndim == 1:
        return array
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    if array.ndim == 2:
        raise ValueError(
            f"{label} has shape {array.shape}; provide a single row/vector, not "
            "multiple simulation samples."
        )
    raise ValueError(f"{label} has shape {array.shape}; expected a 1D output vector.")


def _drop_per_state_raw_columns(raw_values):
    raw_values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    if raw_values.size == RAW_OUTPUTS_PER_STATE:
        keep_mask = np.ones(RAW_OUTPUTS_PER_STATE, dtype=bool)
        keep_mask[list(RESULT_COLS_TO_DROP_PER_STATE)] = False
        return raw_values[keep_mask]
    if raw_values.size == 2 * RAW_OUTPUTS_PER_STATE:
        keep_mask = np.ones(2 * RAW_OUTPUTS_PER_STATE, dtype=bool)
        drop = list(RESULT_COLS_TO_DROP_PER_STATE) + [
            i + RAW_OUTPUTS_PER_STATE for i in RESULT_COLS_TO_DROP_PER_STATE
        ]
        keep_mask[drop] = False
        return raw_values[keep_mask]
    raise ValueError(
        f"Raw simulation vector has {raw_values.size} values; expected "
        f"{RAW_OUTPUTS_PER_STATE} or {2 * RAW_OUTPUTS_PER_STATE}."
    )


def _align_actual_simulation(actual_outputs, n_out):
    actual_outputs = _flatten_one_sample(actual_outputs, "actual simulation output")
    if actual_outputs.size == n_out:
        return actual_outputs

    raw_union_outputs = 2 * RAW_OUTPUTS_PER_STATE
    if n_out == 2 * ALIGNED_OUTPUTS_PER_STATE and actual_outputs.size == raw_union_outputs:
        return _drop_per_state_raw_columns(actual_outputs)
    if n_out == ALIGNED_OUTPUTS_PER_STATE and actual_outputs.size == RAW_OUTPUTS_PER_STATE:
        return _drop_per_state_raw_columns(actual_outputs)

    expected_forms = [f"{n_out} already-aligned outputs"]
    if n_out == 2 * ALIGNED_OUTPUTS_PER_STATE:
        expected_forms.append(f"{raw_union_outputs} raw union outputs")
    elif n_out == ALIGNED_OUTPUTS_PER_STATE:
        expected_forms.append(f"{RAW_OUTPUTS_PER_STATE} raw single-state outputs")

    raise ValueError(
        f"Actual simulation vector has {actual_outputs.size} values; expected "
        + ", or ".join(expected_forms)
        + "."
    )


def _find_auto_actual_file(run_dir):
    search_dirs = (run_dir, run_dir.parent)
    for directory in search_dirs:
        for file_name in AUTO_ACTUAL_FILENAMES:
            path = directory / file_name
            if path.exists():
                return path
    return None


def _load_actual_outputs(run_dir, n_out, actual_path=None, rest_path=None, exercise_path=None):
    if actual_path is not None and (rest_path is not None or exercise_path is not None):
        raise ValueError(
            "Use either --actual-simulation, or --rest-actual-simulation and "
            "--exercise-actual-simulation, not both."
        )

    source = None
    if actual_path is not None:
        source = Path(actual_path)
        actual = _load_numeric_file(source)
        return _align_actual_simulation(actual, n_out), source

    if rest_path is not None or exercise_path is not None:
        if rest_path is None or exercise_path is None:
            raise ValueError(
                "Both --rest-actual-simulation and --exercise-actual-simulation "
                "are required when using separate state files."
            )
        rest = _drop_per_state_raw_columns(
            _flatten_one_sample(_load_numeric_file(rest_path), "rest actual output")
        )
        exercise = _drop_per_state_raw_columns(
            _flatten_one_sample(
                _load_numeric_file(exercise_path), "exercise actual output"
            )
        )
        actual = np.concatenate([rest, exercise])
        if actual.size != n_out:
            raise ValueError(
                f"Combined rest/exercise actual vector has {actual.size} values; "
                f"expected {n_out}."
            )
        return actual, f"{Path(rest_path)} + {Path(exercise_path)}"

    auto_path = _find_auto_actual_file(run_dir)
    if auto_path is None:
        return None, None
    return _align_actual_simulation(_load_numeric_file(auto_path), n_out), auto_path


def _state_boundary_positions(output_names):
    states = []
    for name in output_names:
        name = str(name)
        if name.startswith("Rest_"):
            states.append("Rest")
        elif name.startswith("Exercise_"):
            states.append("Exercise")
        else:
            states.append(None)

    boundaries = []
    for idx in range(1, len(states)):
        if states[idx] != states[idx - 1]:
            boundaries.append(idx - 0.5)
    return boundaries


def plot_union_refined_map_with_simulation(
    run_dir,
    output_path,
    actual_path=None,
    rest_actual_path=None,
    exercise_actual_path=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    run_dir = Path(run_dir).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_plot_data(run_dir)
    output_names = data["output_names"]
    n_out = len(output_names)
    x_pos = np.arange(n_out)
    ratio_groups = _resolve_ratio_groups(output_names)

    obs_means, obs_stds = _convert_pre_a_observations_to_volume(
        data["obs_means"], data["obs_stds"], ratio_groups
    )
    pred_matrix, converted_pred_matrix = _convert_pre_a_matrix_to_volume_if_needed(
        data["pred_matrix"], ratio_groups
    )
    sampled_mu, sampled_sd, converted_sampled = _convert_pre_a_predictions_to_volume_if_needed(
        data["sampled_mu"], data["sampled_sd"], ratio_groups
    )
    refined_mu, refined_sd, converted_refined = _convert_pre_a_predictions_to_volume_if_needed(
        data["refined_mu"], data["refined_sd"], ratio_groups
    )

    actual_outputs, actual_source = _load_actual_outputs(
        run_dir,
        n_out,
        actual_path=actual_path,
        rest_path=rest_actual_path,
        exercise_path=exercise_actual_path,
    )
    actual_outputs, converted_actual = _convert_pre_a_vector_to_volume_if_needed(
        actual_outputs, ratio_groups
    )

    normalised_matrix = None
    if pred_matrix is not None:
        normalised_matrix = (pred_matrix - obs_means) / obs_stds

    sampled_res = (sampled_mu - obs_means) / obs_stds
    sampled_err = sampled_sd / obs_stds
    refined_res = (refined_mu - obs_means) / obs_stds
    refined_err = refined_sd / obs_stds
    actual_res = None
    if actual_outputs is not None:
        actual_res = (actual_outputs - obs_means) / obs_stds

    finite_for_limits = [
        sampled_res - sampled_err,
        sampled_res + sampled_err,
        refined_res - refined_err,
        refined_res + refined_err,
    ]
    if actual_res is not None:
        finite_for_limits.append(actual_res)
    if normalised_matrix is not None:
        matrix_values = normalised_matrix[np.isfinite(normalised_matrix)]
        if matrix_values.size:
            finite_for_limits.extend(
                [
                    np.array([np.nanpercentile(matrix_values, 1)]),
                    np.array([np.nanpercentile(matrix_values, 99)]),
                ]
            )

    limits = np.concatenate([v[np.isfinite(v)] for v in finite_for_limits])
    ymin = min(-3.5, float(np.nanmin(limits)) - 0.5)
    ymax = max(3.5, float(np.nanmax(limits)) + 0.5)

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.linewidth": 1.1,
            "axes.labelsize": 13,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelsize": 8,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "font.size": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(
        figsize=(max(16.0, 0.46 * n_out), 7.2),
        facecolor="white",
    )
    ax.axhspan(
        -1,
        1,
        facecolor=TARGET_BAND_COLOR,
        alpha=0.07,
        edgecolor="none",
        zorder=0,
    )

    if normalised_matrix is not None:
        ax.boxplot(
            normalised_matrix,
            positions=x_pos,
            widths=0.46,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(
                facecolor=POSTERIOR_BOX_COLOR,
                alpha=0.38,
                edgecolor=POSTERIOR_EDGE_COLOR,
                linewidth=1.0,
            ),
            medianprops=dict(color=POSTERIOR_EDGE_COLOR, linewidth=1.4),
            whiskerprops=dict(
                color=POSTERIOR_EDGE_COLOR,
                alpha=0.75,
                linewidth=1.0,
            ),
            capprops=dict(
                color=POSTERIOR_EDGE_COLOR,
                alpha=0.75,
                linewidth=1.0,
            ),
        )

    for boundary in _state_boundary_positions(output_names):
        ax.axvline(
            boundary,
            color="#B8B8B8",
            linewidth=1.0,
            linestyle=(0, (2, 2)),
            zorder=1,
        )

    ax.axhline(0, color=TARGET_LINE_COLOR, ls="-", linewidth=1.1, zorder=1)
    ax.axhline(
        1,
        color=TARGET_LINE_COLOR,
        ls=(0, (4, 3)),
        alpha=0.55,
        linewidth=1.0,
        zorder=1,
    )
    ax.axhline(
        -1,
        color=TARGET_LINE_COLOR,
        ls=(0, (4, 3)),
        alpha=0.55,
        linewidth=1.0,
        zorder=1,
    )
    ax.axhline(
        3,
        color=REFINED_MAP_EDGE_COLOR,
        ls=(0, (1, 2)),
        alpha=0.45,
        linewidth=1.0,
        zorder=1,
    )
    ax.axhline(
        -3,
        color=REFINED_MAP_EDGE_COLOR,
        ls=(0, (1, 2)),
        alpha=0.45,
        linewidth=1.0,
        zorder=1,
    )

    # ax.errorbar(
    #     x_pos - 0.10,
    #     sampled_res,
    #     yerr=sampled_err,
    #     fmt="D",
    #     color=SAMPLED_MAP_COLOR,
    #     markersize=4.9,
    #     markeredgecolor=SAMPLED_MAP_EDGE_COLOR,
    #     markeredgewidth=0.65,
    #     elinewidth=0.9,
    #     ecolor=SAMPLED_MAP_COLOR,
    #     capsize=2.0,
    #     zorder=5,
    # )
    ax.errorbar(
        x_pos + 0.10,
        refined_res,
        yerr=refined_err,
        fmt="*",
        color=REFINED_MAP_COLOR,
        markersize=9.5,
        markeredgecolor=REFINED_MAP_EDGE_COLOR,
        markeredgewidth=0.6,
        elinewidth=0.9,
        ecolor=REFINED_MAP_COLOR,
        capsize=2.0,
        zorder=6,
    )
    if actual_res is not None:
        ax.scatter(
            x_pos,
            actual_res,
            marker="o",
            s=38,
            color=SIMULATION_COLOR,
            edgecolor=SIMULATION_EDGE_COLOR,
            linewidth=0.8,
            zorder=7,
        )

    short_names = [_display_label(name) for name in output_names]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        short_names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )
    ax.set_ylabel(r"Residual / $\sigma_{\mathrm{obs}}$")
    ax.set_title("Without $V_{tot}$ calibration")
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.55, zorder=0)
    ax.tick_params(axis="both", which="major", width=1.0, length=4, colors=TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)

    legend_handles = [
        Patch(
            facecolor=POSTERIOR_BOX_COLOR,
            edgecolor=POSTERIOR_EDGE_COLOR,
            alpha=0.38,
            label="Posterior predictive draws",
        ),
        Patch(
            facecolor=TARGET_BAND_COLOR,
            edgecolor=TARGET_BAND_COLOR,
            alpha=0.07,
            label=r"Target $\pm$ 1 SD",
        ),
        Line2D([0], [0], color=TARGET_LINE_COLOR, linewidth=1.1, label="Target mean"),
        Line2D(
            [0],
            [0],
            color=REFINED_MAP_EDGE_COLOR,
            linestyle=(0, (1, 2)),
            linewidth=1.0,
            alpha=0.55,
            label=r"Target $\pm$ 3 SD",
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     marker="D",
        #     color=SAMPLED_MAP_COLOR,
        #     markerfacecolor=SAMPLED_MAP_COLOR,
        #     markeredgecolor=SAMPLED_MAP_EDGE_COLOR,
        #     linewidth=0,
        #     markersize=5.8,
        #     label="Sampled MAP emulator",
        # ),
        Line2D(
            [0],
            [0],
            marker="*",
            color=REFINED_MAP_COLOR,
            markerfacecolor=REFINED_MAP_COLOR,
            markeredgecolor=REFINED_MAP_EDGE_COLOR,
            linewidth=0,
            markersize=10,
            label="Refined MAP emulator",
        ),
    ]
    if actual_res is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=SIMULATION_COLOR,
                markerfacecolor=SIMULATION_COLOR,
                markeredgecolor=SIMULATION_EDGE_COLOR,
                linewidth=0,
                markersize=6,
                label="Actual simulation output",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=6,
        frameon=False,
    )

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.30, top=0.79)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)

    return {
        "output_path": output_path,
        "output_names": output_names,
        "actual_residual": actual_res,
        "actual_source": actual_source,
        "ratio_groups": ratio_groups,
        "converted_pred_matrix": converted_pred_matrix,
        "converted_sampled": converted_sampled,
        "converted_refined": converted_refined,
        "converted_actual": converted_actual,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the union rest+exercise refined MAP residuals. Pre-atrial "
            "columns are displayed as pre-A contraction volumes."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Path to the union MCMC result directory.",
    )
    parser.add_argument(
        "--actual-simulation",
        default=None,
        help=(
            "Optional path to one union actual simulation vector. Accepts 50 "
            "aligned outputs or 62 raw rest+exercise outputs."
        ),
    )
    parser.add_argument(
        "--rest-actual-simulation",
        default=None,
        help=(
            "Optional path to a rest actual simulation vector. Use together "
            "with --exercise-actual-simulation. Accepts 25 aligned or 31 raw outputs."
        ),
    )
    parser.add_argument(
        "--exercise-actual-simulation",
        default=None,
        help=(
            "Optional path to an exercise actual simulation vector. Use together "
            "with --rest-actual-simulation. Accepts 25 aligned or 31 raw outputs."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "PNG output path. Defaults to refined_map_vs_targets_with_actual_simulation.png "
            "inside --run-dir."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else run_dir / DEFAULT_OUTPUT_NAME
    )

    result = plot_union_refined_map_with_simulation(
        run_dir=run_dir,
        output_path=output_path,
        actual_path=args.actual_simulation,
        rest_actual_path=args.rest_actual_simulation,
        exercise_actual_path=args.exercise_actual_simulation,
    )

    print(f"Saved: {result['output_path']}")
    print(
        "Displayed pre-A contraction columns as volumes for: "
        + ", ".join(group["label"] for group in result["ratio_groups"])
    )
    print(
        "Posterior predictive boxplot pre-A columns were "
        + (
            "converted from ratios to volumes."
            if result["converted_pred_matrix"]
            else "already on the volume scale."
        )
    )

    residuals = result["actual_residual"]
    if residuals is None:
        print(
            "No actual simulation vector was supplied or auto-detected; "
            "saved emulator/posterior residuals only."
        )
        return

    residuals = np.asarray(residuals, dtype=np.float64)
    within_one_sd = int(np.sum(np.abs(residuals) <= 1.0))
    within_three_sd = int(np.sum(np.abs(residuals) <= 3.0))
    worst_idx = int(np.nanargmax(np.abs(residuals)))

    print(f"Actual simulation source: {result['actual_source']}")
    print(f"Actual simulation points within 1 target SD: {within_one_sd}/{residuals.size}")
    print(f"Actual simulation points within 3 target SD: {within_three_sd}/{residuals.size}")
    print(
        "Worst actual simulation residual: "
        f"{result['output_names'][worst_idx]} = {residuals[worst_idx]:.3f} SD"
    )


if __name__ == "__main__":
    main()
