"""One-shot post-processing for the refined union MAP.

Replaces the manual chain that used to follow Refine_MAP_Union_all.py (paste
into check.py -> Run_model_all_correct_order.py at 150 s then 450 s -> paste
targets into Rest_exercise_refined_simulation_HM.txt -> Plot_Union_Refined_MAP_
With_Simulation.py -> Run_model_Paper.py). Everything stays inside this
DGSM project folder:

  * simulation: imports Samples_for_DGSM_Union.py from the parent DGSM folder
    and runs the refined MAP to convergence at Rest then Exercise, exactly like
    the DGSM sampling does;
  * residual figure: reuses plot_union_refined_map_with_simulation from the
    sibling Plot_Union_Refined_MAP_With_Simulation.py (correct pre-atrial
    ratio-to-volume handling);
  * target traces: reimplements the Run_model_Paper.py rest/exercise panels.

All outputs land in <run_dir>/Target_Trace. Nothing outside this
DGSM project folder is imported or called.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_RUN_NAME = "MCMC_Union_50_28_08_copula_prior"
DEFAULT_SIMULATION_TXT = "Rest_exercise_refined_simulation_HM_28-08.txt"
DEFAULT_OUTPUT_FOLDER = "Target_Trace"
DEFAULT_CACHE_NAME = "Run_model_simulation_cache.pkl"
CACHE_VERSION = 1

RAW_OUTPUTS_PER_STATE_HM = 31
RAW_OUTPUTS_PER_STATE_SIMULATOR = 32
RESULT_COLS_TO_DROP_PER_STATE = (11, 14, 17, 20, 27, 30)

RAW_OUTPUT_NAMES_PER_STATE = [
    "Heart_Rate",
    "Systolic_Pressure",
    "Diastolic_Pressure",
    "EDV",
    "ESV",
    "Max_RV_Volume",
    "Min_RV_Volume",
    "Max_RV_Pressure",
    "Min_RV_Pressure",
    "Min_RA_Volume",
    "Max_RA_Volume",
    "Min_RA_Pressure_A_descent",
    "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening",
    "Min_RA_Pressure_V_descent",
    "Min_LA_Volume",
    "Max_LA_Volume",
    "Min_LA_Pressure_A_descent",
    "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening",
    "Min_LA_Pressure_V_descent",
    "Pre_LA_Contraction_Volume",
    "Pre_RA_Contraction_Volume",
    "LV_Pressure_Deriv",
    "RV_Pressure_Deriv",
    "Tidal_Volume",
    "Minute_Ventilation",
    "Cardiac_Output",
    "PaO2",
    "PaCO2",
    "Pericardial_Volume_Percentage_Change",
]

PLOT_OUTPUT_NAMES_PER_STATE = [
    "Heart_Rate",
    "Systolic_Pressure",
    "Diastolic_Pressure",
    "EDV",
    "ESV",
    "Max_RV_Volume",
    "Min_RV_Volume",
    "Max_RV_Pressure",
    "Min_RV_Pressure",
    "Min_RA_Volume",
    "Max_RA_Volume",
    "Min_RA_Pressure_A_descent",
    "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening",
    "Min_RA_Pressure_V_descent",
    "Min_LA_Volume",
    "Max_LA_Volume",
    "Min_LA_Pressure_A_descent",
    "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening",
    "Min_LA_Pressure_V_descent",
    "LA_Volume_Before_Atrial_Contraction",
    "RA_Volume_Before_Atrial_Contraction",
    "LV_Pressure_Deriv",
    "RV_Pressure_Deriv",
    "Tidal_Volume",
    "Minute_Ventilation",
    "Cardiac_Output",
    "PaO2",
    "PaCO2",
    "Pericardial_Volume_Percentage_Change",
]

ACTIVATION_HISTORY_POINTS = 4000
ATRIAL_PV_HISTORY_POINTS = 10000
GAS_EXCHANGE_WINDOW_SECONDS = 10.0
RESULTS_OVERVIEW_WINDOW_SECONDS = 10.0
RESULTS_ATRIAL_DETAIL_WINDOW_SECONDS = 2.5
HEART_RATE_PLOT_SCALE = 60.0
SOLID_LINEWIDTH = 1.8
TARGET_LINEWIDTH = 1.4
FOCUS_LINEWIDTH = 2.0
SUBPLOT_LEGEND_FONT_SIZE = 11

STAGE_COLORS = ["#BBA3D6", "#9DB8D8", "#7DB6C0", "#D68484"]
PLOT_COLORS = {
    "solid_red": "#D68484",
    "solid_blue": "#7DB6C0",
    "lavender": STAGE_COLORS[0],
    "blue": STAGE_COLORS[1],
    "teal": STAGE_COLORS[2],
    "rose": STAGE_COLORS[3],
    "lavender_dark": "#8F78AB",
    "blue_dark": "#789CC4",
    "teal_dark": "#5D9FA8",
    "rose_dark": "#B86A6A",
    "lavender_light": "#C9B8DE",
    "blue_light": "#B4CAE2",
    "teal_light": "#9ACBD1",
    "rose_light": "#E2A4A4",
    "ink": "#4A4E57",
}
GAS_EXCHANGE_COLORS = [
    "#B84E4E",
    "#E8A5A5",
    "#C97A2C",
    "#F0BD7F",
    "#4F8F5A",
    "#A5CFA8",
    "#3F6CAB",
    "#A0BEE0",
    "#6B4A92",
    "#C5AED8",
    "#5A4A3E",
    "#B59880",
]
JOURNAL_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#555555",
    "axes.labelcolor": "#303030",
    "axes.linewidth": 1.1,
    "axes.labelsize": 13,
    "xtick.color": "#303030",
    "ytick.color": "#303030",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": SUBPLOT_LEGEND_FONT_SIZE,
    "font.size": 12,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
TARGET_LABELS = {
    "Heart_Rate": "HR",
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
    "LA_Volume_Before_Atrial_Contraction": r"$V_{\mathrm{pre-A,LA}}$",
    "RA_Volume_Before_Atrial_Contraction": r"$V_{\mathrm{pre-A,RA}}$",
    "LV_Pressure_Deriv": r"$\max\,\mathrm{d}P_{\mathrm{LV}}/\mathrm{d}t$",
    "RV_Pressure_Deriv": r"$\max\,\mathrm{d}P_{\mathrm{RV}}/\mathrm{d}t$",
    "Tidal_Volume": "Inspired/Expired Volume",
    "Minute_Ventilation": r"$\dot{V}_E$",
    "PaO2": r"$P_{\mathrm{a}O_2}$",
    "PaCO2": r"$P_{\mathrm{a}CO_2}$",
}
TARGET_MEAN_LABELS = {
    "Heart_Rate": r"$\overline{\mathrm{HR}}$",
    "PaO2": r"$\overline{P_{\mathrm{a}O_2}}$",
    "PaCO2": r"$\overline{P_{\mathrm{a}CO_2}}$",
    "Tidal_Volume": r"$V_T$",
}

REQUIRED_GAS_KEYS = [
    "Pd_1_O2",
    "Pd_1_CO2",
    "Pd_2_O2",
    "Pd_2_CO2",
    "Pd_3_O2",
    "Pd_3_CO2",
    "Pd_4_O2",
    "Pd_4_CO2",
    "Pd_5_O2",
    "Pd_5_CO2",
    "Pa_O2",
    "Pa_CO2",
    "dPa_O2_dt",
    "dPa_CO2_dt",
    "PA_O2",
    "PA_CO2",
    "PCSFCO2",
    "MRTO2",
    "MRTCO2",
    "CTO2",
    "CvtCO2",
    "CBO2",
    "CvbCO2",
    "MRV",
]
GAS_EXCHANGE_SKIPPED_LABELS = {
    "Pa_O2",
    "Pa_CO2",
    "dPa_O2_dt",
    "dPa_CO2_dt",
    "PCSFCO2",
    "MRTO2",
    "MRTCO2",
    "CTO2",
    "CvtCO2",
    "CBO2",
    "CvbCO2",
    "MRV",
}


@dataclass
class SimulationOutputs:
    rest_full: np.ndarray
    exercise_full: np.ndarray
    rest_hm_raw: np.ndarray
    exercise_hm_raw: np.ndarray
    raw_union_hm: np.ndarray
    aligned_union: np.ndarray
    rest_storage: dict[str, Any]
    exercise_storage: dict[str, Any]
    rest_converged: bool
    exercise_converged: bool
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run refined_map_parameters through the DGSM union simulator."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_DIR,
        help=(
            "DGSM folder containing Samples_for_DGSM_Union.py, fixed_params.py, "
            "All_derivatives_njit.py, and related simulator files. Defaults to "
            "the parent folder of this script."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "MCMC run directory containing refined MAP outputs."
        ),
    )
    parser.add_argument(
        "--simulation-txt",
        default=DEFAULT_SIMULATION_TXT,
        help="HM-compatible raw rest+exercise text file to write inside --run-dir.",
    )
    parser.add_argument(
        "--chunk-time",
        type=float,
        default=None,
        help="Override Samples_for_DGSM_Union.max_time for each convergence chunk.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Override Samples_for_DGSM_Union.MAX_CONVERGENCE_ATTEMPTS.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Override Samples_for_DGSM_Union.MIN_MEASUREMENT_DURATION.",
    )
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=None,
        help="Override Samples_for_DGSM_Union.CONVERGENCE_TOLERANCE.",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable per-chunk timeout wrapper in Samples_for_DGSM_Union.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create the simple residual plot.",
    )
    parser.add_argument(
        "--no-paper-plots",
        action="store_true",
        help="Do not create Run_model_Paper-style rest/exercise target plots.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore any cached simulation and run the ODE solve again.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the simulation cache pickle (always simulate).",
    )
    parser.add_argument(
        "--cache-name",
        default=DEFAULT_CACHE_NAME,
        help="Pickle filename inside the Target_Trace output folder for the cached simulation.",
    )
    return parser.parse_args()


def format_number(value: float, precision: int = 15) -> str:
    value = float(value)
    if not np.isfinite(value):
        return repr(value)
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def import_samples_pipeline(project_dir: Path) -> Any:
    require_file(project_dir / "Samples_for_DGSM_Union.py")
    sys.path.insert(0, str(project_dir))
    import Samples_for_DGSM_Union as pipeline

    return pipeline


def as_float_dict(mapping: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in mapping.items()}


def load_refined_parameters(run_dir: Path, project_dir: Path) -> dict[str, float]:
    npy_path = run_dir / "refined_map_parameters.npy"
    json_path = run_dir / "refined_map_parameters.json"
    if npy_path.exists():
        loaded = np.load(npy_path, allow_pickle=True).item()
        return as_float_dict(loaded)
    if json_path.exists():
        with json_path.open() as f:
            return as_float_dict(json.load(f))

    posterior_dict_path = run_dir / "posterior_param_dict.npy"
    subset_vars_path = run_dir / "subset_vars.npy"
    refined_values_path = run_dir / "refined_map_sample.npy"
    if not (posterior_dict_path.exists() and subset_vars_path.exists() and refined_values_path.exists()):
        raise FileNotFoundError(
            "Could not find refined_map_parameters.npy/json, and could not "
            "reconstruct from posterior_param_dict.npy, subset_vars.npy, and "
            "refined_map_sample.npy."
        )

    parameters = as_float_dict(np.load(posterior_dict_path, allow_pickle=True).item())
    # Only this rare reconstruction path needs the simulator's nominal parameters,
    # so import the heavy pipeline lazily here to keep cache-hit plot reruns fast.
    old_parameters = import_samples_pipeline(project_dir).Old_Parameters
    for name, value in old_parameters.items():
        parameters.setdefault(str(name), float(value))

    subset_vars = np.load(subset_vars_path, allow_pickle=True).tolist()
    refined_values = np.load(refined_values_path, allow_pickle=True)
    if len(subset_vars) != len(refined_values):
        raise ValueError(
            f"Length mismatch: subset_vars has {len(subset_vars)} names, "
            f"refined_map_sample has {len(refined_values)} values."
        )

    for name, value in zip(subset_vars, refined_values):
        parameters[str(name)] = float(value)

    np.save(npy_path, parameters)
    with json_path.open("w") as f:
        json.dump(parameters, f, indent=2)
    return parameters


SIMULATION_OUTPUT_FIELDS = (
    "rest_full",
    "exercise_full",
    "rest_hm_raw",
    "exercise_hm_raw",
    "raw_union_hm",
    "aligned_union",
    "rest_storage",
    "exercise_storage",
    "rest_converged",
    "exercise_converged",
    "elapsed_seconds",
)


def _parameters_fingerprint(parameters: dict[str, float]) -> str:
    items = sorted((str(key), float(value)) for key, value in parameters.items())
    return hashlib.sha1(repr(items).encode("utf-8")).hexdigest()


def load_simulation_cache(
    cache_path: Path,
    parameters: dict[str, float],
    force_rerun: bool,
) -> tuple[SimulationOutputs, int] | None:
    """Return ``(outputs, buffer_limit)`` from a cached pickle when it matches the
    current refined parameters, else ``None`` so the caller re-simulates.

    The cache exists so plot-only tweaks (axis ranges, colours, labels) skip the
    ~40 s ODE solve. It is invalidated automatically when the refined parameters
    change; pass ``--rerun`` to force a fresh simulation regardless.
    """
    if force_rerun or not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except Exception as exc:
        print(f"Ignoring unreadable simulation cache ({exc}); re-simulating.")
        return None
    if not isinstance(cache, dict) or cache.get("cache_version") != CACHE_VERSION:
        print("Simulation cache version mismatch; re-simulating.")
        return None
    if cache.get("parameters_fingerprint") != _parameters_fingerprint(parameters):
        print("Refined parameters changed since the cache was written; re-simulating.")
        return None
    try:
        outputs = SimulationOutputs(**{field: cache[field] for field in SIMULATION_OUTPUT_FIELDS})
        buffer_limit = int(cache["buffer_limit"])
    except KeyError as exc:
        print(f"Simulation cache missing {exc}; re-simulating.")
        return None
    return outputs, buffer_limit


def save_simulation_cache(
    cache_path: Path,
    outputs: SimulationOutputs,
    parameters: dict[str, float],
    buffer_limit: int,
) -> None:
    cache: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "parameters_fingerprint": _parameters_fingerprint(parameters),
        "buffer_limit": int(buffer_limit),
    }
    for field in SIMULATION_OUTPUT_FIELDS:
        cache[field] = getattr(outputs, field)
    # Write to a temp file then atomically replace so an interrupted run never
    # leaves a half-written cache that a later run would load.
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, cache_path)


def configure_pipeline(pipeline: Any, args: argparse.Namespace) -> None:
    if args.chunk_time is not None:
        pipeline.max_time = float(args.chunk_time)
    if args.max_attempts is not None:
        pipeline.MAX_CONVERGENCE_ATTEMPTS = int(args.max_attempts)
    if args.min_duration is not None:
        pipeline.MIN_MEASUREMENT_DURATION = float(args.min_duration)
    if args.convergence_tolerance is not None:
        pipeline.CONVERGENCE_TOLERANCE = float(args.convergence_tolerance)


def copy_storage_snapshot(storage: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in storage.items():
        if isinstance(value, np.ndarray):
            snapshot[key] = value.copy()
        else:
            snapshot[key] = value
    return snapshot


def run_refined_map(pipeline: Any, parameters: dict[str, float], use_timeout: bool) -> SimulationOutputs:
    start = time.time()
    storage = pipeline.make_fresh_storage()

    rest_result, ic_final, storage, breath_coef, rest_converged = pipeline.run_state_to_convergence(
        parameters,
        storage,
        pipeline.Old_Parameters,
        state="Rest",
        use_timeout=use_timeout,
    )
    if storage is None or ic_final is None or pipeline.is_failed_state_result(rest_result):
        raise RuntimeError("Rest simulation failed before producing a valid output vector.")

    rest_storage = copy_storage_snapshot(storage)
    latest_nonzero_index = (storage["i"].item() - 1) % pipeline.BUFFER_LIMIT
    exercise_start_time = storage["all_time"][latest_nonzero_index]

    exercise_result, ic_final, storage, breath_coef, exercise_converged = pipeline.run_state_to_convergence(
        parameters,
        storage,
        pipeline.Old_Parameters,
        IC_final=ic_final,
        state="Exercise",
        breath_coef=breath_coef,
        exercise_start_time=exercise_start_time,
        use_timeout=use_timeout,
    )
    if storage is None or ic_final is None or pipeline.is_failed_state_result(exercise_result):
        raise RuntimeError("Exercise simulation failed before producing a valid output vector.")

    exercise_storage = copy_storage_snapshot(storage)
    rest_full = np.asarray(rest_result, dtype=np.float64)
    exercise_full = np.asarray(exercise_result, dtype=np.float64)
    rest_hm_raw = to_hm_raw_state(rest_full, "rest")
    exercise_hm_raw = to_hm_raw_state(exercise_full, "exercise")
    raw_union_hm = np.concatenate([rest_hm_raw, exercise_hm_raw])
    aligned_union = align_hm_raw_union(raw_union_hm)

    return SimulationOutputs(
        rest_full=rest_full,
        exercise_full=exercise_full,
        rest_hm_raw=rest_hm_raw,
        exercise_hm_raw=exercise_hm_raw,
        raw_union_hm=raw_union_hm,
        aligned_union=aligned_union,
        rest_storage=rest_storage,
        exercise_storage=exercise_storage,
        rest_converged=bool(rest_converged),
        exercise_converged=bool(exercise_converged),
        elapsed_seconds=time.time() - start,
    )


def to_hm_raw_state(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == RAW_OUTPUTS_PER_STATE_HM:
        return values
    if values.size == RAW_OUTPUTS_PER_STATE_SIMULATOR:
        # Samples_for_DGSM_Union.py appends mean max P_peri as a 32nd value.
        # The HM/MCMC target convention uses the first 31 raw outputs.
        return values[:RAW_OUTPUTS_PER_STATE_HM]
    raise ValueError(
        f"{label} output has {values.size} values; expected "
        f"{RAW_OUTPUTS_PER_STATE_HM} or {RAW_OUTPUTS_PER_STATE_SIMULATOR}."
    )


def align_hm_raw_union(raw_union: np.ndarray) -> np.ndarray:
    raw_union = np.asarray(raw_union, dtype=np.float64).reshape(-1)
    expected = 2 * RAW_OUTPUTS_PER_STATE_HM
    if raw_union.size != expected:
        raise ValueError(f"raw_union has {raw_union.size} values; expected {expected}.")

    keep_mask = np.ones(expected, dtype=bool)
    drop = list(RESULT_COLS_TO_DROP_PER_STATE) + [
        idx + RAW_OUTPUTS_PER_STATE_HM for idx in RESULT_COLS_TO_DROP_PER_STATE
    ]
    keep_mask[drop] = False
    return raw_union[keep_mask]


def write_simulation_text(path: Path, rest: np.ndarray, exercise: np.ndarray) -> None:
    def block(label: str, values: np.ndarray) -> str:
        numbers = ", ".join(format_number(value) for value in values)
        return f"{label}: {{{numbers}\n}}"

    path.write_text(block("Rest", rest) + "\n\n" + block("Exercise", exercise) + "\n")


def import_plot_module() -> Any:
    """Import the sibling Plot_Union_Refined_MAP_With_Simulation module.

    It lives in this HM_MCMC folder, so using it does not reach outside the
    project tree. Reusing it guarantees the refined
    MAP residual figure matches the manual workflow exactly, including the
    pre-atrial ratio-to-volume conversion that an in-script residual would
    otherwise get wrong (raw pre-A volumes are ~40 mL, but the calibration target
    is the 0.25 active-emptying fraction).
    """
    require_file(SCRIPT_DIR / "Plot_Union_Refined_MAP_With_Simulation.py")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import Plot_Union_Refined_MAP_With_Simulation as plot_module

    return plot_module


def write_union_residual_plot(
    run_dir: Path,
    output_dir: Path,
    actual_source_path: Path,
    make_plot: bool,
) -> dict[str, Any] | None:
    if not make_plot:
        return None

    try:
        plot_module = import_plot_module()
    except FileNotFoundError as exc:
        print(f"Skipping residual comparison plot: {exc}")
        return None

    output_path = output_dir / "refined_map_vs_targets_with_actual_simulation.png"
    try:
        result = plot_module.plot_union_refined_map_with_simulation(
            run_dir=run_dir,
            output_path=output_path,
            actual_path=actual_source_path,
        )
    except Exception as exc:
        print(f"Skipping residual comparison plot: {exc}")
        return None

    output_names = list(result.get("output_names", []))
    residuals = result.get("actual_residual")
    summary: dict[str, Any] = {
        "residual_plot": str(output_path),
        "actual_source": str(actual_source_path),
    }
    if residuals is None:
        return summary

    residuals = np.asarray(residuals, dtype=np.float64)
    finite = residuals[np.isfinite(residuals)]
    worst_idx = int(np.nanargmax(np.abs(residuals)))
    summary.update(
        {
            "within_1_sd": int(np.sum(np.abs(finite) <= 1.0)),
            "within_3_sd": int(np.sum(np.abs(finite) <= 3.0)),
            "n_outputs": int(residuals.size),
            "worst_output": str(output_names[worst_idx]) if output_names else str(worst_idx),
            "worst_residual_sd": float(residuals[worst_idx]),
        }
    )

    csv_path = output_dir / "refined_map_actual_simulation_residuals.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output", "residual_sd"])
        for name, residual in zip(output_names, residuals):
            writer.writerow([name, residual])
    summary["residual_csv"] = str(csv_path)

    return summary


def get_pyplot() -> Any:
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _sorted_buffer(values: np.ndarray, start_idx: int) -> np.ndarray:
    values = np.asarray(values)
    return np.concatenate((values[start_idx:], values[:start_idx]))


def _finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _target(targets: dict[str, float], name: str) -> float:
    return float(targets.get(name, np.nan))


def _parameter_value(parameters: dict[str, float], name: str, default: float = np.nan) -> float:
    try:
        return float(parameters[name])
    except KeyError:
        return float(default)


def plot_targets_from_raw(raw_state: np.ndarray) -> dict[str, float]:
    raw_state = np.asarray(raw_state, dtype=np.float64).reshape(-1)
    targets = {
        name: float(raw_state[idx]) if idx < raw_state.size else np.nan
        for idx, name in enumerate(PLOT_OUTPUT_NAMES_PER_STATE)
    }
    targets["Pre_LA_Contraction_Volume"] = targets["LA_Volume_Before_Atrial_Contraction"]
    targets["Pre_RA_Contraction_Volume"] = targets["RA_Volume_Before_Atrial_Contraction"]
    return targets


def circular_traces(storage: dict[str, Any], keys: list[str], buffer_limit: int) -> dict[str, np.ndarray]:
    all_time = np.asarray(storage["all_time"], dtype=np.float64)
    if all_time.size == 0:
        raise ValueError("all_time buffer is empty.")

    effective_limit = min(int(buffer_limit), all_time.size)
    i_buffer = int(np.asarray(storage["i"]).item()) % effective_limit
    sorted_times = _sorted_buffer(all_time[:effective_limit], i_buffer)
    valid_time = np.isfinite(sorted_times) & (sorted_times < 1e5)
    if not np.any(valid_time):
        raise ValueError("No valid circular-buffer time values were available.")

    traces = {"time": sorted_times[valid_time]}
    for key in keys:
        if key not in storage:
            traces[key] = np.full(traces["time"].shape, np.nan, dtype=np.float64)
            continue

        values = np.asarray(storage[key], dtype=np.float64)
        if values.size < effective_limit:
            traces[key] = np.full(traces["time"].shape, np.nan, dtype=np.float64)
            continue

        traces[key] = _sorted_buffer(values[:effective_limit], i_buffer)[valid_time]
    return traces


def _history_end_index(storage: dict[str, Any]) -> int:
    time_history = np.asarray(storage["time_history"], dtype=np.float64)
    valid_idx = np.flatnonzero(np.isfinite(time_history) & (time_history < 1e5))
    if valid_idx.size == 0:
        raise ValueError("No valid time_history values were available for plotting.")
    latest_idx = valid_idx[int(np.argmax(time_history[valid_idx]))]
    return int(latest_idx + 1)


def _history_window(storage: dict[str, Any], keys: list[str], end_index: int, n_points: int) -> list[np.ndarray]:
    start_index = max(0, end_index - int(n_points))
    arrays = [np.asarray(storage[key], dtype=np.float64)[start_index:end_index] for key in keys]
    valid = np.ones(arrays[0].shape, dtype=bool)
    for values in arrays:
        valid &= np.isfinite(values) & (values < 1e5)
    if not np.any(valid):
        raise ValueError(f"No valid values were available in the requested history window: {keys}")
    return [values[valid] for values in arrays]


def _history_time_window(storage: dict[str, Any], keys: list[str], seconds: float) -> list[np.ndarray]:
    end_index = _history_end_index(storage)
    arrays = [np.asarray(storage[key], dtype=np.float64)[:end_index] for key in keys]
    time_values = arrays[0]
    valid = np.ones(time_values.shape, dtype=bool)
    for values in arrays:
        valid &= np.isfinite(values) & (values < 1e5)
    if not np.any(valid):
        raise ValueError(f"No valid values were available in the requested history window: {keys}")
    end_time = float(np.nanmax(time_values[valid]))
    valid &= time_values >= end_time - float(seconds)
    if not np.any(valid):
        raise ValueError(f"No valid values were available in the last {seconds:g} seconds: {keys}")
    return [values[valid] for values in arrays]


def _style_journal_axis(ax: Any, secondary_y: bool = False) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", width=1.0, length=4, colors="#303030")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["bottom"].set_linewidth(1.1)
    if secondary_y:
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_color("#555555")
        ax.spines["right"].set_linewidth(1.1)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    else:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color("#555555")
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.margins(y=0.08)


def _add_panel_letters(axes: np.ndarray) -> None:
    for letter, ax in zip("ABCDEFGH", axes):
        ax.text(
            -0.16,
            1.12,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="#303030",
            clip_on=False,
            in_layout=False,
            zorder=20,
        )


def _horizontal_target(ax: Any, value: float, label: str, color: str, linestyle: Any = (0, (2, 2))) -> None:
    if _finite(value):
        ax.axhline(value, linestyle=linestyle, linewidth=TARGET_LINEWIDTH, color=color, alpha=0.9, label=label)


def _vertical_target(ax: Any, value: float, label: str, color: str) -> None:
    if _finite(value):
        ax.axvline(value, linestyle=(0, (2, 2)), linewidth=TARGET_LINEWIDTH, color=color, alpha=0.9, label=label)


def _legend_above(ax: Any, handles: list[Any], labels: list[str]) -> None:
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=max(1, min(len(handles), 4)),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )


def _combined_legend(ax: Any, *other_axes: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    for other_ax in other_axes:
        more_handles, more_labels = other_ax.get_legend_handles_labels()
        handles.extend(more_handles)
        labels.extend(more_labels)
    _legend_above(ax, handles, labels)


def _atrial_targets_legend(ax: Any, pressure_ax: Any) -> None:
    volume_handles, volume_labels = ax.get_legend_handles_labels()
    pressure_handles, pressure_labels = pressure_ax.get_legend_handles_labels()
    if volume_handles:
        volume_legend = ax.legend(
            volume_handles,
            volume_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=max(1, min(len(volume_handles), 4)),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        ax.add_artist(volume_legend)
    if pressure_handles:
        ax.legend(
            pressure_handles,
            pressure_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=max(1, min(len(pressure_handles), 4)),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )


def _ventricular_pv_legend(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    top_handles, top_labels = handles[:4], labels[:4]
    pressure_handles, pressure_labels = handles[4:8], labels[4:8]
    if top_handles:
        top_legend = ax.legend(
            top_handles,
            top_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=max(1, min(len(top_handles), 4)),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        ax.add_artist(top_legend)
    if pressure_handles:
        ax.legend(
            pressure_handles,
            pressure_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=max(1, min(len(pressure_handles), 4)),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )


def _minute_ventilation_series(traces: dict[str, np.ndarray], parameters: dict[str, float]) -> np.ndarray:
    va_flow = traces["VAflow_store"]
    breath_period = traces["t1_store"] + traces["t2_store"]
    result = np.full_like(va_flow, np.nan, dtype=np.float64)
    valid = np.isfinite(va_flow) & np.isfinite(breath_period) & (breath_period > 0)
    if not np.any(valid):
        return result

    gv_dead = _parameter_value(parameters, "GV_dead", 0.0)
    v0_dead = _parameter_value(parameters, "V0_dead", 0.0)
    dead_space = gv_dead * va_flow + v0_dead
    result[valid] = (va_flow[valid] + dead_space[valid] / breath_period[valid]) * 60.0
    return result


def _gas_exchange_plot_keys(storage: dict[str, Any]) -> list[str]:
    return [label for label in REQUIRED_GAS_KEYS if label not in GAS_EXCHANGE_SKIPPED_LABELS and label in storage]


def _gas_exchange_legend_label(label: str) -> str:
    gas_labels = {"O2": "O_2", "CO2": "CO_2"}
    if label.startswith("Pd_"):
        _, compartment, gas = label.split("_", 2)
        return rf"$P_{{d,{compartment},{gas_labels.get(gas, gas)}}}$"
    if label.startswith("PA_"):
        gas = label.split("_", 1)[1]
        return rf"$P_{{A,{gas_labels.get(gas, gas)}}}$"
    return label


def write_result_targets_plot(
    output_path: Path,
    storage: dict[str, Any],
    raw_state: np.ndarray,
    parameters: dict[str, float],
    buffer_limit: int,
    state_name: str,
) -> None:
    plt = get_pyplot()
    targets = plot_targets_from_raw(raw_state)
    keys = [
        "time_since_beat_store",
        "finish_breath_time",
        "HR_store",
        "P_sa_store",
        "V_lv_store",
        "V_rv_store",
        "P_lv_store",
        "P_rv_store",
        "V_la_store",
        "V_ra_store",
        "P_la_store",
        "P_ra_store",
        "tidal_store",
        "VAflow_store",
        "t1_store",
        "t2_store",
        "Pa_O2_every_store",
        "Pa_CO2_every_store",
        "dP_lv_dt_store",
        "dP_rv_dt_store",
    ]
    traces = circular_traces(storage, keys, buffer_limit)
    minute_ventilation = _minute_ventilation_series(traces, parameters)
    time_values = traces["time"]
    if time_values.size == 0:
        raise ValueError("No valid circular-buffer time points were available for plotting.")

    step = max(1, int(np.ceil(time_values.size / 8000)))
    plot_slice = slice(None, None, step)
    t_plot = time_values[plot_slice]
    finite_time = time_values[np.isfinite(time_values)]
    time_axis_start = float(finite_time[0])
    time_axis_end = float(finite_time[-1])
    overview_start = max(time_axis_start, time_axis_end - RESULTS_OVERVIEW_WINDOW_SECONDS)
    atrial_start = max(time_axis_start, time_axis_end - RESULTS_ATRIAL_DETAIL_WINDOW_SECONDS)
    pv_slice = slice(None, None, max(1, int(np.ceil(time_values.size / 12000))))
    colors = PLOT_COLORS

    plt.rcParams.update(JOURNAL_RC_PARAMS)
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 14.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.09, hspace=0.14, wspace=0.04)
    axes = axes.ravel()
    secondary_axes = []
    overview_axes = []
    atrial_axes = []

    ax = axes[0]
    overview_axes.append(ax)
    ax.plot(t_plot, traces["P_sa_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=r"$P_{\mathrm{sa}}$")
    _horizontal_target(ax, _target(targets, "Systolic_Pressure"), TARGET_LABELS["Systolic_Pressure"], colors["rose_dark"])
    _horizontal_target(ax, _target(targets, "Diastolic_Pressure"), TARGET_LABELS["Diastolic_Pressure"], colors["rose_light"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$P_{\mathrm{sa}}$ (mmHg)")
    ax_hr = ax.twinx()
    secondary_axes.append(ax_hr)
    ax_hr.plot(t_plot, traces["HR_store"][plot_slice] * HEART_RATE_PLOT_SCALE, color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["Heart_Rate"])
    _horizontal_target(ax_hr, _target(targets, "Heart_Rate") * HEART_RATE_PLOT_SCALE, TARGET_MEAN_LABELS["Heart_Rate"], colors["teal_dark"])
    ax_hr.set_ylabel("HR (BPM)")
    _combined_legend(ax, ax_hr)

    ax = axes[1]
    ax.plot(traces["V_lv_store"][pv_slice], traces["P_lv_store"][pv_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label="LV")
    ax.plot(traces["V_rv_store"][pv_slice], traces["P_rv_store"][pv_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label="RV")
    _vertical_target(ax, _target(targets, "EDV"), TARGET_LABELS["EDV"], colors["rose_dark"])
    _vertical_target(ax, _target(targets, "ESV"), TARGET_LABELS["ESV"], colors["rose_light"])
    _vertical_target(ax, _target(targets, "Max_RV_Volume"), TARGET_LABELS["Max_RV_Volume"], colors["blue_dark"])
    _vertical_target(ax, _target(targets, "Min_RV_Volume"), TARGET_LABELS["Min_RV_Volume"], colors["blue_light"])
    _horizontal_target(ax, _target(targets, "Max_RV_Pressure"), TARGET_LABELS["Max_RV_Pressure"], colors["blue_dark"])
    _horizontal_target(ax, _target(targets, "Min_RV_Pressure"), TARGET_LABELS["Min_RV_Pressure"], colors["blue_light"])
    ax.set_xlabel(r"$V$ (mL)")
    ax.set_ylabel(r"$P$ (mmHg)")
    _ventricular_pv_legend(ax)

    ax = axes[2]
    ax.plot(traces["V_la_store"][pv_slice], traces["P_la_store"][pv_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label="LA")
    ax.plot(traces["V_ra_store"][pv_slice], traces["P_ra_store"][pv_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label="RA")
    ax.set_xlabel(r"$V$ (mL)")
    ax.set_ylabel(r"$P$ (mmHg)")
    _legend_above(ax, *ax.get_legend_handles_labels())

    ax = axes[3]
    ax.plot(t_plot, traces["V_ra_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=r"$V_{\mathrm{RA}}$")
    _horizontal_target(ax, _target(targets, "Min_RA_Volume"), TARGET_LABELS["Min_RA_Volume"], colors["rose_dark"])
    _horizontal_target(ax, _target(targets, "Max_RA_Volume"), TARGET_LABELS["Max_RA_Volume"], colors["lavender_dark"], linestyle=(0, (5, 2)))
    _horizontal_target(ax, _target(targets, "RA_Volume_Before_Atrial_Contraction"), TARGET_LABELS["RA_Volume_Before_Atrial_Contraction"], colors["rose_light"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$V_{\mathrm{RA}}$ (mL)")
    ax_p = ax.twinx()
    secondary_axes.append(ax_p)
    ax_p.plot(t_plot, traces["P_ra_store"][plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=r"$P_{\mathrm{RA}}$")
    # ax_p.plot(t_plot, traces["P_rv_store"][plot_slice], color=colors["lavender_dark"], linewidth=SOLID_LINEWIDTH, label=r"$P_{\mathrm{RV}}$")
    _horizontal_target(ax_p, _target(targets, "Max_RA_Pressure_Atrial_contraction"), TARGET_LABELS["Max_RA_Pressure_Atrial_contraction"], colors["blue_dark"])
    _horizontal_target(ax_p, _target(targets, "Max_RA_Pressure_Tricuspid_Opening"), TARGET_LABELS["Max_RA_Pressure_Tricuspid_Opening"], colors["teal_light"])
    ax_p.set_ylabel(r"$P (mmHg)")
    _atrial_targets_legend(ax, ax_p)
    atrial_axes.append(ax)

    ax = axes[4]
    ax.plot(t_plot, traces["V_la_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=r"$V_{\mathrm{LA}}$")
    _horizontal_target(ax, _target(targets, "Min_LA_Volume"), TARGET_LABELS["Min_LA_Volume"], colors["rose_dark"])
    _horizontal_target(ax, _target(targets, "Max_LA_Volume"), TARGET_LABELS["Max_LA_Volume"], colors["lavender_dark"], linestyle=(0, (5, 2)))
    _horizontal_target(ax, _target(targets, "LA_Volume_Before_Atrial_Contraction"), TARGET_LABELS["LA_Volume_Before_Atrial_Contraction"], colors["rose_light"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$V_{\mathrm{LA}}$ (mL)")
    ax_p = ax.twinx()
    secondary_axes.append(ax_p)
    ax_p.plot(t_plot, traces["P_la_store"][plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=r"$P_{\mathrm{LA}}$")
    # ax_p.plot(t_plot, traces["P_lv_store"][plot_slice], color=colors["lavender_dark"], linewidth=SOLID_LINEWIDTH, label=r"$P_{\mathrm{LV}}$")
    _horizontal_target(ax_p, _target(targets, "Max_LA_Pressure_Atrial_contraction"), TARGET_LABELS["Max_LA_Pressure_Atrial_contraction"], colors["blue_dark"])
    _horizontal_target(ax_p, _target(targets, "Max_LA_Pressure_Mitral_Opening"), TARGET_LABELS["Max_LA_Pressure_Mitral_Opening"], colors["teal_light"])
    ax_p.set_ylabel(r"$P_{\mathrm{LA}}$ (mmHg)")
    _atrial_targets_legend(ax, ax_p)
    atrial_axes.append(ax)

    ax = axes[5]
    ax.plot(t_plot, traces["dP_lv_dt_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=r"$\mathrm{d}P_{\mathrm{LV}}/\mathrm{d}t$")
    ax.plot(t_plot, traces["dP_rv_dt_store"][plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=r"$\mathrm{d}P_{\mathrm{RV}}/\mathrm{d}t$")
    _horizontal_target(ax, _target(targets, "LV_Pressure_Deriv"), TARGET_LABELS["LV_Pressure_Deriv"], colors["rose_dark"])
    _horizontal_target(ax, _target(targets, "RV_Pressure_Deriv"), TARGET_LABELS["RV_Pressure_Deriv"], colors["blue_dark"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\mathrm{d}P/\mathrm{d}t$ (mmHg/s)")
    _legend_above(ax, *ax.get_legend_handles_labels())
    atrial_axes.append(ax)

    ax = axes[6]
    overview_axes.append(ax)
    ax.plot(t_plot, traces["tidal_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["Tidal_Volume"])
    _horizontal_target(ax, _target(targets, "Tidal_Volume"), TARGET_MEAN_LABELS["Tidal_Volume"], colors["rose_dark"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inspired/Expired Volume (L)")
    ax_mv = ax.twinx()
    secondary_axes.append(ax_mv)
    ax_mv.plot(t_plot, minute_ventilation[plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["Minute_Ventilation"])
    _horizontal_target(ax_mv, _target(targets, "Minute_Ventilation"), TARGET_LABELS["Minute_Ventilation"], colors["teal_dark"])
    ax_mv.set_ylabel(r"$\dot{V}_E$ (L/min)")
    _combined_legend(ax, ax_mv)

    ax = axes[7]
    overview_axes.append(ax)
    ax.plot(t_plot, traces["Pa_O2_every_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["PaO2"])
    ax.plot(t_plot, traces["Pa_CO2_every_store"][plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["PaCO2"])
    _horizontal_target(ax, _target(targets, "PaO2"), TARGET_MEAN_LABELS["PaO2"], colors["rose_dark"])
    _horizontal_target(ax, _target(targets, "PaCO2"), TARGET_MEAN_LABELS["PaCO2"], colors["blue_dark"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$P_{\mathrm{a}O_2}$ / $P_{\mathrm{a}CO_2}$ (mmHg)")
    _legend_above(ax, *ax.get_legend_handles_labels())

    for axis in overview_axes:
        axis.set_xlim(overview_start, time_axis_end)
    for axis in atrial_axes:
        axis.set_xlim(atrial_start, time_axis_end)
    for axis in axes:
        _style_journal_axis(axis)
    for axis in secondary_axes:
        _style_journal_axis(axis, secondary_y=True)
    _add_panel_letters(axes)

    fig.suptitle(f"{state_name.capitalize()} refined MAP targets", fontsize=15)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"Wrote: {output_path}")


def write_activation_atrial_pv_plot(output_path: Path, storage: dict[str, Any], buffer_limit: int) -> None:
    plt = get_pyplot()
    try:
        end_index = _history_end_index(storage)
        time_activation, phi, phi_atr = _history_window(
            storage,
            ["time_history", "phi", "phi_atr"],
            end_index,
            ACTIVATION_HISTORY_POINTS,
        )
        _, vt_ra, p_ra, vt_la, p_la = _history_window(
            storage,
            ["time_history", "VT_ra", "P_ra", "VT_la", "P_la"],
            end_index,
            ATRIAL_PV_HISTORY_POINTS,
        )
    except (KeyError, ValueError):
        traces = circular_traces(
            storage,
            ["phi_store", "phi_atr_store", "V_ra_store", "P_ra_store", "V_la_store", "P_la_store"],
            buffer_limit,
        )
        time_activation = traces["time"]
        phi = traces["phi_store"]
        phi_atr = traces["phi_atr_store"]
        vt_ra = traces["V_ra_store"]
        p_ra = traces["P_ra_store"]
        vt_la = traces["V_la_store"]
        p_la = traces["P_la_store"]

    plt.rcParams.update(JOURNAL_RC_PARAMS)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.05, hspace=0.08)

    axes[0].plot(time_activation, phi, color=PLOT_COLORS["teal"], linewidth=FOCUS_LINEWIDTH, label="Ventricle Activation")
    axes[0].plot(time_activation, phi_atr, color=PLOT_COLORS["rose"], linewidth=FOCUS_LINEWIDTH, label="Atrial Activation")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Activation")
    axes[0].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D5D5D5")

    axes[1].plot(vt_ra, p_ra, color=PLOT_COLORS["teal"], linewidth=FOCUS_LINEWIDTH, label="RA")
    axes[1].plot(vt_la, p_la, color=PLOT_COLORS["rose"], linewidth=FOCUS_LINEWIDTH, label="LA")
    axes[1].set_xlabel("Volume (mL)")
    axes[1].set_ylabel("Atrial Pressure (mmHg)")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D5D5D5")

    for axis in axes:
        _style_journal_axis(axis)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_yticks(np.arange(0, 1.01, 0.25))
    fig.align_ylabels(axes)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote: {output_path}")


def write_gas_exchange_plot(output_path: Path, storage: dict[str, Any], buffer_limit: int) -> None:
    plt = get_pyplot()
    plt.rcParams.update(JOURNAL_RC_PARAMS)
    fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.06)

    plotted = False
    plot_keys = _gas_exchange_plot_keys(storage)
    if plot_keys:
        try:
            arrays = _history_time_window(storage, ["time_history", *plot_keys], GAS_EXCHANGE_WINDOW_SECONDS)
            time_values, value_arrays = arrays[0], arrays[1:]
            for idx, (label, values) in enumerate(zip(plot_keys, value_arrays)):
                ax.plot(
                    time_values,
                    values,
                    color=GAS_EXCHANGE_COLORS[idx % len(GAS_EXCHANGE_COLORS)],
                    linestyle="-",
                    linewidth=SOLID_LINEWIDTH,
                    label=_gas_exchange_legend_label(label),
                )
            plotted = True
        except (KeyError, ValueError):
            plotted = False

    if not plotted:
        traces = circular_traces(storage, ["Pa_O2_every_store", "Pa_CO2_every_store"], buffer_limit)
        time_values = traces["time"]
        valid_time = np.isfinite(time_values)
        end_time = float(np.nanmax(time_values[valid_time]))
        plot_mask = valid_time & (time_values >= end_time - GAS_EXCHANGE_WINDOW_SECONDS)
        ax.plot(
            time_values[plot_mask],
            traces["Pa_O2_every_store"][plot_mask],
            color=GAS_EXCHANGE_COLORS[0],
            linewidth=SOLID_LINEWIDTH,
            label=TARGET_LABELS["PaO2"],
        )
        ax.plot(
            time_values[plot_mask],
            traces["Pa_CO2_every_store"][plot_mask],
            color=GAS_EXCHANGE_COLORS[1],
            linewidth=SOLID_LINEWIDTH,
            label=TARGET_LABELS["PaCO2"],
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State Variables")
    ax.set_title("Evolution of Gas Exchange State Variables")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, facecolor="white", edgecolor="#D5D5D5")
    _style_journal_axis(ax)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote: {output_path}")


def write_run_model_paper_style_plots(
    output_dir: Path,
    outputs: SimulationOutputs,
    parameters: dict[str, float],
    buffer_limit: int,
) -> dict[str, Any]:
    plot_summary: dict[str, Any] = {}
    states = [
        ("rest", outputs.rest_storage, outputs.rest_hm_raw),
        ("exercise", outputs.exercise_storage, outputs.exercise_hm_raw),
    ]
    for state_name, storage, raw_state in states:
        state_summary: dict[str, Any] = {}
        plot_summary[state_name] = state_summary
        plot_specs = [
            (
                "results_targets",
                output_dir / f"Run_model_Paper_results_targets_{state_name}.png",
                lambda path, store=storage, raw=raw_state, state=state_name: write_result_targets_plot(
                    path,
                    store,
                    raw,
                    parameters,
                    buffer_limit,
                    state,
                ),
            ),
            # (
            #     "activation_atrial_pv",
            #     output_dir / f"Run_model_Paper_activation_atrial_pv_{state_name}.png",
            #     lambda path, store=storage: write_activation_atrial_pv_plot(path, store, buffer_limit),
            # ),
            # (
            #     "gas_exchange",
            #     output_dir / f"Run_model_Paper_gas_exchange_{state_name}.png",
            #     lambda path, store=storage: write_gas_exchange_plot(path, store, buffer_limit),
            # ),
        ]
        for plot_name, path, writer in plot_specs:
            try:
                writer(path)
                state_summary[plot_name] = str(path)
            except Exception as exc:
                state_summary[plot_name] = None
                state_summary[f"{plot_name}_error"] = str(exc)
                print(f"Skipping {plot_name} plot for {state_name}: {exc}")
    return plot_summary


def write_simulation_artifacts(
    output_dir: Path,
    outputs: SimulationOutputs,
    simulation_txt_name: str,
) -> Path:
    """Write the raw simulation vectors and the HM-compatible rest+exercise text
    file. Returns the text path so the residual plot can consume it. This runs
    before any plotting so the text file exists when the residual figure is made.
    """
    raw64 = np.concatenate([outputs.rest_full, outputs.exercise_full])
    np.save(output_dir / "refined_map_actual_simulation_raw64.npy", raw64)
    np.save(output_dir / "refined_map_actual_simulation_raw62.npy", outputs.raw_union_hm)
    np.save(output_dir / "refined_map_actual_simulation_aligned50.npy", outputs.aligned_union)
    np.save(output_dir / "refined_map_actual_simulation_rest_raw31.npy", outputs.rest_hm_raw)
    np.save(output_dir / "refined_map_actual_simulation_exercise_raw31.npy", outputs.exercise_hm_raw)

    simulation_txt_path = output_dir / simulation_txt_name
    write_simulation_text(simulation_txt_path, outputs.rest_hm_raw, outputs.exercise_hm_raw)
    return simulation_txt_path


def write_summary_json(
    run_dir: Path,
    output_dir: Path,
    outputs: SimulationOutputs,
    simulation_txt_path: Path,
    residual_summary: dict[str, Any] | None,
    paper_plot_summary: dict[str, Any] | None,
) -> None:
    summary = {
        "rest_converged": outputs.rest_converged,
        "exercise_converged": outputs.exercise_converged,
        "elapsed_seconds": outputs.elapsed_seconds,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "raw64_path": str(output_dir / "refined_map_actual_simulation_raw64.npy"),
        "raw62_path": str(output_dir / "refined_map_actual_simulation_raw62.npy"),
        "aligned50_path": str(output_dir / "refined_map_actual_simulation_aligned50.npy"),
        "simulation_txt": str(simulation_txt_path),
        "residual_summary": residual_summary,
        "paper_plot_summary": paper_plot_summary,
        "raw_output_names_per_state": RAW_OUTPUT_NAMES_PER_STATE,
        "dropped_raw_columns_per_state": list(RESULT_COLS_TO_DROP_PER_STATE),
    }
    with (output_dir / "refined_map_actual_simulation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


class _Tee:
    """Mirror text writes to a console stream and a log file at once."""

    def __init__(self, stream: Any, log_file: Any) -> None:
        self._stream = stream
        self._log = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    run_dir = (args.run_dir if args.run_dir is not None else SCRIPT_DIR / DEFAULT_RUN_NAME).resolve()
    output_dir = run_dir / DEFAULT_OUTPUT_FOLDER
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mirror everything printed during the run into Target_Trace/Run_model.log so
    # each run keeps a durable record (including any traceback) beside its outputs.
    log_path = output_dir / "Run_model.log"
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)
    try:
        _run(args, project_dir, run_dir, output_dir)
        print(f"Wrote: {log_path}")
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_file.close()


def _run(args: argparse.Namespace, project_dir: Path, run_dir: Path, output_dir: Path) -> None:
    print(f"Project dir: {project_dir}")
    print(f"Run dir: {run_dir}")
    print(f"Output dir: {output_dir}")

    parameters = load_refined_parameters(run_dir, project_dir)
    print(f"Loaded {len(parameters)} refined parameters.")

    cache_path = output_dir / args.cache_name
    cached = load_simulation_cache(cache_path, parameters, force_rerun=args.rerun or args.no_cache)
    if cached is None:
        pipeline = import_samples_pipeline(project_dir)
        configure_pipeline(pipeline, args)
        outputs = run_refined_map(pipeline, parameters, use_timeout=not args.no_timeout)
        buffer_limit = int(getattr(pipeline, "BUFFER_LIMIT", 80000))
        if not args.no_cache:
            save_simulation_cache(cache_path, outputs, parameters, buffer_limit)
            print(f"Wrote: {cache_path}")
    else:
        outputs, buffer_limit = cached
        print(f"Loaded cached simulation: {cache_path} (pass --rerun to re-simulate).")

    # Write the raw vectors and HM-compatible text first so the residual figure
    # (which consumes the text file) has it available.
    simulation_txt_path = write_simulation_artifacts(output_dir, outputs, args.simulation_txt)

    residual_summary = write_union_residual_plot(
        run_dir,
        output_dir,
        simulation_txt_path,
        make_plot=not args.no_plot,
    )

    paper_plot_summary = None
    if not args.no_paper_plots:
        paper_plot_summary = write_run_model_paper_style_plots(
            output_dir,
            outputs,
            parameters,
            buffer_limit=buffer_limit,
        )

    write_summary_json(
        run_dir,
        output_dir,
        outputs,
        simulation_txt_path,
        residual_summary,
        paper_plot_summary,
    )

    print()
    print(f"Rest converged: {outputs.rest_converged}")
    print(f"Exercise converged: {outputs.exercise_converged}")
    if cached is None:
        print(f"Simulation elapsed seconds: {outputs.elapsed_seconds:.1f}")
    else:
        print(f"Simulation: loaded from cache (original solve took {outputs.elapsed_seconds:.1f}s).")
    print(f"Wrote: {simulation_txt_path}")
    print(f"Wrote: {output_dir / 'refined_map_actual_simulation_aligned50.npy'}")
    if residual_summary is not None and "residual_plot" in residual_summary:
        print(f"Wrote: {residual_summary['residual_plot']}")
    if paper_plot_summary is not None:
        for state_name, state_summary in paper_plot_summary.items():
            written = [
                path
                for key, path in state_summary.items()
                if not key.endswith("_error") and isinstance(path, str)
            ]
            print(f"Paper-style plots for {state_name}: {len(written)} written")
    if residual_summary is not None and "n_outputs" in residual_summary:
        print(
            "Actual simulation within 1 SD: "
            f"{residual_summary['within_1_sd']}/{residual_summary['n_outputs']}"
        )
        print(
            "Actual simulation within 3 SD: "
            f"{residual_summary['within_3_sd']}/{residual_summary['n_outputs']}"
        )
        print(
            "Worst residual: "
            f"{residual_summary['worst_output']} = {residual_summary['worst_residual_sd']:.3f} SD"
        )


if __name__ == "__main__":
    main()
