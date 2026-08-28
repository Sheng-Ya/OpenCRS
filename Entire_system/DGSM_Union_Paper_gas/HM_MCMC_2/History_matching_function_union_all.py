import logging
import multiprocessing
import warnings
import numpy as np
from scipy.stats import gaussian_kde
import os
import joblib
from joblib.externals.loky import get_reusable_executor
from autoemulate.core.model_selection import evaluate, r2_metric
from autoemulate.core.model_selection import bootstrap
from joblib.externals.loky import get_reusable_executor
import gc
import torch
# from tqdm import tqdm
# import tqdm_joblib

from joblib import Parallel, delayed
import math
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from torch.distributions.multivariate_normal import MultivariateNormal

# silence_resource_tracker_childprocesserror.py
import sys

_old_hook = sys.unraisablehook

def _quiet_resource_tracker(unraisable):
    exc = unraisable.exc_value
    obj = unraisable.object

    # Filter only: multiprocessing.resource_tracker ResourceTracker + ChildProcessError
    if isinstance(exc, ChildProcessError):
        if obj is not None:
            mod = getattr(obj.__class__, "__module__", "")
            name = getattr(obj.__class__, "__name__", "")
            if mod == "multiprocessing.resource_tracker" and name == "ResourceTracker":
                return  # swallow it

    _old_hook(unraisable)

sys.unraisablehook = _quiet_resource_tracker


from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.plotting import display_figure
from autoemulate.core.results import Result
from autoemulate.core.types import DeviceLike, DistributionLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators import TransformedEmulator, get_emulator_class
# from autoemulate.simulations.base import Simulator
from Simulator_Union import Simulator

logger = logging.getLogger("autoemulate")


def _resolve_emulator_model(model_or_name):
    """Accept both string model names and callable/class model references."""
    if isinstance(model_or_name, str):
        return get_emulator_class(model_or_name)
    if not isinstance(model_or_name, type):
        return model_or_name.__class__
    return model_or_name


def _transformed_emulator_kwargs(result, device):
    """Build constructor kwargs from either an AutoEmulate Result or TransformedEmulator."""
    model_or_name = getattr(result, "model_name", None)
    if not isinstance(model_or_name, str):
        model_or_name = getattr(result, "model", result)
        if not isinstance(model_or_name, str) and not isinstance(model_or_name, type):
            model_or_name = model_or_name.__class__

    params = getattr(result, "params", None)
    if params is None:
        params = getattr(result, "model_params", None)
    if params is None:
        params = {}

    return {
        "model": _resolve_emulator_model(model_or_name),
        "x_transforms": getattr(result, "x_transforms", None),
        "y_transforms": getattr(result, "y_transforms", None),
        "device": device,
        **params,
    }

INITIAL_EMULATOR_DIR = "Emulator_union_all_initial"
WAVE_EMULATOR_DIR = "Emulator_union_all_wave"
NROY_SAMPLES_PATH = "nroy_samples_union_all.pt"
LAST_WAVE_PATH = "last_wave_union_all.pt"
X_TRAIN_PATH = "X_train_union_all.pt"
Y_TRAIN_PATH = "Y_train_union_all.pt"
EMULATOR_TRAIN_N_JOBS = 64

RAW_OUTPUT_NAMES_PER_STATE = [
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV", "ESV",
    "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure", "Min_RV_Pressure",
    "Min_RA_Volume", "Max_RA_Volume", "Min_RA_Pressure_Atrial_descent",
    "Max_RA_Pressure_Atrial_contraction", "Max_RA_Pressure_Tricuspid_Opening",
    "Min_RA_Pressure_Tricuspid_descent", "Min_LA_Volume", "Max_LA_Volume",
    "Min_LA_Pressure_Atrial_descent", "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening", "Min_LA_Pressure_Mitral_descent",
    "Pre_LA_Contraction_Volume", "Pre_RA_Contraction_Volume",
    "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume",
    "Minute_Ventilation", "Cardiac_Output", "PaO2", "PaCO2",
    "Pericardial_Volume_Percentage_Change"
]
RAW_SIMULATION_OUTPUT_NAMES = [
    f"Rest_{name}" for name in RAW_OUTPUT_NAMES_PER_STATE
] + [
    f"Exercise_{name}" for name in RAW_OUTPUT_NAMES_PER_STATE
]
RAW_OUTPUT_COLUMNS_TO_DROP = [11, 14, 17, 20, 27, 30, 42, 45, 48, 51, 58, 61]

EMULATOR_OUTPUT_NAMES = [
            "Rest_Heart_Rate", "Rest_Systolic_Pressure", "Rest_Diastolic_Pressure", "Rest_EDV", "Rest_ESV",
            "Rest_Max_RV_Volume", "Rest_Min_RV_Volume", "Rest_Max_RV_Pressure", "Rest_Min_RV_Pressure",
            "Rest_Min_RA_Volume", "Rest_Max_RA_Volume", "Rest_Max_RA_Pressure_Atrial_contraction",
            "Rest_Max_RA_Pressure_Tricuspid_Opening", "Rest_Min_LA_Volume", "Rest_Max_LA_Volume",
            "Rest_Max_LA_Pressure_Atrial_contraction", "Rest_Max_LA_Pressure_Mitral_Opening",
            "Rest_Pre_LA_Contraction_Volume", "Rest_Pre_RA_Contraction_Volume", "Rest_LV_Pressure_Deriv",
            "Rest_RV_Pressure_Deriv", "Rest_Tidal_Volume", "Rest_Minute_Ventilation", "Rest_PaO2", "Rest_PaCO2",

            "Exercise_Heart_Rate", "Exercise_Systolic_Pressure", "Exercise_Diastolic_Pressure", "Exercise_EDV",
            "Exercise_ESV", "Exercise_Max_RV_Volume", "Exercise_Min_RV_Volume", "Exercise_Max_RV_Pressure",
            "Exercise_Min_RV_Pressure", "Exercise_Min_RA_Volume", "Exercise_Max_RA_Volume",
            "Exercise_Max_RA_Pressure_Atrial_contraction", "Exercise_Max_RA_Pressure_Tricuspid_Opening",
            "Exercise_Min_LA_Volume", "Exercise_Max_LA_Volume", "Exercise_Max_LA_Pressure_Atrial_contraction",
            "Exercise_Max_LA_Pressure_Mitral_Opening", "Exercise_Pre_LA_Contraction_Volume",
            "Exercise_Pre_RA_Contraction_Volume", "Exercise_LV_Pressure_Deriv", "Exercise_RV_Pressure_Deriv",
            "Exercise_Tidal_Volume", "Exercise_Minute_Ventilation", "Exercise_PaO2", "Exercise_PaCO2",
]

EMULATOR_OUTPUT_INDEX = {name: idx for idx, name in enumerate(EMULATOR_OUTPUT_NAMES)}
class HistoryMatching(TorchDeviceMixin):
    r"""
    History Matching class for model calibration.

    History matching is a model calibration method, which uses observed data to
    rule out ``implausible`` parameter values. The implausibility metric is:

    .. math::

        I_i(\bar{x_0}) = \frac{|z_i - \mathbb{E}(f_i(\bar{x_0}))|}
        {\sqrt{\text{Var}[z_i - \mathbb{E}(f_i(\bar{x_0}))]}}

    Queried parameters above a given implausibility threshold are ruled out (RO)
    whereas all other parameters are marked as not ruled out yet (NROY).
    """

    def __init__(
        self,
        observations: dict[str, tuple[float, float]] | dict[str, float],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        device: DeviceLike | None = None,
    ):
        """
        Initialize the history matching object.

        Parameters
        ----------
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise] (with noise
            specified as variances). In case of no uncertainty in observations, provides
            just the observed value.
        threshold: float
            Implausibility threshold (query points with implausibility scores that
            exceed this value are ruled out). Defaults to 3, which is considered
            a good value for simulations with a single output.
        model_discrepancy: float
            Additional variance to include in the implausibility calculation.
        rank: int
            Scoring method for multi-output problems. Must be 1 <= rank <= n_outputs.
            When the implausibility scores are ordered across outputs, it indicates
            which rank to use when determining whether the query point is NROY. The
            default of ``1`` indicates that the largest implausibility will be used.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        """
        TorchDeviceMixin.__init__(self, device=device)

        self.threshold = threshold
        self.discrepancy = model_discrepancy
        self.out_dim = len(observations)

        if rank > self.out_dim or rank < 1:
            raise ValueError(
                f"Rank ({rank}) is outside valid range between 1 and output dimension "
                f"of simulator ({self.out_dim})",
            )
        self.rank = rank

        # Save mean and variance of observations, shape: [1, n_outputs]
        self.obs_means, self.obs_vars = self._process_observations(observations)

    def _process_observations(
        self,
        observations: dict[str, tuple[float, float]] | dict[str, float],
    ) -> tuple[TensorLike, TensorLike]:
        """
        Turn observations into tensors of shape [1, n_inputs].

        Parameters
        ----------
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise] (with noise
            specified as variances). In case of no uncertainty in observations, provides
            just the observed value.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Tensors of observations and the associated noise (which can be 0) specified
            as variances.
        """
        values = torch.tensor(list(observations.values()), device=self.device)

        # No variance
        if values.ndim == 1:
            means = values
            variances = torch.zeros_like(means, device=self.device)
        # Values are (mean, variance)
        elif values.ndim == 2:
            means = values[:, 0]
            variances = values[:, 1]
        else:
            msg = "Observations must be either float or tuple of two floats."
            raise ValueError(msg)

        # Reshape observation tensors for broadcasting
        return means.view(1, -1), variances.view(1, -1)

    def _create_nroy_mask(self, implausibility: TensorLike) -> TensorLike:
        """
        Create mask for NROY points based on rank.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested parameters.

        Returns
        -------
        TensorLike
            Tensor indicating whether each implausability score is NROY
            given self.rank and self.threshold values.
        """
        # Sort implausibilities for each sample (descending)
        I_sorted, index_for_sort = torch.sort(implausibility, dim=1, descending=True)
        values, row_idx = torch.sort(I_sorted[:, 0], descending=True)
        implausibility_sorted_by_col0 = I_sorted[row_idx]
        index_of_implausibility_sorted_by_col0 = index_for_sort[row_idx]

        # The rank-th highest output implausibility must be <= threshold
        return I_sorted[:, self.rank - 1] <= self.threshold

    def get_nroy(
        self, implausibility: TensorLike, x: TensorLike | None = None
    ) -> TensorLike:
        """
        Get indices of NROY points from implausibility scores.

        If `x` is provided, returns parameter values at NROY indices.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested input parameters.
        x: Tensorlike | None
            Optional tensor of scored input parameters.

        Returns
        -------
        TensorLike
            Indices of NROY points or `x` parameters at NROY indices.
        """
        nroy_mask = self._create_nroy_mask(implausibility)
        idx = torch.where(nroy_mask)[0]
        if x is None:
            return idx
        return x[idx]

    def get_ro(
        self, implausibility: TensorLike, x: TensorLike | None = None
    ) -> TensorLike:
        """
        Get indices of RO points from implausibility scores.

        If `x` is provided, returns parameter values at RO indices.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested input parameters.
        x: Tensorlike | None
            Optional tensor of scored iput parameters.

        Returns
        -------
        TensorLike
            Indices of RO points or `x` parameters at RO indices.
        """
        nroy_mask = self._create_nroy_mask(implausibility)
        idx = torch.where(~nroy_mask)[0]
        if x is None:
            return idx
        return x[idx]

    def calculate_implausibility(
        self,
        pred_means: TensorLike,  # [n_samples, n_outputs]
        pred_vars: TensorLike,  # [n_samples, n_outputs]
    ) -> TensorLike:
        """
        Calculate implausibility scores.

        Parameters
        ----------
        pred_means: TensorLike
            Tensor of prediction means [n_samples, n_outputs]
        pred_vars: TensorLike
            Tensor of prediction variances [n_samples, n_outputs].

        Returns
        -------
        TensorLike
            Tensor of implausibility scores.
        """
        # Additional variance due to model discrepancy (defaults to 0)
        discrepancy = torch.full_like(
            self.obs_vars, self.discrepancy, device=self.device
        )
        # obs_vars is the obs uncertainty, eg HR 1.1, std 0.1. Discrepancy is the uncertainty of the emulator prediction in that area
        # Calculate total variance
        Vs = pred_vars + discrepancy + self.obs_vars

        # Calculate implausibility
        return torch.abs(self.obs_means - pred_means) / torch.sqrt(Vs)

    @staticmethod
    def _safe_ratio_denominator(denominator: TensorLike, eps: float = 1e-8) -> TensorLike:
        # Prevent divide-by-zero when max and min are very close.
        eps_tensor = torch.full_like(denominator, eps)
        signed_eps = torch.where(denominator < 0, -eps_tensor, eps_tensor)
        return torch.where(denominator.abs() < eps, signed_eps, denominator)

    @staticmethod
    def generate_param_bounds(
        nroy_x: TensorLike,
        buffer_ratio: float = 0.05,
        param_names: list[str] | None = None,
        min_samples: int = 1,
    ) -> dict[str, tuple[float, float]] | None:
        """
        Generate lower/upper parameter bounds as min/max of NROY samples.

        Parameters
        ----------
        nroy_x: TensorLike
            A tensor of NROY parameter samples [n_samples, n_inputs].
        buffer_ratio: float
            A scaling factor used to expand the bounds of the (NROY) parameter space.
            It is applied as a ratio of the range (max_val - min_val) of each input
            parameter to create a buffer around the NROY minimum and maximum values.
        param_names: list[str] | None
            Optional list of parameter names. If None, uses default `["x1", ..., "xn"]`.
        min_samples: int
            Minimum number of samples needed to generate new bounds.

        Returns
        -------
        dict[str, [float, float]] | None
            The generated [lower, upper] parameter bounds. Returns None if there are
            not enough samples to generate bounds from.
        """
        if param_names is None:
            param_names = [f"x{i + 1}" for i in range(nroy_x.shape[1])]

        if nroy_x.shape[0] > min_samples:
            min_val = torch.min(nroy_x, dim=0).values
            max_val = torch.max(nroy_x, dim=0).values
            buffer = (max_val - min_val) * buffer_ratio
            lower_bound = min_val - buffer
            upper_bound = max_val + buffer

            return {
                param: (lower_bound[i].item(), upper_bound[i].item())
                for i, param in enumerate(param_names)
            }
        return None


class HistoryMatchingWorkflow(HistoryMatching):
    """
    History Matching Workflow class.

    Run history matching workflow:
    - sample parameter values to test from the current NROY parameter space
    - use emulator to rule out implausible parameter samples
    - run simulations for a subset of the NROY parameters
    - refit the emulator using the simulated data
    """

    def __init__(
        self,
        simulator: Simulator,
        result: Result,
        observations: dict[str, tuple[float, float]] | dict[str, float],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        train_x: TensorLike | None = None,
        train_y: TensorLike | None = None,
        calibration_params: list[str] | None = None,
        rest_calibration_params: list[str] | None = None,
        exercise_calibration_params: list[str] | None = None,
        atrial_ratio_bounds: tuple[float, float] | None = None,
        atrial_ratio_min_probability: float = 0.0,
        atrial_ratio_mc_samples: int = 128,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
        log_level: str = "debug",
    ):
        """
        Initialize the history matching workflow object.

        Parameters
        ----------
        simulator: Simulator
            A simulator.
        result: Result
            A Result object containing the pre-trained emulator and its hyperparameters.
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise] (with noise
            specified as variances). In case of no uncertainty in observations, provides
            just the observed value.
        threshold: float
            Implausibility threshold (query points with implausibility scores that
            exceed this value are ruled out). Defaults to 3, which is considered
            a good value for simulations with a single output.
        model_discrepancy: float
            Additional variance to include in the implausibility calculation.
        rank: int
            Scoring method for multi-output problems. Must be 1 <= rank <= n_outputs.
            When the implausibility scores are ordered across outputs, it indicates
            which rank to use when determining whether the query point is NROY. The
            default val of ``1`` indicates that the largest implausibility will be used.
        train_x: TensorLike | None
            Optional tensor of input data the emulator was trained on.
        train_y: TensorLike | None
            Optional tensor of output data the emulator was trained on.
        calibration_params: list[str] | None
            Optional subset of parameters to calibrate. These have to correspond to the
            parameters that the emulator was trained on. If None, calibrate all
            simulator parameters.
        atrial_ratio_bounds: tuple[float, float] | None
            Optional acceptable range for the derived atrial contraction ratio.
        atrial_ratio_min_probability: float
            Minimum predictive probability required for the ratio to lie in range.
        atrial_ratio_mc_samples: int
            Monte Carlo samples used to propagate emulator uncertainty to the ratio.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        random_seed: int | None
            Optional random seed for reproducibility. If None, no seed is set.
        log_level: str
            The logging level to use. One of: "debug", "info", "warning", "error",
            "critical", "progress_bar" (default).
        """
        super().__init__(observations, threshold, model_discrepancy, rank, device)
        self.simulator = simulator
        if random_seed is not None:
            set_random_seed(seed=random_seed)
        self.logger, self.progress_bar = get_configured_logger(log_level)

        self.result = result
        self.emulator = result.model
        self.emulator.device = self.device

        # New data is simulated in `run()` and appended here
        # It can be used to refit the emulator
        if train_x is not None and train_y is not None:
            self.train_x = train_x.float().to(self.device)
            self.train_y = train_y.float().to(self.device)
        else:
            self.train_x = torch.empty((0, self.simulator.in_dim), device=self.device)
            self.train_y = torch.empty((0, len(EMULATOR_OUTPUT_NAMES)), device=self.device)

        # New NROY samples are generated in `run()` and used in `cloud_sample()`
        # We only ever use the most recent NROY samples
        # This means `self.nroy_samples` gets overwritten each time `run()` is called
        self.nroy_samples = None

        # If use `run_waves()`, results are stored here
        self.wave_results = []
        self._current_wave_idx: int | None = None
        self._save_wave_artifacts = True
        self._wave_artifacts_dir = "."
        self._last_wave_train_points: TensorLike | None = None

        # Save names and indices of parameters to calibrate.
        # All-union variant: every output emulator sees the same joint parameter
        # set. The rest/exercise-specific arguments are accepted for API
        # compatibility but intentionally ignored here.
        sim_param_names = list(simulator.parameters_range.keys())
        requested_params = set(calibration_params or sim_param_names)
        self.calibration_params = [n for n in sim_param_names if n in requested_params]
        self.rest_calibration_params = list(self.calibration_params)
        self.exercise_calibration_params = list(self.calibration_params)

        self.parameter_idx = [
            self.simulator.get_parameter_idx(param) for param in self.calibration_params
        ]
        self.rest_parameter_idx = [
            self.simulator.get_parameter_idx(param) for param in self.rest_calibration_params
        ]
        self.exercise_parameter_idx = [
            self.simulator.get_parameter_idx(param) for param in self.exercise_calibration_params
        ]
        # Derived atrial ratio is enforced as an interval constraint, not a point target.
        self.atrial_ratio_bounds = atrial_ratio_bounds
        self.atrial_ratio_min_probability = atrial_ratio_min_probability
        self.atrial_ratio_mc_samples = atrial_ratio_mc_samples

    @staticmethod
    def _to_numpy(array: TensorLike | np.ndarray) -> np.ndarray:
        if torch.is_tensor(array):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def _param_idx_for_output(self, output_name: str) -> list[int]:
        """Simulator-space parameter indices feeding `output_name`'s emulator.

        All-union variant: Rest_* and Exercise_* outputs both use the full
        combined calibration set.
        """
        return self.parameter_idx

    def _wave_number(self) -> int | None:
        if self._current_wave_idx is None:
            return None
        return self._current_wave_idx + 1

    def _save_wave_numpy_artifacts(
        self,
        test_x: TensorLike,
        impl_scores: TensorLike,
    ) -> None:
        if not self._save_wave_artifacts:
            return

        wave_number = self._wave_number()
        if wave_number is None:
            return

        os.makedirs(self._wave_artifacts_dir, exist_ok=True)
        nroy_mask = self._create_nroy_mask(impl_scores)
        nroy_points = test_x[nroy_mask]

        np.save(
            os.path.join(self._wave_artifacts_dir, f"test_params_union_all_wave_{wave_number}.npy"),
            self._to_numpy(test_x),
        )
        np.save(
            os.path.join(self._wave_artifacts_dir, f"impl_scores_union_all_wave_{wave_number}.npy"),
            self._to_numpy(impl_scores),
        )
        np.save(
            os.path.join(self._wave_artifacts_dir, f"nroy_mask_union_all_wave_{wave_number}.npy"),
            self._to_numpy(nroy_mask),
        )
        np.save(
            os.path.join(self._wave_artifacts_dir, f"nroy_points_union_all_wave_{wave_number}.npy"),
            self._to_numpy(nroy_points),
        )

        if self._last_wave_train_points is not None:
            np.save(
                os.path.join(self._wave_artifacts_dir, f"train_points_union_all_wave_{wave_number}.npy"),
                self._to_numpy(self._last_wave_train_points),
            )

    def _estimate_ratio_interval_probability(
        self,
        min_mean: TensorLike,
        min_var: TensorLike,
        max_mean: TensorLike,
        max_var: TensorLike,
        pre_mean: TensorLike,
        pre_var: TensorLike,
    ) -> TensorLike:
        if self.atrial_ratio_bounds is None:
            return torch.ones_like(min_mean)

        lower, upper = self.atrial_ratio_bounds
        n_mc = self.atrial_ratio_mc_samples

        # Sample from emulator marginals and estimate P(lower <= ratio <= upper).
        min_draws = min_mean[:, None] + min_var.clamp(min=0).sqrt()[:, None] * torch.randn(
            min_mean.shape[0], n_mc, device=min_mean.device, dtype=min_mean.dtype
        )
        max_draws = max_mean[:, None] + max_var.clamp(min=0).sqrt()[:, None] * torch.randn(
            max_mean.shape[0], n_mc, device=max_mean.device, dtype=max_mean.dtype
        )
        pre_draws = pre_mean[:, None] + pre_var.clamp(min=0).sqrt()[:, None] * torch.randn(
            pre_mean.shape[0], n_mc, device=pre_mean.device, dtype=pre_mean.dtype
        )

        denom = self._safe_ratio_denominator(max_draws - min_draws)
        ratio = (pre_draws - min_draws) / denom
        in_band = (max_draws > min_draws) & (ratio >= lower) & (ratio <= upper)
        return in_band.float().mean(dim=1)


    def _is_within_bounds(
        self, sample: TensorLike, bounds_dict: dict[str, tuple[float, float]]
    ) -> bool:
        """
        Check if `sample` is within the bounds defined in `bounds_dict`.

        Parameters
        ----------
        sample: torch.Tensor
            A single sample of input parameters to check, shape [1, in_dim].
        bounds_dict: dict of {param_name: [lower, upper]}
            A dictionary of parameter bounds for each parameter.

        Returns
        -------
        bool
            True if the sample is within the bounds, False otherwise.
        """
        sample = sample.squeeze(0)  # shape: [in_dim]
        lowers = torch.tensor(
            [bounds[0] for bounds in bounds_dict.values()],
            dtype=sample.dtype,
            device=sample.device,
        )
        uppers = torch.tensor(
            [bounds[1] for bounds in bounds_dict.values()],
            dtype=sample.dtype,
            device=sample.device,
        )
        return bool(torch.all((sample >= lowers) & (sample <= uppers)).item())


    def cloud_sample(self, n: int, scaling_factor: float = 0.1) -> TensorLike:
        """
        Generate `n` additional parameter samples using cloud sampling.

        Handles fixed parameters (min == max) by not sampling those. The constant
        values are inserted at the correct indices in the sampled tensor.

        Parameters
        ----------
        n: int
            The number of samples to generate.
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        TensorLike
            A tensor of sampled (and potentially constant) parameters [n, in_dim].
        """
        assert torch.is_tensor(self.nroy_samples)

        bounds = self.generate_param_bounds(self.nroy_samples, buffer_ratio=0.0)
        assert bounds is not None

        # Identify constant parameters
        min_vals = torch.tensor([b[0] for b in bounds.values()], device=self.device)
        max_vals = torch.tensor([b[1] for b in bounds.values()], device=self.device)
        is_constant = min_vals == max_vals
        constant_params = {
            i: min_vals[i].item() for i, fixed in enumerate(is_constant) if fixed
        }
        sample_params_idx = [i for i, fixed in enumerate(is_constant) if not fixed]

        # If all parameters are constant just return the constant sample n times
        if len(sample_params_idx) == 0:
            msg = "All parameters are constant, cannot sample from them."
            raise ValueError(msg)

        # Only use non-constant parameters for mean and covariance to sample from
        nroy_params_to_sample = self.nroy_samples[:, sample_params_idx]
        stdev = (
                        nroy_params_to_sample.max(dim=0).values
                        - nroy_params_to_sample.min(dim=0).values
                ) * scaling_factor
        # covariance_matrix = torch.diag(stdev ** 2)

        # Shuffle the order of means to sample from
        num_means = nroy_params_to_sample.shape[0]
        perm = torch.randperm(num_means, device=nroy_params_to_sample.device)

        # Determine how many samples to draw for each mean, handle remainder
        min_samples_per_mean = n // num_means
        remainder_to_sample = n % num_means

        # Determine number of parallel jobs
        n_jobs = 64 # multiprocessing.cpu_count()  # use all cores

        # Split permuted means into batches
        chunk_size = math.ceil(num_means / n_jobs)
        batches = [nroy_params_to_sample[perm][i:i + chunk_size] for i in range(0, num_means, chunk_size)]

        # precompute once outside the loop:
        low_all = torch.tensor([b[0] for b in bounds.values()], device=self.device)
        high_all = torch.tensor([b[1] for b in bounds.values()], device=self.device)
        low = low_all[sample_params_idx]
        high = high_all[sample_params_idx]
        std = stdev  # already [d_nonconst]

        # Precompute these once (outside sample_batch)
        sample_idx_t = torch.tensor(sample_params_idx, device=self.device, dtype=torch.long)

        if constant_params:
            const_idx_t = torch.tensor(list(constant_params.keys()), device=self.device, dtype=torch.long)
            const_vals_t = torch.tensor(list(constant_params.values()), device=self.device, dtype=low.dtype)
        else:
            const_idx_t, const_vals_t = None, None

        param_dim = len(bounds)

        def _phi(x):
            return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

        def _phi_inv(u):
            return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

        def truncated_normal_1d(mean, std, low, high, n_samples):
            """
            mean/std/low/high: [d]
            returns: [n_samples, d] all within [low, high]
            """
            eps = 1e-7
            std = torch.clamp(std, min=1e-12)

            a = (low - mean) / std
            b = (high - mean) / std

            pa = torch.clamp(_phi(a), eps, 1 - eps)
            pb = torch.clamp(_phi(b), eps, 1 - eps)

            # sample uniformly between CDF(low) and CDF(high)
            u = torch.rand((n_samples, mean.numel()), device=mean.device, dtype=mean.dtype)
            u = pa + u * (pb - pa)
            u = torch.clamp(u, eps, 1 - eps)

            z = _phi_inv(u)
            x = mean + std * z

            # numerical safety
            return torch.clamp(x, low, high)


        def sample_batch(batch, batch_idx):
            outs = []
            for j, mean in enumerate(batch):
                i = batch_idx * chunk_size + j
                n_samples = min_samples_per_mean + (1 if i < remainder_to_sample else 0)

                x_nonconst = truncated_normal_1d(mean, std, low, high, n_samples)  # [n_samples, d_nonconst]

                full = torch.empty((n_samples, param_dim), device="cpu", dtype=x_nonconst.dtype)
                if const_idx_t is not None:
                    full[:, const_idx_t] = const_vals_t.to(x_nonconst.dtype)
                full[:, sample_idx_t] = x_nonconst

                outs.append(full)

            # print(f"==============Batch {batch_idx + 1} done")
            return torch.cat(outs, dim=0) if outs else torch.empty((0, param_dim), device="cpu")

        results = Parallel(n_jobs=n_jobs)(
            delayed(sample_batch)(batch, idx) for idx, batch in enumerate(batches)
        )
        get_reusable_executor().shutdown(wait=True)
        print(f"==============Batch done")
        return torch.cat(results, dim=0)


    def pre_wave_train_emulators(self, n_simulations: int = 4096, refit_on_all_data: bool = False) -> None:
        """
        Pre-wave step: generate hybrid samples, run them through the simulator,
        train one emulator per rest/exercise output, and save them to
        Emulator_union_all_initial/.

        This must be called BEFORE run_waves(). It populates train_x / train_y
        and creates the initial emulators that wave 0 will load.

        Parameters
        ----------
        n_simulations: int
            Number of samples to generate, simulate, and train emulators on.
        refit_on_all_data: bool
            Whether to refit on all accumulated data (True) or just this batch.
        """
        x_train_path = X_TRAIN_PATH
        y_train_path = Y_TRAIN_PATH

        if os.path.exists(x_train_path) and os.path.exists(y_train_path):
            print("=" * 60)
            print(f"PRE-WAVE: Loading existing {x_train_path} and {y_train_path}")
            print("=" * 60)
            x = torch.load(x_train_path, map_location=self.device)
            y = torch.load(y_train_path, map_location=self.device)
        else:
            print("=" * 60)
            print("PRE-WAVE: Generating hybrid samples for initial emulator training")
            print("=" * 60)

            samples = self.simulator.sample_inputs(n_simulations).to(self.device)

            x, y = [], []
            for chunk in samples.split(2048):
                x_chunk, y_chunk = self.simulate(chunk)
                if x_chunk.shape[0] > 0:
                    x.append(x_chunk)
                    y.append(y_chunk)

            if not x:
                raise RuntimeError("Pre-wave simulation produced no valid rest/exercise training rows.")

            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
            torch.save(x.detach().cpu(), x_train_path)
            torch.save(y.detach().cpu(), y_train_path)

        x = x.float().to(self.device)
        y = y.float().to(self.device)
        self.train_x = x
        self.train_y = y

        # Train and save one emulator per output
        output_names_full = EMULATOR_OUTPUT_NAMES

        def fit_one_initial_output(j, target_name, X_fit_all, Y_fit_all, parameter_idx, result, device):
            if refit_on_all_data:
                X_fit = X_fit_all
                Y_fit = Y_fit_all[:, j:j + 1]
            else:
                X_fit = X_fit_all
                Y_fit = Y_fit_all[:, j:j + 1]

            x_fit = X_fit[:, parameter_idx]
            n = x_fit.shape[0]
            g = torch.Generator(device=x_fit.device)
            g.manual_seed(42)
            perm = torch.randperm(n, generator=g, device=x_fit.device)

            n_test = max(1, int(round(0.2 * n)))
            x_train, y_train = x_fit[perm[n_test:]], Y_fit[perm[n_test:]]
            x_test, y_test = x_fit[perm[:n_test]], Y_fit[perm[:n_test]]

            emulator = TransformedEmulator(
                x_train.float(),
                y_train.float(),
                **_transformed_emulator_kwargs(result, device),
            )
            emulator.fit(x_train, y_train)

            (r2_mean, r2_std), (rmse_mean, rmse_std) = bootstrap(
                emulator,
                x_test.float(),
                y_test.float(),
                n_bootstraps=100,
                device=device,
            )

            print(
                f"[{j + 1}/{len(output_names_full)}] {target_name} "
                f"R² test: {r2_mean:.4f} (±{r2_std:.4f}) | "
                f"RMSE test: {rmse_mean:.4f} (±{rmse_std:.4f})"
            )

            parent = os.path.join(INITIAL_EMULATOR_DIR, target_name)
            os.makedirs(parent, exist_ok=True)
            path1 = os.path.join(parent, f"GaussianProcessMatern32_{target_name}_best.joblib")
            joblib.dump(emulator, path1)
            # print(f"  Saved to {path1}")
            return target_name

        Parallel(n_jobs=EMULATOR_TRAIN_N_JOBS)(
            delayed(fit_one_initial_output)(
                j, target_name, self.train_x, self.train_y,
                self._param_idx_for_output(target_name), self.result, self.device,
            )
            for j, target_name in enumerate(output_names_full)
        )
        get_reusable_executor().shutdown(wait=True)

        print("=" * 60)
        print(f"PRE-WAVE: All emulators trained and saved to {INITIAL_EMULATOR_DIR}/")
        print("=" * 60)

    def _sample_within_bounds(
        self,
        dist: DistributionLike,
        bounds: dict[str, tuple[float, float]],
        n: int,
        constant_params: dict[int, float] | None = None,
        sample_params_idx: list[int] | None = None,
    ) -> list[TensorLike]:
        """
        Sample from distribution until `n` valid samples within the bounds are obtained.

        Handles constant parameters by inserting their values at the correct indices.

        Parameters
        ----------
        dist: DistributionLike
            A distribution to sample from, e.g., MultivariateNormal.
        bounds: dict[str, tuple[float, float]]
            A dictionary of [min, max] parameter bounds for each sampled parameter.
        n: int
            The number of samples to generate.
        constant_params: dict[int, float] | None
            A dictionary of constant parameter indices and their values.
        sample_params_idx: list[int]
            Indices of parameters that are not constant.

        Returns
        -------
        list[TensorLike]
            A list of valid samples that are within the bounds.
        """
        param_dim = len(bounds)
        if sample_params_idx is None:
            sample_params_idx = list(range(len(bounds)))

        valid_samples = []
        while len(valid_samples) < n:
            n_remaining = n - len(valid_samples)
            samples = dist.sample((n_remaining,))
            full = torch.empty(
                (n_remaining, param_dim),
                dtype=samples.dtype,
                device=samples.device,
            )
            if constant_params:
                const_idx = list(constant_params.keys())
                const_vals = torch.tensor(
                    list(constant_params.values()),
                    dtype=samples.dtype,
                    device=samples.device,
                )
                full[:, const_idx] = const_vals
            full[:, sample_params_idx] = samples
            valid_samples.extend([s for s in full if self._is_within_bounds(s, bounds)])
        return valid_samples


    def generate_samples(
        self, n: int, scaling_factor: float = 0.1
    ) -> tuple[TensorLike, TensorLike]:
        """
        Generate parameter samples and evaluate implausibility.

        Draw `n` samples either from the simulator min/max parameter bounds or
        using cloud sampling centered at NROY samples. Evaluate sample
        implausability using emulator predictions.

        Parameters
        ----------
        n: int
            The number of parameter samples to generate.
        scaling_factor: float
            The standard deviation of the Gaussian used in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausability scores.
        """
        use_raw_model = self.nroy_samples is None
        # Generate `n` parameter samples (use simulator if have no NROY samples)
        if use_raw_model:
            test_x = self.simulator.sample_inputs(n).to(self.device)
            parent = INITIAL_EMULATOR_DIR
        else:
            test_x = self.cloud_sample(n, scaling_factor).to(self.device)
            parent = WAVE_EMULATOR_DIR

        models = {}
        for name in EMULATOR_OUTPUT_NAMES:
            folder = name
            path1 = os.path.join(parent, folder, f"GaussianProcessMatern32_{name}_best.joblib")
            models[name] = joblib.load(path1)

        # means = {}
        # variances = {}
        #
        # for name in EMULATOR_OUTPUT_NAMES:
        #     target_emulator = models[name]
        #
        #     with torch.no_grad():
        #         means[name], variances[name] = target_emulator.predict_mean_and_variance(
        #             test_x[:, self.parameter_idx]
        #         )
        #     # means[name], variances[name] = target_emulator.predict_mean_and_variance(test_x[:, self.parameter_idx])

            #
        n_jobs = len(EMULATOR_OUTPUT_NAMES)
        def predict_one_output(name, X):
            target_emulator = models[name]

            mean, var = target_emulator.predict_mean_and_variance(X)
            return name, mean, var

        results = Parallel(n_jobs=n_jobs)(
            delayed(predict_one_output)(name, test_x[:, self._param_idx_for_output(name)])
            for name in EMULATOR_OUTPUT_NAMES
        )

        means = {name: mean for name, mean, var in results}
        variances = {name: var for name, mean, var in results}

        mean_tensor = torch.cat([means[name].reshape(-1, 1) for name in EMULATOR_OUTPUT_NAMES], dim=1)
        var_tensor = torch.cat([variances[name].reshape(-1, 1) for name in EMULATOR_OUTPUT_NAMES], dim=1)

        get_reusable_executor().shutdown(wait=True)

        # assert adjusted_var_tensor is not None
        impl_scores = self.calculate_implausibility(mean_tensor, var_tensor)
        if self.atrial_ratio_bounds is not None:
            la_ratio_probs = self._estimate_ratio_interval_probability(
                mean_tensor[:, 13], var_tensor[:, 13],
                mean_tensor[:, 14], var_tensor[:, 14],
                mean_tensor[:, 17], var_tensor[:, 17],
            )
            ra_ratio_probs = self._estimate_ratio_interval_probability(
                mean_tensor[:, 9], var_tensor[:, 9],
                mean_tensor[:, 10], var_tensor[:, 10],
                mean_tensor[:, 18], var_tensor[:, 18],
            )
            atrial_ratio_mask = (
                (la_ratio_probs >= self.atrial_ratio_min_probability)
                & (ra_ratio_probs >= self.atrial_ratio_min_probability)
            )

            la_ratio_probs_exercise = self._estimate_ratio_interval_probability(
                mean_tensor[:, 38], var_tensor[:, 38],
                mean_tensor[:, 39], var_tensor[:, 39],
                mean_tensor[:, 42], var_tensor[:, 42],
            )
            ra_ratio_probs_exercise = self._estimate_ratio_interval_probability(
                mean_tensor[:, 34], var_tensor[:, 34],
                mean_tensor[:, 35], var_tensor[:, 35],
                mean_tensor[:, 43], var_tensor[:, 43],
            )
            atrial_ratio_mask_exercise = (
                (la_ratio_probs_exercise >= self.atrial_ratio_min_probability)
                & (ra_ratio_probs_exercise >= self.atrial_ratio_min_probability)
            )
            # The ratio is now enforced through the interval-probability filter below.
            impl_scores[:, 17] = 0.0
            impl_scores[:, 18] = 0.0
            impl_scores[:, 42] = 0.0
            impl_scores[:, 43] = 0.0
        else:
            atrial_ratio_mask = torch.ones(
                mean_tensor.shape[0], dtype=torch.bool, device=mean_tensor.device
            )
            atrial_ratio_mask_exercise = torch.ones(
                mean_tensor.shape[0], dtype=torch.bool, device=mean_tensor.device
            )

        rest_min_ra = EMULATOR_OUTPUT_INDEX["Rest_Min_RA_Volume"]
        rest_max_ra = EMULATOR_OUTPUT_INDEX["Rest_Max_RA_Volume"]
        rest_min_la = EMULATOR_OUTPUT_INDEX["Rest_Min_LA_Volume"]
        rest_max_la = EMULATOR_OUTPUT_INDEX["Rest_Max_LA_Volume"]
        rest_pre_la = EMULATOR_OUTPUT_INDEX["Rest_Pre_LA_Contraction_Volume"]
        rest_pre_ra = EMULATOR_OUTPUT_INDEX["Rest_Pre_RA_Contraction_Volume"]
        exercise_min_ra = EMULATOR_OUTPUT_INDEX["Exercise_Min_RA_Volume"]
        exercise_max_ra = EMULATOR_OUTPUT_INDEX["Exercise_Max_RA_Volume"]
        exercise_min_la = EMULATOR_OUTPUT_INDEX["Exercise_Min_LA_Volume"]
        exercise_max_la = EMULATOR_OUTPUT_INDEX["Exercise_Max_LA_Volume"]
        exercise_pre_la = EMULATOR_OUTPUT_INDEX["Exercise_Pre_LA_Contraction_Volume"]
        exercise_pre_ra = EMULATOR_OUTPUT_INDEX["Exercise_Pre_RA_Contraction_Volume"]

        phys_mask = (
                (mean_tensor[:, rest_min_la] > 0.0)
                & (mean_tensor[:, rest_min_ra] > 0.0)
                & (mean_tensor[:, exercise_min_la] > 10.0)
                & (mean_tensor[:, exercise_min_ra] > 10.0)
                & (mean_tensor[:, rest_max_ra] > mean_tensor[:, rest_min_ra])
                & (mean_tensor[:, rest_max_la] > mean_tensor[:, rest_min_la])
                & (mean_tensor[:, exercise_max_ra] > mean_tensor[:, exercise_pre_ra])
                & (mean_tensor[:, exercise_max_la] > mean_tensor[:, exercise_pre_la])
                & (mean_tensor[:, rest_pre_la] > mean_tensor[:, rest_min_la])
                & (mean_tensor[:, rest_pre_ra] > mean_tensor[:, rest_min_ra])
                & (mean_tensor[:, exercise_pre_la] > mean_tensor[:, exercise_min_la])
                & (mean_tensor[:, exercise_pre_ra] > mean_tensor[:, exercise_min_ra])
                & atrial_ratio_mask
                & atrial_ratio_mask_exercise
        )

        impl_scores[~phys_mask] = 4

        mask = self._create_nroy_mask(impl_scores)

        if mask.any():
            min_rest_la = mean_tensor[mask, rest_min_la].min()
            min_exercise_ra = mean_tensor[mask, exercise_min_ra].min()
            min_exercise_la = mean_tensor[mask, exercise_min_la].min()
            min_exercise_pre_ra = mean_tensor[mask, exercise_pre_ra].min()
            min_exercise_pre_la = mean_tensor[mask, exercise_pre_la].min()

            print(f"min predicted Rest_Min_LA_Volume where NROY:", min_rest_la.item())
            print(f"min predicted Exercise_Min_RA_Volume where NROY:", min_exercise_ra.item())
            print(f"min predicted Exercise_Min_LA_Volume where NROY:", min_exercise_la.item())
            print(f"min predicted Exercise_Pre_RA_Contraction_Volume where NROY:", min_exercise_pre_ra.item())
            print(f"min predicted Exercise_Pre_LA_Contraction_Volume where NROY:", min_exercise_pre_la.item())
        else:
            print(f"No NROY samples found below threshold {self.threshold}.")

        return test_x, impl_scores

    def sample_tensor(
        self,
        n: int,
        x: TensorLike,
        return_indices: bool = False,
    ) -> TensorLike | tuple[TensorLike, TensorLike]:
        """
        Randomly sample `n` rows from `x`.

        Parameters
        ----------
        n: int
            The number of samples to draw.
        x: TensorLike
            The tensor to sample from.
        return_indices: bool
            Whether to also return sampled row indices from `x`.

        Returns
        -------
        TensorLike
            A tensor of samples with `n` rows.
        """
        if x.shape[0] < n:
            warnings.warn(
                f"Number of tensor rows {x.shape[0]} is less than {n} samples.",
                stacklevel=2,
            )
        idx = torch.randperm(x.shape[0], device=x.device)[:n]
        samples = x[idx]
        if return_indices:
            return samples, idx
        return samples

    @staticmethod
    def _row_membership_mask(rows: TensorLike, reference_rows: TensorLike) -> TensorLike:
        """Return a mask showing which rows occur in reference_rows."""
        mask = torch.zeros(rows.shape[0], dtype=torch.bool, device=rows.device)
        if rows.shape[0] == 0 or reference_rows.shape[0] == 0:
            return mask
        for start in range(0, reference_rows.shape[0], 64):
            ref_chunk = reference_rows[start:start + 64].to(rows.device)
            mask |= (rows[:, None, :] == ref_chunk[None, :, :]).all(dim=2).any(dim=1)
        return mask

    def simulate(self, x: TensorLike) -> tuple[TensorLike, TensorLike]:
        """
        Simulate `x` parameter inputs and filter out failed simulations.

        Parameters
        ----------
        x: TensorLike
            A tensor of parameters to simulate [n_samples, n_inputs].

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Tensors of succesfully simulated input parameters and predictions.
        """
        requested_n = x.shape[0]
        # if simulation fails, returned y and x have fewer rows than input x
        y, x = self.simulator.forward_batch(x)
        get_reusable_executor().shutdown(wait=True)

        y = y.to(self.device)
        x = x.to(self.device)
        print(f"simulate: {x.shape[0]}/{requested_n} samples left after simulator success filter")

        if y.numel() == 0:
            empty_y = torch.empty((0, len(EMULATOR_OUTPUT_NAMES)), device=self.device)
            self.train_x = x
            self.train_y = empty_y
            print("simulate: 0 samples left after all filters")
            return x, empty_y

        expected_raw_outputs = len(RAW_SIMULATION_OUTPUT_NAMES)
        if y.shape[1] != expected_raw_outputs:
            raise ValueError(
                f"Union simulator returned {y.shape[1]} columns; expected "
                f"{expected_raw_outputs} raw columns before dropping to "
                f"{len(EMULATOR_OUTPUT_NAMES)} targets."
            )

        # Drop output columns
        cols_to_drop = torch.tensor(RAW_OUTPUT_COLUMNS_TO_DROP, device=self.device)
        keep_mask = torch.ones(y.shape[1], dtype=torch.bool, device=self.device)
        keep_mask[cols_to_drop] = False
        y = y[:, keep_mask]

        # Remove non-finite rows first
        finite_mask = torch.isfinite(y).all(dim=1)
        x = x[finite_mask]
        y = y[finite_mask]
        print(f"simulate: {x.shape[0]} samples left after finite filter")
        if y.shape[0] == 0:
            self.train_y = y
            self.train_x = x
            print("simulate: 0 samples left after all filters")
            return x, y

        rest_min_ra = EMULATOR_OUTPUT_INDEX["Rest_Min_RA_Volume"]
        rest_max_ra = EMULATOR_OUTPUT_INDEX["Rest_Max_RA_Volume"]
        rest_min_la = EMULATOR_OUTPUT_INDEX["Rest_Min_LA_Volume"]
        rest_max_la = EMULATOR_OUTPUT_INDEX["Rest_Max_LA_Volume"]
        rest_pre_la = EMULATOR_OUTPUT_INDEX["Rest_Pre_LA_Contraction_Volume"]
        rest_pre_ra = EMULATOR_OUTPUT_INDEX["Rest_Pre_RA_Contraction_Volume"]
        exercise_min_ra = EMULATOR_OUTPUT_INDEX["Exercise_Min_RA_Volume"]
        exercise_max_ra = EMULATOR_OUTPUT_INDEX["Exercise_Max_RA_Volume"]
        exercise_min_la = EMULATOR_OUTPUT_INDEX["Exercise_Min_LA_Volume"]
        exercise_max_la = EMULATOR_OUTPUT_INDEX["Exercise_Max_LA_Volume"]
        exercise_pre_la = EMULATOR_OUTPUT_INDEX["Exercise_Pre_LA_Contraction_Volume"]
        exercise_pre_ra = EMULATOR_OUTPUT_INDEX["Exercise_Pre_RA_Contraction_Volume"]

        phys_mask = (
                (y[:, rest_min_la] > 0.0)
                & (y[:, rest_min_ra] > 0.0)
                & (y[:, exercise_min_la] > 10.0)
                & (y[:, exercise_min_ra] > 10.0)
                & (y[:, rest_max_ra] > y[:, rest_min_ra])
                & (y[:, rest_max_la] > y[:, rest_min_la])
                & (y[:, exercise_max_ra] > y[:, exercise_pre_ra])
                & (y[:, exercise_max_la] > y[:, exercise_pre_la])
                & (y[:, rest_pre_la] > y[:, rest_min_la])
                & (y[:, rest_pre_ra] > y[:, rest_min_ra])
                & (y[:, exercise_pre_la] > y[:, exercise_min_la])
                & (y[:, exercise_pre_ra] > y[:, exercise_min_ra])
        )

        y = y[phys_mask]
        x = x[phys_mask]
        print(f"simulate: {x.shape[0]} samples left after physiology filter")
        if y.shape[0] == 0:
            self.train_y = y
            self.train_x = x
            print("simulate: 0 samples left after all filters")
            return x, y

        # 3-sigma outlier filter (columnwise)
        col_mean = y.mean(dim=0)
        col_std = y.std(dim=0, unbiased=False)
        within = (y >= (col_mean - 3 * col_std)) & (y <= (col_mean + 3 * col_std))
        row_mask = within.all(axis=1)
        print(f"simulate: {(~row_mask).sum()} samples is removed by 3-sigma filter")
        x = x[row_mask]
        y = y[row_mask]

        self.train_y = y
        self.train_x = x

        return x, y

    def refit_emulator(self, x: TensorLike, y: TensorLike) -> None:
        """
        Refit the emulator on the provided data.

        Parameters
        ----------
        x: TensorLike
            Tensor of input data to refit the emulator on.
        y: TensorLike
            Tensor of output data to refit the emulator on.
        """

        # create test and train data
        n = x.shape[0]
        g = torch.Generator(device=x.device)
        g.manual_seed(42)
        perm = torch.randperm(n, generator=g, device=x.device)

        n_test = max(1, int(round(0.2 * n)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        # Create a fresh model with the same configuration
        self.emulator = TransformedEmulator(
            x_train.float(),
            y_train.float(),
            **_transformed_emulator_kwargs(self.result, self.device),
        )

        self.emulator.fit(x_train, y_train)
        # with torch.no_grad():
        #     y_pred = self.emulator.predict_mean(x.float())  # uses transforms internally
        # r2 = evaluate(y_pred, y.float(), r2_metric())
        # print("R² test:", float(r2))

        (r2_mean, r2_std), (rmse_mean, rmse_std) = bootstrap(
            self.emulator,
            x_test.float(),
            y_test.float(),
            n_bootstraps=100,  # or None for single split behaviour (if supported)
            device=self.device,
        )

        print(f"R² test: {r2_mean:.4f} (±{r2_std:.4f}) | RMSE test: {rmse_mean:.4f} (±{rmse_std:.4f})")
        # y_pred, variance = self.emulator.predict_mean_and_variance(x)
        # y_np = y.detach().cpu().numpy().reshape(-1)
        # y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        # r2 = r2_score(y_np, y_pred_np)
        # print(y_pred[:5,:])
        # print(f"R² = {r2:.4f}")

    def run(
        self,
        n_simulations: int = 100,
        n_test_samples: int = 10000,
        max_retries: int = 3,
        scaling_factor: float = 0.1,
        refit_emulator: bool = True,
        refit_on_all_data: bool = True,
    ) -> tuple[TensorLike, TensorLike]:
        """
        Run a wave of the history matching workflow.

        Parameters
        ----------
        n_simulations: int
            Number of simulations to run.
        n_test_samples: int
            Number of input parameters to test for implausibility with the emulator.
            Parameters to simulate are sampled from this NROY subset.
        max_retries: int
            Maximum number of times to try to generate `n_simulations` NROY parameters.
            That is the maximum number of times to repeat the following steps:
                - draw `n_test_samples` parameters (use cloud sampling if possible)
                - use emulator to make predictions for those parameters
                - score implausability of parameters given predictions
                - identify NROY parameters within this set
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.
        refit_emulator: bool
            Whether to refit the emulator at the end of the run. Defaults to True.
        refit_on_all_data: bool
            Whether to refit the emulator on all available data or just the data
            available from the most recent simulation run. Defaults to True.
        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores from
            which simulation samples were then selected.
        """

        msg = (
            f"Running history matching wave with {n_simulations} simulations and "
            f"{n_test_samples} test samples"
        )
        logger.debug(msg)
        self._last_wave_train_points = None

        test_parameters_list, impl_scores_list, nroy_parameters_list = (
            [],
            [],
            [torch.empty((0, self.simulator.in_dim), device=self.device)],
        )

        retries = 0
        nroy_total = 0
        while nroy_total < n_simulations:
            if retries == max_retries:
                msg = (
                    f"Could not generate n_simulations ({n_simulations}) samples "
                    f"that are NROY after {max_retries} retries. "
                    f"Only {torch.cat(nroy_parameters_list, 0).shape[0]} "
                    "samples generated."
                )
                raise RuntimeError(msg)

            if retries > 10:
                scaling_factor = 0.05

            # Generate `n_test_samples` with implausability scores, identify NROY
            test_parameters, impl_scores = self.generate_samples(
                n_test_samples, scaling_factor
            )

            # print("done getting the impl_score")
            # test parameters is a concatenation of every parameter set from before
            nroy_parameters = self.get_nroy(impl_scores, test_parameters)

            # print("done getting the nroy from self.get_nroy")

            # Store results (test_parameters_list will have as many entries as 200000 * no. of retries)
            nroy_parameters_list.append(nroy_parameters)
            test_parameters_list.append(test_parameters)
            impl_scores_list.append(impl_scores)
            nroy_total += nroy_parameters.shape[0]

            msg = (
                f"Generated {nroy_parameters.shape[0]} NROY samples on try "
                f"{retries + 1}, have {torch.cat(nroy_parameters_list, 0).shape[0]} "
                f"total NROY samples so far."
            )
            logger.debug(msg)

            retries += 1

        # # Next time that call run(), will sample using these NROY points
        self.nroy_samples = torch.cat(nroy_parameters_list, 0)
        nroy_simulation_samples, nroy_simulation_idx = self.sample_tensor(
            n_simulations,
            self.nroy_samples,
            return_indices=True,
        )
        print(f"Training simulations: {nroy_simulation_samples.shape[0]} NROY samples")
        # nroy_params = torch.cat(nroy_parameters_list, dim=0)
        #
        # implaus_tensor = torch.cat(impl_scores_list, 0)
        # nroy_impl = self.get_nroy(implaus_tensor, implaus_tensor)
        #
        # # Rank by worst-output implausibility per sample
        # max_impl_per_sample, _ = nroy_impl.max(dim=1)
        # best_idx = torch.argsort(max_impl_per_sample)
        # nroy_params_sorted = nroy_params[best_idx]
        #
        # # Take the best n_simulations to run through the simulator
        # nroy_simulation_samples = nroy_params_sorted[:n_simulations]
        #
        # # Also update nroy_samples so cloud sampling next wave uses best seeds
        # self.nroy_samples = nroy_params_sorted

        # np.save("check.npy", nroy_simulation_samples)
        # A = np.load("check.npy")[64:66]
        # print(A[:,-4:])
        # A = torch.from_numpy(A)

        # Make predictions using simulator (this updates self.x_train and self.y_train)
        x, y = self.simulate(nroy_simulation_samples)
        if x.shape[0] == 0 or y.shape[0] == 0:
            raise RuntimeError("No valid simulated union targets were produced for emulator training.")

        valid_sample_mask = self._row_membership_mask(nroy_simulation_samples, x)
        rejected_sample_idx = nroy_simulation_idx[~valid_sample_mask]
        if rejected_sample_idx.numel() > 0:
            keep_nroy_mask = torch.ones(
                self.nroy_samples.shape[0], dtype=torch.bool, device=self.nroy_samples.device
            )
            keep_nroy_mask[rejected_sample_idx] = False
            before_prune = self.nroy_samples.shape[0]
            self.nroy_samples = self.nroy_samples[keep_nroy_mask]
            print(
                f"Removed {rejected_sample_idx.numel()} simulated NROY sample(s) "
                f"that failed post-simulation filters; "
                f"{self.nroy_samples.shape[0]}/{before_prune} NROY seed samples remain."
            )
        else:
            print(
                f"Removed 0 simulated NROY sample(s) after post-simulation filters; "
                f"{self.nroy_samples.shape[0]} NROY seed samples remain."
            )

        min_rest_la = y[:, EMULATOR_OUTPUT_INDEX["Rest_Min_LA_Volume"]].min()
        min_exercise_ra = y[:, EMULATOR_OUTPUT_INDEX["Exercise_Min_RA_Volume"]].min()
        min_exercise_la = y[:, EMULATOR_OUTPUT_INDEX["Exercise_Min_LA_Volume"]].min()
        min_exercise_pre_ra = y[:, EMULATOR_OUTPUT_INDEX["Exercise_Pre_RA_Contraction_Volume"]].min()
        min_exercise_pre_la = y[:, EMULATOR_OUTPUT_INDEX["Exercise_Pre_LA_Contraction_Volume"]].min()
        print(
            "training atrial minima:",
            min_rest_la,
            min_exercise_ra,
            min_exercise_la,
            min_exercise_pre_ra,
            min_exercise_pre_la,
        )

        if not refit_emulator:
            return torch.cat(test_parameters_list, 0), torch.cat(impl_scores_list, 0)

        # # Keep only simulations whose outputs are within 3 observation standard
        # # deviations of the observation means. `obs_vars` stores variances.
        # obs_means = self.obs_means.to(device=y.device, dtype=y.dtype)
        # obs_stds = torch.sqrt(torch.clamp(self.obs_vars.to(device=y.device, dtype=y.dtype), min=0.0))
        # obs_3sd_mask = (
        #     (y >= obs_means - self.threshold * obs_stds)
        #     & (y <= obs_means + self.threshold * obs_stds)
        # ).all(dim=1)
        #
        # if not bool(obs_3sd_mask.all()):
        #     rejected_x = x[~obs_3sd_mask]
        #     x = x[obs_3sd_mask]
        #     y = y[obs_3sd_mask]
        #     self.train_x = x
        #     self.train_y = y
        #
        #     # Remove each rejected simulated parameter set from the NROY cloud too,
        #     # so it cannot seed samples for the next emulator wave.
        #     keep_nroy_mask = torch.ones(
        #         self.nroy_samples.shape[0], dtype=torch.bool, device=self.device
        #     )
        #     for rejected in rejected_x:
        #         matches = torch.where(
        #             keep_nroy_mask & torch.all(self.nroy_samples == rejected, dim=1)
        #         )[0]
        #         if matches.numel() > 0:
        #             keep_nroy_mask[matches[0]] = False
        #     self.nroy_samples = self.nroy_samples[keep_nroy_mask]
        #     print(
        #         f"Removed {rejected_x.shape[0]} simulated sample(s) outside "
        #         f"the observation +/- {self.threshold} std band from training and NROY samples."
        #     )

        # Save on CPU so it's portable across machines/devices. This happens after
        # filtering so resume/cloud sampling uses the updated NROY set.
        torch.save(self.nroy_samples.detach().cpu(), NROY_SAMPLES_PATH)
        self._last_wave_train_points = x.detach().cpu()

        output_names_full = EMULATOR_OUTPUT_NAMES
        wave_number = self._wave_number()
        snapshot_root = (
            os.path.join(self._wave_artifacts_dir, f"Emulator_union_all_wave_{wave_number}")
            if self._save_wave_artifacts and wave_number is not None
            else None
        )

        def fit_one_output(j, target_name, X_fit, Y_fit, parameter_idx, result, device):
            x_fit = X_fit[:, parameter_idx]
            y_fit = Y_fit[:, j:j + 1]

            n = x_fit.shape[0]
            g = torch.Generator(device=x_fit.device)
            g.manual_seed(42)
            perm = torch.randperm(n, generator=g, device=x_fit.device)

            n_test = max(1, int(round(0.2 * n)))
            x_train, y_train = x_fit[perm[n_test:]], y_fit[perm[n_test:]]
            x_test, y_test = x_fit[perm[:n_test]], y_fit[perm[:n_test]]

            emulator = TransformedEmulator(
                x_train.float(), y_train.float(),
                **_transformed_emulator_kwargs(result, device),
            )
            emulator.fit(x_train, y_train)

            (r2_mean, r2_std), (rmse_mean, rmse_std) = bootstrap(
                emulator,
                x_test.float(),
                y_test.float(),
                n_bootstraps=100,  # or None for single split behaviour (if supported)
                device=device,
            )

            print(
                f"{target_name}: R² test: {r2_mean:.4f} (±{r2_std:.4f}) | RMSE test: {rmse_mean:.4f} (±{rmse_std:.4f})")

            # save
            parent = os.path.join(WAVE_EMULATOR_DIR, target_name)
            # parent = os.path.join("Emulator_wave_V_tot", target_name)
            os.makedirs(parent, exist_ok=True)
            #######################################
            with torch.no_grad():
                y_test_emulator_mean, y_test_emulator_variance = emulator.predict_mean_and_variance(
                    x_test.float()
                )

            numpy_artifacts = {
                "x_train.npy": x_train,
                "y_train.npy": y_train,
                "x_test.npy": x_test,
                "y_test.npy": y_test,
                "y_test_emulator_mean.npy": y_test_emulator_mean,
                "y_test_emulator_variance.npy": y_test_emulator_variance,
            }
            for filename, array in numpy_artifacts.items():
                np.save(os.path.join(parent, filename), array.detach().cpu().numpy())
            #############################################
            model_filename = f"GaussianProcessMatern32_{target_name}_best.joblib"
            joblib.dump(emulator, os.path.join(parent, model_filename))
            if snapshot_root is not None:
                snapshot_parent = os.path.join(snapshot_root, target_name)
                os.makedirs(snapshot_parent, exist_ok=True)
                ############
                for filename, array in numpy_artifacts.items():
                    np.save(os.path.join(snapshot_parent, filename), array.detach().cpu().numpy())
                ############
                joblib.dump(emulator, os.path.join(snapshot_parent, model_filename))

            return target_name, emulator

        results = Parallel(n_jobs=EMULATOR_TRAIN_N_JOBS)(
            delayed(fit_one_output)(
                j, target_name, x, y,
                self._param_idx_for_output(target_name), self.result, self.device,
            )
            for j, target_name in enumerate(output_names_full)
        )
        get_reusable_executor().shutdown(wait=True)
        # for j, target_name in enumerate(output_names_full):
        #     # Optionally refit the emulator using the most recent simulations or all data
        #     if refit_emulator:
        #         # data_msg = "all data" if refit_on_all_data else "most recent data"
        #         # msg = f"Refitting emulator on {data_msg}."
        #         # logger.info(msg)
        #         if refit_on_all_data:
        #             X_fit = self.train_x
        #             Y_fit = self.train_y[:, j:j+1]
        #             self.refit_emulator(X_fit[:, self.parameter_idx], Y_fit)
        #         else:
        #             X_fit = x
        #             Y_fit = y[:, j:j+1]
        #             self.refit_emulator(X_fit[:, self.parameter_idx], Y_fit)
        #
        #     parent = os.path.join("Emulator_wave", target_name)
        #     os.makedirs(parent, exist_ok=True)
        #
        #     path1 = os.path.join(parent, f"GaussianProcessMatern32_{target_name}_best.joblib")
        #     joblib.dump(self.emulator, path1)

        # torch.save(x, f"X_train_wave_{(len(self.wave_results) - 1)}_rest_.pt")
        # torch.save(y, f"Y_train_wave_{(len(self.wave_results) - 1)}_rest_.pt")

        # Return test parameters and impl scores for this run/wave
        return torch.cat(test_parameters_list, 0), torch.cat(impl_scores_list, 0)

    def run_waves(
        self,
        n_waves: int = 5,
        frac_nroy_stop: float = 0.9,
        n_simulations: int = 100,
        n_test_samples: int = 10000,
        max_retries: int = 3,
        scaling_factor: float = 0.1,
        refit_emulator_on_last_wave: bool = True,
        refit_on_all_data: bool = True,
        resume_wave: bool = False,
        save_wave_artifacts: bool = True,
        wave_artifacts_dir: str = ".",
    ) -> list[tuple[TensorLike, TensorLike]]:
        """
        Run multiple waves of the history matching workflow.

        Refits the emulator after each wave (except the last), using all available data.

        Parameters
        ----------
        n_waves: int
            The maximum number of waves to run.
        frac_nroy_stop: float
            Fraction of NROY samples to stop at. If less than this fraction of
            NROY samples is reached, the workflow stops.
        n_simulations: int
            Number of simulations to run in each wave.
        n_test_samples: int
            Number of input parameters to test for implausibility with the emulator.
            Parameters to simulate are sampled from this NROY subset.
        max_retries: int
            Maximum number of times to try to generate `n_simulations` NROY parameters.
            That is the maximum number of times to repeat the following steps:
                - draw `n_test_samples` parameters (use cloud sampling if possible)
                - use emulator to make predictions for those parameters
                - score implausibility of parameters given predictions
                - identify NROY parameters within this set
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.
        refit_emulator_on_last_wave: bool
            Whether to refit the emulator after the last wave. Defaults to True.
        refit_on_all_data: bool
            Whether to refit the emulator on all available data after each wave
            or just the data from the most recent simulation run. Defaults to True.
        save_wave_artifacts: bool
            Whether to save per-wave emulator snapshots and `.npy` artifacts.
        wave_artifacts_dir: str
            Directory where per-wave artifacts are written.
        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores.
        """
        if resume_wave == True:
            self.nroy_samples = torch.load(NROY_SAMPLES_PATH, map_location="cpu").to(self.device)
            last_wave = int(torch.load(LAST_WAVE_PATH, map_location="cpu"))
            start_i = last_wave + 1
            print(start_i)
        else:
            start_i = 0

        self.wave_results = []
        self._save_wave_artifacts = save_wave_artifacts
        self._wave_artifacts_dir = wave_artifacts_dir
        if self._save_wave_artifacts:
            os.makedirs(self._wave_artifacts_dir, exist_ok=True)
        for i in range(start_i, n_waves):
            # 0th wave had 155173
            # if i == 0: # 110599
            #     self.threshold = 3.5 # change
            # if i == 1:  # 154081
            #     self.threshold = 3.5
            # if i == 1: # 154081
            #     self.threshold = 3.25
            #     n_test_samples = 200000
            if i > 0:
                self.threshold = 3
                n_test_samples = 200000

            # if i == 1: # 110599
            #     self.threshold = 1.5
            # if i == 2: # 110599
            #     self.threshold = 1.25
            # if i == 3: # 110599
            #     self.threshold = 1.125
            # if i == 4: # 154081
            #     self.threshold = 1.0
            # if i == 5: # 49467
            #     self.threshold = 1.0
            #     n_simulations = 5000

            logger.info("Running history matching wave %d/%d", i + 1, n_waves)
            self._current_wave_idx = i
            refit_emulator = i != n_waves - 1 or refit_emulator_on_last_wave
            test_x, impl_scores = self.run(
                n_simulations=n_simulations,
                n_test_samples=n_test_samples,
                max_retries=max_retries,
                scaling_factor=scaling_factor,
                refit_emulator=refit_emulator,
                refit_on_all_data=refit_on_all_data,
            )

            if len(test_x) < n_simulations or len(impl_scores) < n_simulations:
                msg = (
                    f"Not enough parameters or impl scores generated in wave {i + 1}"
                    f"/{n_waves}. Stopping history matching workflow. Results are "
                    f"stored until wave {i}/{n_waves}."
                )
                logger.warning(msg)
                break

            self.wave_results.append((test_x, impl_scores))
            # self.plot_wave((len(self.wave_results) - 1), fname=f"200000_wave_{(len(self.wave_results) - 1)}_rest.png")

            # Get NROY points from impl scores and check fraction
            self._save_wave_numpy_artifacts(test_x, impl_scores)
            nroy_x = self.get_nroy(impl_scores, test_x)
            nroy_frac = nroy_x.shape[0] / test_x.shape[0]
            logger.info(
                "Wave %d/%d: NROY fraction is %.2f%%",
                i + 1,
                n_waves,
                nroy_frac * 100,
            )

            torch.save(int(i), LAST_WAVE_PATH)

            if nroy_frac > frac_nroy_stop:
                logger.info(
                    "Stopping history matching workflow at wave %d/%d "
                    "with NROY fraction %.2f%% > %.2f%%",
                    i + 1,
                    n_waves,
                    nroy_frac * 100,
                    frac_nroy_stop * 100,
                )
                break

        self._current_wave_idx = None
        return self.wave_results

    def plot_run(
        self,
        test_parameters: TensorLike,
        impl_scores: TensorLike,
        set_simulator_axis_limits: bool = True,
        ref_val: dict[str, float] | None = None,
        title: str = "History Matching Results",
        fname: str | None = None,
    ) -> None | Figure:
        """
        Plot results of a single history matching run.

        Parameters
        ----------
        test_parameters: TensorLike
            A tensor of tested input parameters [n_samples, n_inputs].
        impl_scores: TensorLike
            A tensor of implausibility scores for the tested input parameters.
        set_simulator_axis_limits: bool
            Whether to keep the simulator parameter ranges as axis limits.
        ref_val:dict[str, float] | None
            Optional dictionary of true parameter values to mark on the plots.
        title: str
            Title for the plot.
        fname: str | None
            Optional filename to save the plot to. If None, the plot is displayed.

        Returns
        -------
        None | Figure
            If `fname` is provided, saves the plot to the file and returns None.
            If `fname` is None, displays the plot and returns the plot figure.
        """
        test_parameters_plausible = self.get_nroy(impl_scores, test_parameters)
        impl_scores_plausible = self.get_nroy(impl_scores, impl_scores)

        df = pd.DataFrame(
            test_parameters_plausible[:, self.parameter_idx],
            columns=self.calibration_params,  # pyright: ignore[reportArgumentType]
        )
        df["Implausibility"] = impl_scores_plausible.cpu().numpy().mean(axis=1)
        g = sns.PairGrid(df, vars=self.calibration_params, corner=True)

        norm = Normalize(
            vmin=df["Implausibility"].min(),  # pyright: ignore[reportArgumentType]
            vmax=df["Implausibility"].max(),  # pyright: ignore[reportArgumentType]
        )
        cmap = plt.cm.get_cmap("viridis")

        # added
        n_params = len(self.calibration_params)
        ncols = 4
        nrows = int(np.ceil(n_params / ncols))

        plt.rcParams.update({
            "font.size": 26,  # base font size
            "axes.titlesize": 28,  # subplot title
            "axes.labelsize": 26,  # axis labels
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "axes.linewidth": 2.5,
            "lines.linewidth": 2.2,
        })

        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows), sharey=True)
        axes = axes.flatten()

        for i, param in enumerate(self.calibration_params):
            ax = axes[i]
            x = df[param].to_numpy()
            y = df["Implausibility"].to_numpy()

            # Compute 2D density using Gaussian KDE
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]  # sort for clean layering

            sc = ax.scatter(x, y, c=z, cmap=cmap, s=15, alpha=0.8)

            if ref_val is not None and param in ref_val:
                ax.axvline(ref_val[param], color="red", linestyle="--", label="True value")

            if set_simulator_axis_limits:
                ax.set_xlim(self.simulator.parameters_range[param])

            ax.set_xlabel(param)
            if i % ncols == 0:
                ax.set_ylabel("Implausibility")
            ax.grid(True, linestyle="--", alpha=0.3)

        # Hide unused subplots if parameter count not divisible by ncols
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # cbar = fig.colorbar(sc, ax=axes, shrink=0.7, label="Point density")

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if fname is None:
            return display_figure(fig)
        fig.savefig(fname, bbox_inches="tight")
        return None

        # def scatter_continuous(x, y, **kwargs):
        #     ax = plt.gca()
        #     sc = ax.scatter(
        #         x,
        #         y,
        #         c=df.loc[x.index, "Implausibility"],
        #         cmap=cmap,
        #         norm=norm,
        #         s=15,
        #         alpha=0.7,
        #     )
        #     # Set axis limits if available
        #     if set_simulator_axis_limits:
        #         ax.set_xlim(self.simulator.parameters_range[x.name])
        #         ax.set_ylim(self.simulator.parameters_range[y.name])
        #     return sc
        #
        # def diag_hist(x, **kwargs):
        #     ax = plt.gca()
        #     sns.histplot(x, kde=False, color="gray", ax=ax)
        #     # Set axis limits if available
        #     if set_simulator_axis_limits:
        #         ax.set_xlim(self.simulator.parameters_range[x.name])
        #
        # g.map_lower(scatter_continuous)
        # g.map_diag(diag_hist)
        #
        # # Add reference points
        # if ref_val is not None:
        #     for i, parami in enumerate(self.calibration_params):
        #         for j, paramj in enumerate(self.calibration_params):
        #             if j < i:  # lower triangle only
        #                 ax = g.axes[i, j]
        #                 ax.scatter(
        #                     ref_val[paramj],
        #                     ref_val[parami],
        #                     color="white",
        #                     s=60,
        #                     edgecolor="black",
        #                     marker="X",
        #                     zorder=5,
        #                     label=(
        #                         "True value"
        #                         if (i == len(self.calibration_params) - 1 and j == 0)
        #                         else None
        #                     ),
        #                 )
        #
        # # Colorbar
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # plt.colorbar(sm, ax=plt.gcf().axes, shrink=0.7, label="Implausibility")
        #
        # # Global legend (handles all subplots)
        # handles, labels = g.axes[-1, 0].get_legend_handles_labels()
        # g.fig.legend(handles, labels, loc="upper right", frameon=True)
        # g.fig.suptitle(title, fontsize=16)
        #
        # if fname is None:
        #     return display_figure(g.fig)
        # g.savefig(fname, bbox_inches="tight")
        # return None

    def plot_wave(
        self,
        wave: int,
        set_simulator_axis_limits: bool = True,
        ref_val: dict[str, float] | None = None,
        fname: str | None = None,
    ) -> None | Figure:
        """
        Plot results for a specific wave.

        Parameters
        ----------
        wave: int
            The wave number to plot (0-indexed).
        set_simulator_axis_limits: bool
            Whether to keep the simulator parameter ranges as axis limits.
        ref_val: dict[str, float] | None
            Optional dictionary of true parameter values to mark on the plots.
        fname: str | None
            Optional filename to save the plot to. If None, the plot is displayed.

        Returns
        -------
        None | Figure
            If `fname` is provided, saves the plot to the file and returns None.
            If `fname` is None, displays the plot and returns the plot figure.
        """
        test_parameters, impl_scores = self.get_wave_results(wave)
        return self.plot_run(
            test_parameters,
            impl_scores,
            set_simulator_axis_limits,
            ref_val,
            f"Results for Wave {wave}",
            fname,
        )

    def get_wave_results(self, wave: int) -> tuple[TensorLike, TensorLike]:
        """
        Get results for a specific wave.

        Parameters
        ----------
        wave: int
            The wave number to get results for (0-indexed).

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores.
        """
        assert self.wave_results, "No wave results, run `run_waves()` first."
        assert 0 <= wave < len(self.wave_results), f"Wave {wave} not available."

        return self.wave_results[wave]

    def plot_wave_evolution(
        self, param, ref_val: dict[str, float] | None = None, fname: str | None = None
    ) -> None | Figure:
        """
        Plot evolution of parameter distributions across all waves.

        Parameters
        ----------
        param: str
            The parameter to plot the evolution for.
        ref_val: dict[str, float] | None
            Optional dictionary of true parameter values to mark on the plots.
        fname: str | None
            Optional filename to save the plot to. If None, the plot is displayed.

        Returns
        -------
        None | Figure
            If `fname` is provided, saves the plot to the file and returns None.
            If `fname` is None, displays the plot and returns the plot figure.
        """
        all_df = []
        for wave_idx, (test_parameters, impl_scores) in enumerate(self.wave_results):
            test_parameters_plausible = self.get_nroy(impl_scores, test_parameters)
            impl_scores_plausible = self.get_nroy(impl_scores, impl_scores)

            # Create DataFrame
            df = pd.DataFrame(
                test_parameters_plausible[:, self.parameter_idx],
                columns=self.calibration_params,  # pyright: ignore[reportArgumentType]
            )
            df["Implausibility"] = impl_scores_plausible.mean(axis=1)  # pyright: ignore[reportCallIssue]
            df["Wave"] = wave_idx

            all_df.append(df)

        # Concatenate all waves into a single DataFrame
        result_df = pd.concat(all_df, ignore_index=True)

        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(data=result_df, x="Wave", y=param)

        # Add horizontal line at true value
        if ref_val is not None:
            plt.axhline(
                ref_val[param],
                color="red",
                linestyle="--",
                linewidth=2,
                label="True value",
            )

        plt.title(f"Distribution of {param} by Wave")
        plt.xlabel("Wave")
        plt.ylabel(param)
        plt.tight_layout()

        # Add global legend only once (first plot)
        plt.legend(loc="upper right", frameon=True)

        if fname is None:
            return display_figure(fig)
        plt.savefig(f"{param}_wave_evolution.png", dpi=300, bbox_inches="tight")
        return None
