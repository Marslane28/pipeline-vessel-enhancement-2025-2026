# core/config/benchmark.py

from dataclasses import dataclass, field
from typing import Any, Optional, Union, Literal
from numpy import ndarray

from core.config.base import ConfigBase
from core.config.experiment import Experiment
from core.config.setup import SetupConfig

import logging

logger = logging.getLogger(__name__)


# TYPES

BenchmarkResults = dict[str, dict[str, Experiment]]

RunnerResultsParsed = dict[
    str,
    dict[str, dict[Any, list[float]]],
]


# THRESHOLD CONFIG

@dataclass
class ThresholdingOptimConfig(ConfigBase):

    fixed_value: Optional[float] = None

    def is_fixed(self) -> bool:
        return self.fixed_value is not None


# GLOBAL OPTIMIZATION CONFIG

@dataclass
@dataclass
class OptimizationConfig(ConfigBase):
    """Configuration du seuillage et de la sélection des métriques calculées."""

    thresholding: ThresholdingOptimConfig = field(
        default_factory=ThresholdingOptimConfig
    )
    metrics: Union[Literal["all"], list] = "all"

    _VALID_METRICS = {
        "dice", "mcc", "recall", "specificity", "precision", "accuracy",
        "roc", "pr", "cldice", "components", "bifurcation",
    }

    def __post_init__(self) -> None:
        if self.metrics != "all":
            invalid = set(self.metrics) - self._VALID_METRICS
            if invalid:
                raise ValueError(
                    f"Métrique(s) invalide(s) dans optimization.metrics : {invalid}. "
                    f"Valeurs valides : {sorted(self._VALID_METRICS)}"
                )

    def to_detailed_metrics_selection(self):
        return self.metrics

# BENCHMARK CONFIG

@dataclass
class BenchmarkConfig(ConfigBase):

    mode: str # "hessian"| "enhancement"
    results_dir: str
    params: dict[str, list]
    params_grid: Optional[dict[str, Any]]
    optimize_scales: bool = False
    optimization: Optional[OptimizationConfig] = None

    def __post_init__(self) -> None:
        self._normalize_gamma_values()
        self._print_summary()

    def _print_summary(self) -> None:
        if self.optimization is None:
            return
        thresh = self.optimization.thresholding
        print("OPTIMIZATION CONFIG")
        print("="* 50)
        if thresh.is_fixed():
            print(f"Threshold: fixed = {thresh.fixed_value}")
        else:
            print("Threshold: optimized via F1 (or 0.5 if no GT)")
        print(f"Metrics computed: {self.optimization.metrics}")
        if self.params_grid:
            print(f"Grid search params: {list(self.params_grid.keys())}")
            for param, values in self.params_grid.items():
                print(f"{param}: {values}")

    def _normalize_gamma_values(self) -> None:
        if not self.params_grid or "gamma"not in self.params_grid:
            return
        gamma_values = self.params_grid["gamma"]
        if not isinstance(gamma_values, list):
            gamma_values = [gamma_values]
        self.params_grid["gamma"] = [
            None if str(v).lower() in {"null", "none", "auto"} else v
            for v in gamma_values
        ]

    # ----------------------------------------------------------
    # Threshold resolution délègue à Segmenter
    # ----------------------------------------------------------

    def resolve_threshold(
        self,
        segmenter,
        data: ndarray,
        ground_truth: Optional[ndarray] = None,
        mask: Optional[ndarray] = None,
    ) -> tuple[ndarray, float]:
        
        thresh_config = self.optimization.thresholding if self.optimization else ThresholdingOptimConfig()

        return segmenter.thresholding(
            data=data,
            threshold=thresh_config.fixed_value,
            ground_truth=ground_truth,
            mask=mask,
        )


# DATA CONTAINER
@dataclass
class BenchmarkData(ConfigBase):
    data_raw: ndarray
    data_gt: ndarray
    results: BenchmarkResults


# RUNNER CONFIG

@dataclass
class RunnerConfig(ConfigBase):
    setup: SetupConfig
    images_dir: str
    labels_dir: str
    masks_dir: Optional[str] = None
    patient_ids: Optional[Union[str, int, list]] = None