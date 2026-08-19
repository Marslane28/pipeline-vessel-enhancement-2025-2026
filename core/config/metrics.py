from dataclasses import dataclass, field
from core.config.base import ConfigBase
from typing import Optional


@dataclass
class Metrics:
    dice: float
    mcc: float
    roc: float
    pr: float
    cldice: Optional[float] = None
    components_ratio: Optional[float] = None
    n_components_pred: Optional[int] = None
    n_components_gt: Optional[int] = None
    excess_components: Optional[int] = None
    largest_ratio: Optional[float] = None
    largest_gt_recall: Optional[float] = None
    largest_component_overlap: Optional[float] = None
    fragmentation_ratio: Optional[float] = None
    threshold: Optional[float] = None
    skeleton_component_connectivity: Optional[float] = None
    largest_component_recall: Optional[float] = None
    gt_fragmentation: Optional[float] = None
    pred_small_components: Optional[int] = None
    pred_medium_components: Optional[int] = None
    pred_large_components: Optional[int] = None
    gt_small_components: Optional[int] = None
    gt_medium_components: Optional[int] = None
    gt_large_components: Optional[int] = None
    operator_name: Optional[str] = None
    
    hessian_time_seconds: Optional[float] = None

    bifurcation_detection_rate: Optional[float] = None
    bifurcation_precision: Optional[float] = None
    n_bifurcations_gt: Optional[int] = None
    n_bifurcations_detected: Optional[int] = None
    n_bifurcations_pred: Optional[int] = None
    metrics_computed: list = field(default_factory=lambda: ["all"])