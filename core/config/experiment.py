# core/config/experiment.py

from numpy import ndarray
from typing import Literal, Callable, Sequence, Tuple, Optional
from dataclasses import dataclass
from core.config.base import ConfigBase
from core.config.metrics import Metrics

### CONFIGS

@dataclass
class LoadingConfig(ConfigBase):
    normalize: bool
    crop: bool
    target_shape: Sequence[int]
    raw_file: Optional[str] = None
    gt_file: Optional[str] = None
    mask_file: Optional[str] = None
    use_physical_units: Optional[bool] = False


@dataclass
class MethodsConfig(ConfigBase):
    derivator: Literal[
        'default', 'gaussian', 'farid', 'cubic',
        'trigonometric', 'catmull', 'bspline', 'bezier', 'scharr',
    ]
    enhancer: Literal['frangi', 'jerman', 'mfat']
    segmenter: Literal['thresholding']


@dataclass
class HessianConfig(ConfigBase):
    mode: Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    cval: float


@dataclass
class ProcessingConfig(ConfigBase):
    use_gpu: bool = True
    normalize: Optional[bool] = False
    parallelize: Optional[bool] = False
    show_progress: Optional[bool] = True
    overlap_size: Optional[int] = None
    chunk_size: Optional[Tuple[int, int, int]] = None

@dataclass
class EnhancementConfig(ConfigBase):
    alpha: float
    beta: float
    black_ridges: Optional[bool] = None

    scales: Optional[Sequence[float]] = None
    scales_mm: Optional[Sequence[float]] = None     
    gamma: Optional[float] = None
    skimage: Optional[bool] = False


    tau: float = 0.75
    hessian_function: Optional[Callable[..., list[ndarray]]] = None
    hessian_params: Optional[dict] = None

    mfat_tau: Optional[float] = None
    mfat_tau2: Optional[float] = None
    mfat_step_size: Optional[float] = None
    variant: Literal['eigen', 'proba'] = 'eigen'


@dataclass
class SegmentationConfig(ConfigBase):
    threshold: Optional[float] = None


@dataclass
class ExperimentConfig(ConfigBase):
    loading: LoadingConfig
    methods: MethodsConfig
    processing: ProcessingConfig
    hessian: HessianConfig
    enhancement: EnhancementConfig
    segmentation: SegmentationConfig


### EXPERIMENT

@dataclass
class Experiment(ConfigBase):
    data_enhanced: ndarray
    data_segmented: ndarray
    config: ExperimentConfig
    metrics: Optional[Metrics] = None
    id: Optional[str] = None
    cldice_score: Optional[float] = None
    conn_metrics: Optional[dict] = None
    data_stats: Optional[dict] = None
    threshold: Optional[float] = None