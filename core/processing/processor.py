
import threading
import numpy as np
import dask.array as da
from numpy import ndarray
from math import ceil
from dask.diagnostics import ProgressBar
from typing import Callable, Optional, Tuple, TYPE_CHECKING

from core.utils.helpers import normalize_data
from core.utils.gpu import is_gpu_available_available
from core.utils.black_ridges import detect_black_ridges
from core.config.experiment import HessianConfig, EnhancementConfig, ProcessingConfig, MethodsConfig, SegmentationConfig
from core.processing.derivator import Derivator
from core.processing.enhancer import Enhancer 
from core.processing.segmenter import Segmenter
from scipy.ndimage import binary_dilation

if TYPE_CHECKING:
    from core.config.benchmark import BenchmarkConfig

## Frangi et MFAT sont appliqués sur le volume complet.
# Seul jerman utilise encore un masque, et uniquement pour restreindre le
# calcul de son maximum global lambda3_max (régularisation lambda_rho) 
# ni en entrée ni en sortie.
MASKED_ENHANCEMENT_METHODS = {'jerman'}
HESSIAN_BASED_METHODS = {'frangi', 'jerman', 'mfat'}

class Processor:
    
    def __init__(self, config: ProcessingConfig):
        self.use_gpu = config.use_gpu and is_gpu_available_available()
        self.normalize = config.normalize
        self.parallelize = config.parallelize
        self.show_progress = config.show_progress
        self.overlap_size = config.overlap_size
        self.chunk_size = config.chunk_size

        self.derivator = Derivator(use_gpu=self.use_gpu)
        self.enhancer = Enhancer(use_gpu=self.use_gpu)
        self.segmenter = Segmenter()

    def enhance_data(self,
            data: ndarray,
            enhancement_function: Callable,
            enhancement_params: dict,
            mask: Optional[ndarray] = None,
        ):
        
        hessian_time = 0.0

        if self.parallelize and data.ndim == 3:
            chunk_size = self.chunk_size or tuple(ceil(s//2) for s in data.shape)
            overlap_size = self.overlap_size or max(enhancement_params.get('scales', [10]))
            chunk_hessian_times: list[float] = []
            times_lock = threading.Lock()

            def _enhancement_wrapper(chunk, **kw):
                result = enhancement_function(chunk, **kw)
                if isinstance(result, tuple) and len(result) == 2:
                    arr, chunk_time = result
                    if chunk_time is not None:
                        with times_lock:
                            chunk_hessian_times.append(chunk_time)
                    return arr
                return result

            data_dask = da.from_array(data, chunks=chunk_size)

            processed_chunks = da.map_overlap(
                _enhancement_wrapper,
                data_dask,
                depth=overlap_size,
                boundary='reflect',
                dtype=np.float32,
                **enhancement_params,
            )

            if self.show_progress:
                with ProgressBar():
                    data_enhanced = processed_chunks.compute(scheduler='threads')
            else:
                data_enhanced = processed_chunks.compute(scheduler='threads')
            hessian_time = float(sum(chunk_hessian_times)) if chunk_hessian_times else None
                
        else:
            kw = dict(enhancement_params)
            
            # Masque transmis tel quel : c'est enhance_data/process_data qui
            # décide en amont s'il faut ou non le passer (cf.
            # MASKED_ENHANCEMENT_METHODS dans process_data). Ici on se
            # contente de l'ajouter aux kwargs s'il est présent.
            if mask is not None:
                kw['mask'] = mask
            
            # Exécution
            result = enhancement_function(data, **kw)
            if isinstance(result, tuple) and len(result) == 2:
                data_enhanced, hessian_time = result
            else:
                data_enhanced = result

        # Normalisation (appliquée dans tous les cas)
        if self.normalize:
            data_enhanced = normalize_data(data_enhanced)
            
        return data_enhanced, hessian_time
    def prepare_enhancement(
            self,
            data: ndarray,
            hessian_config: HessianConfig,
            enhancement_config: EnhancementConfig,
            methods: MethodsConfig,
            mask_liver: Optional[ndarray] = None,
            ground_truth: Optional[ndarray] = None,
        ):
        hessian_function = self.derivator.select_hessian_function(
            methods.derivator
        )
        enhancement_function = self.enhancer.select_enhancement_function(
            methods.enhancer
        )

        if enhancement_config.black_ridges is None:
            black_ridges, black_ridges_info = detect_black_ridges(
                data,
                ground_truth=ground_truth,
            )
        else:
            black_ridges = enhancement_config.black_ridges
            black_ridges_info = "config fixe"
        hessian_params = hessian_config.to_dict()
        enhancement_config.black_ridges = black_ridges
        enhancement_config.hessian_function = hessian_function
        enhancement_config.hessian_params = hessian_params
        enhancement_params = enhancement_config.to_dict()

        scales_mm = enhancement_params.pop("scales_mm", None)

        if methods.enhancer == "mfat":
            enhancement_params.pop("tau", None)

            for cfg_key, fn_key in (
                ("mfat_tau", "tau"),
                ("mfat_tau2", "tau2"),
                ("mfat_step_size", "step_size"),
            ):
                value = enhancement_params.pop(cfg_key, None)
                if value is not None:
                    enhancement_params[fn_key] = value
        else:
            enhancement_params.pop("mfat_tau", None)
            enhancement_params.pop("mfat_tau2", None)
            enhancement_params.pop("mfat_step_size", None)

        mask_for_enhancement = (
            mask_liver
            if methods.enhancer in MASKED_ENHANCEMENT_METHODS
            else None
        )
        return (
            hessian_function,
            enhancement_function,
            hessian_params,
            enhancement_params,
            scales_mm,
            mask_for_enhancement,
        )
    
    def process_data(self,
        data: ndarray,
        hessian_config: HessianConfig,
        enhancement_config: EnhancementConfig,
        segmentation_config: SegmentationConfig,
        methods: MethodsConfig,
        ground_truth: Optional[ndarray] = None,
        mask_liver: Optional[ndarray] = None,
        benchmark_config: Optional['BenchmarkConfig'] = None,  
    ) -> Tuple[ndarray, ndarray, float, Optional[float]]:
        (
            # Préparation de l'enhancement
            hessian_function,
            enhancement_function,
            hessian_params,
            enhancement_params,
            scales_mm,
            mask_for_enhancement,
        ) = self.prepare_enhancement(
            data=data,
            hessian_config=hessian_config,
            enhancement_config=enhancement_config,
            methods=methods,
            mask_liver=mask_liver,
            ground_truth=ground_truth,
        )
        # Préparation Segmentation
        segmentation_function = self.segmenter.select_segmentation_function(
            methods.segmenter
        )
        segmentation_params = segmentation_config.to_dict()
        # Enhancement

        data_enhanced, hessian_time = self.enhance_data(
            data=data,
            enhancement_function=enhancement_function,
            enhancement_params=enhancement_params,
            mask=mask_for_enhancement,
        )
        # Segmentation
        data_segmented, threshold_value = segmentation_function(
            data=data_enhanced,
            ground_truth=ground_truth,
            mask=mask_liver,
            **segmentation_params,
        )
        return data_enhanced, data_segmented, threshold_value, hessian_time