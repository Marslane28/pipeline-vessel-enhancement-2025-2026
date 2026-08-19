import os
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Optional
from core.utils.helpers import normalize_data

import nibabel as nib
from numpy import ndarray
from scipy.ndimage import label as scipy_label

_module_logger = logging.getLogger(__name__)

from core.config.benchmark import BenchmarkData
from core.config.experiment import (
    EnhancementConfig,
    Experiment,
    ExperimentConfig,
    HessianConfig,
    MethodsConfig,
    ProcessingConfig,
    SegmentationConfig,
)
from core.config.figure import FigureData
from core.experiments.metrics import detect_bifurcations, bifurcation_detection_rate
from core.experiments.analytics.hessian import AnalyticsHessian
from core.experiments.benchmarks.base import BenchmarkBase
from core.experiments.metrics import (
    mcc,
    print_confusion_matrix,
    detailed_metrics,
)

from core.io.loader import Loader
from core.io.saver import Saver
from core.processing.processor import Processor
from core.utils.searcher import GridSearcher
from core.config.benchmark import BenchmarkConfig


def _crop_around(data: ndarray, center: np.ndarray, size: int = 128) -> ndarray:
    """Return a cubic sub-volume centred on *center*."""
    slices = []
    for c, s in zip(center, data.shape):
        start = max(0, c - size // 2)
        end = min(s, start + size)
        start = max(0, end - size)
        slices.append(slice(start, end))
    return data[tuple(slices)]


def _liver_center(data_raw: ndarray, data_mask: Optional[ndarray]) -> np.ndarray:
    """Return the centre of the liver mask, or the volume centre."""
    if data_mask is not None and data_mask.any():
        return np.array(np.where(data_mask > 0)).mean(axis=1).astype(int)
    return np.array(data_raw.shape) // 2


def _safe_corrcoef(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def _to_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Picklable standalone functions required by GridSearcher for update_function and eval_function. They are defined outside the class to avoid pickling issues with instance methods.

def _gs_update_function(
    combination: dict,
    data_gt: ndarray,
    data_mask: ndarray,
    cached_eigenvalues,
    cached_ratios,
    image_shape,
    xp,
    mask_arr,
    gt_cache: dict,
    experiment_config: ExperimentConfig,
    benchmark_config: Optional[BenchmarkConfig] = None,
) -> dict:

    exp_config = deepcopy(experiment_config)

    for param, value in combination.items():
        setattr(exp_config.enhancement, param, value)

    return {
        "data_gt": data_gt,
        "data_mask": data_mask,

        "cached_eigenvalues": cached_eigenvalues,
        "cached_ratios": cached_ratios,
        "image_shape": image_shape,
        "xp": xp,
        "mask_arr": mask_arr,
        "gt_cache": gt_cache,

        "processing_config": exp_config.processing,
        "hessian_config": exp_config.hessian,
        "enhancement_config": exp_config.enhancement,
        "segmentation_config": exp_config.segmentation,
        "methods": exp_config.methods,

        "benchmark_config": benchmark_config,
    }

def _gs_eval_func(
    data_gt: ndarray,
    data_mask: ndarray,
    cached_eigenvalues,
    cached_ratios,
    image_shape,
    xp,
    mask_arr,
    gt_cache: dict,
    processing_config: ProcessingConfig,
    hessian_config: HessianConfig,
    enhancement_config: EnhancementConfig,
    segmentation_config: SegmentationConfig,
    methods: MethodsConfig,
    benchmark_config: Optional[BenchmarkConfig] = None,
) -> float:
    """Grid search evaluation function - metric MCC."""
    processing_config = ProcessingConfig(
        use_gpu=processing_config.use_gpu,
        normalize=processing_config.normalize,
        parallelize=False,
    )
    processor = Processor(processing_config)
    method = methods.enhancer
    apply_function = processor.enhancer.select_apply_function(method)

    _t0 = time.perf_counter()

    if method == "frangi":
        data_enhanced = apply_function(
            cached_eigenvalues,
            image_shape,
            xp,
            alpha=enhancement_config.alpha,
            beta=enhancement_config.beta,
            gamma=enhancement_config.gamma,
            cached_ratios=cached_ratios,
        )
    elif method == "jerman":
        data_enhanced = apply_function(
            cached_eigenvalues,
            image_shape,
            xp,
            tau=enhancement_config.tau,
            black_ridges=enhancement_config.black_ridges,
            mask=mask_arr,
        )
    elif method == "mfat":
        kw = {}
        for cfg_field, fn_arg in (
            ("mfat_tau", "tau"),
            ("mfat_tau2", "tau2"),
            ("mfat_step_size", "step_size"),
        ):
            value = getattr(enhancement_config, cfg_field)
            if value is not None:
                kw[fn_arg] = value
        kw["variant"] = enhancement_config.variant
        data_enhanced = apply_function(cached_eigenvalues, image_shape, xp, **kw)
    else:
        raise ValueError(f"Grid search rapide non supporté pour la méthode : {method}")

    _t1 = time.perf_counter()

    if processing_config.normalize:
        data_enhanced = normalize_data(data_enhanced)
    segmentation_function = processor.segmenter.select_segmentation_function(
        methods.segmenter
    )

    _t2 = time.perf_counter()

    segmentation_params = segmentation_config.to_dict()
    data_segmented, threshold = segmentation_function(
        data=data_enhanced,
        ground_truth=data_gt,
        mask=data_mask,
        gt_cache=gt_cache,
        **segmentation_params,
    )

    _t3 = time.perf_counter()
    score = mcc(data_segmented, data_gt, data_mask)
    _t4 = time.perf_counter()

    _module_logger.debug(
        "[GS TIMING] apply_function=%.4fs | normalize+dispatch=%.4fs | "
        "segmentation_function=%.4fs | mcc=%.4fs | total=%.4fs",
        _t1 - _t0, _t2 - _t1, _t3 - _t2, _t4 - _t3, _t4 - _t0,
    )

    return score

class BenchmarkHessian(BenchmarkBase):

    def __init__(
        self,
        save_mode: bool,
        plot_mode: bool,
        logger: Logger,
        loader: Loader,
        saver: Saver,
        params_grid: dict,
        input_dir: str,
    ):
        super().__init__(save_mode, plot_mode, logger, loader, saver)
        self.analytics = AnalyticsHessian()
        self.grid_searcher = GridSearcher(
            params_grid=params_grid,
            update_function=_gs_update_function,
            eval_function=_gs_eval_func,
            show_progress=False,
        )
        self.input_dir = input_dir
        self.detailed_results: dict[str, list[dict]] = {}
        self._all_data_stats: dict[str, dict] = {}
        self._patient_results: dict[str, dict] = {}
        self._segmented_per_patient: dict = {}
        self._enhanced_per_patient: dict = {}
        
    # Internal utilities
        
    def _update_config(
        self, config: ExperimentConfig, param: str, value: Any
    ) -> ExperimentConfig:
        for sub in (
            config.enhancement,
            config.hessian,
            config.segmentation,
            config.methods,
            config.processing,
        ):
            if hasattr(sub, param):
                setattr(sub, param, value)
                return config
        setattr(config.methods, param, value)
        return config

    def _save_or_show(
        self,
        fig: plt.Figure,
        name: str,
        output_dir: Optional[str] = None,
    ) -> None:
        """Save *fig* to *output_dir*, fall back to saver, or display it."""
        saved = False

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self.logger.info(f"[VIZ] Saved → {path}")
            saved = True

        if not saved and self.saver is not None:
            try:
                self.saver.save_plot(fig, name)
                saved = True
            except Exception as e:
                self.logger.warning(f"[VIZ] saver.save_plot failed for {name}: {e}")

        if not saved and self.plot_mode:
            plt.show()

    # Processing helpers

    def _compute_fp_tp_fn(
        self,
        data_segmented: ndarray,
        data_gt: ndarray,
        data_mask: Optional[ndarray],
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Return binary masks for FP, TP and FN (with optional liver mask)."""
        pred_bin = (data_segmented > 0.5).astype(np.uint8)
        gt_bin = (data_gt > 0.5).astype(np.uint8)

        if data_mask is not None:
            mask_bin = (data_mask > 0.5).astype(np.uint8)
            pred_bin = pred_bin * mask_bin
            gt_bin = gt_bin * mask_bin

        tp = (pred_bin & gt_bin ).astype(np.uint8)
        fp = (pred_bin & ~gt_bin ).astype(np.uint8)
        fn = (~pred_bin & gt_bin ).astype(np.uint8)
        return fp, tp, fn



    def _store_patient_continuity_data(
        self,
        patient_id: str,
        derivator_name: str,
        dice_val: float,
        cldice_score: float,
        connectivity_metrics: dict,
    ) -> None:
        """Stocke les métriques de continuité pour un patient/opérateur."""
        if patient_id not in self._patient_results:
            self._patient_results[patient_id] = {}
        if derivator_name not in self._patient_results[patient_id]:
            self._patient_results[patient_id][derivator_name] = {}
        self._patient_results[patient_id][derivator_name]["dice"] = dice_val
        self._patient_results[patient_id][derivator_name]["cldice"] = cldice_score
        self._patient_results[patient_id][derivator_name]["connectivity_metrics"] = connectivity_metrics

    def _fp_statistics(self, fp: ndarray, tp: ndarray, fn: ndarray) -> dict:
        """Compute comprehensive FP statistics."""
        labeled_fp, n_fp_comp = scipy_label(fp)
        fp_sizes = (
            np.bincount(labeled_fp.ravel())[1:] if n_fp_comp > 0 else np.array([])
        )

        # .sum() calculé une seule fois par tableau (au lieu de jusqu'à 5x chacun)
        tp_sum = int(tp.sum())
        fp_sum = int(fp.sum())
        fn_sum = int(fn.sum())

        total_pred = tp_sum + fp_sum
        total_gt = tp_sum + fn_sum
        precision = float(tp_sum / total_pred) if total_pred > 0 else 0.0
        recall = float(tp_sum / total_gt) if total_gt > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return {
            "total_fp_voxels": fp_sum,
            "n_fp_components": int(n_fp_comp),
            "fp_sizes_mean": float(fp_sizes.mean()) if len(fp_sizes) > 0 else 0.0,
            "fp_sizes_median": float(np.median(fp_sizes)) if len(fp_sizes) > 0 else 0.0,
            "fp_sizes_max": int(fp_sizes.max()) if len(fp_sizes) > 0 else 0,
            "fp_sizes_std": float(fp_sizes.std()) if len(fp_sizes) > 0 else 0.0,
            "fp_small_under10": int((fp_sizes < 10).sum()) if len(fp_sizes) > 0 else 0,
            "fp_medium_10_50": int(((fp_sizes >= 10) & (fp_sizes < 50)).sum()) if len(fp_sizes) > 0 else 0,
            "fp_large_over50": int((fp_sizes >= 50).sum()) if len(fp_sizes) > 0 else 0,
            "total_tp_voxels": tp_sum,
            "total_fn_voxels": fn_sum,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_sizes_histogram": fp_sizes.tolist(),
        }

    # Slice helpers

    def _get_center(self, data_raw: ndarray, data_mask: Optional[ndarray]) -> ndarray:
        if data_mask is not None and data_mask.any():
            return np.array(np.where(data_mask > 0)).mean(axis=1).astype(int)
        return np.array(data_raw.shape) // 2

    def _get_best_slices(
        self,
        data_gt: ndarray,
        data_mask: Optional[ndarray],
        center: ndarray,
        threshold=0.5,
    ) -> dict:
        thresh_to_use = threshold
        if data_mask is not None:
            gt_binary = data_gt > thresh_to_use
            mask_binary = data_mask > 0
            filtered = gt_binary & mask_binary
            data_for_sum = filtered.astype(np.float32)
        else:
            data_for_sum = (data_gt > thresh_to_use).astype(np.float32)
        has_vessels = np.any(data_for_sum > 0)
        return {
            "axial": np.argmax(data_for_sum.sum(axis=(0, 1))) if has_vessels else center[2],
            "coronal": np.argmax(data_for_sum.sum(axis=(0, 2))) if has_vessels else center[1],
            "sagittal": np.argmax(data_for_sum.sum(axis=(1, 2))) if has_vessels else center[0],
        }

    # Grid-search

    def _run_grid_search(
        self,
        data_raw: ndarray,
        data_gt: ndarray,
        data_mask: Optional[ndarray],
        experiment_config: ExperimentConfig,
        benchmark_config: Optional[BenchmarkConfig] = None,
    ) -> dict:
        center = _liver_center(data_raw, data_mask)
        data_raw_crop = _crop_around(data_raw, center)
        data_gt_crop = _crop_around(data_gt, center)
        data_mask_crop = (
            _crop_around(data_mask, center) if data_mask is not None else None
        )
        processing_config = ProcessingConfig(
            use_gpu=experiment_config.processing.use_gpu,
            normalize=experiment_config.processing.normalize,
            parallelize=False,
        )
        processor = Processor(processing_config)
        (
            hessian_function,
            enhancement_function,
            hessian_params,
            enhancement_params,
            scales,
            mask_for_enhancement,
        ) = processor.prepare_enhancement(
            data=data_raw_crop,
            hessian_config=experiment_config.hessian,
            enhancement_config=deepcopy(experiment_config.enhancement),
            methods=experiment_config.methods,
            mask_liver=data_mask_crop,
            ground_truth=data_gt_crop,
        )
        cached_eigenvalues, image_shape, xp, mask_arr = (
            processor.enhancer.precompute_eigenvalues(
                image=data_raw_crop,
                method=experiment_config.methods.enhancer,
                scales=scales,
                black_ridges=enhancement_params["black_ridges"],
                hessian_params=hessian_params,
                hessian_function=hessian_function,
                mask=mask_for_enhancement,
            )
        )
        cached_ratios = None
        if experiment_config.methods.enhancer == "frangi":
            cached_ratios = processor.enhancer.precompute_frangi_ratios(
                cached_eigenvalues,
                image_shape,
                xp,
            )
        gt_cache = processor.segmenter.precompute_gt_cache(
            data_gt_crop, data_mask_crop
        )
        best_params, _ = self.grid_searcher.fit(
            params={
                "data_gt": data_gt_crop,
                "data_mask": data_mask_crop,
                "cached_eigenvalues": cached_eigenvalues,
                "cached_ratios": cached_ratios,
                "image_shape": image_shape,
                "xp": xp,
                "mask_arr": mask_arr,
                "gt_cache": gt_cache,
                "experiment_config": deepcopy(experiment_config),
                "benchmark_config": benchmark_config,
            }
        )

        return best_params

    # Experiment

    def _run_experiment(
        self,
        data_raw: ndarray,
        data_gt: ndarray,
        data_mask: Optional[ndarray],
        experiment_config: ExperimentConfig,
        patient_id: str = "unknown_patient",
        benchmark_config: Optional[BenchmarkConfig] = None,
    ) -> Experiment:
        if not hasattr(self, '_grid_logged'):
            self.logger.info(
                f"[HESSIAN BENCHMARK] Filtre de rehaussement : "
                f"{experiment_config.methods.enhancer} | "
            )
            if benchmark_config and benchmark_config.params_grid:
                total=1
                for v in benchmark_config.params_grid.values():
                    total *= len(v)
                self.logger.info(f"Grid-searching {total} combinations for 'MCC'")
                self._grid_logged = True

        best_params = self._run_grid_search(data_raw, data_gt, data_mask, experiment_config, benchmark_config)
        for key, value in best_params.items():
            setattr(experiment_config.enhancement, key, value)
        processor = Processor(experiment_config.processing)
        data_enhanced, data_segmented, threshold, hessian_time = processor.process_data(
            data=data_raw,
            ground_truth=data_gt,
            hessian_config=experiment_config.hessian,
            enhancement_config=experiment_config.enhancement,
            segmentation_config=experiment_config.segmentation,
            methods=experiment_config.methods,
            mask_liver=data_mask,
            benchmark_config=benchmark_config,
        )
        if threshold is None:
            threshold = 0.5
            self.logger.warning(f"No threshold returned by segmentation, defaulting to {threshold}")

        experiment_config.segmentation.threshold = threshold
        mask_for_metrics = data_mask if data_mask is not None else None
        metrics_selection = (
            benchmark_config.optimization.to_detailed_metrics_selection()
            if benchmark_config and benchmark_config.optimization
            else "all"
        )
        all_metrics = detailed_metrics(
            data_segmented, data_gt, mask_for_metrics,
            threshold=threshold,
            skip_bifurcation=False,
            metrics=metrics_selection
        )
        cldice_score = all_metrics["cldice"]
        conn_metrics = {k: all_metrics[k] for k in [
            "n_components_pred",
            "n_components_gt",
            "excess_components",
            "missing_components",
            "components_ratio",
            "pred_small_components",
            "pred_medium_components",
            "pred_large_components",
            "gt_small_components",
            "gt_medium_components",
            "gt_large_components",
            "largest_component_pred",
            "largest_component_gt",
            "largest_component_overlap",
            "largest_gt_recall",
            "fragmentation_ratio",
            "skeleton_component_connectivity",
            "largest_ratio",
        ]}
        dice_val = all_metrics["dice"]


        hessian_time_seconds = hessian_time

        bdr_stats = {
            'bifurcation_detection_rate': all_metrics['bifurcation_detection_rate'],
            'bifurcation_precision': all_metrics['bifurcation_precision'],
            'n_bifurcations_gt': all_metrics['n_bifurcations_gt'],
            'n_bifurcations_pred': all_metrics['n_bifurcations_pred'],
            'n_bifurcations_detected': all_metrics['n_bifurcations_detected'],
            'bifurcation_tp': all_metrics['bifurcation_tp'],
            'bifurcation_fp': all_metrics['bifurcation_fp'],
            'bifurcation_fn': all_metrics['bifurcation_fn'],
        }
        derivator_name = experiment_config.methods.derivator
        if callable(derivator_name):
            derivator_name = derivator_name.__name__
        else:
            derivator_name = str(derivator_name)

        self._store_patient_continuity_data(
            patient_id=patient_id,
            derivator_name=derivator_name,
            dice_val=dice_val,
            cldice_score=cldice_score,
            connectivity_metrics=conn_metrics,
        )
        if patient_id not in self._segmented_per_patient:
            self._segmented_per_patient[patient_id] = {
                'operators': {},
                'data_raw': data_raw,
                'data_gt': data_gt,
                'data_mask': data_mask,
            }
        self._segmented_per_patient[patient_id]['operators'][derivator_name] = {
                'segmented': data_segmented,
                'threshold': threshold,
                'dice': dice_val,
                'cldice': cldice_score,
                'conn_metrics': conn_metrics,
                'hessian_time_seconds': hessian_time_seconds,
                'bdr_stats': bdr_stats,
                'mcc': all_metrics['mcc'],
                'precision': all_metrics['precision'],
                'recall': all_metrics['sensitivity'],
                'specificity': all_metrics['specificity'],
                'roc_auc': all_metrics['roc'],
                'pr_auc': all_metrics['pr'],
            }
        self.last_threshold = threshold
        data_stats = self.analyze_prediction_data(
            data_enhanced=data_enhanced,
            data_segmented=data_segmented,
            patient_id=patient_id,
            derivator_name=derivator_name,
            threshold=threshold,
        )
        if patient_id not in self._all_data_stats:
            self._all_data_stats[patient_id] = {}
        self._all_data_stats[patient_id][derivator_name] = data_stats

        self._log_and_store_confusion(
            data_segmented=data_segmented,
            data_gt=data_gt,
            data_mask=data_mask,
            experiment_config=experiment_config,
            patient_id=patient_id,
            cldice_score=cldice_score,
            conn_metrics=conn_metrics,
            all_metrics=all_metrics,
        )
        self.logger.info(
            f"Hessian time: {hessian_time_seconds}"
        )
        computed = set(all_metrics.get('_metrics_computed', ['bifurcation']))
        if 'bifurcation'in computed or 'all'in computed:
            self.logger.info(
                f"Bifurcation detection rate: {bdr_stats['bifurcation_detection_rate']:.4f} "
                f"(n_gt={bdr_stats['n_bifurcations_gt']}, precision={bdr_stats['bifurcation_precision']:.4f})"
            )

        experiment = Experiment(
            config=experiment_config,
            data_enhanced=data_enhanced,
            data_segmented=data_segmented,
            cldice_score=cldice_score,
            conn_metrics=conn_metrics,
            threshold=threshold,
            data_stats=data_stats,
        )
        experiment.hessian_time_seconds = hessian_time_seconds
        experiment.bdr_stats = bdr_stats
        experiment.all_metrics = all_metrics
        return experiment



    # Logging

    def _log_and_store_confusion(
            self,
            data_segmented: ndarray,
            data_gt: ndarray,
            data_mask: Optional[ndarray],
            experiment_config: ExperimentConfig,
            patient_id: str = "unknown_patient",
            cldice_score: Optional[float] = None,
            conn_metrics: Optional[dict] = None,
            all_metrics: Optional[dict] = None,
    ) -> None:
        derivator = experiment_config.methods.derivator
        if callable(derivator):
            derivator = getattr(derivator, "__name__", str(derivator))
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"DETAILED METRICS - {derivator.upper()} | patient: {patient_id}")
        self.logger.info(f"{'='*60}")

        if data_gt.dtype == bool:
            gt_binary = data_gt.astype(np.uint8)
        elif data_gt.dtype == np.uint8:
            if np.all((data_gt == 0) | (data_gt == 255)):
                gt_binary = (data_gt > 127).astype(np.uint8)
            else:
                gt_binary = (data_gt > 0).astype(np.uint8)
        else:
            if data_gt.max() <= 1.0:
                gt_binary = (data_gt > 0.5).astype(np.uint8)
            else:
                gt_binary = (data_gt > 127).astype(np.uint8)
        cm = print_confusion_matrix(
            y_pred=data_segmented,
            y_true=gt_binary,
            mask=data_mask,
            name=derivator,
            logger=self.logger,
        )
        threshold = experiment_config.segmentation.threshold
        self.logger.info(f"Optimal threshold: {threshold:.4f}")

        computed = set(all_metrics.get('_metrics_computed', ['all'])) if all_metrics else {'all'}
        if 'cldice'in computed or 'all'in computed:
            self.logger.info(f"CL-DICE: {cldice_score:.4f}")
        if 'components'in computed or 'all'in computed:
            self.logger.info("\n--- CONNECTED COMPONENTS (vascular continuity) ---")
            self.logger.info(f"GT components: {conn_metrics['n_components_gt']}")
            self.logger.info(f"Pred components: {conn_metrics['n_components_pred']}")
            self.logger.info(f"Ratio (pred/gt): {conn_metrics['components_ratio']:.2f}")
            self.logger.info(f"Fragmentation: +{conn_metrics['excess_components']} components")
            self.logger.info(f"Largest GT component: {conn_metrics['largest_component_gt']:,} voxels")
            self.logger.info(f"Largest pred component: {conn_metrics['largest_component_pred']:,} voxels")
            self.logger.info(f"Largest ratio: {conn_metrics['largest_ratio']:.2f}")
            self.logger.info(f"Largest GT recall: {conn_metrics.get('largest_gt_recall', conn_metrics.get('largest_component_recall', 0)):.4f}")
            self.logger.info(f"GT fragmentation: {conn_metrics.get('fragmentation_ratio', conn_metrics.get('gt_fragmentation', 0)):.4f}")
            self.logger.info(f"Skeleton connectivity: {conn_metrics['skeleton_component_connectivity']:.4f}")
            self.logger.info(
                f"GT distribution: small={conn_metrics['gt_small_components']}, "
                f"medium={conn_metrics['gt_medium_components']}, large={conn_metrics['gt_large_components']}"
            )
            self.logger.info(
                f"Pred distribution: small={conn_metrics['pred_small_components']}, "
                f"medium={conn_metrics['pred_medium_components']}, large={conn_metrics['pred_large_components']}"
            )

        if patient_id not in self.detailed_results:
            self.detailed_results[patient_id] = []
        self.detailed_results[patient_id].append({
            "derivator": derivator,
            "confusion_matrix": cm,
            "cldice_score": cldice_score,
            "conn_metrics": conn_metrics,
            "all_metrics": all_metrics,
        })

    # Scale optimisation

    def optimize_scales(
        self,
        raw_files: list,
        gt_files: list,
        mask_files: list,
        experiment_config: ExperimentConfig,
        sigma_mins: list,
        sigma_maxs: list,
        n_scales: int = 4,
    ) -> list:
        combinations = [
            (smin, smax) for smin in sigma_mins for smax in sigma_maxs if smax > smin
        ]
        self.logger.info(f"[SCALE OPT] {len(combinations)} combinations to test.")

        # --- Load all patients ---
        patients_data = []
        for raw_file, gt_file, mask_file in zip(raw_files, gt_files, mask_files):
            exp_config = deepcopy(experiment_config)
            exp_config.loading.raw_file = raw_file
            exp_config.loading.gt_file = gt_file
            exp_config.loading.mask_file = mask_file
            data_raw, data_gt, data_mask = self.load_data(exp_config.loading)
            full_path = raw_file if os.path.isabs(raw_file) or os.path.exists(raw_file) else os.path.join("data", self.input_dir, raw_file)
            try:
                spacing = nib.load(full_path).header.get_zooms()
                voxel_size = float(np.min(spacing))
                self.logger.info(f"[SCALE OPT] spacing lu: {spacing}, voxel_size={voxel_size:.6f} mm")
            except Exception as e:
                self.logger.warning(f"[SCALE OPT] Impossible de lire le spacing de {full_path} ({e})")
                voxel_size = 1.0

            center = _liver_center(data_raw, data_mask)
            patients_data.append({
                "data_raw": _crop_around(data_raw, center),
                "data_gt": _crop_around(data_gt, center),
                "data_mask": _crop_around(data_mask, center) if data_mask is not None else None,
                "voxel_size": voxel_size,
                "exp_config": deepcopy(experiment_config),
            })
            patients_data[-1]["exp_config"].loading.raw_file = raw_file
            patients_data[-1]["exp_config"].loading.gt_file = gt_file
            patients_data[-1]["exp_config"].loading.mask_file = mask_file

        # --- Search best (smin, smax) ---
        best_scales = None
        best_mean_mcc = -1.0

        for smin, smax in combinations:
            scales_mm = list(np.linspace(smin, smax, n_scales).round(5))
            mccs = []

            for patient in patients_data:
                exp_config = deepcopy(patient["exp_config"])
                exp_config.methods.derivator = "default"
                exp_config.enhancement.scales = [
                    round(s / patient["voxel_size"], 2) for s in scales_mm
                ]

                processor = Processor(exp_config.processing)
                _, data_segmented, _, _ = processor.process_data(
                    data=patient["data_raw"],
                    ground_truth=patient["data_gt"],
                    hessian_config=exp_config.hessian,
                    enhancement_config=exp_config.enhancement,
                    segmentation_config=exp_config.segmentation,
                    methods=exp_config.methods,
                    mask_liver=patient["data_mask"],
                )
                mccs.append(mcc(data_segmented, patient["data_gt"], patient["data_mask"]))

            mean_mcc = float(np.mean(mccs))
            self.logger.info(f"[SCALE OPT] scales_mm={scales_mm} → mean MCC={mean_mcc:.3f}")
            EPSILON = 0.002 # Seuil de 0.2% d'amélioration
            if mean_mcc > best_mean_mcc+EPSILON:
                best_mean_mcc = mean_mcc
                best_scales = scales_mm
            elif abs(mean_mcc - best_mean_mcc) <= EPSILON:
                # Égalité : choisir la plage la plus large (σ_max - σ_min le plus grand)
                current_span = smax - smin
                best_span = best_scales[-1] - best_scales[0] if best_scales is not None else 0
                if current_span > best_span:
                    best_scales = scales_mm
                    best_mean_mcc = mean_mcc

        self.logger.info(f"[SCALE OPT] Best scales_mm: {best_scales} (MCC={best_mean_mcc:.3f})")

        if best_scales is not None and self.saver is not None:
            scales_file = Path(self.saver.output_dir) / "best_scales.json"
            scales_file.parent.mkdir(parents=True, exist_ok=True)
            with open(scales_file, "w") as f:
                json.dump(
                    {
                        "scales_mm": best_scales,
                        "mean_mcc": best_mean_mcc,
                        "timestamp": str(datetime.now()),
                    },
                    f,
                    indent=4,
                )

        return best_scales if best_scales is not None else []

    # False-positive analysis
    def _save_fp_stats_incremental(self, patient_id: str, operator: str, stats: dict) -> None:
        if self.saver is None:
            return
        fp_stats_file = Path(self.saver.output_dir) / "fp_stats.json"
        fp_stats_file.parent.mkdir(parents=True, exist_ok=True)
        if fp_stats_file.exists():
            try:
                with open(fp_stats_file, "r") as f:
                    fp_stats = json.load(f)
            except json.JSONDecodeError:
                fp_stats = {}
        key=f"{patient_id}_{operator}"
        fp_stats[key] = stats
        with open(fp_stats_file, "w") as f:
            json.dump(fp_stats, f, indent=4)

    
    # Bifurcation analysis

    def analyze_bifurcations(
        self,
        experiment_config: ExperimentConfig,
        benchmark_config: BenchmarkConfig,
        patient_id: str,
        tolerance_radius: int = 2,
        roi_half_size: int = 4,
    ) -> dict[str, dict]:

        results: dict[str, dict] = {}

        derivator_values = benchmark_config.params.get("derivator")
        if derivator_values is None:
            self.logger.warning(
                "[BIFURCATION] benchmark_config.params ne contient pas 'derivator'- "
                "analyse ignorée."
            )
            return results

        data_raw, data_gt, data_mask = self.load_data(experiment_config.loading)

        for value in derivator_values:
            exp_config = self._update_config(deepcopy(experiment_config), "derivator", value)

            processor = Processor(exp_config.processing)
            data_enhanced, data_segmented, threshold, hessian_time = processor.process_data(
                data=data_raw,
                ground_truth=data_gt,
                hessian_config=exp_config.hessian,
                enhancement_config=exp_config.enhancement,
                segmentation_config=exp_config.segmentation,
                methods=exp_config.methods,
                mask_liver=data_mask,
            )
            if threshold is None:
                threshold = 0.5
            pred_bin = data_segmented > threshold
            op_name = str(value) if not callable(value) else getattr(value, "__name__", str(value))

            
            gt_bin = data_gt > 0.5

            bdr_stats = bifurcation_detection_rate(
                pred_bin, gt_bin, mask=data_mask, tolerance_radius=tolerance_radius
            )

            # --- Statistiques locales autour de chaque bifurcation GT ---
            gt_for_bif = gt_bin
            if data_mask is not None:
                gt_for_bif = gt_bin & (data_mask > 0)
            from core.experiments.metrics import _skeletonize_3d
            sk_gt = _skeletonize_3d(gt_for_bif)
            bifurcations_gt = detect_bifurcations(sk_gt)
            bif_coords = np.argwhere(bifurcations_gt)

            local_vesselness_vals = []
            local_dice_vals = []
            center_dip_ratios = []

            h = roi_half_size
            shape = data_enhanced.shape
            for (cz, cy, cx) in bif_coords:
                z0, z1 = max(0, cz - h), min(shape[0], cz + h + 1)
                y0, y1 = max(0, cy - h), min(shape[1], cy + h + 1)
                x0, x1 = max(0, cx - h), min(shape[2], cx + h + 1)

                roi_enhanced = data_enhanced[z0:z1, y0:y1, x0:x1]
                roi_pred = pred_bin[z0:z1, y0:y1, x0:x1]
                roi_gt = gt_bin[z0:z1, y0:y1, x0:x1]

                roi_mean = float(roi_enhanced.mean())
                local_vesselness_vals.append(roi_mean)

                inter = np.logical_and(roi_pred, roi_gt).sum()
                denom = roi_pred.sum() + roi_gt.sum()
                local_dice_vals.append(float(2 * inter / denom) if denom > 0 else 0.0)

                center_value = float(data_enhanced[cz, cy, cx])
                if roi_mean > 1e-12:
                    center_dip_ratios.append(center_value / roi_mean)

            results[op_name] = {
                "bdr_stats": bdr_stats,
                "n_bifurcations_analyzed": len(bif_coords),
                "local_vesselness_mean": float(np.mean(local_vesselness_vals)) if local_vesselness_vals else 0.0,
                "local_dice_mean": float(np.mean(local_dice_vals)) if local_dice_vals else 0.0,
                "center_dip_ratio_mean": float(np.mean(center_dip_ratios)) if center_dip_ratios else 1.0,
                "hessian_time_seconds": getattr(processor.derivator, "last_hessian_time", None),
            }

            self.logger.info(
                f"[BIFURCATION] {patient_id} | {op_name} → "
                f"BDR={bdr_stats['bifurcation_detection_rate']:.3f} "
                f"(n_gt={bdr_stats['n_bifurcations_gt']}) | "
                f"local_dice={results[op_name]['local_dice_mean']:.3f} | "
                f"center_dip={results[op_name]['center_dip_ratio_mean']:.3f} | "
                f"hessian_time={results[op_name]['hessian_time_seconds']}"
            )

        if self.saver is not None and results:
            out_path = Path(self.saver.output_dir) / "bifurcation_analysis.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if out_path.is_file():
                with open(out_path, "r") as f:
                    existing = json.load(f)
            existing[patient_id] = _to_json_serializable(results)
            with open(out_path, "w") as f:
                json.dump(existing, f, indent=4)

        return results

    

    # Data analysis

    def analyze_prediction_data(
        self,
        data_enhanced: ndarray,
        data_segmented: ndarray,
        patient_id: str,
        derivator_name: str,
        threshold,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Analyse enhanced and segmented data at compute time.
        Returns statistics and can generate visualisations.
        """

        enhanced = data_enhanced.ravel()
        segmented = data_segmented.ravel()
        unique_enhanced = np.unique(enhanced)
        unique_seg = np.unique(segmented)

        stats_enhanced = {
            "shape": str(data_enhanced.shape),
            "dtype": str(data_enhanced.dtype),
            "min": float(np.min(enhanced)),
            "max": float(np.max(enhanced)),
            "mean": float(np.mean(enhanced)),
            "std": float(np.std(enhanced)) if not np.isinf(np.std(enhanced)) else 0,
            "n_unique": len(unique_enhanced),
            "is_binary": False,
            "values_0_1": float(np.sum((enhanced >= 0) & (enhanced <= 1)) / len(enhanced) * 100),
            "has_inf": bool(np.isinf(enhanced).any()),
            "has_nan": bool(np.isnan(enhanced).any()),
        }

        stats_segmented = {
            "shape": str(data_segmented.shape),
            "dtype": str(data_segmented.dtype),
            "min": float(np.min(segmented)),
            "max": float(np.max(segmented)),
            "mean": float(np.mean(segmented)),
            "std": float(np.std(segmented)) if not np.isinf(np.std(segmented)) else 0,
            "n_unique": len(unique_seg),
            "is_binary": bool(
                np.array_equal(unique_seg, [0, 1]) or np.array_equal(unique_seg, [0.0, 1.0])
            ),
            "binary_ratio": float(
                np.sum((segmented == 0) | (segmented == 1)) / len(segmented) * 100
            ),
            "n_intermediate": int(np.sum((unique_seg > 0) & (unique_seg < 1))),
        }

        segmented_above_mask = data_segmented > 0
        segmented_voxels = int(segmented_above_mask.sum())
        segmented_ratio = float(segmented_above_mask.mean())


        above_mask = data_enhanced > threshold
        voxels_above = int(above_mask.sum())
        stats_comparison = {
            "threshold_used": float(threshold),
            "voxels_above_threshold": voxels_above,
            "voxels_below_threshold": int(np.sum(data_enhanced <= threshold)),
            "ratio_above": voxels_above / above_mask.size,
            "segmented_voxels": segmented_voxels,
            "segmented_ratio": segmented_ratio,
        }

        return {
            "enhanced": stats_enhanced,
            "segmented": stats_segmented,
            "comparison": stats_comparison,
        }

    # Figures main entry point

    def generate_data_analysis_figures(self, output_dir: Optional[str] = None) -> None:
        """Generate all data-analysis figures."""
        if self._all_data_stats:
            self.plot_patient_data_analysis(self._all_data_stats, output_dir)
            self.plot_global_data_analysis(self._all_data_stats, output_dir)
            self.logger.info("[DATA ANALYSIS] Figures generated.")
        else:
            self.logger.warning("[DATA ANALYSIS] No data stats available.")

    def _create_figures(self, benchmark_data: BenchmarkData) -> list[FigureData]:
        data_raw = benchmark_data.data_raw
        data_gt = benchmark_data.data_gt
        experiments = [
            experiment
            for values in benchmark_data.results.values()
            for experiment in values.values()
            if experiment is not None
        ]

        figures: list[FigureData] = [
            self.analytics.get_histograms(experiments=experiments, data_raw=data_raw, data_gt=data_gt),
            self.analytics.get_configs(experiments=experiments),
            self.analytics.get_metrics(experiments=experiments),
            self.analytics.get_curves(experiments=experiments, ground_truth=data_gt),
            *self.analytics.get_views(experiments=experiments, data_gt=data_gt, data_raw=data_raw),
        ]

        if self.detailed_results:
            figures.append(self._create_confusion_summary_figure())

        return figures

    def _create_confusion_summary_figure(self) -> FigureData:
        """Summary figure: TP / FP / FN / TN aggregated across all operators."""
        aggregated: dict[str, dict] = {}
        for patient_id, records in self.detailed_results.items():
            for r in records:
                d = r["derivator"]
                if d not in aggregated:
                    aggregated[d] = {"tp": 0, "fp": 0, "fn": 0, "gt_vessels": 0, "pred_vessels": 0}
                cm = r["confusion_matrix"]
                for key in aggregated[d]:
                    aggregated[d][key] += cm.get(key, 0)

        derivators = list(aggregated.keys())
        tp = np.array([aggregated[d]["tp"] for d in derivators])
        fp = np.array([aggregated[d]["fp"] for d in derivators])
        fn = np.array([aggregated[d]["fn"] for d in derivators])
        gt_vessels = np.array([aggregated[d]["gt_vessels"] for d in derivators])
        pred_vessels = np.array([aggregated[d]["pred_vessels"] for d in derivators])

        fig, (ax_cm, ax_vol) = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(derivators))
        width = 0.6

        ax_cm.bar(x, tp, width, label="TP (true positives)", color="green", alpha=0.7)
        ax_cm.bar(x, fp, width, bottom=tp, label="FP (false positives)", color="red", alpha=0.7)
        ax_cm.bar(x, fn, width, bottom=tp + fp, label="FN (false negatives)", color="orange", alpha=0.7)
        ax_cm.set_xlabel("Operators")
        ax_cm.set_ylabel("Number of voxels")
        ax_cm.set_title("Confusion matrix per operator\n(voxels in ROI)")
        ax_cm.set_xticks(x)
        ax_cm.set_xticklabels(derivators, rotation=45, ha="right")
        ax_cm.legend()
        ax_cm.grid(alpha=0.3, axis="y")

        ax_vol.bar(x - width / 4, gt_vessels, width / 2, label="GT vessels", color="blue", alpha=0.7)
        ax_vol.bar(x + width / 4, pred_vessels, width / 2, label="Pred vessels", color="cyan", alpha=0.7)
        ax_vol.set_xlabel("Operators")
        ax_vol.set_ylabel("Number of voxels")
        ax_vol.set_title("Vascular volume: GT vs Predictions")
        ax_vol.set_xticks(x)
        ax_vol.set_xticklabels(derivators, rotation=45, ha="right")
        ax_vol.legend()
        ax_vol.grid(alpha=0.3, axis="y")

        if self.saver is not None:
            out = Path(self.saver.output_dir) / "confusion_matrices.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            existing = {}
            if out.is_file():
                with open(out, "r") as f:
                    existing = json.load(f)
            existing.update(_to_json_serializable(self.detailed_results))
            with open(out, "w") as f:
                json.dump(existing, f, indent=4)

        fig.tight_layout()
        return FigureData(figure=fig, name="confusion_matrix_summary", mode="plot")

    # Data-analysis visualisations

    def plot_patient_data_analysis(
        self,
        all_patient_stats: dict,
        output_dir: Optional[str] = None,
    ) -> None:
        """Generate one figure per patient with data-analysis subplots."""
        operators = list(list(all_patient_stats.values())[0].keys()) if all_patient_stats else []
        n_ops = len(operators)

        for patient_id, ops_stats in all_patient_stats.items():
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"Data Analysis - Patient {patient_id}", fontsize=16, fontweight="bold")

            # 1. Enhanced mean per operator
            ax1 = axes[0, 0]
            ax1.bar(operators, [ops_stats[op]["enhanced"]["mean"] for op in operators],
                    color="steelblue", alpha=0.7)
            ax1.set_ylabel("Mean value")
            ax1.set_title("Enhanced data - Mean per operator")
            ax1.tick_params(axis="x", rotation=45, labelsize=8)
            ax1.grid(axis="y", alpha=0.3)

            # 2. Enhanced std per operator
            ax2 = axes[0, 1]
            stds = [ops_stats[op]["enhanced"]["std"] for op in operators]
            colors = ["orange"if s > 0.5 else "green"for s in stds]
            ax2.bar(operators, stds, color=colors, alpha=0.7)
            ax2.set_ylabel("Standard deviation")
            ax2.set_title("Enhanced data - Std per operator")
            ax2.tick_params(axis="x", rotation=45, labelsize=8)
            ax2.axhline(y=0.3, color="red", linestyle="--", label="High variability threshold")
            ax2.legend(fontsize=8)
            ax2.grid(axis="y", alpha=0.3)

            # 3. Ratio of values in [0, 1]
            ax3 = axes[0, 2]
            ratio_0_1 = [ops_stats[op]["enhanced"]["values_0_1"] for op in operators]
            colors_r = ["green"if r > 90 else "orange"if r > 70 else "red"for r in ratio_0_1]
            ax3.bar(operators, ratio_0_1, color=colors_r, alpha=0.7)
            ax3.set_ylabel("Percentage (%)")
            ax3.set_title("Enhanced - Values in [0,1]")
            ax3.tick_params(axis="x", rotation=45, labelsize=8)
            ax3.set_ylim(0, 100)
            ax3.axhline(y=90, color="green", linestyle="--", alpha=0.5)
            ax3.axhline(y=70, color="orange", linestyle="--", alpha=0.5)
            ax3.grid(axis="y", alpha=0.3)

            # 4. Binary status
            ax4 = axes[1, 0]
            is_binary = [ops_stats[op]["segmented"]["is_binary"] for op in operators]
            binary_ratio = [ops_stats[op]["segmented"]["binary_ratio"] for op in operators]
            colors_b = ["green"if b else "red"for b in is_binary]
            ax4.bar(operators, [100 if b else 0 for b in is_binary], color=colors_b, alpha=0.7)
            ax4.set_ylabel("Binary? (%)")
            ax4.set_title("Segmentation - Binary status")
            ax4.tick_params(axis="x", rotation=45, labelsize=8)
            ax4.set_ylim(0, 100)
            ax4.grid(axis="y", alpha=0.3)
            for i, ratio in enumerate(binary_ratio):
                ax4.text(i, 50, f"{ratio:.1f}%", ha="center", va="center", fontsize=8)

            # 5. Threshold vs ratio above threshold
            ax5 = axes[1, 1]
            xpos = np.arange(n_ops)
            w = 0.35
            threshold_numeric = []
            for op in operators:
                thresh = ops_stats[op]["comparison"]["threshold_used"]
                threshold_numeric.append(thresh)
            ratio_above = [ops_stats[op]["comparison"]["ratio_above"] for op in operators]
            ax5.bar(xpos - w/2, threshold_numeric, w, label="Threshold", color="blue", alpha=0.7)
            ax5.bar(xpos + w/2, ratio_above, w, label="Ratio > threshold", color="orange", alpha=0.7)
            

            # 6. Summary table
            ax6 = axes[1, 2]
            ax6.axis("off")
            table_data = [["Operator", "Enhanced\nMean", "Enhanced\nStd", "Binary?", "Threshold"]]
            for op in operators:
                thresh_value = ops_stats[op]['comparison']['threshold_used']
                thresh_display = f"{thresh_value:.3f}"
                table_data.append([
                    op[:12],
                    f"{ops_stats[op]['enhanced']['mean']:.3f}",
                    f"{ops_stats[op]['enhanced']['std']:.3f}",
                    ""if ops_stats[op]["segmented"]["is_binary"] else "",
                    thresh_display,
                ])
            table = ax6.table(
                cellText=table_data, loc="center", cellLoc="center",
                colWidths=[0.2, 0.15, 0.15, 0.1, 0.15],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            for i, op in enumerate(operators, 1):
                row_color = (
                    "#ffcccc"if not ops_stats[op]["segmented"]["is_binary"]
                    else "#ffffcc"if ops_stats[op]["enhanced"]["std"] > 0.5
                    else "white"
                )
                for j in range(5):
                    table[(i, j)].set_facecolor(row_color)
            ax6.set_title("Summary Table", fontsize=10, fontweight="bold", pad=20)

            plt.tight_layout()
            self._save_or_show(fig, f"data_analysis_patient_{patient_id}", output_dir)
            plt.close(fig)

    def plot_global_data_analysis(
        self,
        all_patient_stats: dict,
        output_dir: Optional[str] = None,
    ) -> None:
        """Generate global figures: mean per operator across all patients."""
        if not all_patient_stats:
            return

        operators = list(list(all_patient_stats.values())[0].keys())

        aggr = {op: {
            "enhanced_mean": [],
            "enhanced_std": [],
            "enhanced_range": [],
            "threshold_mean": [],
            "ratio_above_mean": [],
            "binary_count": 0,
            "total_count": 0,
        } for op in operators}

        for patient_stats in all_patient_stats.values():
            for op in operators:
                if op not in patient_stats:
                    continue
                s = patient_stats[op]
                thresh_numeric = float(s["comparison"]["threshold_used"])
                aggr[op]["enhanced_mean"].append(s["enhanced"]["mean"])
                aggr[op]["enhanced_std"].append(s["enhanced"]["std"])
                aggr[op]["enhanced_range"].append(s["enhanced"]["max"] - s["enhanced"]["min"])
                aggr[op]["threshold_mean"].append(thresh_numeric)
                aggr[op]["ratio_above_mean"].append(s["comparison"]["ratio_above"])
                aggr[op]["total_count"] += 1
                if s["segmented"]["is_binary"]:
                    aggr[op]["binary_count"] += 1

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("GLOBAL DATA ANALYSIS - All patients", fontsize=16, fontweight="bold")
        x = np.arange(len(operators))

        # 1. Global enhanced mean ± std
        ax1 = axes[0, 0]
        means = [np.mean(aggr[op]["enhanced_mean"]) for op in operators]
        stds = [np.std(aggr[op]["enhanced_mean"]) for op in operators]
        ax1.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.7, edgecolor="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels(operators, rotation=45, ha="right", fontsize=9)
        ax1.set_ylabel("Mean value")
        ax1.set_title("Enhanced data - Global mean ± std")
        ax1.grid(axis="y", alpha=0.3)

        # 2. Mean variability (std of enhanced)
        ax2 = axes[0, 1]
        std_means = [np.mean(aggr[op]["enhanced_std"]) for op in operators]
        colors2 = ["green"if s < 0.3 else "orange"if s < 0.5 else "red"for s in std_means]
        ax2.bar(x, std_means, color=colors2, alpha=0.7, edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels(operators, rotation=45, ha="right", fontsize=9)
        ax2.set_ylabel("Mean standard deviation")
        ax2.set_title("Enhanced data - Mean variability")
        ax2.axhline(y=0.3, color="green", linestyle="--", alpha=0.5, label="Low")
        ax2.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Medium")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        # 3. Binary segmentation rate
        ax3 = axes[0, 2]
        binary_rates = [
            aggr[op]["binary_count"] / aggr[op]["total_count"] * 100 for op in operators
        ]
        colors3 = ["green"if r == 100 else "orange"if r > 80 else "red"for r in binary_rates]
        ax3.bar(x, binary_rates, color=colors3, alpha=0.7, edgecolor="black")
        ax3.set_xticks(x)
        ax3.set_xticklabels(operators, rotation=45, ha="right", fontsize=9)
        ax3.set_ylabel("Binary segmentation rate (%)")
        ax3.set_title("Segmentation - Binary status")
        ax3.set_ylim(0, 105)
        ax3.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="100% binary")
        ax3.axhline(y=80, color="orange", linestyle="--", alpha=0.5, label="80%")
        ax3.legend(fontsize=8)
        ax3.grid(axis="y", alpha=0.3)
        for i, rate in enumerate(binary_rates):
            ax3.text(i, rate + 2, f"{rate:.0f}%", ha="center", fontsize=8)

        # 4. Mean optimal thresholds
        ax4 = axes[1, 0]
        thresholds_mean = [np.mean(aggr[op]["threshold_mean"]) for op in operators]
        thresholds_std = [np.std(aggr[op]["threshold_mean"]) for op in operators]
        ax4.bar(x, thresholds_mean, yerr=thresholds_std, capsize=5,
                color="purple", alpha=0.7, edgecolor="black")
        ax4.set_xticks(x)
        ax4.set_xticklabels(operators, rotation=45, ha="right", fontsize=9)
        ax4.set_ylabel("Threshold value")
        ax4.set_title("Optimal thresholds - Mean ± std")
        ax4.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Default 0.5")
        ax4.legend(fontsize=8)
        ax4.grid(axis="y", alpha=0.3)

        # 5. Voxels above threshold
        ax5 = axes[1, 1]
        ratio_means = [np.mean(aggr[op]["ratio_above_mean"]) for op in operators]
        ratio_stds = [np.std(aggr[op]["ratio_above_mean"]) for op in operators]
        ax5.bar(x, ratio_means, yerr=ratio_stds, capsize=5,
                color="teal", alpha=0.7, edgecolor="black")
        ax5.set_xticks(x)
        ax5.set_xticklabels(operators, rotation=45, ha="right", fontsize=9)
        ax5.set_ylabel("Ratio")
        ax5.set_title("Voxels above threshold - Mean ± std")
        ax5.set_ylim(0, 1)
        ax5.grid(axis="y", alpha=0.3)

        # 6. Correlation heatmap
        ax6 = axes[1, 2]
        metric_keys = ["enhanced_mean", "enhanced_std", "binary_count", "threshold_mean", "ratio_above_mean"]
        metric_names = ["Mean", "Std", "BinRate", "Threshold", "RatioAbove"]
        n_m = len(metric_keys)
        corr = np.eye(n_m)

        def _op_vals(key):
            out = []
            for op in operators:
                if key == "binary_count":
                    out.append(aggr[op]["binary_count"] / aggr[op]["total_count"])
                else:
                    out.append(np.mean(aggr[op][key]))
            return out

        # _op_vals(key) était recalculé à chaque paire (i, j) alors que sa valeur
        # ne dépend que de i (ou j) : on la calcule une seule fois par clé.
        vals_by_key = [_op_vals(key) for key in metric_keys]

        for i in range(n_m):
            for j in range(i + 1, n_m):
                c = _safe_corrcoef(vals_by_key[i], vals_by_key[j])
                corr[i, j] = corr[j, i] = c

        im = ax6.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax6.set_xticks(range(n_m))
        ax6.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=8)
        ax6.set_yticks(range(n_m))
        ax6.set_yticklabels(metric_names, fontsize=8)
        ax6.set_title("Correlation between metrics", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax6, label="Correlation", shrink=0.8)
        for i in range(n_m):
            for j in range(n_m):
                color = "white"if abs(corr[i, j]) > 0.6 else "black"
                ax6.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color=color)

        plt.tight_layout()
        self._save_or_show(fig, "global_data_analysis", output_dir)
        plt.close(fig)

    # FP summary report (multi-patient × multi-operator)

    def create_fp_summary_report(
        self,
        all_stats: list[dict],
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Generate summary figures comparing FP across patients and operators.
        """
        import pandas as pd

        df = pd.DataFrame(all_stats)
        if df.empty:
            self.logger.warning("[FP REPORT] No statistics to display.")
            return

        patients = df["patient_id"].unique()
        operators = df["operator"].unique()
        n_ops = len(operators)
        n_pats = len(patients)
        colors = plt.cm.tab10(np.linspace(0, 1, n_ops))

        # Figure 1 - grouped bar: FP voxels per patient × operator
        fig1, ax = plt.subplots(figsize=(max(10, n_pats * 1.2), 6))
        x = np.arange(n_pats)
        width = 0.8 / n_ops
        for i, op in enumerate(operators):
            sub = df[df["operator"] == op]
            vals = [
                float(sub[sub["patient_id"] == p]["total_fp_voxels"].values[0])
                if p in sub["patient_id"].values else 0.0
                for p in patients
            ]
            ax.bar(x + i * width, vals, width, label=op, color=colors[i], alpha=0.82)
        ax.set_xticks(x + width * (n_ops - 1) / 2)
        ax.set_xticklabels(patients, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("FP Voxels")
        ax.set_title("Total FP voxels per patient and operator")
        ax.legend(title="Operator", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig1, "fp_report_fp_per_patient_operator", output_dir)
        plt.close(fig1)

        # Figure 2 - mean ± std precision / recall / F1 per operator
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
        for ax2, (metric, label) in zip(
            axes2, [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
        ):
            means = [df[df["operator"] == op][metric].mean() for op in operators]
            stds = [df[df["operator"] == op][metric].std() for op in operators]
            ax2.bar(operators, means, yerr=stds, capsize=5, color=colors[:n_ops], alpha=0.8)
            ax2.set_title(f"Mean {label} per operator")
            ax2.set_xlabel("Operator")
            ax2.set_ylabel(label)
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, axis="y", alpha=0.3)
        plt.suptitle("Mean FP metrics per operator (all patients)", fontsize=11)
        plt.tight_layout()
        self._save_or_show(fig2, "fp_report_metrics_by_operator", output_dir)
        plt.close(fig2)

        # Figure 3 - stacked FP size distribution per operator
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        small_means = [df[df["operator"] == op]["fp_small_under10"].mean() for op in operators]
        medium_means = [df[df["operator"] == op]["fp_medium_10_50"].mean() for op in operators]
        large_means = [df[df["operator"] == op]["fp_large_over50"].mean() for op in operators]
        x3 = np.arange(n_ops)
        ax3.bar(x3, small_means, 0.6, label="FP < 10 vox (noise)", color="#e74c3c", alpha=0.85)
        ax3.bar(x3, medium_means, 0.6, bottom=small_means,
                label="FP 10–50 vox (medium)", color="#e67e22", alpha=0.85)
        ax3.bar(x3, large_means, 0.6,
                bottom=np.array(small_means) + np.array(medium_means),
                label="FP > 50 vox (parasitic structures)", color="#c0392b", alpha=0.85)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(operators, rotation=45, ha="right")
        ax3.set_ylabel("# FP components (mean)")
        ax3.set_title("FP component size distribution per operator")
        ax3.legend(fontsize=9)
        ax3.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig3, "fp_report_fp_size_distribution", output_dir)
        plt.close(fig3)

        if self.saver is not None:
            csv_path = Path(self.saver.output_dir) / "fp_full_stats.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"[FP REPORT] CSV saved → {csv_path}")

        self.logger.info("[FP REPORT] Summary report generated.")

    # JSON persistence

    def _save_metrics_json(self, all_patients_results: dict, output_dir=None):
        output_path = Path(output_dir) if output_dir else Path(self.saver.output_dir) if self.saver else Path(".")
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "all_metrics.json", "w") as f:
            json.dump(_to_json_serializable(all_patients_results), f, indent=4)
            
        self.logger.info(f"[JSON] Saved → {output_path / 'all_metrics.json'}")