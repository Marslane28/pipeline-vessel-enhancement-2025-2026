import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from core.io.loader import Loader
from logging import Logger
from core.io.logger import setup_logger, close_logger
from core.io.saver import Saver
from core.processing.processor import Processor
from core.config.figure import FigureData
from core.config.experiment import ExperimentConfig, Experiment, LoadingConfig
from core.config.benchmark import BenchmarkConfig, BenchmarkResults, BenchmarkData
from core.config.metrics import Metrics
from core.experiments.analytics.base import AnalyticsBase
from core.experiments.metrics import detailed_metrics
from core.utils.decorator import log_section


class BenchmarkBase(ABC):

    def __init__(self,
            save_mode: bool,
            plot_mode: bool,
            logger: Logger,
            loader: Loader,
            saver: Saver,
            analytics: Optional[AnalyticsBase] = None,
        ):
        self.save_mode = save_mode
        self.plot_mode = plot_mode
        self.loader = loader
        self.logger = logger
        self.saver = saver if save_mode else None
        self.analytics = analytics

    @abstractmethod
    def _update_config(self, config: ExperimentConfig, param: str, value: Any) -> ExperimentConfig:
        """Met à jour dans la config de l'expérience le paramètre étudié dans le benchmark"""
        pass

    @abstractmethod
    def _run_experiment(self, data_raw: ndarray,
            data_gt: ndarray,
            data_mask: ndarray,
            experiment_config: ExperimentConfig,
            patient_id: str = "unknown_patient",
            benchmark_config: Optional['BenchmarkConfig'] = None,
        ) -> Experiment:
        
        pass

    @abstractmethod
    def _create_figures(self, benchmark_data: BenchmarkData) -> list[FigureData]:
        """Construit les figures utiles pour l'analyse du benchmark."""
        pass

    def load_data(self, loading_config: LoadingConfig):
        data_raw = self.loader.load_data(
            filename=loading_config.raw_file,
            normalize=loading_config.normalize,
            crop=loading_config.crop,
            target_shape=loading_config.target_shape,
        )
        data_gt = self.loader.load_data(
            filename=loading_config.gt_file,
            normalize=loading_config.normalize,
            crop=loading_config.crop,
            target_shape=loading_config.target_shape,
        )
        data_mask = None
        if loading_config.mask_file is not None:
            data_mask = self.loader.load_data(
                filename=loading_config.mask_file,
                normalize=False,
                crop=loading_config.crop,
                target_shape=loading_config.target_shape,
            )
        return data_raw, data_gt, data_mask

    def _show_figures(self, figures: list[FigureData]):
        for figure in figures:
            if figure.mode == 'text':
                self.logger.info(f'{figure.figure}')
            else:
                plt.show()

    def _compute_metrics(self, 
        data_segmented: ndarray, 
        data_gt: ndarray, 
        data_mask: ndarray,
        threshold: Optional[float] = None,
        operator_name: Optional[str] = None,
        hessian_time_seconds: Optional[float] = None,
        bdr_stats: Optional[dict] = None,
        conn_metrics: Optional[dict] = None, 
        all_metrics: Optional[dict] = None,
    ) -> Metrics:
        """
        Calcule toutes les métriques (classiques + clDice + composantes connexes
        + temps de calcul Hessien + Bifurcation Detection Rate).
        """
        # Métriques classiques
        if all_metrics is None:
            all_metrics = detailed_metrics(data_segmented, data_gt, data_mask, threshold=threshold)
        bdr_stats = bdr_stats or {}
        conn_metrics = conn_metrics or {}
        merged_metrics = {**all_metrics, **conn_metrics}
        metrics_computed = all_metrics.get('_metrics_computed', ["all"])
        return Metrics(
            dice=all_metrics['dice'],
            mcc=all_metrics['mcc'],
            roc=all_metrics['roc'],
            pr=all_metrics['pr'],
            cldice=all_metrics['cldice'],                         
            components_ratio=all_metrics['components_ratio'], 
            n_components_pred=all_metrics['n_components_pred'],
            n_components_gt=all_metrics['n_components_gt'],    
            excess_components=all_metrics['excess_components'], 
            largest_ratio=all_metrics['largest_ratio'],        
            threshold=threshold,
            
            skeleton_component_connectivity=all_metrics.get('skeleton_component_connectivity', 0),
            largest_component_recall=merged_metrics.get('largest_gt_recall', 0),
            gt_fragmentation=merged_metrics.get('fragmentation_ratio', 0),
            pred_small_components=all_metrics.get('pred_small_components', 0),
            pred_medium_components=all_metrics.get('pred_medium_components', 0),
            pred_large_components=all_metrics.get('pred_large_components', 0),
            gt_small_components=all_metrics.get('gt_small_components', 0),
            gt_medium_components=all_metrics.get('gt_medium_components', 0),
            gt_large_components=all_metrics.get('gt_large_components', 0),
            operator_name=operator_name,
            hessian_time_seconds=hessian_time_seconds,
            bifurcation_detection_rate=bdr_stats.get('bifurcation_detection_rate'),
            bifurcation_precision=bdr_stats.get('bifurcation_precision'),
            n_bifurcations_gt=bdr_stats.get('n_bifurcations_gt'),
            n_bifurcations_detected=bdr_stats.get('n_bifurcations_detected'),
            n_bifurcations_pred=bdr_stats.get('n_bifurcations_pred'),
            
            largest_gt_recall=merged_metrics.get('largest_gt_recall', 0),
            largest_component_overlap=merged_metrics.get('largest_component_overlap', 0),
            fragmentation_ratio=merged_metrics.get('fragmentation_ratio', 0),
            metrics_computed=metrics_computed
        )

    @log_section('Benchmark execution')
    def run(self,
            benchmark_config: BenchmarkConfig,
            experiment_config: ExperimentConfig,
        ) -> list[dict]:

        if hasattr(self, 'detailed_results'):
            self.detailed_results = {}

        data_raw, data_gt, data_mask = self.load_data(experiment_config.loading)

        i = 0
        image_name = Path(experiment_config.loading.raw_file).stem
        results: BenchmarkResults = {
            param: {value: None for value in values}
            for param, values in benchmark_config.params.items()
        }
        all_outputs = []
        for param, values in benchmark_config.params.items():
            for value in tqdm(values, desc=f"Processing {image_name} - {param:<12}"):

                exp_config = self._update_config(
                    config=deepcopy(experiment_config),
                    param=param,
                    value=value,
                )

                experiment = self._run_experiment(
                    data_raw=data_raw,
                    data_gt=data_gt,
                    data_mask=data_mask,
                    experiment_config=exp_config,
                    patient_id=image_name,
                    benchmark_config=benchmark_config,
                )

                # Récupérer
                threshold = getattr(experiment, 'threshold', None)
                hessian_time_seconds = getattr(experiment, 'hessian_time_seconds', None)
                bdr_stats = getattr(experiment, 'bdr_stats', None)
                conn_metrics = getattr(experiment, 'conn_metrics', None)
                all_metrics = getattr(experiment, 'all_metrics', None)
                # Calculer les métriques avec le seuil
                metrics = self._compute_metrics(
                    data_segmented=experiment.data_segmented,
                    data_gt=data_gt,
                    data_mask=data_mask,
                    threshold=threshold,
                    hessian_time_seconds=hessian_time_seconds,
                    bdr_stats=bdr_stats,
                    conn_metrics=conn_metrics,
                    all_metrics=all_metrics
                )

                experiment.metrics = metrics
                experiment.id = f"{image_name}_{i}"
                results[param][value] = experiment

                # Log enrichi avec clDice et fragmentation
                computed = set(metrics.metrics_computed)
                parts = [f"[{param}={value}]", f"Dice: {metrics.dice:.3f}", f"MCC: {metrics.mcc:.3f}"]
                if 'cldice' in computed or 'all' in computed:
                    parts.append(f"clDice: {metrics.cldice:.3f}")
                if 'components' in computed or 'all' in computed:
                    parts.append(f"FragRatio: {metrics.components_ratio:.2f}")
                if 'roc' in computed or 'all' in computed:
                    parts.append(f"ROC: {metrics.roc:.3f}")
                if 'pr' in computed or 'all' in computed:
                    parts.append(f"PR: {metrics.pr:.3f}")
                parts.append(f"HessianTime: {metrics.hessian_time_seconds}")
                if 'bifurcation' in computed or 'all' in computed:
                    parts.append(f"BDR: {metrics.bifurcation_detection_rate:.3f}")
                self.logger.info(" | ".join(parts))
                i += 1

        benchmark_data = BenchmarkData(
            data_raw=data_raw,
            data_gt=data_gt,
            results=results,
        )

        if self.plot_mode or self.save_mode:
            figures = self._create_figures(benchmark_data)
        if self.save_mode:
            self.saver.save_results(results, image_name, 'results')
            for figure in figures:
                self.saver.save_figure(figure, image_name)

        output = []
        for param, values in results.items():
            for value, experiment in values.items():
                output.append({
                    'method': value,
                    'param': param,
                    'dice': experiment.metrics.dice,
                    'mcc': experiment.metrics.mcc,
                    'roc': experiment.metrics.roc,
                    'pr': experiment.metrics.pr,
                    'cldice': experiment.metrics.cldice,                    
                    'components_ratio': experiment.metrics.components_ratio, 
                    'n_components_pred': experiment.metrics.n_components_pred,  
                    'n_components_gt': experiment.metrics.n_components_gt,      
                    'excess_components': experiment.metrics.excess_components,  
                    'largest_ratio': experiment.metrics.largest_ratio,         
                    'skeleton_component_connectivity': experiment.metrics.skeleton_component_connectivity,
                    'largest_component_recall': experiment.metrics.largest_component_recall,
                    'gt_fragmentation': experiment.metrics.gt_fragmentation,
                    'pred_small_components': experiment.metrics.pred_small_components,
                    'pred_medium_components': experiment.metrics.pred_medium_components,
                    'pred_large_components': experiment.metrics.pred_large_components,
                    'gt_small_components': experiment.metrics.gt_small_components,
                    'gt_medium_components': experiment.metrics.gt_medium_components,
                    'gt_large_components': experiment.metrics.gt_large_components,
                    'threshold': experiment.metrics.threshold,                 
                    'operator_name': experiment.metrics.operator_name,        
                    'hessian_time_seconds': experiment.metrics.hessian_time_seconds,      
                    'bifurcation_detection_rate': experiment.metrics.bifurcation_detection_rate, 
                    'bifurcation_precision': experiment.metrics.bifurcation_precision,            
                    'n_bifurcations_gt': experiment.metrics.n_bifurcations_gt,                    
                    'n_bifurcations_detected': experiment.metrics.n_bifurcations_detected,        
                    'largest_gt_recall': experiment.metrics.largest_gt_recall,
                    'largest_component_overlap': experiment.metrics.largest_component_overlap,
                    'fragmentation_ratio': experiment.metrics.fragmentation_ratio,
                    'metrics_computed': experiment.metrics.metrics_computed,
                })
        self.last_benchmark_results = results
        return output