import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pathlib import Path
from pprint import pformat
from tabulate import tabulate

from core.config.experiment import Experiment, ExperimentConfig
from core.config.figure import FigureData
from core.experiments.analytics.base import AnalyticsBase
from core.experiments.metrics import roc_curve, precision_recall_curve, detailed_metrics
from core.utils.helpers import create_error_map


class AnalyticsHessian(AnalyticsBase):
    
    # Configuration Methods
    def get_configs(self, experiments: list[Experiment]) -> FigureData:
        from dataclasses import asdict
        from pathlib import Path
        best_configs: dict[str, ExperimentConfig] = {}
        best_mcc: dict[str, float] = {}
        for experiment in experiments:
            derivator = experiment.config.methods.derivator
            mcc = experiment.metrics.mcc if experiment.metrics else -1

            if derivator not in best_configs or mcc > best_mcc.get(derivator, -1):
                best_configs[derivator] = experiment.config
                best_mcc[derivator] = mcc
        file_stem = Path(experiments[0].config.loading.raw_file).stem
        content = f"EXPERIMENT CONFIG FOR < {file_stem} >\n"
        for method, config in best_configs.items():
            config_dict = asdict(config)
            config_str = pformat(config_dict, indent=4, sort_dicts=False)
            content += f"\n{method.upper()} (best MCC={best_mcc[method]:.4f}):\n{config_str}\n"
            
        return self._create_figure(content, 'configs', 'text')
    
    # Metrics Methods
    
    def get_metrics(self, experiments: list[Experiment]) -> FigureData:
        rows = []
        
        for experiment in experiments:
            row = self._build_metrics_row(experiment)
            rows.append(row)
        
        # Add best methods row
        best_methods_row = self._compute_best_methods(rows)
        rows.append(best_methods_row)
        
        table = tabulate(rows, headers="keys", tablefmt='github')
        content = f'\nHESSIAN BENCHMARK METRICS\n{table}'
        
        return self._create_figure(content, 'metrics', 'text')
    
    def _build_metrics_row(self, experiment: Experiment) -> dict:
        """Build a metrics dictionary for a single experiment."""
        metrics_obj = experiment.metrics
        
        return {
            'method': experiment.config.methods.derivator,
            # Basic metrics
            'dice': f"{metrics_obj.dice:.4f}",
            'mcc': f"{metrics_obj.mcc:.4f}",
            'roc': f"{metrics_obj.roc:.4f}",
            'pr': f"{metrics_obj.pr:.4f}",
            'cldice': f"{metrics_obj.cldice:.4f}",
            'components_ratio': f"{metrics_obj.components_ratio:.4f}",
            'excess_components': f"{metrics_obj.excess_components:.0f}",
            # Connectivity metrics
            'skeleton_component_connectivity': f"{metrics_obj.skeleton_component_connectivity:.4f}",
            'largest_component_recall': f"{metrics_obj.largest_component_recall:.4f}",
            'gt_fragmentation': f"{metrics_obj.gt_fragmentation:.4f}",
            'pred_small_components': f"{metrics_obj.pred_small_components:.0f}",
            'pred_medium_components': f"{metrics_obj.pred_medium_components:.0f}",
            'pred_large_components': f"{metrics_obj.pred_large_components:.0f}",
            'largest_gt_recall': f"{metrics_obj.largest_gt_recall:.4f}",
            'largest_component_overlap': f"{metrics_obj.largest_component_overlap:.4f}",
            'fragmentation_ratio': f"{metrics_obj.fragmentation_ratio:.4f}",
            'hessian_time_seconds': f"{metrics_obj.hessian_time_seconds:.4f}",
            'bifurcation_detection_rate': f"{metrics_obj.bifurcation_detection_rate:.4f}",
            'bifurcation_precision': f"{metrics_obj.bifurcation_precision:.4f}",
            'n_bifurcations_gt': f"{metrics_obj.n_bifurcations_gt:.0f}",
            'n_bifurcations_pred': f"{metrics_obj.n_bifurcations_pred:.0f}",
            'n_bifurcations_detected': f"{metrics_obj.n_bifurcations_detected:.0f}",
        }
    
    def _compute_best_methods(self, rows: list[dict]) -> dict:
        """Compute which method performs best for each metric."""
        best_methods = {'method': 'best'}
        
        #  plus haut meilleur
        higher_is_better = ['dice', 'mcc', 'roc', 'pr', 'cldice',
                            'skeleton_component_connectivity', 'largest_component_recall',
                            'largest_gt_recall', 'largest_component_overlap',
                            'bifurcation_detection_rate', 'bifurcation_precision',
                            'n_bifurcations_detected']
        for metric in higher_is_better:
            best_methods[metric] = max(rows, key=lambda r: float(r[metric]))['method']
        
        #  plus bas meilleur
        lower_is_better = ['components_ratio', 'excess_components', 'gt_fragmentation',
                           'pred_small_components', 'pred_medium_components', 'pred_large_components',
                           'fragmentation_ratio', 'hessian_time_seconds',
                           'n_bifurcations_gt', 'n_bifurcations_pred']
        for metric in lower_is_better:
            best_methods[metric] = min(rows, key=lambda r: float(r[metric]))['method']
        
        return best_methods
    
    # Visualization Methods
    
    def visualize_patient_continuity(self, patient_results: dict, patient_id: str, output_dir=None):
        """Visualize continuity vs Dice score for a single patient."""
        operators = list(patient_results.keys())
        ratios = [patient_results[op].get('conn_metrics', {}).get('components_ratio', 0)
                  for op in operators]
        dices = [patient_results[op].get('dice', 0) for op in operators]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot points
        for i, op in enumerate(operators):
            ax.scatter(ratios[i], dices[i], s=120, alpha=0.7, label=op)
            ax.annotate(op, (ratios[i], dices[i]), xytext=(5, 5), textcoords='offset points')
        
        # Add reference lines
        ax.axvline(x=2, color='orange', linestyle='--', alpha=0.5, label='Ratio limite')
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Dice limite')
        
        # Formatting
        ax.set_xlabel('Ratio de fragmentation (plus petit = mieux)')
        ax.set_ylabel('Dice (plus grand = mieux)')
        ax.set_title(f'Patient {patient_id} - Continuité vasculaire VS Dice')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_or_show(fig, f"continuity_{patient_id}", output_dir)
        plt.close(fig)
    
    # Histogram Methods
    
    def get_histograms(self,
            experiments: list[Experiment],
            data_raw: np.ndarray,
            data_gt: np.ndarray,
            bins: int = 50,
            density: bool = False,
            color: str = 'dodgerblue',
        ) -> FigureData:
        
        ncols = len(experiments) + 1
        nrows = 2
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=(4.5 * ncols, 3 * nrows))
        
        # Add vertical labels
        self._add_histogram_vertical_labels(fig)
        
        # Prepare data and titles
        row_data, row_titles = self._prepare_histogram_data(experiments, data_raw, data_gt)
        
        # Create histograms
        self._plot_histograms(axs, row_data, row_titles, bins, density, color)
        
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.95,
                          wspace=0.15, hspace=0.2)
        
        figure = self._create_figure(fig, 'histograms', 'plot')
        plt.close(fig)
        
        return figure
    
    def _add_histogram_vertical_labels(self, fig):
        """Add vertical labels for histogram rows."""
        fig.text(0.015, 0.752, 'Enhanced', va='center', ha='center',
                rotation='vertical', fontsize=11, fontweight='bold')
        fig.text(0.015, 0.278, 'Segmented', va='center', ha='center',
                rotation='vertical', fontsize=11, fontweight='bold')
    
    def _prepare_histogram_data(self, experiments, data_raw, data_gt):
        """Prepare row data and titles for histograms."""
        row_data = [
            [data_raw] + [exp.data_enhanced for exp in experiments],
            [data_gt] + [exp.data_segmented for exp in experiments],
        ]
        
        row_titles = [
            ['raw data'] + [exp.config.methods.derivator for exp in experiments],
            ['ground truth'] + [exp.config.methods.derivator for exp in experiments],
        ]
        
        return row_data, row_titles
    
    def _plot_histograms(self, axs, row_data, row_titles, bins, density, color):
        """Plot all histograms."""
        nrows, ncols = axs.shape
        
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].hist(row_data[row][col].ravel(),
                                  bins=bins, density=density, color=color)
                axs[row, col].set_title(row_titles[row][col], fontsize=9)
                
                if row == 0:
                    axs[row, col].set_ylabel('Density'if density else 'Frequency')
                else:
                    axs[row, col].set_xlabel('Intensity')
                    axs[row, col].set_ylabel('Density'if density else 'Frequency')
                
                axs[row, col].grid(True)
                axs[row, col].tick_params('y', labelsize=8, labelrotation=90)
    
    # Curve Methods
    
    def get_curves(self,
            experiments: list[Experiment],
            ground_truth: ndarray
        ) -> FigureData:
        
        y_true = ground_truth.ravel()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(experiments)))
        
        # Plot curves for each experiment
        for i, experiment in enumerate(experiments):
            y_scores = experiment.data_enhanced.ravel()
            self._plot_roc_curve(ax1, y_true, y_scores, experiment, colors[i])
            self._plot_pr_curve(ax2, y_true, y_scores, experiment, colors[i])
        
        # Format subplots
        self._format_roc_axis(ax1)
        self._format_pr_axis(ax2)
        
        fig.tight_layout()
        figure = self._create_figure(fig, 'curves', 'plot')
        plt.close(fig)
        
        return figure
    
    def _plot_roc_curve(self, ax, y_true, y_scores, experiment, color):
        """Plot ROC curve for a single experiment."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, label=experiment.config.methods.derivator, color=color)
    
    def _plot_pr_curve(self, ax, y_true, y_scores, experiment, color):
        """Plot Precision-Recall curve for a single experiment."""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ax.plot(recall, precision, label=experiment.config.methods.derivator, color=color)
    
    def _format_roc_axis(self, ax):
        """Format ROC curve subplot."""
        ax.plot([0, 1], [0, 1], 'k--', label='random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True)
    
    def _format_pr_axis(self, ax):
        """Format Precision-Recall curve subplot."""
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True)
    
    # View Methods
    
    def get_views(self,
            experiments: list[Experiment],
            data_gt: ndarray,
            data_raw: ndarray
        ) -> list[FigureData]:
        
        dim = data_gt.ndim
        methods = [exp.config.methods.derivator for exp in experiments]
        
        # Prepare data arrays
        data_enhanced = [data_raw] + [exp.data_enhanced for exp in experiments]
        data_segmented = [data_gt] + [exp.data_segmented for exp in experiments]
        error_maps = [data_gt] + [create_error_map(data_gt, exp.data_segmented)
                                  for exp in experiments]
        
        titles_enhanced = ['raw data'] + methods
        titles_segmented = ['ground truth'] + methods
        titles_error_maps = ['ground truth'] + methods
        
        # Generate views based on dimensionality
        if dim == 2:
            figures = self._get_2d_views(data_enhanced, data_segmented, error_maps,
                                        titles_enhanced, titles_segmented, titles_error_maps)
        else: # dim == 3
            figures = self._get_3d_views(data_enhanced, data_segmented, error_maps,
                                        titles_enhanced, titles_segmented, titles_error_maps)
        
        plt.close('all')
        return figures
    
    def _get_2d_views(self, data_enhanced, data_segmented, error_maps,
                      titles_enhanced, titles_segmented, titles_error_maps):
        """Generate 2D visualization views."""
        plot_enhanced = self.viewer.display_images(data_enhanced, titles=titles_enhanced)
        plot_segmented = self.viewer.display_images(data_segmented, titles=titles_segmented,
                                                     binary_mode=True)
        plot_error_maps = self.viewer.display_images(error_maps, titles=titles_error_maps,
                                                      error_mode=True)
        
        fig_enhanced = self._create_figure(plot_enhanced, 'images_enhanced', 'plot')
        fig_segmented = self._create_figure(plot_segmented, 'images_segmented', 'plot')
        fig_error_maps = self._create_figure(plot_error_maps, 'error_maps', 'plot')
        
        return [fig_enhanced, fig_segmented, fig_error_maps]
    
    def _get_3d_views(self, data_enhanced, data_segmented, error_maps,
                      titles_enhanced, titles_segmented, titles_error_maps):
        """Generate 3D visualization views."""
        mip_enhanced = self.viewer.display_mip(data_enhanced, titles=titles_enhanced)
        slices_enhanced = self.viewer.display_slices(data_enhanced, titles=titles_enhanced,
                                                      interval=100)
        slices_segmented = self.viewer.display_slices(data_segmented, titles=titles_segmented,
                                                       interval=100, binary_mode=True)
        slices_error_maps = self.viewer.display_slices(error_maps, titles=titles_error_maps,
                                                        interval=100, error_mode=True)
        
        fig_mip_enhanced = self._create_figure(mip_enhanced, 'mip_enhanced', 'plot')
        fig_slices_enhanced = self._create_figure(slices_enhanced, 'slices_enhanced', 'anim')
        fig_slices_segmented = self._create_figure(slices_segmented, 'slices_segmented', 'anim')
        fig_slices_error_maps = self._create_figure(slices_error_maps, 'slices_error_maps', 'anim')
        
        return [fig_mip_enhanced, fig_slices_enhanced, fig_slices_segmented, fig_slices_error_maps]