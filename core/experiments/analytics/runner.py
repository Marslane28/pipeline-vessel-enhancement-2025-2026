from monai import metrics
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import fields
from tabulate import tabulate
from scipy.ndimage import label as scipy_label
import matplotlib.gridspec as gridspec
from core.experiments.analytics.base import AnalyticsBase
from core.config.benchmark import BenchmarkResults, RunnerResultsParsed
from core.config.figure import FigureData
from core.config.metrics import Metrics
from core.experiments.metrics import enhanced_stats, confusion_matrix, roc_curve, precision_recall_curve

class AnalyticsRunner(AnalyticsBase):
    """Analytics runner for Hessian benchmark analysis."""
    
    # CONSTANTS
    _DATASET_UNIT_LABEL = {
         "vascusynth": "volumes",
    }
    ALL_METRICS = [
        'dice', 'mcc', 'roc', 'pr', 'cldice',
        'components_ratio', 'excess_components', 'largest_ratio',
        'skeleton_component_connectivity', 'largest_component_recall',
        'gt_fragmentation', 'pred_small_components', 'pred_medium_components',
        'pred_large_components', 'gt_small_components', 'gt_medium_components',
        'gt_large_components', 'n_components_pred', 'n_components_gt',
        'largest_gt_recall', 'largest_component_overlap', 'fragmentation_ratio',
        'bifurcation_detection_rate', 'bifurcation_precision', 'hessian_time_seconds',
        'n_bifurcations_gt', 'n_bifurcations_pred', 'n_bifurcations_detected',
        'threshold',
    ]


    _NON_STATISTICAL_METRICS = {'operator_name'}
    

    _METRIC_GROUP_MAP = {
        'dice': {'dice'},
        'mcc': {'mcc'},
        'roc': {'roc'},
        'pr': {'pr'},
        'cldice': {'cldice'},
        'components': {
            'components_ratio', 'excess_components', 'largest_ratio',
            'skeleton_component_connectivity', 'largest_component_recall',
            'gt_fragmentation', 'pred_small_components', 'pred_medium_components',
            'pred_large_components', 'gt_small_components', 'gt_medium_components',
            'gt_large_components', 'n_components_pred', 'n_components_gt',
            'largest_gt_recall', 'largest_component_overlap', 'fragmentation_ratio',
        },
        'bifurcation': {
            'bifurcation_detection_rate', 'bifurcation_precision',
            'n_bifurcations_gt', 'n_bifurcations_pred', 'n_bifurcations_detected',
        },
    }

    _ALWAYS_METRICS = {'threshold', 'hessian_time_seconds'}
    
    METRICS_CONFIG = {
        'dice': 'max',
        'mcc': 'max',
        'roc': 'max',
        'pr': 'max',
        'cldice': 'max',
        'components_ratio': 'min',
        'excess_components': 'min',
        'largest_ratio': 'max',
        'skeleton_component_connectivity': 'max',
        'largest_component_recall': 'max',
        'pred_small_components': 'min',
        'pred_medium_components': 'min',
        'pred_large_components': 'max',
        'gt_fragmentation': 'min',
        'largest_gt_recall': 'max',
        'largest_component_overlap': 'max',
        'fragmentation_ratio': 'min',
        'hessian_time_seconds': 'min', # On veut le plus rapide
        'bifurcation_detection_rate': 'max',
        'bifurcation_precision': 'max',
        'n_bifurcations_gt': 'none',
        'n_bifurcations_pred': 'none',
        'n_bifurcations_detected': 'max',
        'n_components_pred': 'min',
        'n_components_gt': 'none', # constant entre opérateurs (vérité terrain)
        'gt_small_components': 'none', # idem propriété de la GT, pas de l'opérateur
        'gt_medium_components': 'none',
        'gt_large_components': 'none',
    }
    
    OPERATORS = ["default", "gaussian", "farid", "cubic",
                 "trigonometric", "catmull", "bspline", "bezier", "scharr"]
    
    PARAM_LIST = ['alpha', 'beta', 'gamma', 'scales_min', 'scales_max']

    _OVERVIEW_METRICS = ['dice', 'mcc', 'cldice', 'skeleton_component_connectivity',
                         'largest_component_recall']
    
    # METRIC FILTERING METHODS
    def _unit_label(self, dataset_name: str) -> str:
        return self._DATASET_UNIT_LABEL.get((dataset_name or "").lower(), "patients")

    def _dataset_header(self, dataset_name: str, num_patients: int) -> str:
        unit = self._unit_label(dataset_name)
        label = (dataset_name or "IRCAD").upper()
        return f"{label} - {num_patients} {unit}"
    
    def _filter_metrics(self, metrics_computed) -> list:
        """Restreint self.ALL_METRICS à ce qui a réellement été calculé
        (cf. optimization.metrics / detailed_metrics['_metrics_computed'])."""
        computed = set(metrics_computed)
        if 'all'in computed:
            return list(self.ALL_METRICS)
        keep = set(self._ALWAYS_METRICS)
        for group, names in self._METRIC_GROUP_MAP.items():
            if group in computed:
                keep |= names
        return [m for m in self.ALL_METRICS if m in keep]

    def _extract_metrics_computed(self, results_raw: list[BenchmarkResults]) -> list:
        """Lit metrics_computed sur le premier experiment.metrics trouvé
        (identique pour tout le run, cf. optimization.metrics fixé une fois
        par BenchmarkConfig)."""
        for image_result in results_raw:
            for param, values in image_result.items():
                for value, experiment in values.items():
                    mc = getattr(experiment.metrics, 'metrics_computed', None)
                    if mc is not None:
                        return mc
        return ['all']
    
    # DATA PARSING METHODS
    
    def _parse_results(self, results: list[BenchmarkResults], params: dict[str, list[float]]) -> RunnerResultsParsed:
        """Parse benchmark results into structured format."""
        benchmark_data = {
            param: {
                value: {metric: [] for metric in self.ALL_METRICS}
                for value in values
            }
            for param, values in params.items()
        }

        for image_result in results:
            for param, values in image_result.items():
                for value, experiment in values.items():
                    metrics_obj = experiment.metrics
                    conn = getattr(metrics_obj, 'conn_metrics', {})
                    
                    for metric in self.ALL_METRICS:
                        if hasattr(metrics_obj, metric):
                            val = getattr(metrics_obj, metric, 0)
                        elif metric in conn:
                            val = conn.get(metric, 0)
                        else:
                            val = 0
                        
                        benchmark_data[param][value][metric].append(val if val is not None else 0)
                        
        return benchmark_data
    
    def _get_patient_scores(self, results_raw: list[BenchmarkResults]) -> dict:
        """Extract scores per patient."""
        patient_scores = {}
        
        for result in results_raw:
            raw_file = getattr(result, 'config', None)
            if raw_file and hasattr(raw_file, 'loading'):
                patient_name = Path(raw_file.loading.raw_file).stem
            else:
                patient_name = "unknown"
            
            mcc_scores = []
            dice_scores = []
            
            for method, experiment in result.get('derivator', {}).items():
                mcc_scores.append(getattr(experiment.metrics, 'mcc', 0))
                dice_scores.append(getattr(experiment.metrics, 'dice', 0))
            
            patient_scores[patient_name] = {
                'mean_mcc': np.mean(mcc_scores) if mcc_scores else 0,
                'mean_dice': np.mean(dice_scores) if dice_scores else 0,
                'best_mcc': max(mcc_scores) if mcc_scores else 0,
                'best_dice': max(dice_scores) if dice_scores else 0,
                'mcc_scores': mcc_scores,
                'dice_scores': dice_scores
            }
        
        return patient_scores
    
    def _compute_statistics(self, results: dict, methods: list, metrics: list) -> tuple:
        """Compute mean, std, and all scores from results.

        Defensive: si une métrique demandée n'a pas été collectée pour un
        opérateur donné (clé absente, ou liste vide), on l'ignore plutôt que
        de faire planter le calcul (ex: métrique ajoutée à `Metrics` mais pas
        encore à `ALL_METRICS`, ou champ non numérique).
        """
        mean = {method: {metric: 0.0 for metric in metrics} for method in methods}
        std = {method: {metric: 0.0 for metric in metrics} for method in methods}
        all_scores = {metric: [] for metric in metrics}

        for method in methods:
            method_data = results.get('derivator', {}).get(method, {})
            for metric in metrics:
                scores = method_data.get(metric, [])
                numeric_scores = [
                    s for s in scores if isinstance(s, (int, float, np.integer, np.floating))
                ]
                if not numeric_scores:
                    mean[method][metric] = float('nan')
                    std[method][metric] = float('nan')
                    continue

                mean[method][metric] = np.mean(numeric_scores)
                std[method][metric] = np.std(numeric_scores)
                all_scores[metric].extend(numeric_scores)

        return mean, std, all_scores
    
    # TABLE GENERATION METHODS
    
    def _generate_mean_std_table(self, mean: dict, std: dict, methods: list, metrics: list) -> str:
        """Generate mean ± std table."""
        rows = []
        for method in methods:
            rows.append({
                'method': method,
                **{metric: f'{mean[method][metric]:.3f} ± {std[method][metric]:.3f}'for metric in metrics}
            })
        
        best_row = {'method': 'best'}
        for metric in metrics:
            if metric not in self.METRICS_CONFIG:
                best_row[metric] = 'N/A'
                continue
                
            direction = self.METRICS_CONFIG[metric]
            if direction == 'max':
                best_method = max(methods, key=lambda m: mean[m][metric])
            elif direction == 'min':
                best_method = min(methods, key=lambda m: mean[m][metric])
            else:
                best_method = 'N/A'
            best_row[metric] = best_method
            
        rows.append(best_row)
        return tabulate(rows, headers="keys", tablefmt='github')
    
    def _generate_patient_table(self, patient_scores: dict) -> str:
        """Generate per-patient results table."""
        patient_rows = []
        for patient_name, scores in sorted(patient_scores.items()):
            patient_rows.append({
                'patient': patient_name.replace('_images', ''),
                'mean_dice': f'{scores["mean_dice"]:.4f}',
                'best_dice': f'{scores["best_dice"]:.4f}',
                'mean_mcc': f'{scores["mean_mcc"]:.4f}',
                'best_mcc': f'{scores["best_mcc"]:.4f}'
            })
        return tabulate(patient_rows, headers="keys", tablefmt='github')
    
    def _generate_best_worst_summary(self, patient_scores: dict, header: str, unit: str, num_patients: int = None) -> str:
        """Generate best/worst patient summary."""
        best_patient = max(patient_scores.items(), key=lambda x: x[1]['mean_dice'])
        worst_patient = min(patient_scores.items(), key=lambda x: x[1]['mean_dice'])

        all_dices = [s['best_dice'] for s in patient_scores.values()]
        label = header if header else f"{len(patient_scores)} {unit}"

        content = f"\nMEILLEURS ET MOINS BONS {unit.upper()}\n\n"
        content += f"Best {unit[:-1]}: {best_patient[0]} (mean_dice={best_patient[1]['mean_dice']:.4f}, best_dice={best_patient[1]['best_dice']:.4f})\n"
        content += f"Worst {unit[:-1]}: {worst_patient[0]} (mean_dice={worst_patient[1]['mean_dice']:.4f}, best_dice={worst_patient[1]['best_dice']:.4f})\n"
        content += f"\nStatistiques globales sur {label} patients:\n"
        content += f"- Dice moyen: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}\n"
        content += f"- Dice min: {np.min(all_dices):.4f}\n"
        content += f"- Dice max: {np.max(all_dices):.4f}\n"
        content += f"- Median: {np.median(all_dices):.4f}\n"

        return content
    
    # OVERVIEW PLOT METHODS 3 synthetic figures, saved under overview/

    def _create_overview_metrics(self, results: dict, mean: dict, std: dict, methods: list, header: str,
                                  sel: list, num_patients: int) -> FigureData:
        """Overview #1 - boxplot + mean±std bar pour chaque métrique réellement
        calculée (cf. optimization.metrics). Vide → figure texte d'explication."""
        n_metrics = len(sel)
        if n_metrics == 0:
            return self._create_figure(
                "Aucune métrique de cette catégorie n'a été calculée (cf. optimization.metrics).",
                'overview/overview_metrics', 'text'
            )
        n_methods = len(methods)
        colors = plt.cm.tab10(np.linspace(0, 0.9, n_methods))

        fig = plt.figure(figsize=(20, 4 * n_metrics), dpi=120)
        gs = gridspec.GridSpec(n_metrics, 2, figure=fig,
                               width_ratios=[3, 1], hspace=0.5, wspace=0.25)

        for row, metric in enumerate(sel):
            ax_box = fig.add_subplot(gs[row, 0])
            data = [results['derivator'][m][metric] for m in methods]
            bp = ax_box.boxplot(data, patch_artist=True, vert=True,
                                medianprops=dict(color='black', linewidth=2))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax_box.set_xticks(range(1, n_methods + 1))
            ax_box.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
            ax_box.set_ylabel(metric, fontsize=9)
            ax_box.set_title(f'{metric} - distribution ({header})', fontsize=9)
            ax_box.grid(True, axis='y', alpha=0.3)

            ax_bar = fig.add_subplot(gs[row, 1])
            means = [mean[m][metric] for m in methods]
            errs = [std[m][metric] for m in methods]
            y_pos = np.arange(n_methods)
            ax_bar.barh(y_pos, means, xerr=errs, color=colors,
                        align='center', alpha=0.8, ecolor='gray', capsize=3)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(methods, fontsize=8)
            ax_bar.set_xlim(0, 1.05)
            ax_bar.set_title('mean ± std', fontsize=9)
            ax_bar.grid(True, axis='x', alpha=0.3)

            direction = self.METRICS_CONFIG.get(metric, 'max')
            if direction == 'max':
                best_idx = int(np.argmax(means))
            elif direction == 'min':
                best_idx = int(np.argmin(means))
            else:
                best_idx = None
            if best_idx is not None:
                bars_children = [c for c in ax_bar.patches]
                if best_idx < len(bars_children):
                    bars_children[best_idx].set_edgecolor('gold')
                    bars_children[best_idx].set_linewidth(2)

        fig.suptitle(f'Overview - Metrics per Operator ({header})',
                    fontsize=13, fontweight='bold', y=1.005)
        plt.tight_layout()
        return self._create_figure(fig, 'overview/overview_metrics', 'plot')

    def _create_overview_patients(self, results: dict, header: str, patient_scores: dict, methods: list) -> FigureData:
        """Overview #2 - heatmap patients × operators + dice bar + distribution on one figure.

        Replaces the previous best-dice barchart + dice distribution histogram + patient table figure.
        Saved as overview/overview_patients.
        """
        sorted_patients = sorted(patient_scores.keys())
        labels = [p.replace('_images', '') for p in sorted_patients]
        n_patients = len(sorted_patients)
        n_methods = len(methods)

        # Build dice matrix (patients × methods)
        dice_matrix = np.zeros((n_patients, n_methods))
        for i, patient in enumerate(sorted_patients):
            for j, method in enumerate(methods):
                scores = results['derivator'][method]['dice']
                dice_matrix[i, j] = scores[i] if i < len(scores) else 0.0

        fig = plt.figure(figsize=(max(14, n_methods * 1.4), n_patients * 0.45 + 6), dpi=120)
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               height_ratios=[3, 1], hspace=0.5, wspace=0.35)

        # Heatmap (full top row)
        ax_heat = fig.add_subplot(gs[0, :])
        im = ax_heat.imshow(dice_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax_heat.set_xticks(range(n_methods))
        ax_heat.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax_heat.set_yticks(range(n_patients))
        ax_heat.set_yticklabels(labels, fontsize=8)
        ax_heat.set_title('Dice score - patients × operators', fontsize=10, fontweight='bold')
        for i in range(n_patients):
            for j in range(n_methods):
                val = dice_matrix[i, j]
                ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=6, color='black'if 0.3 < val < 0.8 else 'white')
        plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)

        # Best dice per patient (horizontal bars)
        ax_bar = fig.add_subplot(gs[1, 0])
        best_dices = [patient_scores[p]['best_dice'] for p in sorted_patients]
        bar_colors = plt.cm.RdYlGn(np.array(best_dices))
        ax_bar.barh(labels, best_dices, color=bar_colors, edgecolor='none')
        ax_bar.axvline(np.mean(best_dices), color='steelblue', ls='--',
                       label=f'mean={np.mean(best_dices):.3f}')
        ax_bar.set_xlim(0, 1)
        ax_bar.set_title('Best Dice per patient', fontsize=9)
        ax_bar.legend(fontsize=8)
        ax_bar.grid(True, axis='x', alpha=0.3)

        # Dice distribution
        ax_hist = fig.add_subplot(gs[1, 1])
        ax_hist.hist(best_dices, bins=10, color='steelblue', edgecolor='white', alpha=0.8)
        ax_hist.axvline(np.mean(best_dices), color='red', ls='--', label=f'mean={np.mean(best_dices):.3f}')
        ax_hist.axvline(np.median(best_dices), color='green', ls='--', label=f'median={np.median(best_dices):.3f}')
        ax_hist.set_xlabel('Best Dice')
        ax_hist.set_ylabel('# patients')
        ax_hist.set_title('Distribution des best Dice', fontsize=9)
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)

        fig.suptitle(f'Overview - Patient Results ({header})'if header else 'Overview - Patient Results',fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._create_figure(fig, 'overview/overview_patients', 'plot')

    def _create_overview_params(self, results: dict, num_images: int) -> FigureData:
        """Overview #3 - all parameter analyses on a single compact figure.

        Replaces the previous one-figure-per-parameter approach (5 figures → 1).
        Saved as overview/overview_params.
        """
        params = [p for p in self.PARAM_LIST if p in results]
        n_params = len(params)
        if n_params == 0:
            return None

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6']
        fig = plt.figure(figsize=(14, 4 * n_params), dpi=120)
        gs = gridspec.GridSpec(n_params, 2, figure=fig,
                               width_ratios=[3, 2], hspace=0.5, wspace=0.35)

        for row, (param, color) in enumerate(zip(params, colors)):
            param_scores = results[param]
            values_sorted = sorted(param_scores.keys())

            means, stds = [], []
            all_scores_per_value = {v: param_scores[v]['mcc'] for v in values_sorted}
            best_counts = {v: 0 for v in values_sorted}

            for v in values_sorted:
                scores = param_scores[v]['mcc']
                means.append(np.mean(scores))
                stds.append(np.std(scores))

            for i in range(num_images):
                best_v = max(values_sorted,
                             key=lambda v: all_scores_per_value[v][i]
                             if i < len(all_scores_per_value[v]) else 0)
                best_counts[best_v] += 1

            # Mean ± std line
            ax_line = fig.add_subplot(gs[row, 0])
            ax_line.plot(values_sorted, means, '+-', color=color, linewidth=2, markersize=7)
            ax_line.fill_between(values_sorted,
                                 np.array(means) - np.array(stds),
                                 np.array(means) + np.array(stds),
                                 color=color, alpha=0.18)
            ax_line.set_xlabel(param, fontsize=9)
            ax_line.set_ylabel('MCC', fontsize=9)
            ax_line.set_title(f'{param} - influence sur MCC', fontsize=9)
            ax_line.grid(True, alpha=0.3)
            if param in ['alpha', 'beta', 'gamma']:
                ax_line.set_xscale('log')

            # Best-value frequency histogram
            ax_hist = fig.add_subplot(gs[row, 1])
            x_labels = [str(v) for v in best_counts]
            y_vals = list(best_counts.values())
            bars = ax_hist.bar(x_labels, y_vals, color=color, edgecolor='white', alpha=0.85)
            for bar, yi in zip(bars, y_vals):
                if yi > 0:
                    ax_hist.text(bar.get_x() + bar.get_width() / 2, yi + 0.2,
                                 str(yi), ha='center', fontsize=8)
            ax_hist.set_xlabel(param, fontsize=9)
            ax_hist.set_ylabel('Fréquence (best)', fontsize=9)
            ax_hist.set_title(f'{param} - valeur optimale / patient', fontsize=9)
            ax_hist.grid(True, axis='y', alpha=0.3)

        fig.suptitle('Overview - Parameter Analysis (MCC)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._create_figure(fig, 'overview/overview_params', 'plot')

    # LEGACY PLOT HELPERS (kept intact, no longer called in public API)
    
    def _create_boxplots(self, results: dict, methods: list, metrics: list) -> list[FigureData]:
        """Create box plots for each metric."""
        figures = []
        
        for metric in metrics:
            data = [results['derivator'][method][metric] for method in methods]
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(methods)))
            
            fig = plt.figure(figsize=(12, 8))
            bp = plt.boxplot(data, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            plt.xticks(ticks=range(1, len(methods) + 1), labels=methods, rotation=45, ha='right')
            plt.title(f'Distribution de {metric} sur 20 patients')
            plt.ylabel(metric)
            plt.grid(True, axis='y')
            plt.tight_layout()

            figures.append(self._create_figure(fig, f'box_{metric}', 'plot'))
            plt.close(fig)
            
        return figures
    
    def _create_best_dice_chart(self, patient_scores: dict) -> FigureData:
        """Create bar chart of best dice per patient."""
        fig = plt.figure(figsize=(14, 6))
        patients = [name.replace('_images', '') for name in sorted(patient_scores.keys())]
        best_dices = [patient_scores[name]['best_dice'] for name in sorted(patient_scores.keys())]
        
        colors_bar = plt.cm.RdYlGn(np.array(best_dices) / max(best_dices))
        bars = plt.bar(patients, best_dices, color=colors_bar)
        plt.axhline(y=np.mean(best_dices), color='red', linestyle='--', label=f'Moyenne: {np.mean(best_dices):.4f}')
        
        plt.xlabel('Patient')
        plt.ylabel('Meilleur Dice')
        plt.title('Meilleur score Dice par patient')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        return self._create_figure(fig, 'best_dice_per_patient', 'plot')
    
    def _create_radar_chart(self, mean: dict, methods: list, metrics: list) -> FigureData:
        """Create radar chart of mean metrics."""
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(methods)))

        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(111, polar=True)

        for method, color in zip(methods, colors):
            values = [mean[method][metric] for metric in metrics]
            values += values[:1]
            ax.plot(angles, values, label=method, color=color)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        plt.title('Radar métriques moyennes (20 patients)')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()

        return self._create_figure(fig, 'radar', 'plot')
    
    def _create_dice_distribution(self, patient_scores: dict) -> FigureData:
        """Create histogram of dice distribution."""
        all_dices = [s['best_dice'] for s in patient_scores.values()]
        
        fig = plt.figure(figsize=(10, 6))
        plt.hist(all_dices, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(x=np.mean(all_dices), color='red', linestyle='--', label=f'Moyenne: {np.mean(all_dices):.4f}')
        plt.axvline(x=np.median(all_dices), color='green', linestyle='--', label=f'Médiane: {np.median(all_dices):.4f}')
        plt.xlabel('Dice Score')
        plt.ylabel('Nombre de patients')
        plt.title('Distribution des meilleurs Dice scores (20 patients)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return self._create_figure(fig, 'dice_distribution', 'plot')
    
    def _create_param_analysis_plots(self, results: dict, num_images: int) -> list[FigureData]:
        """Create parameter analysis plots."""
        figures = []
        colors = ['red', 'dodgerblue', 'limegreen', 'orangered', 'magenta']
        
        for color, param in zip(colors, self.PARAM_LIST):
            if param not in results:
                continue
                
            param_scores = results[param]
            scores_per_value = {v: metrics['mcc'] for v, metrics in param_scores.items()}
            values_sorted = sorted(scores_per_value.keys())
            
            values, means, stds = [], [], []
            for value, metrics in results[param].items():
                scores = metrics['mcc']
                means.append(np.mean(scores))
                stds.append(np.std(scores))
                values.append(value)
            
            best_counts = {v: 0 for v in values_sorted}
            for i in range(num_images):
                best_value = max(values_sorted, key=lambda v: scores_per_value[v][i])
                best_counts[best_value] += 1
            
            fig = self._create_param_subplot(param, values, means, stds, best_counts, color)
            figures.append(self._create_figure(fig, f'{param}_analysis', 'plot'))
            plt.close(fig)
            
        return figures
    
    def _create_param_subplot(self, param: str, values: list, means: list,
                              stds: list, best_counts: dict, color: str) -> plt.Figure:
        """Create the actual parameter subplot figure."""
        fig = plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(values, means, '+-', label="mean", color=color, linewidth=2, markersize=8)
        plt.fill_between(values, np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds), color=color, alpha=0.2, label='± std')
        plt.xlabel(param)
        plt.ylabel('MCC Score')
        plt.title(f'Influence du paramètre {param} (moyenne sur 20 patients)')
        plt.legend()
        plt.grid(True)
        if param in ['alpha', 'beta', 'gamma']:
            plt.xscale('log')
        
        plt.subplot(1, 2, 2)
        x_labels = [str(v) for v in best_counts.keys()]
        y_values = list(best_counts.values())
        bars = plt.bar(x_labels, y_values, color=color, edgecolor='black')
        plt.xlabel(param)
        plt.ylabel("Fréquence")
        plt.title(f"Histogramme des meilleures valeurs de {param} (sur 20 patients)")
        plt.grid(True, axis='y')
        
        for bar, val in zip(bars, y_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        arr = arr.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)
    
    # ENHANCED STATISTICS METHODS
    
    def get_enhanced_stats_table(self, results_raw: list[BenchmarkResults]) -> list[FigureData]:
        """Generate enhanced statistics table."""
        rows = []
        op_means = {}
        
        for result in results_raw:
            raw_file = getattr(result, 'config', None)
            if raw_file and hasattr(raw_file, 'loading'):
                patient_name = Path(raw_file.loading.raw_file).stem.replace('_images', '')
            else:
                patient_name = "unknown"
                
            for method, experiment in result.get('derivator', {}).items():
                if hasattr(experiment, 'data_enhanced'):
                    stats = enhanced_stats(experiment.data_enhanced)
                    rows.append({
                        'patient': patient_name,
                        'operator': method,
                        'vessel_ratio': f"{stats['vessel_ratio']:.2%}",
                        'mean': f"{stats['mean']:.4f}",
                        'std': f"{stats['std']:.4f}",
                        'is_binary': stats['is_binary']
                    })
                    
                    if method not in op_means:
                        op_means[method] = {'vessel_ratio': [], 'mean': [], 'std': []}
                    op_means[method]['vessel_ratio'].append(stats['vessel_ratio'])
                    op_means[method]['mean'].append(stats['mean'])
                    op_means[method]['std'].append(stats['std'])
        
        if op_means:
            rows.append({})
            for method, stats in op_means.items():
                rows.append({
                    'patient': f"MEAN ({len(stats['vessel_ratio'])} patients)",
                    'operator': method,
                    'vessel_ratio': f"{np.mean(stats['vessel_ratio']):.2%}",
                    'mean': f"{np.mean(stats['mean']):.4f}",
                    'std': f"{np.mean(stats['std']):.4f}",
                })
        
        if rows:
            table = tabulate(rows, headers="keys", tablefmt='github')
            content = f"\n ENHANCED DATA STATISTICS (vessel ratio per operator)\n"+ table
            return [self._create_figure(content, 'enhanced_stats', 'text')]
        
        return []
    
    # MAIN PUBLIC METHODS
    
    def get_hessian_figures(self, results_raw: list[BenchmarkResults], params: dict[str, list], dataset_name: str = None) -> list[FigureData]:
        """Generate all Hessian benchmark figures."""
        results = self._parse_results(results_raw, params)
        methods = list(results['derivator'].keys())

        metrics_computed = self._extract_metrics_computed(results_raw)
        allowed = set(self._filter_metrics(metrics_computed))
        metrics = [m for m in self.ALL_METRICS if m not in self._NON_STATISTICAL_METRICS and m in allowed]

        mean, std, all_scores = self._compute_statistics(results, methods, metrics)
        patient_scores = self._get_patient_scores(results_raw)
        num_patients = len(patient_scores)
        header = self._dataset_header(dataset_name, num_patients)
        unit = self._unit_label(dataset_name)

        figures = []

        table_content = f"\n BENCHMARK HESSIAN - METRICS TABLE (mean over {header})\n"
        table_content += self._generate_mean_std_table(mean, std, methods, metrics)

        figures.append(self._create_figure(table_content, 'table_mean', 'text'))

        patient_table = f"\n RÉSULTATS PAR {unit.upper()[:-1]} ({header})\n"
        patient_table += self._generate_patient_table(patient_scores)
        figures.append(self._create_figure(patient_table, 'table_patients', 'text'))

        sel = [m for m in self._OVERVIEW_METRICS if m in allowed]
        figures.append(self._create_overview_metrics(results, mean, std, methods, header, sel, num_patients))
        figures.append(self._create_overview_patients(results, header, patient_scores, methods))

        self._generate_best_worst_summary(patient_scores, header, unit, num_patients)

        figures.extend(self.get_enhanced_stats_table(results_raw))

        return figures
    
    def get_enhancement_figures(self, results_raw: list[BenchmarkResults], params: dict[str, list]) -> list[FigureData]:
        """Generate enhancement parameter analysis figures."""
        results = self._parse_results(results_raw, params)
        
        first_param = next(iter(results.keys()))
        first_value = next(iter(results[first_param].keys()))
        num_images = len(results[first_param][first_value]['mcc'])
        
        # Single synthetic overview instead of one figure per parameter
        overview = self._create_overview_params(results, num_images)
        return [overview] if overview else []
    
    # PATIENT COMPARISON METHODS
    
    def generate_patient_comparison_figures(self, benchmark, output_dir=None):
        """Generate all patient comparison figures."""
        if not hasattr(benchmark, '_segmented_per_patient'):
            print("No segmented data available")
            return
        
        for patient_id, patient_data in benchmark._segmented_per_patient.items():
            if 'operators'in patient_data:
                ops_data = patient_data['operators']
            else:
                ops_data = {k: v for k, v in patient_data.items()
                            if k not in ['data_raw', 'data_gt', 'data_mask']}
            data_gt = patient_data['data_gt']
            data_mask = patient_data['data_mask']
            data_raw = patient_data.get('data_raw', data_gt)
            
            self._visualize_patient_comparison(benchmark, patient_id, data_raw, data_gt, data_mask, ops_data, output_dir)
            self._visualize_all_operators_cc(benchmark, patient_id, ops_data, data_gt, data_mask, output_dir)
            
            segmented_per_operator = {}
            for op, d in ops_data.items():
                seg = d.get('segmented')
                if seg is not None:
                    segmented_per_operator[op] = seg
        
        self._generate_global_boxplots(benchmark, output_dir)
    
    def _visualize_patient_comparison(self, benchmark, patient_id, data_raw, data_gt, data_mask, ops_data, output_dir):
        """Generate single figure with CC overlays + metrics + confusion."""
        operators = list(ops_data.keys())
        n_ops = len(operators)
        
        gt_bin = data_gt > 0.5
        if data_mask is not None:
            gt_bin &= data_mask > 0.5
        
        dim = data_raw.shape[2]
        gt_per_slice = np.array([gt_bin[:, :, z].sum() for z in range(dim)])
        margin = max(1, dim // 20)
        gt_per_slice[:margin] = 0
        gt_per_slice[-margin:] = 0
        best_z = np.argmax(gt_per_slice) if gt_per_slice.sum() > 0 else dim // 2
        
        fig = plt.figure(figsize=(n_ops * 4, 14), dpi=150)
        gs = gridspec.GridSpec(3, n_ops, figure=fig, hspace=0.35, wspace=0.25)
        
        struct = np.ones((3, 3, 3), dtype=np.uint8)
        gt_labels, n_gt = scipy_label(gt_bin, structure=struct)
        rng_gt = np.random.default_rng(42)
        gt_colors = rng_gt.uniform(0.3, 1.0, (n_gt + 1, 3)).astype(np.float32)
        gt_colors[0] = 0.0
        alpha_c = 0.72
        
        raw_slice = self._normalize(data_raw[:, :, best_z])
        raw_rgb = np.stack([raw_slice, raw_slice, raw_slice], axis=-1)
        
        tp_vals, fp_vals, fn_vals, pred_vals, gt_vals = [], [], [], [], []
        
        for col, op_name in enumerate(operators):
            op_data = benchmark._segmented_per_patient[patient_id]['operators'][op_name]
            if isinstance(op_data, dict):
                seg = op_data.get('segmented')
                thresh = op_data.get('threshold', 0.5)
                dice_val = op_data.get('dice', 0.0)
                cldice_score = op_data.get('cldice', 0.0)
                conn_val = op_data.get('conn_metrics', {}).get('skeleton_component_connectivity', 0.0)
            else:
                seg = op_data
                thresh = 0.5
                dice_val = cldice_score = conn_val = 0.0
            if seg is None:
                print(f"Missing segmented data for {op_name} on patient {patient_id}")
                continue
            
            ax0 = fig.add_subplot(gs[0, col])
            pred_bin = seg > thresh
            if data_mask is not None:
                pred_bin &= data_mask > 0.5
                
            pred_labels, n_pred = scipy_label(pred_bin, structure=struct)
            rng_pred = np.random.default_rng(hash(op_name) % 2**32)
            pred_colors = rng_pred.uniform(0.3, 1.0, (n_pred + 1, 3)).astype(np.float32)
            pred_colors[0] = 0.0
            
            display = raw_rgb.copy()
            gt_slice = gt_labels[:, :, best_z]
            display[gt_slice > 0] = (1 - alpha_c) * display[gt_slice > 0] + alpha_c * np.array([0.0, 0.8, 0.0])
            pred_slice = pred_labels[:, :, best_z]
            m_pred = pred_slice > 0
            display[m_pred] = (1 - alpha_c) * display[m_pred] + alpha_c * pred_colors[pred_slice[m_pred]]
            ax0.imshow(np.clip(display, 0, 1), origin="lower")
            ax0.set_title(f"{op_name}\nGT:{n_gt} | Pred:{n_pred}", fontsize=8)
            ax0.axis("off")
            
            ax1 = fig.add_subplot(gs[1, col])
            dice_val = op_data.get('dice', 0.0)
            cldice_score = op_data.get('cldice', 0.0)
            conn_val = op_data.get('conn_metrics', {}).get('skeleton_component_connectivity', 0.0)
            metrics_vals = [dice_val, cldice_score, conn_val]
            bars = ax1.bar(["Dice", "clDice", "Conn."], metrics_vals,
                           color=["#1565C0", "#00838F", "#558B2F"], alpha=0.8)
            ax1.set_ylim(0, 1)
            ax1.axhline(0.7, color="green", ls="--", alpha=0.5)
            for bar, val in zip(bars, metrics_vals):
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha="center", fontsize=7)
            ax1.set_title("Metrics", fontsize=8)
            
            cm = confusion_matrix(seg, data_gt, data_mask)
            tp_vals.append(cm['tp'])
            fp_vals.append(cm['fp'])
            fn_vals.append(cm['fn'])
            pred_vals.append(cm['pred_vessels'])
            gt_vals.append(cm['gt_vessels'])
        
        for col, (tp, fp, fn) in enumerate(zip(tp_vals, fp_vals, fn_vals)):
            ax2 = fig.add_subplot(gs[2, col])
            ax2.bar(0, tp, color="forestgreen", label="TP", alpha=0.85)
            ax2.bar(0, fp, bottom=tp, color="orange", label="FP", alpha=0.85)
            ax2.bar(0, fn, bottom=tp+fp, color="crimson", label="FN", alpha=0.85)
            ax2.set_xticks([])
            ax2.set_ylabel("Voxels")
            ax2.set_title(f"{operators[col]}", fontsize=8)
            if col == 0:
                ax2.legend(fontsize=6)
        
        ax_last = fig.add_subplot(gs[2, n_ops-1])
        x = np.arange(n_ops)
        ax_last.bar(x - 0.17, pred_vals, 0.35, label="Pred", color="#1976D2", alpha=0.85)
        ax_last.bar(x + 0.17, gt_vals, 0.35, label="GT", color="#9E9E9E", alpha=0.85)
        ax_last.set_xticks(x)
        ax_last.set_xticklabels(operators, rotation=45, ha="right", fontsize=7)
        ax_last.set_ylabel("Voxels")
        ax_last.set_title("Predicted vs GT", fontsize=9, fontweight="bold")
        ax_last.legend(fontsize=7)
        
        plt.tight_layout()
        benchmark._save_or_show(fig, f"patient_comparison_{patient_id}", output_dir)
        plt.close(fig)
    
    def _visualize_all_operators_cc(self, benchmark, patient_id, ops_data, data_gt, data_mask, output_dir):
        """Display GT + all operators side by side."""
        patient_data = benchmark._segmented_per_patient[patient_id]
        data_gt = patient_data['data_gt']
        data_mask = patient_data['data_mask']
        operators = list(ops_data.keys())
        n_cols = len(operators) + 1
        
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5), dpi=150)
        
        gt_bin = data_gt > 0.5
        if data_mask is not None:
            gt_bin &= data_mask > 0.5
            
        gt_labels, n_gt = scipy_label(gt_bin)
        dim = gt_bin.shape[2]
        gt_per_slice = np.array([gt_bin[:, :, z].sum() for z in range(dim)])
        margin = max(1, dim // 20)
        gt_per_slice[:margin] = 0
        gt_per_slice[-margin:] = 0
        best_z = np.argmax(gt_per_slice) if gt_per_slice.sum() > 0 else dim // 2
        
        raw_norm = self._normalize(data_gt[:, :, best_z])
        raw_rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
        alpha_c = 0.72
        
        ax_gt = axes[0]
        gt_slice = gt_labels[:, :, best_z]
        rng_gt = np.random.default_rng(42)
        gt_colors = rng_gt.uniform(0.3, 1.0, (n_gt + 1, 3)).astype(np.float32)
        gt_colors[0] = 0.0
        display = raw_rgb.copy()
        display[gt_slice > 0] = (1 - alpha_c) * display[gt_slice > 0] + alpha_c * gt_colors[gt_slice[gt_slice > 0]]
        ax_gt.imshow(np.clip(display, 0, 1), origin="lower")
        ax_gt.set_title(f"GROUND TRUTH\n{n_gt} components", fontsize=9, fontweight="bold")
        ax_gt.axis("off")
        
        for idx, op_name in enumerate(operators):
            ax = axes[idx + 1]
            op_data = benchmark._segmented_per_patient[patient_id]['operators'][op_name]
            seg = op_data.get('segmented')
            if seg is None:
                print(f"Missing segmented data for {op_name} on patient {patient_id}")
                continue
            pred_bin = seg > op_data['threshold']
            if data_mask is not None:
                pred_bin &= data_mask > 0.5
                
            pred_labels, n_pred = scipy_label(pred_bin)
            rng_pred = np.random.default_rng(hash(op_name) % 2**32)
            pred_colors = rng_pred.uniform(0.3, 1.0, (n_pred + 1, 3)).astype(np.float32)
            pred_colors[0] = 0.0
            
            display = raw_rgb.copy()
            pred_slice = pred_labels[:, :, best_z]
            m_pred = pred_slice > 0
            display[m_pred] = (1 - alpha_c) * display[m_pred] + alpha_c * pred_colors[pred_slice[m_pred]]
            ax.imshow(np.clip(display, 0, 1), origin="lower")
            ax.set_title(f"{op_name.upper()}\nGT:{n_gt} | Pred:{n_pred}", fontsize=8)
            ax.axis("off")
        
        plt.tight_layout()
        benchmark._save_or_show(fig, f"all_operators_cc_{patient_id}", output_dir)
        plt.close(fig)
    
    def _generate_global_boxplots(self, benchmark, output_dir):
        """Generate global boxplots for all operators."""
        if not hasattr(benchmark, '_patient_results'):
            return
        
        op_data = {}
        for op in self.OPERATORS:
            op_data[op] = {"dice": [], "cldice": [], "mcc": []}
        
        for pat_res in benchmark._patient_results.values():
            for op, metrics in pat_res.items():
                if op in op_data:
                    op_data[op]["dice"].append(metrics.get("dice", 0))
                    op_data[op]["cldice"].append(metrics.get("cldice", 0))
                    op_data[op]["mcc"].append(metrics.get("mcc", 0))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle("Global Metrics Comparison", fontsize=14, fontweight="bold")
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.OPERATORS)))
        
        for ax, metric, title, ylim in zip(
            axes, ["dice", "cldice", "mcc"],
            ["Dice Coefficient", "clDice", "MCC"],
            [(0, 1), (0, 1), (-0.2, 1)]
        ):
            data = [op_data[op][metric] for op in self.OPERATORS]
            bp = ax.boxplot(data, labels=self.OPERATORS, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.axhline(0.7 if metric != "mcc"else 0.5, color="green", ls="--", alpha=0.6)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        benchmark._save_or_show(fig, "global_boxplots", output_dir)
        plt.close(fig)