import os
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd

from core.utils.black_ridges import detect_black_ridges
from core.experiments.benchmarks.hessian import BenchmarkHessian
from core.experiments.benchmarks.base import BenchmarkBase
from core.experiments.analytics.runner import AnalyticsRunner
from core.io.saver import Saver
from core.io.loader import Loader
from core.io.logger import setup_logger
from core.config.benchmark import BenchmarkConfig
from core.config.experiment import ExperimentConfig
from core.config.setup import SetupConfig
from core.config.figure import FigureData
from core.processing.derivator import Derivator
from core.utils.patient_loader import PatientLoader
from core.utils.decorator import log_time, log_section, log_init

class BenchmarkRunner:

    @log_init()
    def __init__(self, setup: SetupConfig, dataset: str = None, enhancer: str = None):
        self.setup = setup
        self.plot_mode = setup.plot_mode
        self.save_mode = setup.save_mode
        self.logger = setup_logger(name=setup.log_file, debug_mode=setup.debug_mode)
        self.loader = Loader(setup.input_dir, self.logger)
        self.dataset = dataset
        self.enhancer = enhancer
        self.saver = Saver(
            experiment_name=setup.name,
            output_dir=setup.output_dir,
            logger=self.logger,
            dataset=dataset,
            enhancer=enhancer
        ) if self.save_mode else None
        self.patient_loader = PatientLoader(input_dir=setup.input_dir, logger=self.logger, dataset=dataset)
        self.analytics = AnalyticsRunner()
        self.analytics.logger = self.logger
      
    _AVAILABLE_ENHANCERS = ("frangi", "jerman", "mfat")

    def _apply_enhancer_choice(
        self,
        experiment_config: ExperimentConfig,
        enhancer: Literal["frangi", "jerman", "mfat"] = "frangi",
    ) -> None:
        if enhancer not in self._AVAILABLE_ENHANCERS:
            raise ValueError(
                f"enhancer invalide : {enhancer!r}. "
                f"Valeurs acceptées : {self._AVAILABLE_ENHANCERS}"
            )
        experiment_config.methods.enhancer = enhancer
        self.logger.info(f"[RUNNER] Filtre de rehaussement sélectionné : {enhancer}")
    _FRANGI_GRID_KEYS = frozenset({"alpha", "beta", "gamma"})
    _JERMAN_GRID_KEYS = frozenset({"tau"})
    _MFAT_GRID_KEYS = frozenset({"mfat_tau", "mfat_tau2", "mfat_step_size", "variant"})
    _GRID_KEYS_BY_ENHANCER = {
        "frangi": _FRANGI_GRID_KEYS,
        "jerman": _JERMAN_GRID_KEYS,
        "mfat": _MFAT_GRID_KEYS,
    }

    def _filter_grid_for_enhancer(
        self,
        benchmark_config: BenchmarkConfig,
        enhancer: Literal["frangi", "jerman", "mfat"],
    ) -> None:
        if not benchmark_config.params_grid:
            return

        all_enhancer_specific_keys = frozenset().union(*self._GRID_KEYS_BY_ENHANCER.values())
        keep_keys = self._GRID_KEYS_BY_ENHANCER.get(enhancer, frozenset())
        drop_keys = all_enhancer_specific_keys - keep_keys
        removed = [k for k in drop_keys if k in benchmark_config.params_grid]
        for k in removed:
            del benchmark_config.params_grid[k]

        if removed:
            self.logger.info(
                f"[RUNNER] Grid search restreint à '{enhancer}'- paramètre(s) "
                f"non pertinent(s) retiré(s) : {removed}"
            )

    # BENCHMARK INITIALISATION

    def _get_benchmark(self, benchmark_config: BenchmarkConfig) -> BenchmarkBase:
        if benchmark_config.mode == "hessian":
            return BenchmarkHessian(
                save_mode=self.save_mode,
                plot_mode=self.plot_mode,
                logger=self.logger,
                loader=self.loader,
                saver=self.saver,
                params_grid=benchmark_config.params_grid,
                input_dir=self.setup.input_dir,
            )
        raise ValueError(f"Benchmark mode inconnu : {benchmark_config.mode}")

    # FIGURES

    def _save_figures(self, figures: list[FigureData]):
        for i, figure in enumerate(figures):
            if figure.name is None:
                figure.name = f"figure_{i}"

        for figure in figures:
            if figure.mode == "text":
                self.logger.info(figure.figure)
                if self.save_mode and self.saver:
                    text_path = Path(self.saver.output_dir) / f"{figure.name}.txt"
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.write_text(figure.figure)
                    self.logger.info(f"[TEXT] Saved → {text_path}")
        
            elif self.plot_mode:
                plt.figure(figure.figure.number)
                plt.show()

            if self.save_mode and figure.mode != "text":
                self.saver.save_figure(figure, "overview")

    # EXECUTION PRINCIPALE

    @log_time()
    @log_section("Runner execution")
    def run(
        self,
        images_dir: str,
        labels_dir: str,
        benchmark_config: BenchmarkConfig,
        experiment_config: ExperimentConfig,
        masks_dir: str = None,
        n_patients: Optional[int] = None,
        enhancer: Literal["frangi", "jerman", "mfat"] = "frangi",
        patient_ids: Optional[Union[str, int, list]] = None,

    ) -> Optional[Path]:
        
        if enhancer:
            self._apply_enhancer_choice(experiment_config, enhancer)
            self._filter_grid_for_enhancer(benchmark_config, enhancer)
        elif self.enhancer:
            self._apply_enhancer_choice(experiment_config, self.enhancer)
            self._filter_grid_for_enhancer(benchmark_config, self.enhancer)

        if enhancer:
            self.logger.info(f"[RUNNER] Filtre de rehaussement choisi : {enhancer}")
            self.saver = Saver(
                experiment_name=self.setup.name,
                output_dir=self.setup.output_dir,
                logger=self.logger,
                dataset=self.dataset,
                enhancer=enhancer,
            )
        raw_files, gt_files = self.patient_loader.get_files(images_dir, labels_dir, n_patients, patient_ids=patient_ids)
        mask_files = (
            self.patient_loader.get_mask_files(masks_dir, n_patients, patient_ids=patient_ids)
            if masks_dir
            else [None] * len(raw_files)
        )

        n_total = len(raw_files)
        self.logger.info(f"[RUNNER] Démarrage du benchmark sur {n_total} patient(s).")

        benchmark = self._get_benchmark(benchmark_config)

        # Optimisation groupwise des scales (Lamy 2020)
        if getattr(benchmark_config, "optimize_scales", False):
            first_raw_file = raw_files[0] if raw_files else None
            voxel_size = None
            spacing = None

            if first_raw_file is not None:
                full_path = first_raw_file if os.path.isabs(first_raw_file) or os.path.exists(first_raw_file) else os.path.join("data", self.setup.input_dir, first_raw_file)
                try:
                    spacing = nib.load(full_path).header.get_zooms()
                    voxel_size = float(np.min(spacing))
                    self.logger.info(f"[SCALE OPT] spacing lu: {spacing}, voxel_size={voxel_size:.6f} mm")
                except Exception as e:
                    self.logger.warning(f"[SCALE OPT] Impossible de lire le spacing de {full_path} ({e})")

            if voxel_size is None or spacing is None:
                dataset_type = getattr(self.patient_loader, 'dataset', 'ircad')

                if dataset_type == "bullitt":
                    voxel_size = 0.513
                    self.logger.warning(f"[SCALE OPT] Fallback voxel_size=0.56mm pour Bullitt")
                elif dataset_type == "ircad":
                    voxel_size = 1.0
                    self.logger.warning(f"[SCALE OPT] Fallback voxel_size=1.0mm pour IRCAD")
                elif dataset_type == "sennet":
                    voxel_size = 0.0052
                    self.logger.warning(f"[SCALE OPT] Fallback voxel_size=1.0mm pour SenNet kidney_1_voi")
                elif dataset_type == "vascusynth":
                    voxel_size = 1.0
                    self.logger.warning(f"[SCALE OPT] Fallback voxel_size=01mm pour VascuSynth")

                spacing = (voxel_size, voxel_size, voxel_size)
                self.logger.info(f"[SCALE OPT] voxel_size utilisé: {voxel_size:.6f} mm")
            
            experiment_config.enhancement.voxel_size = voxel_size
            dataset_type = getattr(self.patient_loader, 'dataset', 'ircad')
            # Ces valeurs sont choisies pour chaque dataset en fonction de la taille des vaisseaux dans la vérité de terrain et du voxel_size
            if dataset_type == "bullitt":
                sigma_mins = np.arange(0.2, 0.7, 0.1)
                sigma_maxs = np.arange(0.8, 2.6, 0.2)
                self.logger.info("[RUNNER] Scales adaptées pour Bullitt (vaisseaux fins)")
            elif dataset_type == "ircad":
                sigma_mins = np.arange(0.6, 1.8, 0.2)
                sigma_maxs = np.arange(1.6, 3.0, 0.2)
                self.logger.info("[RUNNER] Scales adaptées pour IRCAD (vaisseaux moyens)")
            elif dataset_type == "sennet":
                sigma_mins = np.arange(0.026, 0.156, 0.026)
                sigma_maxs = np.arange(0.104, 0.442, 0.052)
                self.logger.info("[RUNNER] Scales adaptées pour SenNet kidney_1_voi (vaisseaux très fins)")
            elif dataset_type == "vascusynth":
                sigma_mins = np.arange(0.6, 1.8, 0.2)
                sigma_maxs = np.arange(1.6, 3.0, 0.2)
                self.logger.info("[RUNNER] Scales adaptées pour VascuSynth (vaisseaux moyens, 1-3mm)")
            best_scales = benchmark.optimize_scales(
                raw_files=raw_files,
                gt_files=gt_files,
                mask_files=mask_files,
                experiment_config=experiment_config,
                sigma_mins=sigma_mins,
                sigma_maxs=sigma_maxs,
                n_scales=4
            )
            experiment_config.enhancement.scales_mm = best_scales
            experiment_config.enhancement.scales = best_scales
            self.logger.info(f"[RUNNER] Scales optimales : {best_scales}")

        all_results: list[dict] = []
        all_raw_results: list = [] 

        for idx, (raw_file, gt_file, mask_file) in enumerate(
            zip(raw_files, gt_files, mask_files), start=1
        ):
            patient_id = PatientLoader.patient_id_from_path(raw_file)
            self.logger.info(f"[RUNNER] Patient {idx}/{n_total} : {patient_id}")

            exp_config = deepcopy(experiment_config)
            exp_config.loading.raw_file = raw_file
            exp_config.loading.gt_file = gt_file
            exp_config.loading.mask_file = mask_file

            if getattr(exp_config.loading, "use_physical_units", False):
                full_path = (
                    raw_file
                    if os.path.isabs(raw_file) or os.path.exists(raw_file)
                    else os.path.join("data", self.setup.input_dir, raw_file)
                )
                spacing = nib.load(full_path).header.get_zooms()
                voxel_size = float(np.min(spacing))
                exp_config.enhancement.voxel_size = voxel_size
                scales_mm = exp_config.enhancement.scales_mm
                exp_config.enhancement.scales = [round(s / voxel_size, 2) for s in scales_mm]
                self.logger.debug(
                    f"[RUNNER] {patient_id} | spacing={tuple(round(s, 3) for s in spacing)} mm "
                    f"| voxel_size={voxel_size:.3f} mm"
                )

            results = benchmark.run(benchmark_config, exp_config)
            if results is None:
                self.logger.warning(f"[RUNNER] Pas de résultats pour {patient_id}, on continue.")
                continue

            raw_results = getattr(benchmark, "last_benchmark_results", None)
            if raw_results is not None:
                all_raw_results.append(raw_results)

            for r in results:
                r["patient"] = patient_id
                all_results.append(r)
                if 'threshold'not in r:
                    if hasattr(benchmark, 'last_threshold'):
                        r['threshold'] = benchmark.last_threshold
                    else:
                        r['threshold'] = None
                threshold = r.get("threshold", None)
                hessian_time = r.get("hessian_time_seconds", None)
                thresh_display = f"thresh={threshold:.3f}"if threshold is not None else "thresh=N/A"
                hessian_time_display = f"{hessian_time:.4f}s"if hessian_time is not None else "time=N/A"

                computed = set(r.get('_metrics_computed', ['dice', 'mcc']))
                parts = [f"[{r['method']}] {patient_id} →"]
                parts.append(f"Dice : {r['dice']:.3f}")
                parts.append(f"MCC : {r['mcc']:.3f}")
                if 'cldice'in computed or 'all'in computed:
                    cldice_val = r.get("cldice", r.get("cldice_score", 0.0))
                    parts.append(f"clDice : {cldice_val:.3f}")
                if 'components'in computed or 'all'in computed:
                    conn_metrics = r.get("conn_metrics", {})
                    comp_ratio = conn_metrics.get("components_ratio", 0.0)
                    parts.append(f"Fragmentation : {comp_ratio:.2f}")
                if 'roc'in computed or 'all'in computed:
                    parts.append(f"ROC : {r['roc']:.3f}")
                if 'pr'in computed or 'all'in computed:
                    parts.append(f"PR : {r['pr']:.3f}")
                parts.append(f"HessianTime : {hessian_time_display}")
                if 'bifurcation'in computed or 'all'in computed:
                    bdr = r.get("bifurcation_detection_rate", None)
                    if bdr is not None:
                        parts.append(f"BifurcationDetectionRate : {bdr:.3f}")
                parts.append(thresh_display)
                self.logger.info("| ".join(parts))

        if hasattr(benchmark, "_save_confusion_json"):
            benchmark._save_confusion_json()
            self.logger.info("[RUNNER] JSON des matrices de confusion sauvegardé")
        if benchmark_config.mode == "hessian":
            if self.save_mode and self.saver:
                output_dir = str(self.saver.output_dir)
                self.logger.info(f"[RUNNER] Sauvegarde des matrices de hessian dans {output_dir}")
            else:
                output_dir = "figures/hessian/patients"# Default output dir for figures if saver is not used
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.analytics.generate_patient_comparison_figures(benchmark, output_dir)
            self.logger.info(f"[RUNNER] Figures de comparaison inter-opérateurs générées dans {output_dir}")

            if all_raw_results:
                dataset_name = getattr(self.patient_loader, 'dataset', self.dataset or 'ircad')
                overview_figures = self.analytics.get_hessian_figures(
                    all_raw_results, benchmark_config.params,
                    dataset_name=dataset_name,
                )
                self._save_figures(overview_figures)
                self.logger.info(
                    f"[RUNNER] Overview global généré ({len(overview_figures)} figure(s)/tableau(x)) "
                    f"sur {len(all_raw_results)} patient(s)."
                )
            else:
                self.logger.warning(
                    "[RUNNER] Aucun résultat brut accumulé - overview global non généré."
                )

        # Statistiques globales
        if all_results:
            all_dices = [r["dice"] for r in all_results]

            cldice_computed = any('cldice'in set(r.get('_metrics_computed', ['cldice'])) for r in all_results)
            components_computed = any('components'in set(r.get('_metrics_computed', ['components'])) for r in all_results)

            by_patient = {}
            for r in all_results:
                by_patient.setdefault(r["patient"], []).append(r["dice"])
            patient_means = {p: float(np.mean(d)) for p, d in by_patient.items()}
            best_patient = max(patient_means, key=patient_means.get)
            worst_patient = min(patient_means, key=patient_means.get)

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"RÉSULTATS GLOBAUX - {n_total} patient(s)")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Dice moyen : {np.mean(all_dices):.3f}")
            self.logger.info(f"Dice min : {np.min(all_dices):.4f}")
            self.logger.info(f"Dice max : {np.max(all_dices):.4f}")
            if cldice_computed:
                all_cldices = [r.get("cldice", r.get("cldice_score", 0.0)) for r in all_results]
                self.logger.info(f"clDice moyen : {np.mean(all_cldices):.4f} ± {np.std(all_cldices):.4f}")

            if components_computed:
                all_ratios = [r.get("conn_metrics", {}).get("components_ratio", 0.0) for r in all_results]
                self.logger.info(f"Fragmentation : {np.mean(all_ratios):.2f} ± {np.std(all_ratios):.2f} (ratio pred/GT)")

            self.logger.info(f"Médiane Dice : {np.median(all_dices):.4f}")
            self.logger.info(f"Meilleur patient: {best_patient} (Dice={patient_means[best_patient]:.4f})")
            self.logger.info(f"Moins bon : {worst_patient} (Dice={patient_means[worst_patient]:.4f})")

            if cldice_computed:
                methods = list(set(r["method"] for r in all_results))
                best_op_idx = np.argmax([
                    np.mean([r.get("cldice", 0) for r in all_results if r.get("method") == op])
                    for op in methods
                ])
                self.logger.info(f"Meilleur clDice : {methods[best_op_idx]}")
            self.logger.info(f"{'='*60}")
        

        self.logger.info(f"[RUNNER] Benchmark terminé : {n_total} patient(s) traité(s).")

        output_dir = str(self.saver.output_dir) if self.save_mode and self.saver else None
        if self.save_mode and self.saver:
            return Path(self.saver.output_dir) / "results"
        return None

    # HELPERS FICHIERS

    def _get_patient_files(
        self,
        patient_id: str,
        images_dir: str,
        labels_dir: str,
        masks_dir: str = None,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        raw_file = None
        gt_file = None
        mask_file = None

        images_path = Path(images_dir)
        if images_path.exists():
            for f in sorted(images_path.glob(f"{patient_id}*.nii*")):
                raw_file = str(f)
                break

        labels_path = Path(labels_dir)
        if labels_path.exists():
            for f in sorted(labels_path.glob(f"{patient_id}*.nii*")):
                gt_file = str(f)
                break

        if masks_dir:
            masks_path = Path(masks_dir)
            if masks_path.exists():
                for f in sorted(masks_path.glob(f"{patient_id}*.nii*")):
                    mask_file = str(f)
                    break

        return raw_file, gt_file, mask_file

    def _build_operator_map(self) -> dict:
        
        derivator = Derivator(use_gpu=False)

        return {
            "default": derivator.default,
            "gaussian": derivator.gaussian,
            "farid": derivator.farid,
            "cubic": derivator.cubic,
            "trigonometric": derivator.trigonometric,
            "catmull": derivator.catmull,
            "bspline": derivator.bspline,
            "bezier": derivator.bezier,
            "scharr": derivator.scharr,
        }

    def _get_operator_function(self, derivator_name: str, operator_map: Optional[dict] = None):
        if operator_map is None:
            operator_map = self._build_operator_map()

        if derivator_name not in operator_map:
            raise ValueError(f"Opérateur inconnu : {derivator_name}")
        return operator_map[derivator_name]

    # ANALYSE FP BATCH tous les patients, tous les opérateurs

    @log_time()
    @log_section("Batch FP Analysis - all patients")
    def batch_fp_analysis(
        self,
        experiment_config: ExperimentConfig,
        benchmark_config: BenchmarkConfig,
        images_dir: str,
        labels_dir: str,
        masks_dir: Optional[str] = None,
        n_patients: int=20,
        derivator_names: Optional[list[str]] = None,
        output_dir: Optional[str] = None,
        patient_ids: Optional[Union[str, int, list]] = None,
        enhancer: Literal["frangi", "jerman", "mfat"] = "frangi",
    ) -> pd.DataFrame:
        self._apply_enhancer_choice(experiment_config, enhancer)
        self._filter_grid_for_enhancer(benchmark_config, enhancer)

        raw_files, gt_files = self.patient_loader.get_files(images_dir, labels_dir, n_patients, patient_ids=patient_ids)
        mask_files = (
            self.patient_loader.get_mask_files(masks_dir, n_patients, patient_ids=patient_ids)
            if masks_dir
            else [None] * len(raw_files)
        )

        csv_path: Optional[Path] = None
        if self.save_mode and self.saver:
            csv_path = Path(self.saver.output_dir) / "fp_batch_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

        done_pairs: set[tuple[str, str]] = set()
        if csv_path is not None and csv_path.exists():
            try:
                existing = pd.read_csv(csv_path)
                for _, row in existing.iterrows():
                    done_pairs.add((str(row["patient_id"]), str(row["operator"])))
                self.logger.info(
                    f"[BATCH FP] Reprise détectée - {len(done_pairs)} paire(s) déjà dans le CSV."
                )
            except Exception as e:
                self.logger.warning(f"[BATCH FP] Impossible de lire le CSV existant : {e}")

        CSV_COLUMNS = [
            "patient_id", "operator",
            "total_fp_voxels", "n_fp_components",
            "fp_sizes_mean", "fp_sizes_median", "fp_sizes_max", "fp_sizes_std",
            "fp_small_under10", "fp_medium_10_50", "fp_large_over50",
            "total_tp_voxels", "total_fn_voxels",
            "precision", "recall", "f1",
        ]

        if csv_path is not None and not csv_path.exists():
            pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)
            self.logger.info(f"[BATCH FP] CSV initialisé → {csv_path}")

        all_stats: list[dict] = []
        benchmark = self._get_benchmark(benchmark_config)

        for idx, (raw_file, gt_file, mask_file) in enumerate(
            zip(raw_files, gt_files, mask_files)
        ):
            patient_id = PatientLoader.patient_id_from_path(raw_file)
            self.logger.info(f"[BATCH FP] Patient {idx+1}/{len(raw_files)} : {patient_id}")

            exp_config = deepcopy(experiment_config)
            exp_config.loading.raw_file = raw_file
            exp_config.loading.gt_file = gt_file
            exp_config.loading.mask_file = mask_file

            original_params = None
            if derivator_names is not None:
                original_params = deepcopy(benchmark_config.params)
                benchmark_config.params = {"derivator": derivator_names}

            fp_results = benchmark.analyze_false_positives(
                experiment_config=exp_config,
                benchmark_config=benchmark_config,
                patient_id=patient_id,
            )

            if original_params is not None:
                benchmark_config.params = original_params

            for param, values in fp_results.items():
                for value, stats in values.items():
                    op_name = stats.get("operator", str(value))

                    if (patient_id, op_name) in done_pairs:
                        self.logger.info(
                            f"[BATCH FP] Déjà traité - {patient_id} / {op_name}, ignoré."
                        )
                        continue

                    row = {
                        "patient_id": patient_id,
                        "operator": op_name,
                        "total_fp_voxels": stats.get("total_fp_voxels", 0),
                        "n_fp_components": stats.get("n_fp_components", 0),
                        "fp_sizes_mean": stats.get("fp_sizes_mean", 0.0),
                        "fp_sizes_median": stats.get("fp_sizes_median", 0.0),
                        "fp_sizes_max": stats.get("fp_sizes_max", 0),
                        "fp_sizes_std": stats.get("fp_sizes_std", 0.0),
                        "fp_small_under10": stats.get("fp_small_under10", 0),
                        "fp_medium_10_50": stats.get("fp_medium_10_50", 0),
                        "fp_large_over50": stats.get("fp_large_over50", 0),
                        "total_tp_voxels": stats.get("total_tp_voxels", 0),
                        "total_fn_voxels": stats.get("total_fn_voxels", 0),
                        "precision": stats.get("precision", 0.0),
                        "recall": stats.get("recall", 0.0),
                        "f1": stats.get("f1", 0.0),
                    }
                    all_stats.append(row)

                    if csv_path is not None:
                        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)
                        self.logger.info(
                            f"[BATCH FP] {patient_id} / {op_name} → écrit dans {csv_path.name}"
                        )

        # Rapport de synthèse final (recharger depuis le CSV pour inclure les reprises)
        if csv_path is not None and csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(all_stats)

        if not df.empty:
            benchmark.create_fp_summary_report(
                all_stats=df.to_dict("records"),
                output_dir=output_dir,
            )

        self.logger.info(f"\n{'='*80}")
        self.logger.info("RÉSUMÉ BATCH ANALYSE FP")
        self.logger.info(f"{'='*80}")
        if not df.empty:
            summary = df.groupby("operator").agg(
                fp_voxels_mean=("total_fp_voxels", "mean"),
                fp_voxels_std=("total_fp_voxels", "std"),
                n_components_mean=("n_fp_components", "mean"),
                precision_mean=("precision", "mean"),
                recall_mean=("recall", "mean"),
                f1_mean=("f1", "mean"),
            ).round(3)
            self.logger.info(f"\n{summary.to_string()}")

        return df

    # ANALYSE BIFURCATIONS BATCH - tous les patients, tous les opérateurs

    @log_time()
    @log_section("Batch Bifurcation Analysis - all patients")
    def batch_bifurcation_analysis(
        self,
        experiment_config: ExperimentConfig,
        benchmark_config: BenchmarkConfig,
        images_dir: str,
        labels_dir: str,
        masks_dir: Optional[str] = None,
        n_patients: int = 20,
        patient_ids: Optional[Union[str, int, list]] = None,
        tolerance_radius: int = 2,
        roi_half_size: int = 4,
        output_dir: Optional[str] = None,
        enhancer: Literal["frangi", "jerman", "mfat"] = "frangi",
    ) -> pd.DataFrame:
        self._apply_enhancer_choice(experiment_config, enhancer)
        self._filter_grid_for_enhancer(benchmark_config, enhancer)

        raw_files, gt_files = self.patient_loader.get_files(images_dir, labels_dir, n_patients, patient_ids=patient_ids)
        mask_files = (
            self.patient_loader.get_mask_files(masks_dir, n_patients, patient_ids=patient_ids)
            if masks_dir
            else [None] * len(raw_files)
        )

        benchmark = self._get_benchmark(benchmark_config)
        all_bifurcation_stats: dict[str, dict] = {}

        for idx, (raw_file, gt_file, mask_file) in enumerate(
            zip(raw_files, gt_files, mask_files)
        ):
            patient_id = PatientLoader.patient_id_from_path(raw_file)
            self.logger.info(
                f"[BATCH BIFURCATION] Patient {idx + 1}/{len(raw_files)} : {patient_id}"
            )

            exp_config = deepcopy(experiment_config)
            exp_config.loading.raw_file = raw_file
            exp_config.loading.gt_file = gt_file
            exp_config.loading.mask_file = mask_file

            patient_results = benchmark.analyze_bifurcations(
                experiment_config=exp_config,
                benchmark_config=benchmark_config,
                patient_id=patient_id,
                tolerance_radius=tolerance_radius,
                roi_half_size=roi_half_size,
            )
            all_bifurcation_stats[patient_id] = patient_results

        if all_bifurcation_stats:
            benchmark.create_bifurcation_summary_report(
                all_bifurcation_stats=all_bifurcation_stats,
                output_dir=output_dir,
            )

        rows = []
        for patient_id, op_results in all_bifurcation_stats.items():
            for op_name, stats in op_results.items():
                rows.append({
                    "patient_id": patient_id,
                    "operator": op_name,
                    "bdr": stats["bdr_stats"]["bifurcation_detection_rate"],
                    "bifurcation_precision": stats["bdr_stats"]["bifurcation_precision"],
                    "n_bifurcations_gt": stats["bdr_stats"]["n_bifurcations_gt"],
                    "local_dice_mean": stats["local_dice_mean"],
                    "center_dip_ratio_mean": stats["center_dip_ratio_mean"],
                    "hessian_time_seconds": stats.get("hessian_time_seconds"),
                })
        df = pd.DataFrame(rows)

        self.logger.info(f"\n{'='*80}")
        self.logger.info("RÉSUMÉ BATCH ANALYSE BIFURCATIONS")
        self.logger.info(f"{'='*80}")
        if not df.empty:
            summary = df.groupby("operator").agg(
                bdr_mean=("bdr", "mean"),
                bdr_std=("bdr", "std"),
                local_dice_mean=("local_dice_mean", "mean"),
                center_dip_mean=("center_dip_ratio_mean", "mean"),
                hessian_time_mean=("hessian_time_seconds", "mean"),
            ).round(4)
            self.logger.info(f"\n{summary.to_string()}")

        return df

    # COMPARAISON DES OPÉRATEURS (single patient)

    @log_section("Operators FP Comparison")
    def compare_operators_fp(
        self,
        experiment_config: ExperimentConfig,
        benchmark_config: BenchmarkConfig,
        patient_id: str,
        operators: list[str],
        output_dir: str = "fp_comparison",
        enhancer: Literal["frangi", "jerman", "mfat"] = "frangi",
    ) -> None:
        """Compare les FP entre plusieurs opérateurs pour un même patient."""
        self._apply_enhancer_choice(experiment_config, enhancer)
        self._filter_grid_for_enhancer(benchmark_config, enhancer)

        raw_file, gt_file, mask_file = self._get_patient_files(
            patient_id=patient_id,
            images_dir=benchmark_config.images_dir,
            labels_dir=benchmark_config.labels_dir,
            masks_dir=getattr(benchmark_config, "masks_dir", None),
        )
        if raw_file is None:
            self.logger.error(f"[COMPARE OP] Patient {patient_id} introuvable.")
            return

        exp_config = deepcopy(experiment_config)
        exp_config.loading.raw_file = raw_file
        exp_config.loading.gt_file = gt_file
        exp_config.loading.mask_file = mask_file

        benchmark = self._get_benchmark(benchmark_config)
        self._run_fp_comparison(
            experiment_config=exp_config,
            benchmark=benchmark,
            patient_id=patient_id,
            operators=operators,
            output_dir=output_dir,
        )

    # RAPPORT COMPLET

    @log_section("Full Report Generation")
    def generate_full_report(
        self,
        experiment_config: ExperimentConfig,
        benchmark_config: BenchmarkConfig,
        images_dir: str,
        labels_dir: str,
        masks_dir: Optional[str] = None,
        n_patients: int = 20,
        patient_ids: Optional[Union[str, int, list]] = None,
        max_patients: Optional[int] = None,
        output_dir: Optional[str] = None,
        enhancer: Literal["frangi", "jerman", "mfat"] = "frangi",
    ) -> None:
        actual_n_patients = max_patients if max_patients is not None else n_patients
        results_path = self.run(
            images_dir=images_dir,
            labels_dir=labels_dir,
            benchmark_config=benchmark_config,
            experiment_config=experiment_config,
            masks_dir=masks_dir,
            n_patients=actual_n_patients,
            enhancer=enhancer,
        )
        should_run_batches = (
            bool(patient_ids) if patient_ids is not None else actual_n_patients > 0
        )
        if should_run_batches:
            self.batch_fp_analysis(
                experiment_config=experiment_config,
                benchmark_config=benchmark_config,
                images_dir=images_dir,
                labels_dir=labels_dir,
                masks_dir=masks_dir,
                n_patients=actual_n_patients,
                output_dir=output_dir,
                enhancer=enhancer,
            )
            self.batch_bifurcation_analysis(
                experiment_config=experiment_config,
                benchmark_config=benchmark_config,
                images_dir=images_dir,
                labels_dir=labels_dir,
                masks_dir=masks_dir,
                n_patients=actual_n_patients,
                output_dir=output_dir,
                enhancer=enhancer,
            )

        self.logger.info("\n"+ "="* 80)
        self.logger.info("RAPPORT COMPLET GÉNÉRÉ")
        self.logger.info(f"Résultats dans : {results_path if results_path else 'outputs/'}")
        self.logger.info("="* 80)

    # ANALYSE DES RÉSULTATS

    @log_section("Runner analysis")
    def analyse(self, benchmark_config: BenchmarkConfig, results_dir: str):
        results = [
            self.loader.load_results(os.path.join(results_dir, f))
            for f in os.listdir(results_dir)
            if os.path.isfile(os.path.join(results_dir, f))
        ]

        match benchmark_config.mode:
            case "hessian":
                figures = self.analytics.get_hessian_figures(results, benchmark_config.params)
            case "enhancement":
                figures = self.analytics.get_enhancement_figures(results, benchmark_config.params)
            case _:
                raise ValueError(f"Mode inconnu : {benchmark_config.mode}")

        self._save_figures(figures)