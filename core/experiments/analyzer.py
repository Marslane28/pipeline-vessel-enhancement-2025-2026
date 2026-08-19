import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tabulate import tabulate
from math import ceil
from pathlib import Path
from typing import Optional

from core.utils.helpers import compute_time
from core.utils.decorator import log_time, log_section, log_init
from core.utils.viewer import Viewer
from core.utils.patient_loader import PatientLoader
from core.io.logger import setup_logger
from core.io.loader import Loader
from core.io.saver import Saver
from core.config.setup import SetupConfig
from core.config.experiment import ProcessingConfig, EnhancementConfig, ExperimentConfig
from core.config.benchmark import BenchmarkConfig
from core.processing.enhancer import Enhancer
from core.processing.derivator import Derivator
from core.processing.processor import Processor
from core.experiments.metrics import detailed_metrics
from core.experiments.benchmarks.hessian import BenchmarkHessian

logger = setup_logger(name="benchmark", debug_mode=True)


class Analyzer:

    @log_init()
    def __init__(self, setup: SetupConfig):
        self.save_mode = setup.save_mode
        self.plot_mode = setup.plot_mode

        self.logger = setup_logger(log_file=setup.log_file, debug_mode=setup.debug_mode)
        self.loader  = Loader(setup.input_dir, self.logger)
        self.viewer  = Viewer()
        self.saver   = Saver(
            experiment_name=setup.name,
            output_dir=setup.output_dir,
            logger=self.logger,
        )
        self.patient_loader = PatientLoader(input_dir=setup.input_dir, logger=self.logger)

-    # Métriques sur N patients

    @log_time()
    @log_section("Patient metrics analysis")
    def run_on_patients(
        self,
        experiment_config: ExperimentConfig,
        images_dir: str = "images",
        labels_dir: str = "labels",
        n_patients: int = 20,
    ):
        """Charge, traite et évalue toutes les métriques (Dice, MCC, ROC, PR, clDice, composantes connexes)."""
        raw_files, gt_files = self.patient_loader.get_files(images_dir, labels_dir, n_patients)
        n_total = len(raw_files)

        processor = Processor(experiment_config.processing)
        all_results = []

        metrics_to_collect = [
            "dice", "mcc", "roc", "pr", "cldice","components_ratio", "excess_components", "largest_ratio","skeleton_component_connectivity", "largest_component_recall","gt_fragmentation", "pred_small_components", "pred_medium_components","pred_large_components", "gt_small_components", "gt_medium_components","gt_large_components", "precision", "sensitivity", "specificity","accuracy", "n_components_pred", "n_components_gt"
        ]
        all_metrics = detailed_metrics(data_segmented, data_gt, threshold=threshold)

        # Configuration : True = plus haut = meilleur, False = plus bas = meilleur
        metric_config = {
            "dice": True, "mcc": True, "roc": True, "pr": True, "cldice": True,
            "largest_ratio": True, "skeleton_component_connectivity": True,
            "largest_component_recall": True, "precision": True, "sensitivity": True,
            "specificity": True, "accuracy": True, "pred_large_components": True,
            "components_ratio": False, "excess_components": False,
            "pred_small_components": False,
            "pred_medium_components": False,
        }

        for idx, (raw_file, gt_file) in enumerate(zip(raw_files, gt_files), start=1):
            patient_id = PatientLoader.patient_id_from_path(raw_file)
            self.logger.info(f"[ANALYZER] Patient {idx}/{n_total} : {patient_id}")

            load_kwargs = dict(
                normalize=experiment_config.loading.normalize,
                crop=experiment_config.loading.crop,
                target_shape=experiment_config.loading.target_shape,
            )
            data_raw = self.loader.load_data(filename=raw_file, **load_kwargs)
            data_gt  = self.loader.load_data(filename=gt_file,  **load_kwargs)

            exp_config = deepcopy(experiment_config)
            exp_config.loading.raw_file = raw_file
            exp_config.loading.gt_file  = gt_file

            data_enhanced, data_segmented, threshold = processor.process_data(
                data=data_raw,
                hessian_config=exp_config.hessian,
                enhancement_config=exp_config.enhancement,
                segmentation_config=exp_config.segmentation,
                methods=exp_config.methods,
                ground_truth=data_gt,
            )

            all_metrics = detailed_metrics(data_segmented, data_gt, threshold=threshold)
            all_metrics['conn_metrics'] = {
                k: all_metrics[k] for k in ['skeleton_component_connectivity', 'largest_component_recall', 'gt_fragmentation', 'pred_small_components', 'pred_medium_components', 'pred_large_components', 'gt_small_components', 'gt_medium_components', 'gt_large_components']
            }
            result = {"patient": patient_id}
            for m in metrics_to_collect:
                result[m] = all_metrics.get(m, 0)
            all_results.append(result)

            self.logger.info(
                f"  Dice={all_metrics['dice']:.4f} | MCC={all_metrics['mcc']:.4f} | "
                f"ROC={all_metrics['roc']:.4f} | PR={all_metrics['pr']:.4f} | "
                f"clDice={all_metrics['cldice']:.4f} | FragRatio={all_metrics['components_ratio']:.2f}"
            )

            if self.save_mode:
                self.saver.save_data(data_enhanced,  "data_enhanced",  dirname=patient_id)
                self.saver.save_data(data_segmented, "data_segmented", dirname=patient_id)

        # --- Tableau avec ligne BEST ---
        table_rows = []
        for r in all_results:
            row = [r["patient"]]
            for m in metrics_to_collect:
                val = r.get(m, 0)
                row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            table_rows.append(row)

        best_row = ["BEST"]
        for m in metrics_to_collect:
            if m not in metric_config:
                best_row.append("-")
                continue
            values = [(r[m], r["patient"]) for r in all_results if m in r]
            if not values:
                best_row.append("-")
                continue
            best_val, best_patient = (
                max(values, key=lambda x: x[0]) if metric_config[m]
                else min(values, key=lambda x: x[0])
            )
            best_row.append(f"{best_val:.4f} ({best_patient})")
        table_rows.append(best_row)

        headers = ["Patient"] + [m.replace("_", " ").title() for m in metrics_to_collect]
        table   = tabulate(table_rows, headers=headers, tablefmt="github", floatfmt=".4f")

        summary = (
            f"\n{'='*100}\n"
            f"  RÉSUMÉ - {n_total} patient(s)\n"
            f"{'='*100}\n"
            f"{table}\n\n"
            f"  {'='*40} STATISTIQUES GLOBALES {'='*40}\n"
        )
        for m in metrics_to_collect:
            values = [r[m] for r in all_results if m in r]
            if values:
                summary += (
                    f"  {m.replace('_', ' ').title():30} : "
                    f"{np.mean(values):.4f} ± {np.std(values):.4f}\n"
                )
        summary += f"{'='*100}"

        self.logger.info(summary)
        if self.save_mode:
            self.saver.save_text(summary, "metrics_summary_patients")

        # --- Bar charts ---
        plot_metrics = ["dice", "mcc", "cldice", "roc", "pr", "components_ratio"]
        labels = [r["patient"] for r in all_results]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, metric in zip(axes.flatten(), plot_metrics):
            scores = [r.get(metric, 0) for r in all_results]
            bars = ax.bar(labels, scores, color="#407ff5")

            best_idx = (
                np.argmax(scores) if metric_config.get(metric, True) else np.argmin(scores)
            )
            bars[best_idx].set_color("green")
            bars[best_idx].set_edgecolor("darkgreen")
            bars[best_idx].set_linewidth(2)

            ax.axhline(np.mean(scores), color="red", linestyle="--",
                       label=f"Moy = {np.mean(scores):.3f}")
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Patient")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            ax.grid(True, axis="y")

        for ax in axes.flatten()[len(plot_metrics):]:
            ax.set_visible(False)

        plt.tight_layout()
        if self.plot_mode:
            plt.show()
        if self.save_mode:
            self.saver.save_plot(fig, "metrics_per_patient_all")

        return all_results

    # Analyses techniques

    @log_time()
    @log_section("Chunk analysis")
    def chunk_analysis(
        self,
        volume_sizes: list = [64, 128, 256, 512],
        chunk_ratios: np.ndarray = 100 ** np.linspace(0.62, 0.88, 40),
    ):
        enhancer  = Enhancer(use_gpu=False)
        derivator = Derivator(use_gpu=False)
        processor = Processor(ProcessingConfig(
            use_gpu=False, normalize=True, parallelize=True,
            show_progress=True, overlap_size=10, chunk_size=None,
        ))
        enh_params = EnhancementConfig(
            scales=[2, 4, 6, 8, 10], alpha=0.5, beta=0.5, skimage=False,
            hessian_function=derivator.farid,
            hessian_params={"mode": "reflect", "cval": 0.0, "sigma": 1.0},
        ).to_dict()

        all_times = []
        for vs in volume_sizes:
            vol = np.ones((vs, vs, vs), np.float32)
            times = []
            for ratio in chunk_ratios:
                s = int(vs * ratio / 100)
                processor.chunk_size = (s, s, s)
                times.append(compute_time(
                    processor.enhance_data,
                    data=vol,
                    enhancement_function=enhancer.frangi,
                    enhancement_params=enh_params,
                ))
            all_times.append(times)

        colors = ["#175ddfff", "#407ff5ff", "#6097fcff"]
        for k in range(2):
            fig = plt.figure(figsize=(10, 8))
            for i, vs in enumerate(volume_sizes):
                plt.plot(chunk_ratios, all_times[i], "-+",
                         color=colors[i % len(colors)], label=f"volume={vs}")
            plt.xlim([15, 62])
            plt.title("Chunks vs temps")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if self.plot_mode:
                plt.show()
            if self.save_mode:
                self.saver.save_plot(fig, f"chunk_analysis_{k + 1}")

    @log_time()
    @log_section("Parallelization check")
    def para_check(self, input_file: str):
        vol = self.loader.load_data(input_file)
        s   = vol.shape
        ck  = (ceil(s[0] / 2), ceil(s[1] / 2), ceil(s[2] / 2))

        enh   = Enhancer(use_gpu=False)
        deriv = Derivator(use_gpu=False)
        proc  = Processor(ProcessingConfig(
            use_gpu=False, normalize=True, parallelize=True,
            show_progress=True, overlap_size=10, chunk_size=ck,
        ))
        ep = EnhancementConfig(
            scales=[2, 4, 6, 8, 10], alpha=0.5, beta=0.5, gamma=20,
            skimage=False, hessian_function=deriv.farid,
            hessian_params={"mode": "reflect", "cval": 0.0, "sigma": 1.0},
        ).to_dict()

        proc.parallelize = False
        seq = proc.enhance_data(vol, enh.frangi, ep)
        proc.parallelize = True
        par = proc.enhance_data(vol, enh.frangi, ep)

        fig = self.viewer.display_slices([seq, par, seq - par],
                                         ["Séquentiel", "Parallèle", "Diff"])
        self.saver.save_anim(fig, "para_check")
        plt.close()
        logger.info(f"MAE: {np.abs(seq - par).mean():.4e}")

    @log_time()
    @log_section("PARA vs SEQ")
    def para_vs_seq(self, volume_sizes: list = [32, 64, 128, 256, 512]):
        enh   = Enhancer(use_gpu=False)
        deriv = Derivator(use_gpu=False)
        proc  = Processor(ProcessingConfig(
            use_gpu=False, normalize=True, parallelize=False,
            show_progress=True, overlap_size=10, chunk_size=(64, 64, 64),
        ))
        ep = EnhancementConfig(
            scales=[2, 4, 6, 8, 10], alpha=0.5, beta=0.5, skimage=False,
            hessian_function=deriv.farid,
            hessian_params={"mode": "reflect", "cval": 0.0, "sigma": 1.0},
        ).to_dict()

        t_seq, t_par = [], []
        for s in volume_sizes:
            vol = np.ones((s, s, s), np.float32)
            proc.chunk_size = (ceil(s / 2),) * 3

            proc.parallelize = False
            t_seq.append(compute_time(proc.enhance_data,
                                      data=vol,
                                      enhancement_function=enh.frangi,
                                      enhancement_params=ep))
            proc.parallelize = True
            t_par.append(compute_time(proc.enhance_data,
                                      data=vol,
                                      enhancement_function=enh.frangi,
                                      enhancement_params=ep))

        logger.info("\n" + tabulate(
            zip(volume_sizes, t_seq, t_par),
            headers=["Size", "Seq(s)", "Par(s)"],
            tablefmt="github",
            floatfmt=".3f",
        ))

        fig = plt.figure(figsize=(10, 5))
        plt.plot(volume_sizes, t_seq, "+-", label="Séquentiel", color="red")
        plt.plot(volume_sizes, t_par, "+-", label="Parallèle",  color="dodgerblue")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if self.plot_mode:
            plt.show()
        if self.save_mode:
            self.saver.save_plot(fig, "para_vs_seq")

        return volume_sizes, t_seq, t_par

    @log_time()
    @log_section("CPU vs GPU")
    def cpu_vs_gpu(self, volume_sizes: list = [16, 32, 64, 128, 256]):
        proc = Processor(ProcessingConfig(
            use_gpu=False, normalize=True, parallelize=False,
            show_progress=True, overlap_size=10, chunk_size=(64, 64, 64),
        ))
        ep = EnhancementConfig(
            scales=[2, 4, 6, 8, 10], alpha=0.5, beta=0.5, skimage=False,
            hessian_params={"mode": "reflect", "cval": 0.0, "sigma": 1.0},
        ).to_dict()

        t_cpu, t_gpu = [], []
        for s in volume_sizes:
            vol = np.ones((s, s, s), np.float32)
            proc.chunk_size = (ceil(s / 2),) * 3
            for gpu, tl in [(False, t_cpu), (True, t_gpu)]:
                e = Enhancer(use_gpu=gpu)
                d = Derivator(use_gpu=gpu)
                ep["hessian_function"] = d.farid
                proc.use_gpu = gpu
                tl.append(compute_time(proc.enhance_data, vol, e.frangi, ep))

        fig = plt.figure(figsize=(10, 5))
        plt.plot(volume_sizes, t_cpu, "+-", label="CPU", color="red")
        plt.plot(volume_sizes, t_gpu, "+-", label="GPU", color="dodgerblue")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if self.plot_mode:
            plt.show()
        if self.save_mode:
            self.saver.save_plot(fig, "cpu_vs_gpu")

    # False Positive Analysis  délégation à BenchmarkHessian

    def _make_benchmark(self, benchmark_config: BenchmarkConfig) -> BenchmarkHessian:
        """Construit un BenchmarkHessian avec les paramètres courants de l'Analyzer."""
        return BenchmarkHessian(
            save_mode=self.save_mode,
            plot_mode=self.plot_mode,
            logger=self.logger,
            loader=self.loader,
            saver=self.saver,
            params_grid=getattr(benchmark_config, "params_grid", {}),
        )

    @log_time()
    @log_section("False Positive Analysis")
    def analyze_false_positives(
        self,
        experiment_config: ExperimentConfig,
        benchmark_config: BenchmarkConfig,
        patient_id: str,
        derivator_names: list[str] = None,
        n_slices: int = 9,
        output_dir: str = "fp_analysis",
    ) -> dict:
        """Analyse détaillée des FP pour un patient, pour un ou plusieurs opérateurs."""
        original_params = None
        if derivator_names is not None:
            original_params = deepcopy(benchmark_config.params)
            benchmark_config.params = {"derivator": derivator_names}

        raw_file = gt_file = mask_file = None
        images_path = Path(benchmark_config.images_dir)
        labels_path = Path(benchmark_config.labels_dir)
        masks_dir   = getattr(benchmark_config, "masks_dir", None)

        if images_path.exists():
            for f in sorted(images_path.glob(f"{patient_id}*.nii*")):
                raw_file = str(f)
                break
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

        if raw_file is None:
            self.logger.error(f"[ANALYZER FP] Patient {patient_id} introuvable")
            return {}

        exp_config = deepcopy(experiment_config)
        exp_config.loading.raw_file  = raw_file
        exp_config.loading.gt_file   = gt_file
        exp_config.loading.mask_file = mask_file

        benchmark  = self._make_benchmark(benchmark_config)
        fp_results = benchmark.analyze_false_positives(
            experiment_config=exp_config,
            benchmark_config=benchmark_config,
            patient_id=patient_id,
            output_dir=output_dir,
        )

        if original_params is not None:
            benchmark_config.params = original_params

        return fp_results

    # Operator summary figure

    def generate_operator_summary_figure(
        self,
        experiment_config: ExperimentConfig,
        patient_id: str,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Génère une figure récapitulative pour un patient donné :
        pour chaque opérateur, une figure avec overlay 3 vues + métriques.
        """
        benchmark_config = BenchmarkConfig(
            mode="hessian",
            images_dir="data/3d-échantillonnées/images",
            labels_dir="data/3d-échantillonnées/labels",
            masks_dir="data/3d-échantillonnées/masks",
            params={"derivator": [
                "default", "gaussian", "farid", "cubic",
                "trigonometric", "catmull", "bspline", "bezier",
            ]},
        )

        benchmark = BenchmarkHessian(
            save_mode=self.save_mode,
            plot_mode=self.plot_mode,
            logger=self.logger,
            loader=self.loader,
            saver=self.saver,
            params_grid={},
        )

        raw_file, gt_file, mask_file = self._get_patient_files(
            patient_id=patient_id,
            images_dir=benchmark_config.images_dir,
            labels_dir=benchmark_config.labels_dir,
            masks_dir=benchmark_config.masks_dir,
        )

        if raw_file is None:
            self.logger.error(f"Patient {patient_id} introuvable")
            return

        exp_config = deepcopy(experiment_config)
        exp_config.loading.raw_file  = raw_file
        exp_config.loading.gt_file   = gt_file
        exp_config.loading.mask_file = mask_file

        data_raw, data_gt, data_mask = self.loader.load_data(exp_config.loading)

        for op_name in benchmark_config.params["derivator"]:
            try:
                deriv   = Derivator(use_gpu=False)
                op_func = getattr(deriv, op_name)

                exp_op = deepcopy(experiment_config)
                exp_op.methods.derivator = op_func

                processor = Processor(exp_op.processing)
                _, data_segmented, threshold = processor.process_data(
                    data=data_raw,
                    ground_truth=data_gt,
                    hessian_config=exp_op.hessian,
                    enhancement_config=exp_op.enhancement,
                    segmentation_config=exp_op.segmentation,
                    methods=exp_op.methods,
                    mask_liver=data_mask,
                )

                cldice_val   = benchmark._compute_cldice_metrics(
                    data_segmented, data_gt, data_mask, threshold)
                conn_metrics = benchmark._compute_connectivity_metrics(
                    data_segmented, data_gt, data_mask, threshold)

                benchmark.visualize_operator_summary(
                    data_raw=data_raw,
                    data_gt=data_gt,
                    data_segmented=data_segmented,
                    patient_id=patient_id,
                    derivator_name=op_name,
                    threshold=threshold,
                    cldice_score=cldice_val,
                    conn_metrics=conn_metrics,
                    data_mask=data_mask,
                    output_dir=output_dir,
                )

            except Exception as e:
                self.logger.warning(f"Erreur pour {op_name}: {e}")
                continue

    def _get_patient_files(
        self,
        patient_id: str,
        images_dir: str,
        labels_dir: str,
        masks_dir: str = None,
    ):
        raw_file = gt_file = mask_file = None

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

    # Analyzer run

    @log_time()
    @log_section("Analyzer run")
    def run(self):
        self.cpu_vs_gpu(volume_sizes=[50, 60])