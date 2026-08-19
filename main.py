from pathlib import Path

from core.experiments.benchmarks.runner import BenchmarkRunner
from core.config.builder import ConfigBuilder
from core.config.setup import SetupConfig
from core.config.benchmark import RunnerConfig, BenchmarkConfig
from core.config.experiment import ExperimentConfig
from core.config.cc_filtering import CCFilteringConfig
from core.config.operator_study import (
    HessianStudyConfig,
    EigenvalueStudyConfig,
    VesselnessStudyConfig,
    SegmentationStudyConfig,
)
from core.experiments.post_benchmark.cc_filtering.cc_filtering_study import run_cc_filtering_study
from configs.args import get_parser, DATASET_N_PATIENTS
from Manipulation_des_donnees.core_pre_traitement.multi_dataset_preprocessor import MultiDatasetPreprocessor



parser = get_parser()
args = parser.parse_args()

run_pipeline        = args.run_pipeline
run_benchmark        = args.run_benchmark
run_postbench        = args.run_postbench
run_preprocessing    = args.run_preprocessing
run_operator_study   = args.run_operator_study
run_post_tests       = args.run_post_tests
run_cc_filtering     = args.run_cc_filtering
benchmark_type        = args.benchmark_type
study_type            = args.study_type
test                  = args.test
max_patients          = args.max_patients

ROOT = "tests/" if test else ""

_OPERATOR_STUDY_CONFIG_TYPES = {
    "hessian":      HessianStudyConfig,
    "eigenvalues":  EigenvalueStudyConfig,
    "vesselness":   VesselnessStudyConfig,
    "segmentation": SegmentationStudyConfig,
}

SRC_PIPELINE_SETUP      = ROOT + "configs/pipeline/setup.yaml"
SRC_PIPELINE_EXPERIMENT = ROOT + "configs/pipeline/experiment.yaml"
SRC_BENCHMARK_S3D       = ROOT + "configs/benchmark/s3d.yaml"

SRC_BENCHMARK_RUNNER      = ROOT + "configs/benchmark/runner.yaml"
SRC_BENCHMARK_HESSIAN     = ROOT + "configs/benchmark/hessian.yaml"
SRC_BENCHMARK_EXPERIMENT  = ROOT + "configs/benchmark/experiment.yaml"


def _resolve_preprocess_config_path():
    if args.preprocess_config:
        return args.preprocess_config
    return ROOT + "configs/preprocessing/dataset_config.yaml"


def _resolve_postbench_config_path():
    if args.postbench_config:
        return args.postbench_config
    dataset_key = args.dataset or "ircad"
    return ROOT + f"configs/post_benchmark_studies/datasets/{dataset_key}.yaml"


def _resolve_post_tests_config_dir():
    return args.post_tests_config_dir or (ROOT + "configs/post_benchmark_studies/datasets/")


def _resolve_cc_filtering_config_path():
    if args.cc_filtering_config:
        return args.cc_filtering_config
    dataset_key = args.dataset or "ircad"
    return ROOT + f"configs/post_benchmark_studies/cc_filtering/{dataset_key}.yaml"


def _resolve_operator_study_config_path():
    if args.operator_study_config:
        return args.operator_study_config
    if study_type not in _OPERATOR_STUDY_CONFIG_TYPES:
        raise ValueError(
            f"study_type inconnu : {study_type!r} "
            f"(attendu : {sorted(_OPERATOR_STUDY_CONFIG_TYPES)})"
        )
    return ROOT + f"configs/operator_study/{study_type}.yaml"


def _load_operator_study_module(study_type: str):
    """Import à la demande du module Comparaison_*.py correspondant."""
    if study_type == "hessian":
        from core.experiments.Hessian_evaluation import Comparaison_hessian as mod
    elif study_type == "eigenvalues":
        from core.experiments.Hessian_evaluation import Comparaison_valeurs_propres as mod
    elif study_type == "vesselness":
        from core.experiments.Hessian_evaluation import Comparaison_vesselness as mod
    elif study_type == "segmentation":
        from core.experiments.Hessian_evaluation import Comparaison_segmentation as mod
    else:
        raise ValueError(f"study_type inconnu : {study_type!r}")
    return mod


def main():
    # filtrage CC 
    if run_cc_filtering:
        cc_yaml = _resolve_cc_filtering_config_path()
        print(f"[MAIN] CC filtering config : {cc_yaml}")
        cc_cfg: CCFilteringConfig = ConfigBuilder(cc_yaml, CCFilteringConfig)
        cc_cfg.dataset = args.dataset or cc_cfg.dataset
        run_cc_filtering_study(cc_cfg)

    if run_preprocessing:
        preprocess_yaml = _resolve_preprocess_config_path()
        print(f"[MAIN] Prétraitement config : {preprocess_yaml}")
        pipeline = MultiDatasetPreprocessor(preprocess_yaml)
        pipeline.run(args.preprocess_dataset)
        print("\n Prétraitement terminé !")

    if run_benchmark:
        runner_config: RunnerConfig = ConfigBuilder(SRC_BENCHMARK_RUNNER, RunnerConfig)
        experiment_config = ConfigBuilder(SRC_BENCHMARK_EXPERIMENT, ExperimentConfig)

        if benchmark_type == "hessian":
            benchmark_config: BenchmarkConfig = ConfigBuilder(SRC_BENCHMARK_HESSIAN, BenchmarkConfig)
        else:
            raise ValueError(f"Benchmark type inconnu : {benchmark_type}")

        runner = BenchmarkRunner(runner_config.setup, dataset=args.dataset)

        patient_ids = getattr(runner_config, 'patient_ids', None)
        n_patients = args.n_patients if args.n_patients is not None else getattr(runner_config, 'n_patients', None)

        if patient_ids is None:
            dataset_key = args.dataset or "ircad"
            n_patients = DATASET_N_PATIENTS.get(dataset_key, 20)
            n_patients_to_use = n_patients
            patient_ids_to_use = None
            print(f"[MAIN] patient_ids non spécifié -> chargement de TOUS les patients du dataset {dataset_key} ({n_patients} patients)")
        else:
            n_patients_to_use = None
            patient_ids_to_use = patient_ids
            print(f"[MAIN] Utilisation des patients spécifiés : {patient_ids}")

        dirname = runner.run(
            images_dir=runner_config.images_dir,
            labels_dir=runner_config.labels_dir,
            masks_dir=runner_config.masks_dir,
            benchmark_config=benchmark_config,
            experiment_config=experiment_config,
            n_patients=n_patients_to_use,
            patient_ids=patient_ids_to_use,
            enhancer=args.enhancer,
        )

        print(f"\n Benchmark terminé ! Résultats dans : {dirname}")


    
    # tests statistiques post-benchmark fusionnés 
    if run_post_tests:
        from core.experiments.post_benchmark.tests.post_benchmark_tests import (
            run_post_benchmark_tests,
            load_datasets,
            DATASETS,
        )

        post_tests_config_dir = _resolve_post_tests_config_dir()
        DATASETS.clear()
        DATASETS.update(load_datasets(post_tests_config_dir))

        print(f"[MAIN] Post-tests : config dir : {post_tests_config_dir}")
        print(f"[MAIN] Post-tests : datasets : {list(DATASETS.keys())}")
        run_post_benchmark_tests(
            dataset=args.dataset,
            output=args.post_tests_output,
            skip_sweep=args.skip_sweep,
            debug=args.debug,
        )

    if run_operator_study:
        study_yaml = _resolve_operator_study_config_path()
        config_type = _OPERATOR_STUDY_CONFIG_TYPES[study_type]
        study_config = ConfigBuilder(study_yaml, config_type)
        study_module = _load_operator_study_module(study_type)

        print(f"[MAIN] Étude opérateurs '{study_type}' config : {study_yaml}")
        df, out_dir = study_module.run_study(study_config)
        print(f"\n Étude '{study_type}' terminée ! Résultats dans : {out_dir}")


if __name__ == "__main__":
    main()