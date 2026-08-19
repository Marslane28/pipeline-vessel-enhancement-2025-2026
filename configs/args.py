import argparse

LOG_DIR = "logs"
INPUT_DIR = "data"
OUTPUT_DIR = "outputs"

DATASET_N_PATIENTS = {
    "ircad": 20,
    "bullitt": 33,
    "vascusynth": 30,
}


def get_parser():
    parser = argparse.ArgumentParser(description="Runner pipeline")
    parser.add_argument('--run_pipeline', action='store_true', help='Run the pipeline')
    parser.add_argument('--run_analyzer', action='store_true', help='Run the analyzer')
    parser.add_argument('--run_benchmark', action='store_true', help='Run the benchmark')
    parser.add_argument('--benchmark_type', choices=['hessian'], default='hessian', help='Type of benchmark')
    parser.add_argument('--test', action='store_true', help='Use test config')
    parser.add_argument('--max_patients', type=int, default=None,
                        help='Nombre maximum de patients à traiter (pour test)')
    parser.add_argument("--n_patients", "-n", type=int, default=None,
                        help="Nombre de patients à traiter (défaut: tous)")
    parser.add_argument("--dataset", "-d", choices=["ircad", "bullitt", "sennet", "vascusynth"], default=None,
                        help="Dataset à utiliser (auto-détection si non spécifié)")
    parser.add_argument('--enhancer', choices=['frangi', 'jerman', 'mfat'],
                        default='frangi',
                        help='Filtre de rehaussement à utiliser (frangi ou jerman ou mfat)')
    parser.add_argument("--run_preprocessing", action="store_true")
    parser.add_argument("--preprocess_dataset", default="all",
                        choices=["ircad", "bullitt", "vascusynth", "both", "all"])
    parser.add_argument("--preprocess_config", default=None)
    parser.add_argument('--run_postbench', action='store_true',
                        help="Lance l'étude post-benchmark (filtrage, vesselness/segmentation, 3D) "
                             "après --run_benchmark")
    parser.add_argument('--postbench_config', type=str, default=None,
                        help="Chemin YAML explicite pour la config post-benchmark. "
                             "Par défaut : configs/postbench/{dataset}.yaml (ou study.yaml si dataset non précisé)")
    parser.add_argument('--run_post_tests', action='store_true',
                        help="Lance les tests statistiques post-benchmark fusionnés "
                             "(post_benchmark_tests.py) après --run_benchmark : filtrage, "
                             "proximité au seuil, médiation, distribution des seuils, "
                             "différences inter-opérateurs, sweep de seuil.")
    parser.add_argument('--post_tests_config_dir', type=str, default=None,
                        help="Dossier contenant les YAML par dataset pour --run_post_tests. "
                             "Par défaut : configs/postbench_tests/ (un fichier par dataset : "
                             "bullitt.yaml, ircad.yaml, vascusynth.yaml)")
    parser.add_argument('--skip_sweep', action='store_true',
                        help="Ne pas lancer le sweep de seuil (coûteux) dans --run_post_tests.")
    parser.add_argument('--post_tests_output', type=str, default=None,
                        help="Chemin du rapport texte à sauvegarder pour --run_post_tests "
                             "(par défaut : affiché seulement, non sauvegardé).")
    parser.add_argument('--debug', action='store_true',
                        help="Active les impressions de debug (utilisé par --run_post_tests).")
    parser.add_argument('--run_operator_study', action='store_true',
                        help="Lance une étude analytique d'opérateurs sur fantôme "
                             "AnalyticalVessel (core/Hessian_evaluation/Comparaison_*.py).")
    parser.add_argument('--study_type', type=str, default=None,
                        choices=['hessian', 'eigenvalues', 'vesselness', 'segmentation'],
                        help="Type d'étude à lancer avec --run_operator_study : "
                             "hessian (erreur Hessienne brute), eigenvalues (valeurs propres), "
                             "vesselness (Frangi/Jerman), segmentation (Segmenter + detailed_metrics).")
    parser.add_argument('--operator_study_config', type=str, default=None,
                        help="Chemin YAML explicite pour --run_operator_study. "
                             "Par défaut : configs/operator_study/{study_type}.yaml")
    parser.add_argument('--run_cc_filtering', action='store_true',
                    help="Recalcule les métriques après filtrage par taille de "
                         "composante connexe (min_size) pour chaque opérateur, "
                         "et génère une figure d'évolution des métriques.")
    parser.add_argument('--cc_filtering_config', type=str, default=None,
                    help="Chemin YAML explicite. Par défaut : "
                         "configs/post_benchmark_studies/cc_filtering/{dataset}.yaml")

    return parser