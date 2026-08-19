import argparse
from vessel_preprocessor import VesselPreprocessor


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocessing du dataset de vaisseaux (B-spline isotrope)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py -i data/raw -o data/preprocessed
  python main.py -i data/raw -o data/preprocessed --spacing 0.5 0.5 0.5 --n_patients 10
        """,
    )
    parser.add_argument("--input",  "-i", required=False, default="/Users/arslenebouhadjera/Stage_Arslène_2026/Documents/StageNoé2025/pipeline-vessel-enhancement-master/Chargement_des_données/données/vessel_labels",
                        help="Dossier source (contient patient01, patient02, …) [default: /Users/arslenebouhadjera/Stage_Arslène_2026/Documents/StageNoé2025/pipeline-vessel-enhancement-master/Chargement_des_données/données/vessel_labels]")
    parser.add_argument("--output", "-o", required=False, default="données_pretraitées",
                        help="Dossier de destination [default: données_pretraitées]")
    parser.add_argument("--spacing", "-s", type=float, nargs=3,
                        default=[1.0, 1.0, 1.0], metavar=("X", "Y", "Z"),
                        help="Espacement cible en mm (défaut : 1.0 1.0 1.0)")
    parser.add_argument("--n_patients", "-n", type=int, default=20,
                        help="Nombre de patients à traiter (défaut : 20)")
    return parser


def main():
    args = get_parser().parse_args()

    preprocessor = VesselPreprocessor(
        input_dir=args.input,
        output_dir=args.output,
        target_spacing=tuple(args.spacing),
    )
    preprocessor.run(n_patients=args.n_patients)


if __name__ == "__main__":
    main()