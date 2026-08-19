import argparse
import sys
from pathlib import Path
from .data_config.reglage_data.data_config import load_config
from .dataset.logger import setup_logger
from .dataset.dicom_loader import load_dataset
from .dataset.patient_classifier import classify_dataset
from .dataset.dataset_report import generate_full_report
from .dataset.data_visualisation.export_labels import save_vessel_masks_as_nifti, save_slice_overviews

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline IRCAD - Étape 1 : Acquisition et structuration des données"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Chemin vers config.yaml (défaut : configuration/data.yaml)",
    )
    parser.add_argument(
        "--patients",
        type=int,
        nargs="*",
        default=None,
        help="IDs des patients à traiter (ex. --patients 1 3 8 12). Défaut = tous (1–20).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Désactive la génération du rapport et des figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Configuration ─────────────────────────────────────────────────────────
    config = load_config(args.config)
    out_cfg = config["output"]
    debug_mode = out_cfg.get("log_level", "INFO") == "DEBUG"

    log = setup_logger(
    log_file=out_cfg.get("log_file", "pipeline"),
    debug_mode=debug_mode
    )

    log.info("***Pipeline IRCAD 3D-IRCADb-01 - Étape 1***")

    # ── Chargement du dataset ─────────────────────────────────────────────────
    try:
        dataset = load_dataset(config, patient_ids=args.patients)
    except FileNotFoundError as exc:
        log.error("ERREUR CRITIQUE : %s", exc)
        sys.exit(1)

    if not dataset:
        log.error("Aucun patient chargé. Vérifiez le répertoire du dataset.")
        sys.exit(1)
    log.info("%d patient(s) chargé(s) avec succès.", len(dataset))

    # ── Export des masques ────────────────────────────────────────────────────
    output_labels_dir = Path(config["output"]["processed_dir"]) / "vessel_labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    for pv in dataset.values():
        save_vessel_masks_as_nifti(pv, output_labels_dir)
        save_slice_overviews(pv, output_labels_dir, num_slices=5)
    log.info("Masques vasculaires exportés au format NIfTI et vues générées pour chaque patient.")
    # ── Classification ────────────────────────────────────────────────────────
    log.info("── Classification automatique des patients ──")
    classifications = classify_dataset(dataset, config)

    # ── Rapport ───────────────────────────────────────────────────────────────
    if not args.no_report:
        log.info("── Génération du rapport ──")
        report_dir = Path(out_cfg.get("reports_dir", "outputs_Etape1/reports_Etape1"))
        df = generate_full_report(dataset, classifications, report_dir)

        # Affiche le sous-ensemble de patients par catégorie
        log.info("\n Patients recommandés pour l'analyse de robustesse :")
        for cat in ["Facile", "Artefact métallique", "Faible contraste", "Contact organique"]:
            subset = df[df["label"].str.contains(cat, na=False)]["patient_id"].tolist()
            if subset:
                log.info("  %-25s → patients %s", cat, subset)

    log.info("Étape 1 terminée avec succès.")


if __name__ == "__main__":
    main()