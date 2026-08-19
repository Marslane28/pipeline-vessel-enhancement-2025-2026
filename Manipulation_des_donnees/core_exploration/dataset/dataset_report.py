from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .dicom_loader import PatientVolume
from .patient_classifier import ClassificationResult, DifficultyFlag
from core.io.logger import setup_logger, close_logger
from logging import Logger

log = setup_logger(__name__)

# Palette de couleurs pour les catégories
CATEGORY_COLORS = {
    "Facile": "#4CAF50", # vert
    "Artefact métallique": "#F44336", # rouge
    "Faible contraste": "#FF9800", # orange
    "Contact organique": "#2196F3", # bleu
    "Non classifié": "#9E9E9E", # gris
}


# Construction du DataFrame de synthèse


def build_summary_dataframe(
    dataset: Dict[int, PatientVolume],
    classifications: Dict[int, ClassificationResult],
) -> pd.DataFrame:
    """
    Construit un DataFrame récapitulatif de tous les patients.
    Ajoute des métriques sur les vaisseaux.
    """
    rows = []
    for pid in sorted(dataset.keys()):
        pv = dataset[pid]
        res = classifications.get(pid)

        shape_str = str(pv.shape) if pv.shape else "N/A"
        sp = pv.spacing

        # Métriques des vaisseaux
        vessel_volume_ml = 0
        vessel_voxels = 0
        liver_volume_ml = 0
        vessel_density = 0
        
        if pv.vessel_mask is not None:
            vessel_voxels = int(pv.vessel_mask.sum())
            # Calcul du volume en mL (mm³ → mL)
            voxel_volume_ml = (sp[0] * sp[1] * sp[2]) / 1000
            vessel_volume_ml = vessel_voxels * voxel_volume_ml
        
        if pv.liver_mask is not None:
            liver_voxels = int(pv.liver_mask.sum())
            voxel_volume_ml = (sp[0] * sp[1] * sp[2]) / 1000
            liver_volume_ml = liver_voxels * voxel_volume_ml
            if liver_voxels > 0:
                vessel_density = vessel_voxels / liver_voxels * 100

        row = {
            "patient_id": pid,
            "shape": shape_str,
            "spacing_z_mm": round(sp[0], 3),
            "spacing_y_mm": round(sp[1], 3),
            "spacing_x_mm": round(sp[2], 3),
            "has_liver": pv.has_liver,
            "has_vessels": pv.has_vessels,
            "vessel_sources": "|".join(pv.vessel_mask_sources),
            "load_errors": "|".join(pv.load_errors),
            "vessel_voxels": vessel_voxels,
            "vessel_volume_ml": round(vessel_volume_ml, 2),
            "liver_volume_ml": round(liver_volume_ml, 2),
            "vessel_density_pct": round(vessel_density, 2),
        }

        if res:
            row.update({
                "liver_mean_hu": round(res.liver_mean_hu, 1),
                "liver_std_hu": round(res.liver_std_hu, 1),
                "liver_bg_diff": round(res.liver_background_diff, 1),
                "metal_voxels": res.metal_voxel_count,
                "contact_voxels": res.contact_voxels,
                "adjacent_organs": str(res.adjacent_organ_ids),
                "difficulty_flags": str(res.difficulty),
                "label": res.label,
            })
        else:
            row.update({
                "liver_mean_hu": None, "liver_std_hu": None,
                "liver_bg_diff": None, "metal_voxels": None,
                "contact_voxels": None, "adjacent_organs": None,
                "difficulty_flags": None, "label": "N/A",
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_summary_table(df: pd.DataFrame) -> None:
    """Affiche le tableau récapitulatif dans le terminal."""
    try:
        from tabulate import tabulate
        cols = ["patient_id", "shape", "has_liver", "has_vessels",
                "vessel_volume_ml", "liver_volume_ml", "vessel_density_pct",
                "liver_mean_hu", "metal_voxels", "label"]
        
        print("\n"+ "═"* 100)
        print("RÉSUMÉ DU DATASET 3D-IRCADb-01")
        print("═"* 100)
        print(tabulate(df[cols], headers="keys", tablefmt="rounded_outline", showindex=False))
        
        # STATISTIQUES RÉCAPITULATIVES
        print("\n"+ "─"* 50)
        print("STATISTIQUES GÉNÉRALES")
        print("─"* 50)
        print(f"Nombre de patients : {len(df)}")
        print(f"Patients avec vaisseaux : {df['has_vessels'].sum()}")
        print(f"Volume foie moyen : {df['liver_volume_ml'].mean():.1f} ± {df['liver_volume_ml'].std():.1f} mL")
        print(f"Volume vaisseaux moyen : {df['vessel_volume_ml'].mean():.1f} ± {df['vessel_volume_ml'].std():.1f} mL")
        print(f"Densité vasculaire moyenne: {df['vessel_density_pct'].mean():.2f} ± {df['vessel_density_pct'].std():.2f} %")
        
        # PAR CATÉGORIE
        print("\n"+ "─"* 50)
        print("️ RÉPARTITION PAR CATÉGORIE")
        print("─"* 50)
        category_counts = df['label'].value_counts()
        for cat, count in category_counts.items():
            pct = count / len(df) * 100
            bar = "█"* int(pct / 2)
            print(f"{cat:20s}: {count:2d} patients ({pct:5.1f}%) {bar}")
            
    except ImportError:
        print(df.to_string())


# Heatmap des corrélations

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Trace une heatmap des corrélations entre métriques."""
    numeric_cols = ['liver_volume_ml', 'vessel_volume_ml', 'vessel_density_pct',
                    'liver_mean_hu', 'liver_std_hu', 'metal_voxels']
    
    existing_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]
    
    if len(existing_cols) < 2:
        log.warning("Pas assez de colonnes numériques pour la heatmap")
        return
    
    corr = df[existing_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax,
                cbar_kws={'label': 'Coefficient de corrélation'})
    ax.set_title("Matrice de corrélation des métriques du dataset\n3D-IRCADb-01", fontsize=13)
    
    fig.tight_layout()
    out_path = output_dir / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Heatmap sauvegardée : %s", out_path)


# Boxplot des volumes par catégorie

def plot_volume_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot des volumes vasculaires par catégorie de difficulté."""
    if 'label'not in df.columns or 'vessel_volume_ml'not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = [cat for cat in CATEGORY_COLORS.keys() if cat in df['label'].values]
    data = [df[df['label'] == cat]['vessel_volume_ml'].dropna() for cat in categories]
    colors = [CATEGORY_COLORS[cat] for cat in categories]
    
    bp = ax.boxplot(data, tick_labels=categories, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Volume vasculaire (mL)", fontsize=12)
    ax.set_xlabel("Catégorie de patient", fontsize=12)
    ax.set_title("Distribution des volumes vasculaires par catégorie\n3D-IRCADb-01", fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    
    fig.tight_layout()
    out_path = output_dir / "volume_boxplots_by_category.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Boxplot sauvegardé : %s", out_path)


# Figures existantes

def plot_hu_distributions(
    dataset: Dict[int, PatientVolume],
    classifications: Dict[int, ClassificationResult],
    output_dir: Path,
) -> None:
    """
    Trace les distributions HU intrahépatiques regroupées par catégorie.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    category_hu: Dict[str, List[np.ndarray]] = {cat: [] for cat in CATEGORY_COLORS}

    for pid, pv in dataset.items():
        if pv.ct_volume is None or not pv.has_liver:
            continue
        res = classifications.get(pid)
        if res is None:
            continue

        liver_hu = pv.ct_volume[pv.liver_mask == 1]

        # Catégorie principale
        if DifficultyFlag.METAL_ARTIFACT in res.difficulty:
            cat = "Artefact métallique"
        elif DifficultyFlag.ORGANIC_CONTACT in res.difficulty:
            cat = "Contact organique"
        elif DifficultyFlag.LOW_CONTRAST in res.difficulty:
            cat = "Faible contraste"
        elif DifficultyFlag.EASY in res.difficulty:
            cat = "Facile"
        else:
            cat = "Non classifié"

        category_hu[cat].append(liver_hu)

    fig, ax = plt.subplots(figsize=(12, 6))
    patches = []
    
    # Ajout des statistiques dans la légende
    for cat, samples in category_hu.items():
        if not samples:
            continue
        all_hu = np.concatenate(samples)
        color = CATEGORY_COLORS[cat]
        
        # KDE plot plus lisse
        ax.hist(all_hu, bins=80, alpha=0.55, color=color,
                density=True, label=f"{cat} (n={len(samples)})")
        
        # Ajouter la moyenne ± std
        mean_hu = np.mean(all_hu)
        std_hu = np.std(all_hu)
        ax.axvline(mean_hu, color=color, linestyle='--', alpha=0.7)
        ax.axvspan(mean_hu - std_hu, mean_hu + std_hu, alpha=0.1, color=color)
        
        patches.append(mpatches.Patch(color=color, label=f"{cat} (n={len(samples)}, μ={mean_hu:.0f}±{std_hu:.0f} HU)"))

    ax.set_xlabel("Intensité HU (dans la région hépatique)", fontsize=12)
    ax.set_ylabel("Densité de probabilité", fontsize=12)
    ax.set_title("Distributions HU intrahépatiques par catégorie de patient\n3D-IRCADb-01", fontsize=13)
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "hu_distributions_by_category.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Figure sauvegardée : %s", out_path)


def plot_patient_overview(
    pv: PatientVolume,
    output_dir: Path,
    n_slices: int = 3,
) -> None:
    """
    Visualise n_slices coupes axiales du volume CT superposées avec
    les masques foie (rouge) et vaisseaux (bleu) pour un patient.
    """
    if pv.ct_volume is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    Z = pv.ct_volume.shape[0]
    slice_indices = np.linspace(Z // 4, 3 * Z // 4, n_slices, dtype=int)

    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    for ax, idx in zip(axes, slice_indices):
        slice_ct = pv.ct_volume[idx]
        vmin = np.percentile(slice_ct, 1)
        vmax = np.percentile(slice_ct, 99)
        ax.imshow(slice_ct, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

        if pv.liver_mask is not None:
            liver_slice = pv.liver_mask[idx].astype(float)
            liver_rgb = np.zeros((*liver_slice.shape, 4))
            liver_rgb[..., 0] = 1.0 # Rouge
            liver_rgb[..., 3] = liver_slice * 0.35
            ax.imshow(liver_rgb, origin="lower")

        if pv.vessel_mask is not None:
            vessel_slice = pv.vessel_mask[idx].astype(float)
            vessel_rgb = np.zeros((*vessel_slice.shape, 4))
            vessel_rgb[..., 2] = 1.0 # Bleu
            vessel_rgb[..., 3] = vessel_slice * 0.55
            ax.imshow(vessel_rgb, origin="lower")

        # Ajouter le numéro de coupe et l'échelle
        ax.set_title(f"Coupe z={idx} (spacing={pv.spacing[0]:.2f}mm)", fontsize=10)
        ax.axis("off")
        
        # Ajouter une barre d'échelle
        scale_bar_length_px = 100 # pixels
        scale_bar_length_mm = scale_bar_length_px * pv.spacing[2] # en mm
        ax.plot([10, 10 + scale_bar_length_px], [10, 10], 'w-', linewidth=3)
        ax.text(10, 20, f"{scale_bar_length_mm:.1f} mm", color='white', fontsize=8)

    # Légende
    patches = [
        mpatches.Patch(color="red", alpha=0.5, label="Foie"),
        mpatches.Patch(color="blue", alpha=0.5, label="Vaisseaux (GT)"),
    ]
    
    # Ajout des infos patient
    fig.suptitle(f"Patient {pv.patient_id:02d} - Vue axiale\n"
                 f"Volume foie: {pv.liver_mask.sum() * pv.spacing[0] * pv.spacing[1] * pv.spacing[2] / 1000:.1f} mL | "
                 f"Vaisseaux: {pv.vessel_mask.sum() * pv.spacing[0] * pv.spacing[1] * pv.spacing[2] / 1000:.1f} mL",
                 fontsize=11)
    fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=10)
    fig.tight_layout()

    out_path = output_dir / f"patient_{pv.patient_id:02d}_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Pipeline complet de reporting

def generate_full_report(
    dataset: Dict[int, PatientVolume],
    classifications: Dict[int, ClassificationResult],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Génère le rapport complet :
      - CSV récapitulatif
      - Figures HU par catégorie
      - Heatmap de corrélation
      - Boxplot des volumes
      - Aperçus par patient

    Returns
    -------
    pd.DataFrame du récapitulatif.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # DataFrame
    df = build_summary_dataframe(dataset, classifications)
    print_summary_table(df)

    # Sauvegarde CSV
    csv_path = output_dir / "dataset_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    log.info("Récapitulatif sauvegardé : %s", csv_path)
    
    # Sauvegarde aussi en Markdown pour le rapport
    md_path = output_dir / "dataset_summary.md"
    df.to_markdown(md_path, index=False)
    log.info("Rapport Markdown sauvegardé : %s", md_path)

    # Figures globales
    fig_dir = output_dir / "figures"
    
    # Distributions HU
    plot_hu_distributions(dataset, classifications, fig_dir)
    
    # NOUVELLES FIGURES
    plot_correlation_heatmap(df, fig_dir)
    plot_volume_boxplots(df, fig_dir)

    # Aperçus individuels
    for pid, pv in dataset.items():
        plot_patient_overview(pv, fig_dir / "patients")

    # RAPPORT STATISTIQUE FINAL
    print("\n"+ "═"* 70)
    print("RAPPORT FINAL - DATASET 3D-IRCADb-01")
    print("═"* 70)
    print(f"Patients chargés : {len(dataset)}/20")
    print(f"Avec masque foie : {sum(1 for pv in dataset.values() if pv.has_liver)}")
    print(f"Avec masque vaisseaux : {sum(1 for pv in dataset.values() if pv.has_vessels)}")
    
    # Statistiques par catégorie
    print("\n ️ CLASSIFICATION DES PATIENTS")
    print(""+ "-"* 50)
    for cat in ["Facile", "Artefact métallique", "Faible contraste", "Contact organique"]:
        patients = df[df['label'].str.contains(cat, na=False)]['patient_id'].tolist()
        if patients:
            print(f"{cat:20s}: patients {patients}")
    
    log.info("Rapport complet généré dans : %s", output_dir)
    return df