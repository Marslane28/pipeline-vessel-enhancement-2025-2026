from __future__ import annotations
from dataclasses import dataclass, field
from enum import Flag, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import binary_dilation, label
from .dicom_loader import PatientVolume, _load_dicom_series_sitk, sitk_to_numpy
from ..data_config.reglage_data.data_config import load_config
from logging import Logger
from core.io.logger import setup_logger

log = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Flags de classification (combinables par OR)
# ─────────────────────────────────────────────────────────────────────────────

class DifficultyFlag(Flag):
    """Flags décrivant les difficultés d'un cas."""
    NONE = 0
    EASY = auto() # Foie homogène, bon contraste
    METAL_ARTIFACT = auto() # Artefacts strie métalliques
    LOW_CONTRAST = auto() # Foie peu contrasté
    ORGANIC_CONTACT = auto() # Contact avec organe adjacent


@dataclass
class ClassificationResult:
    """Résultat de la classification pour un patient."""
    patient_id: int
    difficulty: DifficultyFlag = DifficultyFlag.NONE

    # Métriques calculées
    metal_voxel_count: int = 0
    liver_mean_hu: float = 0.0
    liver_std_hu: float = 0.0
    background_mean_hu: float = 0.0
    liver_background_diff: float = 0.0
    contact_voxels: int = 0
    adjacent_organ_ids: List[int] = field(default_factory=list)

    # Informations de débogage
    notes: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Étiquette lisible."""
        parts = []
        if DifficultyFlag.EASY in self.difficulty:
            parts.append("Facile")
        if DifficultyFlag.METAL_ARTIFACT in self.difficulty:
            parts.append("Artefact métallique")
        if DifficultyFlag.LOW_CONTRAST in self.difficulty:
            parts.append("Faible contraste")
        if DifficultyFlag.ORGANIC_CONTACT in self.difficulty:
            parts.append("Contact organique")
        return "| ".join(parts) if parts else "Non classifié"

    def __str__(self) -> str:
        return (
            f"Patient {self.patient_id:02d} → [{self.label}]\n"
            f"HU foie={self.liver_mean_hu:.1f}±{self.liver_std_hu:.1f}, "
            f"fond={self.background_mean_hu:.1f}, "
            f"diff={self.liver_background_diff:.1f}\n"
            f"Voxels métal={self.metal_voxel_count}, "
            f"voxels contact={self.contact_voxels}, "
            f"organes adj.={self.adjacent_organ_ids}\n"
            f"Notes: {self.notes}"
        )


# Détecteurs individuels

def detect_metal_artifact(
    ct: np.ndarray,
    hu_threshold: float = 2500,
    min_voxel_count: int = 500,
) -> Tuple[bool, int]:
    """
    Détecte la présence d'artefacts métalliques par seuillage HU.

    Les implants métalliques génèrent des voxels à très haute intensité
    (> 2500 HU, parfois jusqu'à 3071 ou +∞ selon le scanner).

    Returns
    -------
    (detected, voxel_count)
    """
    high_hu_mask = ct > hu_threshold
    count = int(np.sum(high_hu_mask))
    detected = count >= min_voxel_count
    return detected, count


def detect_low_contrast(
    ct: np.ndarray,
    liver_mask: Optional[np.ndarray],
    diff_threshold: float = 30.0,
    std_threshold: float = 20.0,
) -> Tuple[bool, float, float, float]:
    """
    Détecte un faible contraste hépatique.

    Stratégie :
    - Calcule la moyenne HU dans le foie (région d'intérêt = liver_mask)
    - Calcule la moyenne HU du fond abdominal (tissu mou hors foie)
    - Si la différence < diff_threshold → faible contraste

    Returns
    -------
    (detected, liver_mean, liver_std, background_mean)
    """
    if liver_mask is None or liver_mask.sum() == 0:
        log.warning("Pas de masque foie disponible pour detect_low_contrast.")
        return False, 0.0, 0.0, 0.0

    liver_hu = ct[liver_mask == 1]
    liver_mean = float(np.mean(liver_hu))
    liver_std = float(np.std(liver_hu))

    # Fond = tissu mou abdominal : HU entre -100 et 100, hors foie
    soft_tissue = (ct > -100) & (ct < 100) & (liver_mask == 0)
    if soft_tissue.sum() == 0:
        background_mean = 0.0
    else:
        background_mean = float(np.mean(ct[soft_tissue]))

    diff = abs(liver_mean - background_mean)
    detected = diff < diff_threshold or liver_std < std_threshold

    return detected, liver_mean, liver_std, background_mean


def detect_organic_contact(
    liver_mask: Optional[np.ndarray],
    labelled_volume: Optional[np.ndarray],
    dilation_voxels: int = 3,
    min_contact_voxels: int = 100,
) -> Tuple[bool, int, List[int]]:
    """
    Détecte si le foie est en contact avec d'autres organes.

    Méthode :
    1. Dilate le masque foie de `dilation_voxels` voxels.
    2. Intersecte avec le volume labellisé (LABELLED_DICOM).
    3. Collecte les identifiants d'organes adjacents (labels ≠ 0 et ≠ label_foie).

    Parameters
    ----------
    liver_mask : np.ndarray
        Masque binaire du foie.
    labelled_volume : np.ndarray
        Volume où chaque voxel porte l'identifiant de son organe (0 = fond).
    dilation_voxels : int
        Rayon de dilatation en voxels.
    min_contact_voxels : int
        Seuil minimum de voxels de contact.

    Returns
    -------
    (detected, total_contact_voxels, adjacent_organ_ids)
    """
    if liver_mask is None or labelled_volume is None:
        return False, 0, []

    # Dilatation morphologique du masque foie
    struct = np.ones((dilation_voxels, dilation_voxels, dilation_voxels), dtype=bool)
    dilated = binary_dilation(liver_mask.astype(bool), structure=struct)

    # Zone de contact = dilatation − masque original
    contact_zone = dilated & ~liver_mask.astype(bool)

    # Labels dans la zone de contact
    labels_in_contact = labelled_volume[contact_zone]
    unique_labels = [int(lbl) for lbl in np.unique(labels_in_contact) if lbl != 0]

    # Comptage des voxels de contact (étiquetés, hors fond)
    contact_voxels = int(np.sum(labels_in_contact > 0))

    # Identifier le label du foie pour l'exclure
    liver_label = _infer_liver_label(labelled_volume, liver_mask)
    adjacent_ids = [lbl for lbl in unique_labels if lbl != liver_label]

    detected = contact_voxels >= min_contact_voxels and len(adjacent_ids) > 0

    return detected, contact_voxels, adjacent_ids


def _infer_liver_label(labelled_volume: np.ndarray, liver_mask: np.ndarray) -> int:
    """
    Infère le label entier du foie dans le volume labellisé
    en cherchant le label le plus fréquent au sein du masque foie.
    """
    if liver_mask.sum() == 0:
        return -1
    labels = labelled_volume[liver_mask == 1]
    labels = labels[labels > 0]
    if len(labels) == 0:
        return -1
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du volume labellisé
# ─────────────────────────────────────────────────────────────────────────────

def load_labelled_volume(patient_dir: Path, config: dict) -> Optional[np.ndarray]:
    """
    Charge le volume labellisé depuis LABELLED_DICOM/.
    Retourne un tableau int16 (Z, Y, X) ou None si absent.
    """
    labelled_folder = patient_dir / config["dataset"]["subfolders"]["labelled_dicom"]
    if not labelled_folder.exists():
        log.warning("LABELLED_DICOM absent pour : %s", patient_dir.name)
        return None

    image = _load_dicom_series_sitk(labelled_folder)
    if image is None:
        return None

    arr, _, _, _ = sitk_to_numpy(image)
    return arr.astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
# Classificateur principal
# ─────────────────────────────────────────────────────────────────────────────

def classify_patient(pv: PatientVolume, config: dict) -> ClassificationResult:
    """
    Classifie un patient selon ses caractéristiques image.

    Parameters
    ----------
    pv : PatientVolume
        Volume chargé (CT + masques).
    config : dict
        Configuration YAML.

    Returns
    -------
    ClassificationResult
    """
    cls_cfg = config["classification"]
    result = ClassificationResult(patient_id=pv.patient_id)

    if pv.ct_volume is None:
        result.notes.append("Volume CT absent - classification impossible.")
        return result

    ct = pv.ct_volume

    # ── 1. Artefacts métalliques ──────────────────────────────────────────────
    metal_cfg = cls_cfg["metal_artifact"]
    detected_metal, metal_count = detect_metal_artifact(
        ct,
        hu_threshold=metal_cfg["hu_threshold"],
        min_voxel_count=metal_cfg["min_voxel_count"],
    )
    result.metal_voxel_count = metal_count
    if detected_metal:
        result.difficulty |= DifficultyFlag.METAL_ARTIFACT
        result.notes.append(
            f"Artefact métallique détecté : {metal_count} voxels > {metal_cfg['hu_threshold']} HU"
        )
        log.info("[Patient %02d] Artefact métallique (%d voxels)", pv.patient_id, metal_count)

    # ── 2. Faible contraste ───────────────────────────────────────────────────
    lc_cfg = cls_cfg["low_contrast"]
    detected_lc, liver_mean, liver_std, bg_mean = detect_low_contrast(
        ct,
        pv.liver_mask,
        diff_threshold=lc_cfg["liver_background_diff_threshold"],
        std_threshold=lc_cfg["liver_std_threshold"],
    )
    result.liver_mean_hu = liver_mean
    result.liver_std_hu = liver_std
    result.background_mean_hu = bg_mean
    result.liver_background_diff = abs(liver_mean - bg_mean)

    if detected_lc:
        result.difficulty |= DifficultyFlag.LOW_CONTRAST
        result.notes.append(
            f"Faible contraste : ΔHU={result.liver_background_diff:.1f}, "
            f"std_foie={liver_std:.1f}"
        )
        log.info("[Patient %02d] Faible contraste (ΔHU=%.1f)", pv.patient_id, result.liver_background_diff)

    # ── 3. Contact organique ──────────────────────────────────────────────────
    labelled = load_labelled_volume(pv.patient_dir, config)
    oc_cfg = cls_cfg["organic_contact"]

    # Conversion dilation_mm → voxels (utilise spacing z)
    dil_mm = oc_cfg["dilation_mm"]
    sz = pv.spacing[0] if pv.spacing[0] > 0 else 1.0
    dilation_voxels = max(1, int(round(dil_mm / sz)))

    detected_oc, contact_vox, adj_ids = detect_organic_contact(
        pv.liver_mask,
        labelled,
        dilation_voxels=dilation_voxels,
        min_contact_voxels=oc_cfg["min_contact_voxels"],
    )
    result.contact_voxels = contact_vox
    result.adjacent_organ_ids = adj_ids

    if detected_oc:
        result.difficulty |= DifficultyFlag.ORGANIC_CONTACT
        result.notes.append(
            f"Contact organique : {contact_vox} voxels, organes adj.={adj_ids}"
        )
        log.info(
            "[Patient %02d] Contact organique (%d voxels, organes=%s)",
            pv.patient_id, contact_vox, adj_ids,
        )

    # ── 4. Cas facile (aucune difficulté détectée) ────────────────────────────
    hom_cfg = cls_cfg["homogeneous_liver"]
    is_homogeneous = (
        liver_std < hom_cfg["std_threshold"]
        and result.liver_background_diff >= hom_cfg["liver_background_diff_min"]
    )
    if result.difficulty == DifficultyFlag.NONE and is_homogeneous:
        result.difficulty |= DifficultyFlag.EASY
        result.notes.append(
            f"Foie homogène : std={liver_std:.1f} HU, ΔHU={result.liver_background_diff:.1f}"
        )
        log.info("[Patient %02d] Cas facile", pv.patient_id)
    elif result.difficulty == DifficultyFlag.NONE:
        result.notes.append("Cas non classifié (critères ambigus).")

    return result


def classify_dataset(
    dataset: Dict[int, PatientVolume],
    config: dict,
) -> Dict[int, ClassificationResult]:
    """
    Classifie tous les patients d'un dataset chargé.

    Returns
    -------
    dict {patient_id: ClassificationResult}
    """
    log.info("=== Classification des patients ===")
    results = {}
    for pid, pv in sorted(dataset.items()):
        res = classify_patient(pv, config)
        results[pid] = res
        log.info(str(res))

    # Résumé
    easy = [pid for pid, r in results.items() if DifficultyFlag.EASY in r.difficulty]
    metal = [pid for pid, r in results.items() if DifficultyFlag.METAL_ARTIFACT in r.difficulty]
    low_c = [pid for pid, r in results.items() if DifficultyFlag.LOW_CONTRAST in r.difficulty]
    contact = [pid for pid, r in results.items() if DifficultyFlag.ORGANIC_CONTACT in r.difficulty]

    log.info("─"* 60)
    log.info("Résumé classification :")
    log.info("Cas faciles : %s", easy)
    log.info("Artefacts métalliques: %s", metal)
    log.info("Faible contraste : %s", low_c)
    log.info("Contacts organiques : %s", contact)
    log.info("=== Fin classification ===")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Type hint (import circulaire évité)
# ─────────────────────────────────────────────────────────────────────────────
from typing import Dict # noqa: E402 (already imported above, kept for clarity)