from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import SimpleITK as sitk
from core.io.logger import setup_logger, close_logger
from logging import Logger

log = setup_logger(__name__)


@dataclass
class PatientVolume:
    """Contient le volume CT et les masques d'un patient."""

    patient_id: int
    patient_dir: Path

    ct_volume: Optional[np.ndarray] = None
    spacing: Tuple[float, float, float] = (1., 1., 1.)
    origin: Tuple[float, float, float] = (0., 0., 0.)
    direction: Optional[np.ndarray] = None

    liver_mask: Optional[np.ndarray] = None
    vessel_mask: Optional[np.ndarray] = None # fusion de tous les vaisseaux

    vessel_masks: Dict[str, np.ndarray] = field(default_factory=dict)

    liver_mask_source: str = ""
    vessel_mask_sources: List[str] = field(default_factory=list)
    load_errors: List[str] = field(default_factory=list)

    @property
    def shape(self) -> Optional[Tuple[int, int, int]]:
        return self.ct_volume.shape if self.ct_volume is not None else None

    @property
    def has_liver(self) -> bool:
        return self.liver_mask is not None

    @property
    def has_vessels(self) -> bool:
        return self.vessel_mask is not None

    def summary(self) -> str:
        lines = [
            f"Patient {self.patient_id:02d} | {self.patient_dir.name}",
            f"Volume CT : {self.shape} | spacing={self.spacing}",
            f"Foie : {''if self.has_liver else ''} (source: {self.liver_mask_source})",
            f"Vaisseaux : {''if self.has_vessels else ''} (sources: {self.vessel_mask_sources})",
        ]
        if self.load_errors:
            lines.append(f"Erreurs : {self.load_errors}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Fonctions bas niveau
# -----------------------------------------------------------------------------

def _load_dicom_series_sitk(folder: Path) -> Optional[sitk.Image]:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(folder))
    if not dicom_names:
        log.warning("Aucun fichier DICOM trouvé dans : %s", folder)
        return None
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    try:
        return reader.Execute()
    except RuntimeError as exc:
        log.error("Erreur SimpleITK lors du chargement de %s : %s", folder, exc)
        return None


def sitk_to_numpy(image: sitk.Image) -> Tuple[np.ndarray, tuple, tuple, np.ndarray]:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    origin_xyz = image.GetOrigin()
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    origin = (origin_xyz[2], origin_xyz[1], origin_xyz[0])
    return array, spacing, origin, direction


def _load_mask_from_folder(folder: Path) -> Optional[np.ndarray]:
    image = _load_dicom_series_sitk(folder)
    if image is None:
        return None
    mask, _, _, _ = sitk_to_numpy(image)
    return (mask > 0).astype(np.uint8)


# -----------------------------------------------------------------------------
# Recherche des dossiers
# -----------------------------------------------------------------------------

def _find_mask_folder(masks_root: Path, candidate_names: List[str]) -> Optional[Path]:
    if not masks_root.exists():
        return None
    existing = {p.name.lower(): p for p in masks_root.iterdir() if p.is_dir()}
    for name in candidate_names:
        if name.lower() in existing:
            return existing[name.lower()]
    return None


def _list_all_mask_subfolders(masks_root: Path) -> List[str]:
    if not masks_root.exists():
        return []
    return [p.name for p in sorted(masks_root.iterdir()) if p.is_dir()]


# -----------------------------------------------------------------------------
# Chargeur principal
# -----------------------------------------------------------------------------

def load_patient(patient_dir: Path, config: dict, apply_liver_mask: bool = True) -> PatientVolume:
    ds_cfg = config["dataset"]

    match = re.search(r"(\d+)$", patient_dir.name)
    patient_id = int(match.group(1)) if match else -1
    pv = PatientVolume(patient_id=patient_id, patient_dir=patient_dir)

    # ---- 1. CT ----
    ct_folder = patient_dir / ds_cfg["subfolders"]["patient_dicom"]
    log.info("[Patient %02d] Chargement CT depuis : %s", patient_id, ct_folder)
    ct_image = _load_dicom_series_sitk(ct_folder)
    if ct_image is None:
        pv.load_errors.append(f"CT introuvable dans {ct_folder}")
        log.error("[Patient %02d] Impossible de charger le volume CT.", patient_id)
    else:
        pv.ct_volume, pv.spacing, pv.origin, pv.direction = sitk_to_numpy(ct_image)
        log.info("[Patient %02d] CT chargé : shape=%s, spacing=%s mm",
                 patient_id, pv.ct_volume.shape, pv.spacing)

    # ---- 2. Foie ----
    masks_root = patient_dir / ds_cfg["subfolders"]["masks_dicom"]
    available = _list_all_mask_subfolders(masks_root)
    log.debug("[Patient %02d] Sous-dossiers disponibles : %s", patient_id, available)

    liver_folder = _find_mask_folder(masks_root, ds_cfg["liver_mask_names"])
    if liver_folder is not None:
        pv.liver_mask = _load_mask_from_folder(liver_folder)
        pv.liver_mask_source = liver_folder.name
        log.info("[Patient %02d] Masque foie chargé depuis '%s'.", patient_id, liver_folder.name)
    else:
        pv.load_errors.append("Masque foie introuvable")
        log.warning("[Patient %02d] Masque foie introuvable. Candidats : %s, Disponibles : %s",
                    patient_id, ds_cfg["liver_mask_names"], available)

    # ---- 3. Vaisseaux (individuels + fusion) ----
    vessel_masks_dict = {}
    for vessel_type, candidate_names in ds_cfg["vessel_mask_names"].items():
        folder = _find_mask_folder(masks_root, candidate_names)
        if folder:
            mask = _load_mask_from_folder(folder)
            if mask is not None:
                vessel_masks_dict[vessel_type] = mask
                pv.vessel_mask_sources.append(f"{vessel_type}:{folder.name}")
                log.info(f"[Patient {patient_id:02d}] -> {vessel_type} chargé")

    if vessel_masks_dict:
        # Fusion pour l'attribut vessel_mask (rétrocompatibilité)
        fused = np.zeros_like(next(iter(vessel_masks_dict.values())), dtype=np.uint8)
        for mask in vessel_masks_dict.values():
            fused = np.logical_or(fused, mask).astype(np.uint8)
        if apply_liver_mask and pv.liver_mask is not None:
            fused = fused * pv.liver_mask
            log.info("[Patient %02d] Vaisseaux limités au foie : %d voxels", patient_id, fused.sum())
        pv.vessel_mask = fused
        pv.vessel_masks = vessel_masks_dict
        log.info("[Patient %02d] %d masque(s) vasculaire(s) conservés, fusion effectuée.",
                 patient_id, len(vessel_masks_dict))
    else:
        pv.load_errors.append("Masque vaisseaux introuvable")
        log.warning("[Patient %02d] Aucun masque vasculaire trouvé. Disponibles : %s",
                    patient_id, available)

    return pv


def load_dataset(
    config: dict,
    patient_ids: Optional[List[int]] = None,
) -> Dict[int, PatientVolume]:
    base = Path(config["dataset"]["base_path"])
    prefix = config["dataset"]["patient_prefix"]
    n = config["dataset"]["n_patients"]

    if patient_ids is None:
        patient_ids = list(range(1, n + 1))

    log.info("=== Chargement du dataset IRCAD ===")
    log.info("Répertoire de base : %s", base)
    log.info("Patients à charger : %s", patient_ids)

    if not base.exists():
        raise FileNotFoundError(
            f"Le répertoire du dataset est introuvable : {base}\n"
            "Vérifiez le champ 'dataset.base_path'dans la configuration."
        )

    dataset = {}
    for pid in patient_ids:
        patient_dir = base / f"{prefix}{pid}"
        if not patient_dir.exists():
            log.warning("Dossier patient introuvable : %s - ignoré.", patient_dir)
            continue
        pv = load_patient(patient_dir, config)
        dataset[pid] = pv
        log.info(pv.summary())

    log.info("=== Chargement terminé : %d/%d patients ===", len(dataset), len(patient_ids))
    return dataset