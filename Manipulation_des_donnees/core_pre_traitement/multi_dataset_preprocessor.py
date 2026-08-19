
#inspiré par LAMY 2020
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Manipulation_des_donnees.core_pre_traitement.Bullit_patient.bullit_file_resolver import BullittFileResolver
from Manipulation_des_donnees.core_pre_traitement.Ircad_patient.patient_file_resolver import PatientFileResolver
from Manipulation_des_donnees.core_pre_traitement.vascusynth.vascusynth_file_resolver import VascuSynthFileResolver
from Manipulation_des_donnees.core_pre_traitement.vascusynth.tree_rasterizer import (
    rasterize_tree_to_mask,
    NOISE_LEVELS,
    BIAS_SIGMA,
    TARGET_DATA_INDICES,
)
from Manipulation_des_donnees.core_pre_traitement.report_generator import ReportGenerator
from Manipulation_des_donnees.core_pre_traitement.resampler import Resampler


# Constantes / reproductibilité
SEED = 42
np.random.seed(SEED)

DEFAULT_IRCAD_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)
DEFAULT_MARGIN_PX = 15


# Configuration
@dataclass
class IRCADConfig:
    input_dir: Path
    output_dir: Path = Path("data/ircad_preprocessed")
    target_spacing: Tuple[float, float, float] = DEFAULT_IRCAD_SPACING
    apply_liver_mask: bool = True
    n_patients: int = 20

    @classmethod
    def from_dict(cls, raw: dict) -> "IRCADConfig":
        return cls(
            input_dir=Path(raw["input_dir"]),
            output_dir=Path(raw.get("output_dir", "data/ircad_preprocessed")),
            target_spacing=tuple(raw.get("target_spacing", DEFAULT_IRCAD_SPACING)),
            apply_liver_mask=raw.get("apply_liver_mask", True),
            n_patients=raw.get("n_patients", 20),
        )


@dataclass
class BullittConfig:
    input_dir: Path
    output_dir: Path = Path("data/bullitt_preprocessed")
    margin_px: int = DEFAULT_MARGIN_PX
    apply_tissue_mask: bool = True
    verbose: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> "BullittConfig":
        return cls(
            input_dir=Path(raw["input_dir"]),
            output_dir=Path(raw.get("output_dir", "data/bullitt_preprocessed")),
            margin_px=raw.get("margin_px", DEFAULT_MARGIN_PX),
            apply_tissue_mask=raw.get("apply_tissue_mask", True),
            verbose=raw.get("verbose", True),
        )


@dataclass
class SenNetConfig:
    input_dir: Path
    output_dir: Path = Path("data/sennet_preprocessed")
    dataset_name: str = "kidney_1_voi"
    voxel_size_um: float = 5.20
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    normalize: bool = True
    
    # Options de crop
    crop: bool = False # True = extraire un crop, False = volume entier
    crop_size: int = 384 # Taille du crop (voxels)
    crop_block: int = 128 # Taille du bloc pour le scan
    crop_normalize: bool = True # Normaliser le crop en [0,1]
    crop_min_diameter_um: float = 15.0 # Diamètre min des vaisseaux à cibler
    crop_max_diameter_um: float = 80.0 # Diamètre max des vaisseaux à cibler

    @classmethod
    def from_dict(cls, raw: dict) -> "SenNetConfig":
        return cls(
            input_dir=Path(raw["input_dir"]),
            output_dir=Path(raw.get("output_dir", "data/sennet_preprocessed")),
            dataset_name=raw.get("dataset_name", "kidney_1_voi"),
            voxel_size_um=raw.get("voxel_size_um", 5.20),
            percentile_low=raw.get("percentile_low", 1.0),
            percentile_high=raw.get("percentile_high", 99.0),
            normalize=raw.get("normalize", True),
            crop=raw.get("crop", False),
            crop_size=raw.get("crop_size", 384),
            crop_block=raw.get("crop_block", 128),
            crop_normalize=raw.get("crop_normalize", True),
            crop_min_diameter_um=raw.get("crop_min_diameter_um", 15.0),
            crop_max_diameter_um=raw.get("crop_max_diameter_um", 80.0),
        )


@dataclass
class VascuSynthConfig:
    input_dir: Path
    output_dir: Path = Path("data/vascusynth_preprocessed")
    groups: Optional[list] = None
    n_data_per_group: int = 12
    generate_gt_mask: bool = True # Toujours True pour l'article
    generate_bifurcation_mask: bool = True # ROI 3
    generate_vessel_neighborhood: bool = True # ROI 2
    neighborhood_dilation_iterations: int = 3 # Dilatation pour ROI 2
    noise_levels: Optional[list] = None # Si None, utilise NOISE_LEVELS par défaut

    @classmethod
    def from_dict(cls, raw: dict) -> "VascuSynthConfig":
        return cls(
            input_dir=Path(raw["input_dir"]),
            output_dir=Path(raw.get("output_dir", "data/vascusynth_preprocessed")),
            groups=raw.get("groups", None),
            n_data_per_group=raw.get("n_data_per_group", 12),
            generate_gt_mask=raw.get("generate_gt_mask", True),
            generate_bifurcation_mask=raw.get("generate_bifurcation_mask", True),
            generate_vessel_neighborhood=raw.get("generate_vessel_neighborhood", True),
            neighborhood_dilation_iterations=raw.get("neighborhood_dilation_iterations", 3),
            noise_levels=raw.get("noise_levels", None),
        )



# Utilitaires SimpleITK partagés
class ImageOps:

    @staticmethod
    def resample_to_spacing(
        image: sitk.Image,
        target_spacing: Tuple[float, float, float],
        interpolator=sitk.sitkBSpline,
    ) -> sitk.Image:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        return resampler.Execute(image)

    @staticmethod
    def resample_to_reference(
        image: sitk.Image,
        reference: sitk.Image,
        interpolator=sitk.sitkNearestNeighbor,
    ) -> sitk.Image:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        return resampler.Execute(image)

    @staticmethod
    def compute_bbox_from_mask(mask: sitk.Image, margin: int = 15) -> Tuple[int, int, int, int, int, int]:
        array = sitk.GetArrayFromImage(mask)
        coords = np.where(array > 0.5)
        if len(coords[0]) == 0:
            return (0, array.shape[0], 0, array.shape[1], 0, array.shape[2])

        z_min, z_max = int(coords[0].min()), int(coords[0].max()) + 1
        y_min, y_max = int(coords[1].min()), int(coords[1].max()) + 1
        x_min, x_max = int(coords[2].min()), int(coords[2].max()) + 1
        return (z_min, z_max, y_min, y_max, x_min, x_max)

    @staticmethod
    def crop_with_margin(
        image: sitk.Image,
        bbox: Tuple[int, int, int, int, int, int],
        margin: int = 15,
    ) -> sitk.Image:
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        size = image.GetSize()
        z_min, z_max = max(0, z_min - margin), min(size[2], z_max + margin)
        y_min, y_max = max(0, y_min - margin), min(size[1], y_max + margin)
        x_min, x_max = max(0, x_min - margin), min(size[0], x_max + margin)
        return image[z_min:z_max, y_min:y_max, x_min:x_max]

    @staticmethod
    def normalize_min_max(image: sitk.Image) -> sitk.Image:
        array = sitk.GetArrayFromImage(image)
        min_val, max_val = array.min(), array.max()
        if max_val - min_val > 1e-10:
            array = (array - min_val) / (max_val - min_val)

        normalized = sitk.GetImageFromArray(array.astype(np.float32))
        normalized.CopyInformation(image)
        return normalized

    @staticmethod
    def apply_mask(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
        array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        array[~(mask_array > 0.5)] = 0

        masked = sitk.GetImageFromArray(array.astype(np.float32))
        masked.CopyInformation(image)
        return masked

    @staticmethod
    def normalize_percentile_clip_minmax(
        image: sitk.Image,
        p_low: float = 1.0,
        p_high: float = 99.0,
    ) -> Tuple[sitk.Image, float, float]:
        array = sitk.GetArrayFromImage(image)
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        lo, hi = np.percentile(array, [p_low, p_high])
        np.clip(array, lo, hi, out=array)
        if hi - lo > 1e-10:
            array -= lo
            array /= (hi - lo)
        else:
            array[:] = 0.0

        normalized = sitk.GetImageFromArray(array)
        normalized.CopyInformation(image)
        return normalized, float(lo), float(hi)


class ImageOpsStreaming:

    @staticmethod
    def percentile_from_sample(
        memmap_array: np.ndarray,
        p_low: float,
        p_high: float,
        sample_fraction: float = 0.02,
        min_samples: int = 10_000_000,
        seed: int = 42,
    ) -> Tuple[float, float]:
        rng = np.random.default_rng(seed)
        n_total = memmap_array.size
        n_sample = min(max(int(n_total * sample_fraction), min_samples), n_total)

        flat = memmap_array.reshape(-1)
        idx = rng.integers(0, n_total, size=n_sample)
        idx.sort()
        sample = flat[idx]
        lo, hi = np.percentile(sample, [p_low, p_high])
        return float(lo), float(hi)

    @staticmethod
    def clip_minmax_chunked_inplace(
        memmap_array: np.ndarray,
        lo: float,
        hi: float,
        chunk_z: int = 64,
    ) -> None:
        span = hi - lo
        n_z = memmap_array.shape[0]
        n_chunks = (n_z + chunk_z - 1) // chunk_z
        for i, z0 in enumerate(range(0, n_z, chunk_z), 1):
            z1 = min(z0 + chunk_z, n_z)
            block = memmap_array[z0:z1]
            if span > 1e-10:
                np.clip(block, lo, hi, out=block)
                block -= lo
                block /= span
            else:
                block[:] = 0.0
            memmap_array[z0:z1] = block
            print(f"\r Normalisation : bloc {i}/{n_chunks} (coupes {z0}-{z1})", end="")
        print()

    @staticmethod
    def save_nifti_streaming(array: np.ndarray, spacing_mm: float, out_path: Path) -> None:
        print(f"Écriture {out_path.name}"+ "...")
        affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
        img = nib.Nifti1Image(array.transpose(2, 1, 0), affine)
        img.header.set_data_dtype(array.dtype)
        nib.save(img, str(out_path))
        print(f"OK : {out_path}")


# IRCAD
class IRCADPreprocessor:
    def __init__(self, config: IRCADConfig):
        self.config = config
        self.resolver = PatientFileResolver(config.input_dir)
        self.resampler = Resampler(config.target_spacing)
        self.reporter = ReportGenerator(config.output_dir, config.target_spacing)
        self.results: list[dict] = []

    def run(self) -> None:
        self._print_header()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        patient_ids = self.resolver.get_patient_ids(self.config.n_patients)
        if not patient_ids:
            print("ERROR: Aucun patient trouvé.")
            return

        print(f"\n{len(patient_ids)} patient(s) trouvé(s).\n")

        for pid in tqdm(patient_ids, desc="Preprocessing IRCAD"):
            result = self._process_patient(pid)
            self.results.append(result)
            self._log_result(pid, result)

        self.reporter.generate(self.results)

    def _print_header(self) -> None:
        print("\n"+ "="* 80)
        print("IRCAD PREPROCESSING (VERSION ORIGINALE)")
        print("="* 80)
        print(f"Input : {self.config.input_dir}")
        print(f"Output : {self.config.output_dir}")
        print(f"Spacing : {self.config.target_spacing} mm")
        print(f"Masque foie : {'OUI'if self.config.apply_liver_mask else 'NON'}")
        print("="* 80)

    @staticmethod
    def _log_result(pid: str, result: dict) -> None:
        if result["status"] == "success":
            print(f"OK {pid}: {result['time_seconds']:.2f}s")
        else:
            print(f"FAIL {pid}: {result.get('error', 'Erreur')}")

    def _process_patient(self, patient_id: str) -> dict:
        start = time.time()
        patient_dir = self.config.input_dir / patient_id

        if not patient_dir.exists():
            return self._fail(patient_id, f"Dossier introuvable : {patient_dir}")

        files = self.resolver.resolve(patient_id)
        if files["ct"] is None:
            return self._fail(patient_id, "Aucune image CT trouvée")

        out_dir = self.config.output_dir / patient_id
        out_dir.mkdir(exist_ok=True)

        suffix = "in_liver"if self.config.apply_liver_mask else "original"
        processed: dict = {}

        ct_original, ct_resampled = self._process_ct(files["ct"], out_dir, patient_id, processed)
        self._process_liver_mask(files.get("liver"), ct_resampled, out_dir, patient_id, processed)
        self._process_vessel_fused(files.get("vessel_fused"), ct_resampled, out_dir, patient_id, suffix, processed)
        self._process_individual_vessels(files.get("vessels_individual", []), ct_resampled, out_dir, patient_id, suffix, processed)

        return {
            "patient_id": patient_id,
            "status": "success",
            "dataset": "ircad",
            "version": suffix,
            "original_spacing": list(ct_original.GetSpacing()),
            "original_size": list(ct_original.GetSize()),
            "new_spacing": list(ct_resampled.GetSpacing()),
            "new_size": list(ct_resampled.GetSize()),
            "processed_files": processed,
            "time_seconds": time.time() - start,
        }

    def _process_ct(self, ct_path: Path, out_dir: Path, patient_id: str, processed: dict):
        print(f"CT : {ct_path.name}")
        ct_original = sitk.ReadImage(str(ct_path))
        ct_resampled = self.resampler.resample_image(ct_original)

        ct_out = out_dir / f"{patient_id}_ct_isotropic.nii.gz"
        sitk.WriteImage(ct_resampled, str(ct_out))
        processed["ct"] = str(ct_out)
        return ct_original, ct_resampled

    def _process_liver_mask(self, liver_path: Optional[Path], ct_resampled: sitk.Image, out_dir: Path, patient_id: str, processed: dict) -> None:
        if not liver_path:
            return
        print(f"Liver : {liver_path.name}")
        liver = sitk.ReadImage(str(liver_path))
        liver_out = out_dir / f"{patient_id}_liver_mask_isotropic.nii.gz"
        sitk.WriteImage(self.resampler.resample_mask(liver, reference_image=ct_resampled), str(liver_out))
        processed["liver_mask"] = str(liver_out)

    def _process_vessel_fused(self, vf_path: Optional[Path], ct_resampled: sitk.Image, out_dir: Path, patient_id: str, suffix: str, processed: dict) -> None:
        if not vf_path:
            return
        print(f"Vessels GT : {vf_path.name}")
        vf = sitk.ReadImage(str(vf_path))
        vf_out = out_dir / f"{patient_id}_vessels_gt_isotropic_{suffix}.nii.gz"
        sitk.WriteImage(self.resampler.resample_mask(vf, reference_image=ct_resampled), str(vf_out))
        processed["vessels_gt"] = str(vf_out)

    def _process_individual_vessels(self, vessel_files: list, ct_resampled: sitk.Image, out_dir: Path, patient_id: str, suffix: str, processed: dict) -> None:
        processed["individual_vessels"] = []
        for vfile in vessel_files:
            print(f"Vessel : {vfile.name}")
            stem = vfile.name.replace(".nii.gz", "").replace(".nii", "")
            v = sitk.ReadImage(str(vfile))
            v_out = out_dir / f"{patient_id}_{stem}__{suffix}_isotropic.nii.gz"
            sitk.WriteImage(self.resampler.resample_mask(v, reference_image=ct_resampled), str(v_out))
            processed["individual_vessels"].append(str(v_out))

    @staticmethod
    def _fail(patient_id: str, error: str) -> dict:
        return {"patient_id": patient_id, "status": "failed", "error": error}


# Bullitt
class BullittPreprocessor:
    def __init__(self, config: BullittConfig):
        self.config = config
        self.resolver = BullittFileResolver(config.input_dir)
        self.reporter = ReportGenerator(config.output_dir, "finest-per-patient")
        self.results: list[dict] = []
        self.ops = ImageOps()

    def run(self) -> None:
        self._print_header()
        images_dir, labels_dir, masks_dir = self._make_output_dirs()

        patient_ids = self.resolver.get_patient_ids()
        if not patient_ids:
            print("ERROR: Aucun patient trouvé.")
            return

        print(f"\n{len(patient_ids)} patient(s) trouvé(s).\n")

        for i, pid in enumerate(tqdm(patient_ids, desc="Preprocessing Bullitt"), 1):
            result = self._process_patient(pid, i)
            self.results.append(result)
            self._log_result(pid, i, result)

        self.reporter.generate(self.results)

    def _print_header(self) -> None:
        print("\n"+ "="* 80)
        print("BULLITT PREPROCESSING (NOUVEAU PROTOCOLE)")
        print("="* 80)
        print(f"Input : {self.config.input_dir}")
        print(f"Recadrage (bbox) : DÉSACTIVÉ (réservé à l'évaluation)")
        print(f"Masquage cerveau : DÉSACTIVÉ (réservé à l'évaluation)")
        print("="* 80)

    def _make_output_dirs(self) -> Tuple[Path, Path, Path]:
        images_dir = self.config.output_dir / "images"
        labels_dir = self.config.output_dir / "labels"
        masks_dir = self.config.output_dir / "masks"
        for d in (images_dir, labels_dir, masks_dir):
            d.mkdir(parents=True, exist_ok=True)
        return images_dir, labels_dir, masks_dir

    @staticmethod
    def _log_result(pid: str, idx: int, result: dict) -> None:
        if result["status"] == "success":
            print(f"OK {pid} -> patient_{idx:02d} ({result['time_seconds']:.2f}s)")
        else:
            print(f"FAIL {pid}: {result.get('error', 'Erreur')}")

    def _process_patient(self, patient_id: str, idx: int) -> dict:
        start = time.time()
        patient_dir = self.config.input_dir / patient_id

        if not patient_dir.exists():
            return self._fail(patient_id, f"Dossier introuvable : {patient_dir}")

        files = self.resolver.resolve(patient_id)
        if files.get("mra") is None:
            return self._fail(patient_id, "Fichier MRA non trouvé")
        if files.get("brain_mask") is None:
            return self._fail(patient_id, "Masque cerveau non trouvé")

        try:
            return self._run_patient_pipeline(patient_id, idx, files, start)
        except Exception as exc:
            traceback.print_exc()
            return self._fail(patient_id, str(exc))

    def _run_patient_pipeline(self, patient_id: str, idx: int, files: dict, start: float) -> dict:
        mra = sitk.ReadImage(str(files["mra"]))
        brain_mask = sitk.ReadImage(str(files["brain_mask"]))

        original_spacing = mra.GetSpacing()
        original_size = mra.GetSize()
        if self.config.verbose:
            print(f"\nPatient: {patient_id}")
            print(f"MRA: {original_size} (spacing={original_spacing})")

        finest = min(original_spacing)
        target_spacing = (finest, finest, finest)
        mra_resampled = self.ops.resample_to_spacing(mra, target_spacing, sitk.sitkBSpline)
        brain_mask_resampled = self.ops.resample_to_reference(brain_mask, mra_resampled, sitk.sitkNearestNeighbor)

        mra_final = self.ops.normalize_min_max(mra_resampled)

        prefix = f"patient_{idx:02d}"
        img_out = self.config.output_dir / "images"/ f"{prefix}_images.nii.gz"
        mask_out = self.config.output_dir / "masks"/ f"{prefix}_brain.nii.gz"
        sitk.WriteImage(mra_final, str(img_out))
        sitk.WriteImage(brain_mask_resampled, str(mask_out))

        label_out = self._process_vessel_label(files.get("vessel_mask"), patient_id, mra, mra_resampled, prefix)

        return {
            "patient_id": patient_id,
            "status": "success",
            "dataset": "bullitt",
            "normalization_applied": True,
            "cropping_applied": False,
            "masking_applied": False,
            "original_spacing": list(original_spacing),
            "original_size": list(original_size),
            "new_spacing": list(target_spacing),
            "new_size": list(mra_final.GetSize()),
            "processed_files": {
                "image": str(img_out),
                "label": str(label_out) if label_out else None,
                "mask": str(mask_out),
            },
            "time_seconds": time.time() - start,
        }

    def _process_vessel_label(
        self,
        vessel_mask_path: Optional[Path],
        patient_id: str,
        mra_original: sitk.Image,
        mra_resampled: sitk.Image,
        prefix: str,
    ) -> Optional[Path]:
        if vessel_mask_path is None:
            if self.config.verbose:
                print("Pas de masque vasculaire trouvé.")
            return None

        try:
            vessels = sitk.ReadImage(str(vessel_mask_path))
            if vessels.GetSize() != mra_original.GetSize():
                vessels = self.ops.resample_to_reference(vessels, mra_original, sitk.sitkNearestNeighbor)

            vessels_resampled = self.ops.resample_to_reference(vessels, mra_resampled, sitk.sitkNearestNeighbor)

            label_out = self.config.output_dir / "labels"/ f"{prefix}_label.nii.gz"
            sitk.WriteImage(vessels_resampled, str(label_out))
            return label_out
        except Exception as exc:
            print(f"ERROR: Échec du traitement du masque vasculaire pour {patient_id}: {exc}")
            return None

    @staticmethod
    def _fail(patient_id: str, error: str) -> dict:
        return {"patient_id": patient_id, "status": "failed", "error": error}


    # Pipeline
    def run(self) -> None:
        self._print_header()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        result = self._process_volume()
        self.results.append(result)
        self._log_result(result)

        self.reporter.generate(self.results)

    def _print_header(self) -> None:
        print("\n"+ "="* 80)
        print(f"Input : {self.config.input_dir}")
        print(f"Dataset : {self.config.dataset_name}")
        print(f"Voxel size : {self.config.voxel_size_um} µm (isotrope, natif)")
        print(f"Rééchantillonnage : DÉSACTIVÉ (déjà isotrope, VOI unique)")
        print(f"Mode : {'CROP'if self.config.crop else 'VOLUME ENTIER'}")
        if self.config.crop:
            print(f"Crop size : {self.config.crop_size}³")
            print(f"Diamètre ciblé : {self.config.crop_min_diameter_um}-{self.config.crop_max_diameter_um} µm")
            print(f"Normalisation : {'OUI'if self.config.crop_normalize else 'NON'}")
        else:
            print(f"Normalisation : {'OUI'if self.config.normalize else 'NON'}")
        print("="* 80)

    @staticmethod
    def _log_result(result: dict) -> None:
        if result["status"] == "success":
            print(f"OK {result['patient_id']}: {result['time_seconds']:.2f}s")
        else:
            print(f"FAIL {result['patient_id']}: {result.get('error', 'Erreur')}")

    # Volume unique
    def _process_volume(self) -> dict:
        start = time.time()
        dataset_name = self.config.dataset_name

        files = self.resolver.resolve()
        if files is None:
            return self._fail(
                dataset_name,
                f"Aucune coupe commune images/labels trouvée dans "
                f"{self.config.input_dir / dataset_name}",
            )

        n_common = len(files["common_indices"])
        z_coverage = n_common / files["n_images_total"] if files["n_images_total"] else 0.0
        print(f"Coupes communes images/labels : {n_common} "
              f"({z_coverage:.1%} des {files['n_images_total']} images)")
        if z_coverage < 0.9:
            print(f"️ Couverture z partielle (< 90%) - restriction à l'intersection "
                  f"réelle, PAS à un stack positionnel.")

        spacing_mm = self.config.voxel_size_um / 1000.0

        # Cas 1 : CROP
        if self.config.crop:
            print("\n=== MODE CROP ===")
            
            # Charger tout le volume (memmap)
            imgs_array, lbls_array = self.resolver.load_volume(files)
            
            # Extraire le crop
            img_crop, gt_crop, offset = self._extract_crop(lbls_array, imgs_array)
            
            # Normaliser le crop en [0,1] si demandé
            if self.config.crop_normalize:
                print("Normalisation du crop en [0,1]...")
                img_crop = img_crop.astype(np.float32)
                img_crop = (img_crop - img_crop.min()) / (img_crop.max() - img_crop.min() + 1e-10)
                img_crop = np.clip(img_crop, 0, 1)
                print(f"min={img_crop.min():.3f}, max={img_crop.max():.3f}")
            
            # Sauvegarder le crop
            out_dir = self.config.output_dir / dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            img_out = out_dir / f"{dataset_name}_image_crop_{self.config.crop_size}.nii"
            lbl_out = out_dir / f"{dataset_name}_vessels_gt_crop_{self.config.crop_size}.nii.gz"
            
            # Affine translatée
            affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
            offset_world = np.array(offset) * spacing_mm
            affine[:3, 3] = offset_world
            
            nib.save(nib.Nifti1Image(img_crop, affine), str(img_out))
            nib.save(nib.Nifti1Image(gt_crop, affine), str(lbl_out))
            
            print(f"Crop sauvegardé: {img_out}")
            print(f"Label sauvegardé: {lbl_out}")
            
            return {
                "patient_id": dataset_name,
                "status": "success",
                "dataset": "sennethoa",
                "crop_applied": True,
                "crop_size": self.config.crop_size,
                "crop_offset": offset,
                "normalization_applied": self.config.crop_normalize,
                "normalization_method": "minmax_to_[0,1]"if self.config.crop_normalize else "none",
                "processed_files": {
                    "ct": str(img_out),
                    "vessels_gt": str(lbl_out),
                },
                "time_seconds": time.time() - start,
            }

        # Cas 2 : VOLUME ENTIER
        else:
            print("\n=== MODE VOLUME ENTIER ===")
            imgs_array, lbls_array = self.resolver.load_volume(files)
            
            if self.config.normalize:
                clip_lo, clip_hi = ImageOpsStreaming.percentile_from_sample(
                    imgs_array, self.config.percentile_low, self.config.percentile_high
                )
                print(f"Percentiles: p{self.config.percentile_low}={clip_lo:.1f}, "
                      f"p{self.config.percentile_high}={clip_hi:.1f}")
                ImageOpsStreaming.clip_minmax_chunked_inplace(imgs_array, clip_lo, clip_hi)
                norm_method = f"percentile_clip[{self.config.percentile_low},{self.config.percentile_high}]_then_minmax"
                clip_bounds = [clip_lo, clip_hi]
            else:
                print("Normalisation DÉSACTIVÉE")
                imgs_array = imgs_array.astype(np.float32)
                norm_method = "none"
                clip_bounds = [0, 65535]
            
            out_dir = self.config.output_dir / dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            img_out = out_dir / f"{dataset_name}_image_isotropic.nii"
            lbl_out = out_dir / f"{dataset_name}_vessels_gt_isotropic.nii.gz"
            
            ImageOpsStreaming.save_nifti_streaming(imgs_array, spacing_mm, img_out)
            ImageOpsStreaming.save_nifti_streaming(lbls_array, spacing_mm, lbl_out)
            
            return {
                "patient_id": dataset_name,
                "status": "success",
                "dataset": "sennethoa",
                "crop_applied": False,
                "normalization_applied": self.config.normalize,
                "normalization_method": norm_method,
                "normalization_clip_bounds": clip_bounds,
                "processed_files": {
                    "ct": str(img_out),
                    "vessels_gt": str(lbl_out),
                },
                "time_seconds": time.time() - start,
            }

    @staticmethod
    def _fail(patient_id: str, error: str) -> dict:
        return {"patient_id": patient_id, "status": "failed", "error": error}


# VascuSynth
class VascuSynthPreprocessor:

    def __init__(self, config: VascuSynthConfig):
        self.config = config
        self.resolver = VascuSynthFileResolver(config.input_dir)
        self.reporter = ReportGenerator(config.output_dir, "native-no-resampling")
        self.results: list[dict] = []

    def run(self) -> None:
        self._print_header()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Récupérer tous les cas
        all_cases = self.resolver.get_case_ids(self.config.groups, self.config.n_data_per_group)
        
        # Filtrer pour ne garder que data 7, 9, 11 (complexités 31, 41, 51)
        from vascusynth.tree_rasterizer import TARGET_DATA_INDICES
        cases = [(g, d) for (g, d) in all_cases if d in TARGET_DATA_INDICES]
        
        if not cases:
            print(f"ERROR: Aucun cas VascuSynth trouvé pour data{TARGET_DATA_INDICES}")
            return

        print(f"\n{len(cases)} cas sélectionnés (data{TARGET_DATA_INDICES}).\n")

        for group, data in tqdm(cases, desc="Preprocessing VascuSynth (paper)"):
            result = self._process_case(group, data)
            self.results.append(result)
            self._log_result(result)

        self.reporter.generate(self.results)
# Prétraitement VascuSynth selon Lamy et al. (ICPR 2020)
    def _print_header(self) -> None:
        print("\n"+ "="* 80)
        print("VASCUSYNTH PREPROCESSING ")
        print("="* 80)
        print(f"Input : {self.config.input_dir}")
        print(f"Output : {self.config.output_dir}")
        print(f"Rééchantillonnage : DÉSACTIVÉ (conversion directe .mhd -> .nii.gz)")
        print(f"Normalisation : DÉSACTIVÉE")
        print(f"GT (vaisseaux) : seuillage > 0 de l'image brute")
        print(f"Biais (illumination): 3 gaussiennes (σ={BIAS_SIGMA})")
        print(f"Bruit Rician : σ = {NOISE_LEVELS}")
        print(f"Cas sélectionnés : data{TARGET_DATA_INDICES} (complexités 31, 41, 51)")
        print("="* 80)

    @staticmethod
    def _log_result(result: dict) -> None:
        if result["status"] == "success":
            print(f"OK {result['patient_id']}: {result['time_seconds']:.2f}s")
        else:
            print(f"FAIL {result['patient_id']}: {result.get('error', 'Erreur')}")

    def _process_case(self, group: int, data: int) -> dict:
        start = time.time()
        case = self.resolver.resolve(group, data)
        if case is None:
            return self._fail(f"Group{group}_data{data}", "Fichiers .mhd/.mat introuvables")

        case_id = case.case_id
        bifurcations = 1 + 5 * (data - 1)
        print(f"\n{case_id} (bifurcations={bifurcations}) : {case.image_path.name}")

        try:
            image = sitk.ReadImage(str(case.image_path))
        except Exception as exc:
            return self._fail(case_id, f"Échec lecture image MetaImage : {exc}")

        # Récupérer les métadonnées
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()
        dat_zyx = sitk.GetArrayFromImage(image) # (z, y, x)

        from vascusynth.tree_rasterizer import (
            vessels_and_background, vessels_illumination, rician_noise_variants,
            ground_truth_from_raw_image, bifurcation_mask, vessel_neighborhood_mask
        )

        # Étape 1 : vesselsAndBackground - rescale [50, 100]
        dat_bg = vessels_and_background(dat_zyx)

        # Étape 2 : vesselsIllumination - bias field (3 gaussiennes)
        dat_illum = vessels_illumination(dat_bg)

        # Étape 3 : Bruit Rician (3 niveaux)
        noise_levels = tuple(self.config.noise_levels) if self.config.noise_levels else NOISE_LEVELS
        noisy_variants = rician_noise_variants(dat_illum, noise_levels)

        # Étape 4 : Sauvegarde des images bruitées
        img_dir = self.config.output_dir / "images"/ case_id
        img_dir.mkdir(parents=True, exist_ok=True)
        processed: dict = {}

        for sigma, noisy_arr in noisy_variants:
            out_img = sitk.GetImageFromArray(noisy_arr)
            out_img.SetSpacing(spacing)
            out_img.SetDirection(direction)
            out_img.SetOrigin(origin)
            out_path = img_dir / f"{case_id}_rician_{sigma:.1f}.nii.gz"
            sitk.WriteImage(out_img, str(out_path))
            processed[f"image_rician_{sigma}"] = str(out_path)
            

        # Étape 5 : GT = seuillage > 0 de l'image brute
        label_dir = self.config.output_dir / "labels"/ case_id
        label_dir.mkdir(parents=True, exist_ok=True)

        gt_arr = ground_truth_from_raw_image(dat_zyx)
        gt_img = sitk.GetImageFromArray(gt_arr)
        gt_img.CopyInformation(image)
        gt_path = label_dir / f"{case_id}_vessels_gt.nii.gz"
        sitk.WriteImage(gt_img, str(gt_path))
        processed["vessels_gt"] = str(gt_path)
        print(f"OK {len(noisy_variants)} images + GT générés")

        return {
            "patient_id": case_id,
            "status": "success",
            "dataset": "vascusynth",
            "group": group,
            "data": data,
            "bifurcations": bifurcations,
            "resampling_applied": False,
            "normalization_applied": False,
            "noise_levels": list(NOISE_LEVELS),
            "gt_mask_generated": True,
            "bifurcation_mask_generated": self.config.generate_bifurcation_mask,
            "neighborhood_mask_generated": self.config.generate_vessel_neighborhood,
            "original_spacing": list(spacing),
            "original_size": list(image.GetSize()),
            "processed_files": processed,
            "time_seconds": time.time() - start,
        }

    @staticmethod
    def _fail(case_id: str, error: str) -> dict:
        return {"patient_id": case_id, "status": "failed", "error": error}


# Orchestrateur
class MultiDatasetPreprocessor:

    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    _ALL = ("ircad", "bullitt", "vascusynth")
    _BOTH = ("ircad", "bullitt")

    def run(self, dataset: str = "all") -> None:
        if dataset == "all":
            targets = self._ALL
        elif dataset == "both":
            targets = self._BOTH
        else:
            targets = (dataset,)

        if "ircad"in targets:
            ircad_config = IRCADConfig.from_dict(self.config["datasets"]["ircad"])
            IRCADPreprocessor(ircad_config).run()

        if "bullitt"in targets:
            bullitt_config = BullittConfig.from_dict(self.config["datasets"]["bullitt"])
            BullittPreprocessor(bullitt_config).run()

        if "vascusynth"in targets:
            vascusynth_config = VascuSynthConfig.from_dict(self.config["datasets"]["vascusynth"])
            VascuSynthPreprocessor(vascusynth_config).run()

        print("\n"+ "="* 80)
        print("TOUS LES PRÉTRAITEMENTS SONT TERMINÉS")
        print("="* 80)


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prétraitement unifié IRCAD, Bullitt et VascuSynth"
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["ircad", "bullitt", "vascusynth", "both", "all"],
        default="all",
        help="Dataset à prétraiter (défaut: all). 'both'est conservé pour "
             "compatibilité rétro.",
    )
    parser.add_argument(
        "--config", "-c",
        default="dataset_config.yaml",
        help="Fichier de configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = MultiDatasetPreprocessor(args.config)
    pipeline.run(args.dataset)


if __name__ == "__main__":
    main()