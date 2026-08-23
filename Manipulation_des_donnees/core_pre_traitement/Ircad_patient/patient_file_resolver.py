from pathlib import Path

import numpy as np
import SimpleITK as sitk
import re


class PatientFileResolver:

    # Mots-clés pour identifier chaque type de fichier (mode NIfTI à plat)
    _CT_KEYWORDS      = ("ct", "scan")
    _LIVER_KEYWORDS   = ("liver",)
    _FUSED_KEYWORDS   = ("vessel_fused", "fused")
    _VESSEL_KEYWORDS = ("vessel", "vein", "veno", "artery", "cava", "porte", "portal")
    _EXCLUDE_KEYWORDS = ("mask", "label")

    # Sous-dossiers du format IRCAD brut
    _RAW_CT_DIR = "PATIENT_DICOM"
    _RAW_MASKS_DIR = "MASKS_DICOM"
    _RAW_LIVER_SUBDIR = "liver"
    _CACHE_DIRNAME = ".nii_cache"

    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)

    def get_patient_ids(self, n_patients: int = 20) -> list[str]:
        if not self.input_dir.exists():
            print(f"Warning: dossier introuvable : {self.input_dir}")
            return []

        patients_with_num = []
        for d in self.input_dir.iterdir():
            if d.is_dir() and self._looks_like_patient(d):
                try:
                    num = int(d.name.split('.')[-1])
                    patients_with_num.append((num, d.name))
                except ValueError:
                     patients_with_num.append((999, d.name))
        patients_with_num.sort(key=lambda x: x[0])
        patients = [name for _, name in patients_with_num]

        if not patients:
            print(f"Warning: aucun dossier patient trouvé dans {self.input_dir}")
            return []

        if n_patients and len(patients) > n_patients:
            patients = patients[:n_patients]
        return patients


    
    def _looks_like_patient(self, patient_dir: Path) -> bool:
        if next(patient_dir.glob("*.nii*"), None) is not None:
            return True
        if (patient_dir / self._RAW_CT_DIR).is_dir():
            return True
        return False

    def resolve(self, patient_id: str) -> dict:
        patient_dir = self.input_dir / patient_id

        if (patient_dir / self._RAW_CT_DIR).is_dir():
            return self._resolve_raw_dicom(patient_dir)
        return self._resolve_nifti(patient_dir)

    def _resolve_nifti(self, patient_dir: Path) -> dict:
        files = {
            "ct": None,
            "liver": None,
            "vessel_fused": None,
            "vessels_individual": [],
        }

        for f in sorted(patient_dir.glob("*.nii*")):
            self._classify(f, f.name.lower(), files)

        if files["ct"] is None:
            for f in sorted(patient_dir.glob("*.nii*")):
                if not any(k in f.name.lower() for k in self._EXCLUDE_KEYWORDS):
                    files["ct"] = f
                    break

        return files

    def _classify(self, filepath: Path, name: str, files: dict):
        if any(k in name for k in self._FUSED_KEYWORDS):
            files["vessel_fused"] = filepath
        elif any(k in name for k in self._LIVER_KEYWORDS):
            files["liver"] = filepath
        elif any(k in name for k in self._CT_KEYWORDS):
            files["ct"] = filepath
        elif any(k in name for k in self._VESSEL_KEYWORDS):
            files["vessels_individual"].append(filepath)


    def _resolve_raw_dicom(self, patient_dir: Path) -> dict:
        cache_dir = patient_dir / self._CACHE_DIRNAME
        cache_dir.mkdir(exist_ok=True)

        files = {
            "ct": None,
            "liver": None,
            "vessel_fused": None,
            "vessels_individual": [],
        }

        ct_dicom_dir = patient_dir / self._RAW_CT_DIR
        files["ct"] = self._cached_dicom_to_nifti(ct_dicom_dir, cache_dir / "ct.nii.gz")

        masks_dir = patient_dir / self._RAW_MASKS_DIR
        if masks_dir.is_dir():
            struct_dirs = sorted(d for d in masks_dir.iterdir() if d.is_dir())
            print(f"[IRCAD] MASKS_DICOM : {len(struct_dirs)} structure(s) trouvée(s) -> "
                  f"{[d.name for d in struct_dirs]}")

            for struct_dir in struct_dirs:
                name = struct_dir.name.lower()
                out_path = cache_dir / f"{struct_dir.name}.nii.gz"
                nii_path = self._cached_dicom_to_nifti(struct_dir, out_path, binarize=True)
                if nii_path is None:
                    print(f"[IRCAD]   - {struct_dir.name} : SKIP (conversion échouée)")
                    continue

                if name == self._RAW_LIVER_SUBDIR:
                    files["liver"] = nii_path
                    print(f"[IRCAD]   - {struct_dir.name} : LIVER (masque foie)")
                elif any(k in name for k in self._VESSEL_KEYWORDS):
                    files["vessels_individual"].append(nii_path)
                    print(f"[IRCAD]   - {struct_dir.name} : VESSEL (inclus dans la fusion)")
                else:
                    print(f"[IRCAD]   - {struct_dir.name} : ignoré (ne matche aucun mot-clé vaisseau)")

            if files["vessels_individual"]:
                names = [Path(p).stem for p in files["vessels_individual"]]
                print(f"[IRCAD] Fusion de {len(names)} structure(s) vasculaire(s) : {names}")
                files["vessel_fused"] = self._fuse_masks(
                    files["vessels_individual"], cache_dir / "vessels_fused.nii.gz"
                )
            else:
                print("[IRCAD] Aucune structure vasculaire retenue -> vessels_fused vide")

        return files

    def _cached_dicom_to_nifti(self, dicom_dir: Path, out_path: Path, binarize: bool = False) -> Path | None:
        if not dicom_dir.is_dir():
            return None
        if out_path.exists():
            return out_path

        try:
            image = self._read_dicom_series(dicom_dir)
        except Exception as exc:
            print(f"Warning: lecture DICOM impossible pour {dicom_dir} ({exc})")
            return None

        if binarize:
            array = sitk.GetArrayFromImage(image)
            array = (array > 0).astype(np.uint8)
            binary = sitk.GetImageFromArray(array)
            binary.CopyInformation(image)
            image = binary

        sitk.WriteImage(image, str(out_path))
        return out_path

    @staticmethod
    def _read_dicom_series(dicom_dir: Path) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if series_ids:
            file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        else:
            file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        if not file_names:
            raise RuntimeError("Aucun fichier DICOM trouvé dans le dossier")
        reader.SetFileNames(file_names)
        return reader.Execute()

    @staticmethod
    def _fuse_masks(mask_paths: list, out_path: Path) -> Path:

        fused = None
        reference = None
        for p in mask_paths:
            img = sitk.ReadImage(str(p))
            arr = sitk.GetArrayFromImage(img) > 0
            if fused is None:
                fused = arr
                reference = img
            else:
                fused = fused | arr

        fused_img = sitk.GetImageFromArray(fused.astype(np.uint8))
        fused_img.CopyInformation(reference)
        sitk.WriteImage(fused_img, str(out_path))
        return out_path
