import time
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

from resampler import Resampler
from patient_file_resolver import PatientFileResolver
from report_generator import ReportGenerator


class VesselPreprocessor:

    def __init__(self, input_dir: str | Path, output_dir: str | Path,
                 target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),apply_liver_mask: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_spacing = target_spacing

        self.resolver = PatientFileResolver(input_dir)
        self.resampler = Resampler(target_spacing)
        self.reporter = ReportGenerator(output_dir, target_spacing)
        self.results: list[dict] = []
        self.apply_liver_mask = apply_liver_mask


    def run(self, n_patients: int = 20):
        self._print_header()
        if self.apply_liver_mask:
            print("Vaisseaux DANS le foie uniquement")
        else:
            print("Tous les vaisseaux")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        patient_ids = self.resolver.get_patient_ids(n_patients)
        if not patient_ids:
            print("ERROR: Aucun patient trouvé. Vérifiez le dossier source.")
            return

        print(f"\n{len(patient_ids)} patient(s) trouvé(s).\n")

        for pid in tqdm(patient_ids, desc="Preprocessing"):
            print(f"\nPatient : {pid}")
            result = self._process_patient(pid)
            self.results.append(result)

            if result["status"] == "success":
                print(f"{result['time_seconds']:.2f} s")
            else:
                print(f"{result.get('error', 'Erreur inconnue')}")

        self.reporter.generate(self.results)


    def _process_patient(self, patient_id: str) -> dict:
        start = time.time()
        patient_dir = self.input_dir / patient_id

        if not patient_dir.exists():
            return {"patient_id": patient_id, "status": "failed",
                    "error": f"Dossier introuvable : {patient_dir}"}

        files = self.resolver.resolve(patient_id)

        if files["ct"] is None:
            return {"patient_id": patient_id, "status": "failed",
                    "error": "Aucune image CT trouvée"}

        out_dir = self.output_dir / patient_id
        out_dir.mkdir(exist_ok=True)
        if self.apply_liver_mask:
            suffix="in_liver"
        else:
            suffix="original"
        processed = {}

        # CT B-spline
        print(f"CT : {files['ct'].name}")
        ct_original = sitk.ReadImage(str(files["ct"]))
        ct_resampled = self.resampler.resample_image(ct_original)

        original_spacing = list(ct_original.GetSpacing())
        original_size = list(ct_original.GetSize())
        new_spacing = list(ct_resampled.GetSpacing())
        new_size = list(ct_resampled.GetSize())

        ct_out = out_dir / f"{patient_id}_ct_isotropic.nii.gz"
        sitk.WriteImage(ct_resampled, str(ct_out))
        processed["ct"] = str(ct_out)

        # Liver mask nearest neighbor, aligné sur le CT resamplé
        if files["liver"]:
            print(f"Liver : {files['liver'].name}")
            liver_out = out_dir / f"{patient_id}_liver_mask_isotropic.nii.gz"
            liver = sitk.ReadImage(str(files["liver"]))
            sitk.WriteImage(
                self.resampler.resample_mask(liver, reference_image=ct_resampled),
                str(liver_out),
            )
            processed["liver_mask"] = str(liver_out)

        # Vessel fused GT nearest neighbor, aligné sur le CT resamplé
        if files["vessel_fused"]:
            print(f"Vessels GT : {files['vessel_fused'].name}")
            vf_out = out_dir / f"{patient_id}_vessels_gt_isotropic_{suffix}.nii.gz"
            vf = sitk.ReadImage(str(files["vessel_fused"]))
            sitk.WriteImage(
                self.resampler.resample_mask(vf, reference_image=ct_resampled),
                str(vf_out),
            )
            processed["vessels_gt"] = str(vf_out)

        # Individual vessels nearest neighbor, alignés sur le CT resamplé
        processed["individual_vessels"] = []
        for vfile in files["vessels_individual"]:
            print(f"Vessel : {vfile.name}")
            stem = vfile.name.replace(".nii.gz", "").replace(".nii", "")
            v_out = out_dir / f"{patient_id}_{stem}__{suffix}_isotropic.nii.gz"
            v = sitk.ReadImage(str(vfile))
            sitk.WriteImage(
                self.resampler.resample_mask(v, reference_image=ct_resampled),
                str(v_out),
            )
            processed["individual_vessels"].append(str(v_out))

        return {
            "patient_id": patient_id,
            "status": "success",
            "version": "in_liver"if self.apply_liver_mask else "original",
            "original_spacing": original_spacing,
            "original_size": original_size,
            "new_spacing": new_spacing,
            "new_size": new_size,
            "processed_files": processed,
            "time_seconds": time.time() - start,
        }


    def _print_header(self):
        sep = "="* 80
        print(sep)
        print("VESSEL DATASET PREPROCESSING")
        print("Vesselness Filters: A Survey with Benchmarks (ICPR 2020)")
        print(sep)
        print(f"Input : {self.input_dir}")
        print(f"Output : {self.output_dir}")
        print(f"Spacing : {self.target_spacing} mm")
        print(sep)