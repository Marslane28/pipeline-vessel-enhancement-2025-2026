from pathlib import Path


class BullittFileResolver:

    _VESSEL_KEYWORDS = (
        "binaryvessels",
        "binary_vessels",
        "vesselsiso_s",
        "binary",
        "vessel",
    )
    
    _BRAIN_MASK_KEYWORDS = ("brainmask", "brain_mask", "brain")
    
    _MRA_EXCLUDE_KEYWORDS = (
        "vessel", "mask", "dilated", "bifurcation", "rician",
        "binary", "label", "seg",
    )

    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)

    def get_patient_ids(self, n_patients: int | None = None) -> list[str]:
        patients = sorted(
            p.name for p in self.input_dir.iterdir()
            if p.is_dir() and "-MRA"in p.name
        )
        if n_patients:
            patients = patients[:n_patients]
        return patients

    def resolve(self, patient_id: str) -> dict:
        patient_dir = self.input_dir / patient_id
        files = {"mra": None, "vessel_mask": None, "brain_mask": None}

        all_files = sorted(patient_dir.glob("*.nii*"))

        # 1. Détection des masques
        for f in all_files:
            name = f.name.lower()
            
            # Vessel mask : binaryVesselsIso_S.nii.gz
            if files["vessel_mask"] is None and any(k in name for k in self._VESSEL_KEYWORDS):
                files["vessel_mask"] = f
                
            # Brain mask : brainMaskIso.nii
            if files["brain_mask"] is None and any(k in name for k in self._BRAIN_MASK_KEYWORDS):
                files["brain_mask"] = f

        # 2. MRA = tous les fichiers non exclus (méthode originale qui marchait)
        candidates = [
            f for f in all_files
            if f != files["vessel_mask"]
            and f != files["brain_mask"]
            and not any(k in f.name.lower() for k in self._MRA_EXCLUDE_KEYWORDS)
        ]
        
        if candidates:
            files["mra"] = candidates[0] # Prendre le premier

        if files["mra"] is None:
            print(f"{patient_id} : volume MRA non détecté")

        return files