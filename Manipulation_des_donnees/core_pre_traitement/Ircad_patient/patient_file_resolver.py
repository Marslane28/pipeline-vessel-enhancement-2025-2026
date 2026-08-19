from pathlib import Path


class PatientFileResolver:
    """
    Parcourt le dossier d'un patient et catégorise ses fichiers NIfTI.

    Structure attendue :
        <input_dir>/
            patient01/
                *.nii or *.nii.gz   ← CT, liver, vessel_fused, vessels individuels
            patient02/
            ...
    """

    # Mots-clés pour identifier chaque type de fichier
    _CT_KEYWORDS      = ("ct", "scan")
    _LIVER_KEYWORDS   = ("liver",)
    _FUSED_KEYWORDS   = ("vessel_fused", "fused")
    _VESSEL_KEYWORDS  = ("vessel", "vein", "artery")
    _EXCLUDE_KEYWORDS = ("mask", "label")

    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)

    # ------------------------------------------------------------------

    def get_patient_ids(self, n_patients: int = 20) -> list[str]:
        """
        Retourne les identifiants des patients trouvés dans input_dir.
        Cherche patient01 … patient20 (avec fallback patient1 … patient20).
        """
        patients = []
        for i in range(1, n_patients + 1):
            pid = f"patient_{i:02d}"
            if (self.input_dir / pid).exists():
                patients.append(pid)
                continue
            pid_alt = f"patient_{i}"
            if (self.input_dir / pid_alt).exists():
                patients.append(pid_alt)
            else:
                print(f"Warning: {pid} introuvable dans {self.input_dir}")
        return sorted(patients)

    def resolve(self, patient_id: str) -> dict:
        """
        Retourne un dict avec les chemins catégorisés pour un patient :
            {
                'ct': Path | None,
                'liver': Path | None,
                'vessel_fused': Path | None,
                'vessels_individual': list[Path],
            }
        """
        patient_dir = self.input_dir / patient_id
        files = {
            "ct": None,
            "liver": None,
            "vessel_fused": None,
            "vessels_individual": [],
        }

        for f in sorted(patient_dir.glob("*.nii*")):
            name = f.name.lower()
            self._classify(f, name, files)

        # Fallback CT : premier fichier qui n'est pas un masque/label
        if files["ct"] is None:
            for f in sorted(patient_dir.glob("*.nii*")):
                if not any(k in f.name.lower() for k in self._EXCLUDE_KEYWORDS):
                    files["ct"] = f
                    break

        return files

    # ------------------------------------------------------------------

    def _classify(self, filepath: Path, name: str, files: dict):
        if any(k in name for k in self._FUSED_KEYWORDS):
            files["vessel_fused"] = filepath
        elif any(k in name for k in self._LIVER_KEYWORDS):
            files["liver"] = filepath
        elif any(k in name for k in self._CT_KEYWORDS):
            files["ct"] = filepath
        elif any(k in name for k in self._VESSEL_KEYWORDS):
            files["vessels_individual"].append(filepath)