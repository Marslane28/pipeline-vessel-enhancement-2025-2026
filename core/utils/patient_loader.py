from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
from logging import Logger
import re

from core.io.logger import setup_logger, close_logger
from configs.args import INPUT_DIR


class PatientLoader:
    """
    Chargeur de donnees pour les bases IRCAD, Bullitt et VascuSynth.

    Supporte quatre formats de nommage :
        - IRCAD    : patient_XX, patient_XX, ... (2 chiffres, .nii.gz)
        - Bullitt  : patient_XXX, patient_XXX, ... (3 chiffres, .nii.gz)

    Pour VascuSynth, supporte les fichiers generes par le pre-traitement :
        - images/GroupX_dataY/GroupX_dataY_rician_{sigma}.nii.gz
        - labels/GroupX_dataY/GroupX_dataY_vessels_gt.nii.gz
        - labels/GroupX_dataY/GroupX_dataY_bifurcations_gt.nii.gz
        - labels/GroupX_dataY/GroupX_dataY_vessel_neighborhood.nii.gz
    """

    # Niveaux de bruit pour VascuSynth
    VASCUSYNTH_NOISE_LEVELS = [5.0, 10.0, 20.0]

    def __init__(
        self,
        input_dir: str,
        logger: Logger = None,
        n_patients: int = None,
        dataset: str = "ircad",
        noise_level: Optional[float] = None,
        use_bifurcation_gt: bool = False,
        use_neighborhood_gt: bool = False,
    ):

        self.base_dir = Path(f"{INPUT_DIR}/{input_dir}")
        self.logger = logger or setup_logger()
        self.dataset = dataset.lower()
        self.noise_level = noise_level or 10.0
        self.use_bifurcation_gt = use_bifurcation_gt
        self.use_neighborhood_gt = use_neighborhood_gt

        self._last_resolved_pids = None

        # Configuration selon la base
        if self.dataset == "ircad":
            self.n_patients = n_patients or 20
            self.pid_format = "{:02d}"
            self.pid_pattern = re.compile(r"patient_(\d{2})")
            self.mask_suffix = "liver"
            self.image_ext = ".nii.gz"
            self.label_ext = ".nii.gz"
            self.structure_type = "flat"

        elif self.dataset == "bullitt":
            self.n_patients = n_patients or 33
            self.pid_format = "{:02d}"
            self.pid_pattern = re.compile(r"patient_(\d{3})")
            self.mask_suffix = "brain"
            self.image_ext = ".nii.gz"
            self.label_ext = ".nii.gz"
            self.structure_type = "flat"

        elif self.dataset == "vascusynth":
            self.n_patients = n_patients or 30
            self.pid_format = None
            self.pid_pattern = re.compile(r"Group(\d+)_data(\d+)")
            self.mask_suffix = None
            self.image_ext = ".nii.gz"
            self.label_ext = ".nii.gz"
            self.structure_type = "flat"

            self.allowed_data = (7, 9, 11)

        else:
            raise ValueError(
                f"Dataset '{dataset}' non reconnu. "
                f"Utilisez 'ircad', 'bullitt' ou 'vascusynth'."
            )

        self.logger.info(f"[PatientLoader] Dataset : {self.dataset.upper()}")
        self.logger.info(f"[PatientLoader] Patients attendus : {self.n_patients}")
        if self.dataset == "vascusynth":
            self.logger.info(f"[PatientLoader] Niveau de bruit : {self.noise_level}")
            self.logger.info(f"[PatientLoader] GT bifurcations : {'OUI' if self.use_bifurcation_gt else 'NON'}")
            self.logger.info(f"[PatientLoader] GT voisinage : {'OUI' if self.use_neighborhood_gt else 'NON'}")

        if not self.base_dir.exists():
            self.logger.warning(f"[PatientLoader] Dossier '{self.base_dir}' introuvable.")

    def get_patient_id(self, index: int) -> str:
        """Retourne l'ID du patient formate selon la base."""
        if self.dataset == "vascusynth":
            group = ((index - 1) // 3) + 1
            data_idx = (index - 1) % 3
            data = self.allowed_data[data_idx]
            return f"Group{group}_data{data}"
        return f"patient_{self.pid_format.format(index)}"

    def _resolve_patient_ids(
        self,
        patient_ids: Optional[Union[str, int, List[Union[str, int]]]],
        n_patients: Optional[int],
    ) -> List[str]:

        if patient_ids is None:
            n = n_patients or self.n_patients
            return [self.get_patient_id(i) for i in range(1, n + 1)]

        if not isinstance(patient_ids, list):
            patient_ids = [patient_ids]

        return [
            self.get_patient_id(pid) if isinstance(pid, int) else str(pid)
            for pid in patient_ids
        ]

    def get_files(
        self,
        images_dir: str = "images",
        labels_dir: str = "labels",
        n_patients: Optional[int] = None,
        patient_ids: Optional[Union[str, int, List[Union[str, int]]]] = None,
    ) -> Tuple[List[str], List[str]]:

        pids = self._resolve_patient_ids(patient_ids, n_patients)

        images_path = self.base_dir / images_dir
        labels_path = self.base_dir / labels_dir

        if not images_path.exists():
            raise FileNotFoundError(
                f"[PatientLoader] Dossier images introuvable : {images_path}"
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"[PatientLoader] Dossier labels introuvable : {labels_path}"
            )

        raw_files: List[str] = []
        gt_files: List[str] = []
        resolved_pids: List[str] = []
        missing: List[str] = []

        if self.dataset == "vascusynth":
            for pid in pids:
                img_filename = f"{pid}_rician_{self.noise_level:.1f}{self.image_ext}"
                img = images_path / pid / img_filename

                if not img.exists():
                    rician_files = list((images_path / pid).glob(f"{pid}_rician_*.nii.gz"))
                    if rician_files:
                        img = rician_files[0]
                        self.logger.debug(f"[PatientLoader] Utilisation de {img.name} pour {pid}")
                    else:
                        missing.append(f"Image manquante : {pid}")
                        continue

                if self.use_bifurcation_gt:
                    label_filename = f"{pid}_bifurcations_gt{self.label_ext}"
                elif self.use_neighborhood_gt:
                    label_filename = f"{pid}_vessel_neighborhood{self.label_ext}"
                else:
                    label_filename = f"{pid}_vessels_gt{self.label_ext}"

                label = labels_path / pid / label_filename

                if not label.exists():
                    missing.append(f"Label manquant : {pid}")
                    continue

                raw_files.append(str(img.relative_to(self.base_dir)))
                gt_files.append(str(label.relative_to(self.base_dir)))
                resolved_pids.append(pid)

        else:
            for pid in pids:
                img = images_path / f"{pid}_images{self.image_ext}"
                label = labels_path / f"{pid}_label{self.label_ext}"

                if not img.exists():
                    missing.append(f"Image manquante : {img}")
                    continue
                if not label.exists():
                    missing.append(f"Label manquant : {label}")
                    continue

                raw_files.append(f"{images_dir}/{pid}_images{self.image_ext}")
                gt_files.append(f"{labels_dir}/{pid}_label{self.label_ext}")
                resolved_pids.append(pid)

        if missing:
            self.logger.warning(
                f"[PatientLoader] {len(missing)} fichier(s) manquant(s) :"
            )
            for m in missing[:5]:
                self.logger.warning(f"  {m}")
            if len(missing) > 5:
                self.logger.warning(f"  ... et {len(missing) - 5} autres")

        if not raw_files:
            raise ValueError(
                f"[PatientLoader] Aucun patient complet trouve parmi : {pids}"
            )

        self.logger.info(
            f"[PatientLoader] {len(raw_files)}/{len(pids)} patient(s) charge(s)."
        )

        self._last_resolved_pids = resolved_pids

        return raw_files, gt_files

    def get_mask_files(
        self,
        masks_dir: str = "masks",
        n_patients: Optional[int] = None,
        patient_ids: Optional[Union[str, int, List[Union[str, int]]]] = None,
    ) -> List[Optional[str]]:
        if patient_ids is not None:
            pids = self._resolve_patient_ids(patient_ids, n_patients)
        elif getattr(self, "_last_resolved_pids", None) is not None:
            pids = self._last_resolved_pids
        else:
            n = n_patients or self.n_patients
            pids = [self.get_patient_id(i) for i in range(1, n + 1)]

        if self.mask_suffix is None:
            self.logger.info(
                f"[PatientLoader] {self.dataset.upper()} : pas de masque d'organe utilise."
            )
            return [None] * len(pids)

        masks_path = self.base_dir / masks_dir
        mask_files: List[Optional[str]] = []

        if not masks_path.exists():
            self.logger.warning(
                f"[PatientLoader] Dossier masques introuvable : {masks_path} "
                f"-> evaluation sans masques."
            )
            return [None] * len(pids)

        found = 0
        for pid in pids:
            mask = masks_path / f"{pid}_{self.mask_suffix}.nii.gz"

            if not mask.exists():
                mask_files.append(None)
            else:
                mask_files.append(f"{masks_dir}/{pid}_{self.mask_suffix}.nii.gz")
                found += 1

        self.logger.info(
            f"[PatientLoader] {found}/{len(pids)} masque(s) charge(s) pour {self.dataset.upper()}."
        )

        return mask_files

    def get_vascusynth_noise_levels(self, patient_id: str) -> List[float]:
        """
        Pour VascuSynth : retourne les niveaux de bruit disponibles pour un patient.
        """
        images_path = self.base_dir / "images" / patient_id
        if not images_path.exists():
            return []

        rician_files = list(images_path.glob(f"{patient_id}_rician_*.nii.gz"))
        levels = []
        for f in rician_files:
            match = re.search(r"rician_([\d.]+)", f.name)
            if match:
                levels.append(float(match.group(1)))

        return sorted(levels)

    def get_patient_info(self) -> dict:
        """
        Retourne les informations sur les patients disponibles.
        """
        images_path = self.base_dir / "images"
        labels_path = self.base_dir / "labels"

        info = {
            "dataset": self.dataset.upper(),
            "expected_patients": self.n_patients,
            "available_images": 0,
            "available_labels": 0,
            "patients": [],
            "details": {},
        }

        if not images_path.exists():
            return info

        if self.dataset == "vascusynth":
            for i in range(1, self.n_patients + 1):
                pid = self.get_patient_id(i)
                patient_dir = images_path / pid

                if not patient_dir.exists():
                    continue

                rician_files = list(patient_dir.glob(f"{pid}_rician_*.nii.gz"))
                noise_levels = []
                for f in rician_files:
                    match = re.search(r"rician_([\d.]+)", f.name)
                    if match:
                        noise_levels.append(float(match.group(1)))

                if not noise_levels:
                    continue

                info["available_images"] += len(noise_levels)

                label_dir = labels_path / pid
                has_vessels = (label_dir / f"{pid}_vessels_gt.nii.gz").exists()
                has_bifurcations = (label_dir / f"{pid}_bifurcations_gt.nii.gz").exists()
                has_neighborhood = (label_dir / f"{pid}_vessel_neighborhood.nii.gz").exists()

                if has_vessels:
                    info["available_labels"] += 1

                info["patients"].append(pid)
                info["details"][pid] = {
                    "noise_levels": sorted(noise_levels),
                    "has_vessels_gt": has_vessels,
                    "has_bifurcations_gt": has_bifurcations,
                    "has_vessel_neighborhood": has_neighborhood,
                }
        else:
            for i in range(1, self.n_patients + 1):
                pid = self.get_patient_id(i)
                img = images_path / f"{pid}_images{self.image_ext}"
                label = labels_path / f"{pid}_label{self.label_ext}" if labels_path.exists() else None

                has_img = img.exists()
                has_label = label.exists() if label is not None else False

                if has_img:
                    info["available_images"] += 1
                if has_label:
                    info["available_labels"] += 1
                if has_img and has_label:
                    info["patients"].append(pid)

        return info

    def print_vascusynth_summary(self) -> None:
        """
        Affiche un resume detaille pour VascuSynth.
        """
        if self.dataset != "vascusynth":
            self.logger.warning("Cette methode est specifique a VascuSynth.")
            return

        info = self.get_patient_info()
        print("\n" + "=" * 80)
        print("VASCUSYNTH DATASET SUMMARY")
        print("=" * 80)
        print(f"Patients disponibles : {len(info['patients'])}/{info['expected_patients']}")
        print(f"Images disponibles  : {info['available_images']}")
        print(f"Labels disponibles  : {info['available_labels']}")
        print("\nDetails par patient:")
        print("-" * 60)
        print("\n" + "=" * 80)

    @staticmethod
    def patient_id_from_path(filepath: str) -> str:
        """Extrait l'ID du patient depuis un chemin de fichier."""
        path = Path(filepath)
        name = path.stem

        suffixes = ["_images", "_label", "_liver", "_mask", "_image", "_vessels_gt",
                    "_bifurcations_gt", "_vessel_neighborhood", "_rician_5.0",
                    "_rician_10.0", "_rician_20.0"]

        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        return name

    def detect_dataset(self, base_dir: Path = None) -> str:
        """
        Detecte automatiquement la base de donnees a partir des fichiers presents.
        """
        base = base_dir or self.base_dir
        images_path = base / "images"

        if not images_path.exists():
            return "unknown"

        vascusynth_files = list(images_path.glob("Group*_data*"))
        if vascusynth_files:
            return "vascusynth"

        for subdir in images_path.iterdir():
            if subdir.is_dir() and subdir.name.startswith("Group") and "_data" in subdir.name:
                return "vascusynth"

        files = list(images_path.glob("patient_*.nii*"))
        if not files:
            return "unknown"

        for f in files:
            name = f.stem
            if "_images" in name:
                pid = name.replace("_images", "")
                match = re.search(r"patient_(\d+)", pid)
                if match:
                    digits = len(match.group(1))
                    if digits == 3:
                        return "bullitt"
                    elif digits == 2:
                        return "ircad"

        return "unknown"


def create_patient_loader(
    input_dir: str,
    logger: Logger = None,
    dataset: str = None,
    n_patients: int = None,
    noise_level: float = 10.0,
    use_bifurcation_gt: bool = False,
    use_neighborhood_gt: bool = False,
) -> PatientLoader:
    """
    Cree un PatientLoader avec detection automatique du dataset.
    """
    temp_loader = PatientLoader(
        input_dir=input_dir,
        logger=logger,
        n_patients=1,
        dataset="ircad"
    )

    if dataset is None:
        dataset = temp_loader.detect_dataset()
        if dataset == "unknown":
            if logger:
                logger.warning("[PatientLoader] Dataset non detecte, utilisation de 'ircad' par defaut.")
            dataset = "ircad"

    if n_patients is None:
        n_patients = {
            "ircad": 20,
            "bullitt": 33,
            "vascusynth": 30
        }.get(dataset, 20)

    return PatientLoader(
        input_dir=input_dir,
        logger=logger,
        n_patients=n_patients,
        dataset=dataset,
        noise_level=noise_level,
        use_bifurcation_gt=use_bifurcation_gt,
        use_neighborhood_gt=use_neighborhood_gt,
    )