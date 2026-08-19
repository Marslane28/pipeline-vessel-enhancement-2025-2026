from dataclasses import dataclass, field
from typing import List, Optional

from core.config.base import ConfigBase


@dataclass
class PostBenchTestsDatasetConfig(ConfigBase):
    # Chemins
    enhancer_dir: str
    labels_dir: str
    masks_dir: Optional[str] = None  # None pour VascuSynth (pas de masque d'organe)

    # Naming des dossiers/fichiers par patient (identique à DatasetConfig)
    case_prefix: str = ""
    case_suffix: str = ""
    result_prefix: str = ""
    result_suffix: str = ""

    # Liste des patients ("01".."NN")  vide pour VascuSynth, détectée au
    patients: List[str] = field(default_factory=list)
    has_subdirs: bool = False