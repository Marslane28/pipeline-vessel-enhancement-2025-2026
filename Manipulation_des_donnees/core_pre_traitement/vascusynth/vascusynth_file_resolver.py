from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class VascuSynthCase:
    """Référence vers un cas VascuSynth unique (un groupe / un data)."""

    group: int
    data: int
    image_path: Path
    tree_mat_path: Path

    @property
    def case_id(self) -> str:
        return f"Group{self.group}_data{self.data}"


class VascuSynthFileResolver:
    """Localise les fichiers .mhd/.mat pour chaque cas VascuSynth."""

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)

    def get_case_ids(
        self,
        groups: Optional[List[int]] = None,
        n_data_per_group: int = 12,
    ) -> List[Tuple[int, int]]:
        """Retourne la liste des (group, data) réellement présents sur disque.

        Si `groups` est None, les groupes sont auto-détectés en scannant les
        dossiers `Group*` de `input_dir`.
        """
        if groups is None:
            groups = sorted(
                int(p.name.replace("Group", ""))
                for p in self.input_dir.glob("Group*")
                if p.is_dir() and p.name.replace("Group", "").isdigit()
            )

        case_ids: List[Tuple[int, int]] = []
        for g in groups:
            group_dir = self.input_dir / f"Group{g}"
            if not group_dir.exists():
                continue
            for d in range(1, n_data_per_group + 1):
                if (group_dir / f"data{d}").exists():
                    case_ids.append((g, d))
        return case_ids

    def resolve(self, group: int, data: int) -> Optional[VascuSynthCase]:
        """Retourne le cas résolu (chemins .mhd + .mat), ou None si introuvable."""
        data_dir = self.input_dir / f"Group{group}" / f"data{data}"
        if not data_dir.exists():
            return None

        mhd_files = sorted(data_dir.glob("*.mhd"))
        mat_files = sorted(data_dir.glob("*.mat"))

        if not mhd_files or not mat_files:
            return None

        if len(mhd_files) > 1:
            # Plusieurs .mhd dans le dossier : on privilégie celui dont le nom
            # contient explicitement le numéro du data (ex: "VascuSynth3_...").
            preferred = [f for f in mhd_files if f"{data}_" in f.stem or f.stem.endswith(str(data))]
            mhd_files = preferred or mhd_files

        if len(mat_files) > 1:
            preferred = [f for f in mat_files if f.stem.endswith(str(data))]
            mat_files = preferred or mat_files

        return VascuSynthCase(
            group=group,
            data=data,
            image_path=mhd_files[0],
            tree_mat_path=mat_files[0],
        )