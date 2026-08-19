from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from core.config.base import ConfigBase

_DEFAULT_OPERATORS = ['default', 'gaussian', 'farid', 'cubic', 'trigonometric',
                       'catmull', 'bspline', 'bezier', 'scharr']
_DEFAULT_MIN_SIZES = [0, 10, 50, 100, 200]


@dataclass
class CCFilteringConfig(ConfigBase):
    dataset: str
    results_dir: str
    images_dir: str
    labels_dir: str
    output_dir: str
    masks_dir: Optional[str] = None

    operators: Optional[List[str]] = None
    min_sizes: Optional[List[int]] = None
    generate_figure: bool = True

    # bullitt / ircad
    patients: Optional[List[str]] = None

    # vascusynth uniquement
    noise_level: float = 10.0
    groups: Optional[List[int]] = None
    data_ids: Optional[List[int]] = None

    def __post_init__(self):
        if self.operators is None:
            self.operators = list(_DEFAULT_OPERATORS)
        if self.min_sizes is None:
            self.min_sizes = list(_DEFAULT_MIN_SIZES)

        self.results_dir = Path(self.results_dir)
        self.images_dir = Path(self.images_dir)
        self.labels_dir = Path(self.labels_dir)
        self.masks_dir = Path(self.masks_dir) if self.masks_dir else None
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_csv(self) -> Path:
        return self.output_dir / "metrics_par_min_size.csv"

    def patient_list(self) -> list:
        from configs.args import DATASET_N_PATIENTS

        if self.dataset == "vascusynth":
            groups = self.groups or list(range(1, 11))
            data_ids = self.data_ids or [7, 9, 11]
            return [f"Group{g}_data{d}" for g in groups for d in data_ids]

        if self.patients is not None:
            return self.patients

        n = DATASET_N_PATIENTS.get(self.dataset, 20)
        return [f"{i:02d}" for i in range(1, n + 1)]