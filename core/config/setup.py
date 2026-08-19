from dataclasses import dataclass
from core.config.base import ConfigBase
from typing import Any, Optional, Union, List

@dataclass
class SetupConfig(ConfigBase):
    name: str 
    input_dir: str
    output_dir: str
    log_file: str
    debug_mode: bool
    plot_mode: bool
    save_mode: bool
    images_dir: str = "images"
    labels_dir: str = "labels"
    masks_dir: Optional[str] = None
    patient_ids: Optional[Union[str, int, List[Union[str, int]]]] = None