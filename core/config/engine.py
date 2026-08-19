from dataclasses import dataclass
from core.config.base import ConfigBase


### ENGINE

@dataclass
class SSIConfig(ConfigBase):
    run: bool
    volume_size: int
    scales_range: tuple[int]
    scales_numbers: list
    
@dataclass
class VSIConfig(ConfigBase):
    run: bool
    volume_sizes: list[int]
    chunk_number: int
    
    
@dataclass
class CNIConfig(ConfigBase):
    run: bool
    volume_sizes: list[int]
    chunk_numbers: list[int]
    
@dataclass
class PACConfig(ConfigBase):
    run: bool
    input_file: str


@dataclass
class EngineConfig(ConfigBase):
    ssi: SSIConfig
    vsi: VSIConfig
    cni: CNIConfig
    pac: PACConfig
    
    