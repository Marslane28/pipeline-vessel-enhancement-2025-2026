from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.config.base import Config, ConfigBase


# Blocs partagés

@dataclass
class GeometryConfig(ConfigBase):
    """Géométrie du fantôme analytique (AnalyticalVessel)."""
    shape: List[int] # taille du volume
    sigma_vessel: float # largeur gaussienne du vaisseau (voxels)
    vessel_radius: float # rayon de troncature


@dataclass
class MethodsConfig(ConfigBase):
    """Opérateurs de dérivation à comparer."""
    custom: List[str] 
    skimage: List[str] 


@dataclass
class NoiseConfig(ConfigBase):
    snr_db: float


@dataclass
class TimingConfig(ConfigBase):
    n_repeats: int
    warmup: int


# 1. Pour Comparaison_hessian.py

@dataclass
class HessianStudyConfig(ConfigBase):
    mode: str # "hessian"(informatif)
    results_dir: str
    seed: int
    verbose: bool
    geometry: GeometryConfig
    methods: MethodsConfig
    cases: Dict[str, str] # nom_cas -> méthode AnalyticalVessel (ex. "simple": "simple_vessel")
    noise: NoiseConfig
    timing: TimingConfig
    plot: bool = True


# 2. Pour Comparaison_valeurs_propres.py

@dataclass
class EigenvalueStudyConfig(ConfigBase):
    mode: str # "eigenvalues"
    results_dir: str
    seed: int
    verbose: bool
    geometry: GeometryConfig
    methods: MethodsConfig
    cases: Dict[str, str]
    noise: NoiseConfig
    n_noise_realizations: int
    plot: bool = True


# 3. Pour Comparaison_vesselness.py

@dataclass
class VesselnessStudyConfig(ConfigBase):
    mode: str
    results_dir: str
    seed: int
    verbose: bool
    geometry: GeometryConfig
    methods: MethodsConfig
    cases: Dict[str, str]
    scales: List[float]
    filters_to_test: List[str]
    black_ridges: bool = False
    plot: bool = True


# 4. Pour Comparaison_segmentation.py

@dataclass
class SegmentationStudyConfig(ConfigBase):
    mode: str # "segmentation"
    results_dir: str
    seed: int
    verbose: bool
    geometry: GeometryConfig
    methods: MethodsConfig
    cases: Dict[str, str]
    scales: List[float]
    filters_to_test: List[str]
    black_ridges: bool = False
    skip_bifurcation: bool = False
    plot: bool = True
