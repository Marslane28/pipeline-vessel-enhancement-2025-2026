"""
Chargement et accès à la configuration YAML du pipeline.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        # Remonte deux niveaux depuis src/utils/ jusqu'à la racine du projet
        config_path = Path(__file__).resolve().parents[2] / "data_config" / "reglage_data" / "data.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """
    Accès sécurisé à une valeur imbriquée dans le dictionnaire de config.

    Exemple
    -------
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current