import yaml
from pathlib import Path
from typing import Type, Union
from typing import get_origin, get_args
from dataclasses import is_dataclass, fields

from core.config.base import Config, ConfigBase

class ConfigBuilder:
    def __new__(cls, config_source: Union[str, Path, dict], config_type: Type[Config]) -> Config:
        config_dict = cls._get_config(config_source)
        return cls._parse_config(config_type, config_dict)

    @staticmethod
    def _get_config(config_source: Union[str, Path, dict]) -> dict:
        if isinstance(config_source, (str, Path)):
            path = Path(config_source)
            if not path.exists():
                raise FileNotFoundError(f"File {path} not found.")
            with path.open("r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise TypeError("Config must be initialized with: str, Path or dict.")
        return config_dict

    @staticmethod
    def _parse_config(dataclass_type: Type[ConfigBase], data: dict) -> ConfigBase:
        if not is_dataclass(dataclass_type):
            return data

        kwargs = {}
        for field in fields(dataclass_type):
            value = data.get(field.name)
            field_type = field.type

            if value is None:
                kwargs[field.name] = None
                continue

            origin = get_origin(field_type)
            args = get_args(field_type)

            # Handle Optional[...] or Union[...] where one type is a dataclass
            if origin is Union:
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    field_type = non_none_args[0]
                    origin = get_origin(field_type)
                    args = get_args(field_type)

            # Nested dataclass or subclass of ConfigBase
            if isinstance(value, dict) and (
                is_dataclass(field_type) or (isinstance(field_type, type) and issubclass(field_type, ConfigBase))
            ):
                value = ConfigBuilder._parse_config(field_type, value)

            kwargs[field.name] = value

        return dataclass_type(**kwargs)

