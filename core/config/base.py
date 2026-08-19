from typing import Any, TypeVar
from dataclasses import dataclass, asdict, fields


@dataclass
class ConfigBase:
    def __getattr__(self, key: str) -> Any:
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(f"Attribut '{key}' not found in {self.__class__.__name__}")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def to_dict(self) -> dict:
        return asdict(self)
    
    def from_dict(self, data: dict) -> None:
        field_names = {f.name for f in fields(self)}
        for key, value in data.items():
            if key in field_names:
                setattr(self, key, value)

    def keys(self):
        return self.to_dict().keys()
    
    def items(self):
        return self.to_dict().items()
    
    def values(self):
        return self.to_dict().values()

Config = TypeVar("Config", bound=ConfigBase)