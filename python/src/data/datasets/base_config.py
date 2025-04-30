import dataclasses

from src.utils.base_config import YamlConfig


@dataclasses.dataclass()
class BaseConfigDataset(YamlConfig):
    dir_paths: list[str]
