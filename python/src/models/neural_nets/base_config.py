import dataclasses

from src.utils.base_config import YamlConfig


@dataclasses.dataclass()
class BaseModelConfig(YamlConfig):
    model_name: str
