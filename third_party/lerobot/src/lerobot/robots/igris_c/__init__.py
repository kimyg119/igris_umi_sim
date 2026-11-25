# [wjson] Igris 코드 추가
from .config_igris_c import IgrisCConfig, IgrisCUpperConfig
from .igris_c import IgrisC
from .umi import IgrisCUmi, IgrisCUmiConfig

__all__ = ["IgrisC", "IgrisCConfig", "IgrisCUpperConfig", "IgrisCUmi", "IgrisCUmiConfig"]
