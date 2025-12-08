from dataclasses import dataclass
from typing import Any

from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@dataclass(frozen=True)
class FSPathMemory(FSPathBase):
    obj: Any = None
