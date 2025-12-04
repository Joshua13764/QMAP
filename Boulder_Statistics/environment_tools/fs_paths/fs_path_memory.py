from typing import Any

import attr

from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@attr.define(frozen=True, slots=True)
class FSPathMemory(FSPathBase):
    obj: Any = None
