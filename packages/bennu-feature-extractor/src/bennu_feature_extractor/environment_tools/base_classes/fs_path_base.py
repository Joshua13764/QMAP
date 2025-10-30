from abc import ABC
from typing import Tuple

import attr


@attr.define(frozen=True, slots=True)
class FSPathBase(ABC):
    path: Tuple[str, ...]
