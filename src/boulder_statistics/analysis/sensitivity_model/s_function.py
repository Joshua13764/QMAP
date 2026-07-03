from dataclasses import dataclass
from typing import Callable

from numpy import ndarray


@dataclass(frozen=True)
class SFunction():
    function: Callable[[ndarray], ndarray]
    min_fitting_alpha: float
    max_fitting_alpha: float

    @property
    def all_lods_max_fitting_alpha(self) -> float:
        return self.max_fitting_alpha * (2 ** (4 * 2))
