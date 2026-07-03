from dataclasses import dataclass
from typing import Callable

from numpy import ndarray


@dataclass(frozen=True)
class SFunction():
    function: Callable[[ndarray], ndarray]
    min_fitting_alpha: float
    max_fitting_alpha: float
