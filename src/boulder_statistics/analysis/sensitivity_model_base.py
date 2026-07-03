from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np
from polars import DataFrame


@dataclass(frozen=True)
class SensitivityModelBase(ABC):
    df: DataFrame
    J_threshold: float = field(default=0.7)

    @cached_property
    def best_p_function(self) -> Callable[[np.ndarray], np.ndarray]:
        ...

    def random_p_function(
            self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        ...

    @cached_property
    def min_fitting_alpha(self) -> float:
        ...

    @cached_property
    def max_fitting_alpha(self) -> float:
        ...
