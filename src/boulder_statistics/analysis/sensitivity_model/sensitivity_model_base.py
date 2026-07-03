from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np
from polars import DataFrame

from boulder_statistics.analysis.sensitivity_model.s_function import SFunction


@dataclass(frozen=True)
class SensitivityModelBase(ABC):
    df: DataFrame
    J_threshold: float = field(default=0.7)

    @cached_property
    def best_S_function(self) -> SFunction:
        ...

    def random_S_function(
            self, rng: np.random.Generator) -> SFunction:
        ...
