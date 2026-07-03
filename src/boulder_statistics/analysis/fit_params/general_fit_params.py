from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class FitParams(ABC):
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        ...

    @abstractmethod
    def modify_from_numpy(self, base: np.ndarray) -> None:
        ...

    @abstractmethod
    def get_labels(self) -> List[str]:
        ...
