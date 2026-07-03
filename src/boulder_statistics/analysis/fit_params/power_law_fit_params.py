from dataclasses import dataclass
from typing import List

from numpy import array, ndarray

from boulder_statistics.analysis.fit_params.general_fit_params import FitParams


@dataclass
class PowerLawFitParams(FitParams):
    q: float
    g: float

    def to_numpy(self) -> ndarray:
        return array([self.q, self.g])

    def modify_from_numpy(self, base: ndarray) -> None:
        self.q, self.g = base

    def get_labels(self) -> List[str]:
        return ["q", "g"]
