from dataclasses import dataclass
from typing import List

from numpy import array, ndarray

from boulder_statistics.analysis.fit_params.general_fit_params import FitParams


@dataclass
class LogNormalFitParams(FitParams):
    mu: float
    sigma: float

    def to_numpy(self) -> ndarray:
        return array([self.mu, self.sigma])

    def modify_from_numpy(self, base: ndarray) -> None:
        self.mu, self.sigma = base

    def get_labels(self) -> List[str]:
        return ["mu", "sigma"]
