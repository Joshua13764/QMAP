from dataclasses import dataclass
from typing import List

from numpy import array, ndarray

from boulder_statistics.analysis.fit_params.general_fit_params import FitParams


@dataclass
class PowerLawVarGFitParams(FitParams):
    q: float
    g_mu: float
    g_std: float

    def to_numpy(self) -> ndarray:
        return array([self.q, self.g_mu, self.g_std])

    def modify_from_numpy(self, base: ndarray) -> None:
        self.q, self.g_mu, self.g_std = base

    def get_labels(self) -> List[str]:
        return ["q", "g_mu", "g_std"]
