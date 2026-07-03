from dataclasses import dataclass
from typing import List

from numpy import array, ndarray

from boulder_statistics.analysis.fit_params.general_fit_params import FitParams


@dataclass
class WeibullFitParams(FitParams):
    lambda_: float
    k: float

    def to_numpy(self) -> ndarray:
        return array([self.lambda_, self.k])

    def modify_from_numpy(self, base: ndarray) -> None:
        self.lambda_, self.k = base

    def get_labels(self) -> List[str]:
        return ["lambda", "k"]
