from dataclasses import dataclass
from time import time

import numpy as np

from boulder_statistics.analysis.power_law_fit_params import PowerLawFitParams
from boulder_statistics.analysis.quick_calculate_general import \
    GeneralPSFDFittingFunction
from boulder_statistics.analysis.weibull_fit_params import WeibullFitParams


@dataclass(frozen=True)
class WeibullFittingFunction(GeneralPSFDFittingFunction[WeibullFitParams]):
    LAD_min = 0

    def flat_PSFD_func(self, alphas, phis, phi_weights,
                       fit_params) -> np.ndarray:
        return (alphas ** (fit_params.k - 1)) * self.CFS_sum(
            alphas=alphas,
            phis=phis,
            phi_weights=phi_weights,
            fit_params=fit_params)

    def CFS_sum(self, alphas, phis, phi_weights,
                fit_params: WeibullFitParams) -> np.ndarray:
        return np.sum(
            phi_weights[None, :] *
            phis[None, :] ** (-fit_params.k) *
            np.exp(
                -(
                    alphas[:, None]
                    / (fit_params.lambda_ * phis[None, :])
                ) ** fit_params.k
            ),
            axis=1,
        )
