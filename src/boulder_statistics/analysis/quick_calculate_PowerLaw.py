from dataclasses import dataclass
from time import time

import numpy as np

from boulder_statistics.analysis.power_law_fit_params import PowerLawFitParams
from boulder_statistics.analysis.quick_calculate_general import \
    GeneralPSFDFittingFunction


@dataclass(frozen=True)
class PowerLawFittingFunction(GeneralPSFDFittingFunction[PowerLawFitParams]):
    @property
    def a_min(self) -> np.float32:
        return np.pi * (self.LAD_min) ** 2

    @property
    def a_max(self) -> np.float32:
        # return np.float32(self.cleaned_data.collect()["surface_area"].max())
        # * 1e6
        return np.float32(self.cleaned_data.collect()[
                          "surface_area"].max()) * 1e6

    def flat_PSFD_func(self, alphas, phis, phi_weights,
                       fit_params) -> np.ndarray:
        return (alphas ** (- fit_params.q - 1)) * self.sigma_sum(
            alphas=alphas,
            phis=0.1 * fit_params.g * phis,
            phi_weights=phi_weights,
            q=fit_params.q)

    def sigma_sum(self, alphas, phis, phi_weights, q) -> np.ndarray:
        return np.sum(np.where(
            (self.a_min < alphas[:, None] /
             phis[None, :]) & (alphas[:, None] /
                               phis[None, :] < self.a_max),
            phi_weights[None, :] * phis[None, :] ** (q - 1),
            0
        ), axis=1)
