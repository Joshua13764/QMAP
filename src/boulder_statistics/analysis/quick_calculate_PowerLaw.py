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
        # Reconstruct bin edges from geometric centres
        log_phis = np.log(phis)

        log_edges = np.empty(len(phis) + 1)
        log_edges[1:-1] = 0.5 * (log_phis[:-1] + log_phis[1:])
        log_edges[0] = 2 * log_phis[0] - log_edges[1]
        log_edges[-1] = 2 * log_phis[-1] - log_edges[-2]

        log_left = log_edges[:-1]
        log_right = log_edges[1:]
        log_width = log_right - log_left

        # Window in log(phi)
        log_min = np.log(alphas[:, None] / self.a_max)
        log_max = np.log(alphas[:, None] / self.a_min)

        overlap = (
            np.minimum(log_max, log_right[None, :])
            - np.maximum(log_min, log_left[None, :])
        )

        fraction = np.clip(overlap / log_width[None, :], 0.0, 1.0)

        return np.sum(
            fraction
            * phi_weights[None, :]
            * phis[None, :] ** (q - 1),
            axis=1,
        )

    # def sigma_sum(self, alphas, phis, phi_weights, q) -> np.ndarray:
    #     return np.sum(np.where(
    #         (self.a_min < alphas[:, None] /
    #          phis[None, :]) & (alphas[:, None] /
    #                            phis[None, :] < self.a_max),
    #         phi_weights[None, :] * phis[None, :] ** (q - 1),
    #         0
    #     ), axis=1)
