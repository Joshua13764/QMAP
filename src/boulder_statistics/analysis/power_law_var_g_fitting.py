from dataclasses import dataclass
from time import time

import numpy as np
from scipy.stats import norm

from boulder_statistics.analysis.fit_params.power_law_fit_params import \
    PowerLawFitParams
from boulder_statistics.analysis.fit_params.power_law_var_g_fit_params import \
    PowerLawVarGFitParams
from boulder_statistics.analysis.PSFD_fitting_base import PSFDFittingBase


@dataclass
class PowerLawVarGFitting(PSFDFittingBase[PowerLawVarGFitParams]):
    @property
    def a_min(self) -> np.float32:
        return np.pi * (self.LAD_min / 2) ** 2

    @property
    def a_max(self) -> np.float32:
        # return np.float32(self.cleaned_data.collect()["surface_area"].max())
        # * 1e6
        return self.cleaned_alphas_best_S.max() * 1e6

    def flat_PSFD_func(self, alphas, phis, phi_weights,
                       fit_params) -> np.ndarray:
        return (alphas ** (- fit_params.q - 1)) * self.sigma_sum(
            alphas=alphas,
            phis=phis,
            g_mu=fit_params.g_mu, g_std=fit_params.g_std,
            phi_weights=phi_weights,
            q=fit_params.q)

    def sigma_sum(self, alphas, phis, g_mu, g_std,
                  phi_weights, q) -> np.ndarray:

        cdf_weights = norm.cdf(
            (alphas[:, None] / phis[None, :]) / self.a_min,
            g_mu,
            g_std) if self.a_min != 0 else 1

        return np.sum(
            np.where(
                (alphas[:, None] / phis[None, :] < self.a_max),
                phi_weights[None, :] * phis[None, :] ** (q - 1),
                0
            ) * cdf_weights,
            axis=1
        )
