from dataclasses import dataclass

import numpy as np

from boulder_statistics.analysis.fit_params.log_normal_fit_params import \
    LogNormalFitParams
from boulder_statistics.analysis.fit_params.power_law_fit_params import \
    PowerLawFitParams
from boulder_statistics.analysis.fit_params.weibull_fit_params import \
    WeibullFitParams
from boulder_statistics.analysis.quick_calculate_general import \
    GeneralPSFDFittingFunction


@dataclass(frozen=True)
class LogNormalFittingFunction(
    GeneralPSFDFittingFunction[LogNormalFitParams]
):
    def flat_PSFD_func(
        self,
        alphas,
        phis,
        phi_weights,
        fit_params,
    ) -> np.ndarray:
        return (
            1.0
            / (alphas * fit_params.sigma * np.sqrt(2.0 * np.pi))
            * self.CFS_sum(
                alphas=alphas,
                phis=phis,
                phi_weights=phi_weights,
                fit_params=fit_params,
            )
        )

    def CFS_sum(
        self,
        alphas,
        phis,
        phi_weights,
        fit_params: LogNormalFitParams,
    ) -> np.ndarray:
        return np.sum(
            phi_weights[None, :]
            * np.exp(
                -(
                    (
                        np.log(alphas[:, None] / phis[None, :])
                        - fit_params.mu
                    ) ** 2
                )
                / (2.0 * fit_params.sigma**2)
            ),
            axis=1,
        )
