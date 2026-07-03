from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Literal, Tuple

import numpy as np
import polars as pl
import scipy
import scipy.interpolate
from polars import DataFrame
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

from boulder_statistics.analysis.sensitivity_model_base import \
    SensitivityModelBase


@dataclass(frozen=True)
class KDEBootstrappedSensitivityModel(SensitivityModelBase):
    df: DataFrame
    J_threshold: float = field(default=0.75)
    min_alpha_s_min: float = field(default=0.0)
    max_alpha_s_min: float = field(default=0.0)
    bandwidth: float | Literal['scott', 'silverman'] = field(default="scott")
    kernel: Literal['gaussian',
                    'tophat',
                    'epanechnikov',
                    'exponential',
                    'linear',
                    'cosine'] = field(default="epanechnikov")

    @cached_property
    def log_totals_passed(self) -> Tuple[np.ndarray, np.ndarray]:
        db_jaccard_agg_filter: DataFrame = self.df.filter(
            pl.col("viewport_size") > 0)

        alphas: np.ndarray = db_jaccard_agg_filter["viewport_size"].to_numpy()
        j: np.ndarray = db_jaccard_agg_filter["Jaccard_index"].to_numpy()

        alphas_pass: np.ndarray = alphas[j > self.J_threshold]

        log_alphas: np.ndarray = np.log(alphas)
        log_alphas_pass: np.ndarray = np.log(alphas_pass)

        return log_alphas, log_alphas_pass

    def get_s_KDE(self, bootstrap_rng: np.random.Generator |
                  None = None) -> Callable[[np.ndarray], np.ndarray]:
        log_alphas, log_alphas_pass = self.log_totals_passed

        if bootstrap_rng is not None:
            log_alphas: np.ndarray = bootstrap_rng.choice(
                log_alphas,
                size=len(log_alphas),
                replace=True
            )

            log_alphas_pass: np.ndarray = bootstrap_rng.choice(
                log_alphas_pass,
                size=len(log_alphas_pass),
                replace=True
            )

        kde_log_alphas = KernelDensity(
            kernel=self.kernel,
            bandwidth="scott"
        ).fit(log_alphas[:, None])

        kde_log_alphas_pass = KernelDensity(
            kernel=self.kernel,
            bandwidth="scott"
        ).fit(log_alphas_pass[:, None])

        def log_alphas_kde(alpha) -> np.ndarray:
            return (
                len(log_alphas)
                * np.exp(kde_log_alphas.score_samples(np.log(alpha).reshape(-1, 1)))
                / alpha
            )

        def log_alphas_pass_kde(alpha) -> np.ndarray:
            return (
                len(log_alphas_pass)
                * np.exp(kde_log_alphas_pass.score_samples(np.log(alpha).reshape(-1, 1)))
                / alpha
            )

        sample_alphas = np.geomspace(1, 512 ** 2, 10_000)
        sample_res = np.divide(
            log_alphas_pass_kde(sample_alphas),
            log_alphas_kde(sample_alphas),
            out=np.zeros_like(sample_alphas, dtype=float),
            where=log_alphas_kde(sample_alphas) != 0
        )

        interp_func = scipy.interpolate.interp1d(
            np.log(sample_alphas),
            sample_res,
            kind="linear",
            bounds_error=False,
            fill_value=0.0
        )

        return lambda input_alphas: interp_func(np.log(input_alphas))

    @cached_property
    def best_p_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.get_s_KDE()

    def random_p_function(
            self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        return self.get_s_KDE(bootstrap_rng=rng)

    @cached_property
    def min_fitting_alpha(self) -> float:
        alpha_samples: np.ndarray = np.geomspace(1, 512 ** 2, 10_000)
        S_samples: np.ndarray = self.best_p_function(alpha_samples)

        return alpha_samples[np.flatnonzero(
            S_samples > self.max_alpha_s_min)[0]].item()

    @cached_property
    def max_fitting_alpha(self) -> float:
        alpha_samples: np.ndarray = np.geomspace(1, 512 ** 2, 10_000)
        S_samples: np.ndarray = self.best_p_function(alpha_samples)

        return alpha_samples[np.flatnonzero(
            S_samples > self.min_alpha_s_min)[-1]].item()
