from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Tuple

import numpy as np
import polars as pl
from polars import DataFrame
from scipy.stats import gaussian_kde

from boulder_statistics.analysis.sensitivity_model_base import \
    SensitivityModelBase


@dataclass(frozen=True)
class KDEBootstrappedSensitivityModel(SensitivityModelBase):
    df: DataFrame
    J_threshold: float = field(default=0.7)
    min_alpha_s_min: float = 0.01
    max_alpha_s_min: float = 0.01

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

    def get_s_KDE(self, input_alphas: np.ndarray,
                  bootstrap_rng: np.random.Generator | None = None) -> np.ndarray:
        log_alphas, log_alphas_pass = self.log_totals_passed

        if bootstrap_rng is not None:
            log_alphas_pass: np.ndarray = bootstrap_rng.choice(
                log_alphas_pass,
                size=len(log_alphas_pass),
                replace=True
            )

            log_alphas: np.ndarray = bootstrap_rng.choice(
                log_alphas,
                size=len(log_alphas),
                replace=True
            )

        def log_alphas_kde(alpha) -> np.ndarray: return len(
            log_alphas) * gaussian_kde(log_alphas)(np.log(alpha)) / alpha
        def log_alphas_pass_kde(alpha) -> np.ndarray: return len(
            log_alphas_pass) * gaussian_kde(log_alphas_pass)(np.log(alpha)) / alpha

        return log_alphas_pass_kde(
            input_alphas) / log_alphas_kde(input_alphas)

    @cached_property
    def best_p_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return lambda input_alphas: self.get_s_KDE(input_alphas)

    def random_p_function(
            self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        return lambda input_alphas: self.get_s_KDE(
            input_alphas, bootstrap_rng=rng)

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
