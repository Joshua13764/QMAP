from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from time import time

import numpy as np
import polars as pl
from polars import Expr, LazyFrame
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from boulder_statistics.analysis.data_product_encyclopedia import \
    DataProductEncyclopedia

relative_alpha: Expr = pl.col(
    "alpha") / (2 ** (2 * 4 - 2 * pl.col("tile_lod_number")))


@dataclass(frozen=True)
class GeneralPSFDFittingFunction[T](ABC):
    dp: DataProductEncyclopedia
    LAD_min: float = 2
    max_fitting_alpha: float = 1e8  # 1e4
    min_fitting_alpha: float = 1e1

    @abstractmethod
    def flat_PSFD_func(self, alphas: np.ndarray,
                       phis: np.ndarray, phi_weights: np.ndarray, fit_params: T) -> np.ndarray:
        ...

    @cached_property
    def no_merge_sample(self) -> np.ndarray:
        return self.dp.combined_mask_no_merge.group_by("row_id").agg(
            pl.len().alias("alpha"),
            pl.col("lod_level").first().alias("tile_lod_number")
        ).with_columns(
            relative_alpha.alias("relative_alpha")
        ).collect()["relative_alpha"].to_numpy()

    @cached_property
    def sample(self) -> np.ndarray:
        return self.dp.boulder_agg_data.collect().with_columns(
            relative_alpha.alias("relative_alpha")
        )["relative_alpha"].to_numpy()

    @cached_property
    def S_fast_un_norm(self) -> interp1d:

        log_sample = np.log(self.no_merge_sample[self.no_merge_sample > 5.5])
        kde = gaussian_kde(log_sample)

        def S_un_norm(x): return kde(np.log(x)) / x

        xs_grid = np.geomspace(
            self.no_merge_sample[self.no_merge_sample > 5.5].min(),
            self.no_merge_sample.max(),
            1000)

        ys_grid = S_un_norm(xs_grid)

        return interp1d(
            xs_grid,
            ys_grid,
            kind="linear",
            bounds_error=False,
            fill_value=0.0
        )

    def F(self, alphas: np.ndarray, effectiveness: np.float32,
          fit_params: T) -> np.ndarray:
        def S_estimate(x): return self.S_fast_un_norm(
            x) / (x ** (-q))  # TODO: think about q estimate

        total_p_alpha = self.flat_PSFD_func(
            alphas=alphas,
            phis=0.5 *
            (self.dp.Phi_counts_smoothed_bins[1:] +
             self.dp.Phi_counts_smoothed_bins[:-1]),
            phi_weights=self.dp.Phi_counts_smoothed_counts,
            fit_params=fit_params
        )

        total_s = 1 - np.prod([
            1 - effectiveness * 0.5 * S_estimate(alphas / (2 ** (2 * 4 - 2 * i))) for i in range(5)
        ], axis=0)

        p_estimate = total_p_alpha * total_s
        # We don't fit larger than this as unreliable
        p_estimate[alphas > self.max_fitting_alpha] = 0
        # We don't fit smaller than this as unreliable
        p_estimate[alphas < self.min_fitting_alpha] = 0

        return p_estimate

    def int_F(self, effectiveness: np.float32, fit_params: T) -> np.floating:
        int_samples = 40_000

        int_alphas = np.geomspace(1, 1e6, int_samples)
        int_probs = self.F(
            int_alphas,
            effectiveness=effectiveness,
            fit_params=fit_params)
        finite_alphas = int_alphas[int_probs > 0]
        finite_probs = int_probs[int_probs > 0]

        return np.abs(trapezoid(finite_alphas, finite_probs))

    def F_norm(self, alphas: np.ndarray, effectiveness: np.float32,
               fit_params: T) -> np.ndarray:

        int_F: np.float32 = self.int_F(effectiveness, fit_params)
        return self.F(alphas, effectiveness, fit_params) / int_F

    @property
    def plot_range(self):
        min = self.cleaned_data.collect()["alpha"].min()
        max = self.cleaned_data.collect()["alpha"].max()

        return min, max

    @property
    def cleaned_data(self) -> LazyFrame:
        most = self.dp.boulder_agg_data.filter(
            pl.col("longest_axis_diameter") *
            1000 > self.LAD_min).collect()

        most_ratios = most["longest_axis_diameter"].to_numpy(
        ) / most["surface_area"].to_numpy()
        most_ratios_clean = most_ratios[~np.isnan(most_ratios)]

        mean_p2std = most_ratios_clean.mean() + 2 * most_ratios_clean.std()

        return self.dp.boulder_agg_data.filter(
            pl.col("longest_axis_diameter") * 1000 > self.LAD_min,
            (pl.col("longest_axis_diameter") /
             pl.col("surface_area")) < mean_p2std,
            pl.col("alpha") < self.max_fitting_alpha,
            # We don't fit larger than this as unreliable
            pl.col("alpha") > self.min_fitting_alpha,
            # We don't fit smaller than this as unreliable
        )
