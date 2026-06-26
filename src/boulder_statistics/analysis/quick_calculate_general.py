from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from time import time
from typing import Any, Callable, List, Tuple

import numpy as np
import polars as pl
from polars import Expr, LazyFrame
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from statsmodels.base.model import (GenericLikelihoodModel,
                                    GenericLikelihoodModelResults,
                                    LikelihoodModelResults)

from boulder_statistics.analysis.data_product_encyclopedia import \
    DataProductEncyclopedia
from boulder_statistics.analysis.fit_params.general_fit_params import FitParams

relative_alpha: Expr = pl.col(
    "alpha") / (2 ** (2 * 4 - 2 * pl.col("tile_lod_number")))


@dataclass(frozen=True)
class GeneralPSFDFittingFunction[T: FitParams](ABC):
    dp: DataProductEncyclopedia
    LAD_min: float = 2
    max_fitting_alpha: float = 1e8  # 1e4
    min_fitting_alpha: float = 1e1
    # Does have to be in the database first
    S_manual_interp_Jaccard_threshold: float = 0.5
    clean_Phi: bool = True

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
    def S_fast(self) -> interp1d:

        S_interp = self.dp.S_manual_interp.filter(
            pl.col("J_threshold") == self.S_manual_interp_Jaccard_threshold).collect()

        return interp1d(
            S_interp["view_port_alpha"].to_numpy(),
            S_interp["p_detection"].to_numpy(),
            kind="linear",
            bounds_error=False,
            fill_value=0.0
        )

    @cached_property
    def Cleaned_Phi(self) -> Tuple[np.ndarray, np.ndarray]:
        bin_centers = 0.5 * \
            (self.dp.Phi_counts_smoothed_bins[:-1] +
             self.dp.Phi_counts_smoothed_bins[1:])

        bin_widths = np.abs(
            self.dp.Phi_counts_smoothed_bins[:-1] - self.dp.Phi_counts_smoothed_bins[1:])

        mask: np.ndarray = (self.dp.Phi_counts_smoothed_counts / bin_widths > 10**3
                            ) & (self.dp.Phi_counts_smoothed_counts * bin_widths > 10**3) if self.clean_Phi else np.ones_like(bin_centers, dtype=np.bool)

        return bin_centers[mask], self.dp.Phi_counts_smoothed_counts[mask]

    def F(self, alphas: np.ndarray, fit_params: T) -> np.ndarray:

        total_p_alpha = self.flat_PSFD_func(
            alphas=alphas,
            phis=self.Cleaned_Phi[0],
            phi_weights=self.Cleaned_Phi[1],
            fit_params=fit_params
        )

        # Ok here as multiple same detections are required for intra-tile calculations, but all
        # the collected data is based of inter-tile calculations for which this
        # model works for
        total_s = 1 - np.prod([
            1 - self.S_fast(alphas / (2 ** (2 * 4 - 2 * i))) for i in range(5)
        ], axis=0)

        p_estimate = total_s
        # We don't fit larger than this as unreliable
        p_estimate[alphas > self.max_fitting_alpha] = 0
        # We don't fit smaller than this as unreliable
        p_estimate[alphas < self.min_fitting_alpha] = 0

        return p_estimate

    def int_F(self, fit_params: T) -> np.floating:
        int_samples = 40_000

        int_alphas = np.geomspace(1, 1e6, int_samples)
        int_probs = self.F(
            int_alphas,
            fit_params=fit_params)
        finite_alphas = int_alphas[int_probs > 0]
        finite_probs = int_probs[int_probs > 0]

        return np.abs(trapezoid(finite_alphas, finite_probs))

    def F_norm(self, alphas: np.ndarray, fit_params: T) -> np.ndarray:

        int_F: np.float32 = self.int_F(fit_params)
        return self.F(alphas, fit_params) / int_F

    def MLE_fit(self, optimize_params: T,
                verbose=True) -> GenericLikelihoodModelResults:

        F_norm: Callable[[np.ndarray, T], np.ndarray] = self.F_norm

        class TheoryFit(GenericLikelihoodModel):
            param_names: List[str] = optimize_params.get_labels()

            def __init__(self, x):
                super().__init__(x)

            def loglikeobs(self, params) -> np.ndarray:
                alphas = self.endog

                optimize_params.modify_from_numpy(params)

                if verbose:
                    print(f"Running iteration with params {params}")

                return np.log(F_norm(alphas, optimize_params))

            @property
            def start_params(self):
                return optimize_params.to_numpy()

        mle_model: GenericLikelihoodModelResults = TheoryFit(
            self.cleaned_data.collect()["alpha"].to_numpy()).fit()

        if verbose:
            print(mle_model.summary())

        return mle_model

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
