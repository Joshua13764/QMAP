from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from time import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import polars as pl
from dask.utils import F
from numpy.random import Generator
from polars import DataFrame, Expr, LazyFrame
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from statsmodels.base.model import (GenericLikelihoodModel,
                                    GenericLikelihoodModelResults,
                                    LikelihoodModelResults)
from tqdm import tqdm

from boulder_statistics.analysis.data_product_encyclopedia import \
    DataProductEncyclopedia
from boulder_statistics.analysis.fit_params.general_fit_params import FitParams
from boulder_statistics.analysis.sensitivity_model.s_function import SFunction
from boulder_statistics.analysis.sensitivity_model.sensitivity_model_base import \
    SensitivityModelBase

relative_alpha: Expr = pl.col(
    "alpha") / (2 ** (2 * 4 - 2 * pl.col("tile_lod_number")))


@dataclass(frozen=True)
class PSFDFittingBase[T: FitParams](ABC):
    dp: DataProductEncyclopedia
    LAD_min: float
    sensitivity_model: SensitivityModelBase
    # Does have to be in the database first
    S_manual_interp_Jaccard_threshold: float = field(default=0.7)
    clean_Phi: bool = field(default=True)
    interp_samples: int = field(default=10_000)

    # The true min can be higher than this if the S model requests it
    min_alpha_to_consider: int = field(default=0)

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
    def Cleaned_Phi(self) -> Tuple[np.ndarray, np.ndarray]:
        bin_centers = 0.5 * \
            (self.dp.Phi_counts_smoothed_bins[:-1] +
             self.dp.Phi_counts_smoothed_bins[1:])

        bin_widths = np.abs(
            self.dp.Phi_counts_smoothed_bins[:-1] - self.dp.Phi_counts_smoothed_bins[1:])

        mask: np.ndarray = (self.dp.Phi_counts_smoothed_counts / bin_widths > 10**3
                            ) & (self.dp.Phi_counts_smoothed_counts * bin_widths > 10**3) if self.clean_Phi else np.ones_like(bin_centers, dtype=np.bool)

        return bin_centers[mask], self.dp.Phi_counts_smoothed_counts[mask]

    def F(self, alphas: np.ndarray, fit_params: T,
          s_function: SFunction) -> np.ndarray:

        alphas_sample = np.geomspace(
            alphas.min(),
            alphas.max(),
            self.interp_samples
        )

        total_p_alpha_sample = self.flat_PSFD_func(
            alphas=alphas_sample,
            phis=self.Cleaned_Phi[0],
            phi_weights=self.Cleaned_Phi[1],
            fit_params=fit_params
        )

        total_p_alpha_log = interp1d(
            np.log(alphas_sample),
            np.log(total_p_alpha_sample),
            assume_sorted=True
        )(np.log(alphas))  # Power-law interpolation

        # Ok here as multiple same detections are required for intra-tile calculations, but all
        # the collected data is based of inter-tile calculations for which this
        # model works for
        total_s = 1 - np.prod([
            1 - s_function.function(alphas / (2 ** (2 * 4 - 2 * i))) for i in range(5)
        ], axis=0)

        p_estimate = total_s * np.exp(total_p_alpha_log)

        # If we cut alphas at this point the function will need to reflect this
        p_estimate[alphas > s_function.max_fitting_alpha *
                   (2 ** (4 * 2))] = 0

        # print(s_function.min_fitting_alpha)
        p_estimate[alphas < s_function.min_fitting_alpha] = 0
        p_estimate[alphas < self.min_alpha_to_consider] = 0

        return p_estimate

    def int_F(self, fit_params: T, s_function: SFunction) -> np.floating:
        int_samples = 40_000

        int_alphas = np.geomspace(1, 1e6, int_samples)
        int_probs = self.F(
            int_alphas, fit_params, s_function)
        finite_alphas = int_alphas[int_probs > 0]
        finite_probs = int_probs[int_probs > 0]

        return np.abs(trapezoid(finite_alphas, finite_probs))

    def F_norm(self, alphas: np.ndarray, fit_params: T,
               s_function: SFunction) -> np.ndarray:

        int_F: np.float32 = self.int_F(fit_params, s_function)
        return self.F(alphas, fit_params, s_function) / int_F

    def MLE_fit(self, optimize_params: T, verbose=False,
                summary=True) -> GenericLikelihoodModelResults:
        return self.MLE_fit_general(optimize_params, self.sensitivity_model.best_S_function,
                                    verbose, summary)

    def MLE_fit_general(self, optimize_params: T, s_function: SFunction,
                        verbose=False, summary=True) -> GenericLikelihoodModelResults:

        F_norm: Callable[[np.ndarray, T, SFunction], np.ndarray] = self.F_norm

        class TheoryFit(GenericLikelihoodModel):
            param_names: List[str] = optimize_params.get_labels()

            def __init__(self, x):
                super().__init__(x)

            def loglikeobs(self, params) -> np.ndarray:
                alphas = self.endog

                optimize_params.modify_from_numpy(params)

                if verbose:
                    print(f"Running iteration with params {params}")

                return np.log(F_norm(alphas, optimize_params, s_function))

            @property
            def start_params(self):
                return optimize_params.to_numpy()

        mle_model: GenericLikelihoodModelResults = TheoryFit(
            self.cleaned_alphas(s_function)).fit(disp=verbose)

        if summary:
            print(mle_model.summary())

        return mle_model

    def MultiMLEFit(self, optimize_params: T, numb_runs: int = 10,
                    verbose=False, summary=False) -> DataFrame:

        original_params = optimize_params.to_numpy()

        seed_vals_to_run: np.ndarray = np.random.random_integers(
            0, 100_000, size=numb_runs)

        params_dict: Dict[str, List[float]] = (
            {"aic": [], "bic": [], "numb_alphas": [], "s_max_fitting_alpha": [], "s_min_fitting_alpha": []} |
            {param_name: [] for param_name in optimize_params.get_labels()} |
            {f"{param_name}_err": []
                for param_name in optimize_params.get_labels()}
        )

        for seed in tqdm(seed_vals_to_run, desc="MultiMLE fit running"):
            rng: Generator = np.random.default_rng(seed)
            s_function: SFunction = self.sensitivity_model.random_S_function(
                rng)

            optimize_params.modify_from_numpy(original_params)

            mle_model: GenericLikelihoodModelResults = self.MLE_fit_general(
                optimize_params, s_function, verbose, summary)

            params_dict["aic"].append(mle_model.aic)
            params_dict["bic"].append(mle_model.bic)
            params_dict["numb_alphas"].append(
                self.cleaned_alphas(s_function).size)
            params_dict["s_max_fitting_alpha"].append(
                s_function.max_fitting_alpha)
            params_dict["s_min_fitting_alpha"].append(
                s_function.min_fitting_alpha)

            for param_name, param_value, param_error in zip(
                    optimize_params.get_labels(), mle_model.params, mle_model.bse):

                params_dict[param_name].append(param_value)
                params_dict[f"{param_name}_err"].append(param_error)

        return DataFrame(
            {"seed": seed_vals_to_run} |
            params_dict
        ).with_columns(  # Extra metadata
            pl.lit(self.LAD_min).alias("LAD_min"),
            pl.lit(self.S_manual_interp_Jaccard_threshold).alias("J_min"),
            pl.lit(self.min_alpha_to_consider).alias("min_alpha_to_consider")
        )

    @property
    def plot_range(self) -> Tuple[float, float]:
        """Uses the sensitivity_model best_S_function to find the range

        Returns:
            Tuple[float, float]: _description_
        """
        min: float = self.cleaned_alphas(
            self.sensitivity_model.best_S_function).min()
        max: float = self.cleaned_alphas(
            self.sensitivity_model.best_S_function).max()

        return min, max

    @property
    def cleaned_alphas_best_S(self) -> np.ndarray:
        return self.cleaned_alphas(self.sensitivity_model.best_S_function)

    def cleaned_alphas(self, s_function: SFunction) -> np.ndarray:
        return self.cleaned_data(s_function).collect()["alpha"].to_numpy()

    def cleaned_data(self, s_function: SFunction) -> LazyFrame:
        most = self.dp.boulder_agg_data.filter(
            pl.col("longest_axis_diameter") *
            1000 > self.LAD_min).collect()

        most_ratios = most["longest_axis_diameter"].to_numpy(
        ) / most["surface_area"].to_numpy()
        most_ratios_clean = most_ratios[~np.isnan(most_ratios)]

        mean_p2std = most_ratios_clean.mean() + 2 * most_ratios_clean.std()

        return self.dp.boulder_agg_data.filter(
            pl.col("longest_axis_diameter") * 1000 > self.LAD_min,
            # (pl.col("longest_axis_diameter") /
            #  pl.col("surface_area")) < mean_p2std,

            pl.col("alpha") < s_function.max_fitting_alpha *
            (2 ** (4 * 2)),  # As we want to consider the last LOD

            pl.col("alpha") > s_function.min_fitting_alpha,
            pl.col("alpha") > self.min_alpha_to_consider,

        )
