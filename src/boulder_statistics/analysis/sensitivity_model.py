from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Tuple

import numpy as np
import polars as pl
from polars import DataFrame
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class SensitivityModel():
    df: DataFrame
    J_threshold: float = field(default=0.7)
    number_of_bins: int = field(default=16)

    @cached_property
    def valid_sample_data(
            self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Any less the marking of these is unreliable
        df = self.df.filter(pl.col("viewport_size") > 4)

        alpha_values = df["viewport_size"].to_numpy()
        J_values = df["Jaccard_index"].to_numpy()

        successful_detection_mask = J_values > self.J_threshold
        successful_alpha_values = alpha_values[successful_detection_mask]

        bins = np.geomspace(
            alpha_values.min(),
            alpha_values.max(),
            self.number_of_bins + 1)

        # bins = np.quantile(
        #     successful_alpha_values,
        #     np.linspace(0, 1, len(alpha_values) // self.min_bin_count + 1),
        # )
        total_detections, _ = np.histogram(alpha_values, bins=bins)
        successful_detections, _ = np.histogram(
            successful_alpha_values, bins=bins)

        valid_bin_mask = (
            (np.cumsum(total_detections) != 0)
            & (np.cumsum(total_detections[::-1])[::-1] != 0)
        )

        valid_total_detections = total_detections[valid_bin_mask]
        valid_successful_detections = successful_detections[valid_bin_mask]
        valid_bins_lefts = bins[:-1][valid_bin_mask]
        valid_bins_rights = bins[1:][valid_bin_mask]

        keep = valid_total_detections >= 0

        return (
            valid_bins_lefts[keep],
            valid_bins_rights[keep],
            valid_successful_detections[keep],
            valid_total_detections[keep],
        )

    @cached_property
    def valid_sample_p(self) -> np.ndarray:
        *_, valid_successful_detections, \
            valid_total_detections = self.valid_sample_data

        if not np.all(valid_total_detections > 0):
            raise ValueError(
                "Unexpected zero-count bin in valid_total_detections")

        return valid_successful_detections / valid_total_detections

    @cached_property
    def valid_sample_sigma(self) -> np.ndarray:
        *_,
        valid_total_detections = self.valid_sample_data

        return np.sqrt(
            (self.valid_sample_p * (1 - self.valid_sample_p)) / valid_total_detections)

    def p_function_from_p_values(
            self, p_values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        valid_bins_lefts, valid_bins_rights, *_ = self.valid_sample_data

        bin_centers = 0.5 * (valid_bins_lefts + valid_bins_rights)
        x = np.r_[valid_bins_lefts[0], bin_centers, valid_bins_rights[-1]]
        y = np.r_[0.0, p_values, 0.0]

        return lambda alphas: interp1d(
            np.log(x),
            y,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )(np.log(alphas))

    def sample_random_p(self, rng: np.random.Generator) -> np.ndarray:
        *_, successful, total = self.valid_sample_data

        alpha = successful + 1
        beta = total - successful + 1

        return rng.beta(alpha, beta)

    @cached_property
    def best_p_function(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.p_function_from_p_values(self.valid_sample_p)

    def random_p_function(
            self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:

        sample = self.sample_random_p(rng)
        sample[0] = 0
        sample[1] = 0
        sample[-1] = 0  # Clamping
        sample[-2] = 0

        return self.p_function_from_p_values(sample)

    @cached_property
    def min_fitting_alpha(self) -> float:
        valid_bins_lefts, valid_bins_rights, *_ = self.valid_sample_data

        tests = np.geomspace(
            valid_bins_lefts[0],
            valid_bins_rights[-1],
            10_000
        )

        return tests[np.flatnonzero(
            self.best_p_function(tests)
        )[0]]

    @cached_property
    def max_fitting_alpha(self) -> float:
        valid_bins_lefts, valid_bins_rights, *_ = self.valid_sample_data

        tests = np.geomspace(
            valid_bins_lefts[0],
            valid_bins_rights[-1],
            10_000
        )
        return tests[np.flatnonzero(
            self.best_p_function(tests)
        )[-1]]
