from typing import List

import polars as pl
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

STEP = 128
JS = np.arange(0, 8192 + STEP, STEP)

def get_combined_Phi(Phi_mesh : pl.LazyFrame, Phi_sphere : pl.LazyFrame) -> pl.LazyFrame:

    Phi_mesh = Phi_mesh.filter(pl.col("area") != 0).with_columns(
        (pl.col("area") * 1e6).alias("area_m_sqr"),
        (pl.col("area") * 1e10).alias("area_cm_sqr"),
        (1 / (pl.col("area") * 1e6)).alias("Phi")
        )

    Phi_sphere = Phi_sphere.filter(pl.col("area") != 0).with_columns(
        (pl.col("area") * 1e6).alias("area_m_sqr"),
        (pl.col("area") * 1e10).alias("area_cm_sqr"),
        (1 / (pl.col("area") * 1e6)).alias("Phi_o")
        )

    xs: pl.Expr = (pl.col("j").cast(pl.Float64) - 4096) / 4096
    ys: pl.Expr = (pl.col("i").cast(pl.Float64) - 4096) / 4096
    r: pl.Expr = (xs ** 2 + ys ** 2).sqrt()

    Phi_t: pl.Expr = (32 * (np.pi / 6) * (1 + r ** 2).pow(3/2))
    Phi_s: pl.Expr = (pl.col("Phi") * Phi_t) / pl.col("Phi_o")

    combined_Phi: pl.LazyFrame = Phi_mesh.join(
        Phi_sphere.select(["Phi_o", "face", "i", "j"]), on=["face", "i", "j"], how="inner"
    ).with_columns(
        Phi_t.alias("Phi_t"),
        Phi_s.alias("Phi_s"),
        r.alias("r")
    )

    return combined_Phi

def get_Phi_counts(
        combined_Phi : pl.LazyFrame,
        Phi_counts_path : Path,
        Phi_type_str : str,
        bin_numbers : NDArray[np.int32]
        ) -> None:
    
    if Phi_counts_path.exists():
        return
    else:
        Phi_counts_path.mkdir(parents=True)
    
    max_Phi = -np.inf
    min_Phi = 1e-3 # Hard limit to avoid vertical faces

    for j_left, j_right in tqdm(zip(JS[:-1], JS[1:]),
                            desc=f"Finding min and max for {Phi_counts_path.stem}", total = len(JS) - 1):
            
            iter_max_Phi, iter_min_Phi = (
                combined_Phi
                .filter(pl.col("j").is_between(j_left, j_right, closed="left"))
                .select(
                    pl.max(Phi_type_str).alias("phi_max"),
                    pl.min(Phi_type_str).alias("phi_min"),
                )
                .collect()
                .row(0)
            )

            max_Phi = max(max_Phi, iter_max_Phi)
            min_Phi = min(min_Phi, iter_min_Phi)

    phi_bins_sets_data = {
         f"{bin_number}_bins" : {
              "bins" : np.geomspace(min_Phi, max_Phi, bin_number),
              "counts" : np.zeros(bin_number - 1)
         }
         for bin_number in bin_numbers
    }

    for j_left, j_right in tqdm(zip(JS[:-1], JS[1:]),
                                desc=f"Populating bins for {Phi_counts_path.stem}", total = len(JS) - 1):
        samples = (
             combined_Phi
            .filter(pl.col("j").is_between(j_left, j_right, closed="left"))
            .select(Phi_type_str)
            .collect()[Phi_type_str]
            .to_numpy())
        
        for set_name, set_data in phi_bins_sets_data.items():
            set_data["counts"] += np.histogram(samples, bins = set_data["bins"])[0]

    for set_name, set_data in phi_bins_sets_data.items():
        np.savez(Phi_counts_path / f"{set_name}.npz", **set_data)

def run_compute_Phi_counts(
        Phi_counts_smoothed_path : Path, Phi_counts_noisy_path : Path, Phi_counts_sphere_theoretical_path : Path,
        Phi_counts_sphere_noisy_path : Path,
        Phi_mesh : pl.LazyFrame, Phi_sphere : pl.LazyFrame, bin_numbers : NDArray[np.int32]) -> None:
    
    combined_Phi: pl.LazyFrame = get_combined_Phi(Phi_mesh, Phi_sphere)
    get_Phi_counts(combined_Phi, Phi_counts_smoothed_path, "Phi_s", bin_numbers)
    get_Phi_counts(combined_Phi, Phi_counts_noisy_path, "Phi", bin_numbers)
    get_Phi_counts(combined_Phi, Phi_counts_sphere_theoretical_path, "Phi_o", bin_numbers)
    get_Phi_counts(combined_Phi, Phi_counts_sphere_noisy_path, "Phi_t", bin_numbers)