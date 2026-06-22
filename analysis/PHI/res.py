from pathlib import Path
import polars as pl
import numpy as np

run_head : str = "AO33-Database-Refinement-Improved"
run_folder: Path = Path.cwd().resolve().parent
project_root: Path = next(
    p for p in run_folder.parents
    if p.name == run_head
)

data_products_path: Path = project_root / "data_products"
combined_atlas_path: Path = data_products_path / "combined_atlas.parquet"
combined_mask_path: Path = data_products_path / "combined_mask.parquet"
mask_atlas_combined_path: Path = data_products_path / "mask_atlas_combined.parquet"
boulder_agg_data_path : Path = data_products_path / "boulder_agg_data.parquet"
combined_mask_no_merge_path : Path = data_products_path / "combined_mask_no_merge.parquet"

Phi_data_path: Path = project_root / "refinement_part_3_3"
Phi_mesh_path: Path = Phi_data_path / "Phi_export_mesh"
Phi_sphere_path: Path = Phi_data_path / "Phi_export_sphere"

Phi_counts_path : Path = project_root / "refinement_part_3_4"
Phi_counts_noisy_path : Path = Phi_counts_path / r"Phi_counts_noisy\512_bins.npz"
Phi_counts_smoothed_path : Path = Phi_counts_path / r"Phi_counts_smoothed\512_bins.npz"
Phi_counts_sphere_theory_path : Path = Phi_counts_path / r"Phi_counts_sphere_theoretical\512_bins.npz"
Phi_counts_sphere_noisy_path = Phi_counts_path / r"Phi_counts_sphere_noisy\512_bins.npz"

combined_mask: pl.LazyFrame = pl.scan_parquet(combined_mask_path)
combined_atlas: pl.LazyFrame = pl.scan_parquet(combined_atlas_path)
mask_atlas_combined: pl.LazyFrame = pl.scan_parquet(mask_atlas_combined_path)
boulder_agg_data: pl.LazyFrame = pl.scan_parquet(boulder_agg_data_path)
combined_mask_no_merge: pl.LazyFrame = pl.scan_parquet(combined_mask_no_merge_path)

Phi_mesh: pl.LazyFrame = pl.scan_parquet(Phi_mesh_path)
Phi_sphere: pl.LazyFrame = pl.scan_parquet(Phi_sphere_path)

Phi_counts_noisy_counts = np.load(Phi_counts_noisy_path)["counts"]
Phi_counts_smoothed_counts = np.load(Phi_counts_smoothed_path)["counts"]
Phi_counts_sphere_theory_counts = np.load(Phi_counts_sphere_theory_path)["counts"]
Phi_counts_sphere_noisy_counts = np.load(Phi_counts_sphere_noisy_path)["counts"]

Phi_counts_noisy_bins = np.load(Phi_counts_noisy_path)["bins"]
Phi_counts_smoothed_bins = np.load(Phi_counts_smoothed_path)["bins"]
Phi_counts_sphere_theory_bins = np.load(Phi_counts_sphere_theory_path)["bins"]
Phi_counts_sphere_noisy_bins = np.load(Phi_counts_sphere_noisy_path)["bins"]

print(f"combined_mask : has schema {combined_mask.collect_schema()} and \
      additional information {combined_mask.explain()}")

print(f"combined_atlas : has schema {combined_atlas.collect_schema()} and \
      additional information {combined_atlas.explain()}")

print(f"mask_atlas_combined : has schema {mask_atlas_combined.collect_schema()} and \
      additional information {mask_atlas_combined.explain()}")

print(f"boulder_agg_data : has schema {boulder_agg_data.collect_schema()} and \
      additional information {boulder_agg_data.explain()}")

print(f"Phi_mesh : has schema {Phi_mesh.collect_schema()} and additional \
      information {Phi_mesh.explain()}")

print(f"Phi_sphere : has schema {Phi_sphere.collect_schema()} and additional \
      information {Phi_sphere.explain()}")