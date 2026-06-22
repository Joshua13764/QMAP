import polars as pl
from pathlib import Path
from typing import List

def join_mask_and_atlas_tables(faces : List[str], combined_mask_path : Path, combined_atlas_path : Path, mask_atlas_combined_path : Path, cache_folder : Path = Path(r"refinement_part_2")) -> None:

    cache_folder.mkdir(parents=True, exist_ok=True)

    for face in faces:
        combined_mask: pl.LazyFrame = pl.scan_parquet(combined_mask_path).filter(pl.col("face") == face)
        combined_atlas: pl.LazyFrame = pl.scan_parquet(combined_atlas_path).filter(pl.col("face") == face)

        mask_atlas_combined: pl.LazyFrame = combined_mask.join(
            combined_atlas,
            on=["face", "i", "j"],
            how="inner"
        )

        mask_atlas_combined.sink_parquet(cache_folder / f"mask_atlas_combined_{face}.parquet")


    combined_atlas_data_product: pl.LazyFrame = pl.scan_parquet(f"{cache_folder.as_posix()}/*.parquet")
    combined_atlas_data_product.sink_parquet(mask_atlas_combined_path)