from pathlib import Path
from typing import Callable, List

import numpy as np
import polars as pl
import tifffile
from polars import DataFrame
from tqdm import tqdm

from boulder_statistics.refinement.utils.pan_to_cubemap import PANToCubemap

ATLASES = {
    "uint8_reflectance": pl.UInt8,
    "32bit_reflectance": pl.Float32,
    "positions_x": pl.Float32,
    "positions_y": pl.Float32,
    "positions_z": pl.Float32,
}


def create_combined_atlas_components(df: DataFrame, faces: List[str], max_lod_level: int, calculate_tile_pixel_offset: Callable[[str], np.ndarray],
                                     attribute_atlas_cache_folder: Path = Path(r"refinement_part_1")) -> None:

    pan_img_loaded = False
    pan_img = None

    for atlas_name, atlases_data_type in ATLASES.items():

        pan_img_loaded = False

        for face in faces:
            lod_level = max_lod_level

            lod_tiles = df.filter(
                (pl.col("tile_lod_number") == pl.lit(lod_level)) & (
                    pl.col("tile_face") == pl.lit(face)))
            lod_tile_codes = np.unique(lod_tiles["tile_lod_code"])[::-1]

            lod_scale_factor = 2 ** (max_lod_level - lod_level)

            for lod_code in tqdm(
                    lod_tile_codes, desc=f"Processing face {face} with lod depth {lod_level}", total=len(lod_tile_codes)):

                export_path: Path = attribute_atlas_cache_folder / \
                    atlas_name / f"{face}-{lod_code}.parquet"
                export_path.parent.mkdir(parents=True, exist_ok=True)

                if export_path.exists():
                    continue

                if not pan_img_loaded:
                    # To resolve very large negatives issues
                    pan_img = np.clip(
                        tifffile.imread(
                            rf"external_data\maps\{atlas_name}.tif"), -1e10, np.inf)
                    # <tifffile.TiffPage 0 @8> parsing GDAL_NODATA tag raised ValueError('-3.4028226550889045e+38 is not castable to float32')
                    pan_img_loaded = True

                row = list(
                    df.filter(
                        pl.col("tile_lod_code") == pl.lit(lod_code)).sample(1).iter_rows(
                        named=True))[0]

                sample_array = PANToCubemap.sample_face_roi_simple_super_sample(
                    pan_img,
                    face=face,
                    x_range=[row["tile_x_min"], row["tile_x_max"]],
                    y_range=[row["tile_y_min"], row["tile_y_max"]],
                    sample_resolution=(512, 512),
                    super_sample_factor=1
                )

                _is, _js = np.indices((sample_array.shape[0] *
                                       lod_scale_factor, sample_array.shape[1] *
                                       lod_scale_factor), dtype=np.uint32)

                tile_offset = calculate_tile_pixel_offset(lod_code)

                schema = {
                    "face": pl.String,
                    "i": pl.UInt32,
                    "j": pl.UInt32,
                    atlas_name: atlases_data_type,
                }

                df_to_export = pl.DataFrame({
                    "face": face,
                    "i": _is.flatten() + tile_offset[0] * lod_scale_factor,
                    "j": _js.flatten() + tile_offset[1] * lod_scale_factor,
                    atlas_name: sample_array.repeat(lod_scale_factor, axis=0).repeat(lod_scale_factor, axis=1).flatten(),
                }, schema=schema)

                df_to_export.write_parquet(export_path)


def merge_combined_atlas_components(
        combined_atlas_path: Path, attribute_atlas_cache_folder: Path = Path(r"refinement_part_1")) -> None:

    atlas_dbs: List[pl.LazyFrame] = [
        pl.scan_parquet(
            f"{(attribute_atlas_cache_folder / atlas_name).as_posix()}/*.parquet")
        for atlas_name, atlases_data_type in ATLASES.items()
    ]

    combined_atlas_data_product: pl.LazyFrame = pl.concat(  # Since same shape
        [
            atlas_dbs[0].select(["face", "i", "j"]),
            *[ldf.select(pl.exclude(["face", "i", "j"])) for ldf in atlas_dbs]
        ],
        how="horizontal"
    )

    combined_atlas_data_product.sink_parquet(combined_atlas_path)
