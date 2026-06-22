from shutil import rmtree
from polars import DataFrame
import polars as pl
import numpy as np
from tqdm import tqdm
from typing import List, Callable
from scipy.ndimage import label
from pathlib import Path

def create_combined_mask(row_masks : np.ndarray, min_row_id : int, max_row_id : int, accept_threshold : int = 2, min_island_size = 5) -> np.ndarray:
    """Handles the merging of detections per LOD

    Args:
        row_masks (np.ndarray): _description_
        accept_threshold (int, optional): _description_. Defaults to 3.
        min_row_id : (int): The min row id to be assigned (assumes that all between min_row_id and max_row_id can be assigned)
        max_row_id : (int): The max row_id that can be assigned (inclusive)

    Returns:
        np.ndarray: An 512x512 array with mask * id for all detections
    """
    transform_cum_mask = np.sum(row_masks, axis = 0)
    accepted_transform_cum_mask = transform_cum_mask >= accept_threshold

    structure = np.array([[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]])
    
    labeled, num_islands = label(accepted_transform_cum_mask, structure=structure)
    # values, counts = np.unique(labeled.flatten(), return_counts=True)
    # erroneous_detections = values[counts < min_island_size]

    assert max_row_id - min_row_id + 1 >= num_islands, f"max_row_id : {max_row_id}, min_row_id : {min_row_id} and num_islands : {num_islands}"
    labeled[labeled != 0] += min_row_id # So row_ids are valid

    return labeled

def create_merge_db_cache(
        df : DataFrame, faces : List[str], lod_levels : List[int], cache_folder : Path, calculate_tile_pixel_offset : Callable[[str], np.ndarray],
        combined_mask_no_merge_path : Path, cleanup : bool = True) -> DataFrame:
    
    for face in faces:
        print(f"Processing face {face}")

        for lod_level in lod_levels:

            lod_tiles = df.filter((pl.col("tile_lod_number") == pl.lit(lod_level)) & (pl.col("tile_face") == pl.lit(face)))
            lod_tile_codes = np.unique(lod_tiles["tile_lod_code"])[::-1]

            lod_scale_factor = 2 ** (max(lod_levels) - lod_level)
            # lod_size = 512 * lod_scale_factor

            for lod_code in tqdm(lod_tile_codes, desc=f"Processing lod {lod_level}", total=len(lod_tile_codes)):

                export_path = cache_folder / f"{face}-{lod_code}.parquet"
                export_path.parent.mkdir(parents=True, exist_ok=True)

                if export_path.exists(): continue

                lod_code_df = lod_tiles.filter(pl.col("tile_lod_code") == pl.lit(lod_code))

                row_masks = np.load(lod_code_df["image_detections_path"][0])["masks_uint8"]
                combined_mask = create_combined_mask(
                    row_masks,
                    max_row_id = lod_code_df["row_id"].max(),
                    min_row_id = lod_code_df["row_id"].min()
                    )

                _is, _js = np.indices((combined_mask.shape[0] * lod_scale_factor, combined_mask.shape[1] * lod_scale_factor), dtype=np.uint32)

                tile_offset = calculate_tile_pixel_offset(lod_code)

                schema = {
                    "lod_level": pl.UInt8,
                    "lod_code": pl.String,
                    "face" : pl.String,
                    "i": pl.UInt32,
                    "j": pl.UInt32,
                    "row_id": pl.UInt32,
                }

                df_to_export = pl.DataFrame({
                    "lod_level" : lod_level,
                    "lod_code" : lod_code,
                    "face" : face,
                    "i": _is.flatten() + tile_offset[0] * lod_scale_factor,
                    "j": _js.flatten() + tile_offset[1] * lod_scale_factor,
                    "row_id": combined_mask.repeat(lod_scale_factor, axis=0).repeat(lod_scale_factor, axis=1).flatten(),
                }, schema=schema)

                df_to_export: DataFrame = df_to_export.filter(pl.col("row_id") != 0)

                export_path: Path = cache_folder / f"{face}-{lod_code}.parquet"
                export_path.parent.mkdir(parents=True, exist_ok=True)

                df_to_export.write_parquet(export_path)

    combined_mask_no_merge: pl.LazyFrame = pl.scan_parquet(cache_folder)

    

    print(f"Building {combined_mask_no_merge_path.stem} table...")
    combined_mask_no_merge.sink_parquet(combined_mask_no_merge_path)
    print(f"Exported combined_mask_no_merge to {combined_mask_no_merge_path}")

    print(f"Cleaning {cache_folder.stem} cache...")
    if cleanup:
        rmtree(cache_folder)
        print("Cleaned up folder")

    return active_row_ids