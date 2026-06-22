import polars as pl

def agg_stats(mask_atlas_combined : pl.LazyFrame) -> pl.DataFrame:

    return mask_atlas_combined.group_by("row_id").agg(
        pl.col("i").mean().alias("mean_i"),
        pl.col("j").mean().alias("mean_j"),

        pl.col("i").max().alias("max_i"),
        pl.col("j").max().alias("max_j"),

        pl.col("i").min().alias("min_i"),
        pl.col("j").min().alias("min_j"),

        pl.col("positions_x").mean().alias("mean_position_x"),
        pl.col("positions_y").mean().alias("mean_position_y"),
        pl.col("positions_z").mean().alias("mean_position_z"),

        pl.len().alias("alpha"),

        pl.col("lod_level").first(),
        pl.col("lod_code").first(),
    ).collect()
    