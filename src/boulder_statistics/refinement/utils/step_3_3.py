from pathlib import Path
from typing import List

import polars as pl
from tqdm import tqdm


def normalize_triangles_expr() -> List[pl.Expr]:

    p1size: pl.Expr = (
        pl.col("p1x") ** 2 +
        pl.col("p1y") ** 2 +
        pl.col("p1z")**2).sqrt()
    p2size: pl.Expr = (
        pl.col("p2x") ** 2 +
        pl.col("p2y") ** 2 +
        pl.col("p2z")**2).sqrt()
    p3size: pl.Expr = (
        pl.col("p3x") ** 2 +
        pl.col("p3y") ** 2 +
        pl.col("p3z")**2).sqrt()

    return [
        (pl.col("p1x") / p1size).alias("p1x"),
        (pl.col("p1y") / p1size).alias("p1y"),
        (pl.col("p1z") / p1size).alias("p1z"),

        (pl.col("p2x") / p2size).alias("p2x"),
        (pl.col("p2y") / p2size).alias("p2y"),
        (pl.col("p2z") / p2size).alias("p2z"),

        (pl.col("p3x") / p3size).alias("p3x"),
        (pl.col("p3y") / p3size).alias("p3y"),
        (pl.col("p3z") / p3size).alias("p3z"),
    ]


def extract_triangle_components(triangle_components_folder: Path, faces: List[str],
                                num_i: int, combined_atlas: pl.LazyFrame) -> None:

    triangle_components_folder.mkdir(parents=True, exist_ok=True)

    for face in faces:
        for i in tqdm(range(num_i - 1),
                      desc=f"Creating triangles for face {face}"):
            df_export_folder: Path = triangle_components_folder / \
                f"triangles-{i}-to-{i + 1}-face-{face}.parquet"
            if df_export_folder.exists():
                continue

            i_is_num_i: pl.DataFrame = combined_atlas.filter(pl.col("i") == i, pl.col("face") == face).sort(
                "j").collect()
            i_is_num_i_p1: pl.DataFrame = combined_atlas.filter(pl.col("i") == i + 1, pl.col("face") == face).sort(
                "j").collect()

            js = i_is_num_i["j"].to_numpy()
            xs = i_is_num_i["positions_x"].to_numpy()
            ys = i_is_num_i["positions_y"].to_numpy()
            zs = i_is_num_i["positions_z"].to_numpy()

            jsp1 = i_is_num_i_p1["j"].to_numpy()
            xsp1 = i_is_num_i_p1["positions_x"].to_numpy()
            ysp1 = i_is_num_i_p1["positions_y"].to_numpy()
            zsp1 = i_is_num_i_p1["positions_z"].to_numpy()

            upper_triangles = pl.DataFrame({
                # Point 1
                "p1i": i, "p1j": js[:-1],
                "p1x": xs[:-1], "p1y": ys[:-1], "p1z": zs[:-1],

                # Point 2
                "p2i": i, "p2j": js[1:],
                "p2x": xs[1:], "p2y": ys[1:], "p2z": zs[1:],

                # Point 3
                "p3i": i + 1, "p3j": jsp1[1:],
                "p3x": xsp1[1:], "p3y": ysp1[1:], "p3z": zsp1[1:],
            })

            lower_triangles = pl.DataFrame({
                # Point 1
                "p1i": i, "p1j": js[:-1],
                "p1x": xs[:-1], "p1y": ys[:-1], "p1z": zs[:-1],

                # Point 2
                "p2i": i + 1, "p2j": jsp1[:-1],
                "p2x": xsp1[:-1], "p2y": ysp1[:-1], "p2z": zsp1[:-1],

                # Point 3
                "p3i": i + 1, "p3j": jsp1[1:],
                "p3x": xsp1[1:], "p3y": ysp1[1:], "p3z": zsp1[1:],
            })

            triangles: pl.DataFrame = pl.concat(
                [upper_triangles, lower_triangles])
            triangles = triangles.with_columns(pl.lit(face).alias("face"))
            triangles.write_parquet(df_export_folder)


def find_related_triangles(triangle_components_folder: Path,
                           related_triangles_folder: Path, extra_config: List[pl.Expr]) -> None:

    if related_triangles_folder.exists():
        return

    related_triangles_folder.mkdir(exist_ok=True)
    print(f"Exporting related triangles to {related_triangles_folder.name}")

    triangles: pl.LazyFrame = pl.scan_parquet(triangle_components_folder)
    triangles = triangles.with_columns(*extra_config)

    triangles_area_added: pl.LazyFrame = triangles.with_columns(
        (
            (
                ((
                    (pl.col("p2y") - pl.col("p1y")) *
                    (pl.col("p3z") - pl.col("p1z"))
                    - (pl.col("p2z") - pl.col("p1z")) *
                    (pl.col("p3y") - pl.col("p1y"))
                ) * 1e-8) ** 2
                +
                ((
                    (pl.col("p2z") - pl.col("p1z")) *
                    (pl.col("p3x") - pl.col("p1x"))
                    - (pl.col("p2x") - pl.col("p1x")) *
                    (pl.col("p3z") - pl.col("p1z"))
                ) * 1e-8) ** 2
                +
                ((
                    (pl.col("p2x") - pl.col("p1x")) *
                    (pl.col("p3y") - pl.col("p1y"))
                    - (pl.col("p2y") - pl.col("p1y")) *
                    (pl.col("p3x") - pl.col("p1x"))
                ) * 1e-8) ** 2
            ).sqrt() * 0.5 / 1e-8
        ).alias("triangle_area")
    )

    triangles_area_added.select([
        pl.col("p1i").alias("i"),
        pl.col("p1j").alias("j"),
        pl.col("face").alias("face"),
        pl.col("triangle_area")
    ]).sink_parquet(related_triangles_folder / "p1s.parquet")

    triangles_area_added.select([
        pl.col("p2i").alias("i"),
        pl.col("p2j").alias("j"),
        pl.col("face").alias("face"),
        pl.col("triangle_area")
    ]).sink_parquet(related_triangles_folder / "p2s.parquet")

    triangles_area_added.select([
        pl.col("p3i").alias("i"),
        pl.col("p3j").alias("j"),
        pl.col("face").alias("face"),
        pl.col("triangle_area")
    ]).sink_parquet(related_triangles_folder / "p3s.parquet")


def extract_Phi(faces: List[str], num_i: int,
                related_triangles_folder: Path, Phi_export_path: Path) -> None:

    Phi_export_path.mkdir(parents=True, exist_ok=True)
    related_triangles: pl.LazyFrame = pl.scan_parquet(related_triangles_folder)
    step_size = 1024

    for face in faces:
        for i in tqdm(range(0, num_i, step_size),
                      desc=f"Running steps for face {face}"):

            LAS_atlas = related_triangles.filter(
                pl.col("face") == face, pl.col(
                    "i") >= i, pl.col("i") < i + step_size
            ).group_by(["face", "i", "j"]).agg(
                (2 * pl.col("triangle_area")).mean().alias("area"))

            export_path = Phi_export_path / \
                f"LAS_factors_{face}_from_{i}_to_{i + step_size - 1}.parquet"
            if export_path.exists() == True:
                continue

            LAS_atlas.sink_parquet(export_path)


def find_LAS(cache_folder: Path, faces: List[str], num_i: int, combined_atlas: pl.LazyFrame,
             Phi_export_path_mesh: Path, Phi_export_path_sphere: Path) -> None:
    triangle_components_folder: Path = cache_folder / "triangle_components"
    related_triangles_folder_mesh: Path = cache_folder / "related_triangles_mesh"
    related_triangles_folder_sphere: Path = cache_folder / "related_triangles_sphere"

    extract_triangle_components(
        triangle_components_folder,
        faces,
        num_i,
        combined_atlas)
    find_related_triangles(
        triangle_components_folder,
        related_triangles_folder_mesh,
        extra_config=[])
    find_related_triangles(
        triangle_components_folder,
        related_triangles_folder_sphere,
        extra_config=normalize_triangles_expr())

    extract_Phi(
        faces,
        num_i,
        related_triangles_folder_mesh,
        Phi_export_path_mesh)
    extract_Phi(
        faces,
        num_i,
        related_triangles_folder_sphere,
        Phi_export_path_sphere)
