from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple

import igl
import numpy as np
import polars as pl

from boulder_statistics.analysis.data_product_encyclopedia import \
    DataProductEncyclopedia
from boulder_statistics.refinement_plus.facet_parser import FacetParser
from boulder_statistics.refinement_plus.qcube_chunk import QCubeChunk
from boulder_statistics.refinement_plus.refinement_chunking import \
    ChunkingTools


@dataclass(frozen=True)
class ProjectFacets():
    mesh_folder: Path
    export_pool_folder: Path
    facet_maps: pl.DataFrame  # For example for TIR then would be data_tir_maps
    measurement_types_of_interest: List[str]
    mission_phase: str | Literal["detailed_survey",
                                 "recona", "reconb", "reconc"]  # The str is just do the type safe linter is happy
    instrument_type: Literal["TIR", "VNIR"]
    dp: DataProductEncyclopedia
    pipeline_from_here_needs_running: bool
    cache_folder: Path = field(default=Path(".cache"))

    def process_mission_phase(self) -> None:
        export_mesh_measurements_lfs: List[pl.LazyFrame] = []

        if self.mission_phase_export_folder().exists() and (
                not self.pipeline_from_here_needs_running):
            return

        for mesh_name in self.facet_maps.filter(
                pl.col("mission_phase") == self.mission_phase)["facet_shape_model_name"].unique():

            mesh_stem: str = Path(mesh_name).stem
            facet_id_lookup: pl.LazyFrame = self.get_facet_id_lookup(mesh_name)

            merged_lf: pl.LazyFrame = (
                facet_id_lookup
                .join(
                    self.get_facet_map_data_for_mesh_and_phase(
                        mesh_name).with_columns(pl.col("facet_num").cast(pl.Float64)).lazy(),
                    on="facet_num"
                )
            )

            self.export_mesh(merged_lf, mesh_stem)

            mesh_measurements_exported_lf: pl.LazyFrame = self.export_mesh_measurements(
                merged_lf, mesh_stem)

            export_mesh_measurements_lfs.append(mesh_measurements_exported_lf)

        merged_mesh_measurements_exported: pl.LazyFrame = pl.concat(
            export_mesh_measurements_lfs, how="vertical")

        print(
            f"Exporting whole mission facet measurements to {
                self.mission_phase_export_folder().name}")

        merged_mesh_measurements_exported.sink_parquet(
            self.mission_phase_export_folder()
        )

    def get_tri_id_lookup(self, verts: np.ndarray,
                          tris: np.ndarray, mesh_stem: str) -> pl.LazyFrame:
        rasterized_tri_id_db_cache_folder: Path = self.cache_folder / \
            f"rasterized_tri_id_lookup_for_{mesh_stem}"

        if rasterized_tri_id_db_cache_folder.exists():
            print(f"Found cached tri_id lookup for mesh {mesh_stem}")
            return pl.scan_parquet(rasterized_tri_id_db_cache_folder)

        project_points_df, project_tris_df = FacetParser.load_mesh(
            verts, tris)

        print(f"Building tri_id lookup for mesh {mesh_stem}")
        tri_id_lookup: pl.LazyFrame = self.rasterize_facets(
            project_points_df,
            project_tris_df,
            rasterized_tri_id_db_cache_folder)

        return tri_id_lookup

    def get_facet_id_lookup(self, mesh_name: str) -> pl.LazyFrame:
        mesh_stem: str = Path(mesh_name).stem
        DTM_path: Path = self.mesh_folder / mesh_name

        rasterized_facet_id_db_cache_folder: Path = self.cache_folder / \
            f"rasterized_facet_id_lookup_for_{mesh_stem}.parquet"

        if rasterized_facet_id_db_cache_folder.exists():
            print(f"Found cached facet_id lookup for mesh {mesh_stem}")
            return pl.scan_parquet(rasterized_facet_id_db_cache_folder)

        verts, tris = self.get_triangle_mesh_data(DTM_path)

        print(f"Creating facet_id lookup for mesh {mesh_stem}")
        tri_id_lookup: pl.LazyFrame = self.get_tri_id_lookup(
            verts, tris, mesh_stem)

        facet_to_tri_lookup: pl.DataFrame = self.get_facet_to_tri_lookup(
            verts, tris, mesh_name)

        facet_id_lookup: pl.LazyFrame = (
            tri_id_lookup
            .join(
                facet_to_tri_lookup.lazy(),
                on="tri_num"
            )
        )

        facet_id_lookup.sink_parquet(rasterized_facet_id_db_cache_folder)

        return facet_id_lookup

    def export_mesh_measurements(self, merged_lf: pl.LazyFrame,
                                 mesh_stem: str) -> pl.LazyFrame:

        measurements_export_path: Path = self.mesh_measurement_export_folder(
            mesh_stem)
        if measurements_export_path.exists() and (
                not self.pipeline_from_here_needs_running):
            print(
                f"Export for measurements {measurements_export_path.stem} already found, skipping")
            return pl.scan_parquet(measurements_export_path)

        print(f"Exporting mesh for {measurements_export_path.stem}")

        (
            merged_lf
            .select(
                ["i", "j", "face"] +
                [measurement_type for measurement_type in self.measurement_types_of_interest]
            ).filter(  # For rows with no usable data remove
                pl.any_horizontal(
                    [
                        (pl.col(col).is_not_null())
                        & (pl.col(col).is_not_nan())
                        & (pl.col(col) != -9999.0)
                        for col in self.measurement_types_of_interest
                    ]
                )
            )
            .with_columns(
                pl.lit(mesh_stem).alias(
                    f"{self.instrument_type} {self.mission_phase} facet mesh")
            )
            .rename(
                {
                    measurement_type: f"{
                        self.instrument_type} {
                        self.mission_phase} {measurement_type}"
                    # self.tri_num_export_col_name not needed as name will
                    # be ok
                    for measurement_type in self.measurement_types_of_interest
                }
            )
            .sink_parquet(
                measurements_export_path, engine="streaming"
            )
        )

        return pl.scan_parquet(measurements_export_path)

    def export_mesh(self, merged_lf: pl.LazyFrame, mesh_stem: str) -> None:

        mesh_export_path: Path = self.mesh_export_folder(mesh_stem)
        if mesh_export_path.exists() and (not self.pipeline_from_here_needs_running):
            print(
                f"Export for mesh {
                    mesh_export_path.stem} already found, skipping")
            return

        print(f"Exporting mesh for {mesh_export_path.stem}")

        (
            merged_lf
            .select(
                ["i", "j", "face", "facet_num"]
            )
            .rename(
                {"facet_num": f"{mesh_stem} facet_id"}
            )
            .sink_parquet(
                mesh_export_path, engine="streaming"
            )
        )

    @property
    def cache_export_folder(self) -> Path:
        return self.cache_folder / \
            f"{self.instrument_type}_{self.mission_phase}"

    def mission_phase_export_folder(self) -> Path:
        return self.export_pool_folder / \
            f"{self.instrument_type}_{self.mission_phase}.parquet"

    def mesh_measurement_export_folder(self, mesh_stem: str) -> Path:
        return self.cache_folder / \
            f"{self.instrument_type}_{self.mission_phase}_{mesh_stem}.parquet"

    def mesh_export_folder(self, mesh_stem: str) -> Path:
        return self.export_pool_folder / f"{mesh_stem}.parquet"

    def tri_id_export_col_name(self, measurement_type: str) -> str:
        return f"{self.instrument_type} {measurement_type} {self.mission_phase} tri_id"

    def rasterize_facets(self, project_points_df: pl.DataFrame,
                         project_tris_df: pl.DataFrame, rasterized_facets_db_folder: Path) -> pl.LazyFrame:

        def handle_chunk(chunk: QCubeChunk) -> List[np.ndarray]:
            return [FacetParser.rasterize_facets(
                project_points_df,
                project_tris_df,
                chunk,
            ).T]  # The transpose is very important!!!

        ChunkingTools.bulk_append_by_chunks(
            target_lf=self.dp.combined_atlas.select("i", "j", "face"),
            export_folder=rasterized_facets_db_folder,
            col_names=["tri_num"],
            process_chunk=handle_chunk,
            chunks=QCubeChunk.generate(depth=1),
        )

        return pl.scan_parquet(rasterized_facets_db_folder)

    def get_facet_map_data_for_mesh_and_phase(
            self, mesh_name: str) -> pl.DataFrame:
        return self.facet_maps.filter(
            pl.col("facet_shape_model_name") == mesh_name,
            pl.col("mission_phase") == self.mission_phase
        )

    def check_valid(self, facet_nums_to_tri_nums_lookup: pl.DataFrame) -> bool:
        # Sanity checks
        if not np.all(
           facet_nums_to_tri_nums_lookup["associate_distance"].to_numpy() < 1e-5):
            print("Facets not lining up, skipping")
            return False

        duplicates = (
            facet_nums_to_tri_nums_lookup
            .group_by("tri_num")
            .len()
            .filter(pl.col("len") > 1)
        )

        if duplicates.height != 0:
            print("duplicate triangles found!, skipping")
            print(duplicates)

            return False

        return True

    def get_facet_to_tri_lookup(
            self, verts: np.ndarray, tris: np.ndarray, mesh_name: str) -> pl.DataFrame:

        facets: pl.DataFrame = self.facet_maps.filter(
            pl.col("facet_shape_model_name") == mesh_name)

        facet_data_arr = (
            facets
            .select(["x", "y", "z", "facet_num"])
            .unique()
            .to_numpy()
        )

        P: np.ndarray = facet_data_arr[:, :3].astype(np.float64)
        facet_nums: np.ndarray = facet_data_arr[:, 3].astype(
            np.float64)  # As the other stuff uses it in f64

        sqrD, tri_idx, closest_points = igl.point_mesh_squared_distance(
            P, verts, tris
        )

        facet_to_tri_lookup = pl.DataFrame(
            {
                "facet_num": facet_nums,
                "tri_num": np.array(tri_idx, dtype=np.float64),
                "associate_distance": np.sqrt(sqrD, dtype=np.float64),
            }
        )

        # Has the facet mesh projection worked
        if not self.check_valid(facet_to_tri_lookup):
            raise ValueError("Failed to check valid")

        return facet_to_tri_lookup

    def get_triangle_mesh_data(
            self, DTM_path: Path) -> Tuple[np.ndarray, np.ndarray]:

        V, F = igl.read_triangle_mesh(DTM_path)
        return V, F

    def mission_phase_cache_folder(self) -> Path:
        return self.cache_folder / self.mission_phase
