from dataclasses import dataclass, field
from os import rename
from pathlib import Path
from typing import List, Literal, Tuple

import igl
import numpy as np
import polars as pl
from sympy import false

from boulder_statistics.analysis.data_product_encyclopedia import \
    DataProductEncyclopedia
from boulder_statistics.refinement_plus.bulk_parse_data_tir_maps import \
    FACET_SHAPE_MODELS as TIR_FACET_SHAPE_MODELS
from boulder_statistics.refinement_plus.bulk_parse_data_vnir_maps import \
    FACET_SHAPE_MODELS as VNIR_FACET_SHAPE_MODELS
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
    mission_phase: Literal["detailed_survey", "recona", "reconb", "reconc"]
    instrument_type: Literal["TIR", "VNIR"]
    dp: DataProductEncyclopedia
    cache_folder: Path = field(default=Path(".cache"))

    def process_mission_phase(self) -> None | Path:
        verts, tris = self.get_triangle_mesh_data()
        facet_to_tri_lookup: pl.DataFrame = self.get_facet_to_tri_lookup(
            verts, tris)

        # Has the facet mesh projection worked
        if not self.check_valid(facet_to_tri_lookup):
            return

        project_points_df, project_tris_df = FacetParser.load_mesh(verts, tris)
        self.rasterize_facets(project_points_df, project_tris_df)

        tris_facets_joined: pl.DataFrame = (
            project_tris_df
            .join(
                facet_to_tri_lookup,
                on="tri_num"
            )
            .join(
                self.get_facets(),
                on="facet_num"
            )
        ).with_columns(
            pl.col("tri_num").cast(
                pl.Float64).alias(
                self.tri_num_export_col_name)
        ).select(
            self.measurement_types_of_interest +
            [self.tri_num_export_col_name]
        )

        pl.scan_parquet(self.cache_export_folder).join(
            tris_facets_joined.lazy(),
            on=self.tri_num_export_col_name,
            how="left"
        ).rename(
            {
                name: f"{self.instrument_type} {self.mission_phase} {name}"
                # self.tri_num_export_col_name not needed as name will be ok
                for name in self.measurement_types_of_interest
            }
        ).sink_parquet(
            self.result_export_folder, engine="streaming"
        )

        return self.result_export_folder

    @property
    def cache_export_folder(self) -> Path:
        return self.cache_folder / \
            f"{self.instrument_type}_{self.mission_phase}"

    @property
    def result_export_folder(self) -> Path:
        return self.export_pool_folder / \
            f"{self.instrument_type}_{self.mission_phase}.parquet"

    @property
    def tri_num_export_col_name(self) -> str:
        return f"{self.instrument_type} {self.mission_phase} tri_num"

    def rasterize_facets(self, project_points_df: pl.DataFrame,
                         project_tris_df: pl.DataFrame) -> None:
        def handle_chunk(chunk: QCubeChunk) -> List[np.ndarray]:
            return [FacetParser.rasterize_facets(
                project_points_df,
                project_tris_df,
                chunk,
            ).T]  # The transpose is very important!!!

        ChunkingTools.bulk_append_by_chunks(
            self.dp.combined_atlas.select("i", "j", "face"),
            self.cache_export_folder,
            [self.tri_num_export_col_name],
            handle_chunk,
            chunks=QCubeChunk.generate(depth=1),
        )

    @property
    def facet_shape_model_path(self) -> Path:
        match self.instrument_type:
            case "TIR":
                return self.mesh_folder / \
                    TIR_FACET_SHAPE_MODELS[self.mission_phase]
            case "VNIR":
                return self.mesh_folder / \
                    VNIR_FACET_SHAPE_MODELS[self.mission_phase]

    def get_facets(self) -> pl.DataFrame:
        return self.facet_maps.filter(
            pl.col("mission_phase") == self.mission_phase
        )

    def check_valid(self, facet_nums_to_tri_nums_lookup: pl.DataFrame) -> bool:
        # Sanity checks
        if not np.all(
            facet_nums_to_tri_nums_lookup["associate_distance"].to_numpy()
            < 1e-5
        ):
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
            self, verts: np.ndarray, tris: np.ndarray) -> pl.DataFrame:

        facets: pl.DataFrame = self.get_facets()

        facet_data_arr = (
            facets
            .select(["x", "y", "z", "facet_num"])
            .to_numpy()
        )

        P: np.ndarray = facet_data_arr[:, :3].astype(np.float64)
        facet_nums: np.ndarray = facet_data_arr[:, 3].astype(np.int32)

        sqrD, tri_idx, closest_points = igl.point_mesh_squared_distance(
            P, verts, tris
        )

        facet_nums_to_tri_nums_lookup = pl.DataFrame(
            {
                "facet_num": facet_nums,
                "tri_num": np.array(tri_idx, dtype=np.int32),
                "associate_distance": np.sqrt(sqrD, dtype=np.float64),
            }
        )

        return facet_nums_to_tri_nums_lookup

    def get_triangle_mesh_data(self) -> Tuple[np.ndarray, np.ndarray]:
        V, F = igl.read_triangle_mesh(self.facet_shape_model_path)
        return V, F

    def mission_phase_cache_folder(self) -> Path:
        return self.cache_folder / self.mission_phase
