from typing import List

import polars as pl
from polars import Expr

from boulder_statistics.file_storage_adapters.adapter_custom_classes.PL_obj_data import \
    PLOBJData

FACES: List[str] = ["posx", "negx", "posy", "negy", "posz", "negz"]
POINT_ATTRS: List[str] = ["x", "y", "z"]
PROJECTED_POINT_ATTRS: List[str] = ["u", "v", "N"]
VERT_ID_COLS: List[str] = ["0", "1", "2"]


class BetterPolars3DExpressions:

    @staticmethod
    def process_mesh_projection_scaling(
            shape: PLOBJData) -> PLOBJData:
        # points already have a vertex id column "vid" that tris["0"/"1"/"2"]
        # reference.

        # points =

        points: pl.LazyFrame = BetterPolars3DExpressions._project_points(
            shape.verts)
        tris: pl.LazyFrame = BetterPolars3DExpressions._attach_points_to_tris(
            points, shape.tris)
        tris: pl.LazyFrame = BetterPolars3DExpressions._add_area_and_ratio_columns(
            tris)

        return PLOBJData(verts=points, tris=tris)

    @staticmethod
    def process_mesh_displacement(points: pl.LazyFrame,
                                  tris: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:

        points = BetterPolars3DExpressions._project_points(points)
        tris = BetterPolars3DExpressions._attach_points_to_tris(points, tris)

        return points, tris

    @staticmethod
    def _project_points(points: pl.LazyFrame) -> pl.LazyFrame:
        """Add cubemap projection columns to points."""
        # NOTE: read() already did .with_row_index("vid"), so we assume "vid"
        # exists.

        points_mapped: pl.LazyFrame = points.with_columns(
            BetterPolars3DExpressions.get_project_points_expression()
        )

        points_mapped_calibrated: pl.LazyFrame = points_mapped.with_columns(
            BetterPolars3DExpressions.get_project_points_calibration()
        )

        return points_mapped_calibrated

    @staticmethod
    def _attach_points_to_tris(
        points: pl.LazyFrame,
        tris: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Attach xyz and projected attributes for each triangle vertex."""
        for vert_id in VERT_ID_COLS:
            tris = BetterPolars3DExpressions._join_points_for_vertex(
                points, tris, vert_id
            )
        return tris

    @staticmethod
    def _join_points_for_vertex(
        points: pl.LazyFrame,
        tris: pl.LazyFrame,
        vert_id: str,
    ) -> pl.LazyFrame:
        """Join in point data for a single vertex index column (e.g. '0')."""
        suffix: str = vert_id

        # IMPORTANT CHANGE: keep 'vid' as-is, don't alias to 'vid0'/etc
        selection: List[Expr] = [pl.col("vid")]

        # original xyz -> x0,y0,z0 or x1,y1,z1, ...
        selection.extend(
            pl.col(attr).alias(f"{attr}{suffix}")
            for attr in POINT_ATTRS
        )

        # projected attributes per face -> posx_u0, posx_v0, posx_N0, ...
        for face in FACES:
            selection.extend(
                pl.col(f"{face}_{p_attr}").alias(f"{face}_{p_attr}{suffix}")
                for p_attr in PROJECTED_POINT_ATTRS
            )

        points_for_vert: pl.LazyFrame = points.select(selection)

        # IMPORTANT CHANGE: join on 'vid', not 'vid0'/etc
        return tris.join(
            points_for_vert,
            left_on=vert_id,
            right_on="vid",
            how="left",
        )
        # we can keep the 'vid' column; it won't hurt, and we don't rely on it
        # later

    @staticmethod
    def add_displacement_columns(shape: PLOBJData) -> PLOBJData:

        tris: pl.LazyFrame = shape.tris.with_columns(
            BetterPolars3DExpressions.get_mean_radius(),
        )

        return PLOBJData(verts=shape.verts, tris=tris)

    @staticmethod
    def get_mean_radius() -> pl.Expr:
        x_tri_mean: Expr = (pl.col("x0") + pl.col("x1") +
                            pl.col("x2")) * (1 / 3)
        y_tri_mean: Expr = (pl.col("y0") + pl.col("y1") +
                            pl.col("y2")) * (1 / 3)
        z_tri_mean: Expr = (pl.col("z0") + pl.col("z1") +
                            pl.col("z2")) * (1 / 3)

        r_tri_mean: Expr = (
            x_tri_mean ** 2 +
            y_tri_mean ** 2 +
            z_tri_mean ** 2) ** 0.5

        return r_tri_mean.alias("r_tri_mean")

    @staticmethod
    def get_mean_projected_radius() -> pl.Expr:

        x_mean: Expr = (pl.col("x0") + pl.col("x1") + pl.col("x2")) * (1 / 3)
        y_mean: Expr = (pl.col("y0") + pl.col("y1") + pl.col("y2")) * (1 / 3)
        z_mean: Expr = (pl.col("z0") + pl.col("z1") + pl.col("z2")) * (1 / 3)

        r_mean_projected: Expr = BetterPolars3DExpressions.get_mean_radius() / \
            pl.max_horizontal(x_mean.abs(), y_mean.abs(), z_mean.abs())

        return r_mean_projected

    @staticmethod
    def _add_area_and_ratio_columns(tris: pl.LazyFrame) -> pl.LazyFrame:
        """Add area_xyz, face uv areas, and area ratios."""

        # 1) First compute area_xyz and all <face>_area_uv columns
        tris = tris.with_columns(
            BetterPolars3DExpressions.get_calculate_unprojected_tri_area_expression(),
            *BetterPolars3DExpressions.get_calculate_projected_tri_area_expressions(),
        )

        # 2) Then compute <face>_ratio using those columns
        tris = tris.with_columns(
            *BetterPolars3DExpressions.get_calculate_tri_area_ratios_expressions(),
        )

        return tris

    @staticmethod
    def process_extra_mesh_data(points: pl.LazyFrame,
                                tris: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:

        r_mean: Expr = BetterPolars3DExpressions.get_mean_radius()
        r_mean_projected: Expr = BetterPolars3DExpressions.get_mean_projected_radius()

        tris = tris.with_columns(r_mean.alias("xyz_radius"))
        tris = tris.with_columns(
            r_mean_projected.alias("projected_radius"))
        tris = tris.with_columns(
            (r_mean / r_mean_projected).alias("radius_ratio"))  # r_before / r_after

        angle_exprs: List[Expr] = []
        for face in FACES:

            angle: Expr = ((pl.col("radius_ratio") ** 2)
                           * (1 / pl.col(f"{face}_ratio")))
            angle_exprs.append(
                angle.alias(f"{face}_cos(angle)")
            )

        tris = tris.with_columns(angle_exprs)

        return (points, tris)

    @staticmethod
    def get_project_points_expression() -> List[pl.Expr]:
        """Projects directions (x, y, z) onto a cubemap (faces posx/negx/...) with
        UV in [0, 1] matching sample_face_roi. Calibration has been applied"""
        x, y, z = pl.col("x"), pl.col("y"), pl.col("z")
        sx, sy, sz = x.abs(), y.abs(), z.abs()

        return [
            (0.5 * ((-z / sx) + 1.0)).alias("posx_u"),
            (0.5 * ((-y / sx) + 1.0)).alias("posx_v"),
            x.alias("posx_N"),

            (0.5 * ((z / sx) + 1.0)).alias("negx_u"),
            (0.5 * ((-y / sx) + 1.0)).alias("negx_v"),
            (-x).alias("negx_N"),

            (0.5 * ((x / sy) + 1.0)).alias("posy_u"),
            (0.5 * ((z / sy) + 1.0)).alias("posy_v"),
            y.alias("posy_N"),

            (0.5 * ((x / sy) + 1.0)).alias("negy_u"),
            (0.5 * ((-z / sy) + 1.0)).alias("negy_v"),
            (-y).alias("negy_N"),

            (0.5 * ((x / sz) + 1.0)).alias("posz_u"),
            (0.5 * ((-y / sz) + 1.0)).alias("posz_v"),
            z.alias("posz_N"),

            (0.5 * ((-x / sz) + 1.0)).alias("negz_u"),
            (0.5 * ((-y / sz) + 1.0)).alias("negz_v"),
            (-z).alias("negz_N"),
        ]

    @staticmethod
    def get_project_points_calibration() -> List[pl.Expr]:
        "Ensures that consistent with other LOD export conventions"

        # Calibration
        # LAS negy(Vflip) = LOD posx
        # LAS negz(90 ccw, Vflip) = LOD negy ->
        # LAS posx(90 ccw, Hflip) = LOD negz ->
        # LAS posz(90cw, FlipH) = LOD posy ->
        # LAS posy(FlipH) = LOD negx
        # LAS negx(90cw, FlipH) = LOD posz -> (flipH + flipV)

        return [

            pl.col("negy_u").alias("posx_u"),
            (1.0 - pl.col("negy_v")).alias("posx_v"),
            pl.col("negy_N").alias("posx_N"),

            pl.col("negz_v").alias("negy_u"),
            pl.col("negz_u").alias("negy_v"),
            pl.col("negz_N").alias("negy_N"),

            (1 - pl.col("posx_v")).alias("negz_u"),
            (1 - pl.col("posx_u")).alias("negz_v"),
            pl.col("posx_N").alias("negz_N"),

            pl.col("posz_v").alias("posy_u"),
            pl.col("posz_u").alias("posy_v"),
            pl.col("posz_N").alias("posy_N"),

            (1.0 - pl.col("posy_u")).alias("negx_u"),
            pl.col("posy_v").alias("negx_v"),
            pl.col("posy_N").alias("negx_N"),

            (pl.col("negx_v")).alias("posz_u"),
            (pl.col("negx_u")).alias("posz_v"),
            pl.col("negx_N").alias("posz_N"),
        ]

    @staticmethod
    def get_calculate_unprojected_tri_area_expression() -> pl.Expr:
        dx10: pl.Expr = pl.col("x1") - pl.col("x0")
        dy10: pl.Expr = pl.col("y1") - pl.col("y0")
        dz10: pl.Expr = pl.col("z1") - pl.col("z0")

        dx20: pl.Expr = pl.col("x2") - pl.col("x0")
        dy20: pl.Expr = pl.col("y2") - pl.col("y0")
        dz20: pl.Expr = pl.col("z2") - pl.col("z0")

        cx: pl.Expr = dy10 * dz20 - dz10 * dy20
        cy: pl.Expr = dz10 * dx20 - dx10 * dz20
        cz: pl.Expr = dx10 * dy20 - dy10 * dx20
        return (0.5 * (cx * cx + cy * cy + cz *
                       cz).sqrt()).alias("area_xyz")

    @staticmethod
    def get_calculate_projected_tri_area_expressions() -> List[pl.Expr]:
        expressions: List[pl.Expr] = []

        for face in FACES:
            u0, v0 = pl.col(f"{face}_u0"), pl.col(f"{face}_v0")
            u1, v1 = pl.col(f"{face}_u1"), pl.col(f"{face}_v1")
            u2, v2 = pl.col(f"{face}_u2"), pl.col(f"{face}_v2")
            uv_area: pl.Expr = 0.5 * ((u1 - u0) * (v2 - v0) -
                                      (u2 - u0) * (v1 - v0)).abs()
            expressions.append(uv_area.alias(f"{face}_area_uv"))

        return expressions

    @staticmethod
    def get_calculate_tri_area_ratios_expressions() -> List[pl.Expr]:
        return [
            pl.when(pl.col(f"{face}_area_uv") > 1e-12)
            .then(pl.col("area_xyz") / pl.col(f"{face}_area_uv"))
            .otherwise(None)
            .alias(f"{face}_ratio")
            for face in FACES
        ]
