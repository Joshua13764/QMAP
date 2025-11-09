from typing import List

import polars as pl

FACES: List[str] = ["posx", "negx", "posy", "negy", "posz", "negz"]
POINT_ATTRS: List[str] = ["x", "y", "z"]
PROJECTED_POINT_ATTRS: List[str] = ["u", "v", "N"]
VERT_ID_COLS: List[str] = ["0", "1", "2"]


class Polars3DExpressions:

    @staticmethod
    def filter_faces_for_rasterization(
            tris: pl.DataFrame, face: str, eps_uv=1e-12) -> pl.DataFrame:
        ensure_not_back_facing: pl.Expr = (
            (pl.col(f"{face}_N0") > 0) &
            (pl.col(f"{face}_N1") > 0) &
            (pl.col(f"{face}_N2") > 0)
        )

        ensure_non_degenerate: pl.Expr = pl.col(f"{face}_area_uv") > eps_uv

        return tris.filter(ensure_not_back_facing & ensure_non_degenerate)

    @staticmethod
    def process_mesh(points: pl.DataFrame,
                     tris: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:

        points = points.with_columns(
            Polars3DExpressions.get_project_points_expression())

        tris = tris.with_columns(
            Polars3DExpressions.get_gather_points_to_tris_expression(points))

        tris = tris.with_columns(
            Polars3DExpressions.get_gather_unprojected_tri_positions_expressions(points))

        tris = tris.with_columns(
            Polars3DExpressions.get_calculate_unprojected_tri_area_expression())

        tris = tris.with_columns(
            Polars3DExpressions.get_calculate_projected_tri_area_expressions())

        tris = tris.with_columns(
            Polars3DExpressions.get_calculate_tri_area_ratios_expressions())

        return (points, tris)

    @staticmethod
    def get_project_points_expression() -> List[pl.Expr]:
        """Projects a polars dataframe with headers of "x", "y", "z" onto a cube of side length 2 centered at the origin

        Returns:
            List[pl.Expr]:  A list of expressions to perform the action
        """
        x, y, z = pl.col("x"), pl.col("y"), pl.col("z")
        sx, sy, sz = x.abs(), y.abs(), z.abs()

        return [
            (0.5 * ((-z / sx) + 1.0)).alias("posx_u"),
            (0.5 * ((y / sx) + 1.0)).alias("posx_v"),
            x.alias("posx_N"),

            # -X face (normal = (-1,0,0))
            (0.5 * ((z / sx) + 1.0)).alias("negx_u"),
            (0.5 * ((y / sx) + 1.0)).alias("negx_v"),
            (-x).alias("negx_N"),

            # +Y face (normal = (0,1,0))
            (0.5 * ((x / sy) + 1.0)).alias("posy_u"),
            (0.5 * ((-z / sy) + 1.0)).alias("posy_v"),
            y.alias("posy_N"),

            # -Y face (normal = (0,-1,0))
            (0.5 * ((x / sy) + 1.0)).alias("negy_u"),
            (0.5 * ((z / sy) + 1.0)).alias("negy_v"),
            (-y).alias("negy_N"),

            # +Z face (normal = (0,0,1))
            (0.5 * ((x / sz) + 1.0)).alias("posz_u"),
            (0.5 * ((y / sz) + 1.0)).alias("posz_v"),
            z.alias("posz_N"),

            # -Z face (normal = (0,0,-1))
            (0.5 * ((-x / sz) + 1.0)).alias("negz_u"),
            (0.5 * ((y / sz) + 1.0)).alias("negz_v"),
            (-z).alias("negz_N")
        ]

    @staticmethod
    def get_gather_points_to_tris_expression(
            points: pl.DataFrame) -> List[pl.Expr]:
        """Finds all of the points for each of the tris and adds their u, v positions to columns of tris

        Args:
            points (pl.DataFrame): The data frame containing the points with headers of "x", "y", "z"

        Returns:
            List[pl.Expr]: A list of expressions to perform the action
        """

        return [
            pl.lit(
                points.get_column(
                    f"{face}_{projected_point_attr}")).gather(
                pl.col(vert_id).cast(
                    pl.UInt32)).alias(
                f"{face}_{projected_point_attr}{vert_id}")
            for projected_point_attr in PROJECTED_POINT_ATTRS
            for vert_id in VERT_ID_COLS
            for face in FACES
        ]

    @staticmethod
    def get_gather_unprojected_tri_positions_expressions(
            points: pl.DataFrame) -> List[pl.Expr]:
        """Finds all of the points for each of the tris and adds their pre-projection x, y, z positions to columns of tris

        Args:
            points (pl.DataFrame): The data frame containing the points with headers of "x", "y", "z"

        Returns:
            List[pl.Expr]: A list of expressions to perform the action
        """
        return [
            pl.lit(
                points.get_column(point_attr)).gather(
                pl.col(vert_id).cast(
                    pl.UInt32)).alias(
                f"{point_attr}{vert_id}")
            for point_attr in POINT_ATTRS
            for vert_id in VERT_ID_COLS
        ]

    @staticmethod
    def get_calculate_unprojected_tri_area_expression() -> pl.Expr:
        """Find all of the area's of the tris before they were projected

        Returns:
            pl.Expr: Expression to carry out the above task
        """
        # 3D edges from vertex 0
        dx10: pl.Expr = pl.col("x1") - pl.col("x0")
        dy10: pl.Expr = pl.col("y1") - pl.col("y0")
        dz10: pl.Expr = pl.col("z1") - pl.col("z0")

        dx20: pl.Expr = pl.col("x2") - pl.col("x0")
        dy20: pl.Expr = pl.col("y2") - pl.col("y0")
        dz20: pl.Expr = pl.col("z2") - pl.col("z0")

        # Cross product and 3D area
        cx: pl.Expr = dy10 * dz20 - dz10 * dy20
        cy: pl.Expr = dz10 * dx20 - dx10 * dz20
        cz: pl.Expr = dx10 * dy20 - dy10 * dx20
        return (0.5 * (cx * cx + cy * cy + cz *
                       cz).sqrt()).alias("area_xyz")

    @staticmethod
    def get_calculate_projected_tri_area_expressions() -> List[pl.Expr]:
        """Find all of the area's of the tris after they were projected

        Returns:
            List[pl.Expr]: Expression to carry out the above task
        """

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
        """Find all of the area ratios (xyz_area / uv_area)

        Returns:
            List[pl.Expr]: Expression to carry out the above task
        """
        return [
            pl.when(pl.col(f"{face}_area_uv") > 1e-12)
            .then(pl.col("area_xyz") / pl.col(f"{face}_area_uv"))
            .otherwise(None)
            .alias(f"{face}_ratio")
            for face in FACES
        ]
