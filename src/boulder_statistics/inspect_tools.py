import textwrap
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars import DataFrame, LazyFrame


@dataclass(frozen=True, kw_only=True)
class QMAPInspector():
    full_db: LazyFrame = field(
        default_factory=lambda: pl.scan_parquet("combined_db"))
    agg_db: DataFrame = field(
        default_factory=lambda: pl.read_parquet("combined_db_agg.parquet"))

    def get_boulder_id_df(self, boulder_id: int) -> DataFrame:
        return self.full_db.filter(
            pl.col("boulder_id") == boulder_id).collect()

    def get_full_df_around_boulder(
            self, boulder_id: int, zoom_factor: float = 1.5):
        boulder_df = self.get_boulder_id_df(boulder_id)

        min_i, max_i = int(boulder_df["i"].min()), int(boulder_df["i"].max())
        min_j, max_j = int(boulder_df["j"].min()), int(boulder_df["j"].max())

        center_i = (min_i + max_i) / 2
        center_j = (min_j + max_j) / 2

        half_width = (max_i - min_i) * zoom_factor / 2
        half_height = (max_j - min_j) * zoom_factor / 2

        return self.full_db.filter(
            (pl.col("i") >= center_i - half_width)
            & (pl.col("i") <= center_i + half_width)
            & (pl.col("j") >= center_j - half_height)
            & (pl.col("j") <= center_j + half_height)
        ).collect()

    def render_column_around_boulder(
        self, boulder_id: int, zoom_factor: float, column_name: str
    ) -> np.ndarray:
        boulder_area_df: DataFrame = self.get_full_df_around_boulder(
            boulder_id, zoom_factor)

        i_min = int(boulder_area_df["i"].min())
        i_max = int(boulder_area_df["i"].max())
        j_min = int(boulder_area_df["j"].min())
        j_max = int(boulder_area_df["j"].max())

        arr = np.full(
            (i_max - i_min + 1, j_max - j_min + 1),
            np.nan,
            dtype=np.float32,
        )

        arr[
            boulder_area_df["i"].to_numpy() - i_min,
            boulder_area_df["j"].to_numpy() - j_min,
        ] = boulder_area_df[column_name].to_numpy()

        return arr

    def render_multi_plot(
        self,
        boulder_id: int,
        left_view_column: str,
        right_view_column: str = "32bit_reflectance",
        zoom_factor: float = 1.5,
    ) -> None:
        left = self.render_column_around_boulder(
            boulder_id, zoom_factor, left_view_column
        )
        right = self.render_column_around_boulder(
            boulder_id, zoom_factor, right_view_column
        )

        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(10, 7)
        )

        im_left = ax_left.imshow(left, origin="lower")
        ax_left.set_title(left_view_column)
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        im_right = ax_right.imshow(right, origin="lower")
        ax_right.set_title(right_view_column)
        ax_right.set_xticks([])
        ax_right.set_yticks([])

        fig.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)
        fig.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)

        # Get boulder metadata
        metadata_df = self.agg_db.filter(
            pl.col("boulder_id") == boulder_id
        )

        metadata_items = [
            f"Boulder ID: {boulder_id}",
            f"Zoom factor: {zoom_factor:.2f}",
        ]

        if metadata_df.height > 0:
            row = metadata_df.row(0, named=True)

            for key, value in row.items():
                if isinstance(value, (float, np.floating)):
                    value = f"{value:.3f}"
                metadata_items.append(f"{key}: {value}")

        metadata_text = " | ".join(metadata_items)

        # Estimate available characters from figure width
        fig_width_inches = fig.get_size_inches()[0]
        font_size = 8
        chars_per_inch = 12  # approximate readable density
        wrap_width = int(fig_width_inches * chars_per_inch)

        metadata_text = textwrap.fill(
            metadata_text,
            width=wrap_width,
            break_long_words=False,
        )

        n_lines = metadata_text.count("\n") + 1

        # Automatically reserve only the required space
        line_height = font_size / 72 * 1.2
        bottom_margin = line_height * n_lines + 0.01

        fig.text(
            0.5,
            bottom_margin - 0.01,
            metadata_text,
            ha="center",
            va="bottom",
            fontsize=font_size,
        )

        plt.tight_layout(rect=[0, bottom_margin + 0.03, 1, 1])
        plt.show()
