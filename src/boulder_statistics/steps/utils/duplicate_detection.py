import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Duplicate_Detection:
    @staticmethod
    def remove_near_duplicate_detections(
        df: pd.DataFrame,
        offset_col: str = "relative_offset",
        size_col: str = "fixed_weight_area",
        size_similarity: float = 0.8,   # 80%
        offset_threshold: float = 0.2,  # distance in relative-offset space
    ) -> pd.DataFrame:
        """
        Remove near-duplicate detections based on:
        - detection size similarity: smaller / larger >= size_similarity
        - relative offset closeness: ||offset_i - offset_j|| <= offset_threshold

        Keeps the larger detection (by `size_col`) when duplicates occur.
        Assumes `df[offset_col]` contains np.array([x_rel, y_rel]) for each row.
        """
        if df.empty:
            return df

        df_sorted = df.copy().sort_values(
            size_col, ascending=False).reset_index(
            drop=True)

        # Stack relative_offset arrays into an (N, 2) array
        offsets: NDArray[np.float64] = np.vstack(
            df_sorted[offset_col].to_numpy())
        sizes: NDArray[np.float64] = df_sorted[size_col].to_numpy(dtype=float)

        n = len(df_sorted)
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            if not keep[i]:
                continue

            off_i = offsets[i]
            size_i = sizes[i]

            for j in range(i + 1, n):
                if not keep[j]:
                    continue

                off_j = offsets[j]
                size_j = sizes[j]

                # 1) size similarity
                size_ratio = min(size_i, size_j) / max(size_i, size_j)
                if size_ratio < size_similarity:
                    continue

                # 2) relative-offset closeness (L2)
                dist = np.linalg.norm(off_i - off_j)
                if dist <= offset_threshold:
                    keep[j] = False

        return df_sorted[keep].reset_index(drop=True)
