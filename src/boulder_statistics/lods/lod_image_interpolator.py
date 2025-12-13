from typing import TypeVar

import cv2
import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.floating)


class LODImageInterpolator:

    @staticmethod
    def interpolate_image[T: np.floating](
            interpolation_resolution: int, img_A: NDArray[T], img_B: NDArray[T],
            img_C: NDArray[T], img_D: NDArray[T], interpolation: int = cv2.INTER_AREA) -> NDArray[T]:

        if len({arr.shape for arr in [img_A, img_B, img_C, img_D]}) != 1:
            raise Exception("Image shape miss match")

        top_stack: NDArray[T] = np.hstack((img_A, img_B))
        bottom_stack: NDArray[T] = np.hstack((img_C, img_D))
        full_stack: NDArray[T] = np.vstack(
            (top_stack, bottom_stack))

        full_stack_interpolated: NDArray[T] = cv2.resize(
            full_stack, dsize=(interpolation_resolution, interpolation_resolution), interpolation=interpolation)

        return full_stack_interpolated
