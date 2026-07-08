from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x_min: np.float64
    x_max: np.float64
    y_min: np.float64
    y_max: np.float64
    z_min: np.float64
    z_max: np.float64

    @property
    def width(self) -> np.float64:
        return np.float64(self.x_max - self.x_min)

    @property
    def height(self) -> np.float64:
        return np.float64(self.y_max - self.y_min)

    @property
    def depth(self) -> np.float64:
        return np.float64(self.z_max - self.z_min)

    @property
    def center(self) -> tuple[np.float64, np.float64, np.float64]:
        return (
            np.float64((self.x_min + self.x_max) / 2),
            np.float64((self.y_min + self.y_max) / 2),
            np.float64((self.z_min + self.z_max) / 2),
        )

    @property
    def bounding_sphere_radius(self) -> np.float64:
        return np.float64(
            np.sqrt(
                (self.width / 2) ** 2
                + (self.height / 2) ** 2
                + (self.depth / 2) ** 2
            )
        )
