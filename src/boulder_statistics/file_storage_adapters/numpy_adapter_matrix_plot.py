from dataclasses import dataclass, field
from typing import Any, ClassVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk

# Plot settings
matplotlib.use("Agg")
plt.style.use('science')
plt.rcParams["figure.figsize"] = (7, 7 * ((5**0.5 - 1) / 2))
DPI = 600
plt.rcParams["figure.dpi"] = 600
plt.ioff()


@dataclass(frozen=True)
class FSNumpyAdapterMatrixPlot(FSAdapterBase[NDArray[Any], FSPathLocalDisk]):
    title: str
    colour_bar_title: str

    standard_extension: ClassVar[str] = "png"

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        return np.load(path.actual_path.as_posix())

    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(obj, origin="upper",
                       extent=[0, 1, 0, 1], aspect="equal")

        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(self.title)
        fig.colorbar(im, ax=ax, label=self.colour_bar_title)
        fig.tight_layout()

        fig.savefig(
            path.actual_path.as_posix(),
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close(fig)
