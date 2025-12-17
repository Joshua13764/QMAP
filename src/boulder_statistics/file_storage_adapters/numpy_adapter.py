from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from attr import field
from numpy.typing import NDArray

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk

matplotlib.use("Agg")
plt.style.use('science')
plt.rcParams["figure.figsize"] = (7, 7 * ((5**0.5 - 1) / 2))
DPI = 600
plt.rcParams["figure.dpi"] = 600
plt.ioff()


@dataclass(frozen=True)
class FSNumpyAdapter(FSAdapterBase[NDArray[Any], FSPathLocalDisk]):
    export_debug_plots: bool = field(default=False)
    standard_extension: ClassVar[str] = "npy"

    # --- Plot settings ---
    title: str = field(default="Default title")
    colour_bar_title: str = field(default="Default colour bar title")
    transform: Callable[[NDArray[Any]],
                        NDArray[Any]] = field(default=lambda x: x)
    plot_standard_extension: ClassVar[str] = "png"

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        return np.load(path.actual_path.as_posix())

    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:
        if self.export_debug_plots:
            self.export_debug_plot(obj, path)

        return np.save(path.actual_path.as_posix(), obj)

    def export_debug_plot(
            self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(obj, origin="upper",
                       extent=[0, 1, 0, 1], aspect="equal")

        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(self.title)
        fig.colorbar(im, ax=ax, label=self.colour_bar_title)
        fig.tight_layout()

        fig.savefig(
            path.actual_path.as_posix().replace(
                self.standard_extension, self.plot_standard_extension),
            dpi=DPI,
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close(fig)
