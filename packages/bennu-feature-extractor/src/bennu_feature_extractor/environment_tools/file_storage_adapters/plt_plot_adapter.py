from dataclasses import dataclass
from typing import Any

from matplotlib.figure import Figure

from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass()
class FSPltPlotAdapter(FSAdapterBase[Figure, FSPathLocalDisk]):
    dpi: int

    def read(self, path: FSPathLocalDisk) -> Figure:
        raise NotImplementedError()

    def write(self, fig: Figure, path: FSPathLocalDisk) -> None:

        fig.savefig(
            path.actual_path.as_posix(),
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor="white")
