from dataclasses import dataclass

from matplotlib.figure import Figure

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSPltPlotAdapter(FSAdapterBase[Figure, FSPathLocalDisk]):
    dpi: int

    def read(self, path: FSPathLocalDisk) -> Figure:
        raise NotImplementedError()

    def write(self, obj: Figure, path: FSPathLocalDisk) -> None:

        obj.savefig(
            path.actual_path.as_posix(),
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor="white")
