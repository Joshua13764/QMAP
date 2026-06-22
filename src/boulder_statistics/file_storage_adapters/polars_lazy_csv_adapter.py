from dataclasses import dataclass, field

from polars import LazyFrame, scan_csv

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSPolarsLazyCSVAdapter(FSAdapterBase[LazyFrame, FSPathLocalDisk]):
    standard_extension: str | None | bool = field(default="csv")

    def read(self, path: FSPathLocalDisk) -> LazyFrame:
        return scan_csv(source=path.actual_path)

    def write(self, obj: LazyFrame, path: FSPathLocalDisk) -> None:
        obj.sink_csv(path.actual_path)
