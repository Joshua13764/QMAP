from dataclasses import dataclass, field

from polars import LazyFrame, scan_parquet

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSPolarsLazyParquetAdapter(FSAdapterBase[LazyFrame, FSPathLocalDisk]):
    standard_extension: str | None | bool = field(default="parquet")

    def read(self, path: FSPathLocalDisk) -> LazyFrame:
        return scan_parquet(source=path.actual_path)

    def write(self, obj: LazyFrame, path: FSPathLocalDisk) -> None:
        obj.sink_parquet(path.actual_path)
