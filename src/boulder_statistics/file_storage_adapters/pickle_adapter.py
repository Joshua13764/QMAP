import pickle
from dataclasses import dataclass
from typing import Any

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSPickleAdapter(FSAdapterBase[Any, FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> Any:
        with path.actual_path.open("rb") as f:
            return pickle.load(f)

    def write(self, obj: Any, path: FSPathLocalDisk) -> None:
        path.make_directory()
        with path.actual_path.open("wb") as f:
            pickle.dump(obj, f)
