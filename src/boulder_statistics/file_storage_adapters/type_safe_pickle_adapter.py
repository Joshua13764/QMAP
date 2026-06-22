import pickle
from dataclasses import dataclass, field
from typing import Any, Type

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True, kw_only=True)
class FSTypeSafePickleAdapter[T](FSAdapterBase[T, FSPathLocalDisk]):
    expected_type: Type[T]
    standard_extension: str | None | bool = field(default="pkl")

    def read(self, path: FSPathLocalDisk) -> Any:
        with path.actual_path.open("rb") as f:
            obj: Any = pickle.load(f)

        assert isinstance(obj, self.expected_type)
        return obj

    def write(self, obj: T, path: FSPathLocalDisk) -> None:
        with path.actual_path.open("wb") as f:
            pickle.dump(obj, f)
