from dataclasses import dataclass

from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_memory import \
    FSPathMemory


@dataclass(frozen=True)
class FSMemoryAdapter[ObjType](FSAdapterBase[ObjType, FSPathMemory]):
    def read(self, path: FSPathMemory) -> ObjType:
        return path.obj

    def write(self, obj: ObjType, path: FSPathMemory) -> None:
        pass
