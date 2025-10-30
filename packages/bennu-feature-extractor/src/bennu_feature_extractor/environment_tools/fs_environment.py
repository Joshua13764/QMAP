from typing import Set
import attr
from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import FSAdapterBase
from bennu_feature_extractor.environment_tools.file_storage_persists.runtime_only_persist import 
from bennu_feature_extractor.environment_tools.base_classes.fs_path_base import FSPathBase

@attr.define()
class FSEnvironment():
    paths : Set[FSPathBase] = set()

    def save[ObjType, PathType : FSPathBase](self, obj : ObjType, path : PathType, adapter : FSAdapterBase[ObjType, PathType]) -> None:
        self.paths.add(path)
        adapter.write(obj, path)

    def load[ObjType, PathType : FSPathBase](self, path : PathType, adapter : FSAdapterBase[ObjType, PathType]) -> ObjType:
        return adapter.read(path)