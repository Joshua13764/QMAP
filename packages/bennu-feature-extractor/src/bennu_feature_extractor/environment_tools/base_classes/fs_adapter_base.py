from abc import ABC, abstractmethod
from bennu_feature_extractor.environment_tools.base_classes.fs_path_base import FSPathBase

class FSAdapterBase[ObjType, PathType : FSPathBase](ABC):

    @abstractmethod
    def write(self, obj: ObjType, path : PathType) -> None:
        ...

    @abstractmethod
    def read(self, path : PathType) -> ObjType:
        ...