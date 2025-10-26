import pds4_tools
from typing import Any, Mapping, Sequence, Protocol
from numpy.typing import NDArray

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

class ArrayStructure(Protocol):
    data: NDArray[Any]
    meta_data: Mapping[str, Any]

StructureList = Sequence[ArrayStructure]

class EnvFilePDS4XML(EnvFileBase):

    def read(self) -> StructureList:
        return pds4_tools.read(self.actual_path)
        
    def write(self, data: Any) -> None:
        raise NotImplementedError("Writing PDS4 files is not supported yet.")