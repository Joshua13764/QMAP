from typing import Any
from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

class EnvFilePDS4Fits(EnvFileBase):

    def read(self) -> None:
        raise NotImplementedError("Reading PDS4 fits files is not supported yet please us the XML to read insted.")
        
    def write(self, data: Any) -> None:
        raise NotImplementedError("Writing PDS4 fits files is not supported yet.")