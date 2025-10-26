import pickle
from typing import Any

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

class EnvFilePickle(EnvFileBase):

    def read(self) -> Any:
        with self.actual_path.open("rb") as f:
            return pickle.load(f)
        
    def write(self, data: Any) -> None:
        with self.actual_path.open("wb") as f:
            return pickle.dump(data, f)