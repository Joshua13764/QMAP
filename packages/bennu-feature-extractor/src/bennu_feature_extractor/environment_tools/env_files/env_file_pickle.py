import pickle

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

class EnvFilePickle(EnvFileBase):

    def read(self) -> object:
        with self.actual_path.open("rb") as f:
            return pickle.load(f)
        
    def write(self, data: object) -> None:
        with self.actual_path.open("wb") as f:
            return pickle.dump(data, f)