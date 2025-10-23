from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

class EnvFileTxt(EnvFileBase):

    def read(self) -> object:
        with self.actual_path.open("rb") as f:
            return f.read().decode('utf-8')
        
    def write(self, data: object) -> None:
        with self.actual_path.open("wb") as f:
            f.write(data.__repr__().encode('utf-8'))