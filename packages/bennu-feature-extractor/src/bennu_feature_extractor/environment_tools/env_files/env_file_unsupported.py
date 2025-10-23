from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

class EnvFileUnsupported(EnvFileBase):

    def read(self) -> object:
        raise NotImplementedError(f"Read method doesn't exist for unknown file type of {self.actual_path.suffix.lower}.")
        
    def write(self, data: object) -> None:
        raise NotImplementedError(f"Write method doesn't exist for unknown file type of {self.actual_path.suffix.lower}.")