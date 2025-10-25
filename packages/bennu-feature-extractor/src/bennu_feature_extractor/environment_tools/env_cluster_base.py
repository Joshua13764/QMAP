import pickle
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import List

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase
from bennu_feature_extractor.environment_tools.env_file_factory import EnvFileFactory

@dataclass
class EnvCluster():
    files: List[EnvFileBase]
    logger: Logger = field(repr=False, compare=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("logger", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_pickle_repr(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle_repr(cls, data: bytes, logger: Logger) -> "EnvCluster":
        obj = pickle.loads(data)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj).__name__}, not {cls.__name__}")

        obj.logger = logger
        for f in getattr(obj, "files", []) or []:
            f.logger = logger
        return obj

    @classmethod
    def from_folder(cls, folder_path: Path, virtual_path : Path, logger: Logger) -> "EnvCluster":
        actual_paths: List[Path] = [p.absolute() for p in folder_path.iterdir() if p.is_file()]
        if logger:
            logger.info(f"Found {len(actual_paths)} files in folder {folder_path} to create EnvClusterBase.")

        virtual_paths: List[Path] = [virtual_path / p.relative_to(folder_path) for p in actual_paths]
        if logger:
            logger.info(f"Generated {len(virtual_paths)} virtual paths for EnvClusterBase.")

        files: List[EnvFileBase] = [
            EnvFileFactory.create_env_file(file_path=actual_path, virtual_path=vpath, logger=logger)
            for actual_path, vpath in zip(actual_paths, virtual_paths)
        ]
        if logger:
            logger.info(f"Created {len(files)} EnvFileBase instances for EnvClusterBase using EnvFileFactory.")

        return cls(files=files, logger=logger)

    def check_metadata(self) -> bool:
        return all(file.check_metadata_valid() for file in self.files)

    def delete(self) -> None:
        for file in self.files:
            file.delete()
        if self.logger:
            self.logger.info(f"Deleted all files in cluster with {len(self.files)} files.")

    def get_total_size(self) -> int:
        return sum(file.get_size() for file in self.files)
