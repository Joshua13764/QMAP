from abc import ABC
from logging import Logger, getLogger
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import pickle

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase
from bennu_feature_extractor.environment_tools.env_file_factory import EnvFileFactory

@dataclass
class EnvClusterBase(ABC):
    files : List[EnvFileBase]

    # Logger is excluded from serialization
    logger: Logger = field(
        default_factory=lambda: getLogger(__name__),
        repr=False,
        compare=False,
    )

    def get_pickle_repr(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle_repr(cls, data: bytes) -> "EnvClusterBase":
        obj = pickle.loads(data)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj).__name__}, not {cls.__name__}")
        return obj
    
    @classmethod
    def from_folder(cls, folder_path: Path, virtual_path : Path, logger: Logger) -> "EnvClusterBase":
        actual_paths : List[Path] = [p.absolute() for p in folder_path.iterdir() if p.is_file()]
        logger.info(f"Found {len(actual_paths)} files in folder {folder_path} to create EnvClusterBase.")

        virtual_paths : List[Path] = [virtual_path / p.relative_to(folder_path) for p in actual_paths]
        logger.info(f"Generated {len(virtual_paths)} virtual paths for EnvClusterBase.")

        files : List[EnvFileBase] = [
            EnvFileFactory.create_env_file(file_path=actual_path, virtual_path=virtual_path, logger=logger)
            for actual_path, virtual_path in zip(actual_paths, virtual_paths)
        ]
        logger.info(f"Created {len(files)} EnvFileBase instances for EnvClusterBase using EnvFileFactory.")

        return cls(files=files, logger=logger)
    
    def check_metadata(self) -> bool:
        return all(file.check_metadata_valid() for file in self.files)
    
    def delete(self) -> None:
        for file in self.files:
            file.delete()
        self.logger.info(f"Deleted all files in cluster with {len(self.files)} files.")

    def get_total_size(self) -> int:
        return sum(file.get_size() for file in self.files)
    
    