import attrs
from logging import Logger
from pathlib import Path
from typing import List, Set, Tuple
from prefect import get_run_logger

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase
from bennu_feature_extractor.environment_tools.env_file_factory import EnvFileFactory

@attrs.define(frozen=True, slots=True, cache_hash=True)
class EnvCluster():
    name : str
    files: Tuple[EnvFileBase]

    @property
    def logger(self) -> Logger:
        return get_run_logger()

    @classmethod
    def from_folder(cls, folder_path: Path, virtual_path : Path) -> "EnvCluster":
        actual_paths: List[Path] = [p.resolve() for p in Path(folder_path).rglob('*') if p.is_file()]
        
        get_run_logger().info(f"Found {len(actual_paths)} files in folder {folder_path} to create EnvClusterBase.")

        virtual_paths: List[Path] = [virtual_path / p.relative_to(folder_path) for p in actual_paths]

        get_run_logger().info(f"Generated {len(virtual_paths)} virtual paths for EnvClusterBase.")

        files: List[EnvFileBase] = [
            EnvFileFactory.create_env_file(file_path=actual_path, virtual_path=vpath)
            for actual_path, vpath in zip(actual_paths, virtual_paths)
        ]

        get_run_logger().info(f"Created {len(files)} EnvFileBase instances for EnvClusterBase using EnvFileFactory.")

        return cls(files=tuple(files), name = folder_path.name)

    def check_metadata(self) -> bool:
        return all(file.check_metadata_valid() for file in self.files)

    def get_total_size(self) -> int:
        return sum(file.get_size() for file in self.files)
