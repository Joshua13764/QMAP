from pathlib import Path
from typing import List, Set

from prefect import get_run_logger

from Boulder_Statistics.environment_tools.fs_environment import FSEnvironment
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


class FSEnvironmentFactory():
    @staticmethod
    def from_folder(folder: Path,
                    extensions: Set[str] = set()) -> FSEnvironment:

        paths: List[Path] = [root / name
                             for root, dirs, files in folder.walk()
                             for name in files]

        get_run_logger().info(f"Found {len(paths)} files in {folder}")

        fs_paths = [
            FSPathLocalDisk(
                path=path.relative_to(folder).parts,
                root_path=folder.as_posix(),
                markers=frozenset()
            )
            for path in paths
            if path.suffix.lower() in extensions
        ]

        get_run_logger().info(
            f"Created {
                len(fs_paths)} FSPaths from found files")

        return FSEnvironment(paths=frozenset(fs_paths))
