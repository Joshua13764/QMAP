from pathlib import Path
from typing import List, Set

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


class FSEnvironmentFactory():
    @staticmethod
    def from_folder(folder: Path,
                    extensions: Set[str] = set()) -> FSEnvironment:

        paths: List[Path] = [root / name
                             for root, dirs, files in folder.walk()
                             for name in files]

        fs_paths: List[FSPathLocalDisk] = [
            FSPathLocalDisk(
                path=path.relative_to(folder).parts,
                root_path=folder.as_posix(),
                markers=frozenset()
            )
            for path in paths
            if path.suffix.lower() in extensions
        ]

        return FSEnvironment(paths=frozenset(fs_paths))
