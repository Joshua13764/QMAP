from functools import reduce
from os import listdir, path, scandir
from typing import Callable, Counter, Dict, List, Set

import attr
from jinja2 import Environment
from joblib import delayed
from tqdm_joblib import ParallelPbar

from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from bennu_feature_extractor.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@attr.define(frozen=True, slots=True)
class FSEnvironment():
    paths: frozenset[FSPathBase]

    def get_paths[T: FSPathBase](self, cls: type[T], condition: Callable[[
                                 T], bool] = lambda x: True) -> List[T]:
        return [f for f in self.paths if isinstance(f, cls) and condition(f)]

    def get_paths_from_markers[T: FSPathBase](
            self, cls: type[T], markers: frozenset[FSMarkerBase]) -> List[T]:
        return [f
                for f in self.paths
                if isinstance(f, cls) and markers.isdisjoint(f.markers) == False
                ]

    @staticmethod
    def quick_exists(paths: List[FSPathLocalDisk],
                     threshold=1000) -> Dict[FSPathLocalDisk, bool]:

        item_counts: Counter = Counter(
            p.actual_path.parent.as_posix() for p in paths)

        filtered_folders: Dict[str, List[FSPathLocalDisk]] = {
            parent: [] for parent,
            n in item_counts.items() if n >= threshold}

        def scan_folder(folder_path: str,
                        _paths: List[FSPathLocalDisk]) -> Dict[FSPathLocalDisk, bool]:
            folder_files: Set[str] = {
                e.name for e in scandir(folder_path) if e.is_file()}

            return {p: p.actual_path.name in folder_files for p in _paths}

        other_folders: List[FSPathLocalDisk] = []

        for p in paths:
            if p.actual_path.parent.as_posix() in filtered_folders:
                filtered_folders[p.actual_path.parent.as_posix()].append(p)
            else:
                other_folders.append(p)

        results = ParallelPbar(desc=f"Scanning paths", unit="folders")(n_jobs=-1, verbose=0, prefer="threads")(
            delayed(scan_folder)(folder, _paths) for folder, _paths in filtered_folders.items()
        ) + [{p: p.exists for p in other_folders}]

        return reduce(lambda a, b: a | b, results, {})

    @classmethod
    def empty(cls) -> 'FSEnvironment':
        return cls(paths=frozenset())

    @staticmethod
    def save[ObjType, PathType: FSPathBase](
            obj: ObjType, path: PathType, adapter: FSAdapterBase[ObjType, PathType], skip_if_exists=False) -> FSPathBase:

        if path.exists and skip_if_exists:
            return path

        path.make_directory()
        adapter.write(obj, path)

        return path

    @staticmethod
    def load[ObjType, PathType: FSPathBase](
            path: PathType, adapter: FSAdapterBase[ObjType, PathType]) -> ObjType:
        return adapter.read(path)

    @staticmethod
    def merge(envs: List['FSEnvironment']) -> 'FSEnvironment':
        merged_paths: frozenset[FSPathBase] = frozenset({
            p for e in envs for p in e.paths})
        return FSEnvironment(paths=merged_paths)

    def __iadd__(self, other_env: 'FSEnvironment') -> 'FSEnvironment':
        self = Environment.merge(self, other_env)
        return self
