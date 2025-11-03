from pathlib import Path
from typing import List

import attr

from bennu_feature_extractor.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from bennu_feature_extractor.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@attr.define(frozen=True, slots=True)
class FSPathLocalDisk(FSPathBase):
    root_path: str

    @property
    def actual_path(self) -> Path:
        return Path(self.root_path) / Path(*self.path)

    @property
    def exists(self) -> bool:
        return self.actual_path.exists()

    def make_directory(self) -> None:
        self.actual_path.parent.mkdir(parents=True, exist_ok=True)

    def copy_as_new(self, new_root_path: Path,
                    new_extension: str, markers: List[FSMarkerBase]) -> 'FSPathLocalDisk':
        return FSPathLocalDisk(
            path=Path(*self.path).with_suffix(new_extension).parts,
            markers=frozenset(markers),
            root_path=new_root_path.as_posix(),
        )
