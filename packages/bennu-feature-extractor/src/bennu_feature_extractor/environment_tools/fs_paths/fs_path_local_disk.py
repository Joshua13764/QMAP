from pathlib import Path

import attr

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

    def copy_as_new(self, new_root_path: Path,
                    new_extension: str) -> 'FSPathLocalDisk':
        return FSPathLocalDisk(
            path=Path(*self.path).with_suffix(new_extension).parts,
            root_path=new_root_path
        )
