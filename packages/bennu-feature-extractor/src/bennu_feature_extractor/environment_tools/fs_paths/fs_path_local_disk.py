from pathlib import Path
import attr

from bennu_feature_extractor.environment_tools.base_classes.fs_path_base import FSPathBase

@attr.define(frozen=True, slots=True)
class FSPathLocalDisk(FSPathBase):
    root_path : str

    @property
    def actual_path(self) -> Path:
        return Path(self.root_path) / Path(*self.path)