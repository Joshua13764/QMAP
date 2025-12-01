import shutil
from dataclasses import dataclass
from pathlib import Path

from bennu_feature_extractor.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSShutilCopy2Adapter(FSAdapterBase[None, FSPathLocalDisk]):
    src: Path

    def read(self, path: FSPathLocalDisk) -> None:
        raise NotImplementedError()

    def write(self, obj: None, path: FSPathLocalDisk) -> None:
        path.make_directory()

        shutil.copy2(
            src=self.src,
            dst=path.actual_path.as_posix()
        )
