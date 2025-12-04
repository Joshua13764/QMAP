from dataclasses import dataclass

from PIL import Image

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSPillowImageAdapter(FSAdapterBase[Image.Image, FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> Image.Image:
        return Image.open(path.actual_path.as_posix())

    def write(self, obj: Image.Image, path: FSPathLocalDisk) -> None:
        return obj.save(path.actual_path.as_posix())
