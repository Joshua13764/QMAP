from typing import Any
import cv2
from numpy.typing import NDArray
from bennu_feature_extractor.environment_tools.base_classes.file_storage_adapter_base import FSAdapterBase
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import FSPathLocalDisk

class FSPNGAdapter(FSAdapterBase[NDArray[Any], FSPathLocalDisk]):

    def read(self, path: FSPathLocalDisk) -> NDArray[Any]:
        img = cv2.imread(path.actual_path.as_posix(), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read PNG: {path.actual_path.as_posix()}")
        return img
    
    def write(self, obj: NDArray[Any], path: FSPathLocalDisk) -> None:
        write_state : bool = cv2.imwrite(path.actual_path.as_posix(), obj)
        if not write_state: raise IOError(f"Failed to write PNG: {path.actual_path}")
