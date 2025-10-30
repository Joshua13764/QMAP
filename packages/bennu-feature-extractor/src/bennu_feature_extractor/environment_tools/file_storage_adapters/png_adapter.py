from pathlib import Path
from typing import Any
import cv2
from numpy.typing import NDArray
from bennu_feature_extractor.environment_tools.base_classes.file_storage_adapter_base import FileStorageAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_medium_base import FileStorageMediumBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_persist_base import FileStoragePersistBase
from bennu_feature_extractor.environment_tools.file_storage_environment import FileStorageEnvironment
from bennu_feature_extractor.environment_tools.file_storage_mediums.local_disk import LocalDisk
from bennu_feature_extractor.environment_tools.file_storage_mediums.runtime_memory import RuntimeMemory

class PNGAdapter(FileStorageAdapterBase[NDArray[Any]]):

    def save(self, obj: NDArray[Any], virtual_path: Path, persist: FileStoragePersistBase, medium: FileStorageMediumBase) -> None:
        match medium:
            case RuntimeMemory() as rm:
                rm.save(obj, virtual_path, persist)
                
            case LocalDisk() as ld:
                write_state : bool = cv2.imwrite(ld.get_file_path_to_write(virtual_path).as_posix(), obj)
                if not write_state: raise IOError(f"Failed to write PNG: {ld.get_file_path_to_write(virtual_path)}")

            case _:
                raise NotImplementedError(f"The adapter {type(self).__name__} has no method to handel saving to the medium {type(medium).__name__}")
                
    
    def load(self, virtual_path: Path, env : FileStorageEnvironment) -> NDArray[Any]:
        match env.get_medium(virtual_path):
            case RuntimeMemory() as rm:
                return rm.load(virtual_path)
            
            case LocalDisk() as ld:
                img = cv2.imread(ld.get_file_path_to_write(virtual_path).as_posix(), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Could not read PNG: {ld.get_file_path_to_write(virtual_path).as_posix()}")
                return img

            case _ as medium:
                raise NotImplementedError(f"The adapter {type(self).__name__} has no method to handel loading to the medium {type(medium).__name__}")
        
    # @staticmethod
    # def _to_png_data(arr: NDArray[Any]) -> tuple[NDArray[np.uint8] | NDArray[np.uint16], list[int]]:
    #     x = np.asarray(arr)
    #     x = np.squeeze(x)
    #     if x.dtype in (np.uint8, np.uint16):
    #         img = x
    #     else:
    #         xf = x.astype(np.float32, copy=False)
    #         m, M = float(np.nanmin(xf)), float(np.nanmax(xf))
    #         xf = (xf - m) / (M - m) if M > m else np.zeros_like(xf, dtype=np.float32)
    #         img = (xf * 255.0).clip(0, 255).astype(np.uint8)
    #     if img.ndim == 3 and img.shape[-1] in (3, 4):
    #         img = img[..., ::-1]
    #     img = np.ascontiguousarray(img)
    #     params: list[int] = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    #     return img, params
