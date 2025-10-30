from typing import Any, Mapping, Sequence, Protocol, Tuple
from pathlib import Path
import pds4_tools
from bennu_feature_extractor.environment_tools.base_classes.file_storage_adapter_base import FileStorageAdapterBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_medium_base import FileStorageMediumBase
from bennu_feature_extractor.environment_tools.base_classes.file_storage_persist_base import FileStoragePersistBase
from bennu_feature_extractor.environment_tools.file_storage_environment import FileStorageEnvironment
from bennu_feature_extractor.environment_tools.file_storage_mediums.local_disk import LocalDisk
from bennu_feature_extractor.environment_tools.file_storage_mediums.runtime_memory import RuntimeMemory
import numpy as np
from numpy.typing import NDArray

class ArrayStructure(Protocol):
    data: NDArray[Any]
    meta_data: Mapping[str, Any]

    def as_masked(self) -> "ArrayStructure": ...

StructureList = Sequence[ArrayStructure]

class PDS4Adapter(FileStorageAdapterBase[Tuple[StructureList, NDArray[Any]]]):

    def save(self, obj: Tuple[StructureList, NDArray[Any]], virtual_path: Path, persist: FileStoragePersistBase, medium: FileStorageMediumBase) -> None:
        match medium:
            case _:
                raise NotImplementedError(f"The adapter {type(self).__name__} has no method to handel saving to the medium {type(medium).__name__}")
                
    
    def load(self, virtual_path: Path, env : FileStorageEnvironment) -> Tuple[StructureList, NDArray[Any]]:
        match env.get_medium(virtual_path):
            case RuntimeMemory() as rm:
                return rm.load(virtual_path)
            
            case LocalDisk() as ld:
                sl : StructureList = pds4_tools.read(ld.get_file_path_to_write(virtual_path).as_posix(), lazy_load=True)
                img : NDArray[Any] = PDS4Adapter._to_viewable(sl)
                return sl, img

            case _ as medium:
                raise NotImplementedError(f"The adapter {type(self).__name__} has no method to handel loading to the medium {type(medium).__name__}")

    @staticmethod
    def _to_viewable(sl: StructureList) -> NDArray[Any]:
        # 1) pick first Array_* with NumPy data
        s = next((x for x in sl if hasattr(x, "data") and isinstance(x.data, np.ndarray)), None)
        if s is None:
            raise ValueError("No Array_* structure with NumPy data found.")

        # 2) use masked data if available (fill masked with 0), keep original dtype/range
        data = getattr(s.as_masked(), "data", s.data)
        if np.ma.isMaskedArray(data):
            data = data.filled(0)
        img = np.squeeze(np.asarray(data))

        # 3) reorder axes to (Line, Sample, *rest) if names exist
        aa = getattr(s, "meta_data", {}).get("Axis_Array")
        if aa:
            names = [aa["axis_name"]] if isinstance(aa, dict) else [d["axis_name"] for d in aa]
            if "Line" in names and "Sample" in names:
                vi, hi = names.index("Line"), names.index("Sample")
                if img.ndim == 2:
                    img = np.transpose(img, (vi, hi))
                elif img.ndim >= 3:
                    rest = [i for i in range(img.ndim) if i not in (vi, hi)]
                    img = np.transpose(img, (vi, hi, *rest))

        return img
    
    # def show(self) -> None:
    #     _, img = self.read()  # already squeezed + (Line, Sample, ...)
    #     disp = img

    #     # If float, cv2.imshow expects values in [0, 1]
    #     if np.issubdtype(disp.dtype, np.floating):
    #         m, M = float(np.nanmin(disp)), float(np.nanmax(disp))
    #         disp = ((disp - m) / (M - m) if M > m else np.zeros_like(disp)).astype(np.float32)

    #     # If clearly 3-channel last, assume RGB -> BGR for OpenCV
    #     if disp.ndim == 3 and disp.shape[-1] == 3:
    #         disp = disp[..., ::-1]

    #     cv2.imshow("PDS4 Image", np.ascontiguousarray(disp))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

