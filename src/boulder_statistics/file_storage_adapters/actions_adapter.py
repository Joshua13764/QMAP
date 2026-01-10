from dataclasses import dataclass, field
from email.policy import default
from typing import Callable, List, Tuple

from joblib import delayed
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True, kw_only=True)
class FSActionsAdapter[T](
        FSAdapterBase[List[Callable[[], T]], FSPathLocalDisk]):
    obj_adapter: FSAdapterBase[T, FSPathLocalDisk]
    obj_export_path_name: Callable[[T, int], Tuple[str, ...]] = field(
        default=lambda obj, index: (f"export obj {index}",))
    standard_extension: str | None | bool = field(default=False)
    n_jobs: int = field(default=4)

    def read(self, path: FSPathLocalDisk) -> List[Callable[[], T]]:
        raise NotImplementedError

    def write(self, obj: List[Callable[[], T]], path: FSPathLocalDisk) -> None:

        ParallelPbar(f"Saving action objects with adapter {self.obj_adapter.__class__.__name__} with {self.n_jobs} jobs", unit="object")(n_jobs=self.n_jobs)(
            delayed(
                FSActionsAdapter.export_obj)(
                obj_action, path, self.obj_adapter)
            for obj_action in obj
        )

    @staticmethod
    def export_obj(
            obj_action: Callable[[], T], path: FSPathLocalDisk, adapter: FSAdapterBase[T, FSPathLocalDisk]) -> None:

        FSEnvironment.save(
            obj=obj_action(),
            path=path,
            adapter=adapter)
