from dataclasses import dataclass
from pathlib import Path

from Boulder_Statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from Boulder_Statistics.environment_tools.fs_environment import FSEnvironment
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from Boulder_Statistics.file_storage_adapters.copy_adapter import \
    FSShutilCopy2Adapter
from Boulder_Statistics.task_step_base import TaskStepBase


@dataclass(frozen=True)
class SimpleLocalFile(TaskStepBase):
    """Injects a file from a local path to the workflow

    Args:
        TaskStepBase (_type_): _description_

    Returns:
        _type_: _description_
    """
    local_file_path: Path
    dst_root_path: Path
    dst_sub_path: Path
    markers: frozenset[FSMarkerBase]
    skip_if_exists = True

    def run(self, env: FSEnvironment) -> FSEnvironment:

        pipeline_file = FSPathLocalDisk(
            path=self.dst_sub_path.parts,
            markers=self.markers,
            root_path=self.dst_root_path.as_posix()
        )

        if pipeline_file.exists and self.skip_if_exists:
            return FSEnvironment(paths=frozenset([pipeline_file]))

        FSEnvironment.save(
            None, pipeline_file, FSShutilCopy2Adapter(
                src=self.local_file_path), skip_if_exists=True)

        return FSEnvironment(paths=frozenset([pipeline_file]))
