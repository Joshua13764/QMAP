from dataclasses import dataclass, field
from typing import Any, List, Tuple

from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase
from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class StepDefaultMarkers():
    input_markers: Tuple[FSMarkerBase, ...]
    output_markers: Tuple[FSMarkerBase, ...]

    def include_markers_in_hashable(
            self, *hashable_items: Any) -> Tuple[Any, ...]:
        return (*hashable_items, self.input_markers, self.output_markers)

    def get_files_with_markers(
            self, env: FSEnvironment) -> List[FSPathLocalDisk]:
        return env.get_paths_from_markers(FSPathLocalDisk, self.input_markers)
