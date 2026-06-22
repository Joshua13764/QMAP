from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

from more_itertools import chunked

from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass()
class FSPathLocalDiskChunk():
    files_to_infer: List[FSPathLocalDisk] = field(default_factory=list)
    inference_output_files: List[FSPathLocalDisk] = field(default_factory=list)

    def get_sub_chunks(
            self, batch_size: int = 64) -> Iterator[Tuple[list[FSPathLocalDisk], list[FSPathLocalDisk]]]:

        return zip(chunked(self.files_to_infer, batch_size),
                   chunked(self.inference_output_files, batch_size))
