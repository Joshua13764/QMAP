import os
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from Boulder_Statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from Boulder_Statistics.environment_tools.fs_environment import FSEnvironment
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from Boulder_Statistics.task_step_base import TaskStepBase


@dataclass(frozen=True)
class SimpleRequest(TaskStepBase):
    url: str
    fs_path: str
    sub_path: str
    markers: frozenset[FSMarkerBase]
    skip_if_exists = True

    def run(self, env: FSEnvironment) -> FSEnvironment:

        file = FSPathLocalDisk(
            path=Path(self.sub_path).parts,
            markers=self.markers,
            root_path=Path(self.fs_path).as_posix()
        )

        if file.exists and self.skip_if_exists:
            return FSEnvironment(paths=frozenset([file]))

        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            file.make_directory()
            tmp_dir: Path = file.actual_path.with_name(
                file.actual_path.name + ".part")

            total = int(r.headers.get("content-length", 0))

            with tmp_dir.open("wb") as f, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {file.actual_path.name}",
                dynamic_ncols=True,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            os.replace(tmp_dir, file.actual_path)

        return FSEnvironment(paths=frozenset([file]))
