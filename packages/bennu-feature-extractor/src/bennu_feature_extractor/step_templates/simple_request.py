from pathlib import Path

import attr
import requests

from bennu_feature_extractor.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from bennu_feature_extractor.step_base import StepBase

attr.define()


class SimpleRequest(StepBase):
    url: str
    fs_path: Path
    sub_path: Path
    markers: frozenset[FSMarkerBase]
    skip_if_exists = True

    def run(self, env: FSEnvironment) -> FSEnvironment:

        file = FSPathLocalDisk(
            path=self.sub_path.parts,
            markers=self.markers,
            root_path=self.fs_path.as_posix()
        )

        if file.exists and self.skip_if_exists:
            return FSEnvironment(paths=frozenset([file]))

        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            with file.actual_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return FSEnvironment(paths=frozenset([file]))
