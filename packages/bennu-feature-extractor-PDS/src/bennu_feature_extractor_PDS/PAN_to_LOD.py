from typing import List

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from bennu_feature_extractor.environment_tools.utils.FS_environment_factory import \
    FSEnvironmentFactory
from bennu_feature_extractor.step_base import StepBase


class PANToLOD(StepBase):

    def run(self, env: FSEnvironment) -> FSEnvironment:

        pan_files: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: x.actual_path.suffix.lower() == ".tif")
