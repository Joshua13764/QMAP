from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from bennu_feature_extractor.environment import *
from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.step_base import StepBase
from bennu_feature_extractor_PDS.file_storage_adapters.pds4_adapter import (
    ArrayStructure, FSPDS4Adapter)
from joblib import delayed
from numpy import dtype, ndarray
from numpy.typing import NDArray
from tqdm_joblib import ParallelPbar


@dataclass()
class PDS4BoulderNetInference(StepBase):
    run_path: Path

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_infer: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: FSMarkerString("PDS4BoulderImage") in x.markers)

        inference_output_files: List[FSPathLocalDisk] = [
            f.copy_as_new(
                new_root_path=self.run_path,
                new_extension=".bni"  # BoulderNetInference (bni)
            )
            for f in files_to_infer
        ]

        # ParallelPbar(f"rendering lod_depth {lod_depth} for model {file.actual_path.name}")(n_jobs=1)(
        #         delayed(
        #             LodNode.render_on_all_faces)(
        #             LodNode(
        #                 shape,
        #                 fileData,
        #                 file,
        #                 self.skip_if_exists,
        #                 self.debug_mode),
        #             target_width=self.lod_res)
        #         for shape in PANToLOD.all_binaries(bits=2 * lod_depth)

    @staticmethod
    def analyse_image(image_path: FSPathLocalDisk,
                      inference_output_path: FSPathLocalDisk) -> None:

        metadata, img = FSEnvironment.load(image_path, FSPDS4Adapter())
