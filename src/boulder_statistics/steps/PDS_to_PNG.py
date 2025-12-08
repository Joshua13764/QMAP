import logging
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from joblib import delayed
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.pds4_adapter import FSPDS4Adapter
from boulder_statistics.file_storage_adapters.png_adapter import FSPNGAdapter
from boulder_statistics.task_step_base import TaskStepBase


@dataclass(frozen=True)
class PDS_to_PNG(TaskStepBase):
    cluster_key: str
    run_path: str
    skip_converted: bool = True

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.cluster_key, self.run_path)

    def run(self, env: FSEnvironment) -> FSEnvironment:

        xml_files: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: x.actual_path.suffix.lower() == ".xml")

        pds_files: List[FSPathLocalDisk] = [
            f.copy_as_new(
                new_root_path=Path(self.run_path),
                new_extension=".png",
                markers=[FSMarkerString("InferableImage")]
            )
            for f in xml_files
        ]

        def _quiet_call(fn, *args, **kwargs):
            # Silence prints, stderr, warnings, and logging inside the worker
            with open(os.devnull, "w") as devnull, \
                    redirect_stdout(devnull), \
                    redirect_stderr(devnull), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.disable(logging.CRITICAL)
                try:
                    return fn(*args, **kwargs)
                finally:
                    logging.disable(logging.NOTSET)

        def convert_png(xml_path: FSPathLocalDisk,
                        pds_path: FSPathLocalDisk) -> None:

            _, img = FSEnvironment.load(xml_path, FSPDS4Adapter())
            FSEnvironment.save(img, pds_path, FSPNGAdapter())

        exits = FSEnvironment.quick_exists(pds_files)

        pairs = [
            (xml, pds)
            for xml, pds in zip(xml_files, pds_files)
            if not (exits[pds] and self.skip_converted)
        ]

        self.logger.info(
            f"Converting {
                len(pairs)} PDS4 XML files in cluster '{
                self.cluster_key}' to PNG format..."
        )

        # For hard disk n_jobs = 1 is better for read writing
        ParallelPbar(desc="Converting PDS4 to PNG", unit="img")(n_jobs=1)(
            delayed(_quiet_call)(convert_png, xml, pds) for xml, pds in pairs
        )

        return FSEnvironment(paths=tuple(pds_files))
