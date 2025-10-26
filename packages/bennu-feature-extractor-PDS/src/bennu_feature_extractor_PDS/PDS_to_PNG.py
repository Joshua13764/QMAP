import io
import re
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from bennu_feature_extractor.environment import Environment
from bennu_feature_extractor.environment_tools.env_cluster_base import \
    EnvCluster
from bennu_feature_extractor.environment_tools.env_files.env_file_pds4_xml import \
    EnvFilePDS4XML
from bennu_feature_extractor.environment_tools.env_files.env_file_PNG import \
    EnvFilePNG
from bennu_feature_extractor.step_base import StepBase
from tqdm import tqdm


@dataclass
class PDS_to_PNG(StepBase):
    cluster_key: str
    run_path: Path | None = None

    def run(self, env: Environment) -> Environment:

        cluster: EnvCluster | None = env.clusters.get(self.cluster_key)
        if cluster is None:
            raise ValueError(f"Cluster for key '{self.cluster_key}' is None.")

        pds_files = [f for f in cluster.files if isinstance(f, EnvFilePDS4XML)]
        self._logger.info(
            f"Converting {
                len(pds_files)} PDS4 XML files in cluster '{
                self.cluster_key}' to PNG format..."
        )

        def under_run(base: Path, child: Union[Path, str]) -> Path:
            s = str(child)

            # Strip special UNC prefix (\\?\) if present
            if s.startswith("\\\\?\\"):
                s = s[4:]

            # Drop drive prefix like 'C:' or 'F:'
            s = re.sub(r"^[A-Za-z]:", "", s)

            # Drop any leading slashes/backslashes so it's relative
            s = s.lstrip("\\/")

            return base / Path(s)

        def convert_png(file: EnvFilePDS4XML) -> None:
            _, img = file.read()

            png_actual_path: Path = (
                under_run(self.run_path,
                          file.virtual_path).with_suffix(".png")
                if self.run_path else file.actual_path.with_suffix(".png")
            )
            png_file = EnvFilePNG(
                last_modified=None,
                virtual_path=file.virtual_path.with_suffix(".png"),
                actual_path=png_actual_path,
                logger=file.logger,
                overwrite_allowed=False
            )
            png_file.write(img)

        # Progress bar over sequential conversion
        for f in tqdm(pds_files, desc="Converting PDS4 XML → PNG",
                      unit="file"):
            convert_png(f)

        return env
