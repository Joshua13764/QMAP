import re
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
    skip_converted: bool = True
    run_path: Path | None = None

    def under_run(self, base: Path, child: Union[Path, str]) -> Path:
        s = str(child)

        if s.startswith("\\\\?\\"):
            s = s[4:]

        s = re.sub(r"^[A-Za-z]:", "", s)

        s = s.lstrip("\\/")

        return base / Path(s)

    def run(self, env: Environment) -> Environment:

        cluster: EnvCluster | None = env.clusters.get(self.cluster_key)
        if cluster is None:
            raise ValueError(f"Cluster for key '{self.cluster_key}' is None.")

        pds_files = [f for f in cluster.files if isinstance(f, EnvFilePDS4XML)]

        self.logger.info(
            f"Converting {
                len(pds_files)} PDS4 XML files in cluster '{
                self.cluster_key}' to PNG format..."
        )

        def convert_png(file: EnvFilePDS4XML) -> None:
            png_actual_path: Path = (
                self.under_run(self.run_path,
                               file.virtual_path).with_suffix(".png")
                if self.run_path else file.actual_path.with_suffix(".png")
            )

            if png_actual_path.exists() and self.skip_converted:
                return

            _, img = file.read()

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
