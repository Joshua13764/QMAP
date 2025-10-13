from ..pull_step_base import PullStepBase

import os
import urllib.request
import zipfile
from dataclasses import dataclass

@dataclass
class BoulderNetBestModelLoadStep(PullStepBase):
    Url: str
    DownloadPath: str

    @property
    def name(self) -> str:
        return __name__

    def run(self):
        self._logger.info(f"Starting BoulderNet model load from {self.Url}")

        os.makedirs(self.DownloadPath, exist_ok=True)
        out_zip = os.path.join(self.DownloadPath, "best_model.zip")

        if any(fname.endswith(".pt") for fname in os.listdir(self.DownloadPath)):
            self._logger.info(f"Model already exists in '{self.DownloadPath}', skipping download.")
            return

        self._logger.info(f"Downloading model archive to '{out_zip}'")
        urllib.request.urlretrieve(self.Url, out_zip)
        self._logger.info("Download complete.")

        self._logger.info(f"Extracting archive '{out_zip}'")
        with zipfile.ZipFile(out_zip, "r") as zip_ref:
            zip_ref.extractall(self.DownloadPath)
        self._logger.info(f"Extraction complete. Files available in '{self.DownloadPath}'")

        self._logger.info("✅ BoulderNet best_model.zip successfully downloaded and extracted.")