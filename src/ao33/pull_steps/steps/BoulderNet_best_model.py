from ..pull_step_base import PullStepBase

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import zipfile
import os

@dataclass
class BoulderNetBestModelLoadStep(PullStepBase):
    Url: str
    DownloadPath: str

    @property
    def name(self) -> str:
        return __name__

    def run(self):
        self._logger.info(f"Starting BoulderNet model load from {self.Url}")

        download_dir = Path(self.DownloadPath)
        download_dir.mkdir(parents=True, exist_ok=True)

        # Skip entirely if already extracted
        if any(p.is_dir() or (p.is_file() and p.suffix not in {".zip", ".part"}) for p in download_dir.iterdir()):
            self._logger.info(f"Model already extracted in '{download_dir}', skipping.")
            return

        # Determine zip filename from URL or fallback name
        parsed = urlparse(self.Url)
        basename = os.path.basename(parsed.path) or "bouldernet_best_model.zip"
        out_zip = download_dir / basename
        tmp_download = download_dir / (basename + ".part")

        # Download archive (always fresh, since we assume no zip-only state)
        try:
            self._logger.info(f"Downloading model archive to temporary file '{tmp_download}'")
            urllib.request.urlretrieve(self.Url, tmp_download)
            tmp_download.replace(out_zip)
            self._logger.info("Download complete.")
        except Exception as exc:
            self._logger.error(f"Download failed: {exc}")
            if tmp_download.exists():
                tmp_download.unlink()
            raise

        # Extract archive
        try:
            self._logger.info(f"Extracting archive '{out_zip}'")
            with zipfile.ZipFile(out_zip, "r") as zip_ref:
                zip_ref.extractall(download_dir)
            self._logger.info(f"✅ BoulderNet model successfully downloaded and extracted to '{download_dir}'")
        except Exception as exc:
            self._logger.error(f"Extraction failed: {exc}")
            if out_zip.exists():
                out_zip.unlink()
            raise
