from ..pull_step_base import PullStepBase

import os
import urllib.parse
import requests
import zipfile
import shutil
from dataclasses import dataclass

@dataclass
class PDSPullStep(PullStepBase):
    DownloadPath : str
    Url : str
    BASE_URL = "https://sbnarchive.psi.edu"

    @property
    def name(self) -> str:
        return __name__
    
    def run(self):
        
        self._logger.info("Running PDS Pull Step")
        self.download_and_extract(
            url = self.Url,
            root_dir = self.DownloadPath,
            logger = self._logger
        )

    @staticmethod
    def download_and_extract(url: str, root_dir: str, logger):
        """
        Downloads and extracts a .zip file from the PDS4 archive into a structured directory tree.

        Args:
            url (str): The download URL (must be under https://sbnarchive.psi.edu/).
            root_dir (str): The root directory where files should be stored.
            logger (logging.Logger): Logger for progress and error messages.
        """
        # Validate URL
        if not url.startswith(PDSPullStep.BASE_URL):
            logger.error(f"URL '{url}' is not under {PDSPullStep.BASE_URL}")
            return

        # Parse and determine local paths
        parsed_url = urllib.parse.urlparse(url)
        rel_path = parsed_url.path.lstrip("/")  # e.g. "pds4/orex/downloads_ocams/ocams_data_reduced_orbit_c.zip"
        file_name = os.path.basename(rel_path)
        extract_dir_name = os.path.splitext(file_name)[0]  # remove .zip
        extract_dir = os.path.join(root_dir, os.path.dirname(rel_path), extract_dir_name)
        zip_path = os.path.join(root_dir, rel_path)

        # Check if already extracted
        if os.path.exists(extract_dir):
            logger.info(f"Already extracted: {extract_dir}, skipping.")
            return

        # Ensure directory structure
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        try:
            # Download the zip file
            logger.info(f"Downloading {url} ...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded to {zip_path}")

            # Extract the zip file
            logger.info(f"Extracting {zip_path} to {extract_dir} ...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Extracted to {extract_dir}")

        except Exception as e:
            logger.error(f"Failed processing {url}: {e}")
            # Cleanup partially downloaded or extracted content
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
        finally:
            # Clean up the zip file
            if os.path.exists(zip_path):
                os.remove(zip_path)
                logger.debug(f"Removed temporary file: {zip_path}")



