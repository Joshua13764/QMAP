import os
import zipfile
from typing import Any

from tqdm import tqdm


class ZipExtractor:
    @staticmethod
    def extract(zip_path: str, extract_dir: str, logger: Any) -> None:
        logger.info(f"Extracting {zip_path} → {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = archive.infolist()
            with tqdm(total=len(members), desc="Extracting") as bar:
                for member in members:
                    archive.extract(member, extract_dir)
                    bar.update(1)
        logger.info(f"Extraction complete: {extract_dir}")
