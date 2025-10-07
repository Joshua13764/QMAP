from ..pull_step_base import PullStepBase

import os
import urllib.parse
import requests
import zipfile
import shutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PDSPullStep(PullStepBase):
    DownloadPath: str
    Url: str
    BASE_URL = "https://sbnarchive.psi.edu"

    @property
    def name(self) -> str:
        return __name__

    def run(self):
        self._logger.info("Running PDS Pull Step")
        self.download_and_extract(
            url=self.Url,
            root_dir=self.DownloadPath,
            logger=self._logger
        )

    @staticmethod
    def download_and_extract(url: str, root_dir: str, logger):
        if not url.startswith(PDSPullStep.BASE_URL):
            logger.error(f"URL '{url}' is not under {PDSPullStep.BASE_URL}")
            return

        parsed_url = urllib.parse.urlparse(url)
        rel_path = parsed_url.path.lstrip("/")
        file_name = os.path.basename(rel_path)
        extract_dir_name = os.path.splitext(file_name)[0]
        extract_dir = os.path.join(root_dir, os.path.dirname(rel_path), extract_dir_name)
        zip_path = os.path.join(root_dir, rel_path)
        temp_zip_path = zip_path + ".part"

        # Skip if already extracted
        if os.path.exists(extract_dir):
            logger.info(f"Already extracted: {extract_dir}, skipping.")
            return

        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        try:
            # ---- Download ----
            logger.info(f"Starting download: {url}")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()

                total_size = int(r.headers.get("Content-Length", 0))
                if total_size == 0:
                    logger.warning("No Content-Length header found; progress reporting limited.")

                downloaded = 0
                chunk_size = 16 * 1024 * 1024  # 16 MB chunks
                next_log_point = 0.05  # log every 5%

                with open(temp_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        # ---- Periodic progress logging ----
                        if total_size > 0:
                            progress = downloaded / total_size
                            if progress >= next_log_point:
                                mb_done = downloaded / (1024 ** 2)
                                mb_total = total_size / (1024 ** 2)
                                mb_left = mb_total - mb_done
                                logger.info(
                                    f"Download progress: {progress * 100:.1f}% "
                                    f"({mb_done:.0f} MB of {mb_total:.0f} MB, ~{mb_left:.0f} MB left)"
                                )
                                # Force flush so logs appear live
                                for h in logger.handlers:
                                    try:
                                        h.flush()
                                    except Exception:
                                        pass
                                next_log_point += 0.05
                        else:
                            if downloaded % (1024 ** 3) < chunk_size:  # every ~1 GB
                                mb_done = downloaded / (1024 ** 2)
                                logger.info(f"Downloaded {mb_done:.0f} MB so far...")
                                for h in logger.handlers:
                                    try:
                                        h.flush()
                                    except Exception:
                                        pass

            os.rename(temp_zip_path, zip_path)
            logger.info(f"Download complete → {zip_path}")

            # ---- Extract ----
            logger.info(f"Extracting {zip_path} to {extract_dir} ...")
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.infolist()

                def extract_one(member):
                    zf.extract(member, extract_dir)

                with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
                    list(ex.map(extract_one, members))

            logger.info(f"Extraction complete: {extract_dir}")

        except Exception as e:
            logger.error(f"Failed processing {url}: {e}", exc_info=True)
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)

        finally:
            # Cleanup
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            if os.path.exists(zip_path):
                os.remove(zip_path)
                logger.debug(f"Removed temporary file: {zip_path}")
