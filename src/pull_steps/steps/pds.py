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
    def _download_chunk(url, byte_range, dest_path, logger):
        headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest_path, "r+b") as f:
                f.seek(byte_range[0])
                for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)

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

        if os.path.exists(extract_dir):
            logger.info(f"Already extracted: {extract_dir}, skipping.")
            return

        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        try:
            # ---- HEAD request to get file size ----
            head = requests.head(url, timeout=60)
            total_size = int(head.headers.get("Content-Length", 0))
            supports_range = "bytes" in head.headers.get("Accept-Ranges", "").lower()

            if not supports_range or total_size == 0:
                logger.warning("Server does not support Range requests or unknown file size — falling back to single-threaded download.")
                PDSPullStep._single_stream_download(url, temp_zip_path, logger)
            else:
                # ---- Parallel range download ----
                n_threads = min(8, os.cpu_count() or 4)
                chunk_size = total_size // n_threads
                ranges = [
                    (i * chunk_size, (i + 1) * chunk_size - 1 if i < n_threads - 1 else total_size - 1)
                    for i in range(n_threads)
                ]
                logger.info(f"Downloading {file_name} in {n_threads} parallel chunks (~{chunk_size // (1024**2)} MB each)")

                with open(temp_zip_path, "wb") as f:
                    f.truncate(total_size)

                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    futures = []
                    for r in ranges:
                        futures.append(ex.submit(PDSPullStep._download_chunk, url, r, temp_zip_path, logger))
                    for i, fut in enumerate(futures, 1):
                        fut.result()
                        logger.info(f"Chunk {i}/{len(futures)} complete.")
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
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            if os.path.exists(zip_path):
                os.remove(zip_path)
                logger.debug(f"Removed temporary file: {zip_path}")

    @staticmethod
    def _single_stream_download(url, dest_path, logger):
        """Fallback single-thread download."""
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
