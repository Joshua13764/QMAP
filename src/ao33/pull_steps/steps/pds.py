from ..pull_step_base import PullStepBase

import os
import json
import math
import shutil
import zipfile
import multiprocessing
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Any, Set, Dict, Tuple, List
from joblib import Parallel, delayed
from threading import Lock
from tqdm import tqdm

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class PDSPullStep(PullStepBase):
    DownloadPath: str
    Url: str
    _logger: Any
    BASE_URL: str = "https://sbnarchive.psi.edu"
    Workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 2))
    KeepArchive: bool = False
    ChunkSizeLimitMB: int = 50  # configurable, default 50 MB

    @property
    def name(self) -> str:
        return __name__

    def __post_init__(self):
        os.makedirs(self.DownloadPath, exist_ok=True)
        self._session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    # ------------------- Public API -------------------

    def run(self):
        if not self.Url.startswith(self.BASE_URL):
            raise ValueError(f"URL must start with {self.BASE_URL}")

        zip_path = self._download_file()
        extract_dir = self._get_extract_dir(zip_path)

        # Skip extraction if folder already exists
        if os.path.exists(extract_dir):
            self._logger.info(f"Already extracted: {extract_dir}, skipping extraction.")
            return

        self._extract_zip(zip_path, extract_dir)
        if not self.KeepArchive:
            os.remove(zip_path)

    # ------------------- Internal -------------------

    def _download_file(self) -> str:
        parsed = urlparse(self.Url)
        file_name = os.path.basename(parsed.path)
        file_path = os.path.join(self.DownloadPath, file_name)
        extract_dir = self._get_extract_dir(file_path)
        resume_file = os.path.join(self.DownloadPath, f"{file_name}.resume.json")

        # Skip download entirely if extraction exists
        if os.path.exists(extract_dir):
            self._logger.info(f"Extraction folder exists ({extract_dir}) — skipping download.")
            return file_path

        self._logger.info(f"Fetching headers for: {self.Url}")
        head = self._session.head(self.Url, allow_redirects=True)
        total_size = int(head.headers.get("Content-Length", 0))
        supports_range = "bytes" in head.headers.get("Accept-Ranges", "").lower()
        if total_size == 0:
            raise ValueError("Unknown file size — cannot proceed.")

        # Determine chunk size
        n = self.Workers
        max_chunk = self.ChunkSizeLimitMB * 1024 * 1024
        chunk_size = min(math.ceil(total_size / n), max_chunk)
        ranges = [
            (i * chunk_size, min((i + 1) * chunk_size - 1, total_size - 1))
            for i in range(math.ceil(total_size / chunk_size))
        ]

        completed: Set[int] = set()
        resume_meta: Dict[str, Any] = {}

        # Load resume info if exists
        if os.path.exists(resume_file):
            try:
                with open(resume_file) as f:
                    resume_meta = json.load(f)
                    completed = set(resume_meta.get("completed", []))
            except Exception:
                self._logger.warning("Corrupt resume file — resetting cache.")
                self._invalidate_cache(file_path, resume_file, extract_dir)
                completed.clear()

        # Validate cache metadata
        valid_cache = (
            resume_meta.get("url") == self.Url
            and resume_meta.get("total_size") == total_size
            and resume_meta.get("chunk_size") == chunk_size
            and resume_meta.get("workers") == n
        )

        if not valid_cache and os.path.exists(resume_file):
            self._logger.warning("Cache invalid — settings changed. Clearing old partial files.")
            self._invalidate_cache(file_path, resume_file, extract_dir)
            completed.clear()

        self._logger.info(
            f"Downloading with {n} workers, chunk size = {chunk_size / (1024**2):.2f} MB, total chunks = {len(ranges)}"
        )

        tqdm_bar = tqdm(total=len(ranges), desc="Downloading", initial=len(completed))
        lock = Lock()

        def _download_chunk(idx: int, start: int, end: int, max_retries: int = 5):
            if idx in completed:
                with lock:
                    tqdm_bar.update(1)
                return idx

            headers = {"Range": f"bytes={start}-{end}"} if supports_range else {}
            part_path = f"{file_path}.part{idx}"
            attempt = 0

            while attempt < max_retries:
                try:
                    with self._session.get(self.Url, headers=headers, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(part_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                                if chunk:
                                    f.write(chunk)

                    # If download completes without exception
                    completed.add(idx)
                    with open(resume_file, "w") as f:
                        json.dump(
                            {
                                "url": self.Url,
                                "total_size": total_size,
                                "chunk_size": chunk_size,
                                "workers": n,
                                "completed": list(completed),
                            },
                            f,
                        )
                    break  # exit retry loop if successful

                except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
                    attempt += 1
                    self._logger.warning(f"Chunk {idx} failed (attempt {attempt}/{max_retries}): {e}")
                    if attempt >= max_retries:
                        self._logger.error(f"Chunk {idx} failed after {max_retries} attempts, giving up.")
                        raise
                finally:
                    with lock:
                        tqdm_bar.update(1)

            return idx

        Parallel(n_jobs=n, prefer="threads")(
            delayed(_download_chunk)(i, s, e) for i, (s, e) in enumerate(ranges) if i not in completed
        )

        tqdm_bar.close()
        self._combine_parts(file_path, ranges, resume_file)
        self._logger.info(f"✅ Download complete → {file_path}")
        return file_path

    # ------------------- Helpers -------------------

    def _get_extract_dir(self, zip_path: str) -> str:
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        return os.path.join(self.DownloadPath, base_name)

    def _extract_zip(self, zip_path: str, extract_dir: str):
        self._logger.info(f"Extracting {zip_path} to {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for m in tqdm(zf.infolist(), desc="Extracting"):
                zf.extract(m, extract_dir)
        self._logger.info(f"✅ Extraction complete: {extract_dir}")

    @staticmethod
    def _combine_parts(file_path: str, ranges: List[Tuple[int, int]], resume_file: str):
        tmp_path = file_path + ".part"
        with open(tmp_path, "wb") as outfile:
            for i in range(len(ranges)):
                part_path = f"{file_path}.part{i}"
                with open(part_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
                os.remove(part_path)
        os.rename(tmp_path, file_path)
        if os.path.exists(resume_file):
            os.remove(resume_file)

    @staticmethod
    def _invalidate_cache(file_path: str, resume_file: str, extract_dir: str):
        """
        Remove any .part files and resume metadata for a given download,
        unless the extracted folder exists (fully skip if done).
        """
        if os.path.exists(extract_dir):
            # Extraction exists, nothing to delete
            return

        base_dir = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        for f in os.listdir(base_dir):
            if f.startswith(base_name) and ".part" in f:
                try:
                    os.remove(os.path.join(base_dir, f))
                except Exception:
                    pass

        if os.path.exists(resume_file):
            try:
                os.remove(resume_file)
            except Exception:
                pass
