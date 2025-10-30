import json
import math
import multiprocessing
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse

import requests
from joblib import Parallel, delayed
from prefect import task
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from bennu_feature_extractor.environment_tools.fs_environment import \
    FSEnvironment
from bennu_feature_extractor.environment_tools.utils.FS_environment_factory import \
    FSEnvironmentFactory
from bennu_feature_extractor.step_base import StepBase


@dataclass
class ArchiveDownloadBase(StepBase):
    """
    A reusable base for 'download-then-extract' steps.

    Features:
    - Optional BaseUrl validation
    - Optional parallel, resumable chunked downloads with .resume.json metadata
    - Clean single-stream fallback (with progress bar)
    - Robust requests.Session with retries
    - Zip extraction with progress
    """
    DownloadPath: str
    Url: str

    # Behavior toggles / options
    # If non-empty, enforce Url.startswith(BaseUrl)
    BaseUrl: str = ""
    AllowChunking: bool = True             # If False, force single-stream download
    # If True, leave the .zip on disk after extraction
    KeepArchive: bool = False
    Extract: bool = True                   # If True, extract as a zip
    Resume: bool = True
    virtual_path_root: Path = Path("/data")

    # Performance / networking
    Workers: int = field(
        default_factory=lambda: max(
            1, multiprocessing.cpu_count() - 2))
    ChunkSizeLimitMB: int = 50             # Cap per-chunk size
    TimeoutSeconds: int = 60               # Request timeout per chunk/stream

    # Internal
    _session: Optional[requests.Session] = field(
        init=False, repr=False, default=None)

    def get_hash(self) -> int:
        return (self.DownloadPath, self.Url, self.BaseUrl, self.KeepArchive,
                self.Extract, self.virtual_path_root).__hash__()

    @property
    def name(self) -> str:
        return __name__

    def __post_init__(self):
        os.makedirs(self.DownloadPath, exist_ok=True)
        self._session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    # ------------------- Public entrypoint -------------------

    def run(self, env: FSEnvironment) -> FSEnvironment:
        if self.BaseUrl and not self.Url.startswith(self.BaseUrl):
            raise ValueError(f"URL must start with {self.BaseUrl}")

        zip_path = self._download_archive()
        if self.Extract:
            extract_dir = self._get_extract_dir(zip_path)

            # Skip extraction if folder already exists
            if os.path.exists(extract_dir):
                self.logger.info(
                    f"Already extracted: {extract_dir}, skipping extraction.")
            else:
                self._extract_zip(zip_path, extract_dir)

            if not self.KeepArchive and os.path.isfile(zip_path):
                try:
                    os.remove(zip_path)
                except Exception:
                    self.logger.warning(f"Could not remove archive {zip_path}")

        return FSEnvironment.merge(
            env,
            FSEnvironmentFactory.from_folder(Path(extract_dir))
        )

    # ------------------- Download logic -------------------

    def _download_archive(self) -> str:
        parsed = urlparse(self.Url)
        file_name = os.path.basename(parsed.path) or "download.zip"
        file_path = os.path.join(self.DownloadPath, file_name)
        extract_dir = self._get_extract_dir(file_path)
        resume_file = os.path.join(
            self.DownloadPath,
            f"{file_name}.resume.json")

        # If already extracted, skip download entirely
        if self.Extract and os.path.exists(extract_dir):
            self.logger.info(
                f"Extraction folder exists ({extract_dir}) — skipping download.")
            return file_path

        # HEAD request for size + range support
        self.logger.info(f"Fetching headers for: {self.Url}")
        head = self._session.head(
            self.Url,
            allow_redirects=True,
            timeout=self.TimeoutSeconds)
        total_size = int(head.headers.get("Content-Length", 0))
        supports_range = "bytes" in head.headers.get(
            "Accept-Ranges", "").lower()

        if total_size == 0:
            # Fall back to streaming without known size (still show a
            # spinner-like bar)
            self.logger.warning(
                "Unknown file size from server; proceeding with single-stream download.")
            self._download_single_stream(file_path, total_size=0)
            return file_path

        # Decide chunking vs single-stream
        if self.AllowChunking and supports_range:
            self._download_in_chunks(file_path, total_size, resume_file)
        else:
            if self.AllowChunking and not supports_range:
                self.logger.info(
                    "Server does not support Range requests; falling back to single-stream.")
            elif not self.AllowChunking:
                self.logger.info(
                    "AllowChunking=False; using single-stream download.")
            self._download_single_stream(file_path, total_size)

        self.logger.info(f"✅ Download complete → {file_path}")
        return file_path

    def _download_single_stream(self, file_path: str, total_size: int):
        tmp_path = file_path + ".part"
        desc = "Downloading (single stream)"
        try:
            with self._session.get(self.Url, stream=True, timeout=self.TimeoutSeconds) as r:
                r.raise_for_status()
                # tqdm works with unknown totals; if total_size=0, it will
                # behave like an indeterminate bar
                with tqdm(total=total_size if total_size > 0 else None,
                          unit="B", unit_scale=True, desc=desc) as bar, open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            if total_size > 0:
                                bar.update(len(chunk))
            os.replace(tmp_path, file_path)
        except Exception as exc:
            self.logger.error(f"Single-stream download failed: {exc}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    def _download_in_chunks(self, file_path: str,
                            total_size: int, resume_file: str):
        # Build ranges
        n = self.Workers
        max_chunk = self.ChunkSizeLimitMB * 1024 * 1024
        chunk_size = min(math.ceil(total_size / n), max_chunk)
        ranges = [
            (i * chunk_size, min((i + 1) * chunk_size - 1, total_size - 1))
            for i in range(math.ceil(total_size / chunk_size))
        ]

        # Resume metadata
        completed: Set[int] = set()
        resume_meta: Dict[str, Any] = {}

        if self.Resume and os.path.exists(resume_file):
            try:
                with open(resume_file) as f:
                    resume_meta = json.load(f)
                    completed = set(resume_meta.get("completed", []))
            except Exception:
                self.logger.warning("Corrupt resume file — resetting cache.")
                self._invalidate_cache(
                    file_path, resume_file, self._get_extract_dir(file_path))
                completed.clear()

        valid_cache = (
            self.Resume
            and resume_meta.get("url") == self.Url
            and resume_meta.get("total_size") == total_size
            and resume_meta.get("chunk_size") == chunk_size
            and resume_meta.get("workers") == n
        )

        if not valid_cache and os.path.exists(resume_file):
            self.logger.warning(
                "Cache invalid — settings changed. Clearing old partial files.")
            self._invalidate_cache(
                file_path,
                resume_file,
                self._get_extract_dir(file_path))
            completed.clear()

        self.logger.info(
            f"Downloading with {n} workers, chunk size = {chunk_size /
                                                          (1024**2):.2f} MB, total chunks = {len(ranges)}, total size = {total_size /
                                                                                                                         (1024**3):.2f} GB"
        )

        bar = tqdm(
            total=len(ranges),
            desc="Downloading (chunked)",
            initial=len(completed))
        lock = Lock()

        def _download_chunk(idx: int, start: int, end: int,
                            max_retries: int = 5):
            if idx in completed:
                with lock:
                    bar.update(1)
                return

            headers = {"Range": f"bytes={start}-{end}"}
            part_path = f"{file_path}.part{idx}"
            attempt = 0

            while attempt < max_retries:
                try:
                    with self._session.get(self.Url, headers=headers, stream=True, timeout=self.TimeoutSeconds) as r:
                        r.raise_for_status()
                        with open(part_path, "wb") as f:
                            for chunk in r.iter_content(
                                    chunk_size=16 * 1024 * 1024):
                                if chunk:
                                    f.write(chunk)

                    completed.add(idx)
                    if self.Resume:
                        with open(resume_file, "w") as f:
                            json.dump(
                                {
                                    "url": self.Url,
                                    "total_size": total_size,
                                    "chunk_size": chunk_size,
                                    "workers": n,
                                    "completed": sorted(list(completed)),
                                },
                                f,
                            )

                    with lock:
                        bar.update(1)
                    return
                except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
                    attempt += 1
                    self.logger.warning(
                        f"Chunk {idx} failed (attempt {attempt}/{max_retries}): {e}")
                    if attempt >= max_retries:
                        self.logger.error(
                            f"Chunk {idx} failed after {max_retries} attempts.")
                        raise

        # Parallel downloads for incomplete chunks
        Parallel(n_jobs=n, prefer="threads")(
            delayed(_download_chunk)(i, s, e) for i, (s, e) in enumerate(ranges) if i not in completed
        )

        bar.close()
        self._combine_parts(file_path, len(ranges), resume_file)

    # ------------------- Helpers -------------------

    def _get_extract_dir(self, zip_path: str) -> str:
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        return os.path.join(self.DownloadPath, base_name)

    def _extract_zip(self, zip_path: str, extract_dir: str):
        self.logger.info(f"Extracting {zip_path} → {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            with tqdm(total=len(members), desc="Extracting") as bar:
                for m in members:
                    zf.extract(m, extract_dir)
                    bar.update(1)
        self.logger.info(f"Extraction complete: {extract_dir}")

    @staticmethod
    def _combine_parts(file_path: str, n_parts: int, resume_file: str):
        tmp_path = file_path + ".part"
        with open(tmp_path, "wb") as outfile:
            for i in range(n_parts):
                part_path = f"{file_path}.part{i}"
                with open(part_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
                os.remove(part_path)
        os.replace(tmp_path, file_path)
        if os.path.exists(resume_file):
            try:
                os.remove(resume_file)
            except Exception:
                pass

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

        try:
            for f in os.listdir(base_dir):
                if f.startswith(base_name) and ".part" in f:
                    try:
                        os.remove(os.path.join(base_dir, f))
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

        if os.path.exists(resume_file):
            try:
                os.remove(resume_file)
            except Exception:
                pass
