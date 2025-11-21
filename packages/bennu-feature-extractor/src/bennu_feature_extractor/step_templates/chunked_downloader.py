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
from urllib.parse import ParseResult, parse_qs, urlparse

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
from bennu_feature_extractor.step_templates.utils.chunked_downloader_utils import \
    ChunkedDownloader
from bennu_feature_extractor.step_templates.utils.download_resume_manager import \
    DownloadResumeManager
from bennu_feature_extractor.step_templates.utils.session_factory import \
    DownloadSessionFactory
from bennu_feature_extractor.step_templates.utils.single_stream_downloader import \
    SingleStreamDownloader
from bennu_feature_extractor.step_templates.utils.zip_extractor import \
    ZipExtractor
from bennu_feature_extractor.task_step_base import TaskStepBase


@dataclass(frozen=True)
class ArchiveDownloadBase(TaskStepBase):
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

    BaseUrl: str = ""
    AllowChunking: bool = True
    KeepArchive: bool = False
    Extract: bool = True
    Resume: bool = True
    virtual_path_root: Path = Path("/data")

    Workers: int = field(
        default_factory=lambda: max(
            1, multiprocessing.cpu_count() - 2))
    ChunkSizeLimitMB: int = 50
    TimeoutSeconds: int = 60

    _session: Optional[requests.Session] = field(
        init=False, repr=False, default=None)

    def get_hash(self) -> int:
        return (self.DownloadPath, self.Url, self.BaseUrl, self.KeepArchive,
                self.Extract, self.virtual_path_root).__hash__()

    def run(self, env: FSEnvironment) -> FSEnvironment:
        if self.BaseUrl and not self.Url.startswith(self.BaseUrl):
            raise ValueError(f"URL must start with {self.BaseUrl}")

        zip_path = self._download_archive()
        if self.Extract:
            extract_dir = self._get_extract_dir(zip_path)
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

        return FSEnvironmentFactory.from_folder(
            Path(extract_dir), extensions=".xml")

    def _download_archive(self) -> str:
        parsed: ParseResult = urlparse(self.Url)
        file_name: str = os.path.basename(parsed.path) or "download.zip"
        file_path: str = os.path.join(self.DownloadPath, file_name)
        extract_dir: str = self._get_extract_dir(file_path)
        resume_file: str = os.path.join(
            self.DownloadPath,
            f"{file_name}.resume.json")

        if self.Extract and os.path.exists(extract_dir):
            self.logger.info(
                f"Extraction folder exists ({extract_dir}) — skipping download.")
            return file_path

        self.logger.info(f"Fetching headers for: {self.Url}")
        session: requests.Session = DownloadSessionFactory.create_with_retries()
        head: requests.Response = session.head(
            self.Url,
            allow_redirects=True,
            timeout=self.TimeoutSeconds)
        total_size = int(head.headers.get("Content-Length", 0))
        supports_range: bool = "bytes" in head.headers.get(
            "Accept-Ranges", "").lower()

        if total_size == 0:
            self.logger.warning(
                "Unknown file size from server; proceeding with single-stream download.")
            self._download_single_stream(file_path, 0)
            return file_path

        if self.AllowChunking and supports_range:
            self._download_in_chunks(
                file_path, total_size, resume_file, extract_dir)
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

    def _download_single_stream(self, file_path: str, total_size: int) -> None:
        SingleStreamDownloader.download(
            url=self.Url,
            file_path=file_path,
            timeout_seconds=self.TimeoutSeconds,
            total_size=total_size,
            logger=self.logger,
        )

    def _download_in_chunks(
            self, file_path: str, total_size: int, resume_file: str, extract_dir: str) -> None:
        ChunkedDownloader.download(
            url=self.Url,
            file_path=file_path,
            total_size=total_size,
            resume_file=resume_file,
            workers=self.Workers,
            chunk_size_limit_mb=self.ChunkSizeLimitMB,
            timeout_seconds=self.TimeoutSeconds,
            resume_enabled=self.Resume,
            extract_dir=extract_dir,
            logger=self.logger,
        )

    def _get_extract_dir(self, zip_path: str) -> str:
        base_name: str = os.path.splitext(os.path.basename(zip_path))[0]
        return os.path.join(self.DownloadPath, base_name)

    def _extract_zip(self, zip_path: str, extract_dir: str) -> None:
        ZipExtractor.extract(zip_path, extract_dir, self.logger)

    @staticmethod
    def _combine_parts(file_path: str, n_parts: int, resume_file: str) -> None:
        DownloadResumeManager.combine_parts(file_path, n_parts, resume_file)

    @staticmethod
    def _invalidate_cache(file_path: str, resume_file: str,
                          extract_dir: str) -> None:
        DownloadResumeManager.invalidate_cache(
            file_path, resume_file, extract_dir)
