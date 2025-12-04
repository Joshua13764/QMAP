import json
import math
import os
from threading import Lock
from typing import Any, Dict, Set

import requests
from joblib import Parallel, delayed
from tqdm import tqdm

from Boulder_Statistics.steps.utils.download_resume_manager import \
    DownloadResumeManager
from Boulder_Statistics.steps.utils.session_factory import \
    DownloadSessionFactory


class ChunkedDownloader:
    @staticmethod
    def download(
        url: str,
        file_path: str,
        total_size: int,
        resume_file: str,
        workers: int,
        chunk_size_limit_mb: int,
        timeout_seconds: int,
        resume_enabled: bool,
        extract_dir: str,
        logger: Any,
    ) -> None:
        session: requests.Session = DownloadSessionFactory.create_with_retries()

        n: int = workers
        max_chunk: int = chunk_size_limit_mb * 1024 * 1024
        chunk_size: int = min(math.ceil(total_size / n), max_chunk)
        ranges: list[tuple[int, int]] = [
            (index * chunk_size, min((index + 1) * chunk_size - 1, total_size - 1))
            for index in range(math.ceil(total_size / chunk_size))
        ]

        completed: Set[int] = set()
        resume_meta: Dict[str, Any] = {}

        if resume_enabled and os.path.exists(resume_file):
            try:
                with open(resume_file) as resume_handle:
                    resume_meta = json.load(resume_handle)
                    completed = set(resume_meta.get("completed", []))
            except Exception:
                logger.warning("Corrupt resume file — resetting cache.")
                DownloadResumeManager.invalidate_cache(
                    file_path, resume_file, extract_dir
                )
                completed.clear()

        valid_cache = (
            resume_enabled
            and resume_meta.get("url") == url
            and resume_meta.get("total_size") == total_size
            and resume_meta.get("chunk_size") == chunk_size
            and resume_meta.get("workers") == n
        )

        if not valid_cache and os.path.exists(resume_file):
            logger.warning(
                "Cache invalid — settings changed. Clearing old partial files."
            )
            DownloadResumeManager.invalidate_cache(
                file_path,
                resume_file,
                extract_dir,
            )
            completed.clear()

        chunk_size_mb: float = chunk_size / (1024 ** 2)
        total_size_gb: float = total_size / (1024 ** 3)

        logger.info(
            f"Downloading with {n} workers, chunk size = {
                chunk_size_mb:.2f} MB, total chunks = {
                len(ranges)}, total size = {
                total_size_gb:.2f} GB"
        )

        progress_bar = tqdm(
            total=len(ranges),
            desc="Downloading (chunked)",
            initial=len(completed),
        )
        lock = Lock()

        def download_chunk(index: int, start: int, end: int,
                           max_retries: int = 5):
            if index in completed:
                with lock:
                    progress_bar.update(1)
                return

            headers: Dict[str, str] = {"Range": f"bytes={start}-{end}"}
            part_path: str = f"{file_path}.part{index}"
            attempt = 0

            while attempt < max_retries:
                try:
                    with session.get(
                        url, headers=headers, stream=True, timeout=timeout_seconds
                    ) as response:
                        response.raise_for_status()
                        with open(part_path, "wb") as part_handle:
                            for chunk in response.iter_content(
                                chunk_size=16 * 1024 * 1024
                            ):
                                if chunk:
                                    part_handle.write(chunk)

                    completed.add(index)
                    if resume_enabled:
                        with open(resume_file, "w") as resume_handle:
                            json.dump(
                                {
                                    "url": url,
                                    "total_size": total_size,
                                    "chunk_size": chunk_size,
                                    "workers": n,
                                    "completed": sorted(list(completed)),
                                },
                                resume_handle,
                            )

                    with lock:
                        progress_bar.update(1)
                    return
                except (
                    requests.exceptions.RequestException,
                    requests.exceptions.ChunkedEncodingError,
                ) as error:
                    attempt += 1
                    logger.warning(
                        f"Chunk {index} failed (attempt {attempt}/{max_retries}): {error}"
                    )
                    if attempt >= max_retries:
                        logger.error(
                            f"Chunk {index} failed after {max_retries} attempts."
                        )
                        raise

        Parallel(n_jobs=n, prefer="threads")(
            delayed(download_chunk)(index, start, end)
            for index, (start, end) in enumerate(ranges)
            if index not in completed
        )

        progress_bar.close()
        DownloadResumeManager.combine_parts(
            file_path, len(ranges), resume_file)
