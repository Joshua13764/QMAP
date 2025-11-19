import os
from typing import Any

from requests import Session
from tqdm import tqdm

from bennu_feature_extractor.step_templates.utils.session_factory import \
    DownloadSessionFactory


class SingleStreamDownloader:
    @staticmethod
    def download(
        url: str,
        file_path: str,
        timeout_seconds: int,
        total_size: int,
        logger: Any,
    ) -> None:
        session: Session = DownloadSessionFactory.create_with_retries()
        tmp_path: str = file_path + ".part"
        desc = "Downloading (single stream)"
        try:
            with session.get(url, stream=True, timeout=timeout_seconds) as response:
                response.raise_for_status()
                with tqdm(
                    total=total_size if total_size > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                ) as bar, open(tmp_path, "wb") as file_handle:
                    for chunk in response.iter_content(
                            chunk_size=16 * 1024 * 1024):
                        if chunk:
                            file_handle.write(chunk)
                            if total_size > 0:
                                bar.update(len(chunk))
            os.replace(tmp_path, file_path)
        except Exception as exc:
            logger.error(f"Single-stream download failed: {exc}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise
