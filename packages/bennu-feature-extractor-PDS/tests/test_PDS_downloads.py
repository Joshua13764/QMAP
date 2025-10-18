# test_downloader.py
import logging
import pytest
import requests
from pathlib import Path

from bennu_feature_extractor_PDS.PDS_downloader import PDSDownloader

def test_run_skips_when_already_extracted(tmp_path, monkeypatch, caplog):
    """
    If the extraction directory already exists, the downloader should skip
    both download and extraction entirely (and therefore not perform any HTTP calls).
    """
    # Arrange
    url = "https://sbnarchive.psi.edu/some.zip"
    download_dir = tmp_path
    extract_dir = download_dir / "some"  # derived from 'some.zip' -> 'some'
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Make any accidental .head() call blow up so we know it wasn't used
    def _boom(*args, **kwargs):
        raise AssertionError("HEAD should not be called when extract dir exists")

    monkeypatch.setattr(requests.Session, "head", lambda self, *a, **k: _boom())

    logger = logging.getLogger("test")
    with caplog.at_level(logging.INFO):
        d = PDSDownloader(
            DownloadPath=str(download_dir),
            Url=url,
            _logger=logger,
        )

        # Act
        d.run()

    # Assert
    log_text = " ".join(m.lower() for m in caplog.messages)
    assert "skipping download" in log_text  # from _download_archive
    assert "already extracted" in log_text  # from run()
    # No archive should have been created
    assert not (download_dir / "some.zip").exists()


def test_baseurl_validation(tmp_path):
    """
    When BaseUrl is set and Url doesn't start with it, run() should raise ValueError.
    """
    logger = logging.getLogger("test")
    d = PDSDownloader(
        DownloadPath=str(tmp_path),
        Url="https://not-psi.edu/file.zip",
        _logger=logger,
        BaseUrl="https://sbnarchive.psi.edu",
    )
    with pytest.raises(ValueError):
        d.run()
