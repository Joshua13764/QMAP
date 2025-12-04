from dataclasses import dataclass

from Boulder_Statistics.steps.chunked_downloader import ArchiveDownloadBase


@dataclass(frozen=True)
class PDSDownloader(ArchiveDownloadBase):
    BaseUrl: str = "https://sbnarchive.psi.edu"
    AllowChunking: bool = True
    KeepArchive: bool = False
