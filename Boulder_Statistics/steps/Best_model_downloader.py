from dataclasses import dataclass

from Boulder_Statistics.steps.chunked_downloader import ArchiveDownloadBase


@dataclass(frozen=True)
class BestModelDownloader(ArchiveDownloadBase):
    BaseUrl: str = "https://zenodo.org/records/8171052/files"
    AllowChunking: bool = True
    KeepArchive: bool = False
