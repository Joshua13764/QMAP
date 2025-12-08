from dataclasses import dataclass
from typing import Any

from boulder_statistics.steps.chunked_downloader import ArchiveDownloadBase


@dataclass(frozen=True)
class BestModelDownloader(ArchiveDownloadBase):
    BaseUrl: str = "https://zenodo.org/records/8171052/files"
    AllowChunking: bool = True
    KeepArchive: bool = False

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.BaseUrl, self.AllowChunking, self.KeepArchive)
