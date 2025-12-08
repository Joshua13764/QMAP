from dataclasses import dataclass
from typing import Any

from boulder_statistics.steps.chunked_downloader import ArchiveDownloadBase


@dataclass(frozen=True)
class PDSDownloader(ArchiveDownloadBase):
    BaseUrl: str = "https://sbnarchive.psi.edu"
    AllowChunking: bool = True
    KeepArchive: bool = False

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.BaseUrl, self.AllowChunking, self.KeepArchive)
