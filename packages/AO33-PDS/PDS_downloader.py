from dataclasses import dataclass

from AO33.step_templates.chunked_downloader import ArchiveDownloadBase

@dataclass
class PDSDownloader(ArchiveDownloadBase):
    BaseUrl: str = "https://sbnarchive.psi.edu"
    AllowChunking: bool = True
    KeepArchive: bool = False
    # Everything else (Workers, ChunkSizeLimitMB, etc.) comes from the base class
