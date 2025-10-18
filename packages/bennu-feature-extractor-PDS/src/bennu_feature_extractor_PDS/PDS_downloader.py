from dataclasses import dataclass

from bennu_feature_extractor.step_templates.chunked_downloader import ArchiveDownloadBase

@dataclass
class PDSDownloader(ArchiveDownloadBase):
    BaseUrl: str = "https://sbnarchive.psi.edu"
    AllowChunking: bool = True
    KeepArchive: bool = False
    # Everything else (Workers, ChunkSizeLimitMB, etc.) comes from the base class
