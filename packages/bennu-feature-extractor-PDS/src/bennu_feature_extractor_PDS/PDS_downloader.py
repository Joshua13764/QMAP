from dataclasses import dataclass

from bennu_feature_extractor.step_templates.chunked_downloader import \
    ArchiveDownloadBase


@dataclass
class PDSDownloader(ArchiveDownloadBase):
    BaseUrl: str = "https://sbnarchive.psi.edu"
    AllowChunking: bool = True
    KeepArchive: bool = False
