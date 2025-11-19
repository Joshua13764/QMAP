from dataclasses import dataclass

from bennu_feature_extractor.step_templates.chunked_downloader import \
    ArchiveDownloadBase


@dataclass(frozen=True)
class BestModelDownloader(ArchiveDownloadBase):
    BaseUrl: str = "https://zenodo.org/records/8171052/files"
    AllowChunking: bool = True
    KeepArchive: bool = False
