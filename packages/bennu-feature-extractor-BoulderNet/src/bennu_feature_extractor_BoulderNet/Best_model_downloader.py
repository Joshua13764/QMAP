# B: BoulderNetBestModelLoadStep (no chunking, but now gains retries + progress bars)
from dataclasses import dataclass

from bennu_feature_extractor.step_tools.chunked_downloader import ArchiveDownloadBase

@dataclass
class BestModelDownloader(ArchiveDownloadBase):
    BaseUrl: str = ""           # or set one if you want to enforce
    AllowChunking: bool = False # force single-stream download like the original
    KeepArchive: bool = False
