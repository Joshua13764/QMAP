from typing import Dict
from pathlib import Path

import attr
from bennu_feature_extractor.environment_tools.base_classes.fs_medium_base import FileStorageMediumBase

@attr.define()
class LocalDiskFile():
    virtual_path : Path

    def get_actual_path(self, root_path : Path):
        return root_path / self.virtual_path

class LocalDisk(FileStorageMediumBase):

    def __init__(self, root_path : Path) -> None:
        self.root_path = root_path
        self.local_disk_files : Dict[str, LocalDiskFile] = {}

    def does_path_exist(self, virtual_path: Path) -> bool:
        return virtual_path.as_uri() in self.local_disk_files
    
    def add_local_disk_file(self, virtual_path: Path) -> Path:
        if self.does_path_exist(virtual_path): raise FileExistsError(f"The virtual file path {virtual_path} all-ready exits in LocalDisk")

        disk_file : LocalDiskFile = LocalDiskFile(virtual_path = virtual_path)
        self.local_disk_files[virtual_path.as_uri()] = disk_file

        return disk_file.get_actual_path(self.root_path)

    def get_file_path_to_write(self, virtual_path: Path) -> Path:

        file_path : Path = self.local_disk_files[virtual_path.as_uri()].get_actual_path(self.root_path) if self.does_path_exist(virtual_path)else self.add_local_disk_file(virtual_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path
    