from pathlib import Path
from typing import Final, List


class FileFinder:
    """Responsible only for locating checksum files in a directory."""

    def __init__(self, root_dir: Path, checksum_filename: str = "checksum.txt") -> None:
        self._root_dir: Final[Path] = root_dir
        self._checksum_filename: Final[str] = checksum_filename

    def find_checksum_files(self) -> List[Path]:
        """Recursively find all checksum files in the directory tree."""
        return [p for p in self._root_dir.rglob(self._checksum_filename) if p.is_file()]
