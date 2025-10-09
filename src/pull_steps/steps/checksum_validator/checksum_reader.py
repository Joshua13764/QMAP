from pathlib import Path
from typing import List
from .models import ChecksumRecord


class ChecksumReader:
    """Parses a checksum text file into ChecksumRecord objects."""

    def __init__(self, checksum_file: Path) -> None:
        self._checksum_file: Path = checksum_file

    def read_checksums(self) -> List[ChecksumRecord]:
        """Reads the checksum file into structured records."""
        records: List[ChecksumRecord] = []
        with self._checksum_file.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    checksum_str, rel_path = parts
                    records.append(ChecksumRecord(
                        checksum=checksum_str,
                        file_path=Path(rel_path)
                    ))
        return records
