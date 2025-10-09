import hashlib
from pathlib import Path
from typing import List, Optional, Final
from .models import ChecksumRecord, VerificationResult


class ChecksumVerifier:
    """Validates that file checksums match expected values."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir: Final[Path] = base_dir

    def _compute_md5(self, file_path: Path) -> str:
        """Compute the MD5 hash for a file."""
        hash_md5 = hashlib.md5()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def verify_all(self, records: List[ChecksumRecord]) -> List[VerificationResult]:
        """Verify all checksum records."""
        results: List[VerificationResult] = []

        for record in records:
            full_path = (self._base_dir / record.file_path).resolve()

            if not full_path.exists():
                results.append(
                    VerificationResult(
                        file_path=full_path,
                        expected_checksum=record.checksum,
                        actual_checksum=None,
                        is_valid=False,
                        error="File not found"
                    )
                )
                continue

            try:
                actual = self._compute_md5(full_path)
                is_valid = actual == record.checksum
                results.append(
                    VerificationResult(
                        file_path=full_path,
                        expected_checksum=record.checksum,
                        actual_checksum=actual,
                        is_valid=is_valid,
                        error=None if is_valid else "Checksum mismatch"
                    )
                )
            except OSError as e:
                results.append(
                    VerificationResult(
                        file_path=full_path,
                        expected_checksum=record.checksum,
                        actual_checksum=None,
                        is_valid=False,
                        error=str(e)
                    )
                )

        return results
