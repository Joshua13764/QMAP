from pathlib import Path
from typing import List
from .file_finder import FileFinder
from .checksum_reader import ChecksumReader
from .checksum_verifier import ChecksumVerifier
from .models import VerificationResult


def verify_directory(root_dir: str) -> List[VerificationResult]:
    """Search directory for all checksum files and validate them."""
    base_path = Path(root_dir).resolve()
    finder = FileFinder(base_path)
    verifier = ChecksumVerifier(base_path)

    all_invalid: List[VerificationResult] = []

    for checksum_file in finder.find_checksum_files():
        reader = ChecksumReader(checksum_file)
        records = reader.read_checksums()
        results = verifier.verify_all(records)
        all_invalid.extend([r for r in results if not r.is_valid])

    return all_invalid