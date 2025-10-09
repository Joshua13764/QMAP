from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

@dataclass(frozen=True, slots=True)
class ChecksumRecord:
    """Represents one line in a checksum file."""
    checksum: str
    file_path: Path


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Represents the result of verifying a single checksum."""
    file_path: Path
    expected_checksum: str
    actual_checksum: Optional[str]
    is_valid: bool
    error: Optional[str] = None
