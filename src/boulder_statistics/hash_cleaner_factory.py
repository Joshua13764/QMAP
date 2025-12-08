from pathlib import Path
from typing import Any


class HashCleanerFactory():
    @staticmethod
    def clean_hashable(obj: Any) -> Any:
        match obj:
            case Path() as p:
                return p.as_posix()
            case _:
                return obj
