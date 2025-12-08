from dataclasses import dataclass
from pathlib import Path

from boulder_statistics.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase
from boulder_statistics.environment_tools.base_classes.fs_path_base import \
    FSPathBase


@dataclass(frozen=True)
class FSPathLocalDisk(FSPathBase):
    root_path: str

    @property
    def actual_path(self) -> Path:
        return Path(self.root_path) / Path(*self.path)

    @property
    def exists(self) -> bool:
        return self.actual_path.exists()

    def make_directory(self) -> None:
        self.actual_path.parent.mkdir(parents=True, exist_ok=True)

    def copy_as_new(self, new_root_path: Path,
                    new_extension: str, markers: tuple[FSMarkerBase, ...] = ()) -> 'FSPathLocalDisk':
        return FSPathLocalDisk(
            path=Path(*self.path).with_suffix(new_extension).parts,
            markers=tuple(markers),
            root_path=new_root_path.as_posix(),
        )

    def copy_as_new_name(self, new_root_path: Path,
                         new_extension: str, markers: tuple[FSMarkerBase, ...] = ()) -> 'FSPathLocalDisk':
        return FSPathLocalDisk(
            path=Path(
                *
                self.path).with_name(
                Path(
                    *
                    self.path).stem +
                new_extension).parts,
            markers=tuple(markers),
            root_path=new_root_path.as_posix(),
        )

    def copy_with_stem_prefix_and_suffix(
            self, stem_prefix: str = "", stem_suffix: str = "", markers: tuple[FSMarkerBase, ...] = ()) -> "FSPathLocalDisk":

        return FSPathLocalDisk(
            path=Path(
                *self.path).with_name(
                stem_prefix +
                Path(
                    *self.path).stem +
                stem_suffix + Path(
                    *self.path).suffix).parts,
            markers=markers,
            root_path=self.root_path
        )

    def copy_from_folder(self, new_sub_path: Path,
                         markers: tuple[FSMarkerBase, ...] = ()) -> 'FSPathLocalDisk':
        return FSPathLocalDisk(
            path=(Path(*self.path) / new_sub_path).parts,
            markers=markers,
            root_path=self.root_path,
        )
