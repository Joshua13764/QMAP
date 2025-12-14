from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from boulder_statistics.steps.extract_obj_as_cubemap_lods import \
    ExtractOBJAsCubemapLods


@dataclass(frozen=True)
class OBJToLAS(ExtractOBJAsCubemapLods):
    colour_column_name: Callable[[str], str] = lambda face: f'{face}_ratio'
    message_prefix_generator: Callable[[int, Path], str] = field(
        default=lambda depth, src_path: f"""Rendering LAS for lod_depth {depth} and model {
            src_path.name}"""
    )
    n_jobs: int = field(default=1)
    export_resolution: int = field(default=512)
    verbose: bool = field(default=False)
