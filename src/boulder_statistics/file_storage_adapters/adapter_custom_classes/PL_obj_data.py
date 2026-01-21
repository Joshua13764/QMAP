from dataclasses import dataclass

from polars import LazyFrame


@dataclass
class PLOBJData():
    verts: LazyFrame
    tris: LazyFrame
