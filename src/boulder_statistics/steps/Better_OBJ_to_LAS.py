from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

from numpy import float64
from numpy.typing import NDArray

from boulder_statistics.file_storage_adapters.adapter_custom_classes.PL_obj_data import \
    PLOBJData
from boulder_statistics.lods.lod_cubemap_utils import LODCubemapUtils
from boulder_statistics.steps.base.one_to_one_step_base import OneToOneStepBase
from boulder_statistics.steps.utils.Bennu_OBJ_to_LAS_cubemap_generator import \
    BennuOBJToLASCubemapGenerator
from boulder_statistics.steps.utils.Better_polars_3D_expressions import \
    BetterPolars3DExpressions

ArrayType = NDArray[float64]


@dataclass(frozen=True)
class BetterOBJToLAS(
        OneToOneStepBase[PLOBJData, BennuOBJToLASCubemapGenerator]):

    lod_depth: int = field(default_factory=lambda: 4)
    lod_res: int = field(default_factory=lambda: 512)
    skip_if_exists: bool = field(default_factory=lambda: True)
    colour_column_name: Callable[[str], str] = field(
        default_factory=lambda: lambda face: f'{face}_ratio')

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.task_name,)

    def get_object_relative_export_path(
            self, input_object: PLOBJData, output_object: BennuOBJToLASCubemapGenerator) -> Tuple[str, ...]:
        return (
            f"""cubemap_lod_depth ({
                self.lod_depth}) res ({
                self.lod_res})""",
        )

    def object_operation(
            self, input_object: PLOBJData) -> BennuOBJToLASCubemapGenerator:

        object_data: PLOBJData = BetterPolars3DExpressions.process_mesh_projection_scaling(
            input_object)
        object_data = BetterPolars3DExpressions.add_displacement_columns(
            object_data)

        return BennuOBJToLASCubemapGenerator(
            tiles=LODCubemapUtils.get_all_cubemap_tiles_for_depths(
                max_depth=self.lod_depth),
            tile_resolution=self.lod_res,
            generator_input=object_data,
            colour_column_name=self.colour_column_name
        )
