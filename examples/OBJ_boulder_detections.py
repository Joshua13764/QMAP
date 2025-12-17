from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.steps.OBJ_to_DIS import OBJToDIS
from boulder_statistics.steps.OBJ_to_LAS import OBJToLAS
from boulder_statistics.steps.simple_function_apply import SimpleFunctionApply
from boulder_statistics.steps.simple_function_import_export import \
    SimpleFunctionImportExport
from boulder_statistics.steps.simple_request import SimpleRequest
from boulder_statistics.steps_orchestrator import StepsOrchestrator

detections_from_bennu_model: Path = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA\detections_from_bennu_model")

get_bennu_obj = SimpleRequest(
    task_name=f"Downloader for the bennu OBJ (LQ) mesh",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/g_00880mm_alt_ptm_0000n00000_v020.obj",
    fs_path=detections_from_bennu_model.as_posix(),
    sub_path=Path(
        "OCAMS",
        "Global Bennu 3D model - OLA v20 PTM.obj").as_posix(),
    markers=(
        FSMarkerString(
            value="OCAMS Model"),
        FSMarkerString("ProjectModel"))
)


get_local_area_scaling_lods = OBJToLAS(
    task_name=f"Convert bennu Mesh to LAS maps",
    run_after_task_names=(get_bennu_obj.task_name,),
    export_folder=FSPathLocalDisk(
        path=("Bennu mesh LQ OBJ to LAS",),
        markers=tuple(),
        root_path=detections_from_bennu_model.as_posix()),
    depth=4,
    skip_if_exists=True,
    input_markers=(FSMarkerString("ProjectModel"),),
    output_markers=(FSMarkerString("ProjectModel_LAS"),),
    adapter=FSNumpyAdapter(
        export_debug_plots=True,
        title="LAS export",
        transform=lambda x: 1 / x,
        colour_bar_title="1 / LAS factor"),
    verbose=False
)

get_displacement_lods = OBJToDIS(
    task_name=f"Convert bennu Mesh to displacement maps",
    run_after_task_names=(get_bennu_obj.task_name,),
    export_folder=FSPathLocalDisk(
        path=("Bennu mesh LQ OBJ to DIS",),
        markers=tuple(),
        root_path=detections_from_bennu_model.as_posix()),
    depth=4,
    skip_if_exists=True,
    input_markers=(FSMarkerString("ProjectModel"),),
    output_markers=(FSMarkerString("ProjectModel_DIS"),),
    adapter=FSNumpyAdapter(
        export_debug_plots=True,
        title="DIS export",
        colour_bar_title="DIS factor"
    ),
    verbose=False
)


# def blur_by_fraction(img: NDArray[np.float64],
#                      fraction: float = 0.5, truncate: float = 4.0) -> NDArray[np.float64]:
#     return gaussian_filter(img, sigma=max(img.shape) *
#                            fraction, truncate=truncate)


# apply_blur_to_displacement_lods_tasks: List[SimpleFunctionApply[NDArray[np.float64]]] = [
# ]

# sizes: List[float] = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
# for size in sizes:

#     apply_blur_to_displacement_lods_tasks.append(SimpleFunctionApply(
#         task_name=f"Apply a LPF of {size} to bennu Mesh displacement maps",
#         run_after_task_names=(get_displacement_lods.task_name,),
#         input_markers=(FSMarkerString("ProjectModel_DIS"),),
#         output_markers=(FSMarkerString("ProjectModel_DIS_LPF"),),
#         read_adapter=FSNumpyAdapter(),
#         write_adapter=FSNumpyAdapter(),
#         function_to_apply=lambda img: blur_by_fraction(img, fraction=size),
#         import_folder=FSPathLocalDisk(
#             path=(f"Bennu mesh LQ OBJ to DIS",),
#             markers=tuple(),
#             root_path=detections_from_bennu_model.as_posix()),
#         export_folder=FSPathLocalDisk(
#             path=(f"Bennu mesh LQ OBJ to DIS LPF size {size}",),
#             markers=tuple(),
#             root_path=detections_from_bennu_model.as_posix()),
#         n_jobs=4
#     ))

#     apply_blur_to_displacement_lods_tasks.append(SimpleFunctionApply(
#         task_name=f"Apply a LPF of {size} to bennu Mesh displacement maps (plotted)",
#         run_after_task_names=(get_displacement_lods.task_name,),
#         input_markers=(FSMarkerString("ProjectModel_DIS"),),
#         output_markers=(FSMarkerString("ProjectModel_DIS_LPF_plot"),),
#         read_adapter=FSNumpyAdapter(),
#         write_adapter=FSNumpyAdapterMatrixPlot(
#             title=f"Bennu mesh LQ OBJ to DIS LPF size {size} (plotted)",
#             colour_bar_title="radius"
#         ),
#         function_to_apply=lambda img: blur_by_fraction(img, fraction=size),
#         import_folder=FSPathLocalDisk(
#             path=(f"Bennu mesh LQ OBJ to DIS",),
#             markers=tuple(),
#             root_path=detections_from_bennu_model.as_posix()),
#         export_folder=FSPathLocalDisk(
#             path=(f"Bennu mesh LQ OBJ to DIS LPF size {size} (plotted)",),
#             markers=tuple(),
#             root_path=detections_from_bennu_model.as_posix()),
#         n_jobs=4
#     ))


steps = [get_bennu_obj, get_local_area_scaling_lods,
         #  get_displacement_lods,
         #  #  get_local_area_scaling_lods_plotted,
         #  plot_displacement_lods_plotted,
         #  *apply_blur_to_displacement_lods_tasks,
         ]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
        steps, cache)
