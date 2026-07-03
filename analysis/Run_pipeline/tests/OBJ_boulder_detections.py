from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_doh

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.file_storage_adapters.png_adapter import FSPNGAdapter
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

get_bennu_obj_hq = SimpleRequest(
    task_name=f"Downloader for the bennu OBJ (HQ) mesh",
    url="https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005069/bennu_OLA_v21_PTM_very-high.obj",
    fs_path=detections_from_bennu_model.as_posix(),
    sub_path=Path(
        "OCAMS",
        "Global Bennu 3D model - OLA v20 PTM HQ.obj").as_posix(),
    markers=(
        FSMarkerString(
            value="OCAMS ModelHQ"),
        FSMarkerString("ProjectModelHQ"))
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


def blur_by_fraction(img: NDArray[np.float64],
                     fraction: float = 0.5, truncate: float = 4.0) -> NDArray[np.float64]:
    return gaussian_filter(img, sigma=max(img.shape) *
                           fraction, truncate=truncate)


apply_blur_to_displacement_lods_tasks: List[SimpleFunctionApply[NDArray[np.float64]]] = [
]

sizes: List[float] = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]
for size in sizes:

    apply_blur_to_displacement_lods_tasks.append(SimpleFunctionApply(
        task_name=f"Apply a LPF of {size} to bennu Mesh displacement maps",
        run_after_task_names=(get_displacement_lods.task_name,),
        input_markers=(FSMarkerString("ProjectModel_DIS"),),
        output_markers=(FSMarkerString("ProjectModel_DIS_LPF"),),
        read_adapter=FSNumpyAdapter(),
        write_adapter=FSNumpyAdapter(
            export_debug_plots=True,
            title=f"DIS LPF export with size {size}",
            colour_bar_title=f"DIS factor after LPF of size {size}"
        ),
        function_to_apply=lambda img: blur_by_fraction(img, fraction=size),
        import_folder=FSPathLocalDisk(
            path=(f"Bennu mesh LQ OBJ to DIS",),
            markers=tuple(),
            root_path=detections_from_bennu_model.as_posix()),
        export_folder=FSPathLocalDisk(
            path=(f"Bennu mesh LQ OBJ to DIS LPF size {size}",),
            markers=tuple(),
            root_path=detections_from_bennu_model.as_posix()),
        n_jobs=-1
    ))

    apply_blur_to_displacement_lods_tasks.append(SimpleFunctionApply(
        task_name=f"Apply a HPF of {size} to bennu Mesh displacement maps",
        run_after_task_names=(get_displacement_lods.task_name,),
        input_markers=(FSMarkerString("ProjectModel_DIS"),),
        output_markers=(FSMarkerString("ProjectModel_DIS_HPF"),),
        read_adapter=FSNumpyAdapter(),
        write_adapter=FSNumpyAdapter(
            export_debug_plots=True,
            title=f"DIS HPF export with size {size}",
            colour_bar_title=f"DIS factor after HPF of size {size}"
        ),
        function_to_apply=lambda img: img -
        blur_by_fraction(img, fraction=size),
        import_folder=FSPathLocalDisk(
            path=(f"Bennu mesh LQ OBJ to DIS",),
            markers=tuple(),
            root_path=detections_from_bennu_model.as_posix()),
        export_folder=FSPathLocalDisk(
            path=(f"Bennu mesh LQ OBJ to DIS HPF size {size}",),
            markers=tuple(),
            root_path=detections_from_bennu_model.as_posix()),
        n_jobs=-1
    ))


selected_size: float = 1 / 32

# apply_step_A_B_C = SimpleFunctionApply(
#     task_name=f"Apply a ABC of {selected_size} to bennu Mesh displacement maps",
#     run_after_task_names=(get_displacement_lods.task_name,),
#     input_markers=(FSMarkerString("ProjectModel_DIS"),),
#     output_markers=(FSMarkerString("ProjectModel_DIS_ABC"),),
#     read_adapter=FSNumpyAdapter(),
#     write_adapter=FSPNGAdapter(
#         # export_debug_plots=True,
#         # title=f"DIS ABC export with size {selected_size}",
#         # colour_bar_title=f"DIS factor after ABC of size {selected_size}"
#     ),
#     function_to_apply=lambda img: blur_by_fraction(img -
#                                                    blur_by_fraction(img, fraction=selected_size), fraction=selected_size / 2),
#     import_folder=FSPathLocalDisk(
#         path=(f"Bennu mesh LQ OBJ to DIS",),
#         markers=tuple(),
#         root_path=detections_from_bennu_model.as_posix()),
#     export_folder=FSPathLocalDisk(
#         path=(f"Bennu mesh LQ OBJ to DIS ABC size {selected_size}",),
#         markers=tuple(),
#         root_path=detections_from_bennu_model.as_posix()),
#     n_jobs=4
# )


def detect_boulders(height_map: np.ndarray,
                    min_radius_fraction=0.01,
                    max_radius_fraction=0.08,
                    num_sigma=20,
                    threshold=0.005,
                    detrend_sigma_fraction=0.25,
                    circle_scale=1.8,
                    line_width=3.0,
                    min_radius_absolute: float | None = None) -> np.ndarray:
    """
    Detects boulders in a 2D grayscale height map using Determinant of Hessian (DoH).
    All scale parameters (min/max sigma, detrend sigma) are automatically calculated
    based on the input image size for better out-of-the-box results on any resolution.

    Returns an RGBA NumPy array (H, W, 4) with terrain colormap + red boulder annotations.

    Parameters:
    - height_map (np.ndarray): 2D grayscale array (higher values = higher elevation)
    - min_radius_fraction / max_radius_fraction: boulder radius range as fraction of image width
    - num_sigma: number of scales to test
    - threshold: DoH sensitivity
    - min_radius_absolute: optional hard minimum radius in pixels (overrides fraction)
    - detrend_sigma_fraction: if not None, sigma for background removal as fraction of width
    - circle_scale, line_width: visual appearance of annotations
    """
    if height_map.ndim != 2:
        raise ValueError("height_map must be a 2D numpy array (grayscale).")

    h, w = height_map.shape
    reference_size = w  # use width as reference (common for landscapes)

    # Auto-calculate radii in pixels
    # at least 3 pixels
    min_sigma = max(3, int(min_radius_fraction * reference_size))
    max_sigma = max(min_sigma + 5, int(max_radius_fraction * reference_size))

    # Optional absolute minimum override
    if min_radius_absolute is not None:
        min_sigma = max(min_sigma, int(min_radius_absolute))

    # Auto detrending sigma
    detrend_sigma = None
    if detrend_sigma_fraction is not None:
        detrend_sigma = detrend_sigma_fraction * reference_size

    print(f"Image size: {w}x{h}")
    print(f"Auto parameters -> min_sigma={min_sigma}, max_sigma={max_sigma}, "
          f"detrend_sigma={detrend_sigma if detrend_sigma else 'None'}")

    # Optional detrending
    processed_map = height_map.astype(float)
    if detrend_sigma is not None:
        background = gaussian_filter(height_map, sigma=detrend_sigma)
        processed_map = height_map - background

    # Normalize for blob detection
    p_min, p_max = processed_map.min(), processed_map.max()
    if p_max > p_min:
        normalized_map = (processed_map - p_min) / (p_max - p_min)
    else:
        normalized_map = processed_map.copy()

    # Detect blobs
    blobs = blob_doh(normalized_map,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     num_sigma=num_sigma,
                     threshold=threshold)

    print(f"Detected {len(blobs)} potential boulders before filtering.")

    # Apply minimum radius filter (using the same min_sigma as detection lower
    # bound)
    final_min_radius = min_radius_absolute if min_radius_absolute is not None else min_sigma
    if final_min_radius > 0:
        blobs = blobs[blobs[:, 2] >= final_min_radius]

    print(f"Final: {len(blobs)} boulders after filtering.")

    # Apply terrain colormap
    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=height_map.min(), vmax=height_map.max())
    terrain_rgb = cm.get_cmap('terrain')(
        norm(height_map))[..., :3]  # (H, W, 3) float [0-1]

    # Create RGBA canvas
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = (terrain_rgb * 255).astype(np.uint8)
    rgba[..., 3] = 255

    # Draw annotations directly with NumPy
    red = np.array([255, 0, 0, 255], dtype=np.uint8)
    white = np.array([255, 255, 255, 255], dtype=np.uint8)

    y_grid, x_grid = np.ogrid[:h, :w]

    for blob in blobs:
        cy, cx, r = blob
        radius = r * circle_scale

        # Circle (approximate anti-aliased ring)
        dist_sq = (y_grid - cy)**2 + (x_grid - cx)**2
        inner = (radius - line_width / 2)**2
        outer = (radius + line_width / 2)**2
        circle_mask = (dist_sq >= inner) & (dist_sq <= outer)
        rgba[circle_mask] = red

        # Center marker (+)
        marker_size = max(3, int(radius / 3))
        # Vertical and horizontal lines with white outline
        rgba[int(cy) - marker_size:int(cy) + marker_size + 1, int(cx)] = white
        rgba[int(cy), int(cx) - marker_size:int(cx) + marker_size + 1] = white
        # Red center
        rgba[int(cy), int(cx)] = red

    return rgba


apply_step_RLA = SimpleFunctionApply(
    task_name=f"Apply a RLA to bennu Mesh displacement maps",
    run_after_task_names=(get_displacement_lods.task_name,),
    input_markers=(FSMarkerString("ProjectModel_DIS"),),
    output_markers=(FSMarkerString("ProjectModel_DIS_RLA"),),
    read_adapter=FSNumpyAdapter(),
    write_adapter=FSNumpyAdapter(
        export_debug_plots=True,
        title=f"DIS RLA export",
        colour_bar_title=f"DIS factor after RLA"
    ),
    function_to_apply=lambda img: detect_boulders(
        blur_by_fraction(img -
                         blur_by_fraction(img, fraction=selected_size), fraction=selected_size / 2)),
    import_folder=FSPathLocalDisk(
        path=(f"Bennu mesh LQ OBJ to DIS",),
        markers=tuple(),
        root_path=detections_from_bennu_model.as_posix()),
    export_folder=FSPathLocalDisk(
        path=(f"Bennu mesh LQ OBJ to DIS RLA",),
        markers=tuple(),
        root_path=detections_from_bennu_model.as_posix()),
    n_jobs=4
)

steps = [get_bennu_obj, get_local_area_scaling_lods, get_displacement_lods,
         get_bennu_obj_hq,
         *apply_blur_to_displacement_lods_tasks,
         #  apply_step_A_B_C,
         apply_step_RLA,
         ]

if __name__ == "__main__":
    cache: ResultCache[FSEnvironment] = ResultCache[FSEnvironment](
        cache_folder=Path(".cache"), result_type=FSEnvironment)

    futures: dict[str, FSEnvironment] = StepsOrchestrator.run_steps(
        steps, cache)
