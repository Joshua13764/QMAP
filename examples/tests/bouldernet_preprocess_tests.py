from pathlib import Path
from typing import Any, Callable, Coroutine, List, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ImageFile import ImageFile

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.file_storage_adapters.numpy_adapter import \
    FSNumpyAdapter
from boulder_statistics.file_storage_adapters.pillow_image_adapter import \
    FSPillowImageAdapter
from boulder_statistics.file_storage_adapters.png_adapter import FSPNGAdapter
from boulder_statistics.result_cache import ResultCache
from boulder_statistics.step_base import StepBase
from boulder_statistics.steps.Best_model_downloader import BestModelDownloader
from boulder_statistics.steps.detection_merge import DetectionMerge
from boulder_statistics.steps.OBJ_to_LAS import OBJToLAS
from boulder_statistics.steps.PAN_to_LOD import PANToLOD
from boulder_statistics.steps.pds4_boulderNet_inference import \
    PDS4BoulderNetInference
from boulder_statistics.steps.PDS_downloader import PDSDownloader
from boulder_statistics.steps.PDS_to_PNG import PDS_to_PNG
from boulder_statistics.steps.plot_standard_detection_results import \
    PlotStandardDetectionResults
from boulder_statistics.steps.simple_function_import_export import \
    SimpleFunctionImportExport
from boulder_statistics.steps.simple_local_file import SimpleLocalFile
from boulder_statistics.steps.SPICE_kernels_downloader import \
    SPICEKernelGrabber
from boulder_statistics.steps_orchestrator import StepsOrchestrator

model_download_path: Path = Path(r"F:\AO33\AO33_models")
pds_download_path: Path = Path(r"F:\AO33\AO33_pds_DATA")
pipeline_working_path: Path = Path(r"F:\AO33\AO33_pipeline_DATA")
pipeline_working_path_fast: Path = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA\Pipeline_running_path_fast")
boulderNet_preprocessor_test_folder: Path = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA\BoulderNetPreProcessorTests"
)
spice_download_path: Path = Path(r"F:\AO33\AO33_SPICE_DATA")

# step11 = PlotStandardDetectionResults(
#     task_name=f"Plot standard detection results",
#     run_after_task_names=(step10.task_name),
#     marker_to_plot=FSMarkerString("Merged_BoulderNet_Detections"),
#     output_marker=FSMarkerString("Detection_Plots"),
#     export_folder=pipeline_working_path_fast.as_posix(),
#     result_output_folder=Path("exports/detection_plots").as_posix(),
#     version_index=3
# )

default_lod_extract = SimpleLocalFile(
    task_name="BoulderNet detect test",
    run_after_task_names=(),
    local_file_path=Path(
        r"C:\Users\Joshu\Documents\AO33_DATA\resources\tests\posx_512_1024_512x512_of_2048.png"),
    dst_root_path=boulderNet_preprocessor_test_folder,
    dst_sub_path=Path("lod 0/posx_512_1024_512x512_of_2048.png"),
    markers=(FSMarkerString("default_lod_export_example"))
)


brightness_to_try: dict[str, Callable[[Image.Image], Image.Image]] = {
    f"brightness_x{x}": lambda img: ImageEnhance.Brightness(img).enhance(x)
    for x in np.linspace(-0.5, 0.5, 100)
}

to_try: dict[str, Callable[[Image.Image], Image.Image]] = {

    # ── Brightness ─────────────────────────────────────
    "brightness_x1.3": lambda img: ImageEnhance.Brightness(img).enhance(1.3),
    "brightness_x1.5": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
    "brightness_x1.8": lambda img: ImageEnhance.Brightness(img).enhance(1.8),
    "brightness_x0.7": lambda img: ImageEnhance.Brightness(img).enhance(0.7),
    "brightness_x0.5": lambda img: ImageEnhance.Brightness(img).enhance(0.5),

    # ── Contrast ───────────────────────────────────────
    "contrast_x1.4": lambda img: ImageEnhance.Contrast(img).enhance(1.4),
    "contrast_x1.8": lambda img: ImageEnhance.Contrast(img).enhance(1.8),
    "contrast_x2.2": lambda img: ImageEnhance.Contrast(img).enhance(2.2),
    "contrast_x0.6": lambda img: ImageEnhance.Contrast(img).enhance(0.6),

    # ── Saturation (Color) ─────────────────────────────
    "saturation_x1.5": lambda img: ImageEnhance.Color(img).enhance(1.5),
    "saturation_x2.0": lambda img: ImageEnhance.Color(img).enhance(2.0),
    "saturation_x0.0_grayscale": lambda img: ImageEnhance.Color(img).enhance(0.0),
    "saturation_x0.3": lambda img: ImageEnhance.Color(img).enhance(0.3),

    # ── Sharpness ──────────────────────────────────────
    "sharpness_x2.0": lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
    "sharpness_x3.0_strong": lambda img: ImageEnhance.Sharpness(img).enhance(3.0),
    "sharpness_x0.5_soft": lambda img: ImageEnhance.Sharpness(img).enhance(0.5),

    # ── Gaussian Blur ──────────────────────────────────
    "gaussian_blur_r=2": lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
    "gaussian_blur_r=4": lambda img: img.filter(ImageFilter.GaussianBlur(radius=4)),
    "gaussian_blur_r=6_heavy": lambda img: img.filter(ImageFilter.GaussianBlur(radius=6)),
    "gaussian_blur_r=1_light": lambda img: img.filter(ImageFilter.GaussianBlur(radius=1)),

    # ── Unsharp Mask (professional sharpening) ─────────
    "unsharp_r2_p150_t3": lambda img: img.filter(
        ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)),
    "unsharp_r1.5_p200_t5_aggressive": lambda img: img.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=200, threshold=5)),
    "unsharp_r3_p100_t2_soft": lambda img: img.filter(
        ImageFilter.UnsharpMask(radius=3, percent=100, threshold=2)),

    # ── Edge Enhancement ──────────────────────────────
    "edge_enhance": lambda img: img.filter(ImageFilter.EDGE_ENHANCE),
    "edge_enhance_more": lambda img: img.filter(ImageFilter.EDGE_ENHANCE_MORE),
    "find_edges_sobel": lambda img: img.filter(ImageFilter.FIND_EDGES),

    # ── Artistic / Diagnostic Filters ──────────────────
    "emboss": lambda img: img.filter(ImageFilter.EMBOSS),
    "contour": lambda img: img.filter(ImageFilter.CONTOUR),
    "detail_enhance": lambda img: img.filter(ImageFilter.DETAIL),

    # ── Simple Built-in Filters ────────────────────────
    "sharpen_basic": lambda img: img.filter(ImageFilter.SHARPEN),
    "blur_box": lambda img: img.filter(ImageFilter.BLUR),
    "smooth": lambda img: img.filter(ImageFilter.SMOOTH),
    "smooth_more": lambda img: img.filter(ImageFilter.SMOOTH_MORE),

    # ── Noise Reduction ───────────────────────────────
    "median_filter_3x3": lambda img: img.filter(ImageFilter.MedianFilter(size=3)),
    "median_filter_5x5": lambda img: img.filter(ImageFilter.MedianFilter(size=5)),

    # ── Morphological ─────────────────────────────────
    "min_filter_3x3_erode": lambda img: img.filter(ImageFilter.MinFilter(size=3)),
    "max_filter_3x3_dilate": lambda img: img.filter(ImageFilter.MaxFilter(size=3)),

    # ── Identity (control) ─────────────────────────────
    "identity": lambda img: img,  # no change
}

to_try_tasks = [
    SimpleFunctionImportExport[Image.Image](
        run_after_task_names=StepBase.get_task_names(default_lod_extract),
        task_name=func_name,
        output_name_suffix=func_name,
        adapter=FSPillowImageAdapter(),
        input_markers=(FSMarkerString("default_lod_export_example"),),
        output_markers=(FSMarkerString("BoulderNetDetects"),),
        function_to_apply=func,
    )
    for func_name, func in to_try.items()
]


infer_default_lod_extract = PDS4BoulderNetInference(
    task_name=f"Infer default lod_extract",
    cuda=True,
    skip_converted=True,
    run_after_task_names=StepBase.get_task_names(
        default_lod_extract, *to_try_tasks),
    run_path=boulderNet_preprocessor_test_folder.as_posix(),
    detection_input_markers=(
        FSMarkerString("default_lod_export_example"),
        FSMarkerString("BoulderNetDetects")),
    detection_output_markers=(FSMarkerString("BoulderNet_Detections"),),
)

merge = DetectionMerge(
    task_name=f"Merge detections",
    run_after_task_names=(infer_default_lod_extract.task_name,),
    marker_to_merge=FSMarkerString("BoulderNet_Detections"),
    output_marker=FSMarkerString("Merged_BoulderNet_Detections"),
    run_path=boulderNet_preprocessor_test_folder.as_posix(),
    result_output_path=Path("exports/merge_detections.pkl").as_posix()
)

step11 = PlotStandardDetectionResults(
    task_name=f"Plot standard detection results",
    run_after_task_names=(merge.task_name,),
    marker_to_plot=FSMarkerString("Merged_BoulderNet_Detections"),
    output_marker=FSMarkerString("Detection_Plots"),
    export_folder=pipeline_working_path_fast.as_posix(),
    result_output_folder=Path("exports/detection_plots").as_posix(),
    version_index=3
)

pool = [default_lod_extract, *to_try_tasks, infer_default_lod_extract, merge]

cache: ResultCache[StepBase, FSEnvironment] = ResultCache[StepBase, FSEnvironment](
    cache_folder=Path("cache"), result_type=FSEnvironment)

futures: dict[str, FSEnvironment] = StepsOrchestrator.run_tasks_with_dependencies([
                                                                                  merge], pool, cache)
