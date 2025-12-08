from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from joblib import delayed
from tqdm_joblib import ParallelPbar

from boulder_statistics.environment_tools.fs_environment import FSEnvironment
from boulder_statistics.environment_tools.fs_markers.fs_marker_string import \
    FSMarkerString
from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk
from boulder_statistics.file_storage_adapters.pandas_pickle_adapter import \
    FSPandasPickleAdapter
from boulder_statistics.file_storage_adapters.plt_plot_adapter import \
    FSPltPlotAdapter
from boulder_statistics.steps.detection_merge import HEADERS
from boulder_statistics.steps.utils.duplicate_detection import \
    Duplicate_Detection
from boulder_statistics.task_step_base import TaskStepBase

FIGSIZE = (7, 7 * ((5**0.5 - 1) / 2))
DPI = 800

matplotlib.use("Agg")
plt.style.use('science')
plt.rcParams["figure.figsize"] = FIGSIZE
plt.rcParams["figure.dpi"] = DPI
plt.ioff()


@dataclass(frozen=True)
class PlotStandardDetectionResults(TaskStepBase):
    marker_to_plot: FSMarkerString
    output_marker: FSMarkerString
    export_folder: str
    result_output_folder: str
    version_index: int

    @property
    def hashable(self) -> tuple[Any, ...]:
        return (self.marker_to_plot, self.output_marker, self.export_folder,
                self.result_output_folder, self.version_index)

    def run(self, env: FSEnvironment) -> FSEnvironment:

        files_to_plot: List[FSPathLocalDisk] = env.get_paths(
            FSPathLocalDisk, lambda x: self.marker_to_plot in x.markers)

        # processed_inference_result_paths = ParallelPbar(
        #     f"Processing plots for {len(files_to_plot)} detection results", unit="detection results")(n_jobs=-1)(
        #     delayed(PlotStandardDetectionResults.plot_detection_results)
        #     (path, FSPathLocalDisk(path=Path(self.result_output_folder).parts, markers=frozenset(
        #         [self.output_marker]), root_path=self.export_folder))
        #     for path in files_to_plot
        # )

        processed_inference_result_paths: List[List[FSPathLocalDisk]] = [
            PlotStandardDetectionResults.plot_detection_results(
                path,
                FSPathLocalDisk(
                    path=Path(self.result_output_folder).parts,
                    markers=(self.output_marker,),
                    root_path=self.export_folder,
                ),
            )
            for path in files_to_plot
        ]

        return FSEnvironment(
            tuple(path for paths in processed_inference_result_paths for path in paths))

    @staticmethod
    def plot_detection_results(results_data_path: FSPathLocalDisk,
                               results_folder: FSPathLocalDisk) -> List[FSPathLocalDisk]:

        results: pd.DataFrame = FSEnvironment.load(
            results_data_path, FSPandasPickleAdapter())

        plots_to_make: List[Callable[[pd.DataFrame, FSPathLocalDisk]]] = [
            PlotStandardDetectionResults.plot_sfd,
            PlotStandardDetectionResults.plot_sfd_pixel_size,
            PlotStandardDetectionResults.plot_sfd_remove_duplicates,
        ]

        return [plot_to_make(results, results_folder)
                for plot_to_make in plots_to_make]

    @staticmethod
    def plot_sfd(df: pd.DataFrame,
                 results_folder: FSPathLocalDisk) -> FSPathLocalDisk:

        save_folder: FSPathLocalDisk = results_folder.copy_from_folder(
            Path("SFD_Raw_Counts.png"))

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

        ax.hist(df["fixed_weight_area"], bins=30)
        ax.set_xlabel("Size (fixed weight area)")
        ax.set_ylabel("Frequency")
        ax.set_title("Detection SFD (Raw counts)")
        ax.set_yscale("log")
        fig.tight_layout()

        FSEnvironment.save(fig, save_folder, FSPltPlotAdapter(dpi=DPI))
        plt.close(fig)

        return save_folder

    @staticmethod
    def plot_sfd_pixel_size(df: pd.DataFrame,
                            results_folder: FSPathLocalDisk) -> FSPathLocalDisk:

        save_folder: FSPathLocalDisk = results_folder.copy_from_folder(
            Path("SFD_Raw_Counts_pixel_size.png"))

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

        ax.hist(df["fixed_weight_area"] / df["relative_scale"], bins=10)
        ax.set_xlabel("Pixel size (fixed weight area / relative scale)")
        ax.set_ylabel("Frequency")
        ax.set_title("Detection SFD (Raw counts pixel size)")
        fig.tight_layout()

        FSEnvironment.save(fig, save_folder, FSPltPlotAdapter(dpi=DPI))
        plt.close(fig)

        return save_folder

    @staticmethod
    def plot_sfd_remove_duplicates(df: pd.DataFrame,
                                   results_folder: FSPathLocalDisk) -> FSPathLocalDisk:

        save_folder: FSPathLocalDisk = results_folder.copy_from_folder(
            Path("SFD_Raw_Counts_duplicates_removed.png"))

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

        duplicate_removed_df: pd.DataFrame = Duplicate_Detection.remove_near_duplicate_detections(
            df)

        ax.hist(duplicate_removed_df["fixed_weight_area"], bins=10)
        ax.set_xlabel("Size (fixed weight area)")
        ax.set_ylabel("Frequency")
        ax.set_title("Detection SFD (Raw counts duplicates removed)")
        ax.set_yscale("log")
        fig.tight_layout()

        FSEnvironment.save(fig, save_folder, FSPltPlotAdapter(dpi=DPI))
        plt.close(fig)

        return save_folder
