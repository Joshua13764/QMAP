import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from cv2.gapi import mask
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

TRANSFORMS = [
    lambda x: x,  # Identity
    lambda x: np.rot90(x, k=1),  # 90 clockwise
    lambda x: np.rot90(x, k=2),  # 180 clockwise
    lambda x: np.rot90(x, k=3),  # 270 clockwise
    lambda x: np.flip(x, axis=0),  # Mirror across horizontal axis
    lambda x: np.flip(x, axis=1),  # Mirror across vertical axis
    lambda x: x.swapaxes(0, 1),                 # fixed diagonal transpose
    lambda x: np.rot90(x, k=2).swapaxes(0, 1),  # fixed version of last one
]

TRANSFORMS_INV = [
    lambda x: x,  # Identity inverse
    lambda x: np.rot90(x, k=-1),  # 90 clockwise inverse
    lambda x: np.rot90(x, k=-2),  # 180 clockwise inverse
    lambda x: np.rot90(x, k=-3),  # 270 clockwise inverse
    lambda x: np.flip(x, axis=0),  # Mirror across horizontal axis inverse
    lambda x: np.flip(x, axis=1),  # Mirror across vertical axis inverse
    lambda x: x.swapaxes(0, 1),
    lambda x: np.rot90(x.swapaxes(0, 1), k=2),
]


def export_inference_data(
        boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
        masks: np.ndarray, save_path: Path):

    # boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)  # (N,4)
    # scores = instances.scores.numpy().astype(np.float32)            # (N,)
    # classes = instances.pred_classes.numpy().astype(np.int64)       # (N,)
    # masks = instances.pred_masks.numpy().astype(np.uint8)           #
    # (N,H,W) 0/1

    np.savez_compressed(
        save_path,
        boxes_xyxy=boxes,
        scores=scores,
        class_ids=classes,
        masks_uint8=masks,
    )
    print(f"Exported infer data to {save_path}")


def build_predictor(cfg_path: str, weights_path: str,
                    score_thresh: float = 0.5) -> DefaultPredictor:
    cfg = get_cfg()
    # allow custom BoulderNet keys (e.g., INPUT.MIN_AREA_NPIXELS)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path

    # --- CUDA change (was "cpu") ---
    cfg.MODEL.DEVICE = "cuda"

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)


def load_image_bgr(in_path: Path):

    print(f"Loading file {in_path.as_posix()}")
    match in_path.suffix.lower():
        case ".png" | ".jpg" | ".jpeg" | ".tif" | ".tiff" | ".bmp":
            bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
            return bgr
        case ".npy":
            gray = np.load(in_path.as_posix())
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return bgr
        case _:
            raise ValueError(
                f"""Unsupported image format: {
                    in_path.suffix.lower()}""")


def infer_image(in_path: Path, overlay_export_path: Path,
                inference_export_path: Path, predictor):

    bgr: np.ndarray = load_image_bgr(in_path)

    transform_boxes: List[np.ndarray] = []
    transform_scores: List[np.ndarray] = []
    transform_classes: List[np.ndarray] = []
    transform_masks: List[np.ndarray] = []

    for transform, transform_inv in zip(TRANSFORMS, TRANSFORMS_INV):

        outputs = predictor(
            transform(bgr.astype(np.uint8))
        )

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)  # (N,4)
        scores = instances.scores.numpy().astype(np.float32)            # (N,)
        classes = instances.pred_classes.numpy().astype(np.int64)       # (N,)
        masks = instances.pred_masks.numpy().astype(np.uint8)           # (N,H,W)

        masks_actual = np.array([transform_inv(mask) for mask in masks])

        transform_boxes.append(boxes)
        transform_scores.append(scores)
        transform_classes.append(classes)
        transform_masks.append(masks_actual)

    total_boxes = np.concatenate(transform_boxes)
    total_scores = np.concatenate(transform_scores)
    total_classes = np.concatenate(transform_classes)
    total_masks = np.concatenate(transform_masks)

    export_inference_data(
        total_boxes,
        total_scores,
        total_classes,
        total_masks,
        inference_export_path)

    print(f"[Result] detections: {len(total_masks)}")


def main() -> None:
    # paths from env (set in Dockerfile) with sensible defaults
    cfg_path: str = os.environ.get(
        "BOULDERNET_CONFIG",
        "/models/bouldernet/config.yaml")
    weights_path: str = os.environ.get(
        "BOULDERNET_WEIGHTS",
        "/models/bouldernet/model_0055999.pth")
    out_dir: Path = Path(os.environ.get("OUT_DIR", "/workspace/out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    detection_export_custom_name_tag: str = os.environ.get(
        "detection_export_custom_name_tag", "")

    # # require an input image path
    # if len(sys.argv) < 2:
    #     print("Usage: python bouldernet_infer_overlay.py /path/to/image.png")
    #     sys.exit(2)

    print("LIST /in:", os.listdir("/in"))

    predictor = build_predictor(cfg_path, weights_path, score_thresh=0.2)
    in_paths: list[Path] = [Path(p) for p in sys.argv[1:]]

    for in_path in in_paths:
        out_base_path: Path = out_dir / in_path.name
        overlay_export_path: Path = out_base_path.with_name(
            f"{out_base_path.stem}{detection_export_custom_name_tag}_overlay.png")
        inference_export_path: Path = out_base_path.with_name(
            f"{out_base_path.stem}{detection_export_custom_name_tag}_detections.npz")

        if not os.path.exists(inference_export_path):
            infer_image(
                in_path,
                overlay_export_path,
                inference_export_path,
                predictor)

    time.sleep(2)


if __name__ == "__main__":
    main()
