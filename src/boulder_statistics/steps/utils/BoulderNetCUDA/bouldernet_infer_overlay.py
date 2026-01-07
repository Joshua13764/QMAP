import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


def ensure_gpu_headroom(threshold: float = 0.95):
    """
    Abort if GPU memory utilisation is above `threshold` (e.g. 0.95 = 95%).

    Uses CUDA driver info, so it accounts for all processes, not just PyTorch.
    """
    if not torch.cuda.is_available():
        return  # nothing to check on CPU-only runs

    # Optionally clear PyTorch's cached blocks first, so the reading is
    # realistic
    torch.cuda.empty_cache()

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    util = used_bytes / total_bytes

    if util >= threshold:
        used_gib = used_bytes / (1024 ** 3)
        total_gib = total_bytes / (1024 ** 3)
        msg = (
            f"GPU memory utilisation too high: {util * 100:.1f}% "
            f"(used {used_gib:.2f} GiB / {total_gib:.2f} GiB). "
            f"""Threshold is {threshold *
                              100:.1f}%. Aborting to avoid instability."""
        )
        # You can either raise or exit; raising is nicer for Docker logs.
        raise RuntimeError(msg)


def render_overlay(bgr, instances, save_path: Path):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    meta_name = "bouldernet_demo"
    MetadataCatalog.get(meta_name).set(
        thing_classes=["boulder"])  # single-class label
    vis = Visualizer(
        rgb,
        metadata=MetadataCatalog.get(meta_name),
        instance_mode=ColorMode.IMAGE,
    )

    rgb_overlay = vis.draw_instance_predictions(instances).get_image()
    bgr_overlay = cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path.as_posix(), bgr_overlay)

    print(f"Exported overlay to {save_path}")


def export_inference_data(instances, save_path: Path):
    boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)  # (N,4)
    scores = instances.scores.numpy().astype(np.float32)            # (N,)
    classes = instances.pred_classes.numpy().astype(np.int64)       # (N,)
    masks = instances.pred_masks.numpy().astype(np.uint8)           # (N,H,W) 0/1

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

    bgr = load_image_bgr(in_path)

    ensure_gpu_headroom()

    outputs = predictor(bgr)

    # Move results back to CPU so .numpy() calls work
    instances = outputs["instances"].to("cpu")
    print(f"[Result] detections: {len(instances)}")

    render_overlay(bgr, instances, overlay_export_path)
    export_inference_data(instances, inference_export_path)


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

    # require an input image path
    if len(sys.argv) < 2:
        print("Usage: python bouldernet_infer_overlay.py /path/to/image.png")
        sys.exit(2)

    predictor = build_predictor(cfg_path, weights_path)
    in_paths: list[Path] = [Path(p) for p in sys.argv[1:]]

    for in_path in in_paths:
        src_path: Path = out_dir / in_path.name
        overlay_export_path: Path = src_path.with_name(
            f"{src_path.stem}{detection_export_custom_name_tag}_overlay.png")
        inference_export_path: Path = src_path.with_name(
            f"{src_path.stem}{detection_export_custom_name_tag}_detections.npz")

        if (not os.path.exists(overlay_export_path) or
                not os.path.exists(inference_export_path)):
            infer_image(
                in_path,
                overlay_export_path,
                inference_export_path,
                predictor)

    time.sleep(2)


if __name__ == "__main__":
    main()
