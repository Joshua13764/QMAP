import os
import sys
from pathlib import Path

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


def render_overlay(bgr, instances, src_path: Path):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    meta_name = "bouldernet_demo"
    MetadataCatalog.get(meta_name).set(
        thing_classes=["boulder"])  # single-class label
    vis = Visualizer(rgb, metadata=MetadataCatalog.get(
        meta_name), instance_mode=ColorMode.IMAGE)

    rgb_overlay = vis.draw_instance_predictions(instances).get_image()
    bgr_overlay = cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR)

    save_path: Path = src_path.with_name(f"{src_path.stem}_overlay.png")
    cv2.imwrite(save_path.as_posix(), bgr_overlay)

    print(f"Exported overlay to {save_path}")


def export_inference_data(instances, src_path: Path):
    boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)        # (N,4)
    scores = instances.scores.numpy().astype(np.float32)                 # (N,)
    classes = instances.pred_classes.numpy().astype(np.int64)            # (N,)
    masks = instances.pred_masks.numpy().astype(
        np.uint8)                # (N,H,W) 0/1

    npz_path: Path = src_path.with_name(f"{src_path.stem}_detections.npz")
    print(f"Exported infer data to {npz_path}")
    np.savez_compressed(
        npz_path,
        boxes_xyxy=boxes,
        scores=scores,
        class_ids=classes,
        masks_uint8=masks,
    )


def build_predictor(cfg_path: str, weights_path: str,
                    score_thresh: float = 0.5) -> DefaultPredictor:
    cfg = get_cfg()
    # allow custom BoulderNet keys (e.g., INPUT.MIN_AREA_NPIXELS)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)


def infer_image(in_path: Path, out_dir: Path, predictor):

    bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV failed to read image: {in_path}")

    outputs = predictor(bgr)
    instances = outputs["instances"].to("cpu")
    print(f"[Result] detections: {len(instances)}")

    render_overlay(bgr, instances, out_dir / in_path.name)
    export_inference_data(instances, out_dir / in_path.name)


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

    # require an input image path
    if len(sys.argv) < 2:
        print("Usage: python bouldernet_infer_overlay.py /path/to/image.png")
        sys.exit(2)

    predictor = build_predictor(cfg_path, weights_path)
    in_paths: list[Path] = [Path(p) for p in sys.argv[1:]]

    for in_path in in_paths:
        infer_image(in_path, out_dir, predictor)


if __name__ == "__main__":
    main()
