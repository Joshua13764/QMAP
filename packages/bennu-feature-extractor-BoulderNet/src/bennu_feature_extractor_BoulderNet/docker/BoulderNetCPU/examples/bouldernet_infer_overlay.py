# examples/bouldernet_infer_overlay.py
# CPU-only overlay demo for BoulderNet + Detectron2

import os
import sys
from pathlib import Path

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


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


def main() -> None:
    # paths from env (set in Dockerfile) with sensible defaults
    cfg_path = os.environ.get(
        "BOULDERNET_CONFIG",
        "/models/bouldernet/config.yaml")
    weights_path = os.environ.get(
        "BOULDERNET_WEIGHTS",
        "/models/bouldernet/model_0055999.pth")
    out_dir = Path(os.environ.get("OUT_DIR", "/workspace/out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # require an input image path
    if len(sys.argv) < 2:
        print("Usage: python bouldernet_infer_overlay.py /path/to/image.png")
        sys.exit(2)
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # load image (BGR), run predictor
    bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV failed to read image: {in_path}")

    predictor = build_predictor(cfg_path, weights_path)
    outputs = predictor(bgr)
    instances = outputs["instances"].to("cpu")
    print(f"[Result] detections: {len(instances)}")

    # visualise (Visualizer expects RGB)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    meta_name = "bouldernet_demo"
    MetadataCatalog.get(meta_name).set(
        thing_classes=["boulder"])  # single-class label
    vis = Visualizer(rgb, metadata=MetadataCatalog.get(
        meta_name), instance_mode=ColorMode.IMAGE)
    rgb_overlay = vis.draw_instance_predictions(instances).get_image()
    bgr_overlay = cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR)

    # save
    out_png = out_dir / f"{in_path.stem}_overlay.png"
    cv2.imwrite(str(out_png), bgr_overlay)
    print("Saved overlay:", out_png)


if __name__ == "__main__":
    main()
