# examples/test_mltools_inference.py
import os
import sys

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# Importing MLtools here ensures the package import path is correct
from MLtools import inference  # noqa: F401  (import check)


def build_predictor(config_path: str, weights_path: str):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"       # force CPU
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg)


def load_image(arg_path: str | None):
    if arg_path and os.path.exists(arg_path):
        img = cv2.imread(arg_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"OpenCV could not read: {arg_path}")
        return img
    # fallback: synthetic image to exercise the pipeline
    return np.zeros((256, 256, 3), dtype=np.uint8)


def main():
    config = os.environ.get(
        "BOULDERNET_CONFIG",
        "/models/bouldernet/config.yaml")
    weights = os.environ.get(
        "BOULDERNET_WEIGHTS",
        "/models/bouldernet/model_0055999.pth")
    img_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("[Paths]")
    print("  config :", config)
    print("  weights:", weights)
    print("  img    :", img_path or "(synthetic)")

    predictor = build_predictor(config, weights)
    img = load_image(img_path)
    outputs = predictor(img)

    # Summarize results (instances: boxes, scores, classes)
    inst = outputs["instances"].to("cpu")
    n = len(inst) if hasattr(inst, "__len__") else 0
    print(f"[Result] detections: {n}")
    if n:
        print("  boxes :", inst.pred_boxes.tensor.numpy()[:5])
        print("  scores:", inst.scores.numpy()[:5])
        print("  classes:", inst.pred_classes.numpy()[:5])


if __name__ == "__main__":
    main()
