# examples/minimal_demo.py
# Runs entirely on CPU. No model weights needed.

import os
import platform
from pathlib import Path

import numpy as np
from PIL import Image

# prove MLtools and detectron2 import
from MLtools import inference  # noqa: F401
import torch, torchvision, detectron2

def main() -> None:
    print("[Python]", platform.python_version())
    print("[Torch]", torch.__version__)
    print("[Torchvision]", torchvision.__version__)
    import PIL as _PIL
    print("[Pillow]", _PIL.__version__)
    print("[Detectron2]", getattr(detectron2, "__version__", "unknown"))
    print("[MLtools.inference]", inference.__file__)

    # verify DefaultPredictor importable
    from detectron2.engine import DefaultPredictor  # noqa: F401
    print("[Detectron2] DefaultPredictor import OK")

    # tiny op: synthetic image, resize using Detectron2 ResizeTransform
    from detectron2.data.transforms import ResizeTransform

    H, W = 64, 96
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)  # horizontal grad R
    img[..., 1] = np.linspace(255, 0, H, dtype=np.uint8)[:, None]  # vertical grad G

    # compute target short edge and max size like ResizeShortestEdge would
    short_edge = 32
    max_size = 64
    min_orig = min(H, W)
    max_orig = max(H, W)
    scale = float(short_edge) / float(min_orig)
    # enforce max_size
    if round(max_orig * scale) > max_size:
        scale = float(max_size) / float(max_orig)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    # create concrete ResizeTransform (has apply_image)
    t = ResizeTransform(H, W, new_h, new_w, interp=Image.BILINEAR)
    resized = t.apply_image(img)

    print("Input shape:", tuple(img.shape), "→ Output shape:", tuple(resized.shape))

    out_dir = Path(os.environ.get("OUT_DIR", "./out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_dir / "input.png")
    Image.fromarray(resized).save(out_dir / "resized.png")
    print("Saved:", out_dir / "input.png")
    print("Saved:", out_dir / "resized.png")


if __name__ == "__main__":
    main()
