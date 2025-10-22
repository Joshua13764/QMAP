# minimal_demo_run_test.py
# CPU-only, no CLI — just call docker_helpers.run_script on the overlay
# example.

from pathlib import Path

from bennu_feature_extractor_BoulderNet.docker.BoulderNetCPU.docker_helpers import \
    run_script

IMAGE_TAG = "mltools:py3.10"

# Resolve the overlay script that lives in ./examples/
overlay_script = (Path(__file__).parent / "examples" /
                  "bouldernet_infer_overlay.py").resolve()
if not overlay_script.exists():
    raise FileNotFoundError(overlay_script)

# Test tile that the Dockerfile downloaded into the image at
# /data/bouldernet/...
test_tile_in_container = (
    "/data/bouldernet/Apr2023-Mars-Moon-Earth-mask-5px/"
    "preprocessing/test/images/stonegarden_2139_image.png"
)

# The helpers mount the script’s folder as /workspace inside the container.
# We’ll write outputs to /workspace/out so they appear on your host at:
# <this folder>/examples/out
env = {"OUT_DIR": "/workspace/out"}

exit_code, logs = run_script(
    IMAGE_TAG,
    str(overlay_script),
    env=env,
    extra_args=[test_tile_in_container],
)

print("Exit:", exit_code)
print(logs)

# Friendly hint about where to look for results on the host:
host_out = overlay_script.parent / "out"
print(
    f"\nOutputs (host): {host_out}  (PNG overlay + JSON detections expected)")
