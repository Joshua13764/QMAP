from pathlib import Path

from bennu_feature_extractor_BoulderNet.docker.BoulderNetCPU.docker_helpers import \
    run_script

IMAGE_TAG = "mltools:py3.10"
overlay_script = Path(__file__).parent / "examples" / \
    "bouldernet_infer_overlay.py"

# Your host image (anywhere on disk)
host_img = Path(
    r"C:\Users\Joshu\Documents\AO33_DATA_Testing\A.jpg")
img_dir = host_img.parent
img_name = host_img.name

# results appear next to the script under examples/out
env = {"OUT_DIR": "/workspace/out"}

code, logs = run_script(
    IMAGE_TAG,
    str(overlay_script),
    env=env,
    extra_args=[f"/in/{img_name}"],                 # container path
    extra_mounts=[(img_dir, "/in", "ro")],          # bind host folder -> /in
)
print("Exit:", code)
print(logs)
print("Host outputs:", (overlay_script.parent / "out"))
