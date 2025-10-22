# docker_helpers.py
# Minimal, CPU-only Docker SDK helpers for Python 3.13
# pip install "docker>=7,<8"

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union
import docker


def build_image(
    tag: str = "mltools:py3.10",
    context_dir: Union[str, Path] = ".",
    dockerfile: str = "Dockerfile",
    *,
    no_cache: bool = False,
    pull: bool = False,
    build_args: Optional[Dict[str, str]] = None,
) -> str:
    """Build Docker image and return its ID."""
    client = docker.from_env()
    image, logs = client.images.build(
        path=str(Path(context_dir)),
        dockerfile=dockerfile,
        tag=tag,
        rm=True,
        pull=pull,
        nocache=no_cache,
        buildargs=build_args or {},
    )
    for chunk in logs:
        line = chunk.get("stream") or chunk.get("status") or chunk.get("message")
        if line:
            print(line, end="")
    print(f"\nBuilt image: {image.id} (tags={image.tags})")
    return image.id


def run_script(
    image: str,
    script_path: Union[str, Path],
    *,
    mount_into: str = "/workspace",
    workdir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> Tuple[int, str]:
    """Run a local Python script inside the container."""
    host_script = Path(script_path).resolve(strict=True)
    host_dir = str(host_script.parent)
    container_workdir = workdir or mount_into
    container_script = f"{mount_into.rstrip('/')}/{host_script.name}"

    volumes = {host_dir: {"bind": mount_into, "mode": "rw"}}
    client = docker.from_env()

    container = client.containers.run(
        image=image,
        command=["python", container_script] + (list(extra_args) if extra_args else []),
        name=name,
        working_dir=container_workdir,
        volumes=volumes,
        environment=env or {},
        detach=True,
    )

    result = container.wait()
    exit_code = int(result.get("StatusCode", 1))
    logs = container.logs(stdout=True, stderr=True)
    output = logs.decode("utf-8", errors="replace")
    container.remove(force=True)
    return exit_code, output
