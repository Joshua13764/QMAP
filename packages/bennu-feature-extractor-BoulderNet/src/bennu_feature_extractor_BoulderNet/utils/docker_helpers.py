from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import docker
from bennu_feature_extractor.environment import *

DOCKER_IMAGE_TAG = "mltools:py3.10"


class DockerHelpers():

    @staticmethod
    def analyse_image(image_path: FSPathLocalDisk,
                      inference_output_path: FSPathLocalDisk) -> None:
        """
        Ensure the Docker image exists (build it from BoulderNetCPU if missing), then run the overlay script.
        """
        # Locate overlay script and Dockerfile directory
        overlay_script: Path = Path(
            __file__).parent / "BoulderNetCPU" / "bouldernet_infer_overlay.py"
        overlay_dir: Path = overlay_script.parent
        dockerfile_rel = "Dockerfile"
        dockerfile_abs = overlay_dir / dockerfile_rel

        # --- ensure the image exists (build if needed from BoulderNetCPU) ---
        try:
            client: docker.DockerClient = docker.from_env()
            client.images.get(DOCKER_IMAGE_TAG)
        except docker.errors.ImageNotFound:
            if not dockerfile_abs.is_file():
                raise FileNotFoundError(
                    f"Expected Dockerfile at {dockerfile_abs} but it was not found."
                )
            print(
                f"Image '{DOCKER_IMAGE_TAG}' not found. Building from {overlay_dir} ...")
            DockerHelpers.build_image(
                tag=DOCKER_IMAGE_TAG,
                context_dir=overlay_dir,   # build context is BoulderNetCPU/
                dockerfile=dockerfile_rel,  # Dockerfile within that context
                pull=False,
                no_cache=False,
                build_args=None,
            )
        except docker.errors.APIError as e:
            raise RuntimeError(
                "Could not communicate with Docker. Is the daemon running?"
            ) from e

        # --- set up and run the analysis ---
        # results appear next to the script under examples/out
        env: Dict[str, str] = {
            "OUT_DIR": inference_output_path.actual_path.parent.as_posix()
        }

        code, logs = DockerHelpers.run_script(
            DOCKER_IMAGE_TAG,
            overlay_script.as_posix(),
            env=env,
            extra_args=[f"/in/{image_path.actual_path.name}"],
            # container path
            # bind host folder -> /in
            extra_mounts=[(image_path.actual_path.parent, "/in", "ro")],
        )

        print("Exit:", code)
        print(logs)
        print("Host outputs:", (overlay_dir / "out"))

    @staticmethod
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
        client: docker.DockerClient = docker.from_env()
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
            line = chunk.get("stream") or chunk.get(
                "status") or chunk.get("message")
            if line:
                print(line, end="")

        print(f"\nBuilt image: {image.id} (tags={image.tags})")
        return image.id

    @staticmethod
    def run_script(
        image: str,
        script_path: Union[str, Path],
        *,
        mount_into: str = "/workspace",
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        extra_args: Optional[Iterable[str]] = None,
        extra_mounts: Optional[Iterable[tuple]] = None,
        # [(host_dir, "/in", "ro"), ...]

    ) -> Tuple[int, str]:
        """Run a local Python script inside the container.

        extra_mounts: iterable of (host_dir, container_dir, mode) tuples.
        Example: [(Path('C:/data/tiles'), '/in', 'ro')]
        """
        host_script = Path(script_path).resolve(strict=True)
        host_dir = str(host_script.parent)

        volumes = {host_dir: {"bind": mount_into, "mode": "rw"}}
        if extra_mounts:
            for host_d, bind_to, mode in extra_mounts:
                h = str(Path(host_d).resolve())
                volumes[h] = {"bind": bind_to, "mode": mode}

        container_workdir = workdir or mount_into
        container_script = f"{mount_into.rstrip('/')}/{host_script.name}"

        client = docker.from_env()
        container = client.containers.run(
            image=image,
            command=["python", container_script] +
            (list(extra_args) if extra_args else []),
            name=name,
            working_dir=container_workdir,
            volumes=volumes,
            environment=env or {},
            detach=True,
        )
        result = container.wait()
        exit_code = int(result.get("StatusCode", 1))
        output = container.logs(
            stdout=True,
            stderr=True).decode(
            "utf-8",
            errors="replace")
        container.remove(force=True)
        return exit_code, output
