from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import docker
from docker.errors import APIError, ImageNotFound
from docker.models.containers import Container
from docker.types import DeviceRequest

from boulder_statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk

DOCKER_IMAGE_TAG = "mltools:py3.10-cuda"

Mount = Tuple[Path, str, str]


class DockerHelpers:

    @staticmethod
    def analyse_image(
        image_paths: List[FSPathLocalDisk],
        inference_output_paths: List[FSPathLocalDisk],
        verbose: bool = False,
        detection_export_custom_name_tag: str = "",
    ) -> None:
        overlay_script: Path = (
            Path(__file__).parent / "BoulderNetCUDA" /
            "bouldernet_infer_overlay.py"
        )

        mounts, env = DockerHelpers.get_mounts_env(
            image_paths, inference_output_paths
        )

        env["detection_export_custom_name_tag"] = detection_export_custom_name_tag

        print("env :", env)
        print("mounts :", mounts)

        code, logs = DockerHelpers.run_script(
            DOCKER_IMAGE_TAG,
            overlay_script.as_posix(),
            env=env,
            extra_args=[
                *[f"/in/{image_path.actual_path.name}" for image_path in image_paths],
            ],
            extra_mounts=mounts,
        )

        if verbose:
            print("Exit:", code)
            print(logs)

    @staticmethod
    def ensure_image_exists() -> None:
        overlay_script: Path = (
            Path(__file__).parent / "BoulderNetCUDA" /
            "bouldernet_infer_overlay.py"
        )
        overlay_dir: Path = overlay_script.parent
        dockerfile_rel: str = "Dockerfile"
        dockerfile_abs: Path = overlay_dir / dockerfile_rel

        try:
            client: docker.DockerClient = docker.from_env()
            client.images.get(DOCKER_IMAGE_TAG)

        except ImageNotFound:
            if not dockerfile_abs.is_file():
                raise FileNotFoundError(
                    f"Expected Dockerfile at {dockerfile_abs} but it was not found."
                )

            print(
                f"Image '{DOCKER_IMAGE_TAG}' not found. Building from {overlay_dir} ..."
            )
            DockerHelpers.build_image(
                tag=DOCKER_IMAGE_TAG,
                context_dir=overlay_dir,
                dockerfile=dockerfile_rel,
                pull=False,
                no_cache=False,
                build_args=None,
            )

        except APIError as e:
            raise RuntimeError(
                "Could not communicate with Docker. Is the daemon running?"
            ) from e

    @staticmethod
    def get_mounts_env(
        image_paths: List[FSPathLocalDisk],
        inference_output_paths: List[FSPathLocalDisk],
    ) -> Tuple[List[Mount], Dict[str, str]]:
        mounts: List[Mount] = []
        env: Dict[str, str] = {}

        # Use asserts to confirm assumptions

        assert len(image_paths) == len(inference_output_paths), \
            "image_paths and inference_output_paths length mismatch"

        input_dirs = {p.actual_path.parent.resolve() for p in image_paths}
        assert len(input_dirs) == 1, \
            f"Multiple input directories detected: {input_dirs}"

        output_dirs = {p.actual_path.parent.resolve()
                       for p in inference_output_paths}
        assert len(output_dirs) == 1, \
            f"Multiple output directories detected: {output_dirs}"

        in_dir: Path = image_paths[0].actual_path.parent.resolve()
        out_dir: Path = inference_output_paths[0].actual_path.parent.resolve()
        mounts.append((in_dir, "/in", "ro"))
        mounts.append((out_dir, "/out", "rw"))

        env["OUT_DIR"] = "/out"

        return mounts, env

    @staticmethod
    def build_image(
        tag: str = DOCKER_IMAGE_TAG,
        context_dir: Union[str, Path] = ".",
        dockerfile: str = "Dockerfile",
        *,
        no_cache: bool = False,
        pull: bool = False,
        build_args: Optional[Dict[str, str]] = None,
    ) -> str:
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
    @contextmanager
    def _container_for_script(
        image: str,
        command: List[str],
        *,
        container_workdir: str,
        volumes: Dict[str, Dict[str, str]],
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ):
        client: docker.DockerClient = docker.from_env()
        device_requests = [
            DeviceRequest(count=-1, capabilities=[["gpu"]])
        ]

        container: Optional[Container] = None
        try:
            container = client.containers.run(
                image=image,
                command=command,
                name=name,
                working_dir=container_workdir,
                volumes=volumes,
                environment=env or {},
                detach=True,
                device_requests=device_requests,
            )
            yield container
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except APIError:
                    pass

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
    ) -> Tuple[int, str]:
        host_script = Path(script_path).resolve(strict=True)
        host_dir = str(host_script.parent)

        volumes: Dict[str, Dict[str, str]] = {
            host_dir: {"bind": mount_into, "mode": "rw"}
        }

        if extra_mounts:
            for host_d, bind_to, mode in extra_mounts:
                h = str(Path(host_d).resolve())
                volumes[h] = {"bind": bind_to, "mode": mode}

        container_workdir: str = workdir or mount_into
        container_script: str = f"{mount_into.rstrip('/')}/{host_script.name}"

        command: List[str] = ["python", container_script] + (
            list(extra_args) if extra_args else []
        )

        with DockerHelpers._container_for_script(
            image=image,
            command=command,
            container_workdir=container_workdir,
            volumes=volumes,
            env=env,
            name=name,
        ) as container:
            try:
                result = container.wait()
            except KeyboardInterrupt:
                try:
                    container.kill()
                except APIError:
                    pass
                raise

            exit_code = int(result.get("StatusCode", 1))
            output: str = container.logs(stdout=True, stderr=True).decode(
                "utf-8", errors="replace"
            )

        return exit_code, output
