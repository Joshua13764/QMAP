# run_container.py
import pathlib
import sys
import time
from logging import Logger

import bennu_feature_extractor.logger_factory
import docker

logger: Logger = bennu_feature_extractor.logger_factory.get_logger(
    "docker-test", pathlib.Path("./logs")
)

IMAGE_TAG = "mltools:py3.10"
CONTAINER_WORKDIR = "/workspace"
CONTAINER_COMMAND = (
    "python -c \"from MLtools import inference; print('MLtools inference import OK')\""
)


def stream_build(log_iter):
    aux_image_id = None
    for chunk in log_iter:
        if isinstance(chunk, (bytes, str)):
            line = chunk.decode(
                "utf-8",
                "ignore") if isinstance(
                chunk,
                bytes) else str(chunk)
            line = line.rstrip("\n")
            if line:
                logger.info(line)
            continue
        if "errorDetail" in chunk:
            logger.error(chunk["errorDetail"].get("message", "Build error"))
            continue
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                if line:
                    logger.info(line.rstrip())
        if "status" in chunk:
            ident = chunk.get("id", "")
            status = chunk["status"]
            progress = chunk.get("progress", "")
            line = f"{
                ident +
                ': ' if ident else ''}{status}{
                (
                    ' ' +
                    progress) if progress else ''}"
            logger.info(line)
        aux = chunk.get("aux")
        if isinstance(aux, dict):
            candidate = aux.get("ID") or aux.get("id")
            if candidate:
                aux_image_id = candidate
                logger.info(f"Aux image ID: {aux_image_id}")
    return aux_image_id


def build_image(docker_client: docker.DockerClient,
                context_dir: pathlib.Path, dockerfile_name: str, image_tag: str):
    logger.info(f"Build context : {context_dir}")
    logger.info(f"Dockerfile    : {dockerfile_name}")
    logger.info(f"Target image  : {image_tag}")
    try:
        image, logs = docker_client.images.build(
            path=str(context_dir),
            dockerfile=dockerfile_name,
            tag=image_tag,
            rm=True,
            pull=False,
        )
    except docker.errors.BuildError as e:
        logger.error("BuildError raised. Streaming partial logs if available.")
        if getattr(e, "build_log", None):
            stream_build(e.build_log)
        logger.exception("Build failed")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Failed to start build: {e}")
        sys.exit(1)
    aux_id = stream_build(logs)
    try:
        image.reload()
        logger.info("Build complete.")
        logger.info(f"Image ID   : {image.id}")
        logger.info(f"Image tags : {', '.join(image.tags) or '(untagged)'}")
        if aux_id and aux_id != image.id:
            logger.warning(
                f"Aux image ID differs from final: aux={aux_id} final={
                    image.id}")
    except Exception:
        pass
    return image


def run_container(docker_client: docker.DockerClient, image_tag: str,
                  host_mount_dir: pathlib.Path, workdir_in_container: str, command: str):
    volumes = {
        str(host_mount_dir): {
            "bind": workdir_in_container,
            "mode": "rw"}}
    container_name = f"mltools-run-{int(time.time())}"
    logger.info(f"Bind mount   : {host_mount_dir} -> {workdir_in_container}")
    logger.info(f"Starting container '{container_name}' from {image_tag} …")
    try:
        container = docker_client.containers.run(
            image_tag,
            command=command,
            volumes=volumes,
            working_dir=workdir_in_container,
            tty=True,
            detach=True,
            name=container_name,
        )
    except Exception as e:
        logger.exception(f"Failed to start container: {e}")
        sys.exit(1)
    logger.info(f"Container ID : {container.id[:12]}")
    logger.info("--------- live container logs ---------")
    try:
        for line in container.logs(stream=True, follow=True):
            msg = line.decode("utf-8", "ignore").rstrip("\n")
            if msg:
                logger.info(msg)
    except KeyboardInterrupt:
        logger.warning("Interrupted log streaming.")
    result = container.wait()
    exit_code = result.get("StatusCode", -1)
    logger.info("---------------------------------------")
    if exit_code == 0:
        logger.info(f"Container exited with code {exit_code}")
    else:
        logger.error(f"Container exited with code {exit_code}")
    try:
        tail = container.logs(tail=20).decode("utf-8", errors="ignore")
        if tail:
            logger.info("----- last 20 log lines -----")
            for line in tail.splitlines():
                logger.info(line)
    except Exception:
        pass
    try:
        container.remove(force=True)
        logger.info("Container removed.")
    except Exception as e:
        logger.warning(f"Could not remove container: {e}")


def main():
    docker_client = docker.from_env()
    context_dir = pathlib.Path(__file__).resolve().parent
    dockerfile_name = "Dockerfile"
    build_image(docker_client, context_dir, dockerfile_name, IMAGE_TAG)
    host_mount_dir = pathlib.Path.cwd()
    run_container(
        docker_client,
        IMAGE_TAG,
        host_mount_dir,
        CONTAINER_WORKDIR,
        CONTAINER_COMMAND)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
