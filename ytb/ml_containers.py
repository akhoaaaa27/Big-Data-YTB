"""Execute ML jobs inside the persistent Compose runner containers."""

from __future__ import annotations

import os
from typing import Sequence


def run_ml_job(container_name: str, arguments: Sequence[str], timeout: int = 6 * 60 * 60) -> None:
    """Run one Python job inside an existing container and stream its logs to Airflow."""
    client = None
    try:
        import docker

        socket_url = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
        client = docker.DockerClient(base_url=socket_url, timeout=timeout)
        container = client.containers.get(container_name)
        container.reload()
        if container.status != "running":
            container.start()
        command = ["python3", "-u", *map(str, arguments)]
        print(f"[{container_name}] Chạy: {' '.join(command)}", flush=True)
        exec_id = client.api.exec_create(
            container.id, command, workdir="/opt/jobs", stdout=True, stderr=True, tty=False
        )["Id"]
        for chunk in client.api.exec_start(exec_id, stream=True, demux=False):
            if chunk:
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
        exit_code = client.api.exec_inspect(exec_id)["ExitCode"]
        if exit_code != 0:
            raise RuntimeError(f"Job kết thúc với exit code {exit_code}")
    except Exception as error:
        raise RuntimeError(f"Không chạy được job trong container {container_name}: {error}") from error
    finally:
        if client is not None:
            client.close()
