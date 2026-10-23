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

def check_ml_runtime() -> dict:
    """Fail early if Airflow cannot access the Docker daemon or shared ML runners."""
    client = None
    try:
        import docker

        socket_url = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
        client = docker.DockerClient(base_url=socket_url, timeout=15)
        client.ping()
        result = {}
        for name in ("ytb-gsdmm-runner", "ytb-sentiment-runner"):
            container = client.containers.get(name)
            container.reload()
            if container.status != "running":
                container.start()
                container.reload()
            if container.status != "running":
                raise RuntimeError(f"Container {name} chưa ở trạng thái running.")
            mounts = {mount.get("Destination") for mount in (container.attrs.get("Mounts") or [])}
            required_mounts = {"/opt/data", "/opt/jobs"}
            missing = required_mounts - mounts
            if missing:
                raise RuntimeError(
                    f"Container {name} thiếu mount {sorted(missing)}; "
                    "kiểm tra docker-compose.yml và recreate runner."
                )
            result[name] = {"status": container.status, "mounts": sorted(mounts)}
        return {"docker": "reachable", "runners": result}
    except Exception as error:
        raise RuntimeError(
            "Kiểm tra môi trường ML thất bại. Hãy kiểm tra Docker Desktop đang chạy, "
            "Docker socket mount và hai service runner trong docker-compose.yml. "
            f"Chi tiết: {error}"
        ) from error
    finally:
        if client is not None:
            client.close()