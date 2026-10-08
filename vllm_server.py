#!/usr/bin/env python3
"""One vLLM server shared by every stage of the suite.

``lm_eval --model vllm`` builds a fresh engine *inside every* ``lm_eval run``
process. A six-stage suite therefore loads the checkpoint six times and never
shares the KV/prefix cache. This module starts a single
``vllm.entrypoints.openai.api_server`` process instead; the stages talk to it
through the harness' own ``local-completions`` model, which is a pure HTTP
client. The model is loaded once and released once.

The lifecycle is explicit: ``run_builtin_suite.py`` calls ``start()`` after
every preflight has passed and ``stop()`` from a ``finally`` block, so the GPU
is freed no matter how the run ends. ``PR_SET_PDEATHSIG`` is armed as a
best-effort extra guard: if the suite itself is killed with SIGKILL, the kernel
still takes the server down instead of leaving a model resident on a shared
card.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

PR_SET_PDEATHSIG = 1


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Return a TCP port that is free right now.

    The probe socket is closed before vLLM binds the port, so this is a
    best-effort reservation rather than a lock. ``start()`` still fails loudly
    if something else grabs the port in between.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _arm_parent_death_signal() -> None:
    """Ask the kernel to SIGKILL this child when the suite process dies.

    Runs in the forked child before ``exec``, which is the only place where
    ``PR_SET_PDEATHSIG`` can be armed. Failure is not fatal: the caller's
    ``finally`` block still terminates the process group.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass


def server_argv(
    *,
    python: str,
    checkpoint: Path,
    served_model_name: str,
    host: str,
    port: int,
    dtype: str,
    context: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    seed: int,
    api_key: str,
    generation_config: str = "vllm",
) -> list[str]:
    """The ``vllm serve`` command line, mirroring the in-process engine settings.

    Every value here has a counterpart in the old ``--model vllm`` model args
    (dtype, context, tensor parallel size, memory fraction, batching limits,
    seed, generation config) so the two backends describe the same engine.
    ``--disable-uvicorn-access-log`` only keeps the log readable: a full suite
    sends tens of thousands of requests and the per-stage ``native.log`` files
    already record what each stage asked for.
    """
    return [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(checkpoint),
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--max-model-len",
        str(context),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--enforce-eager",
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--generation-config",
        generation_config,
        "--seed",
        str(seed),
        "--api-key",
        api_key,
        "--disable-uvicorn-access-log",
    ]


class VllmServer:
    """A running vLLM OpenAI server plus the bookkeeping the suite records."""

    def __init__(self, argv, log_path: Path, host: str, port: int, served_model_name: str, api_key: str, cwd: Path):
        self.argv = [str(item) for item in argv]
        self.log_path = Path(log_path)
        self.host = host
        self.port = port
        self.served_model_name = served_model_name
        self.api_key = api_key
        self.cwd = Path(cwd)
        self.process = None
        self.stream = None
        self.started_at = None
        self.ready_seconds = None

    @property
    def base_url(self) -> str:
        """The exact URL the harness posts to when ``--model local-completions``."""
        return f"http://{self.host}:{self.port}/v1/completions"

    @property
    def models_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/models"

    def start(self, env: dict, timeout: float = 900.0, interval: float = 1.0) -> float:
        """Launch the server and block until ``/v1/models`` answers."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.log_path.open("w")
        self.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        started = time.monotonic()
        self.process = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd),
            stdout=self.stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            preexec_fn=_arm_parent_death_signal,
        )
        try:
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"shared vLLM server exited before it was ready (code {self.process.returncode}); "
                        f"log tail from {self.log_path}:\n{self.tail()}"
                    )
                if self._models_ready():
                    self.ready_seconds = round(time.monotonic() - started, 3)
                    return self.ready_seconds
                if time.monotonic() - started > timeout:
                    raise RuntimeError(
                        f"shared vLLM server did not answer {self.models_url} within {int(timeout)}s; "
                        f"log tail from {self.log_path}:\n{self.tail()}"
                    )
                time.sleep(interval)
        except BaseException:
            self.stop()
            raise

    def _models_ready(self) -> bool:
        request = urllib.request.Request(self.models_url, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def describe(self) -> dict:
        return dict(
            backend="server",
            argv=self.argv,
            host=self.host,
            port=self.port,
            base_url=self.base_url,
            served_model_name=self.served_model_name,
            pid=self.process.pid if self.process is not None else None,
            started_at=self.started_at,
            ready_seconds=self.ready_seconds,
            log=str(self.log_path),
        )

    def stop(self, grace: float = 30.0) -> dict:
        """Terminate the server's whole process group and report how it ended."""
        info = dict(pid=None, exit_code=None, signaled_by=None, stopped_at=None)
        if self.process is None:
            return info
        process = self.process
        info["pid"] = process.pid
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        info["exit_code"] = process.returncode
        if process.returncode is not None and process.returncode < 0:
            info["signaled_by"] = signal.Signals(-process.returncode).name
        info["stopped_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
        self.process = None
        return info

    def tail(self, size: int = 6000) -> str:
        if not self.log_path.is_file():
            return "(no server log)"
        with self.log_path.open("rb") as handle:
            handle.seek(max(0, self.log_path.stat().st_size - size))
            return handle.read().decode(errors="replace").strip()


def write_server_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n")
