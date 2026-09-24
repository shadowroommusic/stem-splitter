"""The job store: one JSON file per separation, plus the process that runs it.

Separating a song takes minutes, which is far longer than an MCP call should ever block. So a job is
started as a **detached process** (`runner.py`) that owns the work and writes the job file as it
moves through its stages; the MCP server only creates the file and reads it back. That also means a
bridge restart, a plugin restart or a closed laptop lid cannot lose a running separation, and
`stem_job_cancel` is just a signal to the recorded pid.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .backends import STEPS, workspace

__all__ = ["JobStore", "default_job_dir"]

AUDIO_SUFFIXES = (".wav", ".wave", ".aif", ".aiff", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus")


def default_job_dir() -> Path:
    return Path(os.environ.get("STEM_SPLIT_JOB_DIR") or workspace() / "stem-splitter-jobs")


def default_out_root() -> Path:
    return Path(os.environ.get("STEM_SPLIT_OUT_DIR") or workspace() / "stems")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobStore:
    """Reads and writes the job files; execution is owned by the resident queue thread."""

    def __init__(self, directory: "str | Path | None" = None) -> None:
        self.directory = Path(directory).expanduser() if directory else default_job_dir()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.json"

    def job_id(self, name: str) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in name)[:40]
        return f"{stamp}-{safe or 'job'}"

    def start(
        self,
        source: "str | Path",
        *,
        out_dir: "str | Path | None" = None,
        model: str = "htdemucs",
        quality: str = "fast",
        backend: str = "auto",
        run: bool = False,
    ) -> dict:
        target = Path(source).expanduser()
        if not target.is_file():
            raise FileNotFoundError(str(target))
        if target.suffix.lower() not in AUDIO_SUFFIXES:
            raise ValueError(f"not an audio file: {target.name}")
        job_id = self.job_id(target.stem)
        destination = Path(out_dir).expanduser() if out_dir else default_out_root() / f"{target.stem}-{job_id.rsplit('-', 1)[-1]}"
        record = {
            "schema_version": 1,
            "job_id": job_id,
            "status": "running",
            "stage": "queued",
            "stages": [],
            "source": str(target),
            "out_dir": str(destination),
            "model": model,
            "quality": quality,
            "backend": backend,
            "stems": {},
            "warnings": [],
            "started_at": time.time(),
            "finished_at": None,
            "pid": 0,
            "error": None,
        }
        self.path(job_id).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        # 不再起 detached runner（每 job 一进程 → pool 单例跨 job 失效 → 模型每 job 重载，
        # 实测 3 job = 3 次 model loaded 只省 5.4%）。默认 job 就地留在 queued，由插件进程
        # 里的常驻队列线程（queue_runner.py）串行取走——worker 与模型只加载一次。
        # run=True 是同步调试/测试路径：当场在调用进程里执行完（同样复用 pool 单例）。
        self.update(job_id, pid=os.getpid())
        if run:
            from .runner import run_job as _run_job

            return _run_job(self.path(job_id))
        return self.status(job_id)

    def _source_root(self) -> str:
        root = str(Path(__file__).resolve().parents[1])
        existing = os.environ.get("PYTHONPATH", "")
        return f"{root}{os.pathsep}{existing}" if existing else root

    def read(self, job_id: str) -> dict:
        path = self.path(job_id)
        if not path.is_file():
            raise FileNotFoundError(f"no such job: {job_id}")
        return json.loads(path.read_text())

    def update(self, job_id: str, **fields) -> dict:
        record = self.read(job_id)
        record.update(fields)
        self.path(job_id).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return record

    def status(self, job_id: str) -> dict:
        """The job, with a running-but-dead process reported as failed rather than forever running."""
        record = self.read(job_id)
        if record.get("status") == "running" and record.get("pid") and not _pid_alive(int(record["pid"])):
            record = self.update(
                job_id,
                status="failed",
                error=record.get("error") or "the separation process stopped without finishing",
                finished_at=record.get("finished_at") or time.time(),
            )
        return record

    def recent(self, limit: int = 10) -> "list[dict]":
        files = sorted(self.directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        return [self.read(path.stem) for path in files[:limit]]

    def cancel(self, job_id: str) -> dict:
        record = self.read(job_id)
        # 执行者现在是插件进程里的队列线程（pid=插件自己），kill 它会带走整个 MCP 服务——
        # 改为协作取消：写 cancel_requested，runner 在每个阶段边界检查到就停；queued 的
        # job 队列线程下一轮直接跳过。分离中的 demucs 由 pool 的 stall 看门狗兜底。
        record["cancel_requested"] = True
        self.path(job_id).write_text(json.dumps(record, ensure_ascii=False, indent=2))
        pid = int(record.get("runner_pid") or 0)
        if pid > 0 and _pid_alive(pid) and record.get("status") == "running":
            # runner_pid 只在 CLI 直跑（调试路径）时存在——那条路还是进程式的，可以杀。
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
        # half-written stems are not a result: remove the directory the job was filling
        out_dir = Path(record.get("out_dir") or "")
        if out_dir != Path("") and out_dir.exists():
            for entry in sorted(out_dir.rglob("*"), reverse=True):
                try:
                    entry.unlink() if entry.is_file() else entry.rmdir()
                except OSError:
                    pass
            try:
                out_dir.rmdir()
            except OSError:
                pass
        return self.update(job_id, status="cancelled", finished_at=time.time(), stems={})


def stage(record: dict, name: str, **extra) -> dict:
    """Append one finished stage to a job record (used by the runner)."""
    stages = list(record.get("stages") or [])
    stages.append({"name": name, "at": time.time(), **extra})
    record["stages"] = stages
    record["stage"] = name
    return record
