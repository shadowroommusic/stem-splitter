"""热 worker 的父进程侧：拉起、串行复用、30 分钟看门狗、按 job 缓存分离结果。

设计（照 stemdeckapp 的 separate.py/jobqueue.py 精神，Apache-2.0 参考）：

- `WorkerPool` 持有一个 demucs 热进程（模型只加载一次）。MCP 桥是多进程 fork 出来的，
  每个插件进程一个——所以 pool 就放在进程级单例上，串行队列天然成立（同一进程的 job
  一个接一个写 stdin）。
- 看 30 分钟 stall 狗：job 派发后若 stdout 在窗口内一行都没有（DONE/ERROR/READY 都算），
  判定卡死，杀进程组重启。demucs 正常一首歌 ≤10 分钟，30 分钟是「绝不误杀正常长歌」
  的宽线。
- 缓存（任务②）：`<workspace>/_stems-cache/<sha1(源文件指纹+模型)>/<model>/<stem>.wav`。
  同一首歌同一模型第二次分轨直接从缓存 collect，不启动 demucs。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .backends import find_demucs, plugin_env

__all__ = ["WorkerPool", "stems_cache_dir", "cache_key"]

#: stall 看门狗窗口：stdout 超过这么久一行都没有 → 杀进程组重启。
STALL_TIMEOUT_SECONDS = 30 * 60
#: worker 就绪（@@READY@@）的等待上限——模型加载最多给 10 分钟（首次会下载权重）。
READY_TIMEOUT_SECONDS = 10 * 60


def _cache_root() -> Path:
    override = os.environ.get("STEM_SPLIT_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    workspace = os.environ.get("SHADOWROOM_HOME") or str(Path.home() / "Documents" / "ShadowRoom")
    return Path(workspace) / "_stems-cache"


def cache_key(source: Path, model: str) -> str:
    """sha1(源文件内容指纹 + 模型名)：同歌同模型 = 同 key。

    用「大小 + 全文件 sha1」而不是 mtime：mtime 会因复制/同步变，内容不会——缓存键的
    语义就是「同一首歌」。
    """
    digest = hashlib.sha1()
    digest.update(str(model).encode())
    with source.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stems_cache_dir(source: Path, model: str) -> Path:
    return _cache_root() / cache_key(source, model) / model / source.stem


class WorkerPool:
    """一个 demucs 热进程，串行服务；失败/卡死后自动按需重启。"""

    def __init__(self) -> None:
        self._proc: "subprocess.Popen | None" = None
        self._model: str = ""
        self._loads = 0

    @property
    def loads(self) -> int:
        """模型加载次数（验收①：连续多首只允许 =1）。"""
        return self._loads

    def _spawn(self, model: str) -> subprocess.Popen:
        python, reason = find_demucs()
        if python is None:
            raise ValueError(reason)
        env = {**plugin_env(), "SHADOW_SPLITTER_PARENT_PID": str(os.getpid())}
        # 把 src/ 加进 PYTHONPATH：worker 与插件同仓（`python -m shadow_stem_splitter.worker`）。
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.Popen(
            [python, "-m", "shadow_stem_splitter.worker", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr 进日志文件（验收①要数「模型加载了几次」；tqdm 进度条也在这里，
            # 父进程不解析但不销毁证据）。SHADOW_SPLITTER_WORKER_LOG 未设时丢弃。
            stderr=(
                open(os.environ.get("SHADOW_SPLITTER_WORKER_LOG", "/dev/null"), "ab")
                if os.environ.get("SHADOW_SPLITTER_WORKER_LOG")
                else subprocess.DEVNULL
            ),
            env=env,
            start_new_session=True,  # 自己一个进程组：cancel 可以 killpg 整组
        )
        self._loads += 1
        self._model = model
        self._proc = proc
        return proc

    def _readline_with_watchdog(self, proc: subprocess.Popen, timeout: float) -> str:
        """读一行 stdout，超时杀进程组并报卡死。

        stdout 是阻塞读——看门狗靠一条守护线程在超时后 killpg，让阻塞读以 EOF 崩出来。
        """
        import threading

        def _stall_kill() -> None:
            time.sleep(timeout)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

        watchdog = threading.Thread(target=_stall_kill, daemon=True)
        watchdog.start()
        line = proc.stdout.readline() if proc.stdout else b""
        # watchdog 还在睡也没关系：daemon 线程随进程退出
        if not line:
            if proc.poll() is not None:
                raise ValueError(f"worker exited (code {proc.returncode}) without a marker")
            raise ValueError(f"worker stalled for {timeout}s without output — killed")
        return line.decode().strip()

    def _ensure_ready(self, model: str) -> subprocess.Popen:
        proc = self._proc
        if proc is not None and proc.poll() is None and self._model == model:
            return proc
        if proc is not None and proc.poll() is None:
            # 换模型：老进程的服务对象不同了，礼貌关掉
            self.shutdown()
        proc = self._spawn(model)
        ready = self._readline_with_watchdog(proc, READY_TIMEOUT_SECONDS)
        if not ready.startswith("@@READY@@"):
            raise ValueError(f"worker did not become ready: {ready[:120]}")
        return proc

    def separate(self, source: Path, out_dir: Path, model: str, shifts: int = 0) -> None:
        """跑一个分离 job（串行：上 一个没回 DONE 前不会发下一个）。"""
        proc = self._ensure_ready(model)
        request = json.dumps({"source": str(source), "out_dir": str(out_dir), "model": model, "shifts": shifts})
        assert proc.stdin is not None
        proc.stdin.write((request + "\n").encode())
        proc.stdin.flush()
        marker = self._readline_with_watchdog(proc, STALL_TIMEOUT_SECONDS)
        if marker.startswith("@@DONE@@"):
            return
        if marker.startswith("@@ERROR@@"):
            message = marker[len("@@ERROR@@"):]
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                pass
            self.shutdown()  # 失败后进程状态不可信，下次重启（stemdeck 同款裁定）
            raise ValueError(str(message))
        raise ValueError(f"unexpected worker marker: {marker[:120]}")

    def shutdown(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()  # EOF → worker 干净退出
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass


_POOL: "WorkerPool | None" = None


def get_pool() -> WorkerPool:
    global _POOL
    if _POOL is None:
        _POOL = WorkerPool()
    return _POOL
