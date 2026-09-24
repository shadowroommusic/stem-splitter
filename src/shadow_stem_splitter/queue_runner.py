"""常驻 job 消费线程（splitter-worker 修正版：模型只加载一次的真路径）。

第一版把热 worker 的复用寄望在「调用方在同一个进程里连跑多个 job」上，而真实路径是
`JobStore.start()` 每 job 起一个 detached runner——pool 单例随进程死，worker 每次
BrokenPipe 后被重新 spawn、模型重载（实测 3 个 job = 3 次 model loaded，只省 5.4%）。

正解（Codex 裁定的方案 A）：**消费线程住在插件进程里**。`separate_stems` 只写 job 文件
（status=queued），这里的后台线程串行取 queued 的 job、在进程内直接调 `run_job`——
pool 是进程级单例，跨 job 存活，worker 与模型只起一次。线程随插件进程（MCP 桥的子进程）
存活；插件死 → 线程死 → atexit/看门狗把 worker 带走。
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from .jobs import JobStore, default_job_dir
from .runner import run_job

__all__ = ["start_queue_thread"]

_POLL_SECONDS = 1.0
#: 同一 job 卡在 queued 超过这个窗口且没有任何 pid 认领 → 视为无主，重新入队执行。
_ORPHAN_SECONDS = 30


def _queued_jobs(store: JobStore) -> "list[Path]":
    """queued 的 job 文件，按创建时间（文件名里的时间戳）排序。"""
    paths = []
    for path in store.directory.glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 — 坏 job 文件跳过，不拖垮队列
            continue
        if record.get("status") == "running" and record.get("stage") == "queued":
            if record.get("cancel_requested"):
                # 已取消但还没跑：就地落 cancelled（下一轮就不再出现）
                record["status"] = "cancelled"
                record["finished_at"] = time.time()
                try:
                    path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                except Exception:  # noqa: BLE001
                    pass
                continue
            paths.append(path)
    return sorted(paths, key=lambda p: p.name)


def _drain_forever() -> None:
    store = JobStore(default_job_dir())
    while True:
        for job_path in _queued_jobs(store):
            try:
                run_job(job_path)
            except Exception:  # noqa: BLE001 — run_job 自己已把失败写进 job 文件；这里只保证线程不死
                traceback.print_exc()
        time.sleep(_POLL_SECONDS)


def start_queue_thread() -> None:
    """起常驻消费线程（幂等：多次调用只有一个线程）。"""
    if getattr(start_queue_thread, "_started", False):
        return
    start_queue_thread._started = True
    threading.Thread(target=_drain_forever, name="stem-queue", daemon=True).start()
