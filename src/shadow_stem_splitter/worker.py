"""持久 demucs 热进程（任务：splitter-worker ①）。

参考 stemdeckapp/app/pipeline/demucs_worker.py（Apache-2.0）的做法：模型只加载一次，
父进程每 job 往 stdin 写一行 JSON，worker 从 stdout 回结果标记。

协议（与 stemdeck 的差异：我们用 stdout 传标记，demucs 自己的进度条留在 stderr——
父进程不用解析它，job 文件里记 stage 就够）：

  - 父进程每 job 写一行：{"source": "<前置衰减后的 wav>", "out_dir": "<写 stems 的目录>",
    "model": "htdemucs", "shifts": 0|2}
  - 成功：worker 写一行 "@@DONE@@" 到 stdout，**继续服务下一个 job**（这就是省 35–42% 的地方）
  - 失败：worker 写 "@@ERROR@@<消息>" 并 exit(1)——推理中途出错后 GPU/内存状态不可信，
    父进程下次重新拉起而不是复用（stemdeck 同款裁定）
  - stdin EOF（父进程关了管道）→ 循环干净退出
  - SHADOW_SPLITTER_PARENT_PID 环境变量武装父进程死亡看门狗：推理中 worker 不读 stdin，
    SIGKILL 杀掉的父进程不会关管道——看门狗每秒轮询，父进程消失就 os._exit（守护线程上
    sys.exit 打不断 C 层推理）
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

__all__ = ["main"]

#: 父进程死亡看门狗的轮询间隔（秒）。
_PARENT_POLL_SECONDS = 1.0


def _parent_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 存在但属别人——看门狗不许在含糊时开枪


def _watch_parent(parent_pid: int) -> None:
    while True:
        if not _parent_exists(parent_pid):
            sys.stdout.write("@@ERROR@@parent process exited\n")
            sys.stdout.flush()
            os._exit(1)
        time.sleep(_PARENT_POLL_SECONDS)


def arm_parent_watchdog() -> None:
    raw = os.environ.get("SHADOW_SPLITTER_PARENT_PID", "").strip()
    if not raw:
        return
    try:
        parent_pid = int(raw)
    except ValueError:
        return
    if parent_pid <= 0 or parent_pid == os.getpid():
        return
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()


def _run_one_job(model, req: dict) -> None:
    """一个 job：调 demucs 的库函数（与 CLI 同一条音频路径），写出四条 float stem。

    输入是父进程已做过前置衰减（ADR-025 v2 的 PRE_GAIN）的临时 wav；输出保持 float
    （--float32 语义），组增益还原/缩放在父进程的 collect 阶段做——worker 只管分离。
    """
    from demucs.apply import apply_model
    from demucs.audio import save_audio
    from demucs.separate import load_track

    source = Path(req["source"])
    out_dir = Path(req["out_dir"])
    shifts = int(req.get("shifts", 0))
    device = str(req.get("device", "cpu"))

    wav = load_track(source, model.audio_channels, model.samplerate)
    ref = wav.mean(0)
    # out-of-place（不是 CLI 的 `wav -= ref.mean()`）：mono 输入下 channels 共享内存，
    # in-place 减法会触发 torch 的 RuntimeError（实测 demucs 4.0.1 CLI 对 mono wav 直接崩；
    # 这里给每一步分配新张量，mono/stereo 都安全）。
    wav = wav - ref.mean()
    wav = wav / ref.std()
    sources = apply_model(
        model,
        wav[None],
        device=device,
        shifts=shifts,
        split=True,
        overlap=0.25,
        progress=False,
        num_workers=0,
        segment=None,
    )[0]
    sources = sources * ref.std()
    sources = sources + ref.mean()

    out_dir.mkdir(parents=True, exist_ok=True)
    for stem_tensor, name in zip(sources, model.sources):
        save_audio(
            stem_tensor,
            str(out_dir / f"{name}.wav"),
            samplerate=model.samplerate,
            clip="rescale",  # 永不触发：输入已前置衰减（ADR-025 v2）
            bits_per_sample=32,
            as_float=True,
        )


def main() -> None:
    arm_parent_watchdog()
    model_name = sys.argv[1] if len(sys.argv) > 1 else "htdemucs"
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.eval()
    if hasattr(model, "cpu"):
        model.cpu()

    print(f"@@READY@@{model_name}", flush=True)  # 父进程等到这行才发第一个 job
    # stderr 侧的加载证据（验收①数这个：连续多首应只出现一次）
    print(f"[worker] model loaded: {model_name}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            _run_one_job(model, req)
        except Exception as exc:  # noqa: BLE001 — 消息就是产品（job 的 error 字段）
            sys.stdout.write(f"@@ERROR@@{json.dumps(str(exc))}\n")
            sys.stdout.flush()
            sys.exit(1)
        sys.stdout.write("@@DONE@@\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
