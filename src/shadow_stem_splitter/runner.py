"""One separation job, run in its own process.

Stages, in order: `decode` (whatever the file is → a WAV the separator reads) → `separate` (the model
or the remote command writes vocals/drums/bass/other) → `collect` (rewrite them as readable 16-bit
PCM WAVs next to each other and record where they are).

The job file is updated after every stage, so `stem_job_status` can always say where the work is, and
a crash leaves a record rather than a mystery.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import sys
import tempfile
import time
from pathlib import Path

from .audio import decode_to_wav, duration_ms, read_wav, write_wav
from .backends import find_demucs, plugin_env, resolve, stems_for
from .jobs import JobStore, stage
from .pool import get_pool, stems_cache_dir

__all__ = ["run_job", "main"]


def _save(store: JobStore, job_id: str, record: dict) -> dict:
    store.path(job_id).write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def _demucs_command(python: str, model: str, quality: str, work: Path, source: Path) -> "list[str]":
    chosen = "htdemucs_ft" if quality == "best" and model == "htdemucs" else model
    # --float32 + 前置衰减的输入（见 run_job 的 PRE_GAIN）让 rescale 永不触发；CLI 的
    # --clip-mode 只有 rescale|clamp 且都是逐条独立（separate.py:104,185-215），所以不用它。
    command = [python, "-m", "demucs", "--name", chosen, "--float32", "--out", str(work), "--filename", "{stem}.{ext}"]
    if quality == "best":
        # the fine-tuned bag is four models: shifts buy accuracy with time
        command += ["--shifts", "2"]
    command.append(str(source))
    return command


def _command_backend(template: "list[str]", source: Path, out_dir: Path, model: str) -> "list[str]":
    return [piece.format(input=str(source), output_dir=str(out_dir), model=model) for piece in template]


def _decode_stem(path: Path) -> "tuple":
    """Decode a non-WAV stem a remote command may have produced."""
    with tempfile.TemporaryDirectory(prefix="shadow-stems-") as tmp:
        decoded = decode_to_wav(path, Path(tmp) / "decoded.wav")
        return read_wav(decoded)


#: 前置衰减（−12 dB）：让 Demucs 的逐条 rescale 永远不触发（ADR-025 v2）。
PRE_GAIN = 0.25
#: 组峰值上限：超过就整组共用一个线性增益缩到它（一个标量管四条）。
GROUP_PEAK = 0.99
#: Demucs rescale 的特征峰值：`wav / max(1.01 * |wav|.max(), 1)` ⇒ 恰好 1/1.01。
RESCALE_SIGNATURE = 1.0 / 1.01


def _collect(work: Path, out_dir: Path, pre_gain: float = PRE_GAIN, model: str = "htdemucs") -> dict:
    """读回四条 stem，整体还原前置衰减，按**组峰值**缩一次，写 16-bit。

    Demucs 用整条混音的 ref 归一（separate.py:171-177，所有 source 乘回同一个
    ref.std()），输入整体 ×k ⇒ 四路整体 ×k——乘回 1/pre_gain 精确还原电平；组峰值
    超 GROUP_PEAK 就整组共用一个增益，四条的相对电平从头到尾只被一个标量动过。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stems_expected = stems_for(model)
    found: "dict[str, str]" = {}
    loaded: "dict[str, tuple]" = {}
    for stem in stems_expected:
        candidates = sorted(
            (path for path in work.rglob("*") if path.is_file() and path.stem.lower() == stem),
            key=lambda path: (path.suffix.lower() not in (".wav", ".flac", ".aiff", ".aif", ".m4a"), len(path.parts)),
        )
        if not candidates:
            continue
        source = candidates[0]
        samples, rate = read_wav(source) if source.suffix.lower() in (".wav", ".wave") else _decode_stem(source)
        restored = samples * (1.0 / pre_gain)
        peak = float(np.max(np.abs(restored))) if restored.size else 0.0
        # 安全带：峰值恰好落在 1/1.01（±3e-4）说明 Demucs 还在逐条 rescale——相对电平已破。
        # 窗口比 ADR 原文的 1e-4 略宽：16-bit 往返在 ×0.25 衰减下的量化误差经 ×4 还原会放大
        # 到 ~1.1e-4（实测），自然峰值落进 3e-4 窗口的概率仍然可忽略。
        if abs(peak - RESCALE_SIGNATURE) < 3e-4:
            raise ValueError(
                f"stem {stem} 峰值 {peak:.6f} 落在 Demucs rescale 特征区（1/1.01±1e-4）——"
                "它仍在逐条缩放，相对电平已破；请降低前置衰减重跑，绝不静默交付。"
            )
        loaded[stem] = (restored, rate)
    group_peak = max((float(np.max(np.abs(samples))) for samples, _ in loaded.values() if samples.size), default=0.0)
    group_gain = min(1.0, GROUP_PEAK / group_peak) if group_peak > 0 else 1.0
    for stem, (samples, rate) in loaded.items():
        found[stem] = str(write_wav(out_dir / f"{stem}.wav", samples * group_gain, rate))
    return found


def run_job(job_path: "str | Path") -> dict:
    """Run one job to completion; the job file is the only thing callers ever read."""
    store = JobStore(Path(job_path).expanduser().parent)
    job_id = Path(job_path).stem
    record = store.read(job_id)
    if record.get("kind") == "install":
        from .installer import run_install_job

        return run_install_job(job_path)
    work = Path(tempfile.mkdtemp(prefix="shadow-separate-"))

    def _cancel_requested() -> bool:
        try:
            return bool(store.read(job_id).get("cancel_requested"))
        except Exception:  # noqa: BLE001 — job 文件读不了就当没取消
            return False

    try:
        if record.get("cancel_requested"):
            record.update({"status": "cancelled", "finished_at": time.time()})
            return _save(store, job_id, record)
        try:
            backend = resolve(record.get("backend"))
        except ValueError:
            # 首次分轨：本地 demucs 还没装 → 先自举（下载 torch/Demucs + 模型，
            # stage 里明示「约 XXX MB，仅一次」），装完重新解析。配了 STEM_SPLIT_COMMAND
            # 的机器 resolve() 早就成功，不会走到这——远端命令是天然的降级路径；
            # 用户显式点名 command 而没配时维持原失败语义，不偷换后端。
            from . import installer

            if record.get("backend") not in (None, "", "auto", "demucs"):
                raise

            record["stage"] = "install"
            record["install_step"] = "检查分轨组件"
            _save(store, job_id, record)

            def _report(text: str) -> None:
                record["install_step"] = text
                _save(store, job_id, record)

            try:
                install_result = installer.install(report=_report, cancel=_cancel_requested)
            except installer.InstallerError:
                if _cancel_requested():
                    record.update({"status": "cancelled", "stage": "cancelled", "finished_at": time.time()})
                    return _save(store, job_id, record)
                raise
            record["install"] = install_result
            _save(store, job_id, record)
            backend = resolve(record.get("backend"))
        record.update({"backend": backend["id"], "stage": "decode"})
        _save(store, job_id, record)

        source = Path(record["source"])
        staged = decode_to_wav(source, work / "input.wav")
        samples, rate = read_wav(staged)
        record = stage(record, "decode", seconds=round(duration_ms(samples, rate) / 1000, 2), rate=rate)
        _save(store, job_id, record)

        model_name = str(record.get("model") or "htdemucs")

        if _cancel_requested():
            record.update({"status": "cancelled", "stage": "cancelled", "finished_at": time.time()})
            return _save(store, job_id, record)

        if backend["id"] == "demucs":
            # demucs 的 AudioFile shell 出 ffmpeg（torchaudio 2.2.2 在 mac 上没有它读不了
            # wav）——resolve() 成功不代表 ffmpeg 在（venv 已装而 bin/ffmpeg 缺的重试路径），
            # 这里幂等补链，便宜（一次 which + glob）。
            from . import installer

            installer.ensure_ffmpeg()
            record["stage"] = "separate"
            _save(store, job_id, record)
            # 任务②：stems 缓存——同歌同模型（sha1(内容+模型)）直接复用，不启动 demucs。
            cache_dir = stems_cache_dir(source, model_name)
            stems_expected = stems_for(model_name)
            if all((cache_dir / f"{stem}.wav").is_file() for stem in stems_expected):
                record["stage"] = "collect"
                record["cached"] = True
                _save(store, job_id, record)
                stems = _collect(cache_dir.parent, Path(record["out_dir"]), model=model_name)
            else:
                # ADR-025 v2：先整体 ×PRE_GAIN 写临时文件再交给热 worker——rescale 只在峰值
                # 越 ~0.99 时触发，前置衰减让任何 stem 都远低于阈值；ref 归一对所有 source 用
                # 同一个标量，输出整体 ×k 精确可逆。
                quiet = work / "quiet-input.wav"
                write_wav(quiet, samples * PRE_GAIN, rate)
                shifts = 2 if str(record.get("quality") or "fast") == "best" else 0
                pool = get_pool()
                try:
                    pool.separate(quiet, cache_dir, model_name, shifts=shifts)
                except ValueError as exc:
                    # cancel 杀 worker 时 separate 会以「worker exited/stalled」抛出——
                    # 若这个 job 被 cancel 过，落 cancelled 而不是 failed（半成品由
                    # JobStore.cancel 清理）。
                    if _cancel_requested():
                        record.update({"status": "cancelled", "stage": "cancelled", "finished_at": time.time()})
                        return _save(store, job_id, record)
                    raise
                record["stage"] = "collect"
                _save(store, job_id, record)
                stems = _collect(cache_dir.parent, Path(record["out_dir"]), model=model_name)
        else:
            remote = work / "remote"
            remote.mkdir(parents=True, exist_ok=True)
            command = _command_backend(backend["command"], staged, remote, model_name)
            record["stage"] = "separate"
            record["command"] = command
            _save(store, job_id, record)
            result = subprocess.run(command, capture_output=True, text=True, env=plugin_env())
            if result.returncode != 0:
                tail = _tail(result.stderr or result.stdout or "")
                raise ValueError(f"separator exited {result.returncode}: {tail}")
            record["stage"] = "collect"
            _save(store, job_id, record)
            stems = _collect(work, Path(record["out_dir"]), model=model_name)
        missing = [stem for stem in stems_for(model_name) if stem not in stems]
        if missing:
            raise ValueError(f"the separator did not write: {', '.join(missing)}")
        record = stage(record, "collect", stems=sorted(stems))
        record.update({"status": "ok", "stems": stems, "finished_at": time.time(), "error": None})
        return _save(store, job_id, record)
    except Exception as error:  # noqa: BLE001 — the message is the product here
        record.update({"status": "failed", "error": str(error), "finished_at": time.time()})
        _save(store, job_id, record)
        _quarantine_failed_job(store, job_id, record, error)
        return record
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _quarantine_failed_job(store: JobStore, job_id: str, record: dict, error: Exception) -> None:
    """任务③：失败证据进 jobs/failed/<id>/——error.txt（原文 + 各 stage 耗时），照 stemdeckapp。

    半成品的 out_dir 保留在原地（cancel 的清理逻辑管它），这里只搬 job 文件与诊断：
    出错后 job 文件若留在主目录，status 会继续读到 failed——搬走后 sweep/列表自然不再
    把它当活跃 job，而证据（何时、哪个阶段、多久）都在。
    """
    failed_root = store.directory / "failed" / job_id
    try:
        failed_root.mkdir(parents=True, exist_ok=True)
        stages = record.get("stages") or []
        timings = "\n".join(
            f"{entry.get('name', '?')}: {entry.get('ms', '?')}ms" for entry in stages
        ) or "(no stage completed)"
        (failed_root / "error.txt").write_text(
            f"time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
            f"job: {job_id}\n"
            f"source: {record.get('source', '?')}\n"
            f"backend: {record.get('backend', '?')}\n"
            f"stage_at_failure: {record.get('stage', '?')}\n"
            f"stage_timings:\n{timings}\n"
            f"error: {type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        job_file = store.path(job_id)
        if job_file.is_file():
            shutil.copy2(job_file, failed_root / job_file.name)
    except Exception:  # noqa: BLE001 — 隔离失败不影响 failed 状态本身
        pass


def _tail(output: str, lines: int = 25) -> str:
    """The informative end of a separator's output.

    Demucs prints a progress bar on stderr, so "the last line" is a download percentage rather than
    the error (measured: a failed separation reported `100%|███| 80.2M/80.2M` and hid the real
    "FFmpeg is not installed" two lines above it).
    """
    cleaned = [
        line.strip()
        for line in output.strip().splitlines()
        if line.strip() and "%|" not in line and not set(line.strip()) <= set("━█▏▎▍▌▋▊▉ ")
    ]
    return " | ".join(cleaned[-lines:]) or "no output"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m shadow_stem_splitter.runner <job.json> [more.json …]", file=sys.stderr)
        return 2
    # 多个 job 文件 = 一个进程串行跑完——热 worker 的复用就发生在这条路径上（MCP 桥常驻时
    # 每个 job 各自 main() 的调用方其实是桥本身，等价于同一个常驻进程）。
    failures = 0
    for job_path in sys.argv[1:]:
        record = run_job(job_path)
        if record.get("status") != "ok":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
