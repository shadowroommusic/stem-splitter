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
import sys
import tempfile
import time
from pathlib import Path

from .audio import decode_to_wav, duration_ms, read_wav, write_wav
from .backends import STEMS, find_demucs, plugin_env, resolve
from .jobs import JobStore, stage

__all__ = ["run_job", "main"]


def _save(store: JobStore, job_id: str, record: dict) -> dict:
    store.path(job_id).write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def _demucs_command(python: str, model: str, quality: str, work: Path, source: Path) -> "list[str]":
    chosen = "htdemucs_ft" if quality == "best" and model == "htdemucs" else model
    command = [python, "-m", "demucs", "--name", chosen, "--out", str(work), "--filename", "{stem}.{ext}"]
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


def _collect(work: Path, out_dir: Path) -> dict:
    """Find the four stems the backend wrote, and rewrite them as plain PCM WAVs.

    Both backends may write wherever they like inside the work directory (demucs nests them under
    `<model>/<track>/`), so the stems are found by name rather than by guessing a layout.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    found: "dict[str, str]" = {}
    for stem in STEMS:
        candidates = sorted(
            (path for path in work.rglob("*") if path.is_file() and path.stem.lower() == stem),
            key=lambda path: (path.suffix.lower() not in (".wav", ".flac", ".aiff", ".aif", ".m4a"), len(path.parts)),
        )
        if not candidates:
            continue
        source = candidates[0]
        samples, rate = read_wav(source) if source.suffix.lower() in (".wav", ".wave") else _decode_stem(source)
        found[stem] = str(write_wav(out_dir / f"{stem}.wav", samples, rate))
    return found


def run_job(job_path: "str | Path") -> dict:
    """Run one job to completion; the job file is the only thing callers ever read."""
    store = JobStore(Path(job_path).expanduser().parent)
    job_id = Path(job_path).stem
    record = store.read(job_id)
    work = Path(tempfile.mkdtemp(prefix="shadow-separate-"))
    try:
        backend = resolve(record.get("backend"))
        record.update({"backend": backend["id"], "stage": "decode"})
        _save(store, job_id, record)

        source = Path(record["source"])
        staged = decode_to_wav(source, work / "input.wav")
        samples, rate = read_wav(staged)
        record = stage(record, "decode", seconds=round(duration_ms(samples, rate) / 1000, 2), rate=rate)
        _save(store, job_id, record)

        if backend["id"] == "demucs":
            python, reason = find_demucs()
            if python is None:
                raise ValueError(reason)
            command = _demucs_command(
                python,
                str(record.get("model") or "htdemucs"),
                str(record.get("quality") or "fast"),
                work,
                staged,
            )
        else:
            remote = work / "remote"
            remote.mkdir(parents=True, exist_ok=True)
            command = _command_backend(backend["command"], staged, remote, str(record.get("model") or "htdemucs"))

        record["stage"] = "separate"
        record["command"] = command
        _save(store, job_id, record)
        result = subprocess.run(command, capture_output=True, text=True, env=plugin_env())
        if result.returncode != 0:
            tail = _tail(result.stderr or result.stdout or "")
            raise ValueError(f"separator exited {result.returncode}: {tail}")

        stems = _collect(work, Path(record["out_dir"]))
        missing = [stem for stem in STEMS if stem not in stems]
        if missing:
            raise ValueError(f"the separator did not write: {', '.join(missing)}")
        record = stage(record, "collect", stems=sorted(stems))
        record.update({"status": "ok", "stems": stems, "finished_at": time.time(), "error": None})
        return _save(store, job_id, record)
    except Exception as error:  # noqa: BLE001 — the message is the product here
        record.update({"status": "failed", "error": str(error), "finished_at": time.time()})
        return _save(store, job_id, record)
    finally:
        shutil.rmtree(work, ignore_errors=True)


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
        print("usage: python -m shadow_stem_splitter.runner <job.json>", file=sys.stderr)
        return 2
    record = run_job(sys.argv[1])
    return 0 if record.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
