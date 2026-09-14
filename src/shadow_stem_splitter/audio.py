"""Reading and writing the WAVs a separation job needs.

No dependencies beyond numpy: the input is decoded with whatever the machine already has (ffmpeg, or
macOS's `afconvert`), the stems are written as plain 16-bit PCM so every player and DAW opens them.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

__all__ = ["read_wav", "write_wav", "decode_to_wav", "duration_ms"]


def read_wav(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Read a PCM WAV as float frames: `(N,)` mono, `(N, channels)` interleaved.

    Handles the 16/24/32-bit PCM and 32-bit float files that separators and DAWs write, including
    `WAVE_FORMAT_EXTENSIBLE` headers (which Python's `wave` refuses).
    """
    try:
        return _read_wave_module(path)
    except Exception:
        return _read_riff(path)


def _read_wave_module(path: "str | Path") -> "tuple[np.ndarray, int]":
    with wave.open(str(Path(path).expanduser()), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width * 8} bit")
    if channels > 1:
        data = data.reshape(-1, channels)
    return data, rate


def _read_riff(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Minimal RIFF reader for the headers `wave` refuses (extensible PCM, float, 24-bit)."""
    raw = Path(path).expanduser().read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    offset = 12
    fmt: "tuple[int, int, int, int] | None" = None
    payload = b""
    while offset + 8 <= len(raw):
        chunk = raw[offset : offset + 4]
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        body = raw[offset + 8 : offset + 8 + size]
        if chunk == b"fmt ":
            tag = int.from_bytes(body[0:2], "little")
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            if tag == 0xFFFE and len(body) >= 26:
                tag = int.from_bytes(body[24:26], "little")
            fmt = (tag, channels, rate, bits)
        elif chunk == b"data":
            payload = body
        offset += 8 + size + (size % 2)
    if fmt is None or payload == b"":
        raise ValueError(f"no audio data in {path}")
    tag, channels, rate, bits = fmt
    if tag == 3:
        data = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 16:
        data = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 24:
        packed = np.frombuffer(payload, dtype=np.uint8)
        packed = packed[: (packed.size // 3) * 3].reshape(-1, 3)
        values = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16)
        data = np.where(values >= 1 << 23, values - (1 << 24), values).astype(np.float32) / 8388608.0
    elif tag == 1 and bits == 32:
        data = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    elif tag == 1 and bits == 8:
        data = (np.frombuffer(payload, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV format tag {tag} ({bits}-bit) in {path}")
    if channels > 1:
        data = data[: (data.size // channels) * channels].reshape(-1, channels)
    return data.astype(np.float32), rate


def write_wav(path: "str | Path", samples: np.ndarray, rate: int) -> Path:
    """Write 16-bit PCM WAV (mono for 1-D input, interleaved otherwise)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = samples if samples.ndim == 2 else samples.reshape(-1, 1)
    clipped = np.clip(frames, -1.0, 1.0)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(clipped.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((clipped * 32767.0).astype("<i2").tobytes())
    return target


def _decoder() -> "tuple[str, ...] | None":
    if (ffmpeg := shutil.which("ffmpeg")) is not None:
        return (ffmpeg, "-v", "error", "-y", "-i")
    if (afconvert := shutil.which("afconvert")) is not None:
        return (afconvert, "-f", "WAVE", "-d", "LEI16", "-o")
    return None


def decode_to_wav(path: "str | Path", target: "str | Path") -> Path:
    """Make sure `path` is a WAV the separator can read, and return the file to hand it.

    Demucs reads WAV natively but needs ffmpeg for anything else; doing the decode here with whatever
    the machine has (ffmpeg, or CoreAudio's afconvert on macOS) keeps the separator's own
    requirements out of the way.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if source.suffix.lower() in (".wav", ".wave"):
        try:
            read_wav(source)
            return source
        except Exception:
            pass  # let the decoder rewrite the header it cannot read
    decoder = _decoder()
    if decoder is None:
        raise ValueError("reading this format needs ffmpeg (or macOS afconvert); WAV works as it is")
    destination = Path(target).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = (
        [*decoder, str(destination), str(source)]
        if decoder[0].endswith("afconvert")
        else [*decoder, str(source), str(destination)]
    )
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0 or not destination.exists():
        detail = (result.stderr or result.stdout or "decoder failed").strip().splitlines()
        raise ValueError(f"could not decode {source.name}: {detail[-1] if detail else 'unknown error'}")
    return destination


def duration_ms(samples: np.ndarray, rate: int) -> int:
    return int(round(samples.shape[0] / rate * 1000))
