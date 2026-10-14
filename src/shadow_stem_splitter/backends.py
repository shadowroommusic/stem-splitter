"""Where the separation actually happens.

Two backends, in the order they are tried:

1. **demucs** — the local model (`pip install demucs`), run through its own interpreter. It is what
   most machines will use, and it splits into four stems: vocals, drums, bass, other.
2. **command** — a template in `STEM_SPLIT_COMMAND` (ssh to a GPU box, a container, a wrapper
   script). The command receives `{input}`, `{output_dir}` and `{model}` and is expected to leave the
   four stems in `{output_dir}`.

Neither is a dependency of this plugin: without both, `stem_backends` says so and `separate_stems`
fails with the sentence that explains how to fix it.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

__all__ = ["STEMS", "STEMS_6S", "MODELS", "stems_for", "find_demucs", "command_template", "describe", "resolve"]

#: The four stems every Demucs model produces, in the order the studio shows them.
STEPS = ("vocals", "drums", "bass", "other")
STEMS = STEPS

#: `htdemucs_6s` adds guitar and piano. Official README: "the piano source is not working
#: great at the moment" — offered as an explicit choice, never the default.
STEMS_6S = STEPS + ("guitar", "piano")

#: Models worth offering: `htdemucs` is the fast one, `htdemucs_ft` the fine-tuned (4× slower).
MODELS = ("htdemucs", "htdemucs_ft", "mdx_extra", "htdemucs_6s")


def stems_for(model: str) -> "tuple[str, ...]":
    """Which stems this model writes — the 6-source model adds guitar and piano."""
    return STEMS_6S if model == "htdemucs_6s" else STEPS


def workspace() -> Path:
    return Path(os.environ.get("SHADOWROOM_HOME") or Path.home() / "Documents" / "ShadowRoom")


def find_demucs() -> "tuple[str | None, str]":
    """`(python, reason)` where `python` runs `-m demucs`, or `None` plus why not."""
    override = os.environ.get("STEM_SPLIT_PYTHON")
    candidates = [
        override,
        str(workspace() / "_venvs" / "stem-splitter" / "bin" / "python"),
        shutil.which("python3"),
        shutil.which("python"),
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", "import demucs.separate"],
            capture_output=True,
            text=True,
            env=plugin_env(),
        )
        if probe.returncode == 0:
            return candidate, "available"
        if candidate == override:
            return None, f"STEM_SPLIT_PYTHON={candidate} cannot import demucs: {probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else 'unknown error'}"
    return None, (
        "demucs is not installed: create the plugin's virtualenv and install it "
        "(python3 -m venv _venvs/stem-splitter && _venvs/stem-splitter/bin/pip install demucs)"
    )


def command_template() -> "list[str] | None":
    """The remote command, split like a shell would but never handed to a shell."""
    raw = os.environ.get("STEM_SPLIT_COMMAND")
    if not raw:
        return None
    parts = shlex.split(raw)
    return parts or None


def plugin_env() -> "dict[str, str]":
    """Environment for the separator: the shared `bin` first, so ffmpeg is found."""
    return {
        **os.environ,
        "PATH": f"{workspace() / 'bin'}:{os.environ.get('PATH', '')}",
        "PYTHONUNBUFFERED": "1",
    }


def describe() -> dict:
    """What this machine can do right now — the answer to `stem_backends`."""
    python, reason = find_demucs()
    command = command_template()
    from . import installer  # 延迟导入：mcp_server 顶层只拉 backends，装没装后端都先能列工具

    return {
        "install": installer.status(demucs_available=python is not None),
        "backends": [
            {
                "id": "demucs",
                "available": python is not None,
                "detail": reason,
                "python": python,
                "models": list(MODELS),
                "stems": list(STEPS),
                "stems_by_model": {"htdemucs_6s": list(STEMS_6S), "default": list(STEPS)},
            },
            {
                "id": "command",
                "available": command is not None,
                "detail": "STEM_SPLIT_COMMAND is set" if command else "STEM_SPLIT_COMMAND is not set",
                "command": command,
                "stems": list(STEPS),
            },
        ],
        "preferred": "command" if command else ("demucs" if python else None),
    }


def resolve(preferred: str | None = None) -> dict:
    """Pick a backend: the requested one, else `command`, else `demucs`, else fail loudly."""
    described = describe()
    order = [preferred] if preferred and preferred != "auto" else []
    # honour the documented preference (a configured command wins over the local model) before
    # falling back to whatever else is available
    if not order and described.get("preferred"):
        order.append(str(described["preferred"]))
    order += [entry["id"] for entry in described["backends"] if entry["available"]]
    for wanted in order:
        for entry in described["backends"]:
            if entry["id"] == wanted and entry["available"]:
                return entry
    raise ValueError(
        "no separation backend: install demucs in the plugin's virtualenv, or set STEM_SPLIT_COMMAND "
        "to a command that writes the four stems"
    )
