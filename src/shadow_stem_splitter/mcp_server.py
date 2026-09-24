"""MCP server (stdio): separate a finished song into vocals / drums / bass / other.

The four tools are deliberately split the way the work is: `separate_stems` starts a job and returns
immediately, `stem_job_status` polls it, `stem_job_cancel` stops it, and `stem_backends` says what
this machine can actually do. Nothing here blocks for minutes on end — an MCP call that did would
time out in the client and leave the caller guessing.
"""
from __future__ import annotations

import json
import sys
import traceback

from . import __version__
from .backends import MODELS, describe
from .jobs import JobStore, default_job_dir, default_out_root

SERVER_NAME = "stem-splitter"
SERVER_VERSION = __version__
PROTOCOL_VERSION = "2024-11-05"

TOOLS = {
    "separate_stems": "Start separating one audio file into vocals / drums / bass / other. Returns a job id immediately.",
    "stem_job_status": "Read one separation job (or the recent ones): status, stage, stems, errors.",
    "stem_job_cancel": "Stop a running separation and remove its half-written stems.",
    "stem_backends": "Which separation backends this machine has (local demucs, or STEM_SPLIT_COMMAND) and how to enable them.",
}

EMPTY_RESULTS = {
    "resources/list": {"resources": []},
    "resources/templates/list": {"resourceTemplates": []},
    "prompts/list": {"prompts": []},
    "logging/setLevel": {},
}


def _schema(name: str) -> dict:
    if name == "separate_stems":
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the song (WAV, or mp3/m4a/flac through a decoder)."},
                "out_dir": {"type": "string", "description": f"Where the stems go (default {default_out_root()}/<name>-<stamp>)."},
                "model": {"type": "string", "enum": list(MODELS), "default": "htdemucs"},
                "quality": {
                    "type": "string",
                    "enum": ["fast", "best"],
                    "default": "fast",
                    "description": "fast = htdemucs once; best = the fine-tuned bag with 2 shifts (roughly 4× slower).",
                },
                "backend": {"type": "string", "enum": ["auto", "demucs", "command"], "default": "auto"},
                "job_dir": {"type": "string", "description": f"Where job records live (default {default_job_dir()})."},
            },
            "required": ["path"],
            "additionalProperties": False,
        }
    if name == "stem_job_status":
        return {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Omit to list the recent jobs."},
                "job_dir": {"type": "string"},
            },
            "additionalProperties": False,
        }
    if name == "stem_job_cancel":
        return {
            "type": "object",
            "properties": {"job_id": {"type": "string"}, "job_dir": {"type": "string"}},
            "required": ["job_id"],
            "additionalProperties": False,
        }
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _run(name: str, arguments: dict) -> dict:
    store = JobStore(arguments.get("job_dir"))
    if name == "separate_stems":
        return store.start(
            arguments["path"],
            out_dir=arguments.get("out_dir"),
            model=str(arguments.get("model") or "htdemucs"),
            quality=str(arguments.get("quality") or "fast"),
            backend=str(arguments.get("backend") or "auto"),
        )
    if name == "stem_job_status":
        job_id = arguments.get("job_id")
        if job_id:
            return store.status(str(job_id))
        return {"jobs": store.recent()}
    if name == "stem_job_cancel":
        return store.cancel(str(arguments["job_id"]))
    if name == "stem_backends":
        return describe()
    raise ValueError(f"unknown tool: {name}")


def response(request_id: object, result: object = None, error: object = None) -> dict:
    value = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        value["error"] = {"code": -32000, "message": str(error)}
    else:
        value["result"] = result
    return value


def handle(message: dict) -> "dict | None":
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        return response(
            request_id,
            {
                "protocolVersion": requested if isinstance(requested, str) and requested else PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if method == "ping":
        return response(request_id, {})
    if method in EMPTY_RESULTS:
        return response(request_id, EMPTY_RESULTS[method])
    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    if method == "tools/list":
        return response(
            request_id,
            {"tools": [{"name": name, "description": description, "inputSchema": _schema(name)} for name, description in TOOLS.items()]},
        )
    if method != "tools/call":
        return response(request_id, error=f"Unsupported method: {method}")
    name = params.get("name")
    arguments = params.get("arguments") or {}
    try:
        payload = _run(str(name), arguments)
        return response(request_id, {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]})
    except Exception as exc:  # reported inside the result, like every other ShadowRoom server
        return response(
            request_id,
            {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True},
        )


def main() -> None:
    # 插件进程退出（stdin EOF）时把热 worker 一并带走——不留孤儿 demucs。
    import atexit

    from .pool import get_pool
    from .queue_runner import start_queue_thread

    atexit.register(get_pool().shutdown)
    # 常驻消费线程：queued 的 job 在本进程内串行执行，pool/worker 跨 job 存活。
    start_queue_thread()
    for line in sys.stdin:
        text = line.strip()
        if text == "":
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            print(json.dumps(response(None, error="invalid JSON")), flush=True)
            continue
        try:
            reply = handle(message)
        except Exception as exc:  # noqa: BLE001 — never take the server down for one bad call
            reply = response(message.get("id"), error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        if reply is not None:
            print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
