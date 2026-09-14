"""Command line: the same three moves as the MCP tools, for using the splitter by hand.

    shadow-stem-splitter separate song.mp3
    shadow-stem-splitter status            # recent jobs
    shadow-stem-splitter status <job_id>
    shadow-stem-splitter cancel <job_id>
    shadow-stem-splitter backends
"""
from __future__ import annotations

import argparse
import json
import sys

from .backends import describe
from .jobs import JobStore

__all__ = ["main"]


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="shadow-stem-splitter", description="Split a song into stems.")
    parser.add_argument("--job-dir", help="where job records live")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("separate", help="start a separation job")
    start.add_argument("path")
    start.add_argument("--out-dir")
    start.add_argument("--model", default="htdemucs")
    start.add_argument("--quality", choices=("fast", "best"), default="fast")
    start.add_argument("--backend", choices=("auto", "demucs", "command"), default="auto")

    status = sub.add_parser("status", help="read one job, or the recent ones")
    status.add_argument("job_id", nargs="?")

    cancel = sub.add_parser("cancel", help="stop a job and remove its half-written stems")
    cancel.add_argument("job_id")

    sub.add_parser("backends", help="what this machine can do")

    args = parser.parse_args(argv)
    store = JobStore(args.job_dir)
    if args.command == "backends":
        _print(describe())
        return 0
    if args.command == "separate":
        _print(
            store.start(
                args.path,
                out_dir=args.out_dir,
                model=args.model,
                quality=args.quality,
                backend=args.backend,
            )
        )
        return 0
    if args.command == "cancel":
        _print(store.cancel(args.job_id))
        return 0
    _print(store.status(args.job_id) if args.job_id else {"jobs": store.recent()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
