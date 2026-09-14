# Internals

## Layout

| Path | What lives there |
| --- | --- |
| `mcp_server.py` (root) | stdio entry point: puts `src/` on `sys.path` and calls the package's `main` |
| `src/shadow_stem_splitter/mcp_server.py` | the four tools, their schemas, and the JSON-RPC loop |
| `src/shadow_stem_splitter/jobs.py` | `JobStore`: job files, the detached runner, cancel |
| `src/shadow_stem_splitter/runner.py` | one job: decode → separate → collect |
| `src/shadow_stem_splitter/backends.py` | demucs discovery, command template, `stem_backends` |
| `src/shadow_stem_splitter/audio.py` | WAV read/write and decode-through-ffmpeg/afconvert |
| `tests/` | job lifecycle, backend selection, failure messages, the tool list |

## The job file is the contract

`<JobDir>/<job_id>.json` carries `status`, `stage`, `stages[]`, `stems{}`, `out_dir`, `error`, `pid`.
Everything else — the MCP server, `stem_job_status`, the workbench UI — reads that file; nothing
depends on the runner being alive. A `running` record whose pid is gone is downgraded to `failed` on
read, so a crashed separation can never look like a hung one.

Stages: `decode` (any input → a WAV the separator reads), `separate` (demucs or the command template
writes the four stems somewhere under the work directory), `collect` (find them by name, rewrite each
as 16-bit PCM WAV into `out_dir`).

## Backends

`find_demucs()` probes, in order: `STEM_SPLIT_PYTHON`, `<ShadowRoom>/_venvs/stem-splitter/bin/python`,
`python3`, `python` — and asks each one `import demucs.separate`. `STEM_SPLIT_COMMAND` is split with
`shlex.split` and never handed to a shell; `{input}`, `{output_dir}` and `{model}` are substituted
per argument. A configured command wins over demucs, because the person who set it meant it.

## Measured on this machine (2026-09-15)

- A 20-second excerpt (44.1 kHz stereo) separated in **11.7 s** on CPU with `htdemucs`, i.e. roughly
  1.7× realtime; a 4-minute song is a few minutes.
- Stem sum vs the mix: correlation **0.999**, residual RMS 0.0103 against a 0.19 original (−25 dB).
- Stems are decorrelated from each other (|r| < 0.3), and an instrumental source yields a silent
  `vocals.wav` — which is the correct answer, not a failure.
