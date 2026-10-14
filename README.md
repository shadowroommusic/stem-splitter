# Stem Splitter

An MCP server that separates a finished song into **vocals, drums, bass and other** — or, with the
six-source model, **guitar and piano on top of those** — locally with Demucs, or through a command
you point at a GPU box, for any MCP-compatible agent.

把一首成品歌拆成人声、鼓、贝斯和其它四条分轨的 MCP 服务：本地跑 Demucs，或用一条命令指向你的 GPU
机器；任何支持 MCP 的 agent 都能调用。

> **Platform support:** macOS — supported (developed & verified on this machine) · Windows — unverified (paths are cross-platform in the codebase, but no real-machine testing yet).

[中文说明](README.zh-CN.md) · License: [AGPL-3.0](LICENSE)

## Why it is shaped like this

Separation takes minutes, and an MCP call that blocks for minutes gets killed by its client. So the
work runs as a **job**: `separate_stems` writes a job file and starts a detached process, then returns
a job id immediately. The client polls `stem_job_status`, and can stop the work with
`stem_job_cancel`. A bridge restart, a plugin restart or a closed laptop lid does not lose the job —
the record on disk is the truth.

The model is not a dependency of the server: the plugin itself is stdlib + numpy, and it runs Demucs
through whichever interpreter has it (by default a virtualenv at
`<ShadowRoom>/_venvs/stem-splitter`).

## Features

- **Four stems**, the split DJs and producers actually ask for: `vocals`, `drums`, `bass`, `other`.
- **Six stems when you ask for them.** `model: "htdemucs_6s"` adds `guitar` and `piano`. It is not
  the default, and that is deliberate: Demucs's own README says its piano source is not working
  well, and the six sources sum back to the mix more loosely than the four-stem model does —
  measured on this machine, correlation **0.963–0.997** against the mix (it varies with the
  material), where four stems hold **0.999**. Use it to pull a guitar or a piano out; use the
  four-stem model when the sum has to be the song.
- **Two backends.** Local Demucs (`htdemucs`, or `htdemucs_ft` for the slow fine-tuned bag), or a
  command template (`STEM_SPLIT_COMMAND`) for a remote GPU box, a container, or a wrapper script.
- **Any input.** WAV is read directly; mp3 / m4a / flac / aiff / ogg go through ffmpeg, or through
  macOS's own `afconvert` when ffmpeg is not installed.
- **Plain output.** Every stem is written as 16-bit PCM WAV next to the others, whatever the backend
  produced, so every player and DAW opens them.
- **Honest errors.** No backend → the message says how to install one; a separator that writes three
  stems → the missing one is named.

## Requirements

| | |
| --- | --- |
| OS | macOS, Linux or Windows |
| Python | 3.9 or newer (for the MCP server itself: stdlib + numpy) |
| Model | none of its own — Demucs runs from its own virtualenv, or a remote command |
| Decoder | ffmpeg, or macOS `afconvert`, for anything that is not WAV |

## Install

### As an MCP server (any client)

```json
{
  "mcpServers": {
    "stem-splitter": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/stem-splitter"
    }
  }
}
```

### The separator itself (local Demucs)

```sh
python3 -m venv ~/Documents/ShadowRoom/_venvs/stem-splitter
~/Documents/ShadowRoom/_venvs/stem-splitter/bin/pip install -r requirements-local.txt
```

That installs torch 2.2.2, torchaudio, Demucs 4.0.1 and soundfile (about 300 MB). The first
separation downloads the `htdemucs` weights (~80 MB). Measured on an M4 Mac: a 20-second excerpt
separated in **11.7 seconds**, and the four stems summed back to the mix at 0.999 correlation.

The versions matter: torch 2.2.2 was built against numpy 1.x, so `numpy<2` is pinned; and without
`soundfile` (or ffmpeg) torchaudio cannot open even a plain WAV — Demucs then fails with
"FFmpeg is not installed".

## Configuration

| Option | Default | Used for |
| --- | --- | --- |
| `STEM_SPLIT_PYTHON` | `<ShadowRoom>/_venvs/stem-splitter/bin/python` | the interpreter that has Demucs |
| `STEM_SPLIT_COMMAND` | – | remote command: `ssh gpu 'split {input} {output_dir}'` with `{input}`, `{output_dir}`, `{model}` |
| `STEM_SPLIT_OUT_DIR` | `<ShadowRoom>/stems` | where the stems are written |
| `STEM_SPLIT_JOB_DIR` | `<ShadowRoom>/stem-splitter-jobs` | job records |
| `SHADOWROOM_HOME` | `~/Documents/ShadowRoom` | where the venv and output directories live |

## Tools

| Tool | What it does |
| --- | --- |
| `separate_stems` | Start a separation (`path`, `out_dir?`, `model?`, `quality?`, `backend?`) → `{job_id, status, out_dir}`. `model` is one of `htdemucs` (default), `htdemucs_ft`, `mdx_extra`, `htdemucs_6s` |
| `stem_job_status` | One job's status, stage, stems and errors (or the recent jobs) |
| `stem_job_cancel` | Stop a job and delete its half-written stems |
| `stem_backends` | Which backends this machine has, and how to enable the missing one |

## Usage

```sh
shadow-stem-splitter backends
shadow-stem-splitter separate ~/Music/artist - track.mp3
shadow-stem-splitter status            # the recent jobs
shadow-stem-splitter status 20260915-011407-song-20s
shadow-stem-splitter cancel 20260915-011407-song-20s
```

The stages a job moves through are `decode` → `separate` → `collect`; `collect` is where the stems
are rewritten as plain WAVs and named after the model's stem set — `vocals.wav`, `drums.wav`,
`bass.wav`, `other.wav` for the four-stem models, plus `guitar.wav` and `piano.wav` for
`htdemucs_6s`. `stem_backends` reports both sets as `stems_by_model`.

## Safety

Nothing is deleted except the job's own half-written stems when you cancel it, inside the output
directory that job was given. The source file is only ever read.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `no separation backend` | Nothing to fix on a fresh machine — the first `separate_stems` bootstraps itself (see below). To pre-install, call `stem_install_backend`. Or set `STEM_SPLIT_COMMAND` to use a remote separator instead |
| `FFmpeg is not installed` from Demucs | `pip install soundfile` into the same venv |
| `numpy` import error | Pin `numpy<2` (torch 2.2.2 was built against numpy 1.x) |
| A job stays `running` | Its process died: `stem_job_status` reports that as `failed`, and the log tail is in `error` |

## First use downloads the backend (once, ~169 MB)

The heavy dependencies (torch + Demucs ≈ 85 MB of wheels, plus the 84 MB htdemucs weights) do not
ship with the app. The first `separate_stems` on a machine without them installs them as job stages
— the job's `install_step` says what is happening, `stem_backends` reports the estimate up front
(`首次分轨需下载模型（约 169 MB，仅一次）`), and `stem_install_backend` pre-installs without waiting
for a real song. Pins match the frozen set the production venv uses, including the two that pip
would otherwise get wrong: `numpy==1.26.4` (torch 2.2.2 is built against numpy 1.x) and
`soundfile==0.13.1` (torchaudio 2.2.2 has no other wav backend on macOS). The model download
resumes (`Range`, verified 206), verifies sha256 + size before landing, tries the next URL on a bad
asset, and every network step has timeouts — a failure leaves a readable Chinese error and a retry
picks up where it left off. `STEM_SPLIT_COMMAND` machines never download anything: the command
backend wins resolution, which is the documented fallback path.

## License

[AGPL-3.0](LICENSE). The separation model (Demucs, MIT) is installed separately and is not bundled.
