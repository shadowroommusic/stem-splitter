# Stem Splitter

An MCP server that separates a finished song into **vocals, drums, bass and other** — locally with
Demucs, or through a command you point at a GPU box — for any MCP-compatible agent.

把一首成品歌拆成人声、鼓、贝斯和其它四条分轨的 MCP 服务：本地跑 Demucs，或用一条命令指向你的 GPU
机器；任何支持 MCP 的 agent 都能调用。

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
| `separate_stems` | Start a separation (`path`, `out_dir?`, `model?`, `quality?`, `backend?`) → `{job_id, status, out_dir}` |
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

The stages a job moves through are `decode` → `separate` → `collect`; `collect` is where the four
files are rewritten as plain WAVs and named `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`.

## Safety

Nothing is deleted except the job's own half-written stems when you cancel it, inside the output
directory that job was given. The source file is only ever read.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `no separation backend` | Install Demucs into the plugin's venv, or set `STEM_SPLIT_COMMAND` |
| `FFmpeg is not installed` from Demucs | `pip install soundfile` into the same venv |
| `numpy` import error | Pin `numpy<2` (torch 2.2.2 was built against numpy 1.x) |
| A job stays `running` | Its process died: `stem_job_status` reports that as `failed`, and the log tail is in `error` |

## License

[AGPL-3.0](LICENSE). The separation model (Demucs, MIT) is installed separately and is not bundled.
