# Agent notes — Stem Splitter

Repo conventions and the decisions behind this plugin, for anyone (human or agent) picking it up.

## What this repo is

One plugin in the ShadowRoom tool layer: it turns a finished song into four stems
(`vocals` / `drums` / `bass` / `other`). `producer-tools` bundles it as the 人声分轨 plugin and its
agent drives the same four tools the panel does.

## House rules

- **License:** AGPL-3.0 (see `LICENSE`). The model (Demucs, MIT) is installed separately, never
  vendored.
- **README:** user-facing only, bilingual intro, full Chinese text in `README.zh-CN.md`; internals go
  in `docs/internals.md`.
- **Commits:** `feat(scope):`, `fix(scope):`, `docs(scope):`, `chore(scope):`; author
  `shadowroommusic <shadowroommusic@users.noreply.github.com>`.
- **No guessing.** Only measured behaviour goes into code, tests and docs.

## Decisions

- **2026-09-15 — a job, not a request.** Separation takes minutes; an MCP call that blocks that long
  gets killed by its client. `separate_stems` writes a job file and starts a **detached** process
  (`shadow_stem_splitter.runner`), then returns immediately; `stem_job_status` reads the file, and a
  `running` job whose pid is gone is reported as `failed` rather than forever running.
  `stem_job_cancel` signals the process group and removes the half-written stems.

- **2026-09-15 — the model is not a dependency.** The MCP server is stdlib + numpy; Demucs lives in
  its own virtualenv (`<ShadowRoom>/_venvs/stem-splitter`), discovered by `backends.find_demucs()`,
  with `STEM_SPLIT_PYTHON` as an override. The same shape covers a remote GPU box through
  `STEM_SPLIT_COMMAND`.

- **2026-09-15 — measured when it was built (macOS arm64, Python 3.9, M4 CPU).** torch 2.2.2 is the
  last release with a cp39 arm64 wheel and needs `numpy<2` (pip otherwise installs numpy 2 and
  `import torch` fails with `_ARRAY_API not found`); Demucs also needs `soundfile` or ffmpeg to open
  even a plain WAV (`FFmpeg is not installed`). A 20-second excerpt separated in 11.7 s, and the four
  stems summed back to the mix at 0.999 correlation with a −25 dB residual.

- **2026-09-15 — errors carry the fix.** "No backend" names the venv command and
  `STEM_SPLIT_COMMAND`; a missing stem names the stem; the separator's own output is trimmed to its
  informative lines (Demucs prints a progress bar on stderr, which used to hide the real error).
