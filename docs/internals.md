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

## First-use bootstrap (`installer.py`, 2026-10-15)

The desktop package ships neither torch nor the model weights (~169 MB together). The plugin
bootstraps itself the first time separation is actually requested:

- `runner.run_job` catches the "no backend" resolution failure (only for `auto`/`demucs` requests)
  and runs `installer.install()` as job stages before re-resolving. A machine with
  `STEM_SPLIT_COMMAND` resolves to the command backend and never downloads.
- `stem_install_backend` exposes the same flow as a job, so the plugins page can pre-install and
  poll `install_step` through `stem_job_status`; cancellation lands on the download loop's chunk
  boundary and the pip subprocess is bounded by a 30-minute timeout.
- Pins are the production freeze: `torch==2.2.2 torchaudio==2.2.2 numpy==1.26.4 soundfile==0.13.1
  demucs==4.0.1`. numpy and soundfile are load-bearing (README troubleshooting has the symptoms);
  the base interpreter prefers the packaged pbs runtime via `SHADOWROOM_PYTHON_RUNTIME`.
- The model lands in the torch hub checkpoints dir under the demucs filename convention
  (`955717e8-8726e21a.th` = sig + sha256 prefix), verified against a pinned sha256 and byte size
  before it is renamed into place; `.part` files resume via HTTP Range (206 verified).
- `backends.describe()` carries an `install` block (needed / estimate_bytes / note) so the UI can
  say 「首次分轨需下载模型（约 XXX MB，仅一次）」 before the user commits.

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

## Six stems: a sanity band, not the 0.999 line

`htdemucs_6s` adds `guitar` and `piano` to the four sources (`STEMS_6S` in `backends.py`;
`stems_for(model)` is the single place that answers "which stems does this model write", and
`describe()` exposes both sets as `stems_by_model`). It is an explicit option, never the default.

The four-stem acceptance test is a hard line: four stems summed must correlate **≥ 0.999** with the
mix. That line exists to catch *our own* gain-path bugs — a wrong pre-attenuation, a per-stem rescale
that fired when it should not have. Six stems fail it for a reason that has nothing to do with our
code: same song, same gain path, four stems measured 0.9992 while six measured 0.9630 (0.9968 in an
earlier run — the number moves with the material), because guitar and piano carve energy out of the
other four and Demucs's own README says the piano source is not working well.

So six stems are accepted on a **sanity band** instead: correlation ≥ 0.95, RMS difference ≤ 3 dB,
every stem present and not silent. Anything tighter would have to be met by padding the guitar and
piano energy back into the four main stems — writing a post-processor to make a metric pass rather
than to make the audio better. We don't do that.
