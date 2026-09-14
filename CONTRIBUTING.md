# Contributing

Thanks for looking. This plugin follows the ShadowRoom house style; `AGENTS.md` has the decisions and
the reasons behind them.

## Before you open a pull request

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Every test must pass without a model installed: the separation itself is exercised through a fake
`STEM_SPLIT_COMMAND`, which is also the backend a CI machine would use.

## Conventions

- **Comments and docs explain *why*.** A comment that restates the code is noise; a comment that
  records a measurement ("174 BPM read as 174.93 until the tempo was refined") is the point.
- **No new dependencies for the server.** It stays stdlib + numpy. Anything heavier belongs in a
  virtualenv the plugin drives, or behind a command template.
- **Errors are for humans.** Say what failed and how to fix it, in one sentence.
- **Commits:** `feat(scope):`, `fix(scope):`, `docs(scope):`, `chore(scope):`.
