---
name: vllm-python-env
description: Ensure vLLM checkouts have their managed Python environment before Python is used. Use whenever Codex is working in the vllm-project/vllm repository or a linked worktree and needs to run Python, pytest, Python-based lint or build tools, imports, or scripts. If `.venv` is absent, run `ve init` when available; if `ve` is unavailable, ask the user how to proceed.
---

# vLLM Python Environment

Before running Python in a vLLM checkout:

1. Resolve the checkout root with `git rev-parse --show-toplevel`.
2. Confirm it is vLLM by checking for both `setup.py` and the `vllm/` package.
3. If `<root>/.venv` does not exist:
   - If `ve` is available, run `ve init` from the root and wait for it to
     finish.
   - If `ve` is unavailable, ask the user how to proceed. Do not install `ve`,
     create a different virtual environment, or use system Python without
     their direction.
4. If initialization fails, report the failure and do not run Python-dependent
   commands.
5. Run Python through `<root>/.venv/bin/python`; never use system `python3` or
   bare `pip`.

Use this check once per session before the first Python-dependent command.
Check again only if the environment disappears or a Python command fails in a
way that suggests the environment is broken or stale. Do not initialize an
environment for read-only source inspection or tasks that never run Python.
