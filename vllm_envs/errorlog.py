"""Append-only local incident log of ve warnings, errors, and workaround notes.

Records are JSON lines in ``<cache_dir>/errors.jsonl`` so recurring failures
(including those hit by agents, which tend to work around them silently) can
be grouped and fixed in ve itself. Recording is enabled only by ``configure()``
from the CLI entry point and never raises: a broken log must not break the
command being logged.
"""

import fcntl
import functools
import gzip
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

LOG_NAME = "errors.jsonl"
MAX_BYTES = 10 * 1024 * 1024
OUTPUT_TAIL_CHARS = 4096
FULL_LOG_DIR = "incident-logs"
MAX_FULL_LOGS = 50

_log_path: Path | None = None
_argv: list[str] = []
_run_id = uuid.uuid4().hex[:12]


def configure(cache_dir: Path | None, argv: list[str]) -> None:
    """Enable recording into ``cache_dir`` (None disables it)."""
    global _log_path, _argv
    disabled = os.environ.get("VE_NO_ERROR_LOG", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    _log_path = None if cache_dir is None or disabled else log_path(cache_dir)
    _argv = list(argv)


def enabled() -> bool:
    return _log_path is not None


def log_path(cache_dir: Path) -> Path:
    return cache_dir / LOG_NAME


def _git(args: list[str], cwd: str | Path) -> str:
    try:
        p = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5,
            env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""


@functools.cache
def _ve_version() -> str:
    try:
        from importlib.metadata import version

        ver = version("vllm-envs")
    except Exception:
        ver = "unknown"
    src = Path(__file__).parent
    sha = _git(["rev-parse", "--short", "HEAD"], src)
    if sha:
        dirty = _git(["status", "--porcelain", "--untracked-files=no"], src)
        ver += f"+g{sha}" + (".dirty" if dirty else "")
    return ver


def _agent() -> tuple[str | None, str | None]:
    if os.environ.get("CLAUDECODE"):
        return "claude-code", os.environ.get("CLAUDE_CODE_SESSION_ID")
    if any(k.startswith("CODEX_") for k in os.environ):
        return "codex", os.environ.get("CODEX_THREAD_ID")
    return None, None


@functools.cache
def _repo_context_for(cwd: str) -> tuple[tuple[str, str], ...]:
    root = _git(["rev-parse", "--show-toplevel"], cwd)
    if not root:
        return ()
    return (
        ("repo", root),
        ("head", _git(["rev-parse", "--short", "HEAD"], cwd)),
        ("branch", _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)),
    )


def _repo_context() -> dict:
    try:
        return dict(_repo_context_for(os.getcwd()))
    except OSError:
        return {}


def _tail_all(text) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode(errors="replace")
    return text


_NINJA_PROGRESS = re.compile(r"^\[\d+/\d+\] ")
_ERROR_LINE = re.compile(
    r"(?<![\w-])(?:fatal )?error\b|undefined reference|\bKilled\b|"
    r"No space left|out of memory|No solution found|"
    r"Because .* we can conclude",
    re.IGNORECASE,
)
_NOT_ERROR = re.compile(r"\b0 errors?\b|-Werror|error_?(?:code|handler)", re.I)
EXCERPT_MAX_LINES = 120
EXCERPT_LINE_CHARS = 400


def _clip(line: str) -> str:
    return line if len(line) <= EXCERPT_LINE_CHARS else (
        line[:EXCERPT_LINE_CHARS] + " …"
    )


def _focus(lines: list[str], before: int = 2, after: int = 6) -> list[str]:
    """Error lines with a little context; build output is mostly warnings."""
    hits = [
        i for i, line in enumerate(lines)
        if _ERROR_LINE.search(line) and not _NOT_ERROR.search(line)
    ]
    keep: list[int] = []
    for i in hits:
        for j in range(max(i - before, 0), min(i + after + 1, len(lines))):
            if not keep or j > keep[-1]:
                if keep and j > keep[-1] + 1:
                    keep.append(-1)
                keep.append(j)
    return [lines[i] if i >= 0 else "  ..." for i in keep]


def error_excerpt(text: str) -> str:
    """The lines that explain a failure, pulled from (possibly huge) output:
    ninja FAILED blocks narrowed to their errors, else error lines anywhere."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines) and len(out) < EXCERPT_MAX_LINES:
        if not lines[i].startswith("FAILED: "):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not (
            _NINJA_PROGRESS.match(lines[j])
            or lines[j].startswith(("FAILED: ", "ninja: "))
        ):
            j += 1
        body = lines[i + 1:j]
        focused = _focus(body) if len(body) > 40 else body
        out += [lines[i], *(focused or body[:20]), ""]
        i = j
    if not out:
        out = _focus(lines)
    return "\n".join(_clip(line) for line in out[:EXCERPT_MAX_LINES]).strip()


def _save_full_output(text: str, rec_id: str) -> str | None:
    assert _log_path is not None
    out_dir = _log_path.parent / FULL_LOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rec_id}.log.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)
    logs = sorted(out_dir.glob("*.log.gz"), key=lambda q: q.stat().st_mtime)
    for old in logs[:-MAX_FULL_LOGS]:
        old.unlink(missing_ok=True)
    return str(path)


def _subprocess_fields(err: subprocess.CalledProcessError, rec_id: str) -> dict:
    cmd = err.cmd
    text = None
    spool = getattr(err, "output_log", None)
    if spool:
        try:
            text = Path(spool).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = None
    if text is None:
        out, errs = _tail_all(err.stdout), _tail_all(err.stderr)
        text = out + ("\n" if out and errs else "") + errs
    fields = {
        "cmd": [str(c) for c in cmd] if isinstance(cmd, (list, tuple))
        else [str(cmd)],
        "returncode": err.returncode,
        "output_tail": text[-OUTPUT_TAIL_CHARS:],
    }
    excerpt = error_excerpt(text)
    if excerpt:
        fields["error_excerpt"] = excerpt
    if len(text) > OUTPUT_TAIL_CHARS:
        fields["full_log"] = _save_full_output(text, rec_id)
    return fields


def _exception_fields(exc: BaseException, rec_id: str) -> dict:
    fields = {
        "exc_type": type(exc).__name__,
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-2 * OUTPUT_TAIL_CHARS:],
    }
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, subprocess.CalledProcessError):
            fields["subprocess"] = _subprocess_fields(cur, rec_id)
            break
        cur = cur.__cause__ or cur.__context__
    return fields


_NORMALIZE = (
    (re.compile(r"(?:~|\.{0,2})?(?:/[^\s'\":,()\[\]]+)+"), "<path>"),
    (re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,64}\b"), "<hash>"),
    (re.compile(r"\d+(?:\.\d+)*"), "<n>"),
)


def fingerprint(level: str, message: str) -> str:
    """Stable short id for an incident type: paths, hashes and numbers are
    normalized away so repeats of the same failure group together."""
    text = message.splitlines()[0] if message else ""
    for pattern, repl in _NORMALIZE:
        text = pattern.sub(repl, text)
    return hashlib.sha1(f"{level}:{text}".encode()).hexdigest()[:10]


def _write(rec: dict) -> None:
    assert _log_path is not None
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str) + "\n"
    with open(_log_path.with_name(LOG_NAME + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if _log_path.stat().st_size > MAX_BYTES:
                os.replace(_log_path, _log_path.with_name(LOG_NAME + ".1"))
        except FileNotFoundError:
            pass
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(line)


def _base_record(level: str, message: str) -> dict:
    agent, session = _agent()
    rec = {
        "id": uuid.uuid4().hex[:12],
        "run_id": _run_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "message": message,
        "fingerprint": fingerprint(level, message),
        "argv": _argv or ["ve", *sys.argv[1:]],
        "cwd": os.getcwd(),
        **_repo_context(),
        "ve_version": _ve_version(),
        "host": socket.gethostname(),
        "agent": agent,
        "pid": os.getpid(),
    }
    if session:
        rec["agent_session"] = session
    return rec


def record(
    level: str, message: str, exc: BaseException | None = None
) -> str | None:
    """Append a warning/error; returns its id, or None when not recording.
    Called from an ``except`` block, the active exception is captured too."""
    if _log_path is None:
        return None
    try:
        if exc is None:
            active = sys.exc_info()[1]
            if active is not None and not isinstance(active, SystemExit):
                exc = active
        rec = _base_record(level, message)
        if exc is not None:
            rec.update(_exception_fields(exc, rec["id"]))
        _write(rec)
        return rec["id"]
    except Exception:
        return None


def record_drift(env_root: Path, drift, context: str) -> str | None:
    """Record hand-made venv changes (see drift.py); not an error, but a sign
    that the env ve built was not usable as-is."""
    if _log_path is None:
        return None
    try:
        rec = _base_record("drift", f"venv drift: {drift.summary()}")
        rec.update(
            env=str(env_root),
            context=context,
            drift_mode=drift.mode,
            drift={
                "added": drift.added,
                "removed": drift.removed,
                "changed": {k: list(v) for k, v in drift.changed.items()},
                "files": drift.files,
                "n_files": drift.n_files,
            },
            **dict(_repo_context_for(str(env_root))),
        )
        # group by which packages/areas were touched, not by versions
        rec["fingerprint"] = fingerprint(
            "drift",
            " ".join([*sorted(f"+{n}" for n in drift.added),
                      *sorted(f"-{n}" for n in drift.removed),
                      *sorted(f"~{n}" for n in drift.changed),
                      *sorted({"files:" + f.split("/", 1)[0] for f in drift.files})]),
        )
        _write(rec)
        return rec["id"]
    except Exception:
        return None


def append_note(message: str, target: dict | None) -> str | None:
    """Record how an incident was handled; links to ``target`` when given."""
    if _log_path is None:
        return None
    try:
        rec = _base_record("note", message)
        if target:
            rec["ref"] = target.get("id")
            rec["ref_fingerprint"] = target.get("fingerprint")
        _write(rec)
        return rec["id"]
    except Exception:
        return None


def read(cache_dir: Path) -> list[dict]:
    """All records (rotated file first), oldest to newest."""
    path = log_path(cache_dir)
    out: list[dict] = []
    for p in (path.with_name(LOG_NAME + ".1"), path):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "id" in rec:
                out.append(rec)
    return out
