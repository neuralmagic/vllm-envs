import sys

from . import errorlog

VERBOSE = True


def say(msg: str) -> None:
    if VERBOSE:
        print(f"[ve] {msg}", file=sys.stderr, flush=True)


def warn(msg: str, log: bool = True) -> None:
    print(f"[ve] WARNING: {msg}", file=sys.stderr, flush=True)
    if log:
        errorlog.record("warning", msg)


def error(msg: str, exc: BaseException | None = None) -> None:
    print(f"[ve] ERROR: {msg}", file=sys.stderr, flush=True)
    errorlog.record("error", msg, exc)


def die(msg: str, code: int = 1) -> "None":
    error(msg)
    sys.exit(code)
