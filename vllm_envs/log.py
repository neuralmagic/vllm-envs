import sys

VERBOSE = True


def say(msg: str) -> None:
    if VERBOSE:
        print(f"[ve] {msg}", file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"[ve] WARNING: {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> "None":
    print(f"[ve] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)
