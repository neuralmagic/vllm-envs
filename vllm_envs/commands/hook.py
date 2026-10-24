"""`ve hook <event>`: entry point of the git hooks ve installs."""

from ..config import Config
from ..hooks import (
    handle_post_checkout,
    handle_post_commit,
    handle_post_merge,
    handle_post_rewrite,
)


def cmd_hook(cfg: Config, args) -> int:
    rest = args.rest
    if args.event == "post-checkout":
        old, new, flag = (rest + ["", "", ""])[:3]
        return handle_post_checkout(cfg, old, new, flag)
    if args.event == "post-commit":
        return handle_post_commit(cfg)
    if args.event == "post-merge":
        return handle_post_merge(cfg)
    if args.event == "post-rewrite":
        return handle_post_rewrite(cfg, rest[0] if rest else "")
    return 0
