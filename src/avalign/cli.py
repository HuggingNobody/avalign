"""avalign 命令行入口。

目前只提供 ``--version``，后续会补充各子命令。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from avalign import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avalign",
        description="音视频跨模态对齐与检索框架",
    )
    parser.add_argument(
        "--version", action="version", version=f"avalign {__version__}"
    )
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
