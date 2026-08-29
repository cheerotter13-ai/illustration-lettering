#!/usr/bin/env python3
"""Mode B CLI (Mac / Windows / Linux). No CUDA, no LaMa.

Requires a clean unlettered plate per file (--clean).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    argv = sys.argv[1:]
    if "--mode-b" not in argv:
        argv = ["--mode-b", *argv]
    sys.argv = [str(HERE / "letter.py"), *argv]
    sys.path.insert(0, str(HERE))
    import runpy

    runpy.run_path(str(HERE / "letter.py"), run_name="__main__")


if __name__ == "__main__":
    main()
