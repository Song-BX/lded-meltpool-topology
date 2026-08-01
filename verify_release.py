"""Verify the source-only R1 reproducibility package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from scripts.release_package.verification import verify_package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the source-only R1 release package.")
    parser.add_argument("--after-reproduction", action="store_true")
    args = parser.parse_args()
    verify_package(Path(__file__).resolve().parent, allow_generated=args.after_reproduction)
    print("Release verification passed.")
