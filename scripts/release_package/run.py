"""Command-line entry point for the non-destructive local release builder."""

from __future__ import annotations

import argparse
from pathlib import Path

from .archive import create_archive
from .config import DEFAULT_OUTPUT_ROOT, ROOT
from .stage import build_release
from .verification import verify_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the source-only R1 GitHub upload package.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--replace", action="store_true", help="Replace only the named staged release directory.")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create the deterministic sibling ZIP and SHA-256 sidecar after verification.",
    )
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    package = build_release(args.output_root, source_root=args.source_root, replace=args.replace)
    if not args.skip_verify:
        verify_package(package, source_root=args.source_root)
    if args.archive:
        archive = create_archive(package, args.output_root, replace=args.replace)
        print(archive.archive_path)
        print(archive.checksum_path)
    print(package)


if __name__ == "__main__":
    main()
