from __future__ import annotations

import argparse

from .inputs import build_manifest, load_legacy_summary, load_point_clouds, move_inputs, write_manifest
from .legacy_centroid_shift import calculate_legacy_centroid_shift_context
from .reconciliation import reconcile_legacy_summary
from .report import write_outputs
from .slice_support import audit_slice_support


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit XZ slice support and exclude unsupported spatial geometry.")
    parser.add_argument(
        "--move-inputs",
        action="store_true",
        help="Move the three canonical CSV inputs into 图/spatial_support_audit/source_inputs after hash verification.",
    )
    args = parser.parse_args()

    manifest = move_inputs() if args.move_inputs else build_manifest()
    manifest_path = write_manifest(manifest)
    support = audit_slice_support(load_point_clouds())
    legacy_summary = load_legacy_summary()
    reconciliation = reconcile_legacy_summary(legacy_summary, support)
    legacy_shift_context = calculate_legacy_centroid_shift_context(legacy_summary, reconciliation)
    paths = write_outputs(support, reconciliation, legacy_shift_context)
    print(manifest_path)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
