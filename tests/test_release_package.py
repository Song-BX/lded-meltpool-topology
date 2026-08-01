from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from scripts.analysis.wls_q import nearest_neighbor_indices
from scripts.release_package.archive import create_archive
from scripts.release_package.config import (
    RAW_CSV_COUNT,
    RELEASE_NAME,
    RETIRED_ROOT_SCRIPTS,
)
from scripts.release_package.inventory import select_release_files


class ReleaseSelectionTests(unittest.TestCase):
    def test_release_expectations_match_source_only_scope(self) -> None:
        self.assertEqual(RAW_CSV_COUNT, 30)

    def test_knn_ties_use_stable_input_index_order(self) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float
        )
        indices = nearest_neighbor_indices(points, k=1)
        np.testing.assert_array_equal(indices[0], np.array([0, 1]))

    def test_allow_list_has_complete_raw_grid_and_excludes_retired_scripts(self) -> None:
        files = select_release_files()
        raw_inputs = [item for item in files if item.category == "raw_input"]
        destinations = {item.relative_path.as_posix() for item in files}
        self.assertEqual(len(raw_inputs), RAW_CSV_COUNT)
        self.assertSetEqual(
            {item.category for item in files},
            {"raw_input", "reference_input", "reproduction_metadata", "code", "test"},
        )
        self.assertFalse(any(path.startswith(("图/", "latex_restructure/")) for path in destinations))
        for legacy_name in RETIRED_ROOT_SCRIPTS:
            self.assertNotIn(f"scripts/{legacy_name}", destinations)

    def test_release_has_no_tiff_or_submission_material(self) -> None:
        destinations = {item.relative_path.as_posix().lower() for item in select_release_files()}
        self.assertFalse(any(path.endswith((".tif", ".tiff", ".docx")) for path in destinations))
        self.assertFalse(any("decision_letter" in path or "reviewer" in path for path in destinations))

    def test_archive_is_a_sibling_with_a_stable_top_level_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / RELEASE_NAME
            package.mkdir()
            (package / "README.md").write_text("release", encoding="utf-8")
            artifacts = create_archive(package, root)
            self.assertTrue(artifacts.archive_path.is_file())
            self.assertTrue(artifacts.checksum_path.is_file())
            self.assertFalse((package / artifacts.archive_path.name).exists())
            with ZipFile(artifacts.archive_path) as archive:
                self.assertEqual(archive.namelist(), [f"{RELEASE_NAME}/README.md"])


if __name__ == "__main__":
    unittest.main()
