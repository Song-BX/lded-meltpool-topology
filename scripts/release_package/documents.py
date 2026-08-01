"""Generate documentation and entry points for the source-only R1 package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RELEASE_DESCRIPTION, RELEASE_NAME, RELEASE_TAG
from .inventory import ReleaseFile, raw_csv_headers, sha256


MIT_LICENSE = """MIT License

Copyright (c) 2026 Song-BX

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

CC_BY_LICENSE = """Creative Commons Attribution 4.0 International

Copyright (c) 2026 Song-BX

The raw numerical exports and solver-provenance records in this package are
licensed under the Creative Commons Attribution 4.0 International License
(CC BY 4.0). To view a copy of this license, visit
https://creativecommons.org/licenses/by/4.0/.
"""


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _readme(manifest: pd.DataFrame) -> str:
    raw_count = int((manifest["category"] == "raw_input").sum())
    reference_count = int((manifest["category"] == "reference_input").sum())
    code_count = int((manifest["category"] == "code").sum())
    test_count = int((manifest["category"] == "test").sum())
    return f"""# {RELEASE_NAME}

{RELEASE_DESCRIPTION} (`{RELEASE_TAG}`). This repository contains only the
source materials needed to reproduce the R1 analysis workflow:

- {raw_count} FLOW-3D CSV point-cloud snapshots in `raw data/`;
- {reference_count} compact reference inputs in `reference_data/` for deterministic
  baseline checks and the archived XZ support audit;
- supplied model-configuration and 300 W run-record files;
- {code_count} Python source files in `scripts/` and {test_count} tests in `tests/`;
- dependency, integrity, and reproduction instructions.

No generated audit CSV/JSON files, figures, manuscript tables, PDFs, reviewer
correspondence, author information, or submission assets are included. Running
the workflow generates outputs locally under `图/` and
`latex_restructure/figures/`; these paths are ignored by Git.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_release.py
python run_reproduction.py
python -m unittest discover -s tests
python verify_release.py --after-reproduction
```

See [REPRODUCTION.md](REPRODUCTION.md) for the fixed execution order and
[DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the export schema. The first
verification checks the source-only package. The second permits generated local
outputs while continuing to check the immutable data, code, and documentation.

## Scope

The scripts reproduce numerical-export diagnostics only. They do not validate
CFD physical fidelity, solver-native velocity gradients, a Marangoni mechanism,
or a continuous power response. Reconstructed Q-derived quantities remain
audit descriptors rather than comparative or physical-mechanism evidence.

## License and citation

Code is released under [MIT](LICENSE-CODE). Raw numerical exports and
solver-provenance records are released under [CC BY 4.0](LICENSE-DATA). See
[CITATION.cff](CITATION.cff), [RELEASE_CONTENTS.csv](RELEASE_CONTENTS.csv),
and [SHA256SUMS.txt](SHA256SUMS.txt).
"""


def _reproduction_guide() -> str:
    return """# Reproduction workflow

## Inputs

The workflow reads the 30 exported FLOW-3D CSV snapshots in `raw data/`, the
supplied configuration records (`Flow3D.md` and `Flow3D设置.txt`), and the
supplied 300 W run record (`running.md`). `reference_data/` contains only two
baseline-comparison tables, two historical XZ point tables, one historical XZ
summary table, and one prior-model-context record required by the code. It
contains no rendered figures, manuscript tables, or complete audit outputs.
The run record is retained as provenance for that case only; it is not evidence
of solver health for the other power cases.

## Execution order

`run_reproduction.py` runs the analysis modules in a fixed order:

1. export-structure and kNN/support audits;
2. reconstruction, conditioning, weighting, and temporal audits;
3. six-power, thermal-gradient, spatial-support, complementary-descriptor,
   velocity, and transferability audits;
4. deterministic claim classification; and
5. figure builders.

The scripts write derived outputs beneath `图/` and figures beneath
`latex_restructure/figures/`. These generated paths are intentionally absent
from the repository and may be removed and regenerated at any time.

## Integrity checks

Run `python verify_release.py` before reproduction to check the source-only
inventory, SHA-256 entries, input count, and exported-column schema. Run
`python verify_release.py --after-reproduction` afterwards; it checks the same
immutable material while allowing the documented generated-output directories.

## Computational boundary

This workflow reproduces the published post-processing sequence from the
provided numerical exports. It does not rerun FLOW-3D, add experimental data,
or establish physical validity of the simulated melt-pool fields.
"""


def _data_dictionary(headers: pd.DataFrame) -> str:
    schema_rows = "\n".join(
        f"- `{row.relative_path}`: {row.field_count} exported fields."
        for row in headers.itertuples(index=False)
    )
    return f"""# Data dictionary

## Raw FLOW-3D CSV inputs

The package contains 30 CSV snapshots: five time points (0.50, 0.55, 0.60,
0.65, and 0.70 s) for each of six laser powers (200, 250, 300, 350, 400, and
450 W). The six 0.70 s inputs are at the root of `raw data/`; the other 24 are
in `raw data/temporal_validation/`.

| Exported field | Standardized name | Description |
| --- | --- | --- |
| `Points_0`, `Points_1`, `Points_2` | `x`, `y`, `z` | Cartesian coordinates. |
| `Fraction Of Fluid` | `fof` | Field used for the interface proxy. |
| `Heat Flux Spatial Distribution` | `heat_flux` | Field used for audit masks. |
| `Temperature` | `T` | Exported temperature. |
| `Temperature Gradient At Tgrdout` | `gradT` | Exported scalar gradient magnitude. |
| `Velocity_0`, `Velocity_1`, `Velocity_2` | `u`, `v`, `w` | Velocity components. |
| `Velocity_Magnitude` | `V` | Exported velocity magnitude. |

## Per-file schema audit

{schema_rows}
"""


def _run_reproduction_script() -> str:
    return """\"\"\"Run the fixed R1 analysis and figure-generation workflow.\"\"\"

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "scripts" / "figures"
sys.dont_write_bytecode = True


def _run(label: str, command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\\n==> {label}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["R1_RELEASE_NON_TIFF"] = "1"
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) if not prior_pythonpath else str(ROOT) + os.pathsep + prior_pythonpath
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def main() -> None:
    module_steps = (
        ("export-structure audit", "scripts.export_diagnostics.run_analysis"),
        ("dense kNN/support audit", "scripts.robustness.run_analysis"),
        ("gradient-validation audit", "scripts.gradient_validation.run_analysis"),
        ("conditioning-cutoff audit", "scripts.conditioning_sensitivity.run_analysis"),
        ("distance-exponent audit", "scripts.weight_exponent_sensitivity.run_analysis"),
        ("temporal validation", "scripts.temporal_validation.run_analysis"),
        ("thermal-fidelity audit", "scripts.thermal_fidelity_audit.run_analysis"),
        ("six-power response audit", "scripts.power_response_audit.run_analysis"),
        ("thermal-gradient audit", "scripts.thermal_gradient_audit.run_analysis"),
        ("spatial-support audit", "scripts.spatial_support_audit.run"),
        ("complementary-tensor audit", "scripts.complementary_descriptor_audit.run_analysis"),
        ("velocity-extreme audit", "scripts.velocity_extreme_audit.run_analysis"),
        ("velocity-distribution overlap audit", "scripts.velocity_distribution_overlap_audit.run_analysis"),
        ("model-fidelity boundary", "scripts.model_fidelity_boundary.run_analysis"),
        ("cross-context scope audit", "scripts.transferability_scope_audit.run_analysis"),
        ("claim classification", "scripts.claim_classification.run_analysis"),
    )
    for label, module in module_steps:
        _run(label, [sys.executable, "-m", module])
    for filename in (
        "build_nature_figures.py",
        "build_export_diagnostics_figure.py",
        "build_knn_robustness_figure.py",
        "build_gradient_validation_figure.py",
        "build_temporal_validation_figure.py",
        "build_conditioning_sensitivity_figure.py",
        "build_weight_exponent_sensitivity_figure.py",
        "build_power_response_audit_figure.py",
        "build_thermal_gradient_audit_figure.py",
        "build_complementary_descriptor_audit_figure.py",
        "build_velocity_distribution_overlap_figure.py",
        "build_velocity_extreme_audit_figure.py",
        "build_thermal_fidelity_audit_figure.py",
    ):
        _run(f"figure builder: {filename}", [sys.executable, filename], cwd=FIGURE_DIR)


if __name__ == "__main__":
    main()
"""


def _verify_release_script() -> str:
    return """\"\"\"Verify the source-only R1 reproducibility package.\"\"\"

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
"""


def _citation() -> str:
    return f"""cff-version: 1.2.0
message: "If you use this R1 source package, please cite it."
title: "{RELEASE_NAME}"
version: "{RELEASE_TAG}"
date-released: 2026-08-01
authors:
  - family-names: "Song-BX"
license: "MIT (code); CC-BY-4.0 (raw numerical exports and provenance records)"
abstract: "Source data, code, and fixed reproduction workflow for the R1 revision."
"""


def _write_hash_sums(package_root: Path) -> None:
    lines = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(f"{sha256(path)}  {path.relative_to(package_root).as_posix()}")
    _write(package_root / "SHA256SUMS.txt", "\n".join(lines))


def write_release_documents(
    package_root: Path, manifest: pd.DataFrame, source_files: list[ReleaseFile]
) -> None:
    """Write source-only documentation after the allow-listed files are staged."""
    manifest[["relative_path", "category", "size_bytes", "sha256"]].to_csv(
        package_root / "RELEASE_CONTENTS.csv", index=False, encoding="utf-8"
    )
    headers = raw_csv_headers(source_files)
    _write(package_root / "README.md", _readme(manifest))
    _write(package_root / "REPRODUCTION.md", _reproduction_guide())
    _write(package_root / "DATA_DICTIONARY.md", _data_dictionary(headers))
    _write(package_root / "LICENSE-CODE", MIT_LICENSE)
    _write(package_root / "LICENSE-DATA", CC_BY_LICENSE)
    _write(package_root / "CITATION.cff", _citation())
    _write(
        package_root / ".gitignore",
        "__pycache__/\n*.py[cod]\n.venv/\n图/\nlatex_restructure/\n*.tiff\n*.tif\n",
    )
    _write(package_root / "run_reproduction.py", _run_reproduction_script())
    _write(package_root / "verify_release.py", _verify_release_script())
    _write_hash_sums(package_root)
