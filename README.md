# lded-meltpool-topology-r1-reproducibility

Source data, code, and reproduction workflow for the R1 revision (`r1-review-2026-07-31`). This repository contains only the
source materials needed to reproduce the R1 analysis workflow:

- 30 FLOW-3D CSV point-cloud snapshots in `raw data/`;
- 7 compact reference inputs in `reference_data/` for deterministic
  baseline checks and the archived XZ support audit;
- supplied model-configuration and 300 W run-record files;
- 166 Python source files in `scripts/` and 11 tests in `tests/`;
- dependency, integrity, and reproduction instructions.

No generated audit CSV/JSON files, figures, manuscript tables, PDFs, reviewer
correspondence, author information, or submission assets are included. Running
the workflow generates outputs locally under `图/` and
`latex_restructure/figures/`; these paths are ignored by Git.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
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
