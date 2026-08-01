# Reproduction workflow

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
