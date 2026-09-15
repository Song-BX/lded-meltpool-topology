# Data dictionary

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

- `raw data/0.7s_200W.csv`: 11 exported fields.
- `raw data/0.7s_250W.csv`: 11 exported fields.
- `raw data/0.7s_300W.csv`: 11 exported fields.
- `raw data/0.7s_350W.csv`: 11 exported fields.
- `raw data/0.7s_400W.csv`: 11 exported fields.
- `raw data/0.7s_450W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.55s_200W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.55s_250W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.55s_300W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.55s_350W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.55s_400W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.55s_450W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.5s_200W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.5s_250W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.5s_300W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.5s_350W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.5s_400W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.5s_450W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.65s_200W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.65s_250W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.65s_300W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.65s_350W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.65s_400W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.65s_450W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.6s_200W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.6s_250W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.6s_300W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.6s_350W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.6s_400W.csv`: 11 exported fields.
- `raw data/temporal_validation/0.6s_450W.csv`: 11 exported fields.
