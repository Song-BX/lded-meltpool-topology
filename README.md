# lded-meltpool-topology

Raw data and Python code for reproducing the main results of our study on topology-based diagnostics and robustness evaluation of quasi-steady melt-pool point clouds in laser directed energy deposition (L-DED).

## Paper

**A Topology-Based Diagnostic and Robustness-Evaluation Framework for Quasi-Steady Melt Pool Point Clouds in Laser Directed Energy Deposition**

## Repository Structure

```text
lded-meltpool-topology/
├─ data/
│  ├─ raw/               # raw point-cloud data for each power condition
│  ├─ processed/         # processed data used for plotting and statistics
│  └─ cases.csv          # case list and basic metadata
├─ code/
│  ├─ preprocess.py      # deduplication and slice extraction
│  ├─ reconstruct_q.py   # WLS gradient reconstruction and Q calculation
│  ├─ statistics.py      # regional statistics, extreme-set geometry, robustness
│  ├─ plot_figures.py    # reproduce figures and tables
│  └─ run_all.py         # one-step reproduction pipeline
├─ figures/              # output figures
├─ tables/               # output tables
├─ requirements.txt
└─ README.md
