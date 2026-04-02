# lded-meltpool-topology

Repository for the raw data and analysis code accompanying our study on topology-based diagnostics and robustness evaluation of quasi-steady melt-pool point clouds in laser directed energy deposition (L-DED).

## Overview

This repository contains the raw data, intermediate processed data, analysis scripts, and figure-generation code used in the study:

**A Topology-Based Diagnostic and Robustness-Evaluation Framework for Quasi-Steady Melt Pool Point Clouds in Laser Directed Energy Deposition**

The purpose of this repository is to support transparent and reproducible analysis of quasi-steady melt-pool point clouds exported from Flow3D simulations under different laser-power conditions. The workflow implemented here covers point-cloud deduplication, local weighted least-squares reconstruction of the velocity-gradient tensor, Q-criterion-based topological diagnostics, region-wise statistics, extreme-point-set geometry analysis, and robustness evaluation with respect to threshold and neighborhood size.

## Main Contents

The repository is organized around the following components:

- raw point-cloud data exported from Flow3D simulations
- metadata describing simulation settings and case information
- preprocessing scripts for coordinate deduplication and cross-sectional extraction
- gradient-reconstruction and topology-analysis modules
- statistical analysis scripts for region-wise metrics and robustness summaries
- plotting scripts for reproducing the figures and tables reported in the paper

## Repository Structure

```text
lded-meltpool-topology/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ metadata/
├─ src/
│  ├─ io/
│  ├─ preprocessing/
│  ├─ reconstruction/
│  ├─ topology/
│  ├─ statistics/
│  ├─ plotting/
│  ├─ pipelines/
│  └─ utils/
├─ configs/
├─ notebooks/
├─ outputs/
│  ├─ figures/
│  ├─ tables/
│  └─ logs/
├─ tests/
└─ paper/
