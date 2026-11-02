# Two-Leader Doppler AUV Localization

Simulation code and data for:

> **Active Follower Localization Using Two-Leader Doppler Measurements for 3-D Autonomous Underwater Vehicle Formation Tracking**

This project studies how an underwater follower can estimate its position from
Doppler signals transmitted by two moving leaders and then join their formation.
The follower combines the measurement history with dead reckoning. While its
position remains uncertain, it chooses maneuvers that help distinguish competing
position estimates. It starts formation tracking after the estimates pass the
specified consistency checks.

The repository includes the localization method, acquisition planner, switching
logic, delay-aware tracking controller, comparison methods, and scripts for the
paper's experiments and figures. The proportional controller is retained for
comparison. These are simulation models, not software validated on a vehicle.

## Code and data

- [Version 1.2.0: source code and difficult-start data](https://github.com/lukisp2/two-leader-doppler-auv-localization/releases/tag/v1.2.0)
- [Original software archive, version 1.0.0](https://doi.org/10.5281/zenodo.22214022)
- [Original raw-data archive, version 1.0.0](https://doi.org/10.5281/zenodo.22214031)

The latest release compares the two controllers in 50 scenarios with large
initial position and heading errors, without changing their settings. It
includes all 100 simulation results, trajectories, and noise sequences.
The delay-aware controller achieved terminal success in 48 of 50 scenarios
and sustained tracking, as measured by Tail80, in 46 of 50.
[Experiment details](docs/IET_REVISION_20261102.md) explain the scenario selection,
both controllers' results, failures, and reproduction commands.

The repository also contains per-run results from the earlier 400-run controller
comparison. Its full trajectory archive is not included in the new download.
Neither later experiment is included in the original Zenodo archives.
See [data coverage](data/DATA_INVENTORY.csv) for individual datasets.

## Installation

The release was checked with Python 3.11.9, NumPy 2.3.3, Gymnasium 1.2.3, and
Matplotlib 3.10.6. Use the supplied dependencies without optional Numba acceleration
when reproducing the results.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/verify_environment.py
```

Run the tests from the repository root:

```bash
PYTHONPATH=code python -m pytest tests -v
```

Some tests require the original raw-data archive. Extract it and make its
`data/raw/` directory available at the same location in this repository, either
by copying it or using a symbolic link. The earlier closed-loop measurement-error
experiment used Python 3.12.8 and NumPy 2.2.1. The recorded environments and data
setup are described in the [reproduction guide](docs/REPRODUCIBILITY.md).

## Reproduce the experiments

- [Experiment commands and data setup](docs/REPRODUCIBILITY.md)
- [Inputs and scripts for the tables and figures](docs/TABLES_AND_DATA.md)
- [Large initial position and heading errors](docs/IET_REVISION_20261102.md)
- [Delayed vehicle response and current](docs/V40_QUALIFICATION_POSTPROCESSOR.md)
- [Proportional and delay-aware tracking](docs/V41_CONTROLLER_REPAIR_AUDIT.md)

The experiments include negative results. Some measurement errors cause
localization or tracking to fail. The delay-aware controller improves tracking
substantially but does not meet every predefined acceptance criterion: one
localization-error increase exceeds the allowed limit. These limitations are
reported alongside the improvements.

## Repository layout

```text
code/       simulation, estimation, control, and experiment runners
tests/      automated tests
protocols/  experiment settings and evaluation rules
data/       per-run results, table inputs, and links to raw data
scripts/    analysis, export, and file-checking tools
docs/       reproduction instructions and experiment details
results/    locally generated results (not committed by default)
```

Historical version numbers in module names are retained so published commands
continue to work. Small result tables are stored in Git; larger measurement and
trajectory files are distributed through the releases and Zenodo.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite source version 1.2.0. The two Zenodo DOIs
above identify only version 1.0.0. No new DOI has been assigned to the later
experiments.

Source code uses the [BSD 3-Clause License](LICENSE). Simulation data use
[CC BY 4.0](data/LICENSE.md), unless an individual file states otherwise.
