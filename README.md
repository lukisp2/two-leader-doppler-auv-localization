# Two-Leader Doppler AUV Localization

Reference implementation and reproducibility materials for:

> **Active Follower Localization Using Two-Leader Doppler Measurements for 3-D Autonomous Underwater Vehicle Formation Tracking**

The repository contains the causal full-history Doppler estimator, retained-mode active acquisition planner, evidence-qualified acquire-to-track gate, formation controller, comparison estimators, stress generators, and scripts used to reproduce the manuscript tables and figures.

## Scope

The follower receives broadcast leader positions and velocities, measures one-way Doppler range rate on one or two leader links, and dead-reckons its own motion. During `ACQUIRE`, it retains competing initial-position hypotheses and commands bounded maneuvers that improve weak information directions and separate plausible modes. It enters `TRACK` only after independent residual, consistency, separation, and stability checks pass.

This is simulation research code. It is **not** certified guidance, navigation, or control software.

## Release identifiers

- source repository: <https://github.com/lukisp2/two-leader-doppler-auv-localization>;
- version and tag: `1.0.0` / `v1.0.0`;
- release date: `2026-08-31`;
- software DOI: <https://doi.org/10.5281/zenodo.22214022>;
- raw-data DOI: <https://doi.org/10.5281/zenodo.22214031>.

The version-specific Zenodo records correspond respectively to the tagged
software source and to the separately licensed raw simulation archive.

## Release-validation environment

- CPython 3.11.9
- NumPy 2.3.3
- Gymnasium 1.2.3
- Matplotlib 3.10.6 for publication-figure regeneration
- no Numba/JIT dependency

This stack validates the released source tree, tests, compact table and figure
generators, and the documented campaign reruns except the historical V35
closed-loop stress campaign behind descriptive Table 11. That frozen campaign
was executed with CPython 3.12.8, NumPy 2.2.1, Gymnasium 1.2.3, and no Numba,
as recorded in its released campaign contract. Running V35 under the
release-validation stack is a new replication, not an exact reconstruction of
the historical software environment.

Both documented stacks exclude Numba. Installing it can change floating-point
execution order and prevent bit-for-bit reproduction of paths that are defined
as bit-exact.

## Repository layout

```text
code/                 flat source tree; historical module names are preserved
tests/                standard-library unittest suite
protocols/            frozen experiment protocols
data/                  compact table inputs and archive pointers
scripts/               table exporters, V40 postprocessor, and release checks
docs/                  table mapping, provenance, and reproduction guide
results/               generated summaries (not committed by default)
```

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Check the exact interpreter and package versions:

```bash
python scripts/verify_environment.py
```

## Tests

The suite contains standard-library `unittest` cases and a small number of
pytest-native parametrized checks. Run all of them through pytest:

```bash
PYTHONPATH=code python -m pytest tests -v
```

Verify the lightweight source repository from its root:

```bash
python scripts/verify_sha256_manifest.py SHA256SUMS
```

The separate DOI dataset has its own manifest. After extracting
`two-leader-doppler-auv-data-v1.0.0`, verify it independently and then expose
its `data/raw/` tree at the same path in this repository (copying or a symbolic
link are both acceptable):

```bash
python /path/to/repository/scripts/verify_sha256_manifest.py \
  /path/to/two-leader-doppler-auv-data-v1.0.0/SHA256SUMS
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for campaign commands and [docs/TABLES_AND_DATA.md](docs/TABLES_AND_DATA.md) for the complete manuscript table/figure-to-artifact map and headless rendering commands.
The archival publication sequence is recorded in
[docs/RELEASE_WORKFLOW.md](docs/RELEASE_WORKFLOW.md).
The summary-independent dynamic-response audit and compact eight-cell export
are documented in
[docs/V40_QUALIFICATION_POSTPROCESSOR.md](docs/V40_QUALIFICATION_POSTPROCESSOR.md).

## Data policy

Compact per-scenario CSV/JSON summaries used by the manuscript belong under
`data/tables/` and are suitable for Git. Small path-sanitized campaign
contracts, decisions, and audits for the planner and closed-loop stress
experiments are kept under `data/provenance/`; their export manifests retain
both original and public SHA-256 values. Large episode JSON files, measurement
histories, noise tapes, and trajectory NPZ files belong in the separately
versioned DOI dataset. The file-level inventory and expected destinations are
recorded in `data/DATA_INVENTORY.csv`. The repository `SHA256SUMS` covers the
Git release, while the dataset's own `SHA256SUMS` covers every external raw
artifact; neither manifest is presented as covering the other deposit.

The dynamic-response campaign is deposited through the tested
`scripts/project_v40_doi_campaign.py` portability boundary: only its inert
environment-metadata path changes, while all recorded outcomes, trajectories,
noise tapes, and frozen source files remain byte-for-byte identical.

The closed-loop stress data used descriptively in manuscript Table 11
come from a campaign that was **invalid under its prespecified joint runtime
gate**: 5 of 1200 optional bias-switch runs exceeded the 2-s decision limit
(maximum 2.7017 s). The primary controller cells remained below 1.49 s, but
the frozen rule was campaign-wide. The released rows are retained to expose
that negative audit result; they are not presented as a valid robustness
qualification. No closed-loop-stress figure is used in the resubmission.
Regeneration of the archived diagnostic plot requires the explicit
`--allow-invalid-descriptive` acknowledgement documented in
`docs/TABLES_AND_DATA.md`.

## Citation

Use the metadata in [CITATION.cff](CITATION.cff). The software release is
identified by DOI `10.5281/zenodo.22214022`, and its separately archived raw
data by DOI `10.5281/zenodo.22214031`. The journal DOI will be added after
article publication.

## License

Source code is released under the [BSD 3-Clause License](LICENSE). Released
simulation data use [CC BY 4.0](data/LICENSE.md), unless an individual artifact
states otherwise; DOI-archive metadata must repeat the applicable license.
