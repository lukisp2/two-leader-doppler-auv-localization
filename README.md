# Two-Leader Doppler AUV Localization

Reference implementation and reproducibility materials for:

> **Active Follower Localization Using Two-Leader Doppler Measurements for 3-D Autonomous Underwater Vehicle Formation Tracking**

The repository contains the causal full-history Doppler estimator, retained-mode active acquisition planner, evidence-qualified acquire-to-track gate, delay-aware formation tracker, comparison estimators, stress generators, and scripts used to reproduce the manuscript tables and figures.

## Scope

The follower receives broadcast leader positions and velocities, measures one-way Doppler range rate on one or two leader links, and dead-reckons its own motion. During `ACQUIRE`, it retains competing initial-position hypotheses and commands bounded maneuvers that improve weak information directions and separate plausible modes. It enters `TRACK` only after independent residual, consistency, separation, and stability checks pass.

After an evidence-qualified TRACK release, the current resubmission branch uses
a deterministic delay-aware tracker with a Smith predictor, a causal reference
governor, bumpless transfer, and command slew limits. The former stateless
proportional tracker is retained as the paired reference arm.

This is simulation research code. It is **not** certified guidance, navigation, or control software.

## Release identifiers

- source repository: <https://github.com/lukisp2/two-leader-doppler-auv-localization>;
- current version and tag: `1.1.0` / `v1.1.0`;
- release date: `2026-09-02`;
- preceding software archive (`v1.0.0`): <https://doi.org/10.5281/zenodo.22214022>;
- preceding raw-data archive (`v1.0.0`): <https://doi.org/10.5281/zenodo.22214031>.

The immutable Zenodo records correspond to `v1.0.0`. The current `v1.1.0`
GitHub tag adds the delay-aware controller, the frozen controller-only
qualification protocol, 400 row-level outcomes, compact paired results,
provenance, tests, and Table 10/Fig. 9 regeneration. Those additions are
public in Git but are not claimed to be present in the earlier DOI records.
A validated projector is provided for a future versioned raw-trace deposit.

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

The final archive-connected `v1.1.0` validation used the environment above,
disabled the optional Numba import, enabled the full baseline-equivalence
smoke, and reported **275 passed tests and 64 passed subtests**.

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
The paired post-TRACK controller repair, its one failed composite acceptance
condition, and the corrected decision/seed-marker semantics are documented in
[docs/V41_CONTROLLER_REPAIR_AUDIT.md](docs/V41_CONTROLLER_REPAIR_AUDIT.md).

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

The controller-repair update follows the same boundary through
`scripts/project_v41_doi_campaign.py`. Git contains all 400 row-level outcomes,
four compact publication cells, a row-derived results document, and a
path-sanitized provenance bundle. Its complete JSON/NPZ campaign remains local
until it is assigned to the next DOI data version.

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

The controller-repair qualification is integrity-valid but did not meet every
prespecified acceptance condition. Terminal success increased from 38% to 95%
without current and from 26% to 95% with visible current; Tail80 success rose
from 2% to 91% and from 6% to 92%. The no-current terminal-localization p95
difference was 0.5669 m, exceeding the frozen 0.50-m non-inferiority margin by
0.0669 m. The frozen ambiguous label is preserved, while the public audit
records the precise interpretation
`VALID_COMPLETE__COMPOSITE_ACCEPTANCE_NOT_MET`.

## Citation

Use the metadata in [CITATION.cff](CITATION.cff) for the current GitHub tag
`v1.1.0`. DOI `10.5281/zenodo.22214022` identifies only the preceding
`v1.0.0` software snapshot, and DOI `10.5281/zenodo.22214031` identifies its
separate raw-data archive; neither DOI contains the controller addendum. The
journal DOI will be added after article publication.

## License

Source code is released under the [BSD 3-Clause License](LICENSE). Released
simulation data use [CC BY 4.0](data/LICENSE.md), unless an individual artifact
states otherwise; DOI-archive metadata must repeat the applicable license.
