# Release notes

## Version 1.1.0 — controller-only follow-up (2026-09-02)

This release provides the reviewer-requested repair of post-TRACK
control under the delayed first-order plant. It does not alter the estimator,
active-acquisition planner, evidence-qualified gate, plant, horizon, or success
definitions. It adds a deterministic Smith-predictor-based tracker with a
causal reference governor, bumpless ACQUIRE-to-TRACK transfer, and command slew
limits. The earlier stateless proportional tracker remains the paired
reference.

The frozen 100-seed, four-arm qualification is complete and integrity-valid.
Terminal joint success changed from 38% to 95% without current and from 26% to
95% with ground-velocity-visible current. Tail80 joint success changed from 2%
to 91% and from 6% to 92%, respectively. There were no reference-only
discordant successes, no truth-audited unsafe TRACK events, and strong
reductions in post-TRACK saturation and command curvature.

The result did not satisfy the complete prespecified composite acceptance gate. In the
no-current condition, localization p95 increased from 0.9932 m to 1.5600 m;
the 0.5669-m difference exceeded the frozen 0.50-m margin by 0.0669 m. The
correct release interpretation is therefore
`VALID_COMPLETE__COMPOSITE_ACCEPTANCE_NOT_MET`, not an unqualified pass. The
original runner label `CONTROLLER_REPAIR_REJECT_OR_INVALID` is retained as
historical provenance and is not silently rewritten.

Added material includes:

- the three-file controller/runner implementation and frozen protocol;
- component, integration, export, and DOI-projection tests;
- all 400 row-level outcomes and a four-cell publication table;
- row-derived paired results and source hashes;
- path-sanitized contract, summary, decision, and additive semantic audit;
- a tested complete-campaign DOI projector with fixed 400/400/100/15
  JSON/trace/tape/source cardinalities.

The inherited reserved-range markers are also clarified: they verify that the
selected 51000--51099 cohort is disjoint from reserved range 50000--50999 and
that the runner excludes that range. They are not global proof that no earlier
external process ever inspected it.

This material is released in immutable GitHub tag `v1.1.0`. It is not
contained in tag `v1.0.0` or its version-specific Zenodo records. A validated
projector prepares the complete controller campaign for a future data-record
version without changing the frozen source artifacts. The final
archive-connected validation, with the optional full baseline-equivalence
smoke enabled and Numba unavailable, reported 275 passed tests and 64 passed
subtests.

## Version 1.0.0

This archival release accompanies the resubmission of *Active Follower
Localization Using Two-Leader Doppler Measurements for 3-D Autonomous
Underwater Vehicle Formation Tracking*. It freezes the manuscript-relevant
simulation code, protocols, tests, and row-level result tables.

## Release identifiers

- source: <https://github.com/lukisp2/two-leader-doppler-auv-localization>;
- version/tag: `1.0.0` / `v1.0.0`;
- release date: `2026-08-31`;
- software DOI: `10.5281/zenodo.22214022`;
- raw-data DOI: `10.5281/zenodo.22214031`.

The two DOI values identify separate, version-specific Zenodo records for the
software snapshot and the raw simulation archive.

## Included scientific components

- causal full-history Doppler estimator with retained separated solutions;
- moving-horizon, particle-filter, EKF, local-NLS, and global-search comparators;
- information-guided acquisition, prescribed S-turn, and planner-component
  ablations;
- evidence-qualified acquire-to-track switching and reacquisition;
- paired leader-link/acquisition experiment, estimator benchmark,
  planner-component ablation, integrated stress screen, and the resubmission
  command-response/current qualification;
- table regeneration, integrity, environment, and source/data-hash tools.
- a summary-independent, read-only dynamic-response postprocessor that verifies the frozen
  contract and traces before producing the eight-cell table, paired effects,
  interactions, and plot-ready data.

The compact per-scenario tables are committed in `data/tables/`. Larger trace
and measurement-history bundles are distributed as release/DOI assets and are
identified in `data/DATA_INVENTORY.csv`.

## Scope and limitations

This is simulation research code, not certified vehicle software. The release
preserves historical module names for traceability. The release-validation
environment is CPython 3.11.9 with the pinned pure-NumPy dependencies; optional
JIT stacks are outside this bit-level validation path. This environment
validates the current release and is not presented as the recorded runtime of
every historical campaign.

## 2026-08-31 final archive-connected test

- Interpreter: CPython 3.11.9
- NumPy: 2.3.3
- Gymnasium: 1.2.3
- Matplotlib: 3.10.6
- pytest: 9.1.1
- Numba discoverable: no
- Result: **241 passed, no tests skipped, 64 subtests passed**
- Runtime: 111.70 s with the released raw-data tree connected

The lightweight Git tree excludes raw numerical arrays. For this check,
`UUV_ARCHIVE_ROOT` pointed to the staged DOI `data/raw` tree, whose minimal
V18.1 trace/noise-tape fixture passed the bit-exact capture/replay integration
test. The same run also exercised the fail-closed DOI path projector and the
exact-tree source/data manifest tools added during release qualification.

## Descriptive-only stress export

The closed-loop stress rows behind manuscript Table 11 are included for
transparent reproduction of a negative audit result. Their frozen campaign
decision is `V35_INVALID`: 5 of 1200 optional bias-switch runs exceeded the
joint 2-s runtime gate (maximum 2.7017 s), although the primary controller
cells remained below 1.49 s. These rows and plots are descriptive and are not
claimed as a passed robustness qualification. The frozen V35 contract records
its historical software environment as CPython 3.12.8, NumPy 2.2.1, Gymnasium
1.2.3, and no Numba; this is distinct from the release-validation environment
used for the final source-tree test suite.

## Command-response qualification outcome

The prespecified 800-run factorial export passed its structural and
summary-independent integrity audits. Information-guided acquisition retained
a positive terminal advantage under delayed first-order command response, but
its Tail80 advantage was not demonstrated. Localization and evidence-qualified
TRACK release remained accurate, while the complete loop failed the frozen
sustained-tracking screens under the combined delay and first-order response
factor. Because the controller and individual response channels were not
separately ablated, the result does not assign sole causality to the tracker or
to one channel. The release preserves this negative qualification result
without retuning or excluding runs.
