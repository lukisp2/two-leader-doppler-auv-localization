# DOI archive component inventories

These components form raw-data version `1.0.0`, DOI
`10.5281/zenodo.22214031`. The companion software record has DOI
`10.5281/zenodo.22214022` and source repository
<https://github.com/lukisp2/two-leader-doppler-auv-localization>.

## Estimator benchmark

The large Table 8 inputs are not duplicated in Git. The following inventory
was taken from the frozen source tree before deposition; `.DS_Store` files are
excluded. Paths in the second column are relative to the private source root
`UUV_ART2/`, while destination paths are the public archive layout.

| DOI destination | Frozen source-relative path | Files | Bytes | Contents |
|---|---|---:|---:|---|
| `data/raw/estimator_benchmark/v27_campaign/` | `experiments_v27_publication_baselines_dev100/` | 8011 | 12061437 | 4000 scored wrappers, 4000 unscored outputs, contract/control files, summary, audit, and decision |
| `data/raw/estimator_benchmark/v34_campaign/` | `experiments_v34_mhe60_baseline_dev100/` | 1014 | 7375279 | 500 scored wrappers, 500 unscored MHE outputs, contract/control files, summary, audit, test report, and decision |
| `data/raw/estimator_benchmark/measurement_history/` | `experiments_v19_observability_estimator_benchmark_dev100/measurement_archive/` | 300 | 7222970 | 100 triplets of causal online inputs, separately held truth labels, and capture metadata |
| `data/raw/estimator_benchmark/legacy_pf_traces/` | `experiments_v18_1_guard_ablation_dev_3seed/evaluations/dev100_seed_28001_range_45000_45099/traces_npz/deterministic_greedy_grid/` | 100 | 14361046 | Historical 1024-particle-filter estimates and covariance traces used by the legacy arm |
| `data/raw/estimator_benchmark/v18_replay_fixture/` | Sanitized frozen metadata plus episode-0000 trace and matching noise tape from the V18.1 evaluation | 3 | 320418 | Minimal fixture for the bit-exact capture/replay integration test; the trace is intentionally duplicated from `legacy_pf_traces/` so the fixture has the directory layout required by the frozen replay API |

The staged estimator component therefore contains 9428 files and 41341150
bytes before archive-container overhead. The release includes a final
per-file SHA-256 manifest at the archive root. The compact CSV records source-result,
online-input, truth-label, unscored-output, and campaign-contract hashes, so the
deposited objects can be joined back to every released row.

Noise tapes used to create the already frozen measurement archive are upstream
simulation lineage, not inputs to the released estimator rerun. The deposit
therefore includes only the single tape in `v18_replay_fixture/` needed to run
the bit-exact capture/replay integration test; it does not claim to reproduce
all 100 capture-stage simulations.

## Paired leader-source/acquisition campaign

Figures 2 and 7 require recorded trajectories from the frozen exact
replication. The whole campaign is staged once, rather than copying a
single visually selected trajectory into Git. This also keeps the plotted
trace connected to the Tables 6--7 audit and permits regeneration of the
full-cohort dynamics figure.

| DOI destination | Frozen source-relative path | Files | Bytes | Contents |
|---|---|---:|---:|---|
| `data/raw/leader_source_ablation/` | `experiments_v38_leader_source_ablation_replication_48600_48699/` | 1335 | 91582872 | 600 episode-result JSON files, 600 trajectory NPZ files, 100 shared noise tapes, 29 control/source-snapshot files, and 6 root-level summary/audit artifacts |

`.DS_Store` files are excluded. The component contains the complete 100-seed,
six-arm campaign; the mission-geometry figure uses seed 48698 only as a
descriptive illustration, whereas the source-policy dynamics figure reads all
600 traces. The final archive-level SHA-256 manifest covers this component
after its metadata freeze.

## Dynamic-response/current qualification

The completed 100-seed, eight-arm factorial campaign is staged as a portable
projection rather than as a direct copy containing a private workstation
path.

| DOI destination | Frozen source-relative path | Files | Bytes | Contents |
|---|---|---:|---:|---|
| `data/raw/dynamic_plant_stress/campaign/` | `experiments_v40_dynamic_current_factorial_qualification100/` | 1721 | 135379146 | 800 episode-result JSON files, 800 trajectory NPZ files, 100 shared noise tapes, 14 frozen source-snapshot files, frozen control/summary records, approved public environment metadata, and the projection manifest |

The private campaign has 1719 files. The public projection adds
`control/environment_metadata.json` and `public_projection_manifest.json`.
The only modified campaign file is `control/campaign_contract.json`, where the
inert private `environment_metadata_path` is replaced by
`${CAMPAIGN_ROOT}/control/environment_metadata.json`. The manifest covers all
1720 other exported objects with original and exported SHA-256 values and
states `scientific_values_changed: false`. All 800 JSON outcomes, all 900 NPZ
files (including noise tapes), and the complete source snapshot are
byte-for-byte unchanged. The public summary-independent postprocessor accepts
the projection without an external metadata override and returns
`integrity_valid: true` and `V40_QUALIFICATION_COMPLETE`.

## Planner-component and closed-loop stress outcomes

No separate raw-trace DOI component is required to verify manuscript Tables 9
and 11. Their complete per-scenario outcome records are committed directly as
`data/tables/planner_component_episode_rows.csv` (400 rows) and
`data/tables/closed_loop_stress_episode_rows.csv` (1200 rows), together with
their summaries and row-derived verification/generation code. Table 11
must retain its explicit descriptive-only status because the source campaign
failed the prespecified joint runtime gate. A future deposit may include the
larger traces for provenance, but the release does not promise or depend on an
uninventoried `planner_component_ablation/` or `closed_loop_stress/` archive.
