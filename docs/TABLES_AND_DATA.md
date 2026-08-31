# Manuscript table-to-artifact map

The separate raw archive is identified by data DOI
`10.5281/zenodo.22214031`; its companion software release is identified by DOI
`10.5281/zenodo.22214022` at
<https://github.com/lukisp2/two-leader-doppler-auv-localization>.

Internal campaign identifiers below are provenance labels only. The manuscript
refers to methods and experiments descriptively. The item numbers below match
the frozen resubmission layout; the two row-level tables numbered 5--6 in the
reviewed version are Tables 6--7 in the resubmission because the modality
comparison was added as Table 1.

| Resubmission item | Scientific comparison | Code entry point | Compact source data | Raw archive or provenance bundle |
|---|---|---|---|---|
| Table 6 | One vs. two Doppler links crossed with prescribed vs. active acquisition | `code/run_v38_leader_source_ablation_replication.py`; `code/make_publication_table_source_policy_ablation.py` | `data/tables/leader_source_policy_episode_rows.csv` | `leader_source_ablation/` |
| Table 7 | Prespecified paired link/acquisition contrasts | `code/run_v38_leader_source_ablation_replication.py`; `code/make_publication_table_source_policy_contrasts.py` | `data/tables/leader_source_policy_episode_rows.csv` | `leader_source_ablation/` |
| Table 8 | Estimator benchmark: PF, EKF/NLS/global-history, and MHE | `scripts/prepare_v27_public_archive.py`; `code/run_v27_publication_baselines.py`; `code/run_v34_mhe60_baseline.py`; `scripts/export_estimator_benchmark.py` | `data/tables/estimator_checkpoint_rows.csv`; `data/tables/estimator_benchmark_summary.json` | `estimator_benchmark/v27_campaign/`, `estimator_benchmark/v34_campaign/`, `estimator_benchmark/measurement_history/`, `estimator_benchmark/legacy_pf_traces/`, and the minimal `estimator_benchmark/v18_replay_fixture/` used only by the capture/replay integration test |
| Table 9 | Planner-component ablation | `code/run_v39_planner_component_ablation.py`; `code/make_publication_table_planner_component_ablation.py` | `data/tables/planner_component_episode_rows.csv`; `data/tables/planner_component_summary.json` | Git provenance bundle: `data/provenance/planner_component_ablation/`; no separate raw archive is required because all 400 outcome rows are committed |
| Table 10 | Active vs. prescribed acquisition crossed with kinematic vs. delayed first-order execution and absent vs. ground-velocity-visible current | `code/run_v40_dynamic_plant_stress.py`; `scripts/postprocess_v40_qualification.py` | `data/tables/dynamic_plant_episode_rows.csv`; `data/tables/v40_publication_arms.csv`; `data/tables/v40_publication_results.json`; `data/tables/v40_interaction_plot.csv` | `dynamic_plant_stress/campaign/` |
| Table 11 | Descriptive eight-family closed-loop stress screen; source campaign invalid under its joint runtime gate | `code/run_v35_closed_loop_stress.py`; `code/make_publication_table_closed_loop_stress.py --allow-invalid-descriptive` | `data/tables/closed_loop_stress_episode_rows.csv`; `data/tables/closed_loop_stress_summary.json` | Git provenance bundle: `data/provenance/closed_loop_stress/`; no separate raw archive is required because all 1200 outcome rows are committed |

The release-validation stack is CPython 3.11.9 with NumPy 2.3.3. The sole
historical exception in this table map is the frozen V35 campaign behind Table
11, whose contract records CPython 3.12.8, NumPy 2.2.1, Gymnasium 1.2.3, and no
Numba. Use that historical stack for an exact software-environment replication;
a V35 run in the release-validation stack is a new replication.

## Manuscript figure reproduction

Figures 1 and 5 are black-and-white TikZ diagrams embedded in the manuscript
source and have no external figure-data dependency. The remaining manuscript
figures have the following stable release mapping. The paired-episode module is
listed because it supplies the validation/loading closure imported by the
mission-geometry generator; it is not a separate evidentiary result.

| Manuscript item | Output file | Code entry point | Released input |
|---|---|---|---|
| Figure 2 | `fig_mission_geometry_seed48698.pdf` | `code/make_publication_figure_mission_geometry.py`; closure: `code/make_publication_figure_paired_episode.py` | DOI component `data/raw/leader_source_ablation/` |
| Figure 3 | `figure3_doppler_degeneracies.pdf` | `code/make_publication_figure_doppler_geometry.py` | Data-free mathematical schematic |
| Figure 4 | `figure4_full_history_estimator.pdf` | `code/make_publication_figure_full_history_estimator.py` | Data-free explanatory schematic |
| Figure 6 | `fig_source_policy_ablation.pdf` | `code/make_publication_figure_source_policy_ablation.py` | `data/tables/leader_source_policy_episode_rows.csv` |
| Figure 7 | `fig_source_policy_dynamics.pdf` | `code/make_publication_figure_source_policy_dynamics.py` | DOI component `data/raw/leader_source_ablation/` |
| Figure 8 | `fig_estimator_history.pdf` | `code/make_publication_figures_v33.py --figure estimator-history` | `data/tables/estimator_benchmark_summary.json` |
| Figure 9 | `fig_policy_execution_current.pdf` | `scripts/plot_v40_qualification_figure.py` | `data/tables/dynamic_plant_episode_rows.csv`; `data/tables/v40_publication_results.json` |

Install the pinned figure dependency through `requirements-dev.txt` or the
`figures` project extra. For a headless host, force the noninteractive backend:

```bash
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figure_doppler_geometry.py
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figure_full_history_estimator.py
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figure_source_policy_ablation.py
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figures_v33.py --figure estimator-history
MPLBACKEND=Agg PYTHONPATH=code python scripts/plot_v40_qualification_figure.py
```

Those commands need only files committed to Git and write under
`results/figures/`. After extracting the DOI archive at the documented public
path, the two trace-derived figures are:

```bash
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figure_mission_geometry.py
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figure_source_policy_dynamics.py
```

The mission illustration uses the frozen descriptive seed 48698. It explains
geometry only; all quantitative claims use the complete paired cohort. The
source-policy dynamics figure aggregates all 100 paired seeds and validates
campaign completeness before plotting. `tests/test_publication_figure_scripts.py`
checks every CLI without data, renders both data-free schematics, validates the
mission-frame transform on a synthetic trace, and renders Figures 6 and 8--9
from the committed compact artifacts. Matplotlib is pinned at 3.10.6 so layout
changes do not silently enter the public workflow.

## Original frozen campaigns

- Tables 6--7: `experiments_v38_leader_source_ablation_replication_48600_48699` (600 paired arm rows).
- Table 8: `experiments_v27_publication_baselines_dev100` (4000 checkpoint-method JSON records) and `experiments_v34_mhe60_baseline_dev100` (500 MHE records). Complete reruns additionally require the measurement-history archive and the referenced resampling-guard traces.
- Table 9: `experiments_v39_planner_component_ablation_dev100` (400 arm rows).
- Table 10: `experiments_v40_dynamic_current_factorial_qualification100` (800 arm rows).
- Table 11: `experiments_v35_closed_loop_stress_dev100` (1200 arm rows),
  retained as descriptive evidence only. Its frozen decision is `V35_INVALID`
  because 5 optional bias-switch runs exceeded the campaign-wide 2-s runtime
  gate (maximum 2.7017 s); the primary controller cells remained below
  1.49 s. The frozen contract records CPython 3.12.8, NumPy 2.2.1, Gymnasium
  1.2.3, and no Numba.

## Lightweight provenance bundles for Tables 9 and 11

The complete row-level outcomes for these two tables are small enough for Git,
so no separate trace archive is required to verify the printed values. Their
control records are nevertheless released beside the rows:

- `data/provenance/planner_component_ablation/` contains 5 files and 92,900
  bytes: the campaign contract, campaign summary, decision, independent audit,
  and `provenance_export_manifest.json`;
- `data/provenance/closed_loop_stress/` contains 4 files and 117,238 bytes:
  the campaign contract, campaign summary, decision, and
  `provenance_export_manifest.json`.

The export manifest in each directory records the original SHA-256 and the
public exported SHA-256 for every copied campaign artifact. Only records that
contained the private source-workstation root differ. Their sole transformation
is a literal replacement of that prefix with `${FROZEN_SOURCE_ROOT}`; the
manifests state `scientific_values_changed: false`. Unchanged summaries and
decisions retain identical original and exported hashes. In particular, the
path-sanitized Table 11 records preserve `integrity_valid: false` and
`V35_INVALID` exactly.

## Dynamic-response qualification fast path

The resubmission qualification is a paired 2 x 2 x 2 campaign: two acquisition
rules, two command-execution models, two current conditions, 100 common seeds,
and 800 arm records. The compact eight-row table and all paired effects must be
generated from the raw episode records and traces with:

```bash
python scripts/postprocess_v40_qualification.py \
  data/raw/dynamic_plant_stress/campaign \
  --analysis-output-dir data/tables
```

The DOI campaign contains the approved path-sanitized environment metadata at
`control/environment_metadata.json`, so no private path or external override
is needed. To validate the archive without writing or replacing publication
outputs, omit `--analysis-output-dir` and redirect standard output outside the
repository, or to `/dev/null` when only the exit status is required.

`v40_publication_arms.csv` has exactly eight marginal rows.
`v40_publication_results.json` contains active-minus-fixed,
dynamic-minus-kinematic, current-minus-none, all two-factor interactions, the
three-factor interaction, qualification screens, and command-execution
semantics. `v40_interaction_plot.csv` is a long-format cell table for plotting.
The full audit remains in `v40_postprocessed_full.json`. None of these files is
valid evidence until the postprocessor reports `integrity_valid: true`.
The staged DOI projection passed this check with exactly 800 JSON records, 800
NPZ traces, 100 seeds, eight arms per seed, all 14 source hashes matching, and
decision `V40_QUALIFICATION_COMPLETE`.

## Estimator benchmark fast path and archive regeneration

The committed estimator CSV has one row per method, causal checkpoint, and
episode: nine methods by five checkpoints by 100 common histories, or 4500
rows. It preserves both `<7 m` and `<=7 m` labels, endpoint and initial errors,
the nominal local radius when defined, residual and runtime fields with an
explicit runtime definition, and hashes linking every row to its archived
source result and input/truth files.

The source-free manuscript cross-check is:

```bash
python scripts/export_estimator_benchmark.py --check-only
```

After extracting the DOI archive under `data/raw/estimator_benchmark/`, rebuild
the compact CSV and JSON summary with:

```bash
python scripts/export_estimator_benchmark.py \
  --v27-dir data/raw/estimator_benchmark/v27_campaign \
  --v34-dir data/raw/estimator_benchmark/v34_campaign
```

The row-derived check reproduces all 45 method--checkpoint cells, every
terminal median/p95/maximum printed in the manuscript, and the prespecified
50000-resample paired MHE-minus-full-history interval at 440 s. The
measurement-history archive (100 three-file episode directories) and the 100
historical particle-filter traces are retained in the DOI deposit because a
complete campaign rerun requires them; they are intentionally not duplicated
in Git. Their exact pre-deposit inventory is in
`data/DOI_ARCHIVE_COMPONENTS.md`.

Historical capture metadata preserves the original source-workstation trace
path as provenance. Before a full V27 rerun,
`scripts/prepare_v27_public_archive.py` creates a generated portable view and
changes only `provenance.trace_path`, after matching episode, seed, and the
recorded trace SHA-256. The frozen runner and all scientific arrays remain
unchanged; unit tests reject a substituted trace, an identity mismatch, or an
existing output directory.

## Required release rule

Every number printed in a manuscript table must be reproducible from a row-level compact file committed here or from a DOI-archived raw file named in `data/DATA_INVENTORY.csv`. Summary-only JSON files are insufficient as the sole public evidence.
