# Reproducibility guide

Version `1.0.0` uses source repository
<https://github.com/lukisp2/two-leader-doppler-auv-localization>, software DOI
`10.5281/zenodo.22214022`, and raw-data DOI `10.5281/zenodo.22214031`.

## 1. Create the release-validation environment

Use CPython 3.11.9 exactly. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/verify_environment.py
```

This is the authoritative environment for the released test suite, compact
table and figure generators, and the campaign reruns below except V35. It is
pure NumPy; do not install Numba when checking bit-level archived values.

The descriptive Table 11 rows are a historical exception. Their frozen V35
contract records CPython 3.12.8, NumPy 2.2.1, Gymnasium 1.2.3, and no Numba.
Use that separate stack when attempting an exact software-environment
replication of V35. Running V35 under CPython 3.11.9 / NumPy 2.3.3 is a new
replication and must not be described as a bit-exact reconstruction of the
archived campaign.

## 2. Verify the source and data snapshots

The source release and raw-data deposit are independently versioned and have
separate manifests. Verification checks both the listed hashes and the exact
logical payload tree by default. Thus an unlisted result, an old preliminary
manifest, `.DS_Store`, `._*`, or `__MACOSX` content causes failure. Fixed local
checkout/runtime directories such as `.git`, `.venv`, and `__pycache__` are
not release payload and are ignored. From the source-repository root, verify
the repository:

```bash
python scripts/verify_sha256_manifest.py SHA256SUMS
```

Download and extract the data record as
`two-leader-doppler-auv-data-v1.0.0/`, then verify its manifest with the same
script:

```bash
python /path/to/repository/scripts/verify_sha256_manifest.py \
  /path/to/two-leader-doppler-auv-data-v1.0.0/SHA256SUMS
```

After verification, copy or symbolically link the extracted dataset's
`data/raw/` directory to `data/raw/` in the source repository. All subsequent
commands assume that combined runtime view. The repository manifest does not
claim to cover external raw files, and the dataset manifest does not claim to
cover the source checkout.

The repository manifest must be created only after source, protocols, compact
data, and archive pointers are frozen and committed. In a Git top-level tree,
the default generator requires clean tracked content and inventories only
`git ls-files`; ignored and untracked `results/` output cannot enter the
manifest:

```bash
python scripts/generate_sha256_manifest.py --output SHA256SUMS
```

For the separate, non-Git data tree, select the filesystem inventory
explicitly after the public archive has been sanitized and frozen:

```bash
python /path/to/repository/scripts/generate_sha256_manifest.py \
  --root /path/to/two-leader-doppler-auv-data-v1.0.0 \
  --inventory filesystem --output SHA256SUMS
```

Use `--no-exact-tree` with the verifier only for a diagnostic hash-only check;
it is not an archival acceptance check.

## 3. Run tests

```bash
PYTHONPATH=code python -m pytest tests -v
```

If the archive is stored elsewhere, point the replay integration test to its
`data/raw` equivalent with `UUV_ARCHIVE_ROOT=/path/to/data/raw`.

The complete archive-connected suite was rerun under the exact
release-validation stack: CPython 3.11.9, NumPy 2.3.3, Gymnasium 1.2.3,
Matplotlib 3.10.6, and pytest 9.1.1, with no Numba. With `UUV_ARCHIVE_ROOT`
pointed to the DOI data tree, the final post-qualification run reported 241
passed tests, no skipped tests, and 64 passed subtests in 111.70 s. This
included the bit-exact V18.1 capture/replay integration test, fail-closed path
projection checks, and exact-tree manifest checks. A failure caused solely by
installing an optional JIT stack is not a release-validation result.

## 4. Reproduce manuscript tables

Campaign entry points are listed in `docs/TABLES_AND_DATA.md`. Invoke their `--help` before a full run because each runner records its own frozen contract and seed policy:

```bash
PYTHONPATH=code python code/run_v38_leader_source_ablation_replication.py --help
PYTHONPATH=code python code/run_v27_publication_baselines.py --help
PYTHONPATH=code python code/run_v34_mhe60_baseline.py --help
PYTHONPATH=code python code/run_v39_planner_component_ablation.py --help
PYTHONPATH=code python code/run_v35_closed_loop_stress.py --help
PYTHONPATH=code python code/run_v40_dynamic_plant_stress.py --help
```

For manuscript verification, regenerating tables from the released row-level files is the fast path. A complete campaign rerun is the strong path and may take several hours.

The canonical full-run commands are below. They use the frozen cohort and
publication settings; do not substitute the sealed final range. Except for the
explicit V35 block, they use the release-validation environment from Section
1. Wall times are the recorded order of magnitude on the reference Mac and are
not acceptance criteria.

```bash
# Tables 6--7: 100 seeds 48600--48699 x 6 arms = 600 rows; about 91 min.
PYTHONPATH=code python code/run_v38_leader_source_ablation_replication.py \
  --output-dir results/campaigns/leader_source_ablation \
  --seed-start 48600 --progress-every 2

# Build a portable view for the byte-frozen V27 runner. Historical metadata
# contains the source-workstation trace path; this command replaces only that
# path after matching episode, seed, and the recorded SHA-256 against the
# released legacy trace. Scientific arrays are hard-linked or copied unchanged.
python scripts/prepare_v27_public_archive.py \
  --archive-dir data/raw/estimator_benchmark/measurement_history \
  --legacy-trace-dir data/raw/estimator_benchmark/legacy_pf_traces \
  --output-dir results/portable_inputs/v27_measurement_history

# Table 8 core comparators: 100 archived histories x 8 methods = 800
# episode-method results (4000 method-checkpoint rows); about 6 min.
PYTHONPATH=code python code/run_v27_publication_baselines.py \
  --archive-dir results/portable_inputs/v27_measurement_history \
  --output-dir results/campaigns/estimator_v27 --progress-every 1

# Required two-history audited FEJ-MHE smoke.  The output directory is
# intentionally omitted because the frozen full runner verifies this exact
# source-root prerequisite before opening the 100-history cohort.
PYTHONPATH=code python code/run_v34_mhe60_baseline.py --smoke \
  --archive-dir results/portable_inputs/v27_measurement_history \
  --reference-v27-dir results/campaigns/estimator_v27 \
  --progress-every 1

# Table 8 FEJ-MHE: 100 histories x 5 checkpoints = 500 rows; about 50 s on
# the documented release-validation environment.
PYTHONPATH=code python code/run_v34_mhe60_baseline.py \
  --archive-dir results/portable_inputs/v27_measurement_history \
  --reference-v27-dir results/campaigns/estimator_v27 \
  --output-dir results/campaigns/estimator_v34_mhe --progress-every 1

# Table 9: 100 seeds 48800--48899 x 4 arms = 400 rows; about 70 min.
PYTHONPATH=code python code/run_v39_planner_component_ablation.py \
  --output-dir results/campaigns/planner_components \
  --episodes 100 --episode-start 0 \
  --coarse-candidates 4096 --coarse-sweeps 2 --local-starts 48 \
  --progress-every 1
```

### Historical V35 environment exception

The V35 protocol's `Frozen environment` heading describes the simulated
mission configuration. The campaign contract is authoritative for the runtime
packages. To reproduce its historical software stack, create a separate
CPython 3.12.8 environment outside the repository and install NumPy 2.2.1 and
Gymnasium 1.2.3 without Numba:

```bash
python3.12 -m venv ../two-leader-v35-py3128
source ../two-leader-v35-py3128/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==2.2.1 gymnasium==1.2.3
python -c "import sys; assert sys.version_info[:3] == (3, 12, 8), sys.version"
python -c "import numpy; assert numpy.__version__ == '2.2.1'"
python -c "import gymnasium; assert gymnasium.__version__ == '1.2.3'"

# Table 11 source: 100 seeds 48100--48199, 1200 rows; about 154 min.
# The archived source campaign failed its joint runtime gate and is
# descriptive only even though the numerical rows are fully released.
PYTHONPATH=code python code/run_v35_closed_loop_stress.py \
  --output-dir results/campaigns/closed_loop_stress \
  --episodes 100 --episode-start 0 \
  --coarse-candidates 4096 --coarse-sweeps 2 --local-starts 48 \
  --progress-every 4
deactivate
```

The released rows, hashes, and `V35_INVALID` decision remain the authoritative
byte-level record. Matching the software versions alone does not guarantee
bit-identical floating-point output on different operating systems or
hardware.

Reactivate and verify the release-validation environment before continuing:

```bash
source .venv/bin/activate
python scripts/verify_environment.py

# Table 10 and Figure 9: seeds 49900--49999 x 8 arms = 800 rows;
# approximately three hours on the reference Mac.
PYTHONPATH=code python code/run_v40_dynamic_plant_stress.py \
  --output-dir results/campaigns/dynamic_current_qualification \
  --episodes 100 --episode-start 0 \
  --coarse-candidates 4096 --coarse-sweeps 2 --local-starts 48 \
  --progress-every 4
```

Each runner writes its own contract, source hashes, seed list, checkpointed
row files, independent-audit inputs, and final decision. `--resume` continues
only a contract-compatible partial output; it does not relax a seed or source
check.

Table 8 has a dedicated source-free cross-check and archive exporter:

```bash
python scripts/export_estimator_benchmark.py --check-only
python scripts/export_estimator_benchmark.py \
  --v27-dir data/raw/estimator_benchmark/v27_campaign \
  --v34-dir data/raw/estimator_benchmark/v34_campaign
```

The second command intentionally requires the archived, independently audited
campaign directories. It does not silently substitute smoke data or regenerate
measurements.

## 5. Reproduce manuscript figures

Matplotlib 3.10.6 is pinned by both `requirements-dev.txt` and the `figures`
project extra. The data-free and compact-data figures can be rebuilt on a
headless host with:

```bash
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figure_doppler_geometry.py
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figure_full_history_estimator.py
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figure_source_policy_ablation.py
MPLBACKEND=Agg PYTHONPATH=code python code/make_publication_figures_v33.py --figure estimator-history
MPLBACKEND=Agg PYTHONPATH=code python scripts/plot_v40_qualification_figure.py
```

The corresponding row-derived table checks are:

```bash
PYTHONPATH=code python code/make_publication_table_source_policy_ablation.py
PYTHONPATH=code python code/make_publication_table_source_policy_contrasts.py
PYTHONPATH=code python code/make_publication_table_planner_component_ablation.py
PYTHONPATH=code python code/make_publication_table_closed_loop_stress.py --allow-invalid-descriptive
```

After the `leader_source_ablation/` DOI component is extracted under
`data/raw/`, the recorded mission-geometry and full-cohort dynamics figures
also run without private paths. The exact figure-to-input map and commands are
in `docs/TABLES_AND_DATA.md`.

The closed-loop-stress diagnostic plot is no longer a manuscript figure. It
can still be regenerated with
`code/make_publication_figures_v33.py --figure closed-loop-stress
--allow-invalid-descriptive`. The explicit acknowledgement remains required
because 5 of 1200 optional bias-switch runs exceeded 2 s (maximum 2.7017 s).
Its primary cells were below 1.49 s, but Table 11 remains descriptive and must
not be cited as a passed robustness qualification.

## 6. Audit and export the dynamic-response qualification

After extracting the staged DOI projection to
`data/raw/dynamic_plant_stress/campaign/`, run the independent, summary-free
postprocessor:

```bash
python scripts/postprocess_v40_qualification.py \
  data/raw/dynamic_plant_stress/campaign \
  --analysis-output-dir data/tables
```

The archive already contains the approved path-sanitized metadata at
`control/environment_metadata.json`. For a validation-only run that cannot
overwrite the five committed publication outputs, omit
`--analysis-output-dir`; the postprocessor then writes only its JSON report to
standard output.

The command reads only the campaign contract, 800 episode JSON records, and
800 NPZ traces. It does not trust the runner-generated summary or decision. It
verifies exact seeds 49900--49999, all eight arms, source/configuration hashes,
pairing, finite outcomes, causal first-TRACK semantics, and trace-derived
scores before recomputing the factorial effects and qualification screens.
The output directory must be outside the immutable campaign tree.

The five release artifacts are `dynamic_plant_episode_rows.csv`,
`v40_postprocessed_full.json`, `v40_publication_results.json`,
`v40_publication_arms.csv`, and `v40_interaction_plot.csv`. Their definitions,
including the strict separation between transport-delay mismatch and
first-order response mismatch, are in
`docs/V40_QUALIFICATION_POSTPROCESSOR.md`.

The DOI projection itself is generated by
`scripts/project_v40_doi_campaign.py`. It refuses an existing destination,
requires exact frozen cardinalities, rejects any private path beyond the one
declared contract field, verifies the approved metadata/configuration hashes,
and records file-level original/exported SHA-256 values. The source campaign
is never edited.

## 7. Provenance expectations

Each campaign output must include:

1. the frozen protocol and campaign contract;
2. interpreter and dependency versions;
3. seed range and pairing rules;
4. hashes of all execution sources;
5. row-level episode outputs;
6. an independent audit report;
7. the explicit pass/fail decision, including invalidated optional arms.

The final holdout seed range must not be opened merely to demonstrate repository functionality. Use the documented smoke/development seeds for examples.
