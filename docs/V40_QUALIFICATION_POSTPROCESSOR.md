# Frozen V40 qualification: public read-only postprocessor

This directory contains an independent postprocessor for the frozen V40
policy--execution--current qualification.  It is intentionally separate from
the campaign runner and frozen source tree.

## What it validates

The command accepts a **completed** campaign output directory and fails closed
unless it finds:

- exactly seeds 49900--49999 and no other seed;
- all eight unique policy x execution x current arms for every seed;
- the exact frozen 2 x 2 x 2 contract, publication estimator settings,
  220-action horizon, timing, final sealed range, and zero-retuning flag;
- source-snapshot hashes and their manifest hash matching the campaign
  contract;
- the frozen environment-metadata hash;
- identical paired initial state, mission support, sensor-noise tape, and
  stochastic-current tape within every seed;
- finite required outcomes and complete traces;
- a finite first evidence transition and first TRACK-action time whenever a
  lock occurred.  Null/missing first-TRACK time is accepted only for a
  no-lock episode;
- row-level terminal, Tail80, Dwell15, gate-safety, plant, current, and command
  diagnostics that can be reproduced from each NPZ trace.

The postprocessor never reads `campaign_summary.json`, `decision.json`, or
`episode_arm_summary.csv`.  Those files can neither validate the campaign nor
change the recomputed decision.

## Analysis produced

All effects use the episode as the paired unit.  The output contains:

- eight cellwise summaries;
- active-minus-fixed contrasts in each execution--current cell;
- dynamic-minus-kinematic contrasts within policy and current;
- current-minus-none contrasts within policy and execution;
- dynamics x current, policy x dynamics, and policy x current
  difference-in-differences;
- the three-way policy x dynamics x current interaction;
- the change in active-policy advantage from nominal conditions to joint
  dynamics-plus-current nonideality;
- terminal, Tail80, ever-lock, conditional first-TRACK time, terminal and
  final-50-action mean \(e_p/e_f\), truth-invalid TRACK starts/ends, and
  decision-runtime outcomes;
- the frozen absolute robustness and policy-claim screens.

Binary paired effects include discordances, paired Newcombe method-10 95%
intervals, and exact McNemar tests. Numeric paired effects and interactions use
50,000 deterministic PCG64 percentile bootstrap resamples. First-TRACK contrasts state their complete missingness
and are explicitly conditional on all contributing arms reaching TRACK.

## Command-execution semantics

Two quantities are reported separately:

1. **requested-to-delayed command mismatch**, computed between the normalized
   command selected by the controller and the command leaving the frozen
   one-action delay queue;
2. **first-order response mismatch**, computed in physical channels between
   that delayed command and the end-of-action executed acceleration/yaw-rate/
   pitch-rate state.

The latter is not defined for the kinematic plant.  In the nominal
kinematic/no-current compatibility path the stored executed-rate array is
zero by construction.  The postprocessor therefore reconstructs physical
kinematic commands from the saved normalized actions and the hash-verified
environment scaling instead of misinterpreting those zeros as zero vehicle
command.

## Usage after the campaign has completed

Print the complete recomputed JSON to standard output:

```bash
python3 scripts/postprocess_v40_qualification.py \
  /absolute/path/to/experiments_v40_dynamic_current_factorial_qualification100 \
  > v40_postprocessed_full.json
```

If the absolute metadata path recorded in the contract is no longer
available, provide the same hash-matching file explicitly:

```bash
python3 scripts/postprocess_v40_qualification.py CAMPAIGN \
  --environment-metadata \
    code/experiments_v18_1_guard_ablation_dev_3seed/evaluations/dev100_seed_28001_range_45000_45099/metadata.json
```

The repository file used above is the approved path-sanitized public
projection.  The postprocessor accepts it only when both its full-file hash
and the canonical hash of the complete frozen `environment_config` match the
values recorded in `docs/SOURCE_PROVENANCE.md`; an arbitrary edited metadata
file is rejected.

Generate the full audit plus compact publication artifacts in a directory
**outside** the immutable campaign:

```bash
python3 scripts/postprocess_v40_qualification.py CAMPAIGN \
  --environment-metadata \
    code/experiments_v18_1_guard_ablation_dev_3seed/evaluations/dev100_seed_28001_range_45000_45099/metadata.json \
  --analysis-output-dir data/tables
```

This writes:

- `dynamic_plant_episode_rows.csv` -- 800 validated per-arm episode rows used
  for paired reanalysis;
- `v40_postprocessed_full.json` -- complete audit and analysis;
- `v40_publication_results.json` -- compact JSON with eight arms, paired main
  effects, interactions/DiD, screens, and command semantics;
- `v40_publication_arms.csv` -- exactly eight rows, ready for a manuscript
  table;
- `v40_interaction_plot.csv` -- long-format marginal cell data ready for
  interaction plots.

The tool refuses an analysis output directory inside the campaign directory.

## Tests

The tests use only generated synthetic fixtures; they do not access any V40
campaign result:

```bash
python3 -m pytest tests/test_postprocess_v40_qualification.py -v
```
