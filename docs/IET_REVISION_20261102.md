# Difficult-initial-condition comparison

The frozen protocol is `protocols/EXPERIMENT_PROTOCOL_IET_DIFFICULT_STARTS.txt`.
This addition changes no estimator, planner, evidence gate, controller,
sensor-noise setting, prior or success threshold. It is a descriptive stress
comparison, not a replacement for the earlier qualification campaigns.

## Population and results

The first 50 qualifying initial states in seeds 52000–52999 were selected
before their outcomes were evaluated: formation error at least 350 m and
absolute heading difference from leader-centroid course at least 120 degrees.
There were 57 eligible states. The follower remains within the original
120–350 m centroid-distance prior; formation error and centroid distance are
different quantities. Both arms use information-guided acquisition, both
Doppler links, the delayed first-order plant, no current and a 440-s horizon.

| Outcome | Proportional | Delay-aware |
|---|---:|---:|
| Terminal success | 22/50 | 48/50 |
| Tail80 success | 2/50 | 46/50 |
| Dwell15 success | 1/50 | 48/50 |
| Terminal formation error, median / p95 (m) | 8.587 / 13.243 | 0.656 / 1.699 |
| Terminal localization error, median / p95 (m) | 0.416 / 0.806 | 0.549 / 1.146 |
| Mean TRACK command saturation fraction | 0.9531 | 0.0595 |

All terminal localization errors are below 7 m, but localization did not
improve numerically under delay-aware tracking. No truth-invalid TRACK
release/start/end was observed. There are no reference-only successes for
any of the three binary outcomes. These are empirical results for the
selected population, not guarantees for arbitrary initialization or a
calibrated physical vehicle.

The four delay-aware Tail80 failures first entered TRACK at 184 s. Two reached
joint success late (408 s and 388 s); two ended with formation error 56.99 m
and 20.89 m. Each had accurate localization and falling formation error near
the end. These observations indicate a finite-horizon recovery limitation;
they do not prove sole gate causality or guaranteed eventual convergence.
No horizon was extended and no parameter was retuned after observing outcomes.

## Files and verification

- `data/tables/iet_difficult_initial_episode_rows.csv`: all 100 arm records.
- `data/tables/iet_difficult_initial_summary.json`: frozen campaign summary.
- `data/provenance/iet_difficult_initial_conditions/`: all 1000 screened
  initial states, selected cohort, projected contract, pairing checks,
  independent outcome/integrity audits and projection hash mapping.
- `data/tables/initial_condition_{analysis,episode_rows}.json`: descriptive
  reanalysis of 600 prior link/acquisition and 400 prior controller records.
- The separate asset `iet-difficult-initial-conditions-v1.2.0.zip` contains
  all 100 episode JSONs, 100 trajectory NPZs, 50 noise NPZs, 27 frozen source
  files and experiment records. It is available from the
  [version 1.2.0 release](https://github.com/lukisp2/two-leader-doppler-auv-localization/releases/tag/v1.2.0).

The numerical results, trajectories, and noise sequences are unchanged.
Private local paths were removed from three metadata files; these changes are
listed in `PROJECTION_MANIFEST.json`. The original source files record how the
experiment was run. To reproduce it on another computer, use
`code/run_iet_difficult_initial_conditions.py`.

After extracting the raw asset, independently recalculate outcomes without
running any simulation:

```bash
python scripts/verify_sha256_manifest.py \
  /path/to/extracted-asset/iet_difficult_initial_conditions/SHA256SUMS
python scripts/audit_iet_difficult_outcomes.py \
  /path/to/extracted-asset/iet_difficult_initial_conditions \
  --output results/independent_difficult_start_audit.json
PYTHONPATH=code python -m unittest discover -s tests -p 'test_iet*.py' -v
```

To reproduce only initial-state selection (no episode execution):

```bash
python code/run_iet_difficult_initial_conditions.py \
  --replicate-published --prepare-only --output results/iet_selection
```

To run the complete 100-episode replication explicitly:

```bash
python code/run_iet_difficult_initial_conditions.py \
  --replicate-published --output results/iet_replication
```

This flag reruns the published scenarios and checks their initial states against
the released data. Use a new output directory, or `--resume` to continue a run
with unchanged settings. The runner continues to exclude the reserved scenario
range 50000–50999.

The historical reanalysis accepts `UUV_DATA_ARCHIVE` (the v1.0.0 raw archive
root containing its SHA256SUMS), `UUV_CONTROLLER_CAMPAIGN` (the complete earlier
400-run controller campaign), optional `UUV_NOMINAL_MIRROR`, and
`UUV_INITIAL_ANALYSIS_OUTPUT`. Run `scripts/analyze_historical_initial_conditions.py`.
The earlier controller raw campaign is a separate deposit from this 100-run
asset; the published compact 400 rows alone cannot reproduce trace-level
reanalysis. Do not attribute either later campaign to the v1.0.0 Zenodo DOI.
The analysis script was rerun on all 1000 historical records and reproduced the
original individual results and summaries. Changes to source labels and the
script are listed in `historical_analysis_projection.json`.

The main saturation metric uses any normalized command magnitude at least
0.98. Historical nested controller diagnostics use a different near-exact
saturation threshold; these fields are not interchangeable. Coordinate-level
auditing uses constant-velocity leader truth extrapolated from the first
broadcast rather than treating later nominal timestamp labels as exact
action-end truth. This reproduces all stored formation errors to numerical
precision without changing their definition.
