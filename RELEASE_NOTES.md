# Release notes

## Version 1.2.0 - large initial position and heading errors (2026-11-02)

This release compares proportional and delay-aware tracking in 50 paired
scenarios, each starting with formation error of at least 350 m and heading
difference of at least 120 degrees. Scenarios were selected from their initial
states before simulation. The method and controller settings were not retuned.

Terminal success was 22/50 for proportional tracking and 48/50 for delay-aware
tracking. Tail80 success was 2/50 and 46/50; Dwell15 success was 1/50 and 48/50.
All final localization errors were below 7 m. The four delay-aware Tail80
failures, including two terminal failures, are included and discussed.

The [release download](https://github.com/lukisp2/two-leader-doppler-auv-localization/releases/tag/v1.2.0)
contains all 100 results, trajectories, noise sequences, and source files used
for the experiment. The repository also includes its runner, protocol, tests,
and an analysis of initial conditions in 1000 earlier runs.
See [experiment details](docs/IET_REVISION_20261102.md) for reproduction commands.
These new data are not part of the version 1.0.0 Zenodo archives.

## Version 1.1.1 - figure presentation (2026-09-23)

Figure 4 annotations were repositioned and the Figure 6 outcome label clarified.
Simulation code, results, and conclusions were unchanged.

## Version 1.1.0 - delay-aware tracking (2026-09-02)

This release adds a controller that accounts for delayed first-order vehicle
response. It combines a Smith predictor, reference governor, smooth transition
from acquisition, and command slew limits. The proportional controller is
retained for comparison. Estimation and acquisition are unchanged.

In 100 paired scenarios per current condition, terminal success increased from
38% to 95% without current and from 26% to 95% with current. Tail80 success
increased from 2% to 91% and from 6% to 92%. Command saturation also decreased.
Without current, however, the increase in localization p95 was 0.5669 m,
exceeding the predefined 0.50-m limit. The overall acceptance criterion was
therefore not met.

The source release contains all 400 per-run results, analysis scripts, tests,
and settings. Its full trajectory archive remains separate from the version
1.2.0 download and the original Zenodo archives.
[Controller comparison details](docs/V41_CONTROLLER_REPAIR_AUDIT.md)

## Version 1.0.0 - original reproducibility release

The original release contains the full-history estimator, active acquisition,
switching logic, comparison estimators, and the original simulation studies.

- [Software archive](https://doi.org/10.5281/zenodo.22214022)
- [Raw-data archive](https://doi.org/10.5281/zenodo.22214031)

The results include unsuccessful tests. Five of 1200 runs in the robustness
experiment exceeded the two-second computation limit; all five used the optional
bias-estimation extension. The initial delayed-response study also showed poor
sustained tracking despite accurate localization. These findings are retained.

The original validation records remain in the corresponding tagged versions.
Tags, numerical results, and experiment protocols have not been rewritten by
later documentation changes.
