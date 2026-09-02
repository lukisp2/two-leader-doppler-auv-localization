# Source population status

This directory intentionally remains flat because the frozen research modules import one another by their historical filenames (for example, `uuv_v19_observability`).

The authoritative file list is `docs/SOURCE_FILES.txt`. Populate it only after the resubmission code is frozen, then generate `SHA256SUMS`. Do not copy the parent 44 GB monorepository.

The publication-figure entry points are the eight `make_publication_figure*.py`
modules. `make_publication_figure_mission_geometry.py` imports
`make_publication_figure_paired_episode.py`; both must remain in this flat
directory. Data-free explanatory figures and compact-data quantitative figures
run directly from the repository. Mission/source-policy trajectory figures
additionally require the `leader_source_ablation/` DOI component described in
`docs/TABLES_AND_DATA.md`.

The post-`v1.0.0` controller-repair source closure consists of
`delay_aware_formation_tracker.py`, `uuv_v41_controller_repair.py`, and
`run_v41_controller_repair.py`. The complete protocol is duplicated in
`code/` and `protocols/` because the historical flat runners and the public
release inventory use both locations. Component tests remain under
`code/tests/`; release-level export and layout tests are under `tests/`.

Historical source comments are retained byte-for-byte where campaign manifests
bind them. In particular, the phrase `post-lock PID` in the frozen command-
response module is legacy shorthand: the implemented tracker is the
proportional position-error law with leader-centroid velocity feedforward
described in the manuscript, with no integral or derivative term.
