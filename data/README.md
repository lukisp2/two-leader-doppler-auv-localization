# Released data

The raw-data archive for version `1.0.0` is identified by DOI
`10.5281/zenodo.22214031`; the companion software record is identified by DOI
`10.5281/zenodo.22214022`, with source at
<https://github.com/lukisp2/two-leader-doppler-auv-localization>.

GitHub tag `v1.1.0` adds the complete 400-row controller-study outcomes,
compact paired statistics, provenance, and regeneration scripts. These files
are public in Git but are not claimed to be contained in the earlier DOI
records.

`tables/` contains compact, row-level CSV/JSON material needed to regenerate the manuscript tables. `DATA_INVENTORY.csv` describes both compact files and larger artifacts distributed through the release/DOI archive.

`provenance/` contains the lightweight control records for the two campaigns
whose complete row-level outcomes are already committed in `tables/` and
therefore do not require separate raw-trace deposits. The planner-component
bundle contains 5 files (92,900 bytes): its frozen contract, summary,
decision, independent audit, and export manifest. The closed-loop-stress
bundle contains 4 files (117,238 bytes): its frozen contract, summary,
decision, and export manifest. Each `provenance_export_manifest.json` records
the original and exported SHA-256 of every copied campaign artifact. The only
export transformation was a literal replacement of the private source-root
prefix by `${FROZEN_SOURCE_ROOT}`; no scientific value was changed.

Version `v1.1.0` adds a third controller-study provenance bundle,
`provenance/controller_repair/`, and three compact table artifacts. It retains
all 400 episode-arm outcomes in Git. Its frozen decision label joined rejection
and invalidity; the additive public audit records that the campaign was valid
and complete but missed one prespecified composite acceptance condition. The
audit also narrows inherited `sealed_seed_range_untouched` language to what is
actually verified: the selected 51000--51099 cohort is disjoint from the
reserved 50000--50999 range. It does not claim that no external process ever
opened the reserved seeds.

`closed_loop_stress_episode_rows.csv` and its summary are deliberately
retained as **descriptive data from an integrity-invalid campaign**. Five of
1200 optional bias-switch runs exceeded the frozen 2-s runtime gate (maximum
2.7017 s); the primary controller cells stayed below 1.49 s, but the protocol
made validity joint across the campaign. Downstream tools must not silently
treat these rows as a passed robustness qualification. Its provenance bundle
preserves the same `V35_INVALID` decision and does not convert the campaign
into a passed qualification. Its contract also preserves the historical
runtime stack: CPython 3.12.8, NumPy 2.2.1, Gymnasium 1.2.3, and no Numba. This
differs from the CPython 3.11.9 / NumPy 2.3.3 release-validation environment;
using the latter for V35 constitutes a new replication rather than an exact
reconstruction of the frozen software environment.

Large `episode_results/`, `unscored/`, `noise_tapes/`, and `traces_npz/` trees
are excluded from Git. Download
the archive, place its contents under `data/raw/`, and verify it against the
published SHA-256 manifest.

The estimator benchmark is the first fully normalized compact export. Verify
all 4500 rows and every rounded value printed in its manuscript table with:

```bash
python scripts/export_estimator_benchmark.py --check-only
```

To regenerate that CSV from the DOI archive, pass the extracted V27 and V34
campaign directories explicitly. The script rejects smoke or unaudited
campaigns, incomplete method--checkpoint cells, altered contract hashes, and
seed/index mismatches; see `docs/TABLES_AND_DATA.md` for the full command.
Exact file counts, byte counts, source-relative locations, and DOI-record
destinations for the five estimator components (including the small replay
fixture) and the complete
leader-source/acquisition trace campaign used by Figures 2 and 7 are listed in
`DOI_ARCHIVE_COMPONENTS.md`.

The complete dynamic-response qualification is staged under
`raw/dynamic_plant_stress/campaign/` in the DOI archive. Its 1721 files
(135,379,146 bytes) comprise all 800 episode JSON records, all 800 trajectory
NPZ files, 100 shared noise tapes, the 14-file frozen source snapshot, the
remaining frozen campaign records, an approved public environment-metadata
projection, and `public_projection_manifest.json`. The manifest records
original and exported SHA-256 values for every file in its scope. Only the
contract's inert `environment_metadata_path` was replaced by
`${CAMPAIGN_ROOT}/control/environment_metadata.json`; all episode records,
traces, tapes, and source files are byte-for-byte copies. After extraction,
run the independent postprocessor documented in
`../docs/V40_QUALIFICATION_POSTPROCESSOR.md`; its five validated
row-level/compact outputs belong in `tables/`, never inside the campaign tree.

The complete controller-repair qualification is not part of data DOI
`10.5281/zenodo.22214031`. Before publishing the next data version, create its
portable component with `scripts/project_v41_doi_campaign.py`; do not copy the
private campaign directory directly. The validated projection has 924 files:
922 frozen campaign files, approved public environment metadata, and the
projection manifest.
