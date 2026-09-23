# Release status

This document records the current freeze state for version `1.1.1`, tag
`v1.1.1`, dated `2026-09-23`. The source repository is
<https://github.com/lukisp2/two-leader-doppler-auv-localization>. The software
record DOI `10.5281/zenodo.22214022` and separate data-record DOI
`10.5281/zenodo.22214031` identify the preceding immutable `v1.0.0` archives;
the controller addendum is public in the `v1.1.0` GitHub tag and is not claimed
to be present in those earlier DOI versions.

## Version 1.1.1 figure-presentation patch

The Figure 4 annotation placement and Figure 6 outcome title now match the
final manuscript. Both generators were rerun with the pinned figure stack.
The source manifest covers this patch; all simulation code and result data
retain their `v1.1.0` contents.

## Version 1.1.0 controller-repair update

The reviewer-driven controller-repair addendum is part of `v1.1.0` but not the
immutable `v1.0.0` tag or its two DOI versions. Its 100-seed, four-arm campaign
is complete and integrity-valid, but it did not meet every prespecified
acceptance condition. The unambiguous state is
`VALID_COMPLETE__COMPOSITE_ACCEPTANCE_NOT_MET`; see
`docs/V41_CONTROLLER_REPAIR_AUDIT.md`.

- [x] Delay-aware tracker, integration runner, protocol, and component tests
      are staged.
- [x] All 400 row-level outcomes, four publication cells, row-derived paired
      effects, and path-sanitized provenance are staged.
- [x] The complete-campaign DOI projector was validated on 922 input campaign files;
      all episode results, traces, tapes, and frozen sources were preserved.
- [x] Replacement Figure 9 is reproducible from the two compact release JSON
      files and is pixel-identical to the current manuscript raster.
- [x] Run the complete archive-connected suite under pinned CPython 3.11.9
      without Numba and with the full equivalence smoke: 275 tests and 64
      subtests passed.
- [x] Freeze manuscript Table 10/Fig. 9 numbering and claims against the
      compact outputs.
- [x] Assign source version/tag `1.1.0` / `v1.1.0`.
- [ ] Create new Zenodo software and data record versions. Until then, cite the
      `v1.1.0` GitHub tag for the controller addendum and do not attribute it
      to the `v1.0.0` DOI records.
- [x] Generate and verify the current clean-tree manifest immediately before
      tagging; the immutable `v1.0.0` tag retains its historical manifest.

## Frozen dynamic-response source snapshot

The dynamic/current factorial source and protocol were frozen before the qualification campaign began. SHA-256 values for the exported files are:

```text
db1ac7cb2d743553eec86dd7ec7b8f6386a690ad522bd8d1fa8d147f47fcb42f  code/uuv_v40_dynamic_plant_stress.py
ed3f1cf0998df29efaaa06bc5d60522fcce5cbc37762f451d36fcf15518c5d3b  code/run_v40_dynamic_plant_stress.py
d8ae5befd599e70197e6f6fc3d0a014d4f628a0107de5654491fba31a279e857  code/audit_v40_dynamic_plant_stress.py
086253d6e3130b0ec78c846e027911e7255165f7786a2b83886c2efc3af83d76  tests/test_uuv_v40_dynamic_plant_stress.py
ec25d414b409c1005c3722fd7c51750d6c3a64fd3d7487a7bd7975c96456176f  protocols/EXPERIMENT_PROTOCOL_V40_DYNAMIC_PLANT_STRESS.md
```

## Independent dynamic-response postprocessor

`scripts/postprocess_v40_qualification.py` is staged as a public, read-only
analysis path with synthetic tests in
`tests/test_postprocess_v40_qualification.py`. It does not read the runner's
campaign summary or decision and refuses to write derived artifacts inside the
campaign directory. The validated 800-row CSV, eight-row CSV, compact/full
JSON, and interaction-plot CSV have been generated outside the frozen campaign
and the postprocessor independently accepted the DOI-staged projection.

## Known descriptive-only campaign

The closed-loop stress export behind manuscript Table 11 preserves a
negative integrity decision. Its frozen summary reports `integrity_valid:
false` and `V35_INVALID`: 5 of 1200 runs exceeded the campaign-wide 2-s
decision-runtime gate, with a maximum of 2.7017 s. All exceedances occurred in
the optional late bias-switch arm; primary controller cells stayed below
1.49 s. The rows and plots are released for transparent descriptive analysis,
not as a passed robustness qualification. Public generators must require the
explicit `--allow-invalid-descriptive` flag.

The campaign contract records the historical V35 software environment as
CPython 3.12.8, NumPy 2.2.1, Gymnasium 1.2.3, and no Numba. Those versions are
required when attempting an exact software-stack replication of V35. The
CPython 3.11.9 / NumPy 2.3.3 stack below is the separate release-validation
environment for the current source tree and archive-connected tests.

## Release checklist

- [x] Qualification campaign `experiments_v40_dynamic_current_factorial_qualification100` has completed.
- [x] The runner decision and independent summary-free audit are present and
      both report `V40_QUALIFICATION_COMPLETE` with valid integrity checks.
- [x] The 800 row-level arm records and compact summary have been exported.
- [x] The summary-independent V40 postprocessor and eight synthetic integrity
      tests are staged. A validation-only run on the DOI projection returned
      `integrity_valid: true` without writing publication artifacts.
- [x] The validated eight-row table, compact effects JSON, full audit JSON, and
      interaction-plot CSV have been generated outside the campaign tree.
- [x] Dynamic/current manuscript claims match the observed results without post-hoc method changes.
- [x] All source/test/protocol/script inventories pass
      `scripts/check_source_inventory.py`.
- [x] The tested V40 DOI projector created a 1721-file path-sanitized campaign
      with 800 byte-identical episode JSON files, 800 byte-identical trajectory
      NPZ files, 100 byte-identical noise tapes, and a byte-identical 14-file
      source snapshot. All 1720 manifest entries rehashed successfully; the
      package security/privacy and NPZ `allow_pickle=False` scans passed.
- [x] Archive-connected pre-results pytest suite was rerun in the exact
      release-validation stack: CPython 3.11.9 /
      NumPy 2.3.3 / Gymnasium 1.2.3 / Matplotlib 3.10.6 / pytest 9.1.1
      environment: 185 passed, no tests skipped, and 60 subtests passed in
      112.39 s. The DOI-staged V18.1 fixture passed its bit-exact replay test.
      The final post-sanitization rerun in the same pinned environment then
      reported 241 passed tests, no skipped tests, and 64 passed subtests in
      111.70 s with the released raw-data tree connected.
- [x] The fail-closed DOI path projection replaced 1713 allowlisted private
      source-root values in 105 JSON files, preserved all scientific values,
      changed none of the 1902 NPZ files, repaired and verified all dependent
      historical hashes, and left no private home path in the public tree.
- [x] Compact inputs for Tables 6--11 are exported.
- [x] Lightweight provenance bundles for Tables 9 and 11 are committed with
      5 and 4 files, respectively. Their export manifests preserve original
      and public SHA-256 values and document source-root path replacement as
      the only transformation, with no scientific values changed. The Table 11
      bundle retains the frozen `V35_INVALID` decision.
- [x] Table 8 has 4500 normalized row-level records; its regeneration script,
      cross-check tests, and manuscript count/tail verification pass exactly.
      Regeneration from the DOI-staging directories `v27_campaign` and
      `v34_campaign` reproduces the committed CSV byte for byte; the exporter
      accepts only those public aliases or the two frozen original campaign
      names and rejects arbitrary renamed directories.
- [x] The frozen runner source-manifest layout is executable from the public
      `code/` tree. Six required test-path mirrors are byte-identical to the
      repository-level tests, and 12 public-layout checks cover all affected
      runners. The documented V34 smoke and full FEJ-MHE sequence was executed
      successfully against the DOI-staging histories and V27 references.
- [x] The public V27 portability helper validates all 100 released legacy PF
      traces against episode/seed identities and recorded SHA-256 values,
      changes only the historical provenance path in a generated view, and
      passes four fail-closed tests. The frozen V27 runner returned
      `SMOKE_PASS` from that projected archive without source modification.
- [x] Figure generators for Figures 2--4 and 6--9 are included with
      repository-relative defaults; data-free and compact-data render tests
      pass under a noninteractive backend. Figure 9 is the command-response
      qualification plot; the earlier closed-loop-stress diagnostic is released
      but is not a manuscript figure. Trace-derived inputs are inventoried for
      the DOI archive rather than duplicated in Git.
- [x] Separate version DOIs identify the software
      (`10.5281/zenodo.22214022`) and raw-data
      (`10.5281/zenodo.22214031`) records.
- [x] `CITATION.cff` contains the current repository URL, version, and release
      date. The README and release notes identify the two DOI records as
      belonging only to the preceding `v1.0.0` software/data snapshots. The
      controller addendum has no DOI yet, and the article DOI is unavailable
      until article publication.

`SHA256SUMS.preliminary` is not part of the release. The authoritative source
and data archives each contain their own `SHA256SUMS`. Both are generated only
after metadata freeze and must pass the verifier's default exact-tree mode; an
ignored result, unlisted file, symlink, or macOS archive sidecar is not
permitted in either public payload.
