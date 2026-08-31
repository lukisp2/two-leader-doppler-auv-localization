# Source provenance and release boundary

This provenance record accompanies source version `1.0.0` at
<https://github.com/lukisp2/two-leader-doppler-auv-localization>. The software
record has DOI `10.5281/zenodo.22214022`, and the separate data record has DOI
`10.5281/zenodo.22214031`.

The research working directory is part of a larger workspace containing
unrelated and work-in-progress material. This public repository is therefore a
reviewed stand-alone export rather than a copy of the parent workspace.

## Included

- the transitive local-import closure listed in `SOURCE_FILES.txt`;
- only the tests listed in `TEST_FILES.txt`;
- protocols directly supporting manuscript Tables 6--11;
- the public scripts listed in `SCRIPT_FILES.txt`, including the independent
  V40 read-only postprocessor;
- compact per-scenario result rows and integrity metadata;
- the frozen environment metadata at the legacy relative path required by the
  campaign runners (`code/experiments_v18_1_guard_ablation_dev_3seed/.../metadata.json`).

## Excluded

- model-training experiments unrelated to the final method;
- IDE files, caches, logs, checkpoints, and exploratory notebooks;
- invalidated or confounded campaigns, except the complete row-level V35
  stress export retained because the manuscript reports its outcomes
  explicitly as descriptive after the prespecified joint runtime gate failed;
- the parent `.git` directory;
- large NPZ/JSON trace trees from ordinary Git history.

Every exported file was compared against the frozen campaign
`control/source_manifest.json` files. Each intentional portability change is
documented, tested, and separately hashed.

`FIXTURE_FILES.txt` lists four small duplicated fixtures required to preserve
unchanged historical relative-path defaults. They are configuration/summary
fixtures, not hidden training artifacts.

The original V18.1 metadata contained absolute private execution paths and
unrelated training provenance. The public copy is a documented projection that
retains `environment_config` byte-for-value while removing those private paths:

```text
original metadata SHA-256: b486d280e57661f4b526b926e0f8e32a5373b99f585de09e3354d87b3342dfe7
public projection SHA-256:  33aff4c6794c9eab2d92bd4c9aa2f3ff3b89b4f3fc121904df40124f584b626f
canonical environment_config SHA-256: 74e83dae26f198604803d50f2da3c14453118e9418e41dc3316684391eb7e211
```

Both duplicated public copies must retain the latter hash.
`scripts/postprocess_v40_qualification.py` accepts either the contract-exact
private file or this one approved public projection. In the latter case it
requires both the public full-file hash and the canonical hash of the complete
`environment_config`; matching a few action-scale fields is insufficient.

## Complete V40 DOI projection

`scripts/project_v40_doi_campaign.py` creates the public
`data/raw/dynamic_plant_stress/campaign/` tree atomically and never edits the
private frozen campaign. It requires 800 episode JSON files, 800 trajectory
NPZ files, 100 shared noise tapes, and 14 source-snapshot files. Every campaign
file except `control/campaign_contract.json` must retain its original SHA-256.
The contract is changed only at `environment_metadata_path`, which becomes
`${CAMPAIGN_ROOT}/control/environment_metadata.json`; the approved public
metadata projection above is copied to that relative location.

`public_projection_manifest.json` records original and exported SHA-256 values
for all 1720 files in its scope, the original/public metadata hashes, the
canonical environment-configuration hash, exact cardinalities, and
`scientific_values_changed: false`. The manifest itself is the 1721st file and
is covered later by the DOI archive's final top-level `SHA256SUMS`.

## Publication-figure portability projection

The figure modules were copied from the frozen private source tree, then
changed only at their public-release boundary: absolute/private defaults were
replaced by repository-relative paths, command-line help was completed, and
the aggregate figure driver was taught to consume released compact estimator
and closed-loop-stress artifacts. The scientific computations, plotted
quantities, labels, styles, and manuscript layouts were retained. The
paired-episode module is included because it is the complete local import and
data-validation closure of the mission-geometry generator. The
closed-loop-stress plot remains an archived diagnostic and is not Figure 9 of
the resubmission; the resubmission's Figure 9 comes from the command-response
qualification export.

| Public module | Frozen private SHA-256 | Public portability-projection SHA-256 |
|---|---|---|
| `make_publication_figure_paired_episode.py` | `991a92d981c38e83e56724420eae494dc5744ebcfa96eeb7abc711d59fdab883` | `ff2efde0e565f54f1fbbc3f83dff7504a83c5e484622e2ffe668882f6eb4d9b8` |
| `make_publication_figure_mission_geometry.py` | `49ae7e8b87f77e1a2a9436b32218e7131ed3f1e00bb60674cfa99ca7c5ee62a8` | `61f4037f7c9903b95e8ad8677a5fa7bd17ee9df6b3891fdcaba1326868e8efa3` |
| `make_publication_figure_doppler_geometry.py` | `7aa7424c48752ecb22db19b4458c41620cf6cb762a838bf2690ea8af636e6994` | `933aa7adbca66312c21664b33280ac968bbd7d45c8d2e26688b816dcabe3950b` |
| `make_publication_figure_full_history_estimator.py` | `81ab603258fca11497684372aed285a8774b5f60363f3dcb740ba2399f5777de` | `66b41b086c4559b74e59a8241f7a98389a4763f0124a2dd33152bb05455d3f35` |
| `make_publication_figure_source_policy_ablation.py` | `7ec548dcc3f71ed653a536afb1cee78a9c02db4b226519a84221f6a25da54f74` | `8d7cc1547e527960520c1da1321a90939757bc9265750e3219e96840b679e174` |
| `make_publication_figure_source_policy_dynamics.py` | `daa89e6293f20dfd6c5d366c5d8af7041c90e4e6e941f1f4d7a9be26d6264a20` | `393e02696fef035f1f521f8d44469d327eefeb952413a1b9eb4a9eeee2116b64` |
| `make_publication_figures_v33.py` | `bcb88a38b64c882b9894f7e637d7b8efde3f76d907d95be3a7d2dabe88500493` | `11687133e07159f7e0093020fc82c2de97ade2b80a39b58f61436fe9d9ae740c` |

The public portability-projection hashes above were regenerated after the
final label and safety changes. The same-environment render comparison is part
of the clean-checkout release audit. `SHA256SUMS.preliminary` is temporary;
only the final `SHA256SUMS` verified from a clean checkout is authoritative.
