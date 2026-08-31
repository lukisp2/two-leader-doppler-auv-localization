# Audited DOI-archive path projection

The frozen estimator and leader-source records include inert provenance paths
from the workstation on which the campaigns were run. Before deposit, those
paths are projected to the literal token `${FROZEN_SOURCE_ROOT}` by
`scripts/project_doi_archive_paths.py`.

This is not a recursive sanitizer. The script contains a closed allowlist of
six file classes and their exact JSON pointers. It accepts only an exact,
caller-supplied source prefix followed by an unchanged suffix. Any other
`/Users/`, `/home/`, or Windows `C:\Users\` path fails the complete operation
before a byte is written. Non-JSON files, including all NPZ arrays and traces,
are read only.

## Apply and verify

Run the projection from outside the public command log so the private prefix
is not captured in a release artifact:

```bash
python scripts/project_doi_archive_paths.py apply /path/to/doi_archive \
  --source-prefix /exact/private/frozen/source/root
python scripts/project_doi_archive_paths.py verify /path/to/doi_archive \
  --source-prefix /exact/private/frozen/source/root
```

The exact prefix is never stored in the output. The deterministic top-level
`path_projection_manifest.json` records only its SHA-256 digest. A second
application is a byte-identical no-op.

## Scientific-value and hash invariants

For every projected JSON file, the manifest records:

- the archive-relative path;
- original and public byte-level SHA-256 values;
- every changed JSON pointer and its replacement count;
- a canonical scientific SHA-256 after normalizing only the allowlisted path
  prefix; and
- `scientific_values_changed: false`.

The canonical scientific hash includes every JSON key and value after the
audited path normalization. Equality before and after projection therefore
proves that no other JSON value changed. Literal byte replacement preserves the
original whitespace, key order, and numeric spelling.

Historical contracts intentionally keep their original self-hash fields. The
validator bridges them by substituting the caller-supplied original prefix back
only at the manifest-listed pointers. It then checks the reconstructed V27 and
V34 canonical contract hashes, the V34 reference to the original V27 contract
file hash, all dependent `contract_sha256` values, and both campaigns' 200
references to the 100 original capture-metadata hashes. Changing those stored
hash fields would alter non-path provenance and is therefore forbidden.

The manifest also commits to the complete NPZ relpath/size/SHA-256 inventory.
The inventory is recomputed after the write phase and must remain identical.
Outputs are validated in memory before commit, each file is replaced atomically,
and ordinary write failures trigger rollback. The projection manifest is
written last, so an interrupted mixed state without a manifest can be rerun
deterministically and repaired.
