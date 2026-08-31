# Frozen public-release workflow

This checklist is for the archival release that accompanies the manuscript
resubmission. Do not publish a tag until all campaign and data checks pass.

For version `1.0.0` (`v1.0.0`, release date `2026-08-31`), software DOI
`10.5281/zenodo.22214022` and data DOI `10.5281/zenodo.22214031` were reserved
before the immutable tag was created.

1. Run the complete release-validation test suite with CPython 3.11.9, NumPy
   2.3.3, and no Numba. This validates the current source tree; it is separate
   from the historical CPython 3.12.8 / NumPy 2.2.1 environment recorded by
   the frozen V35 campaign contract.
2. Verify every compact per-scenario table against its archived campaign.
3. Add the audited dynamic-response/current qualification rows and summary.
4. Confirm that `data/DATA_INVENTORY.csv` maps every manuscript result table to
   a compact source file and, where needed, a raw archive component.
5. From clean committed source content, generate `SHA256SUMS` with the default
   tracked-Git inventory. Generate the data manifest separately with explicit
   `--inventory filesystem`. Verify both with the default exact-tree check;
   do not accept a hash-only (`--no-exact-tree`) result.
6. Check that no credentials, caches, editor state, private absolute paths, or
   unrelated development files are present.
7. Create two unpublished manual Zenodo drafts: one software record and one
   raw-data record. For this release, reserve software DOI
   `10.5281/zenodo.22214022` and data DOI `10.5281/zenodo.22214031` before
   creating the immutable source tag. Relate the data record to the software
   record and the article using the applicable DataCite relations.
8. Insert the reserved software DOI, data DOI, and release date into
   `CITATION.cff`, the README, the data inventory, and release notes. Rebuild
   and verify the final manifests after these identifiers are frozen.
9. Create the public GitHub repository, push the frozen `main` commit, and
   create the immutable tag `v1.0.0` at that exact commit.
10. Upload the tagged source archive to the reserved software record and the
    complete raw archive to the reserved data record. Publish the GitHub
    release and both Zenodo records only after their assets and hashes match.
11. Verify all URLs, downloads, tag, commit, licenses, and DOI landing pages
    without authentication. Only then insert the verified identifiers into
    the manuscript and response to reviewers.

The manual software deposit avoids a circular workflow in which a GitHub tag is
archived before its DOI can be written into `CITATION.cff`. The GitHub source
archive is not a substitute for separately depositing raw data that are outside
the committed tree. Conversely, a summary-only JSON file is not sufficient
evidence for a manuscript table when row-level values exist.
