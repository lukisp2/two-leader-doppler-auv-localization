# Frozen runner test mirrors

The publication runners record their scientific source closure using the
historical path `tests/<name>.py` relative to the source directory.  In the
stand-alone repository, the executable source directory is `code/`, whereas
the ordinary pytest suite remains at repository-level `tests/`.

The six Python files in this directory are byte-identical mirrors of their
repository-level counterparts.  They preserve the frozen runner manifests
without modifying the publication source files or their recorded SHA-256
hashes.  `tests/test_public_release_layout.py` fails if a mirror is missing,
differs, or no longer satisfies a runner's source-manifest preflight.

