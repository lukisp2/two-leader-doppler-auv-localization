# Test population status

The authoritative test list is `docs/TEST_FILES.txt`. Preserve filenames and run the suite with:

```bash
PYTHONPATH=code python -m pytest tests -v
```

`test_project_doi_archive_paths.py` exercises the fail-closed path allowlist,
historical-hash reconstruction bridge, NPZ immutability guard, deterministic
idempotency, manifest tamper detection, and transactional rollback.
