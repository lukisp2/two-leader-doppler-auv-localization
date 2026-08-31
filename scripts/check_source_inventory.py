#!/usr/bin/env python3
"""Check that release inventories and top-level artifact directories agree."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def entries(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            yield value


def main() -> int:
    source_entries = set(entries(ROOT / "docs/SOURCE_FILES.txt"))
    protocol_entries = set(entries(ROOT / "docs/PROTOCOL_FILES.txt"))
    groups = (
        (source_entries | protocol_entries, ROOT / "code"),
        (set(entries(ROOT / "docs/TEST_FILES.txt")), ROOT / "tests"),
        (protocol_entries, ROOT / "protocols"),
        (set(entries(ROOT / "docs/SCRIPT_FILES.txt")), ROOT / "scripts"),
    )
    missing = []
    unlisted = []
    for expected, destination in groups:
        for name in expected:
            if not (destination / name).is_file():
                missing.append((destination / name).relative_to(ROOT).as_posix())
        actual = {
            path.name
            for path in destination.iterdir()
            if path.is_file() and path.name not in {"README.md", ".DS_Store"}
        }
        for name in sorted(actual - expected):
            unlisted.append((destination / name).relative_to(ROOT).as_posix())
    for name in entries(ROOT / "docs/FIXTURE_FILES.txt"):
        if not (ROOT / name).is_file():
            missing.append(name)
    if missing:
        for path in missing:
            print(f"MISSING {path}")
    if unlisted:
        for path in unlisted:
            print(f"UNLISTED {path}")
    if missing or unlisted:
        return 1
    print("Source, test, protocol, and script inventories are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
