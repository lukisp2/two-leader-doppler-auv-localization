#!/usr/bin/env python3
"""Create a path-sanitized complete DOI projection of frozen V41.1.

The implementation deliberately reuses the audited V40 portability boundary:
only ``control/campaign_contract.json:environment_metadata_path`` is replaced,
the approved path-sanitized environment metadata is added, and every other
campaign file must remain byte-for-byte identical.  V41.1-specific
cardinalities are fixed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import project_v40_doi_campaign as projector


V41_COUNTS = projector.ExpectedCounts(
    episode_json=400,
    trace_npz=400,
    noise_npz=100,
    source_snapshot_files=15,
)


def _json_strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_strings(child)
    elif isinstance(value, str):
        yield value


def private_path_hits_allowing_spaces(root: Path) -> list[tuple[str, str]]:
    """Return complete JSON path values even when directories contain spaces.

    The V40 scanner intentionally used a whitespace-delimited text regex.  The
    V41 source path contains ``Mój dysk``, so a JSON-aware scanner is required
    to compare the complete recorded contract value rather than a truncated
    prefix.  Non-JSON text retains the original fail-closed regex scan.
    """

    hits: list[tuple[str, str]] = []
    for path in projector.regular_files(root):
        if path.suffix.lower() not in projector.TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            value = json.loads(text)
            for item in _json_strings(value):
                markers = [
                    marker
                    for marker in ("/Users/", "/home/")
                    if marker in item
                ]
                if projector.PRIVATE_ABSOLUTE_PATH.search(item) and not markers:
                    markers = [projector.PRIVATE_ABSOLUTE_PATH.search(item).group(0)]
                for marker in markers:
                    index = item.index(marker)
                    hits.append((relative, item[index:]))
        else:
            for match in projector.PRIVATE_ABSOLUTE_PATH.finditer(text):
                hits.append((relative, match.group(0)))
    return hits


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="immutable completed V41.1 campaign")
    parser.add_argument("output", type=Path, help="new DOI campaign projection")
    parser.add_argument(
        "--public-environment-metadata",
        type=Path,
        required=True,
        help="approved path-sanitized metadata projection",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    original_scanner = projector.private_path_hits
    projector.private_path_hits = private_path_hits_allowing_spaces
    try:
        manifest = projector.project_campaign(
            args.campaign,
            args.output,
            args.public_environment_metadata,
            expected_counts=V41_COUNTS,
            # V41 was executed from the already approved public metadata
            # projection, so its frozen contract records the public hash
            # rather than the V40-era private metadata hash.
            expected_original_metadata_sha256=(
                projector.PUBLIC_ENVIRONMENT_METADATA_SHA256
            ),
        )
    finally:
        projector.private_path_hits = original_scanner
    print(
        "V41.1 DOI projection PASS: "
        f"{manifest['source_campaign_file_count']} source files, "
        f"{manifest['exported_file_count_excluding_manifest']} exported files "
        "plus manifest; scientific_values_changed=false."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
