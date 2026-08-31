#!/usr/bin/env python3
"""Export path-sanitized campaign control and audit records with source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "control/campaign_contract.json",
    "campaign_summary.json",
    "decision.json",
)
OPTIONAL_FILES = ("independent_audit.json",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(value: Any, source_root: str) -> tuple[Any, int]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        replacements = 0
        for key, item in value.items():
            converted, count = sanitize(item, source_root)
            result[str(key)] = converted
            replacements += count
        return result, replacements
    if isinstance(value, list):
        result_list: list[Any] = []
        replacements = 0
        for item in value:
            converted, count = sanitize(item, source_root)
            result_list.append(converted)
            replacements += count
        return result_list, replacements
    if isinstance(value, str):
        count = value.count(source_root)
        return value.replace(source_root, "${FROZEN_SOURCE_ROOT}"), count
    return value, 0


def export_campaign(
    campaign: Path,
    output: Path,
    *,
    source_root: Path,
) -> dict[str, Any]:
    campaign = campaign.expanduser().resolve()
    output = output.expanduser().resolve()
    source_root_text = str(source_root.expanduser().resolve())
    if not campaign.is_dir():
        raise FileNotFoundError(campaign)
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    paths = list(REQUIRED_FILES)
    for relative in REQUIRED_FILES:
        if not (campaign / relative).is_file():
            raise FileNotFoundError(campaign / relative)
    paths.extend(
        relative for relative in OPTIONAL_FILES if (campaign / relative).is_file()
    )

    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    try:
        for relative in paths:
            source = campaign / relative
            document = json.loads(source.read_text(encoding="utf-8"))
            sanitized, replacements = sanitize(document, source_root_text)
            serialized = (
                json.dumps(
                    sanitized,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            if source_root_text in serialized:
                raise RuntimeError(f"private source root remains in {relative}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(serialized, encoding="utf-8")
            rows.append(
                {
                    "relative_path": relative,
                    "original_sha256": sha256(source),
                    "exported_sha256": sha256(destination),
                    "source_root_replacements": replacements,
                }
            )
        manifest = {
            "schema_version": 1,
            "campaign_name": campaign.name,
            "source_root_token": "${FROZEN_SOURCE_ROOT}",
            "scientific_values_changed": False,
            "only_transformation": "literal source-root path replacement",
            "files": rows,
        }
        (output / "provenance_export_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_campaign(
        args.campaign,
        args.output,
        source_root=args.source_root,
    )
    print(
        f"Provenance export PASS: {manifest['campaign_name']}, "
        f"{len(manifest['files'])} records."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

