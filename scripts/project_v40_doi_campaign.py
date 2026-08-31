#!/usr/bin/env python3
"""Create a fail-closed, path-sanitized DOI projection of frozen V40.

The exporter copies the complete campaign without changing the private source.
Every campaign file except ``control/campaign_contract.json`` is required to
remain byte-for-byte identical. In that contract, and only in that contract,
``environment_metadata_path`` is replaced by a portable token. The approved
public environment-metadata projection is added below ``control/`` and a
file-level original/exported SHA-256 manifest is written at the campaign root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PORTABLE_ENVIRONMENT_METADATA_PATH = (
    "${CAMPAIGN_ROOT}/control/environment_metadata.json"
)
PROJECTION_MANIFEST_NAME = "public_projection_manifest.json"

ORIGINAL_ENVIRONMENT_METADATA_SHA256 = (
    "b486d280e57661f4b526b926e0f8e32a5373b99f585de09e3354d87b3342dfe7"
)
PUBLIC_ENVIRONMENT_METADATA_SHA256 = (
    "33aff4c6794c9eab2d92bd4c9aa2f3ff3b89b4f3fc121904df40124f584b626f"
)
FROZEN_ENVIRONMENT_CONFIG_SHA256 = (
    "74e83dae26f198604803d50f2da3c14453118e9418e41dc3316684391eb7e211"
)

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MAC_HOME_PREFIX = "/" + "Users/"
UNIX_HOME_PREFIX = "/" + "home/"
PRIVATE_ABSOLUTE_PATH = re.compile(
    rf"(?:{re.escape(MAC_HOME_PREFIX)}|{re.escape(UNIX_HOME_PREFIX)})[^\s\"']+"
    r"|[A-Za-z]:\\Users\\[^\s\"']+"
)


@dataclass(frozen=True)
class ExpectedCounts:
    """Frozen V40 campaign cardinalities checked before and after export."""

    episode_json: int = 800
    trace_npz: int = 800
    noise_npz: int = 100
    source_snapshot_files: int = 14


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def regular_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are not allowed: {path}")
        if path.is_file():
            paths.append(path)
    return paths


def relative_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in regular_files(root)
    }


def private_path_hits(root: Path) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for path in regular_files(root):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"declared text file is not UTF-8: {path}") from exc
        for match in PRIVATE_ABSOLUTE_PATH.finditer(text):
            hits.append((path.relative_to(root).as_posix(), match.group(0)))
    return hits


def private_path_hits_in_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot inspect text file: {path}") from exc
    return [match.group(0) for match in PRIVATE_ABSOLUTE_PATH.finditer(text)]


def _count_files(root: Path, directory: str, suffix: str) -> int:
    return sum(
        1
        for path in (root / directory).rglob(f"*{suffix}")
        if path.is_file() and not path.is_symlink()
    )


def validate_campaign_cardinality(root: Path, expected: ExpectedCounts) -> None:
    checks = {
        "episode JSON": (
            _count_files(root, "episode_results", ".json"),
            expected.episode_json,
        ),
        "trace NPZ": (
            _count_files(root, "traces_npz", ".npz"),
            expected.trace_npz,
        ),
        "noise NPZ": (
            _count_files(root, "noise_tapes", ".npz"),
            expected.noise_npz,
        ),
        "source snapshot": (
            len(regular_files(root / "control/source_snapshot")),
            expected.source_snapshot_files,
        ),
    }
    for label, (actual, required) in checks.items():
        if actual != required:
            raise RuntimeError(
                f"unexpected {label} count: expected {required}, found {actual}"
            )

    expected_extensions = {
        "episode_results": ".json",
        "traces_npz": ".npz",
        "noise_tapes": ".npz",
    }
    for directory, extension in expected_extensions.items():
        unexpected = [
            path
            for path in regular_files(root / directory)
            if path.suffix.lower() != extension
        ]
        if unexpected:
            raise RuntimeError(f"unexpected file in {directory}: {unexpected[0]}")


def _replace_contract_path(
    source: Path,
    destination: Path,
    *,
    original_path: str,
) -> None:
    source_bytes = source.read_bytes()
    original_literal = json.dumps(original_path, ensure_ascii=False)[1:-1].encode(
        "utf-8"
    )
    replacement_literal = json.dumps(
        PORTABLE_ENVIRONMENT_METADATA_PATH,
        ensure_ascii=False,
    )[1:-1].encode("utf-8")
    if source_bytes.count(original_literal) != 1:
        raise RuntimeError(
            "the recorded environment_metadata_path is not a unique literal in "
            "the contract"
        )
    destination.write_bytes(source_bytes.replace(original_literal, replacement_literal))


def _assert_only_contract_path_changed(
    source_document: Mapping[str, Any],
    exported_document: Mapping[str, Any],
) -> None:
    expected = copy.deepcopy(dict(source_document))
    expected["environment_metadata_path"] = PORTABLE_ENVIRONMENT_METADATA_PATH
    if dict(exported_document) != expected:
        raise RuntimeError(
            "exported campaign contract differs beyond environment_metadata_path"
        )


def _file_row(
    *,
    relative_path: str,
    original_sha256: str,
    exported_sha256: str,
    original_size_bytes: int,
    exported_size_bytes: int,
    source_kind: str,
    transformation: str,
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "source_kind": source_kind,
        "original_sha256": original_sha256,
        "exported_sha256": exported_sha256,
        "original_size_bytes": int(original_size_bytes),
        "exported_size_bytes": int(exported_size_bytes),
        "byte_for_byte_preserved": original_sha256 == exported_sha256,
        "transformation": transformation,
    }


def project_campaign(
    campaign: Path,
    output: Path,
    public_environment_metadata: Path,
    *,
    expected_counts: ExpectedCounts = ExpectedCounts(),
    expected_original_metadata_sha256: str = ORIGINAL_ENVIRONMENT_METADATA_SHA256,
    expected_public_metadata_sha256: str = PUBLIC_ENVIRONMENT_METADATA_SHA256,
    expected_environment_config_sha256: str = FROZEN_ENVIRONMENT_CONFIG_SHA256,
) -> dict[str, Any]:
    """Create and verify one immutable public projection.

    The destination must not exist. It is populated via a sibling temporary
    directory and atomically renamed only after every preservation, privacy,
    and manifest check has passed.
    """

    campaign = campaign.expanduser().resolve()
    output = output.expanduser().resolve()
    metadata_source = public_environment_metadata.expanduser().resolve()
    if not campaign.is_dir():
        raise FileNotFoundError(campaign)
    if not metadata_source.is_file():
        raise FileNotFoundError(metadata_source)
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if output == campaign or campaign in output.parents or output in campaign.parents:
        raise RuntimeError("source and output campaign trees must be disjoint")

    validate_campaign_cardinality(campaign, expected_counts)
    source_hashes_before = relative_file_hashes(campaign)
    if PROJECTION_MANIFEST_NAME in source_hashes_before:
        raise RuntimeError(f"private campaign already contains {PROJECTION_MANIFEST_NAME}")

    contract_relative = "control/campaign_contract.json"
    contract_source = campaign / contract_relative
    contract = read_json_object(contract_source)
    if int(contract.get("expected_runs", -1)) != expected_counts.episode_json:
        raise RuntimeError("campaign contract expected_runs differs from frozen count")
    recorded_metadata_path = contract.get("environment_metadata_path")
    if not isinstance(recorded_metadata_path, str) or not recorded_metadata_path:
        raise RuntimeError("contract lacks environment_metadata_path")
    if contract.get("environment_metadata_sha256") != expected_original_metadata_sha256:
        raise RuntimeError("contract environment metadata hash is not the frozen hash")

    source_private_hits = private_path_hits(campaign)
    expected_hit = (contract_relative, recorded_metadata_path)
    if source_private_hits != [expected_hit]:
        raise RuntimeError(
            "private campaign must contain exactly its one recorded environment "
            f"metadata path; found {source_private_hits!r}"
        )

    public_metadata_hash = sha256_file(metadata_source)
    if public_metadata_hash != expected_public_metadata_sha256:
        raise RuntimeError("approved public environment metadata hash mismatch")
    public_metadata = read_json_object(metadata_source)
    environment_config = public_metadata.get("environment_config")
    if not isinstance(environment_config, dict):
        raise RuntimeError("public environment metadata lacks environment_config")
    if canonical_sha256(environment_config) != expected_environment_config_sha256:
        raise RuntimeError("public environment_config differs from the frozen config")
    if private_path_hits_in_file(metadata_source):
        raise RuntimeError("approved public environment metadata contains a private path")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        shutil.copytree(
            campaign,
            temporary,
            dirs_exist_ok=True,
            copy_function=shutil.copy2,
        )
        contract_destination = temporary / contract_relative
        _replace_contract_path(
            contract_source,
            contract_destination,
            original_path=recorded_metadata_path,
        )
        exported_contract = read_json_object(contract_destination)
        _assert_only_contract_path_changed(contract, exported_contract)

        metadata_destination = temporary / "control/environment_metadata.json"
        if metadata_destination.exists():
            raise RuntimeError("campaign already contains control/environment_metadata.json")
        shutil.copy2(metadata_source, metadata_destination)

        source_hashes_after = relative_file_hashes(campaign)
        if source_hashes_after != source_hashes_before:
            raise RuntimeError("private campaign changed while projection was being built")

        rows: list[dict[str, Any]] = []
        for relative, original_hash in sorted(source_hashes_before.items()):
            source_path = campaign / relative
            exported_path = temporary / relative
            exported_hash = sha256_file(exported_path)
            if relative != contract_relative and exported_hash != original_hash:
                raise RuntimeError(f"byte-for-byte preservation failure: {relative}")
            rows.append(
                _file_row(
                    relative_path=relative,
                    original_sha256=original_hash,
                    exported_sha256=exported_hash,
                    original_size_bytes=source_path.stat().st_size,
                    exported_size_bytes=exported_path.stat().st_size,
                    source_kind="frozen_campaign",
                    transformation=(
                        "environment_metadata_path replaced by portable token"
                        if relative == contract_relative
                        else "none"
                    ),
                )
            )

        rows.append(
            _file_row(
                relative_path="control/environment_metadata.json",
                original_sha256=public_metadata_hash,
                exported_sha256=sha256_file(metadata_destination),
                original_size_bytes=metadata_source.stat().st_size,
                exported_size_bytes=metadata_destination.stat().st_size,
                source_kind="approved_path_sanitized_environment_metadata",
                transformation="none; copied from the approved public projection",
            )
        )

        unchanged_campaign_rows = [
            row
            for row in rows
            if row["source_kind"] == "frozen_campaign"
            and row["relative_path"] != contract_relative
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "campaign_name": campaign.name,
            "projection_kind": "path-sanitized complete DOI campaign",
            "manifest_scope": (
                "all exported files except this public_projection_manifest.json"
            ),
            "scientific_values_changed": False,
            "portable_environment_metadata_path": PORTABLE_ENVIRONMENT_METADATA_PATH,
            "source_campaign_file_count": len(source_hashes_before),
            "exported_file_count_excluding_manifest": len(rows),
            "frozen_cardinality": {
                "episode_json": expected_counts.episode_json,
                "trace_npz": expected_counts.trace_npz,
                "noise_npz": expected_counts.noise_npz,
                "source_snapshot_files": expected_counts.source_snapshot_files,
            },
            "preservation_checks": {
                "unchanged_campaign_files": len(unchanged_campaign_rows),
                "all_unchanged_campaign_files_byte_for_byte": all(
                    row["byte_for_byte_preserved"]
                    for row in unchanged_campaign_rows
                ),
                "all_episode_json_byte_for_byte": True,
                "all_trace_npz_byte_for_byte": True,
                "all_noise_npz_byte_for_byte": True,
                "source_snapshot_byte_for_byte": True,
                "private_source_unchanged_during_export": True,
            },
            "environment_metadata": {
                "frozen_private_original_sha256": expected_original_metadata_sha256,
                "approved_public_projection_sha256": public_metadata_hash,
                "frozen_environment_config_sha256": expected_environment_config_sha256,
                "environment_config_preserved": True,
            },
            "transformations": [
                {
                    "relative_path": contract_relative,
                    "field": "environment_metadata_path",
                    "replacement": PORTABLE_ENVIRONMENT_METADATA_PATH,
                    "scientific_value_changed": False,
                },
                {
                    "relative_path": "control/environment_metadata.json",
                    "action": "added approved path-sanitized public projection",
                    "scientific_value_changed": False,
                },
            ],
            "files": rows,
        }
        manifest_path = temporary / PROJECTION_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        remaining_private_paths = private_path_hits(temporary)
        if remaining_private_paths:
            raise RuntimeError(
                f"private absolute path remains in export: {remaining_private_paths!r}"
            )
        validate_campaign_cardinality(temporary, expected_counts)
        if sha256_file(metadata_destination) != expected_public_metadata_sha256:
            raise RuntimeError("exported environment metadata changed during copy")
        if len(regular_files(temporary)) != len(source_hashes_before) + 2:
            raise RuntimeError("unexpected exported file count")

        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="immutable completed V40 campaign")
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
    manifest = project_campaign(
        args.campaign,
        args.output,
        args.public_environment_metadata,
    )
    print(
        "V40 DOI projection PASS: "
        f"{manifest['source_campaign_file_count']} source files, "
        f"{manifest['exported_file_count_excluding_manifest']} exported files "
        "plus manifest; scientific_values_changed=false."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
