#!/usr/bin/env python3
"""Project audited private JSON paths to a portable DOI-archive token.

The projection is deliberately narrower than a recursive string sanitizer.
Only the file/pointer pairs in ``ALLOWLIST`` may contain a private path, and
only one caller-supplied source prefix may be replaced.  Every other macOS,
Linux, or Windows home path fails closed before a byte is written.

The transformation is a literal byte substitution, so JSON formatting and all
non-path values remain byte-for-byte unchanged.  A deterministic manifest maps
the original and public file hashes, records every JSON pointer, guards the NPZ
inventory, and documents the bridge for historical hashes.  Full verification
reconstructs the original bytes from the public token using ``--source-prefix``.

The commit phase uses validated-in-memory outputs, atomic ``os.replace`` calls,
and rollback on an ordinary write failure.  The manifest is committed last;
an interrupted mixed state without a manifest is recoverable and idempotent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PUBLIC_ROOT_TOKEN = "${FROZEN_SOURCE_ROOT}"
MANIFEST_NAME = "path_projection_manifest.json"
DEFAULT_EXPECTED_FILES = 105
DEFAULT_EXPECTED_REPLACEMENTS = 1713
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_HOME_RE = re.compile(r"(?i)[a-z]:[\\/]+users[\\/]")


@dataclass(frozen=True)
class AllowRule:
    file_re: re.Pattern[str]
    pointer_res: tuple[re.Pattern[str], ...]


def _compiled(*values: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value) for value in values)


# This allowlist is the machine-readable result of the pre-publication path
# audit.  Array indices are the only wildcarded pointer component.
ALLOWLIST: tuple[AllowRule, ...] = (
    AllowRule(
        re.compile(
            r"^data/raw/estimator_benchmark/measurement_history/"
            r"episode_[0-9]{4}_seed_[0-9]+/capture_metadata\.json$"
        ),
        _compiled(
            r"^/provenance/evaluation_metadata_path$",
            r"^/provenance/noise_tape_path$",
            r"^/provenance/trace_path$",
        ),
    ),
    AllowRule(
        re.compile(
            r"^data/raw/estimator_benchmark/v27_campaign/control/"
            r"campaign_contract\.json$"
        ),
        _compiled(
            r"^/source_archive$",
            r"^/source_episodes/[0-9]+/(directory|metadata_path|online_path|trace_path|truth_path)$",
        ),
    ),
    AllowRule(
        re.compile(
            r"^data/raw/estimator_benchmark/v34_campaign/control/"
            r"campaign_contract\.json$"
        ),
        _compiled(
            r"^/smoke_prerequisite/directory$",
            r"^/source_archive$",
            r"^/source_episodes/[0-9]+/(directory|metadata_path|online_path|truth_path)$",
            r"^/v27_reference/directory$",
            r"^/v27_reference/global_full_cells/[0-9]+/path$",
        ),
    ),
    AllowRule(
        re.compile(
            r"^data/raw/leader_source_ablation/control/campaign_contract\.json$"
        ),
        _compiled(r"^/environment_metadata_path$"),
    ),
    AllowRule(
        re.compile(r"^data/raw/leader_source_ablation/independent_audit\.json$"),
        _compiled(r"^/campaign$"),
    ),
    AllowRule(
        re.compile(r"^data/raw/leader_source_ablation/vertical_datum_audit\.json$"),
        _compiled(
            r"^/campaign_dir$",
            r"^/clearance/campaign_contract$",
            r"^/clearance/campaign_dir$",
            r"^/clearance/progress_record$",
            r"^/environment_metadata$",
            r"^/output_json$",
            r"^/output_md$",
        ),
    ),
)


@dataclass(frozen=True)
class ProjectedFile:
    path: Path
    relpath: str
    current_bytes: bytes
    original_bytes: bytes
    public_bytes: bytes
    original_document: Any
    public_document: Any
    pointers: tuple[tuple[str, int], ...]
    canonical_scientific_sha256: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _escape_pointer_component(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def iter_json_strings(value: Any, components: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_json_strings(item, components + (_escape_pointer_component(str(key)),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_json_strings(item, components + (str(index),))
    elif isinstance(value, str):
        yield "/" + "/".join(components), value


def iter_json_keys(value: Any, components: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            pointer = "/" + "/".join(components + (_escape_pointer_component(key_text),))
            yield pointer, key_text
            yield from iter_json_keys(
                item, components + (_escape_pointer_component(key_text),)
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_json_keys(item, components + (str(index),))


def _is_allowed(relpath: str, pointer: str) -> bool:
    return any(
        rule.file_re.fullmatch(relpath)
        and any(pointer_re.fullmatch(pointer) for pointer_re in rule.pointer_res)
        for rule in ALLOWLIST
    )


def _contains_private_home(value: str) -> bool:
    return "/Users/" in value or "/home/" in value or WINDOWS_HOME_RE.search(value) is not None


def _has_prefix_boundary(value: str, prefix: str) -> bool:
    return value == prefix or value.startswith(prefix + "/")


def _replace_at_allowed_pointers(
    value: Any,
    *,
    relpath: str,
    source_prefix: str,
    replacement: str,
    components: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_at_allowed_pointers(
                item,
                relpath=relpath,
                source_prefix=source_prefix,
                replacement=replacement,
                components=components + (_escape_pointer_component(str(key)),),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_at_allowed_pointers(
                item,
                relpath=relpath,
                source_prefix=source_prefix,
                replacement=replacement,
                components=components + (str(index),),
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        pointer = "/" + "/".join(components)
        if _is_allowed(relpath, pointer) and _has_prefix_boundary(value, source_prefix):
            return replacement + value[len(source_prefix) :]
    return value


def _validate_source_prefix(source_prefix: str) -> str:
    value = source_prefix.rstrip("/")
    if not value or not value.startswith("/"):
        raise ValueError("--source-prefix must be an absolute POSIX path")
    if value == PUBLIC_ROOT_TOKEN or PUBLIC_ROOT_TOKEN in value:
        raise ValueError("--source-prefix cannot contain the public token")
    if not _contains_private_home(value + "/"):
        raise ValueError("--source-prefix must identify a private home path")
    return value


def _parse_json_bytes(raw: bytes, relpath: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid UTF-8 JSON in {relpath}: {exc}") from exc


def _project_one_json(
    path: Path,
    *,
    archive_root: Path,
    source_prefix: str,
) -> ProjectedFile | None:
    relpath = path.relative_to(archive_root).as_posix()
    current = path.read_bytes()
    document = _parse_json_bytes(current, relpath)
    source_occurrences: dict[str, int] = {}
    public_occurrences: dict[str, int] = {}

    for pointer, key in iter_json_keys(document):
        if _contains_private_home(key) or source_prefix in key:
            raise RuntimeError(f"private path is forbidden in a JSON key: {relpath}#{pointer}")

    for pointer, value in iter_json_strings(document):
        source_count = value.count(source_prefix)
        public_count = value.count(PUBLIC_ROOT_TOKEN)
        if _contains_private_home(value):
            if not _is_allowed(relpath, pointer):
                raise RuntimeError(
                    f"private path outside the audited allowlist: {relpath}#{pointer}"
                )
            if not _has_prefix_boundary(value, source_prefix) or source_count != 1:
                raise RuntimeError(
                    f"allowlisted path is not the exact approved prefix plus suffix: "
                    f"{relpath}#{pointer}"
                )
            # Reject a second private-home marker hidden in the suffix.
            if _contains_private_home(value[len(source_prefix) :]):
                raise RuntimeError(f"additional private path in {relpath}#{pointer}")
        if source_count:
            if not _is_allowed(relpath, pointer):
                raise RuntimeError(
                    f"approved source prefix outside the audited allowlist: {relpath}#{pointer}"
                )
            source_occurrences[pointer] = source_occurrences.get(pointer, 0) + source_count
        if public_count and _is_allowed(relpath, pointer):
            if not _has_prefix_boundary(value, PUBLIC_ROOT_TOKEN) or public_count != 1:
                raise RuntimeError(f"malformed public token in {relpath}#{pointer}")
            public_occurrences[pointer] = public_occurrences.get(pointer, 0) + public_count

    if not source_occurrences and not public_occurrences:
        return None

    source_bytes = source_prefix.encode("utf-8")
    public_bytes_token = PUBLIC_ROOT_TOKEN.encode("utf-8")
    decoded_source_count = sum(source_occurrences.values())
    decoded_public_count = sum(public_occurrences.values())
    if current.count(source_bytes) != decoded_source_count:
        raise RuntimeError(f"source prefix is JSON-escaped or occurs outside a string: {relpath}")
    if current.count(public_bytes_token) < decoded_public_count:
        raise RuntimeError(f"public token byte count is inconsistent: {relpath}")

    original = current.replace(public_bytes_token, source_bytes)
    public = current.replace(source_bytes, public_bytes_token)
    original_document = _parse_json_bytes(original, relpath)
    public_document = _parse_json_bytes(public, relpath)
    normalized_original = _replace_at_allowed_pointers(
        original_document,
        relpath=relpath,
        source_prefix=source_prefix,
        replacement=PUBLIC_ROOT_TOKEN,
    )
    normalized_public = _replace_at_allowed_pointers(
        public_document,
        relpath=relpath,
        source_prefix=PUBLIC_ROOT_TOKEN,
        replacement=PUBLIC_ROOT_TOKEN,
    )
    if normalized_original != normalized_public:
        raise RuntimeError(f"a non-path JSON value would change in {relpath}")
    scientific_hash = canonical_json_sha256(normalized_original)

    merged: dict[str, int] = dict(source_occurrences)
    for pointer, count in public_occurrences.items():
        merged[pointer] = merged.get(pointer, 0) + count
    return ProjectedFile(
        path=path,
        relpath=relpath,
        current_bytes=current,
        original_bytes=original,
        public_bytes=public,
        original_document=original_document,
        public_document=public_document,
        pointers=tuple(sorted(merged.items())),
        canonical_scientific_sha256=scientific_hash,
    )


def _iter_key_values(value: Any, key_name: str) -> Iterator[Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == key_name:
                yield item
            yield from _iter_key_values(item, key_name)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_key_values(item, key_name)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {label}")
    return value


def _build_internal_hash_bridge(
    archive_root: Path,
    projected_by_relpath: Mapping[str, ProjectedFile],
) -> dict[str, Any]:
    contract_relpaths = (
        "data/raw/estimator_benchmark/v27_campaign/control/campaign_contract.json",
        "data/raw/estimator_benchmark/v34_campaign/control/campaign_contract.json",
    )
    self_checks: list[dict[str, Any]] = []
    contracts: dict[str, dict[str, Any]] = {}
    for relpath in contract_relpaths:
        record = projected_by_relpath.get(relpath)
        if record is None:
            raise RuntimeError(f"audited projected contract is missing: {relpath}")
        original = _require_object(copy.deepcopy(record.original_document), relpath)
        public = _require_object(copy.deepcopy(record.public_document), relpath)
        stored = str(original.pop("contract_sha256", ""))
        public.pop("contract_sha256", None)
        reconstructed = canonical_json_sha256(original)
        projected = canonical_json_sha256(public)
        if SHA256_RE.fullmatch(stored) is None or reconstructed != stored:
            raise RuntimeError(f"reconstructed contract self-hash mismatch: {relpath}")
        contracts[relpath] = _require_object(record.original_document, relpath)
        self_checks.append(
            {
                "relpath": relpath,
                "stored_original_contract_sha256": stored,
                "reconstructed_original_contract_sha256": reconstructed,
                "public_projected_contract_sha256": projected,
                "valid": True,
            }
        )

    metadata_original_hashes = {
        Path(relpath).parent.name: sha256_bytes(record.original_bytes)
        for relpath, record in projected_by_relpath.items()
        if relpath.endswith("/capture_metadata.json")
    }
    metadata_reference_count = 0
    metadata_targets: set[str] = set()
    for relpath in contract_relpaths:
        contract = contracts[relpath]
        episodes = contract.get("source_episodes")
        if not isinstance(episodes, list):
            raise RuntimeError(f"source_episodes is missing from {relpath}")
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                raise RuntimeError(f"invalid source_episodes/{index} in {relpath}")
            directory = Path(str(episode.get("directory", ""))).name
            expected = str(episode.get("metadata_sha256", ""))
            observed = metadata_original_hashes.get(directory)
            if observed is None or observed != expected:
                raise RuntimeError(
                    f"historical metadata hash bridge mismatch: {relpath} episode {index}"
                )
            metadata_targets.add(directory)
            metadata_reference_count += 1

    v27_relpath, v34_relpath = contract_relpaths
    v34_reference = _require_object(
        contracts[v34_relpath].get("v27_reference"), f"{v34_relpath}#/v27_reference"
    )
    expected_v27_file_hash = str(v34_reference.get("campaign_contract_sha256", ""))
    observed_v27_file_hash = sha256_bytes(projected_by_relpath[v27_relpath].original_bytes)
    if expected_v27_file_hash != observed_v27_file_hash:
        raise RuntimeError("V34-to-V27 original contract file-hash bridge mismatch")

    dependent_checks: list[dict[str, Any]] = []
    for relpath in contract_relpaths:
        campaign_root = archive_root / Path(relpath).parent.parent
        stored = str(contracts[relpath]["contract_sha256"])
        count = 0
        for path in sorted(campaign_root.rglob("*.json")):
            if path == archive_root / relpath:
                continue
            document = _parse_json_bytes(
                path.read_bytes(), path.relative_to(archive_root).as_posix()
            )
            for value in _iter_key_values(document, "contract_sha256"):
                count += 1
                if str(value) != stored:
                    raise RuntimeError(
                        f"dependent contract_sha256 mismatch in "
                        f"{path.relative_to(archive_root).as_posix()}"
                    )
        dependent_checks.append(
            {
                "contract_relpath": relpath,
                "dependent_contract_sha256_values_checked": count,
                "valid": True,
            }
        )

    return {
        "strategy": (
            "Historical hash fields remain unchanged. The validator replaces the public "
            "token by the caller-supplied original prefix only at audited JSON pointers, "
            "then verifies original bytes and canonical contract hashes."
        ),
        "self_hashed_contracts": self_checks,
        "capture_metadata_original_sha256_references": {
            "references_checked": metadata_reference_count,
            "unique_targets_checked": len(metadata_targets),
            "valid": True,
        },
        "v34_to_v27_original_contract_file_sha256": {
            "expected_sha256": expected_v27_file_hash,
            "reconstructed_original_sha256": observed_v27_file_hash,
            "valid": True,
        },
        "dependent_contract_sha256_values": dependent_checks,
        "valid": True,
    }


def _inventory_npz(archive_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(archive_root.rglob("*.npz")):
        relpath = path.relative_to(archive_root).as_posix()
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(f"{relpath}\0{size}\0{file_hash}\n".encode("utf-8"))
        count += 1
        total_bytes += size
    return {
        "file_count": count,
        "total_bytes": total_bytes,
        "canonical_inventory_sha256": digest.hexdigest(),
        "modified_by_projection": False,
    }


def _scan_non_json_private_paths(archive_root: Path, manifest_path: Path) -> None:
    byte_patterns = (b"/Users/", b"/home/")
    windows_re = re.compile(rb"(?i)[a-z]:[\\/]+users[\\/]")
    for path in sorted(item for item in archive_root.rglob("*") if item.is_file()):
        if path.suffix.lower() == ".json" or path == manifest_path:
            continue
        with path.open("rb") as stream:
            carry = b""
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                sample = carry + chunk
                if any(pattern in sample for pattern in byte_patterns) or windows_re.search(sample):
                    raise RuntimeError(
                        f"private home path found in non-JSON payload: "
                        f"{path.relative_to(archive_root).as_posix()}"
                    )
                carry = sample[-32:]


def build_projection(
    archive_root: Path,
    *,
    source_prefix: str,
    expected_files: int | None = None,
    expected_replacements: int | None = None,
) -> tuple[dict[str, Any], list[ProjectedFile], dict[str, Any]]:
    archive_root = archive_root.expanduser().resolve()
    if not archive_root.is_dir():
        raise FileNotFoundError(archive_root)
    source_prefix = _validate_source_prefix(source_prefix)
    manifest_path = archive_root / MANIFEST_NAME
    symlinks = sorted(path for path in archive_root.rglob("*") if path.is_symlink())
    if symlinks:
        raise RuntimeError(
            "DOI archive must not contain symbolic links: "
            + ", ".join(path.relative_to(archive_root).as_posix() for path in symlinks[:10])
        )
    _scan_non_json_private_paths(archive_root, manifest_path)

    records: list[ProjectedFile] = []
    for path in sorted(archive_root.rglob("*.json")):
        if path == manifest_path:
            continue
        projected = _project_one_json(
            path,
            archive_root=archive_root,
            source_prefix=source_prefix,
        )
        if projected is not None:
            records.append(projected)

    replacement_count = sum(sum(count for _, count in record.pointers) for record in records)
    if expected_files is not None and len(records) != int(expected_files):
        raise RuntimeError(
            f"audited projected-file count mismatch: {len(records)} != {expected_files}"
        )
    if expected_replacements is not None and replacement_count != int(expected_replacements):
        raise RuntimeError(
            f"audited replacement count mismatch: {replacement_count} != "
            f"{expected_replacements}"
        )

    projected_by_relpath = {record.relpath: record for record in records}
    if len(projected_by_relpath) != len(records):
        raise RuntimeError("duplicate projected relative path")
    bridge = _build_internal_hash_bridge(archive_root, projected_by_relpath)
    npz_inventory = _inventory_npz(archive_root)
    rows = []
    for record in records:
        rows.append(
            {
                "relpath": record.relpath,
                "original_sha256": sha256_bytes(record.original_bytes),
                "public_sha256": sha256_bytes(record.public_bytes),
                "canonical_scientific_sha256": record.canonical_scientific_sha256,
                "canonical_scientific_hash_basis": (
                    "canonical UTF-8 JSON with sorted keys and compact separators after "
                    "normalizing only audited source-root prefixes to ${FROZEN_SOURCE_ROOT}"
                ),
                "json_pointers": [
                    {"pointer": pointer, "replacement_count": count}
                    for pointer, count in record.pointers
                ],
                "replacement_count": sum(count for _, count in record.pointers),
                "scientific_values_changed": False,
            }
        )
    manifest = {
        "schema_version": 1,
        "projection": "audited DOI-archive private-path projection",
        "public_source_root_token": PUBLIC_ROOT_TOKEN,
        "source_prefix_sha256": sha256_bytes(source_prefix.encode("utf-8")),
        "deterministic": True,
        "scientific_values_changed": False,
        "only_transformation": (
            "literal source-prefix replacement at the enumerated allowlisted JSON pointers; "
            "original JSON formatting is preserved"
        ),
        "audited_expectations": {
            "projected_file_count": expected_files,
            "replacement_count": expected_replacements,
        },
        "totals": {
            "projected_file_count": len(records),
            "json_pointer_count": sum(len(record.pointers) for record in records),
            "replacement_count": replacement_count,
        },
        "npz_guard": npz_inventory,
        "internal_hash_bridge": bridge,
        "files": rows,
    }
    return manifest, records, npz_inventory


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_projection(
    archive_root: Path,
    *,
    source_prefix: str,
    expected_files: int | None = DEFAULT_EXPECTED_FILES,
    expected_replacements: int | None = DEFAULT_EXPECTED_REPLACEMENTS,
) -> dict[str, Any]:
    archive_root = archive_root.expanduser().resolve()
    manifest, records, npz_before = build_projection(
        archive_root,
        source_prefix=source_prefix,
        expected_files=expected_files,
        expected_replacements=expected_replacements,
    )
    manifest_path = archive_root / MANIFEST_NAME
    expected_manifest_bytes = _manifest_bytes(manifest)
    old_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
    if old_manifest is not None and old_manifest != expected_manifest_bytes:
        raise RuntimeError("existing path-projection manifest does not match the archive")

    writes = [(record.path, record.current_bytes, record.public_bytes) for record in records]
    writes = [row for row in writes if row[1] != row[2]]
    committed: list[tuple[Path, bytes]] = []
    manifest_written = False
    try:
        for path, old_bytes, public_bytes in writes:
            _atomic_write(path, public_bytes)
            committed.append((path, old_bytes))
        if old_manifest is None:
            _atomic_write(manifest_path, expected_manifest_bytes)
            manifest_written = True
        npz_after = _inventory_npz(archive_root)
        if npz_after != npz_before:
            raise RuntimeError("NPZ inventory changed during path projection")
        # A complete second build is both the postcondition check and the
        # idempotency proof for the just-written public view.
        post_manifest, post_records, _ = build_projection(
            archive_root,
            source_prefix=source_prefix,
            expected_files=expected_files,
            expected_replacements=expected_replacements,
        )
        if _manifest_bytes(post_manifest) != expected_manifest_bytes:
            raise RuntimeError("post-projection deterministic manifest mismatch")
        if any(record.current_bytes != record.public_bytes for record in post_records):
            raise RuntimeError("private source prefix remains after projection")
    except Exception:
        for path, old_bytes in reversed(committed):
            _atomic_write(path, old_bytes)
        if manifest_written and manifest_path.exists():
            manifest_path.unlink()
        elif old_manifest is not None and manifest_path.read_bytes() != old_manifest:
            _atomic_write(manifest_path, old_manifest)
        raise

    result = dict(manifest)
    result["application"] = {
        "json_files_written": len(writes),
        "manifest_written": manifest_written,
        "idempotent_noop": not writes and not manifest_written,
    }
    return result


def verify_projection(
    archive_root: Path,
    *,
    source_prefix: str,
    expected_files: int | None = DEFAULT_EXPECTED_FILES,
    expected_replacements: int | None = DEFAULT_EXPECTED_REPLACEMENTS,
) -> dict[str, Any]:
    archive_root = archive_root.expanduser().resolve()
    manifest_path = archive_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"projection manifest is missing: {manifest_path}")
    manifest, records, _ = build_projection(
        archive_root,
        source_prefix=source_prefix,
        expected_files=expected_files,
        expected_replacements=expected_replacements,
    )
    if manifest_path.read_bytes() != _manifest_bytes(manifest):
        raise RuntimeError("projection manifest or projected files fail verification")
    if any(record.current_bytes != record.public_bytes for record in records):
        raise RuntimeError("archive still contains the private source prefix")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("apply", "verify"))
    parser.add_argument("archive_root", type=Path)
    parser.add_argument(
        "--source-prefix",
        required=True,
        help=(
            "Exact private prefix to reconstruct/replace. It is never written to the "
            "public manifest; only its SHA-256 is recorded."
        ),
    )
    parser.add_argument("--expected-files", type=int, default=DEFAULT_EXPECTED_FILES)
    parser.add_argument(
        "--expected-replacements", type=int, default=DEFAULT_EXPECTED_REPLACEMENTS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    function = apply_projection if args.mode == "apply" else verify_projection
    result = function(
        args.archive_root,
        source_prefix=args.source_prefix,
        expected_files=args.expected_files,
        expected_replacements=args.expected_replacements,
    )
    totals = result["totals"]
    if args.mode == "apply":
        application = result["application"]
        print(
            "DOI path projection PASS: "
            f"{totals['projected_file_count']} JSON files, "
            f"{totals['replacement_count']} replacements, "
            f"{application['json_files_written']} JSON writes, "
            f"idempotent_noop={str(application['idempotent_noop']).lower()}."
        )
    else:
        print(
            "DOI path projection verification PASS: "
            f"{totals['projected_file_count']} JSON files, "
            f"{totals['replacement_count']} reconstructed replacements, "
            "scientific_values_changed=false."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
