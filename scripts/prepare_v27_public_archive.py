#!/usr/bin/env python3
"""Create a hash-checked portable view of the frozen V27 measurement archive.

Historical capture metadata records the original workstation path of each
legacy particle-filter trace.  This helper changes only that provenance path
in a generated archive view, after matching episode/seed identity and the
recorded SHA-256 digest against the separately released trace directory.  The
frozen V27 runner itself remains byte-identical to the publication snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


EPISODE_RE = re.compile(r"^episode_(\d{4})_seed_(\d+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_EPISODES = 100
DEFAULT_SEED_START = 45_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def project_archive(
    source_archive: Path,
    legacy_trace_dir: Path,
    output_archive: Path,
    *,
    expected_episodes: int = DEFAULT_EPISODES,
    seed_start: int = DEFAULT_SEED_START,
) -> dict[str, Any]:
    source_archive = source_archive.expanduser().resolve()
    legacy_trace_dir = legacy_trace_dir.expanduser().resolve()
    output_archive = output_archive.expanduser().resolve()
    if not source_archive.is_dir():
        raise FileNotFoundError(f"measurement archive is missing: {source_archive}")
    if not legacy_trace_dir.is_dir():
        raise FileNotFoundError(f"legacy trace directory is missing: {legacy_trace_dir}")
    if output_archive == source_archive or output_archive in source_archive.parents:
        raise RuntimeError("output archive must be disjoint from the source archive")
    if output_archive.exists():
        raise RuntimeError(f"output archive already exists: {output_archive}")

    directories = sorted(path for path in source_archive.iterdir() if path.is_dir())
    expected_indices = set(range(int(expected_episodes)))
    observed_indices: set[int] = set()
    rows: list[dict[str, Any]] = []

    output_archive.mkdir(parents=True)
    try:
        for directory in directories:
            match = EPISODE_RE.fullmatch(directory.name)
            if match is None:
                raise RuntimeError(f"unexpected measurement-archive directory: {directory.name}")
            episode_index = int(match.group(1))
            episode_seed = int(match.group(2))
            if episode_index in observed_indices:
                raise RuntimeError(f"duplicate episode index: {episode_index}")
            observed_indices.add(episode_index)
            if episode_seed != int(seed_start) + episode_index:
                raise RuntimeError(f"episode/seed identity mismatch: {directory.name}")

            metadata_path = directory / "capture_metadata.json"
            online_path = directory / "online_inputs.npz"
            truth_path = directory / "truth_labels.npz"
            metadata = read_json(metadata_path)
            if int(metadata.get("episode_index", -1)) != episode_index:
                raise RuntimeError(f"metadata episode mismatch: {metadata_path}")
            if int(metadata.get("episode_seed", -1)) != episode_seed:
                raise RuntimeError(f"metadata seed mismatch: {metadata_path}")
            if not online_path.is_file() or not truth_path.is_file():
                raise RuntimeError(f"measurement payload is incomplete: {directory}")
            online_sha = sha256_file(online_path)
            truth_sha = sha256_file(truth_path)
            if online_sha != str(metadata.get("online_inputs_sha256")):
                raise RuntimeError(f"online-input hash mismatch: {online_path}")
            if truth_sha != str(metadata.get("truth_labels_sha256")):
                raise RuntimeError(f"truth-label hash mismatch: {truth_path}")

            provenance = metadata.get("provenance")
            if not isinstance(provenance, dict):
                raise RuntimeError(f"metadata lacks provenance: {metadata_path}")
            recorded_trace_sha = str(provenance.get("trace_sha256", ""))
            if SHA256_RE.fullmatch(recorded_trace_sha) is None:
                raise RuntimeError(f"invalid recorded trace hash: {metadata_path}")
            trace = legacy_trace_dir / f"episode_{episode_index:04d}_seed_{episode_seed}.npz"
            if not trace.is_file() or trace.resolve().parent != legacy_trace_dir:
                raise RuntimeError(f"released legacy trace is missing: {trace.name}")
            observed_trace_sha = sha256_file(trace)
            if observed_trace_sha != recorded_trace_sha:
                raise RuntimeError(f"legacy trace hash mismatch: {trace}")

            destination = output_archive / directory.name
            destination.mkdir()
            online_mode = _link_or_copy(online_path, destination / online_path.name)
            truth_mode = _link_or_copy(truth_path, destination / truth_path.name)
            original_metadata_sha = sha256_file(metadata_path)
            projected = dict(metadata)
            projected_provenance = dict(provenance)
            projected_provenance["trace_path"] = str(trace)
            projected["provenance"] = projected_provenance
            projected["public_portability_projection"] = {
                "schema_version": 1,
                "original_capture_metadata_sha256": original_metadata_sha,
                "changed_scientific_values": False,
                "changed_field": "provenance.trace_path",
                "released_trace_sha256": observed_trace_sha,
            }
            projected_path = destination / metadata_path.name
            projected_path.write_text(
                json.dumps(projected, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            rows.append(
                {
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "source_directory": directory.name,
                    "original_capture_metadata_sha256": original_metadata_sha,
                    "projected_capture_metadata_sha256": sha256_file(projected_path),
                    "online_inputs_sha256": online_sha,
                    "truth_labels_sha256": truth_sha,
                    "legacy_trace_file": trace.name,
                    "legacy_trace_sha256": observed_trace_sha,
                    "payload_materialization": {
                        "online_inputs": online_mode,
                        "truth_labels": truth_mode,
                    },
                }
            )

        if observed_indices != expected_indices:
            missing = sorted(expected_indices - observed_indices)
            extra = sorted(observed_indices - expected_indices)
            raise RuntimeError(
                f"episode support mismatch: missing={missing[:10]}, extra={extra[:10]}"
            )

        manifest = {
            "schema_version": 1,
            "purpose": "portable path projection for the byte-frozen V27 runner",
            "episode_count": len(rows),
            "episode_indices": sorted(observed_indices),
            "seed_start": int(seed_start),
            "scientific_arrays_reused_without_change": True,
            "only_metadata_change": "provenance.trace_path",
            "rows": sorted(rows, key=lambda row: int(row["episode_index"])),
        }
        (output_archive / "public_projection_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output_archive, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--legacy-trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    args = parser.parse_args()
    result = project_archive(
        args.archive_dir,
        args.legacy_trace_dir,
        args.output_dir,
        expected_episodes=args.expected_episodes,
        seed_start=args.seed_start,
    )
    print(
        "Portable V27 input archive PASS: "
        f"{result['episode_count']} episodes and every released legacy trace hash matched."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

