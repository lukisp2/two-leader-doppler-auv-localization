#!/usr/bin/env python3
"""Export one sanitized V18.1 trace/tape pair for replay verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


CONTROLLER = "deterministic_greedy_grid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evaluation_directory",
        type=Path,
        help="Frozen V18.1 dev100 evaluation directory.",
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-seed", type=int, default=45_000)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Private source prefix to replace by ${FROZEN_SOURCE_ROOT}.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(value: Any, source_root: str) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item, source_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, source_root) for item in value]
    if isinstance(value, str):
        return value.replace(source_root, "${FROZEN_SOURCE_ROOT}")
    return value


def main() -> int:
    args = parse_args()
    evaluation = args.evaluation_directory.expanduser().resolve()
    output = args.output_directory.expanduser().resolve()
    source_root = str(args.source_root.expanduser().resolve())

    expected_seed = 45_000 + int(args.episode_index)
    if int(args.episode_seed) != expected_seed:
        raise RuntimeError("episode seed must equal 45000 + episode index")

    metadata_source = evaluation / "metadata.json"
    trace_source = (
        evaluation
        / "traces_npz"
        / CONTROLLER
        / f"episode_{args.episode_index:04d}_seed_{args.episode_seed}.npz"
    )
    tape_source = (
        evaluation
        / "noise_tapes"
        / f"episode_{args.episode_index:04d}_seed_{args.episode_seed}.npz"
    )
    for path in (metadata_source, trace_source, tape_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    sanitized = sanitize(metadata, source_root)
    serialized = json.dumps(
        sanitized,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    if source_root in serialized:
        raise RuntimeError("private source prefix remains in sanitized metadata")

    trace_destination = output / "traces_npz" / CONTROLLER / trace_source.name
    tape_destination = output / "noise_tapes" / tape_source.name
    trace_destination.parent.mkdir(parents=True, exist_ok=True)
    tape_destination.parent.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(serialized, encoding="utf-8")
    shutil.copy2(trace_source, trace_destination)
    shutil.copy2(tape_source, tape_destination)

    print(f"metadata_sha256={sha256(output / 'metadata.json')}")
    print(f"trace_sha256={sha256(trace_destination)}")
    print(f"tape_sha256={sha256(tape_destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
