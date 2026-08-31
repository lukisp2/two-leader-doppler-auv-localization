from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_v27_public_archive.py"
SPEC = importlib.util.spec_from_file_location("prepare_v27_public_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path, *, trace_hash_override: str | None = None) -> tuple[Path, Path]:
    source = tmp_path / "measurement_history"
    traces = tmp_path / "legacy_pf_traces"
    episode = source / "episode_0000_seed_45000"
    episode.mkdir(parents=True)
    traces.mkdir()
    online = b"online scientific array"
    truth = b"truth labels"
    trace = b"legacy particle-filter trace"
    (episode / "online_inputs.npz").write_bytes(online)
    (episode / "truth_labels.npz").write_bytes(truth)
    (traces / "episode_0000_seed_45000.npz").write_bytes(trace)
    metadata = {
        "episode_index": 0,
        "episode_seed": 45000,
        "online_inputs_sha256": _sha(online),
        "truth_labels_sha256": _sha(truth),
        "provenance": {
            "trace_path": "/private/original/workstation/trace.npz",
            "trace_sha256": trace_hash_override or _sha(trace),
        },
    }
    (episode / "capture_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return source, traces


def test_portable_projection_changes_only_trace_path_and_adds_audit(tmp_path: Path) -> None:
    source, traces = _fixture(tmp_path)
    output = tmp_path / "portable"
    manifest = projection.project_archive(
        source, traces, output, expected_episodes=1
    )

    original = json.loads(
        (source / "episode_0000_seed_45000/capture_metadata.json").read_text()
    )
    projected = json.loads(
        (output / "episode_0000_seed_45000/capture_metadata.json").read_text()
    )
    assert manifest["episode_count"] == 1
    assert projected["provenance"]["trace_path"] == str(
        (traces / "episode_0000_seed_45000.npz").resolve()
    )
    assert projected["provenance"]["trace_sha256"] == original["provenance"]["trace_sha256"]
    assert projected["online_inputs_sha256"] == original["online_inputs_sha256"]
    assert projected["truth_labels_sha256"] == original["truth_labels_sha256"]
    assert projected["public_portability_projection"]["changed_scientific_values"] is False
    assert projection.sha256_file(output / "episode_0000_seed_45000/online_inputs.npz") == original["online_inputs_sha256"]
    assert projection.sha256_file(output / "episode_0000_seed_45000/truth_labels.npz") == original["truth_labels_sha256"]


def test_trace_hash_mismatch_is_rejected_and_partial_output_removed(tmp_path: Path) -> None:
    source, traces = _fixture(tmp_path, trace_hash_override="0" * 64)
    output = tmp_path / "portable"
    with pytest.raises(RuntimeError, match="legacy trace hash mismatch"):
        projection.project_archive(source, traces, output, expected_episodes=1)
    assert not output.exists()


def test_wrong_episode_seed_identity_is_rejected(tmp_path: Path) -> None:
    source, traces = _fixture(tmp_path)
    wrong = source / "episode_0000_seed_45001"
    (source / "episode_0000_seed_45000").rename(wrong)
    with pytest.raises(RuntimeError, match="episode/seed identity mismatch"):
        projection.project_archive(
            source, traces, tmp_path / "portable", expected_episodes=1
        )


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    source, traces = _fixture(tmp_path)
    output = tmp_path / "portable"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exists"):
        projection.project_archive(source, traces, output, expected_episodes=1)
    assert marker.read_text(encoding="utf-8") == "keep"

