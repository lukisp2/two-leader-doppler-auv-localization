from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "project_doi_archive_paths.py"
SPEC = importlib.util.spec_from_file_location("project_doi_archive_paths", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
projector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = projector
SPEC.loader.exec_module(projector)

SOURCE_PREFIX = "/Users/tester/frozen-uuv"
EPISODE_DIR = "episode_0000_seed_45000"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        serialized = json.dumps(value, separators=(",", ":")) + "\n"
    else:
        serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8")


def _contract(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["contract_sha256"] = _canonical_hash(value)
    return result


def _make_archive(root: Path) -> dict[str, Any]:
    capture = root / (
        "data/raw/estimator_benchmark/measurement_history/"
        f"{EPISODE_DIR}/capture_metadata.json"
    )
    capture_value = {
        "episode_index": 0,
        "scientific_value": [1.25, -3.0, 9],
        "provenance": {
            "evaluation_metadata_path": SOURCE_PREFIX + "/evaluation/metadata.json",
            "noise_tape_path": SOURCE_PREFIX + "/evaluation/noise.npz",
            "trace_path": SOURCE_PREFIX + "/evaluation/trace.npz",
        },
    }
    _write_json(capture, capture_value, compact=True)
    capture_sha = hashlib.sha256(capture.read_bytes()).hexdigest()

    source_episode = {
        "directory": SOURCE_PREFIX + "/measurement/" + EPISODE_DIR,
        "metadata_path": SOURCE_PREFIX
        + "/measurement/"
        + EPISODE_DIR
        + "/capture_metadata.json",
        "metadata_sha256": capture_sha,
        "online_path": SOURCE_PREFIX + "/measurement/" + EPISODE_DIR + "/online_inputs.npz",
        "trace_path": SOURCE_PREFIX + "/evaluation/trace.npz",
        "truth_path": SOURCE_PREFIX + "/measurement/" + EPISODE_DIR + "/truth_labels.npz",
    }
    v27_value = _contract(
        {
            "source_archive": SOURCE_PREFIX + "/measurement",
            "source_episodes": [source_episode],
            "scientific_threshold_m": 7.0,
        }
    )
    v27 = root / "data/raw/estimator_benchmark/v27_campaign/control/campaign_contract.json"
    _write_json(v27, v27_value)
    v27_file_sha = hashlib.sha256(v27.read_bytes()).hexdigest()
    _write_json(
        root / "data/raw/estimator_benchmark/v27_campaign/episode_result.json",
        {"contract_sha256": v27_value["contract_sha256"], "score": 4.5},
    )

    v34_episode = dict(source_episode)
    v34_episode.pop("trace_path")
    v34_value = _contract(
        {
            "smoke_prerequisite": {"directory": SOURCE_PREFIX + "/v34-smoke"},
            "source_archive": SOURCE_PREFIX + "/measurement",
            "source_episodes": [v34_episode],
            "v27_reference": {
                "campaign_contract_sha256": v27_file_sha,
                "directory": SOURCE_PREFIX + "/v27",
                "global_full_cells": [
                    {"path": SOURCE_PREFIX + "/v27/episode_result.json", "sha256": "0" * 64}
                ],
            },
            "scientific_window_s": 60.0,
        }
    )
    v34 = root / "data/raw/estimator_benchmark/v34_campaign/control/campaign_contract.json"
    _write_json(v34, v34_value)
    _write_json(
        root / "data/raw/estimator_benchmark/v34_campaign/episode_result.json",
        {"contract_sha256": v34_value["contract_sha256"], "score": 5.5},
    )

    leader_contract = root / "data/raw/leader_source_ablation/control/campaign_contract.json"
    _write_json(
        leader_contract,
        {"environment_metadata_path": SOURCE_PREFIX + "/evaluation/metadata.json", "runs": 600},
    )
    independent = root / "data/raw/leader_source_ablation/independent_audit.json"
    _write_json(independent, {"campaign": SOURCE_PREFIX + "/leader-source", "valid": True})
    vertical = root / "data/raw/leader_source_ablation/vertical_datum_audit.json"
    _write_json(
        vertical,
        {
            "campaign_dir": SOURCE_PREFIX + "/leader-source",
            "clearance": {
                "campaign_contract": SOURCE_PREFIX + "/leader-source/control/campaign_contract.json",
                "campaign_dir": SOURCE_PREFIX + "/leader-source",
                "progress_record": SOURCE_PREFIX + "/leader-source/control/progress.json",
            },
            "environment_metadata": SOURCE_PREFIX + "/evaluation/metadata.json",
            "output_json": SOURCE_PREFIX + "/leader-source/vertical.json",
            "output_md": SOURCE_PREFIX + "/leader-source/vertical.md",
            "minimum_clearance_m": 603.6,
        },
    )

    npz = root / "data/raw/payload.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(b"not-a-real-npz-but-an-immutable-binary-fixture\x00\x01")
    # A pre-existing portable token outside this projection's allowlist is
    # legitimate and must not be mistaken for a private path.
    _write_json(root / "data/raw/preexisting_projection.json", {"path": "${FROZEN_SOURCE_ROOT}/x"})
    return {
        "capture": capture,
        "v27": v27,
        "v34": v34,
        "leader_contract": leader_contract,
        "independent": independent,
        "vertical": vertical,
        "npz": npz,
        "projected": (capture, v27, v34, leader_contract, independent, vertical),
    }


def _apply(root: Path) -> dict[str, Any]:
    return projector.apply_projection(
        root,
        source_prefix=SOURCE_PREFIX,
        expected_files=6,
        expected_replacements=26,
    )


def test_projection_is_allowlisted_hash_bridged_and_scientifically_invariant(
    tmp_path: Path,
) -> None:
    paths = _make_archive(tmp_path)
    originals = {path: path.read_bytes() for path in paths["projected"]}
    npz_before = hashlib.sha256(paths["npz"].read_bytes()).hexdigest()

    result = _apply(tmp_path)

    assert result["totals"] == {
        "projected_file_count": 6,
        "json_pointer_count": 26,
        "replacement_count": 26,
    }
    assert result["application"] == {
        "json_files_written": 6,
        "manifest_written": True,
        "idempotent_noop": False,
    }
    assert result["scientific_values_changed"] is False
    assert result["npz_guard"]["file_count"] == 1
    assert result["npz_guard"]["modified_by_projection"] is False
    bridge = result["internal_hash_bridge"]
    assert bridge["valid"] is True
    assert bridge["capture_metadata_original_sha256_references"] == {
        "references_checked": 2,
        "unique_targets_checked": 1,
        "valid": True,
    }
    assert [row["dependent_contract_sha256_values_checked"] for row in bridge["dependent_contract_sha256_values"]] == [1, 1]
    assert all(row["valid"] for row in bridge["self_hashed_contracts"])

    for path, original in originals.items():
        expected = original.replace(SOURCE_PREFIX.encode(), b"${FROZEN_SOURCE_ROOT}")
        assert path.read_bytes() == expected
        assert SOURCE_PREFIX.encode() not in path.read_bytes()
    assert hashlib.sha256(paths["npz"].read_bytes()).hexdigest() == npz_before

    manifest_path = tmp_path / projector.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert SOURCE_PREFIX not in manifest_path.read_text(encoding="utf-8")
    assert manifest["source_prefix_sha256"] == hashlib.sha256(SOURCE_PREFIX.encode()).hexdigest()
    assert all(row["scientific_values_changed"] is False for row in manifest["files"])
    assert all(row["canonical_scientific_sha256"] for row in manifest["files"])

    verified = projector.verify_projection(
        tmp_path,
        source_prefix=SOURCE_PREFIX,
        expected_files=6,
        expected_replacements=26,
    )
    assert verified == manifest


def test_second_application_is_a_byte_identical_noop(tmp_path: Path) -> None:
    paths = _make_archive(tmp_path)
    _apply(tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    result = _apply(tmp_path)

    after = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert result["application"] == {
        "json_files_written": 0,
        "manifest_written": False,
        "idempotent_noop": True,
    }
    assert all(SOURCE_PREFIX.encode() not in path.read_bytes() for path in paths["projected"])


@pytest.mark.parametrize(
    "rogue_value",
    (
        "/Users/someone-else/private/file.json",
        "/home/someone/private/file.json",
        r"C:\Users\someone\private\file.json",
    ),
)
def test_other_private_home_paths_fail_closed_before_any_write(
    tmp_path: Path, rogue_value: str
) -> None:
    paths = _make_archive(tmp_path)
    rogue = tmp_path / "data/raw/rogue.json"
    _write_json(rogue, {"path": rogue_value})
    before = {path: path.read_bytes() for path in paths["projected"]}

    with pytest.raises(RuntimeError, match="outside the audited allowlist"):
        _apply(tmp_path)

    assert not (tmp_path / projector.MANIFEST_NAME).exists()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_approved_prefix_outside_allowlist_fails_closed(tmp_path: Path) -> None:
    paths = _make_archive(tmp_path)
    rogue = tmp_path / "data/raw/rogue.json"
    _write_json(rogue, {"path": SOURCE_PREFIX + "/secret"})
    before = {path: path.read_bytes() for path in paths["projected"]}

    with pytest.raises(RuntimeError, match="outside the audited allowlist"):
        _apply(tmp_path)

    assert not (tmp_path / projector.MANIFEST_NAME).exists()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_private_path_in_json_key_fails_closed(tmp_path: Path) -> None:
    paths = _make_archive(tmp_path)
    rogue = tmp_path / "data/raw/rogue.json"
    _write_json(rogue, {SOURCE_PREFIX + "/secret": "value"})
    before = {path: path.read_bytes() for path in paths["projected"]}

    with pytest.raises(RuntimeError, match="forbidden in a JSON key"):
        _apply(tmp_path)

    assert not (tmp_path / projector.MANIFEST_NAME).exists()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_invalid_historical_self_hash_fails_before_projection(tmp_path: Path) -> None:
    paths = _make_archive(tmp_path)
    value = json.loads(paths["v27"].read_text(encoding="utf-8"))
    value["contract_sha256"] = "f" * 64
    _write_json(paths["v27"], value)
    before = {path: path.read_bytes() for path in paths["projected"]}

    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        _apply(tmp_path)

    assert not (tmp_path / projector.MANIFEST_NAME).exists()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_write_failure_rolls_back_every_projected_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_archive(tmp_path)
    before = {path: path.read_bytes() for path in paths["projected"]}
    real_atomic_write = projector._atomic_write
    state = {"calls": 0, "failed": False}

    def fail_once(path: Path, value: bytes) -> None:
        state["calls"] += 1
        if state["calls"] == 2 and not state["failed"]:
            state["failed"] = True
            raise OSError("injected atomic-write failure")
        real_atomic_write(path, value)

    monkeypatch.setattr(projector, "_atomic_write", fail_once)
    with pytest.raises(OSError, match="injected atomic-write failure"):
        _apply(tmp_path)

    assert state["failed"] is True
    assert not (tmp_path / projector.MANIFEST_NAME).exists()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_tampered_manifest_is_not_silently_replaced(tmp_path: Path) -> None:
    paths = _make_archive(tmp_path)
    _apply(tmp_path)
    manifest_path = tmp_path / projector.MANIFEST_NAME
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["scientific_values_changed"] = True
    _write_json(manifest_path, tampered)
    before = {path: path.read_bytes() for path in paths["projected"]}

    with pytest.raises(RuntimeError, match="existing path-projection manifest"):
        _apply(tmp_path)

    assert all(path.read_bytes() == value for path, value in before.items())
