from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/project_v40_doi_campaign.py"
SPEC = importlib.util.spec_from_file_location("project_v40_doi_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
projector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = projector
SPEC.loader.exec_module(projector)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_fixture(tmp_path: Path) -> tuple[Path, Path, object, str, str]:
    campaign = tmp_path / "private" / "campaign"
    metadata_original = tmp_path / "private" / "metadata.json"
    config = {"action_dt": 2.0, "max_steps": 4, "sub_dt": 0.1}
    write_json(metadata_original, {"private": True, "environment_config": config})
    original_metadata_hash = projector.sha256_file(metadata_original)

    public_metadata = tmp_path / "approved" / "environment_metadata.json"
    write_json(
        public_metadata,
        {
            "schema": "public_environment_config_v1",
            "provenance": {"transformation": "private paths removed"},
            "environment_config": config,
        },
    )
    public_metadata_hash = projector.sha256_file(public_metadata)
    recorded_private_path = str(
        Path("/") / "Users" / "example" / "frozen" / "metadata.json"
    )
    contract = {
        "expected_runs": 2,
        "environment_metadata_path": recorded_private_path,
        "environment_metadata_sha256": original_metadata_hash,
        "scientific_setting": 17,
    }
    write_json(campaign / "control/campaign_contract.json", contract)
    write_json(campaign / "campaign_summary.json", {"status": "complete"})
    write_json(campaign / "decision.json", {"decision": "PASS"})
    write_json(campaign / "control/progress.json", {"complete": True})

    for index in range(2):
        write_json(
            campaign / "episode_results" / "arm" / f"episode_{index}.json",
            {"episode": index, "value": index + 0.5},
        )
        trace = campaign / "traces_npz" / "arm" / f"episode_{index}.npz"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_bytes(b"synthetic-npz-" + bytes([index]))
    noise = campaign / "noise_tapes" / "noise_0.npz"
    noise.parent.mkdir(parents=True, exist_ok=True)
    noise.write_bytes(b"synthetic-noise")
    snapshot = campaign / "control/source_snapshot/source.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"frozen = True\n")
    counts = projector.ExpectedCounts(
        episode_json=2,
        trace_npz=2,
        noise_npz=1,
        source_snapshot_files=1,
    )
    return (
        campaign,
        public_metadata,
        counts,
        original_metadata_hash,
        public_metadata_hash,
    )


def project_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    campaign, metadata, counts, original_hash, public_hash = make_fixture(tmp_path)
    output = tmp_path / "doi" / "campaign"
    config_hash = projector.canonical_sha256(
        json.loads(metadata.read_text(encoding="utf-8"))["environment_config"]
    )
    manifest = projector.project_campaign(
        campaign,
        output,
        metadata,
        expected_counts=counts,
        expected_original_metadata_sha256=original_hash,
        expected_public_metadata_sha256=public_hash,
        expected_environment_config_sha256=config_hash,
    )
    return campaign, output, manifest


def test_projection_is_complete_path_sanitized_and_byte_preserving(tmp_path: Path) -> None:
    campaign, output, manifest = project_fixture(tmp_path)

    assert manifest["scientific_values_changed"] is False
    assert manifest["source_campaign_file_count"] == 10
    assert manifest["exported_file_count_excluding_manifest"] == 11
    assert len(projector.regular_files(output)) == 12
    assert projector.private_path_hits(output) == []

    source_contract = json.loads(
        (campaign / "control/campaign_contract.json").read_text(encoding="utf-8")
    )
    exported_contract = json.loads(
        (output / "control/campaign_contract.json").read_text(encoding="utf-8")
    )
    assert source_contract["environment_metadata_path"].startswith("/")
    assert exported_contract == {
        **source_contract,
        "environment_metadata_path": projector.PORTABLE_ENVIRONMENT_METADATA_PATH,
    }

    rows = {row["relative_path"]: row for row in manifest["files"]}
    assert rows["control/campaign_contract.json"]["byte_for_byte_preserved"] is False
    assert rows["control/environment_metadata.json"]["byte_for_byte_preserved"] is True
    for relative, row in rows.items():
        if relative not in {
            "control/campaign_contract.json",
            "control/environment_metadata.json",
        }:
            assert row["byte_for_byte_preserved"] is True
            assert (campaign / relative).read_bytes() == (output / relative).read_bytes()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    campaign, metadata, counts, original_hash, public_hash = make_fixture(tmp_path)
    output = tmp_path / "doi" / "campaign"
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    config_hash = projector.canonical_sha256(
        json.loads(metadata.read_text(encoding="utf-8"))["environment_config"]
    )

    with pytest.raises(RuntimeError, match="output already exists"):
        projector.project_campaign(
            campaign,
            output,
            metadata,
            expected_counts=counts,
            expected_original_metadata_sha256=original_hash,
            expected_public_metadata_sha256=public_hash,
            expected_environment_config_sha256=config_hash,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_unexpected_second_private_path_fails_closed(tmp_path: Path) -> None:
    campaign, metadata, counts, original_hash, public_hash = make_fixture(tmp_path)
    extra = campaign / "notes.txt"
    second_private_path = str(
        Path("/") / "Users" / "example" / "secret" / "input.json"
    )
    extra.write_text(f"other={second_private_path}\n", encoding="utf-8")
    output = tmp_path / "doi" / "campaign"
    config_hash = projector.canonical_sha256(
        json.loads(metadata.read_text(encoding="utf-8"))["environment_config"]
    )

    with pytest.raises(RuntimeError, match="exactly its one recorded"):
        projector.project_campaign(
            campaign,
            output,
            metadata,
            expected_counts=counts,
            expected_original_metadata_sha256=original_hash,
            expected_public_metadata_sha256=public_hash,
            expected_environment_config_sha256=config_hash,
        )
    assert not output.exists()


def test_source_is_unchanged_by_successful_projection(tmp_path: Path) -> None:
    campaign, _, _ = project_fixture(tmp_path)
    contract = campaign / "control/campaign_contract.json"
    before = hashlib.sha256(contract.read_bytes()).hexdigest()
    assert json.loads(contract.read_text(encoding="utf-8"))[
        "environment_metadata_path"
    ].startswith("/")
    after = hashlib.sha256(contract.read_bytes()).hexdigest()
    assert after == before


def test_private_path_detector_covers_windows_user_homes() -> None:
    separator = "\\"
    windows_path = separator.join(("C:", "Users", "example", "metadata.json"))
    assert projector.PRIVATE_ABSOLUTE_PATH.search(windows_path)
