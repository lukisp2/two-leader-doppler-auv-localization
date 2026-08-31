from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_campaign_provenance.py"
SPEC = importlib.util.spec_from_file_location("export_campaign_provenance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "private_source"
    campaign = source_root / "campaign"
    (campaign / "control").mkdir(parents=True)
    payload = {"path": str(source_root / "input.json"), "value": 7}
    for relative in exporter.REQUIRED_FILES:
        path = campaign / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    (campaign / "independent_audit.json").write_text(
        json.dumps({"valid": True, "campaign": str(campaign)}),
        encoding="utf-8",
    )
    return campaign, source_root


def test_export_preserves_original_hashes_and_sanitizes_paths(tmp_path: Path) -> None:
    campaign, source_root = _campaign(tmp_path)
    output = tmp_path / "public"
    manifest = exporter.export_campaign(campaign, output, source_root=source_root)
    assert len(manifest["files"]) == 4
    assert sum(row["source_root_replacements"] for row in manifest["files"]) == 4
    for row in manifest["files"]:
        assert row["original_sha256"] == exporter.sha256(campaign / row["relative_path"])
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in output.rglob("*.json")
    )
    assert str(source_root.resolve()) not in serialized
    assert "${FROZEN_SOURCE_ROOT}" in serialized


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    campaign, source_root = _campaign(tmp_path)
    output = tmp_path / "public"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exists"):
        exporter.export_campaign(campaign, output, source_root=source_root)
    assert marker.read_text(encoding="utf-8") == "keep"

