from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/project_v41_doi_campaign.py"
SPEC = importlib.util.spec_from_file_location("project_v41_doi_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(ROOT / "scripts"))
try:
    projector = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(projector)
finally:
    sys.path.remove(str(ROOT / "scripts"))


def test_v41_cardinality_is_frozen() -> None:
    assert projector.V41_COUNTS.episode_json == 400
    assert projector.V41_COUNTS.trace_npz == 400
    assert projector.V41_COUNTS.noise_npz == 100
    assert projector.V41_COUNTS.source_snapshot_files == 15


def test_json_private_path_scanner_preserves_spaces(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    path = root / "control/campaign_contract.json"
    path.parent.mkdir(parents=True)
    private = "/Users/example/My Research/metadata.json"
    path.write_text(
        json.dumps({"environment_metadata_path": private}),
        encoding="utf-8",
    )
    assert projector.private_path_hits_allowing_spaces(root) == [
        ("control/campaign_contract.json", private)
    ]

