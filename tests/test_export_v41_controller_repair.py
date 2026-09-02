from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_v41_controller_repair.py"
SPEC = importlib.util.spec_from_file_location("export_v41_controller_repair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_released_rows_reconstruct_controller_repair_headlines() -> None:
    rows = exporter.load_rows(
        ROOT / "data/tables/controller_repair_episode_rows.csv"
    )
    validation = exporter.validate_rows(rows)
    assert validation["row_count"] == 400
    assert validation["selected_seeds_disjoint_reserved_range"] is True

    cells = {}
    for current in exporter.CURRENTS:
        for controller in exporter.CONTROLLERS:
            arm_rows = [
                row
                for row in rows
                if row["current"] == current and row["controller"] == controller
            ]
            cells[(current, controller)] = exporter.aggregate_arm(arm_rows)

    assert cells[("no_current", "baseline_pid")]["terminal_success_rate"] == 0.38
    assert cells[("no_current", "delay_aware")]["terminal_success_rate"] == 0.95
    assert cells[("bottom_track_visible", "baseline_pid")][
        "terminal_success_rate"
    ] == 0.26
    assert cells[("bottom_track_visible", "delay_aware")][
        "terminal_success_rate"
    ] == 0.95
    assert cells[("no_current", "delay_aware")]["tail80_success_rate"] == 0.91
    assert cells[("bottom_track_visible", "delay_aware")][
        "tail80_success_rate"
    ] == 0.92


def test_public_semantic_audit_does_not_rewrite_frozen_decision() -> None:
    result = json.loads(
        (ROOT / "data/tables/controller_repair_results.json").read_text(
            encoding="utf-8"
        )
    )
    audit = result["semantic_audit"]
    assert audit["campaign_integrity"] == "valid"
    assert audit["prespecified_composite_acceptance"] == "not_met"
    assert audit["legacy_decision_label"] == exporter.LEGACY_DECISION
    assert audit["public_unambiguous_decision_label"] == exporter.PUBLIC_DECISION
    assert audit["legacy_artifacts_modified"] is False
    assert audit["reserved_seed_marker_audit"]["global_never_opened_claim_made"] is False
    failed = audit["failed_prespecified_condition"]
    assert failed["excess_over_margin_m"] > 0.0
    assert failed["excess_over_margin_m"] < 0.067

