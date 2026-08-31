from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_estimator_benchmark.py"
ROWS = ROOT / "data/tables/estimator_checkpoint_rows.csv"
SUMMARY = ROOT / "data/tables/estimator_benchmark_summary.json"

SPEC = importlib.util.spec_from_file_location("export_estimator_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
table = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(table)


def test_released_rows_cross_check_every_manuscript_cell() -> None:
    rows = table.load_rows(ROWS)
    aggregate = table.aggregate_rows(rows)
    table.cross_check_manuscript(aggregate)

    assert len(rows) == 4500
    assert aggregate["row_count"] == 4500
    assert aggregate["episode_count"] == 100
    assert aggregate["checkpoints_s"] == [30, 60, 120, 240, 440]
    assert set(aggregate["methods"]) == set(table.METHOD_ORDER)

    with SUMMARY.open("r", encoding="utf-8") as stream:
        released_summary = json.load(stream)
    assert released_summary["manuscript_cross_check"] == "PASS"
    assert released_summary["row_file_sha256"] == table.sha256_file(ROWS)
    assert released_summary["methods"] == aggregate["methods"]


def test_terminal_paired_mhe_difference_matches_frozen_analysis() -> None:
    aggregate = table.aggregate_rows(table.load_rows(ROWS))
    paired = aggregate["paired_mhe_minus_full_history_error_m_at_440s"]
    assert paired["mean"] == pytest.approx(-0.0021658954329139904, abs=1e-15)
    assert paired["lower95"] == pytest.approx(-0.006076846171050714, abs=1e-15)
    assert paired["upper95"] == pytest.approx(0.000822052467604904, abs=1e-15)
    assert paired["resamples"] == 50000
    assert paired["seed"] == 34045000


def test_duplicate_cell_is_rejected() -> None:
    rows = table.load_rows(ROWS)
    corrupted = list(rows)
    corrupted[-1] = dict(corrupted[0])
    with pytest.raises(RuntimeError, match="duplicate estimator cell"):
        table.validate_rows(corrupted)


def test_missing_cell_is_rejected() -> None:
    rows = table.load_rows(ROWS)
    with pytest.raises(RuntimeError, match="exactly 4500"):
        table.validate_rows(rows[:-1])


@pytest.mark.parametrize(
    ("campaign", "public_alias"),
    [
        (table.V27_CAMPAIGN, "v27_campaign"),
        (table.V34_CAMPAIGN, "v34_campaign"),
    ],
)
def test_campaign_validation_accepts_public_archive_alias(
    tmp_path: Path,
    campaign: str,
    public_alias: str,
) -> None:
    directory = tmp_path / public_alias
    directory.mkdir()
    (directory / "campaign_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "smoke": False,
                "episode_count": 100,
                "cell_count": 4000 if campaign == table.V27_CAMPAIGN else 500,
            }
        ),
        encoding="utf-8",
    )
    (directory / "independent_audit.json").write_text(
        json.dumps({"valid": True}),
        encoding="utf-8",
    )

    table._validate_campaign(
        directory,
        campaign=campaign,
        cells=4000 if campaign == table.V27_CAMPAIGN else 500,
    )


def test_campaign_validation_rejects_arbitrary_archive_name(tmp_path: Path) -> None:
    directory = tmp_path / "renamed_campaign"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="expected archived directory named"):
        table._validate_campaign(
            directory,
            campaign=table.V27_CAMPAIGN,
            cells=4000,
        )
