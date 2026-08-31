from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import pytest

import make_publication_table_planner_component_ablation as table


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SUMMARY = (
    ROOT
    / "experiments_v39_planner_component_ablation_smoke"
    / "campaign_summary.json"
)


def _synthetic_complete_summary() -> dict:
    with SMOKE_SUMMARY.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    output = copy.deepcopy(summary)
    output.update(
        {
            "smoke": False,
            "status": "complete",
            "decision": "V39_COMPLETE",
            "row_count": 400,
            "expected_row_count": 400,
            "material_rate_difference": 0.05,
        }
    )
    output["component_decisions"] = {
        "pair_term": "NO_MATERIAL_PAIR_TERM_EFFECT_DETECTED",
        "retained_hypotheses": (
            "NO_MATERIAL_RETAINED_HYPOTHESIS_EFFECT_DETECTED"
        ),
        "informed_selection": "NO_MATERIAL_INFORMED_SELECTION_EFFECT_DETECTED",
    }
    output["seed_pairing"] = [
        {"episode_seed": 70000 + index, "arm_count": 4}
        for index in range(100)
    ]
    for cell in output["by_arm"].values():
        cell["episodes"] = 100
        cell["terminal_success_count"] = 100
        cell["terminal_success_rate"] = 1.0
        cell["tail80_success_count"] = 100
        cell["tail80_success_rate"] = 1.0
        cell["ever_lock_count"] = 100
        cell["ever_lock_rate"] = 1.0
    for name in table.EXPECTED_CONTRASTS:
        contrast = output["paired_contrasts"][name]
        contrast["pairs"] = 100
        contrast["terminal_success_rate_difference_full_minus_comparator"] = 0.0
        contrast["tail80_success_rate_difference_full_minus_comparator"] = 0.0
        contrast["terminal_discordance"] = {
            "full_only": 0,
            "comparator_only": 0,
            "exact_mcnemar_p_two_sided": 1.0,
        }
        contrast["tail80_discordance"] = {
            "full_only": 0,
            "comparator_only": 0,
            "exact_mcnemar_p_two_sided": 1.0,
        }
    return output


def test_smoke_summary_is_rejected() -> None:
    with SMOKE_SUMMARY.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    with pytest.raises(RuntimeError, match="refusing smoke summary"):
        table.validate_summary(summary)


def test_incomplete_summary_is_rejected() -> None:
    summary = _synthetic_complete_summary()
    summary["row_count"] = 396
    with pytest.raises(RuntimeError, match="exactly 400"):
        table.validate_summary(summary)


def test_integrity_failure_is_rejected() -> None:
    summary = _synthetic_complete_summary()
    summary["integrity_valid"] = False
    with pytest.raises(RuntimeError, match="integrity_valid is false"):
        table.validate_summary(summary)


def test_render_uses_scientific_labels_and_not_internal_arm_names() -> None:
    summary = _synthetic_complete_summary()
    by_arm, contrasts, decisions = table.validate_summary(summary)
    latex = table.render_table(by_arm, contrasts, decisions)
    assert "Complete hypothesis-conditioned planner" in latex
    assert "Uniform random feasible" in latex
    assert "No pairwise separation" in latex
    assert "Best hypothesis only" in latex
    assert "full_active" not in latex
    assert "no_pair_term" not in latex
    assert "442 s" in latex
    assert r"\(e_p\)" in latex
    assert r"\(e_f\)" in latex
    assert "paired Newcombe hybrid-score" in latex


@pytest.mark.parametrize(
    ("contrast_name", "endpoint", "expected_cells", "expected_interval_pp"),
    (
        (
            "full_vs_no_pair",
            "terminal",
            (97, 3, 0, 0),
            (3.0, -1.1933312904, 8.4519364291),
        ),
        (
            "full_vs_no_pair",
            "tail80",
            (92, 4, 2, 2),
            (2.0, -3.8537795128, 8.2718889847),
        ),
        (
            "full_vs_best_only",
            "terminal",
            (98, 2, 0, 0),
            (2.0, -1.9733007116, 7.0011790729),
        ),
        (
            "full_vs_best_only",
            "tail80",
            (93, 3, 1, 3),
            (2.0, -3.1172040294, 7.7063603194),
        ),
        (
            "full_vs_random",
            "terminal",
            (72, 28, 0, 0),
            (28.0, 19.3126679772, 37.4880287099),
        ),
        (
            "full_vs_random",
            "tail80",
            (54, 42, 3, 1),
            (39.0, 27.6875350863, 49.2565037365),
        ),
    ),
)
def test_full_campaign_newcombe_intervals_are_exact(
    contrast_name: str,
    endpoint: str,
    expected_cells: tuple[int, int, int, int],
    expected_interval_pp: tuple[float, float, float],
) -> None:
    summary = table.load_summary(
        table.DEFAULT_SUMMARY
    )
    by_arm, contrasts, _ = table.validate_summary(summary)
    contrast = contrasts[contrast_name]
    comparator = table.EXPECTED_CONTRASTS[contrast_name]
    discordance = contrast[f"{endpoint}_discordance"]
    cells = table._paired_cells(
        by_arm,
        comparator=comparator,
        endpoint=endpoint,
        full_only=int(discordance["full_only"]),
        comparator_only=int(discordance["comparator_only"]),
    )
    assert cells == expected_cells
    interval_pp = tuple(
        100.0 * value
        for value in table.paired_stats.paired_newcombe_hybrid_score_interval(
            *cells
        )
    )
    assert interval_pp == pytest.approx(expected_interval_pp, abs=5.0e-10)


def test_full_campaign_render_contains_all_rounded_intervals() -> None:
    summary = table.load_summary(
        table.DEFAULT_SUMMARY
    )
    by_arm, contrasts, decisions = table.validate_summary(summary)
    latex = table.render_table(by_arm, contrasts, decisions)
    for expected in (
        r"\(+3.0\,[-1.2,\,8.5]\)",
        r"\(+2.0\,[-3.9,\,8.3]\)",
        r"\(+2.0\,[-2.0,\,7.0]\)",
        r"\(+2.0\,[-3.1,\,7.7]\)",
        r"\(+28.0\,[19.3,\,37.5]\)",
        r"\(+39.0\,[27.7,\,49.3]\)",
    ):
        assert expected in latex


def test_impossible_paired_cells_are_rejected() -> None:
    summary = _synthetic_complete_summary()
    summary["by_arm"]["full_active"]["terminal_success_count"] = 1
    summary["by_arm"]["full_active"]["terminal_success_rate"] = 0.01
    summary["by_arm"]["no_pair_term"]["terminal_success_count"] = 0
    summary["by_arm"]["no_pair_term"]["terminal_success_rate"] = 0.0
    summary["paired_contrasts"]["full_vs_no_pair"]["terminal_discordance"] = {
        "full_only": 2,
        "comparator_only": 1,
        "exact_mcnemar_p_two_sided": 1.0,
    }
    summary["paired_contrasts"]["full_vs_no_pair"][
        "terminal_success_rate_difference_full_minus_comparator"
    ] = 0.01
    with pytest.raises(RuntimeError, match="impossible paired cell counts"):
        table.validate_summary(summary)


def test_released_rows_crosscheck_the_summary() -> None:
    summary = table.load_summary(table.DEFAULT_SUMMARY)
    by_arm, contrasts, _ = table.validate_summary(summary)
    table.validate_rows_against_summary(
        summary,
        table.load_rows(table.DEFAULT_ROWS),
        by_arm,
        contrasts,
    )


def test_row_tampering_is_rejected(tmp_path: Path) -> None:
    rows = table.load_rows(table.DEFAULT_ROWS)
    rows[0]["terminal_joint_success"] = (
        "False" if rows[0]["terminal_joint_success"] == "True" else "True"
    )
    tampered = tmp_path / "planner_rows.csv"
    with tampered.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = table.load_summary(table.DEFAULT_SUMMARY)
    by_arm, contrasts, _ = table.validate_summary(summary)
    with pytest.raises(RuntimeError, match="differs from episode rows"):
        table.validate_rows_against_summary(
            summary,
            table.load_rows(tampered),
            by_arm,
            contrasts,
        )
