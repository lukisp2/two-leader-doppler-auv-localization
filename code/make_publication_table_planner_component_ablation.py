#!/usr/bin/env python3
"""Generate the publication table for the planner-component ablation.

Only a complete, integrity-valid, 100-scenario campaign is accepted.  Smoke
summaries and partial campaigns are rejected before an output file is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import make_publication_table_source_policy_contrasts as paired_stats


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data/tables"
DEFAULT_SUMMARY = TABLES / "planner_component_summary.json"
DEFAULT_ROWS = TABLES / "planner_component_episode_rows.csv"
DEFAULT_OUTPUT = ROOT / "results/tables/table_planner_component_ablation.tex"

EXPECTED_EPISODES = 100
EXPECTED_ARMS = (
    "full_active",
    "no_pair_term",
    "best_hypothesis_only",
    "random_feasible",
)
EXPECTED_CONTRASTS = {
    "full_vs_no_pair": "no_pair_term",
    "full_vs_best_only": "best_hypothesis_only",
    "full_vs_random": "random_feasible",
}

ARM_LABELS = {
    "full_active": "Complete hypothesis-conditioned planner",
    "no_pair_term": "No pairwise separation",
    "best_hypothesis_only": "Best hypothesis only",
    "random_feasible": "Uniform random feasible",
}
COMPARATOR_LABELS = {
    "no_pair_term": "No pairwise separation",
    "best_hypothesis_only": "Best hypothesis only",
    "random_feasible": "Uniform random feasible",
}
COMPONENT_FOR_CONTRAST = {
    "full_vs_no_pair": "pair_term",
    "full_vs_best_only": "retained_hypotheses",
    "full_vs_random": "informed_selection",
}
ALLOWED_COMPONENT_DECISIONS = {
    "pair_term": {
        "MATERIAL_PAIR_TERM_EFFECT": "Material effect",
        "NO_MATERIAL_PAIR_TERM_EFFECT_DETECTED": "No material effect detected",
    },
    "retained_hypotheses": {
        "MATERIAL_RETAINED_HYPOTHESIS_EFFECT": "Material effect",
        "NO_MATERIAL_RETAINED_HYPOTHESIS_EFFECT_DETECTED": (
            "No material effect detected"
        ),
    },
    "informed_selection": {
        "MATERIAL_INFORMED_SELECTION_EFFECT": "Material effect",
        "NO_MATERIAL_INFORMED_SELECTION_EFFECT_DETECTED": (
            "No material effect detected"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help=(
            "Optional DOI campaign directory containing campaign_summary.json "
            "and episode_arm_summary.csv; overrides --summary and --rows."
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Released compact campaign summary.",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=DEFAULT_ROWS,
        help="Released per-arm episode CSV used for row-derived cross-checks.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.campaign_dir is not None:
        campaign = args.campaign_dir.expanduser().resolve()
        return (
            campaign / "campaign_summary.json",
            campaign / "episode_arm_summary.csv",
        )
    return (
        args.summary.expanduser().resolve(),
        args.rows.expanduser().resolve(),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"campaign summary does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read campaign summary: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError("campaign summary must be a JSON object")
    return value


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"episode-row CSV does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise RuntimeError(f"episode-row CSV has no header: {path}")
            return list(reader)
    except OSError as error:
        raise RuntimeError(f"cannot read episode-row CSV: {path}") from error


def _parse_csv_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise RuntimeError(f"{context} is not Boolean: {value!r}")


def _parse_csv_int(value: str, context: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} is not an integer: {value!r}") from error
    return parsed


def _parse_csv_float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise RuntimeError(f"{context} is not finite")
    return parsed


def _row_distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != EXPECTED_EPISODES or not np.all(np.isfinite(array)):
        raise RuntimeError("row-derived distributions require 100 finite values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _assert_close(context: str, saved: Any, row_derived: float) -> None:
    if not math.isclose(
        _number(saved, context),
        float(row_derived),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(f"{context} differs from the released episode rows")


def _crosscheck_distribution(
    context: str,
    saved_value: Any,
    values: list[float],
) -> None:
    saved = _mapping(saved_value, context)
    derived = _row_distribution(values)
    for key, value in derived.items():
        _assert_close(f"{context}.{key}", saved.get(key), value)


def _exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / (
        2.0**discordant
    )
    return float(min(1.0, 2.0 * probability))


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be an object")
    return value


def _exact_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{context} must be Boolean")
    return value


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{context} must be an integer")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{context} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise RuntimeError(f"{context} must be finite")
    return output


def _count(value: Any, context: str, maximum: int) -> int:
    output = _integer(value, context)
    if not 0 <= output <= maximum:
        raise RuntimeError(f"{context} must be in [0, {maximum}]")
    return output


def _validate_distribution(
    value: Any,
    context: str,
) -> Mapping[str, Any]:
    distribution = _mapping(value, context)
    median = _number(distribution.get("median"), f"{context}.median")
    p95 = _number(distribution.get("p95"), f"{context}.p95")
    if median < 0.0 or p95 < 0.0:
        raise RuntimeError(f"{context} cannot contain negative values")
    if p95 + 1.0e-12 < median:
        raise RuntimeError(f"{context}.p95 is below its median")
    return distribution


def _validate_arm(
    arm: str,
    cell_value: Any,
) -> Mapping[str, Any]:
    cell = _mapping(cell_value, f"by_arm.{arm}")
    episodes = _integer(cell.get("episodes"), f"by_arm.{arm}.episodes")
    if episodes != EXPECTED_EPISODES:
        raise RuntimeError(
            f"by_arm.{arm}.episodes is {episodes}, expected {EXPECTED_EPISODES}"
        )

    for count_name in (
        "terminal_success_count",
        "tail80_success_count",
        "ever_lock_count",
    ):
        _count(
            cell.get(count_name),
            f"by_arm.{arm}.{count_name}",
            episodes,
        )

    rate_pairs = (
        ("terminal_success_count", "terminal_success_rate"),
        ("tail80_success_count", "tail80_success_rate"),
        ("ever_lock_count", "ever_lock_rate"),
    )
    for count_name, rate_name in rate_pairs:
        count = _integer(
            cell.get(count_name),
            f"by_arm.{arm}.{count_name}",
        )
        rate = _number(cell.get(rate_name), f"by_arm.{arm}.{rate_name}")
        expected_rate = count / episodes
        if not math.isclose(
            rate,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"by_arm.{arm}.{rate_name} is inconsistent with {count_name}"
            )

    _validate_distribution(
        cell.get("first_track_analysis_time_s"),
        f"by_arm.{arm}.first_track_analysis_time_s",
    )
    _validate_distribution(
        cell.get("terminal_localization_error_m"),
        f"by_arm.{arm}.terminal_localization_error_m",
    )
    _validate_distribution(
        cell.get("terminal_formation_error_m"),
        f"by_arm.{arm}.terminal_formation_error_m",
    )
    runtime = _number(
        cell.get("maximum_combined_decision_runtime_s"),
        f"by_arm.{arm}.maximum_combined_decision_runtime_s",
    )
    if runtime < 0.0:
        raise RuntimeError(f"by_arm.{arm} has a negative decision runtime")
    return cell


def _validate_contrast(
    name: str,
    value: Any,
    by_arm: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    contrast = _mapping(value, f"paired_contrasts.{name}")
    comparator = EXPECTED_CONTRASTS[name]
    if contrast.get("full") != "full_active":
        raise RuntimeError(f"paired_contrasts.{name}.full is unexpected")
    if contrast.get("comparator") != comparator:
        raise RuntimeError(
            f"paired_contrasts.{name}.comparator is unexpected"
        )
    if _integer(contrast.get("pairs"), f"paired_contrasts.{name}.pairs") != (
        EXPECTED_EPISODES
    ):
        raise RuntimeError(
            f"paired_contrasts.{name} does not contain 100 pairs"
        )

    endpoint_specs = (
        (
            "terminal",
            "terminal_success_count",
            "terminal_success_rate_difference_full_minus_comparator",
            "terminal_discordance",
        ),
        (
            "tail80",
            "tail80_success_count",
            "tail80_success_rate_difference_full_minus_comparator",
            "tail80_discordance",
        ),
    )
    for endpoint, count_name, difference_name, discordance_name in endpoint_specs:
        difference = _number(
            contrast.get(difference_name),
            f"paired_contrasts.{name}.{difference_name}",
        )
        expected_difference = (
            _integer(by_arm["full_active"][count_name], count_name)
            - _integer(by_arm[comparator][count_name], count_name)
        ) / EXPECTED_EPISODES
        if not math.isclose(
            difference,
            expected_difference,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"paired_contrasts.{name}.{endpoint} rate difference "
                "is inconsistent with arm counts"
            )

        discordance = _mapping(
            contrast.get(discordance_name),
            f"paired_contrasts.{name}.{discordance_name}",
        )
        full_only = _count(
            discordance.get("full_only"),
            f"paired_contrasts.{name}.{discordance_name}.full_only",
            EXPECTED_EPISODES,
        )
        comparator_only = _count(
            discordance.get("comparator_only"),
            (
                f"paired_contrasts.{name}.{discordance_name}."
                "comparator_only"
            ),
            EXPECTED_EPISODES,
        )
        if full_only + comparator_only > EXPECTED_EPISODES:
            raise RuntimeError(
                f"paired_contrasts.{name}.{discordance_name} has too many "
                "discordant pairs"
            )
        if full_only - comparator_only != round(
            difference * EXPECTED_EPISODES
        ):
            raise RuntimeError(
                f"paired_contrasts.{name}.{discordance_name} is inconsistent "
                "with the reported rate difference"
            )
        _paired_cells(
            by_arm,
            comparator=comparator,
            endpoint=endpoint,
            full_only=full_only,
            comparator_only=comparator_only,
        )
        p_value = _number(
            discordance.get("exact_mcnemar_p_two_sided"),
            (
                f"paired_contrasts.{name}.{discordance_name}."
                "exact_mcnemar_p_two_sided"
            ),
        )
        if not 0.0 <= p_value <= 1.0:
            raise RuntimeError(
                f"paired_contrasts.{name}.{discordance_name} has invalid p"
            )
    return contrast


def _paired_cells(
    by_arm: Mapping[str, Mapping[str, Any]],
    *,
    comparator: str,
    endpoint: str,
    full_only: int,
    comparator_only: int,
) -> tuple[int, int, int, int]:
    """Recover the paired 2x2 table from audited marginals and discordances.

    The returned order matches
    :func:`paired_newcombe_hybrid_score_interval`: both successful,
    complete-only, comparator-only, and neither successful.
    """

    if endpoint == "terminal":
        count_name = "terminal_success_count"
    elif endpoint == "tail80":
        count_name = "tail80_success_count"
    else:
        raise ValueError(f"unsupported endpoint: {endpoint}")

    full_success = _integer(
        by_arm["full_active"][count_name],
        f"by_arm.full_active.{count_name}",
    )
    comparator_success = _integer(
        by_arm[comparator][count_name],
        f"by_arm.{comparator}.{count_name}",
    )
    both_success = full_success - full_only
    neither_success = (
        EXPECTED_EPISODES
        - both_success
        - full_only
        - comparator_only
    )
    cells = (
        both_success,
        full_only,
        comparator_only,
        neither_success,
    )
    if any(cell < 0 for cell in cells):
        raise RuntimeError(
            f"{comparator} {endpoint}: impossible paired cell counts {cells}"
        )
    if both_success + comparator_only != comparator_success:
        raise RuntimeError(
            f"{comparator} {endpoint}: paired cells disagree with comparator "
            "success count"
        )
    if sum(cells) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"{comparator} {endpoint}: paired cells do not sum to 100"
        )
    return cells


def validate_summary(
    summary: Mapping[str, Any],
) -> tuple[
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
    Mapping[str, str],
]:
    if _exact_bool(summary.get("smoke"), "smoke"):
        raise RuntimeError(
            "refusing smoke summary: publication table requires 100 scenarios"
        )
    if summary.get("status") != "complete":
        raise RuntimeError("campaign status is not complete")
    if summary.get("decision") != "V39_COMPLETE":
        raise RuntimeError("campaign decision is not V39_COMPLETE")
    if not _exact_bool(summary.get("integrity_valid"), "integrity_valid"):
        raise RuntimeError("campaign integrity_valid is false")
    if not _exact_bool(
        summary.get("publication_settings"),
        "publication_settings",
    ):
        raise RuntimeError("campaign did not use publication settings")

    expected_rows = EXPECTED_EPISODES * len(EXPECTED_ARMS)
    row_count = _integer(summary.get("row_count"), "row_count")
    declared_expected = _integer(
        summary.get("expected_row_count"),
        "expected_row_count",
    )
    if row_count != expected_rows or declared_expected != expected_rows:
        raise RuntimeError(
            f"campaign requires exactly {expected_rows} complete arm-runs"
        )

    integrity_checks = _mapping(
        summary.get("integrity_checks"),
        "integrity_checks",
    )
    if not integrity_checks:
        raise RuntimeError("integrity_checks is empty")
    failed_checks = sorted(
        str(name)
        for name, value in integrity_checks.items()
        if not isinstance(value, bool) or not value
    )
    if failed_checks:
        raise RuntimeError(
            "campaign has failed or malformed integrity checks: "
            + ", ".join(failed_checks)
        )

    pairing = summary.get("seed_pairing")
    if not isinstance(pairing, list) or len(pairing) != EXPECTED_EPISODES:
        raise RuntimeError("seed_pairing must contain exactly 100 entries")
    pairing_seeds: set[int] = set()
    for index, entry_value in enumerate(pairing):
        entry = _mapping(entry_value, f"seed_pairing[{index}]")
        seed = _integer(entry.get("episode_seed"), f"seed_pairing[{index}].seed")
        if seed in pairing_seeds:
            raise RuntimeError(f"duplicate paired episode seed: {seed}")
        pairing_seeds.add(seed)
        if _integer(
            entry.get("arm_count"),
            f"seed_pairing[{index}].arm_count",
        ) != len(EXPECTED_ARMS):
            raise RuntimeError(f"seed_pairing[{index}] does not have four arms")

    by_arm_value = _mapping(summary.get("by_arm"), "by_arm")
    if set(by_arm_value) != set(EXPECTED_ARMS):
        raise RuntimeError(
            "by_arm must contain exactly the four prespecified arms"
        )
    by_arm = {
        arm: _validate_arm(arm, by_arm_value[arm])
        for arm in EXPECTED_ARMS
    }

    contrast_value = _mapping(
        summary.get("paired_contrasts"),
        "paired_contrasts",
    )
    contrasts = {
        name: _validate_contrast(name, contrast_value.get(name), by_arm)
        for name in EXPECTED_CONTRASTS
    }

    threshold = _number(
        summary.get("material_rate_difference"),
        "material_rate_difference",
    )
    if not math.isclose(threshold, 0.05, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("material-effect threshold differs from 5 pp")

    decisions_value = _mapping(
        summary.get("component_decisions"),
        "component_decisions",
    )
    if set(decisions_value) != set(ALLOWED_COMPONENT_DECISIONS):
        raise RuntimeError("component_decisions has unexpected components")
    decisions: dict[str, str] = {}
    for component, labels in ALLOWED_COMPONENT_DECISIONS.items():
        raw = decisions_value[component]
        if raw not in labels:
            raise RuntimeError(
                f"component_decisions.{component} has unexpected value {raw!r}"
            )
        decisions[component] = labels[raw]

    return by_arm, contrasts, decisions


def validate_rows_against_summary(
    summary: Mapping[str, Any],
    rows: list[dict[str, str]],
    by_arm: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
) -> None:
    """Recompute every displayed result from the released episode rows.

    The summary retains contract and audit metadata that cannot be reconstructed
    from the compact CSV.  Counts, distributions, runtimes, paired binary
    contrasts, and the resulting material-effect labels are nevertheless
    checked independently here before any LaTeX is written.
    """

    required = {
        "episode_seed",
        "arm",
        "terminal_joint_success",
        "tail80_joint_success",
        "ever_locked",
        "first_track_analysis_time_s",
        "terminal_localization_error_m",
        "terminal_formation_error_m",
        "unsafe_transition_count",
        "unsafe_track_start_count",
        "unsafe_track_end_count",
        "maximum_combined_decision_runtime_s",
    }
    if len(rows) != EXPECTED_EPISODES * len(EXPECTED_ARMS):
        raise RuntimeError("episode-row CSV must contain exactly 400 rows")
    indexed: dict[str, dict[int, dict[str, str]]] = {
        arm: {} for arm in EXPECTED_ARMS
    }
    for row_number, row in enumerate(rows, start=2):
        missing = required.difference(row)
        if missing:
            raise RuntimeError(
                f"CSV row {row_number} is missing columns {sorted(missing)}"
            )
        arm = row["arm"]
        if arm not in indexed:
            raise RuntimeError(f"CSV row {row_number} has unexpected arm {arm!r}")
        seed = _parse_csv_int(row["episode_seed"], f"CSV row {row_number}.seed")
        if seed in indexed[arm]:
            raise RuntimeError(f"{arm}: duplicate episode seed {seed}")
        indexed[arm][seed] = row

    declared_pairing = summary.get("seed_pairing")
    if not isinstance(declared_pairing, list):
        raise RuntimeError("seed_pairing is not a list")
    declared_seeds = {
        _integer(entry.get("episode_seed"), "seed_pairing.episode_seed")
        for entry in declared_pairing
        if isinstance(entry, Mapping)
    }
    if len(declared_seeds) != EXPECTED_EPISODES:
        raise RuntimeError("seed_pairing does not contain 100 unique seeds")
    for arm, by_seed in indexed.items():
        if set(by_seed) != declared_seeds:
            raise RuntimeError(f"{arm}: row seeds differ from seed_pairing")
        ordered = [by_seed[seed] for seed in sorted(by_seed)]
        cell = by_arm[arm]
        count_specs = (
            ("terminal_success_count", "terminal_joint_success"),
            ("tail80_success_count", "tail80_joint_success"),
            ("ever_lock_count", "ever_locked"),
        )
        for summary_key, row_key in count_specs:
            derived = sum(
                _parse_csv_bool(row[row_key], f"{arm}.{row_key}")
                for row in ordered
            )
            if _integer(cell.get(summary_key), f"by_arm.{arm}.{summary_key}") != derived:
                raise RuntimeError(
                    f"by_arm.{arm}.{summary_key} differs from episode rows"
                )
        sum_specs = (
            "unsafe_transition_count",
            "unsafe_track_start_count",
            "unsafe_track_end_count",
        )
        for key in sum_specs:
            derived = sum(_parse_csv_int(row[key], f"{arm}.{key}") for row in ordered)
            if _integer(cell.get(key), f"by_arm.{arm}.{key}") != derived:
                raise RuntimeError(f"by_arm.{arm}.{key} differs from episode rows")
        distribution_specs = (
            ("first_track_analysis_time_s", "first_track_analysis_time_s"),
            ("terminal_localization_error_m", "terminal_localization_error_m"),
            ("terminal_formation_error_m", "terminal_formation_error_m"),
        )
        for summary_key, row_key in distribution_specs:
            _crosscheck_distribution(
                f"by_arm.{arm}.{summary_key}",
                cell.get(summary_key),
                [
                    _parse_csv_float(row[row_key], f"{arm}.{row_key}")
                    for row in ordered
                ],
            )
        derived_runtime = max(
            _parse_csv_float(
                row["maximum_combined_decision_runtime_s"],
                f"{arm}.maximum_combined_decision_runtime_s",
            )
            for row in ordered
        )
        _assert_close(
            f"by_arm.{arm}.maximum_combined_decision_runtime_s",
            cell.get("maximum_combined_decision_runtime_s"),
            derived_runtime,
        )

    decision_labels: dict[str, str] = {}
    for name, comparator in EXPECTED_CONTRASTS.items():
        saved = contrasts[name]
        full_rows = indexed["full_active"]
        comparator_rows = indexed[comparator]
        for endpoint, row_key in (
            ("terminal", "terminal_joint_success"),
            ("tail80", "tail80_joint_success"),
        ):
            full_only = comparator_only = 0
            for seed in sorted(declared_seeds):
                full_success = _parse_csv_bool(
                    full_rows[seed][row_key], f"full_active.{row_key}"
                )
                comparator_success = _parse_csv_bool(
                    comparator_rows[seed][row_key], f"{comparator}.{row_key}"
                )
                full_only += int(full_success and not comparator_success)
                comparator_only += int(comparator_success and not full_success)
            discordance = _mapping(
                saved.get(f"{endpoint}_discordance"),
                f"paired_contrasts.{name}.{endpoint}_discordance",
            )
            if _integer(discordance.get("full_only"), "full_only") != full_only:
                raise RuntimeError(f"{name}: row-derived {endpoint} full_only differs")
            if (
                _integer(discordance.get("comparator_only"), "comparator_only")
                != comparator_only
            ):
                raise RuntimeError(
                    f"{name}: row-derived {endpoint} comparator_only differs"
                )
            _assert_close(
                f"paired_contrasts.{name}.{endpoint}.McNemar",
                discordance.get("exact_mcnemar_p_two_sided"),
                _exact_mcnemar(full_only, comparator_only),
            )
            difference_key = (
                f"{endpoint}_success_rate_difference_full_minus_comparator"
            )
            _assert_close(
                f"paired_contrasts.{name}.{difference_key}",
                saved.get(difference_key),
                (full_only - comparator_only) / EXPECTED_EPISODES,
            )

        component = COMPONENT_FOR_CONTRAST[name]
        safety = all(
            sum(_parse_csv_int(row[key], f"full_active.{key}") for row in full_rows.values())
            <= sum(_parse_csv_int(row[key], f"{comparator}.{key}") for row in comparator_rows.values())
            for key in (
                "unsafe_transition_count",
                "unsafe_track_start_count",
                "unsafe_track_end_count",
            )
        )
        material = bool(
            float(saved["terminal_success_rate_difference_full_minus_comparator"])
            >= 0.05
            or float(saved["tail80_success_rate_difference_full_minus_comparator"])
            >= 0.05
        )
        raw_labels = ALLOWED_COMPONENT_DECISIONS[component]
        positive = next(label for label in raw_labels if label.startswith("MATERIAL_"))
        negative = next(label for label in raw_labels if label.startswith("NO_MATERIAL_"))
        decision_labels[component] = (
            positive if bool(summary["integrity_valid"]) and safety and material else negative
        )
    if decision_labels != summary.get("component_decisions"):
        raise RuntimeError("component decisions differ from row-derived decisions")


def _count_pair(cell: Mapping[str, Any]) -> str:
    return (
        f"{int(cell['terminal_success_count'])}/"
        f"{int(cell['tail80_success_count'])}"
    )


def _distribution_pair(cell: Mapping[str, Any], key: str) -> str:
    distribution = _mapping(cell[key], key)
    return (
        f"{float(distribution['median']):.2f}/"
        f"{float(distribution['p95']):.2f}"
    )


def _format_p(value: float) -> str:
    if value < 0.001:
        return r"\(<0.001\)"
    return rf"\({value:.3f}\)"


def _difference_interval_pair(
    contrast: Mapping[str, Any],
    by_arm: Mapping[str, Mapping[str, Any]],
    *,
    comparator: str,
) -> str:
    formatted: list[str] = []
    endpoint_specs = (
        (
            "terminal",
            "terminal_success_rate_difference_full_minus_comparator",
            "terminal_discordance",
        ),
        (
            "tail80",
            "tail80_success_rate_difference_full_minus_comparator",
            "tail80_discordance",
        ),
    )
    for endpoint, difference_name, discordance_name in endpoint_specs:
        discordance = _mapping(contrast[discordance_name], discordance_name)
        cells = _paired_cells(
            by_arm,
            comparator=comparator,
            endpoint=endpoint,
            full_only=int(discordance["full_only"]),
            comparator_only=int(discordance["comparator_only"]),
        )
        difference, lower, upper = (
            paired_stats.paired_newcombe_hybrid_score_interval(*cells)
        )
        reported = float(contrast[difference_name])
        if not math.isclose(
            difference,
            reported,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"{comparator} {endpoint}: Newcombe point estimate differs "
                "from the campaign summary"
            )
        formatted.append(
            f"{100.0 * difference:+.1f}"
            rf"\,[{100.0 * lower:.1f},\,{100.0 * upper:.1f}]"
        )
    return (
        r"\shortstack[c]{"
        + rf"\({formatted[0]}\)\\"
        + rf"\({formatted[1]}\)"
        + "}"
    )


def _discordance_pair(contrast: Mapping[str, Any]) -> str:
    terminal = _mapping(
        contrast["terminal_discordance"],
        "terminal_discordance",
    )
    tail80 = _mapping(
        contrast["tail80_discordance"],
        "tail80_discordance",
    )
    return (
        f"{int(terminal['full_only'])}/{int(terminal['comparator_only'])}; "
        f"{int(tail80['full_only'])}/{int(tail80['comparator_only'])}"
    )


def _p_pair(contrast: Mapping[str, Any]) -> str:
    terminal = _mapping(
        contrast["terminal_discordance"],
        "terminal_discordance",
    )
    tail80 = _mapping(
        contrast["tail80_discordance"],
        "tail80_discordance",
    )
    return (
        f"{_format_p(float(terminal['exact_mcnemar_p_two_sided']))}/"
        f"{_format_p(float(tail80['exact_mcnemar_p_two_sided']))}"
    )


def render_table(
    by_arm: Mapping[str, Mapping[str, Any]],
    contrasts: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, str],
) -> str:
    lines = [
        r"\begin{table*}[t]",
        (
            r"\caption{Planner-component ablation on 100 paired scenarios. "
            r"Counts are out of 100; no-TRACK episodes receive 442 s in the "
            r"time-to-first-TRACK summary; errors are median/p95. Lower-panel lines "
            r"show terminal then Tail80 complete-minus-comparator differences "
            r"with paired Newcombe hybrid-score method-10 95\% intervals and discordances. "
            r"A material effect required at least 5 percentage points without "
            r"increasing any truth-invalid transition, TRACK-start, or "
            r"TRACK-end count. A negative decision does not establish "
            r"equivalence.}"
        ),
        r"\label{tab:planner_component_ablation}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        (
            r"\begin{tabularx}{\textwidth}{@{}"
            r">{\raggedright\arraybackslash}p{0.245\textwidth}"
            r"*{6}{>{\centering\arraybackslash}X}@{}}"
        ),
        r"\toprule",
        (
            r"Acquisition planner & Terminal/Tail80 [count] & Ever-lock "
            r"[count] & Time to first TRACK, median [s] & Terminal \(e_p\) "
            r"median/p95 [m] & Terminal \(e_f\) median/p95 [m] & "
            r"Maximum decision time [s]\\"
        ),
        r"\midrule",
    ]
    for arm in EXPECTED_ARMS:
        cell = by_arm[arm]
        first_track = _mapping(
            cell["first_track_analysis_time_s"],
            "first_track_analysis_time_s",
        )
        lines.append(
            " & ".join(
                (
                    ARM_LABELS[arm],
                    _count_pair(cell),
                    str(int(cell["ever_lock_count"])),
                    f"{float(first_track['median']):.0f}",
                    _distribution_pair(cell, "terminal_localization_error_m"),
                    _distribution_pair(cell, "terminal_formation_error_m"),
                    f"{float(cell['maximum_combined_decision_runtime_s']):.3f}",
                )
            )
            + r"\\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabularx}",
            r"\vspace{2pt}",
            (
                r"\begin{tabularx}{\textwidth}{@{}"
                r">{\raggedright\arraybackslash}p{0.22\textwidth}"
                r">{\centering\arraybackslash}p{0.21\textwidth}"
                r">{\centering\arraybackslash}p{0.17\textwidth}"
                r">{\centering\arraybackslash}p{0.135\textwidth}"
                r">{\raggedright\arraybackslash}X@{}}"
            ),
            r"\toprule",
            (
                r"Comparator to complete planner & \(\Delta\) terminal/"
                r"Tail80 [pp; paired 95\% interval] & Discordant complete/comparator "
                r"(terminal; Tail80) & McNemar \(p\) terminal/Tail80 & "
                r"Prespecified material-effect decision\\"
            ),
            r"\midrule",
        )
    )
    for name, comparator in EXPECTED_CONTRASTS.items():
        contrast = contrasts[name]
        component = COMPONENT_FOR_CONTRAST[name]
        lines.append(
            " & ".join(
                (
                    COMPARATOR_LABELS[comparator],
                    _difference_interval_pair(
                        contrast,
                        by_arm,
                        comparator=comparator,
                    ),
                    _discordance_pair(contrast),
                    _p_pair(contrast),
                    decisions[component],
                )
            )
            + r"\\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        )
    )
    return "\n".join(lines)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    paired_stats._validate_newcombe_implementation()
    summary_path, rows_path = resolve_inputs(args)
    summary = load_summary(summary_path)
    by_arm, contrasts, decisions = validate_summary(summary)
    validate_rows_against_summary(
        summary,
        load_rows(rows_path),
        by_arm,
        contrasts,
    )
    output = args.output.expanduser().resolve()
    write_atomic(output, render_table(by_arm, contrasts, decisions))
    print(output)


if __name__ == "__main__":
    main()
