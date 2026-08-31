#!/usr/bin/env python3
"""Generate the descriptive closed-loop stress table from released rows.

The released V35 campaign is complete but failed its frozen runtime integrity
criterion.  Accordingly, this script refuses to create Table 11 unless the user
explicitly supplies ``--allow-invalid-descriptive``.  With that acknowledgement,
every displayed number is derived from the public episode CSV and cross-checked
against the compact campaign summary before LaTeX is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data/tables"
DEFAULT_SUMMARY = TABLES / "closed_loop_stress_summary.json"
DEFAULT_ROWS = TABLES / "closed_loop_stress_episode_rows.csv"
DEFAULT_OUTPUT = ROOT / "results/tables/table_closed_loop_stress.tex"

EXPECTED_EPISODES = 100
PRIMARY_ARM = "v24_nominal_model"
PROFILE_ARM = "v35_bias_gate_300"
CONDITIONS = (
    "nominal",
    "doppler_common_bias_p003",
    "doppler_differential_bias_003",
    "doppler_scale_102",
    "colored_noise_rho09_sd003",
    "dropout_iid10",
    "broadcast_delay_2s",
    "dead_reckoning_scale_101",
)
PROFILE_CONDITIONS = (
    "nominal",
    "doppler_common_bias_p003",
    "doppler_differential_bias_003",
    "colored_noise_rho09_sd003",
)
CONDITION_LABELS = {
    "nominal": "Nominal",
    "doppler_common_bias_p003": "Common Doppler bias",
    "doppler_differential_bias_003": "Differential Doppler bias",
    "doppler_scale_102": r"Doppler scale \(1.02\)",
    "colored_noise_rho09_sd003": "Colored AR(1) noise",
    "dropout_iid10": r"10\% row dropout",
    "broadcast_delay_2s": "Nominal 2-s position-broadcast lag",
    "dead_reckoning_scale_101": r"Dead-reckoning scale \(1.01\)",
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
        help="Released per-arm episode CSV.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-invalid-descriptive",
        action="store_true",
        help=(
            "Acknowledge that integrity_valid is false and generate only the "
            "explicitly labelled descriptive table."
        ),
    )
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


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be an object")
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


def _csv_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise RuntimeError(f"{context} is not Boolean: {value!r}")


def _csv_int(value: str, context: str) -> int:
    try:
        output = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} is not an integer: {value!r}") from error
    if output < 0:
        raise RuntimeError(f"{context} must be nonnegative")
    return output


def _csv_float(value: str, context: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{context} is not numeric: {value!r}") from error
    if not math.isfinite(output):
        raise RuntimeError(f"{context} must be finite")
    return output


def _assert_close(context: str, saved: Any, derived: float) -> None:
    if not math.isclose(
        _number(saved, context),
        float(derived),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(f"{context} differs from the released episode rows")


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != EXPECTED_EPISODES or not np.all(np.isfinite(array)):
        raise RuntimeError("each stress-cell distribution requires 100 finite values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _crosscheck_distribution(
    context: str,
    saved_value: Any,
    derived: Mapping[str, float],
) -> None:
    saved = _mapping(saved_value, context)
    for key, value in derived.items():
        _assert_close(f"{context}.{key}", saved.get(key), value)


def validate_and_derive(
    summary: Mapping[str, Any],
    rows: list[dict[str, str]],
    *,
    allow_invalid_descriptive: bool,
) -> dict[str, dict[str, Any]]:
    if summary.get("status") != "complete":
        raise RuntimeError("campaign status is not complete")
    if summary.get("smoke") is not False:
        raise RuntimeError("publication table requires a non-smoke campaign")
    integrity_valid = summary.get("integrity_valid")
    if not isinstance(integrity_valid, bool):
        raise RuntimeError("integrity_valid must be Boolean")
    integrity_checks = _mapping(
        summary.get("integrity_checks"),
        "integrity_checks",
    )
    failed_checks = {
        str(name)
        for name, value in integrity_checks.items()
        if not isinstance(value, bool) or not value
    }
    if integrity_valid != (not failed_checks):
        raise RuntimeError("integrity_valid disagrees with integrity_checks")
    if not integrity_valid and failed_checks != {"maximum_runtime_below_2s"}:
        raise RuntimeError(
            "the descriptive caption is valid only for the frozen V35 "
            "runtime-only integrity failure"
        )
    if not integrity_valid and not allow_invalid_descriptive:
        raise RuntimeError(
            "refusing integrity-invalid V35 evidence; pass "
            "--allow-invalid-descriptive to generate the labelled descriptive table"
        )
    if _integer(summary.get("row_count"), "row_count") != 1200:
        raise RuntimeError("campaign summary row_count is not 1200")
    if _integer(summary.get("expected_row_count"), "expected_row_count") != 1200:
        raise RuntimeError("campaign summary expected_row_count is not 1200")
    if len(rows) != 1200:
        raise RuntimeError("episode-row CSV must contain exactly 1200 rows")

    required = {
        "episode_seed",
        "condition",
        "arm",
        "terminal_joint_success",
        "tail80_joint_success",
        "dwell15_joint_success",
        "terminal_localization_error_m",
        "terminal_formation_error_m",
        "ever_locked",
        "unsafe_transition_count",
        "unsafe_track_start_count",
        "unsafe_track_end_count",
        "maximum_combined_decision_runtime_s",
    }
    expected_cells = {(condition, PRIMARY_ARM) for condition in CONDITIONS}
    expected_cells.update((condition, PROFILE_ARM) for condition in PROFILE_CONDITIONS)
    grouped: dict[tuple[str, str], dict[int, dict[str, str]]] = {
        cell: {} for cell in expected_cells
    }
    for row_number, row in enumerate(rows, start=2):
        missing = required.difference(row)
        if missing:
            raise RuntimeError(
                f"CSV row {row_number} is missing columns {sorted(missing)}"
            )
        cell = (row["condition"], row["arm"])
        if cell not in grouped:
            raise RuntimeError(f"CSV row {row_number} has unexpected cell {cell}")
        seed = _csv_int(row["episode_seed"], f"CSV row {row_number}.episode_seed")
        if seed in grouped[cell]:
            raise RuntimeError(f"duplicate seed {seed} in {cell}")
        grouped[cell][seed] = row
    seed_sets = {frozenset(cell_rows) for cell_rows in grouped.values()}
    if len(seed_sets) != 1 or len(next(iter(seed_sets))) != EXPECTED_EPISODES:
        raise RuntimeError("the 12 closed-loop stress cells are not fully paired")

    saved_cells = _mapping(summary.get("by_condition_arm"), "by_condition_arm")
    if set(saved_cells) != {f"{condition}@{arm}" for condition, arm in expected_cells}:
        raise RuntimeError("by_condition_arm contains an unexpected cell set")

    derived_primary: dict[str, dict[str, Any]] = {}
    for condition, arm in sorted(expected_cells):
        key = f"{condition}@{arm}"
        saved = _mapping(saved_cells.get(key), f"by_condition_arm.{key}")
        ordered = [grouped[(condition, arm)][seed] for seed in sorted(grouped[(condition, arm)])]
        counts = {
            "terminal_success_count": sum(
                _csv_bool(row["terminal_joint_success"], f"{key}.terminal")
                for row in ordered
            ),
            "tail80_success_count": sum(
                _csv_bool(row["tail80_joint_success"], f"{key}.tail80")
                for row in ordered
            ),
            "dwell15_success_count": sum(
                _csv_bool(row["dwell15_joint_success"], f"{key}.dwell15")
                for row in ordered
            ),
            "ever_lock_count": sum(
                _csv_bool(row["ever_locked"], f"{key}.ever_locked")
                for row in ordered
            ),
            "unsafe_transition_count": sum(
                _csv_int(row["unsafe_transition_count"], f"{key}.unsafe_transition")
                for row in ordered
            ),
            "unsafe_track_start_count": sum(
                _csv_int(row["unsafe_track_start_count"], f"{key}.unsafe_start")
                for row in ordered
            ),
            "unsafe_track_end_count": sum(
                _csv_int(row["unsafe_track_end_count"], f"{key}.unsafe_end")
                for row in ordered
            ),
        }
        if _integer(saved.get("episodes"), f"{key}.episodes") != EXPECTED_EPISODES:
            raise RuntimeError(f"{key}.episodes is not 100")
        for count_name, value in counts.items():
            if _integer(saved.get(count_name), f"{key}.{count_name}") != value:
                raise RuntimeError(f"{key}.{count_name} differs from episode rows")
        localization = _distribution(
            [
                _csv_float(row["terminal_localization_error_m"], f"{key}.e_p")
                for row in ordered
            ]
        )
        formation = _distribution(
            [
                _csv_float(row["terminal_formation_error_m"], f"{key}.e_f")
                for row in ordered
            ]
        )
        _crosscheck_distribution(
            f"{key}.terminal_localization_error_m",
            saved.get("terminal_localization_error_m"),
            localization,
        )
        _crosscheck_distribution(
            f"{key}.terminal_formation_error_m",
            saved.get("terminal_formation_error_m"),
            formation,
        )
        maximum_runtime = max(
            _csv_float(
                row["maximum_combined_decision_runtime_s"],
                f"{key}.maximum_combined_decision_runtime_s",
            )
            for row in ordered
        )
        _assert_close(
            f"{key}.maximum_combined_decision_runtime_s",
            saved.get("maximum_combined_decision_runtime_s"),
            maximum_runtime,
        )
        if arm == PRIMARY_ARM:
            invalid_track_episodes = sum(
                _csv_int(row["unsafe_track_start_count"], f"{key}.unsafe_start") > 0
                or _csv_int(row["unsafe_track_end_count"], f"{key}.unsafe_end") > 0
                for row in ordered
            )
            derived_primary[condition] = {
                **counts,
                "terminal_localization_error_m": localization,
                "terminal_formation_error_m": formation,
                "invalid_track_episode_count": invalid_track_episodes,
            }
    return derived_primary


def render_table(cells: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "% WARNING: descriptive output from an integrity-invalid V35 campaign.",
        r"\begin{table*}[t]",
        (
            r"\caption{Integrated stress screen of the both-link information-guided "
            r"loop on 100 fresh scenarios per condition. Outcomes are counts out of "
            r"100. Results are descriptive because the joint campaign failed its "
            r"frozen runtime rule. The last column is the truth-invalid release "
            r"count / episodes with any truth-invalid TRACK sample.}"
        ),
        r"\label{tab:integrated_stress}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{@{}p{0.19\textwidth}rrrrp{0.17\textwidth}p{0.17\textwidth}p{0.13\textwidth}@{}}",
        r"\toprule",
        (
            r"Condition & Terminal & Tail80 & Dwell15 & Ever-lock & Terminal "
            r"\(e_p\), median/p95 [m] & Terminal \(e_f\), median/p95 [m] & "
            r"Truth-invalid release / episodes with invalid TRACK\\"
        ),
        r"\midrule",
    ]
    for condition in CONDITIONS:
        cell = cells[condition]
        ep = _mapping(cell["terminal_localization_error_m"], "e_p")
        ef = _mapping(cell["terminal_formation_error_m"], "e_f")
        lines.append(
            " & ".join(
                (
                    CONDITION_LABELS[condition],
                    str(int(cell["terminal_success_count"])),
                    str(int(cell["tail80_success_count"])),
                    str(int(cell["dwell15_success_count"])),
                    str(int(cell["ever_lock_count"])),
                    f"{float(ep['median']):.2f} / {float(ep['p95']):.2f}",
                    f"{float(ef['median']):.2f} / {float(ef['p95']):.2f}",
                    (
                        f"{int(cell['unsafe_transition_count'])} / "
                        f"{int(cell['invalid_track_episode_count'])}"
                    ),
                )
            )
            + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""))
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
    summary_path, rows_path = resolve_inputs(args)
    summary = load_summary(summary_path)
    cells = validate_and_derive(
        summary,
        load_rows(rows_path),
        allow_invalid_descriptive=args.allow_invalid_descriptive,
    )
    if summary.get("integrity_valid") is False:
        print(
            "WARNING: generating descriptive Table 11 from integrity-invalid V35 "
            "data; the frozen runtime criterion failed.",
            file=sys.stderr,
        )
    output = args.output.expanduser().resolve()
    write_atomic(output, render_table(cells))
    print(output)


if __name__ == "__main__":
    main()
