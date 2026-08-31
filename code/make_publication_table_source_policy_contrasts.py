#!/usr/bin/env python3
"""Generate the compact paired-contrast table for the source-policy campaign."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import make_publication_figure_source_policy_ablation as figure


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data/tables"
DEFAULT_SUMMARY = TABLES / "leader_source_policy_summary.json"
DEFAULT_ROWS = TABLES / "leader_source_policy_episode_rows.csv"
DEFAULT_OUTPUT = ROOT / "results/tables/table_source_policy_contrasts.tex"


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
    parser.add_argument("--allow-smoke", action="store_true")
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


def format_p(value: float) -> str:
    if value < 0.001:
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"
    if value < 0.1:
        return rf"\({value:.4f}\)"
    return rf"\({value:.3f}\)"


def paired_newcombe_hybrid_score_interval(
    both_success: int,
    right_only: int,
    left_only: int,
    neither_success: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float, float]:
    """Newcombe (1998) method 10 CI for a paired risk difference.

    The point estimate and interval are oriented as ``right - left``.
    Wilson score intervals are formed for both marginal proportions, and the
    paired correction uses the continuity-corrected phi coefficient specified
    by Newcombe's method 10.
    """

    cells = (both_success, right_only, left_only, neither_success)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in cells):
        raise TypeError("paired cell counts must be integers")
    if any(value < 0 for value in cells):
        raise ValueError("paired cell counts must be nonnegative")
    total = sum(cells)
    if total <= 0:
        raise ValueError("paired interval requires at least one pair")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("z must be finite and positive")

    right_success = both_success + right_only
    left_success = both_success + left_only
    right_rate = right_success / total
    left_rate = left_success / total
    difference = right_rate - left_rate

    right_lower, right_upper = figure.wilson_interval(
        right_success,
        total,
        z=z,
    )
    left_lower, left_upper = figure.wilson_interval(
        left_success,
        total,
        z=z,
    )

    phi_denominator = math.sqrt(
        (both_success + right_only)
        * (left_only + neither_success)
        * (both_success + left_only)
        * (right_only + neither_success)
    )
    phi_numerator = (
        both_success * neither_success - right_only * left_only
    )
    # Newcombe method 10 applies the continuity correction to positive phi.
    if phi_numerator > 0:
        phi_numerator = max(phi_numerator - total / 2.0, 0.0)
    phi = phi_numerator / phi_denominator if phi_denominator > 0.0 else 0.0
    phi = min(1.0, max(-1.0, phi))

    right_lower_half = right_rate - right_lower
    right_upper_half = right_upper - right_rate
    left_lower_half = left_rate - left_lower
    left_upper_half = left_upper - left_rate
    lower_radicand = (
        right_lower_half**2
        - 2.0 * phi * right_lower_half * left_upper_half
        + left_upper_half**2
    )
    upper_radicand = (
        right_upper_half**2
        - 2.0 * phi * right_upper_half * left_lower_half
        + left_lower_half**2
    )
    lower = difference - math.sqrt(max(0.0, lower_radicand))
    upper = difference + math.sqrt(max(0.0, upper_radicand))
    return (
        difference,
        max(-1.0, lower),
        min(1.0, upper),
    )


def _validate_newcombe_implementation() -> None:
    """Reproduce Newcombe's example and the boundary case used in the paper."""

    estimate, lower, upper = paired_newcombe_hybrid_score_interval(
        20,
        12,
        2,
        16,
    )
    expected = (0.2, 0.0562, 0.3292)
    observed = (estimate, lower, upper)
    for name, value, target in zip(
        ("estimate", "lower", "upper"),
        observed,
        expected,
    ):
        if not math.isclose(value, target, rel_tol=0.0, abs_tol=5.0e-5):
            raise RuntimeError(
                "Newcombe method-10 self-test failed for "
                f"{name}: observed {value:.8f}, expected {target:.4f}"
            )
    boundary = paired_newcombe_hybrid_score_interval(96, 4, 0, 0)
    boundary_expected = (0.04, -0.0042808501, 0.0983707144)
    for name, value, target in zip(
        ("estimate", "lower", "upper"),
        boundary,
        boundary_expected,
    ):
        if not math.isclose(value, target, rel_tol=0.0, abs_tol=5.0e-10):
            raise RuntimeError(
                "Newcombe boundary-case self-test failed for "
                f"{name}: observed {value:.10f}, expected {target:.10f}"
            )


def _paired_binary_counts(
    grouped: dict[str, list[dict[str, str]]],
    *,
    left_arm: str,
    right_arm: str,
    success_column: str,
) -> tuple[int, int, int, int]:
    required_columns = {"episode_seed", success_column}
    indexed: dict[str, dict[int, dict[str, str]]] = {}
    for arm in (left_arm, right_arm):
        if arm not in grouped:
            raise RuntimeError(f"missing paired arm {arm!r}")
        by_seed: dict[int, dict[str, str]] = {}
        for row_number, row in enumerate(grouped[arm], start=2):
            missing = required_columns.difference(row)
            if missing:
                raise RuntimeError(
                    f"{arm}: CSV row {row_number} is missing {sorted(missing)}"
                )
            try:
                seed = int(row["episode_seed"])
            except ValueError as error:
                raise RuntimeError(
                    f"{arm}: CSV row {row_number} has an invalid episode_seed"
                ) from error
            if seed in by_seed:
                raise RuntimeError(f"{arm}: duplicate episode seed {seed}")
            by_seed[seed] = row
        indexed[arm] = by_seed

    left = indexed[left_arm]
    right = indexed[right_arm]
    if set(left) != set(right):
        raise RuntimeError(
            f"{left_arm} and {right_arm}: paired seed sets differ"
        )
    if not left:
        raise RuntimeError(f"{left_arm} and {right_arm}: no paired rows")

    both_success = 0
    right_only = 0
    left_only = 0
    neither_success = 0
    for seed in sorted(left):
        try:
            left_value = figure.parse_bool(left[seed][success_column])
            right_value = figure.parse_bool(right[seed][success_column])
        except ValueError as error:
            raise RuntimeError(
                f"seed {seed}: invalid {success_column} in "
                f"{left_arm} or {right_arm}"
            ) from error
        if left_value and right_value:
            both_success += 1
        elif right_value:
            right_only += 1
        elif left_value:
            left_only += 1
        else:
            neither_success += 1
    return both_success, right_only, left_only, neither_success


def contrast_row(
    label: str,
    contrast: dict[str, Any],
    grouped: dict[str, list[dict[str, str]]],
    *,
    endpoint: str,
    expected_left: str,
    expected_right: str,
) -> str:
    if contrast.get("left") != expected_left or contrast.get("right") != expected_right:
        raise RuntimeError(
            f"{label}: unexpected contrast orientation "
            f"{contrast.get('left')!r} -> {contrast.get('right')!r}"
        )
    if endpoint == "terminal":
        success_key = "terminal_success_rate_difference_right_minus_left"
        discordance_key = "terminal_discordance"
        improvement_key = "terminal_localization_improvement_m"
        success_column = "terminal_joint_success"
    elif endpoint == "checkpoint_60s":
        success_key = (
            "checkpoint_60s_localization_success_rate_difference_right_minus_left"
        )
        discordance_key = "checkpoint_60s_localization_discordance"
        improvement_key = "checkpoint_60s_localization_improvement_m"
        success_column = "checkpoint_60s_localization_below_7m"
    else:
        raise ValueError(f"unsupported endpoint: {endpoint}")

    discordance = contrast[discordance_key]
    improvement = contrast[improvement_key]
    interval = improvement["bootstrap_mean_95"]
    difference_pp = 100.0 * float(contrast[success_key])
    favored = int(discordance["right_only"])
    opposed = int(discordance["left_only"])
    p_value = float(discordance["exact_mcnemar_p_two_sided"])
    paired_cells = _paired_binary_counts(
        grouped,
        left_arm=expected_left,
        right_arm=expected_right,
        success_column=success_column,
    )
    difference, lower, upper = paired_newcombe_hybrid_score_interval(
        *paired_cells
    )
    if not math.isclose(
        difference,
        float(contrast[success_key]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            f"{label}: paired risk difference differs from campaign summary"
        )
    if paired_cells[1] != favored or paired_cells[2] != opposed:
        raise RuntimeError(
            f"{label}: discordant cell counts differ from campaign summary"
        )

    fields = (
        label,
        (
            f"{difference_pp:+.1f} "
            f"[{100.0 * lower:.1f}, {100.0 * upper:.1f}]"
        ),
        f"{favored}/{opposed}",
        format_p(p_value),
        (
            f"{float(improvement['mean']):.2f} "
            f"[{float(interval[0]):.2f}, {float(interval[1]):.2f}]; "
            f"med. {float(improvement['median']):.2f}"
        ),
    )
    return " & ".join(fields) + r"\\"


def render_table(
    summary: dict[str, Any],
    grouped: dict[str, list[dict[str, str]]],
    smoke: bool,
) -> str:
    contrasts = summary["paired_contrasts"]
    rows = [
        (
            "Information-guided versus fixed S-turn; leader 1 link (terminal)",
            contrasts["active_vs_fixed@leader1_only"],
            "terminal",
            "leader1_only__fixed_s_turn",
            "leader1_only__belief_active",
        ),
        (
            "Information-guided versus fixed S-turn; leader 2 link (terminal)",
            contrasts["active_vs_fixed@leader2_only"],
            "terminal",
            "leader2_only__fixed_s_turn",
            "leader2_only__belief_active",
        ),
        (
            "Information-guided versus fixed S-turn; both links (terminal)",
            contrasts["active_vs_fixed@both_leaders"],
            "terminal",
            "both_leaders__fixed_s_turn",
            "both_leaders__belief_active",
        ),
        (
            "Both vs leader 1 link; information-guided (terminal)",
            contrasts["both_vs_leader1_only@belief_active"],
            "terminal",
            "leader1_only__belief_active",
            "both_leaders__belief_active",
        ),
        (
            "Both vs leader 2 link; information-guided (terminal)",
            contrasts["both_vs_leader2_only@belief_active"],
            "terminal",
            "leader2_only__belief_active",
            "both_leaders__belief_active",
        ),
        (
            "Both vs leader 1 link; fixed S-turn (60 s)",
            contrasts["both_vs_leader1_only@fixed_s_turn"],
            "checkpoint_60s",
            "leader1_only__fixed_s_turn",
            "both_leaders__fixed_s_turn",
        ),
        (
            "Both vs leader 2 link; fixed S-turn (60 s)",
            contrasts["both_vs_leader2_only@fixed_s_turn"],
            "checkpoint_60s",
            "leader2_only__fixed_s_turn",
            "both_leaders__fixed_s_turn",
        ),
    ]

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Prespecified paired contrasts. Information-guided-versus-fixed-S-turn rows use terminal joint success; fixed-S-turn both-versus-single rows use 60-s localization on identical histories; information-guided both-versus-single rows describe the closed loop. Positive values favor the first condition. Binary differences show paired Newcombe hybrid-score 95\% intervals (method 10); continuous \(e_p\) reductions show paired means with deterministic-bootstrap 95\% intervals and paired medians.}",
        r"\label{tab:source_policy_contrasts}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X>{\centering\arraybackslash}p{0.16\textwidth}>{\centering\arraybackslash}p{0.12\textwidth}>{\centering\arraybackslash}p{0.09\textwidth}>{\centering\arraybackslash}p{0.27\textwidth}@{}}",
        r"\toprule",
        r"Comparison and endpoint & Difference [percentage points; paired 95\% interval] & Discordant pairs favorable/opposite & Exact McNemar \(p\) & \(e_p\) reduction [m]: mean [95\% interval]; median\\",
        r"\midrule",
    ]
    if smoke:
        lines.append(
            r"\multicolumn{5}{c}{\textbf{Smoke test---not reporting evidence}}\\"
        )
    for label, contrast, endpoint, expected_left, expected_right in rows:
        lines.append(
            contrast_row(
                label,
                contrast,
                grouped,
                endpoint=endpoint,
                expected_left=expected_left,
                expected_right=expected_right,
            )
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


def main() -> None:
    args = parse_args()
    _validate_newcombe_implementation()
    summary_path, rows_path = resolve_inputs(args)
    summary = figure.load_json(summary_path)
    csv_rows = figure.load_csv(rows_path)
    grouped, smoke = figure.validate_and_group(
        summary,
        csv_rows,
        args.allow_smoke,
    )
    figure.validate_publication_paired_contrasts(summary, grouped)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_table(summary, grouped, smoke), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
