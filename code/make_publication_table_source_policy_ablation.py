#!/usr/bin/env python3
"""Generate the publication table for the source-count/acquisition ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import make_publication_figure_source_policy_ablation as figure


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data/tables"
DEFAULT_SUMMARY = TABLES / "leader_source_policy_summary.json"
DEFAULT_ROWS = TABLES / "leader_source_policy_episode_rows.csv"
DEFAULT_OUTPUT = ROOT / "results/tables/table_source_policy_ablation.tex"

SOURCE_LABELS = {
    "leader1_only": "Leader 1 link",
    "leader2_only": "Leader 2 link",
    "both_leaders": "Both links",
}
POLICY_LABELS = {
    "fixed_s_turn": "Fixed S-turn",
    "belief_active": "Information-guided",
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


def compact_distribution(cell: dict[str, Any], digits: int = 2) -> str:
    values = (cell["median"], cell["p95"], cell["max"])
    return "/".join(f"{float(value):.{digits}f}" for value in values)


def first_track_summary(cell: dict[str, Any]) -> str:
    distribution = cell["first_track_action_time_s"]
    if distribution is None:
        return "--"
    return (
        f"{float(distribution['median']):.0f}--{float(distribution['p95']):.0f}"
    )


def render_table(summary: dict[str, Any], smoke: bool) -> str:
    episodes = int(next(iter(summary["by_arm"].values()))["episodes"])
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{Paired \(3\times2\) Doppler-link--acquisition experiment ({episodes} common scenarios per cell). Fixed S-turn values at \(60\unit{{s}}\) compare links on identical histories; information-guided \(60\)-s and all terminal values describe the closed loop. Single-link rows mask one Doppler link only in localization, planning, and the evidence gate; both leaders and the task remain. First-TRACK median/p95 uses only episodes that entered TRACK. Errors are median/p95/max. The final column reports offline truth-invalid release, TRACK-start, and TRACK-end counts.}}",
        r"\label{tab:source_policy_ablation}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.145\textwidth}>{\raggedright\arraybackslash}p{0.155\textwidth}*{5}{>{\centering\arraybackslash}X}@{}}",
        r"\toprule",
        r"Doppler data used & Acquisition rule & \(e_p<7\unit{m}\) at \(60\unit{s}\) [\%] & Ever-lock [\%] & Time to first TRACK, median--p95 [s] & Terminal joint success [\%] & Tail80 success [\%]\\",
        r"\midrule",
    ]
    if smoke:
        lines.append(r"\multicolumn{7}{c}{\textbf{Smoke test---not reporting evidence}}\\")
    for source in figure.SOURCE_ORDER:
        for policy in figure.POLICY_ORDER:
            cell = summary["by_arm"][figure.arm_name(source, policy)]
            episodes = int(cell["episodes"])
            checkpoint = 100.0 * int(
                cell["checkpoint_60s_localization_success_count"]
            ) / episodes
            gate_release = 100.0 * int(cell["ever_lock_count"]) / episodes
            terminal = 100.0 * int(cell["terminal_success_count"]) / episodes
            tail80 = 100.0 * int(cell["tail80_success_count"]) / episodes
            lines.append(
                " & ".join(
                    (
                        SOURCE_LABELS[source],
                        POLICY_LABELS[policy],
                        f"{checkpoint:.0f}\\%",
                        f"{gate_release:.0f}\\%",
                        first_track_summary(cell),
                        f"{terminal:.0f}\\%",
                        f"{tail80:.0f}\\%",
                    )
                )
                + r"\\"
            )
        if source != figure.SOURCE_ORDER[-1]:
            lines.append(r"\addlinespace[1pt]")
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabularx}",
            r"\vspace{2pt}",
            r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.145\textwidth}>{\raggedright\arraybackslash}p{0.155\textwidth}>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}p{0.17\textwidth}@{}}",
            r"\toprule",
            r"Doppler data used & Acquisition rule & Terminal \(e_p\), median/p95/max [m] & Terminal \(e_f\), median/p95/max [m] & Truth-invalid release/start/end [count]\\",
            r"\midrule",
        )
    )
    if smoke:
        lines.append(r"\multicolumn{5}{c}{\textbf{Smoke test---not reporting evidence}}\\")
    for source in figure.SOURCE_ORDER:
        for policy in figure.POLICY_ORDER:
            cell = summary["by_arm"][figure.arm_name(source, policy)]
            invalid = (
                f"{int(cell['unsafe_transition_count'])}/"
                f"{int(cell['unsafe_track_start_count'])}/"
                f"{int(cell['unsafe_track_end_count'])}"
            )
            lines.append(
                " & ".join(
                    (
                        SOURCE_LABELS[source],
                        POLICY_LABELS[policy],
                        compact_distribution(
                            cell["terminal_localization_error_m"]
                        ),
                        compact_distribution(cell["terminal_formation_error_m"]),
                        invalid,
                    )
                )
                + r"\\"
            )
        if source != figure.SOURCE_ORDER[-1]:
            lines.append(r"\addlinespace[1pt]")
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
    summary_path, rows_path = resolve_inputs(args)
    summary = figure.load_json(summary_path)
    rows = figure.load_csv(rows_path)
    grouped, smoke = figure.validate_and_group(
        summary,
        rows,
        args.allow_smoke,
    )
    figure.validate_ablation_table_fields(summary, grouped)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_table(summary, smoke), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
