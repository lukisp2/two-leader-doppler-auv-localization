#!/usr/bin/env python3
"""Build the monochrome execution/current qualification figure.

The left panel reproduces the frozen acquisition-by-execution-by-current
diagnosis.  The two right panels use the fresh paired controller study and
show the task-level repair and its command-level mechanism.  All values are
read from compact row-validated release artifacts; no private campaign path or
numerical result is embedded in the drawing code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS_JSON = REPOSITORY_ROOT / "data/tables/v40_publication_results.json"
REPAIR_JSON = REPOSITORY_ROOT / "data/tables/controller_repair_results.json"
OUTPUT_PDF = REPOSITORY_ROOT / "results/figures/fig_policy_execution_current.pdf"
OUTPUT_PNG = (
    REPOSITORY_ROOT / "results/figures/fig_policy_execution_current_600dpi.png"
)

FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 2.58
PNG_DPI = 600

POLICIES = ("fixed_s_turn", "belief_active")
EXECUTIONS = ("kinematic", "low_order_dynamic")
CURRENTS = ("no_current", "bottom_track_visible")
TRACKERS = ("baseline_pid", "delay_aware")


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def _rate(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be a finite rate in [0,1]")
    return number


def load_diagnosis(
    path: Path = DIAGNOSIS_JSON,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    document = _load_json(path)
    if document.get("integrity_valid") is not True:
        raise ValueError("The acquisition/execution/current diagnosis is not integrity-valid")
    arms = document.get("arms")
    if not isinstance(arms, list) or len(arms) != 8:
        raise ValueError("The diagnosis must contain the complete eight-arm design")

    result: dict[tuple[str, str, str], tuple[float, float]] = {}
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict) or int(arm.get("N", -1)) != 100:
            raise ValueError(f"Diagnosis arm {index} is not a 100-episode arm")
        key = (str(arm["policy"]), str(arm["execution"]), str(arm["current"]))
        if key in result:
            raise ValueError(f"Duplicate diagnosis arm: {key}")
        result[key] = (
            _rate(arm["terminal_success_rate"], f"{key} terminal"),
            _rate(arm["Tail80_rate"], f"{key} Tail80"),
        )
    expected = {(p, e, c) for p in POLICIES for e in EXECUTIONS for c in CURRENTS}
    if set(result) != expected:
        raise ValueError("The diagnosis does not contain the exact 2 x 2 x 2 design")
    return result


def load_repair(
    path: Path = REPAIR_JSON,
) -> dict[tuple[str, str], dict[str, float]]:
    document = _load_json(path)
    validation = document.get("validation")
    semantic_audit = document.get("semantic_audit")
    if not isinstance(validation, dict) or validation.get("row_count") != 400:
        raise ValueError("The compact controller export is not the complete 400-row design")
    if validation.get("complete_factorial_support") is not True:
        raise ValueError("The compact controller export lacks complete factorial support")
    if (
        not isinstance(semantic_audit, dict)
        or semantic_audit.get("campaign_integrity") != "valid"
    ):
        raise ValueError("The controller study is not integrity-valid")
    by_arm = document.get("by_arm")
    if not isinstance(by_arm, dict) or len(by_arm) != 4:
        raise ValueError("The controller study must contain four arms")

    result: dict[tuple[str, str], dict[str, float]] = {}
    for arm_name, arm in by_arm.items():
        if not isinstance(arm, dict) or int(arm.get("episodes", -1)) != 100:
            raise ValueError(f"Controller arm {arm_name} is not a 100-episode arm")
        tracker = "delay_aware" if "__delay_aware__" in arm_name else "baseline_pid"
        current = "bottom_track_visible" if arm_name.endswith("bottom_track_visible") else "no_current"
        key = (tracker, current)
        if key in result:
            raise ValueError(f"Duplicate controller arm: {key}")
        result[key] = {
            "terminal": _rate(arm["terminal_success_rate"], f"{key} terminal"),
            "tail80": _rate(arm["tail80_success_rate"], f"{key} Tail80"),
            "saturation": _rate(
                arm["post_track_saturation_fraction"]["mean"], f"{key} saturation"
            ),
            "curvature": _rate(
                arm["post_track_action_curvature_rms"]["mean"], f"{key} curvature"
            ),
        }
    expected = {(tracker, current) for tracker in TRACKERS for current in CURRENTS}
    if set(result) != expected:
        raise ValueError("The controller study does not contain the exact paired design")
    return result


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.6,
            "legend.fontsize": 6.1,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.linewidth": 0.65,
            "lines.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _format_rate_axis(axis: plt.Axes, *, ylabel: bool = False) -> None:
    axis.set_ylim(-0.03, 1.055)
    axis.yaxis.set_major_locator(MultipleLocator(0.20))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", color="0.86", linewidth=0.48, zorder=0)
    axis.tick_params(direction="out", length=2.2, width=0.6, pad=1.8)
    if ylabel:
        axis.set_ylabel("Episode success rate", labelpad=2.0)


def _plot_two_metrics(
    axis: plt.Axes,
    center: float,
    values: tuple[float, float],
    *,
    filled: bool,
) -> None:
    offsets = (-0.026, 0.026)
    axis.plot(
        [center + offsets[0], center + offsets[1]],
        values,
        color="0.55",
        linewidth=0.6,
        zorder=2,
    )
    for x, value, marker in zip(
        (center + offsets[0], center + offsets[1]), values, ("o", "s")
    ):
        axis.plot(
            x,
            value,
            linestyle="none",
            marker=marker,
            markersize=4.35,
            markeredgewidth=0.75,
            markeredgecolor="black",
            markerfacecolor="black" if filled else "white",
            zorder=4,
        )


def _method_legend(axis: plt.Axes, labels: tuple[str, str]) -> None:
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.0,
            markeredgewidth=0.7,
            markeredgecolor="black",
            markerfacecolor=fill,
            label=label,
        )
        for fill, label in zip(("white", "black"), labels)
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.012),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.75,
        borderaxespad=0.0,
    )


def build_figure(
    diagnosis: dict[tuple[str, str, str], tuple[float, float]],
    repair: dict[tuple[str, str], dict[str, float]],
) -> Figure:
    configure_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        gridspec_kw={"width_ratios": (1.33, 1.00, 1.00)},
    )
    figure.subplots_adjust(
        left=0.057,
        right=0.993,
        top=0.785,
        bottom=0.245,
        wspace=0.34,
    )

    # (a) Complete frozen 2 x 2 x 2 diagnosis.
    axis = axes[0]
    cells = tuple((execution, current) for execution in EXECUTIONS for current in CURRENTS)
    for x, (execution, current) in enumerate(cells):
        _plot_two_metrics(
            axis,
            x - 0.15,
            diagnosis[("fixed_s_turn", execution, current)],
            filled=False,
        )
        _plot_two_metrics(
            axis,
            x + 0.15,
            diagnosis[("belief_active", execution, current)],
            filled=True,
        )
    _format_rate_axis(axis, ylabel=True)
    axis.set_title("(a) Acquisition-execution diagnosis", loc="left", pad=19.0)
    axis.set_xlim(-0.46, 3.46)
    axis.set_xticks(
        range(4),
        ("Kinematic\nNo current", "Kinematic\nVisible", "Delayed\nNo current", "Delayed\nVisible"),
    )
    axis.axvline(1.5, color="0.55", linestyle=(0, (2.2, 2.0)), linewidth=0.65)
    _method_legend(axis, ("Fixed S-turn", "Information-guided"))

    # (b) Fresh paired controller comparison, task-level endpoints.
    axis = axes[1]
    for x, current in enumerate(CURRENTS):
        for tracker, offset, filled in (
            ("baseline_pid", -0.16, False),
            ("delay_aware", 0.16, True),
        ):
            arm = repair[(tracker, current)]
            _plot_two_metrics(
                axis,
                x + offset,
                (arm["terminal"], arm["tail80"]),
                filled=filled,
            )
    _format_rate_axis(axis)
    axis.set_title("(b) Controller repair: task success", loc="left", pad=19.0)
    axis.set_xlim(-0.43, 1.43)
    axis.set_xticks((0, 1), ("No current", "Visible current"))
    _method_legend(axis, ("Proportional", "Delay-aware"))

    # (c) Command-level mechanism; both statistics are normalized and dimensionless.
    axis = axes[2]
    metric_markers = {"saturation": "^", "curvature": "D"}
    for x, current in enumerate(CURRENTS):
        for tracker, offset, filled in (
            ("baseline_pid", -0.16, False),
            ("delay_aware", 0.16, True),
        ):
            arm = repair[(tracker, current)]
            xs = (x + offset - 0.026, x + offset + 0.026)
            ys = (arm["saturation"], arm["curvature"])
            axis.plot(xs, ys, color="0.55", linewidth=0.6, zorder=2)
            for metric, x_value, y_value in zip(("saturation", "curvature"), xs, ys):
                axis.plot(
                    x_value,
                    y_value,
                    linestyle="none",
                    marker=metric_markers[metric],
                    markersize=4.35,
                    markeredgewidth=0.75,
                    markeredgecolor="black",
                    markerfacecolor="black" if filled else "white",
                    zorder=4,
                )
    axis.set_ylim(-0.03, 1.055)
    axis.yaxis.set_major_locator(MultipleLocator(0.20))
    axis.set_ylabel("Mean normalized statistic", labelpad=2.0)
    axis.set_title("(c) Controller repair: command use", loc="left", pad=19.0)
    axis.set_xlim(-0.43, 1.43)
    axis.set_xticks((0, 1), ("No current", "Visible current"))
    axis.grid(axis="y", color="0.86", linewidth=0.48, zorder=0)
    axis.tick_params(direction="out", length=2.2, width=0.6, pad=1.8)
    _method_legend(axis, ("Proportional", "Delay-aware"))

    metric_handles = (
        Line2D(
            [], [], linestyle="none", marker="o", markerfacecolor="0.45",
            markeredgecolor="black", markersize=4.1, label="Terminal (circle)"
        ),
        Line2D(
            [], [], linestyle="none", marker="s", markerfacecolor="0.45",
            markeredgecolor="black", markersize=4.1, label="Tail80 (square)"
        ),
        Line2D(
            [], [], linestyle="none", marker="^", markerfacecolor="0.45",
            markeredgecolor="black", markersize=4.1, label="Saturation (triangle)"
        ),
        Line2D(
            [], [], linestyle="none", marker="D", markerfacecolor="0.45",
            markeredgecolor="black", markersize=3.9, label="Curvature RMS (diamond)"
        ),
    )
    figure.legend(
        handles=metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        handletextpad=0.32,
        columnspacing=0.95,
        borderaxespad=0.0,
    )
    return figure


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-json", type=Path, default=DIAGNOSIS_JSON)
    parser.add_argument("--repair-json", type=Path, default=REPAIR_JSON)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="write only the vector PDF",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    diagnosis = load_diagnosis(args.diagnosis_json)
    repair = load_repair(args.repair_json)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_png:
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(diagnosis, repair)
    try:
        figure.savefig(
            args.output_pdf,
            format="pdf",
            metadata={
                "Title": "Execution-current diagnosis and delay-aware controller repair",
                "Subject": "Task success and post-TRACK command statistics",
                "Creator": Path(__file__).name,
            },
        )
        if not args.no_png:
            figure.savefig(
                args.output_png,
                format="png",
                dpi=PNG_DPI,
                metadata={"Software": Path(__file__).name},
            )
    finally:
        plt.close(figure)
    print(f"PDF: {args.output_pdf}")
    if not args.no_png:
        print(f"PNG: {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
