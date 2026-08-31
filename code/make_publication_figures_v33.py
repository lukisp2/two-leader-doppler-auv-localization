#!/usr/bin/env python3
"""Generate compact IEEE Access figures from frozen aggregate artifacts.

The script reads only campaign summaries and the V22 episode CSV.  It never
opens simulator truth archives or reserved/final seed ranges.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data/tables"
V27 = TABLES / "estimator_benchmark_summary.json"
V22 = ROOT / "data/raw/experiments_v22_active_acquisition_dev100/episode_arm_summary.csv"
V28 = ROOT / "data/raw/experiments_v28_estimator_stress_dev100/campaign_summary.json"
V29 = ROOT / "data/raw/experiments_v29_nuisance_identifiability_dev100/campaign_summary.json"
V32 = ROOT / "data/raw/experiments_v32_bias_time_to_evidence_dev100/campaign_summary.json"
V33 = ROOT / "data/raw/experiments_v33_offline_nuisance_acquisition_dev100/campaign_summary.json"
V34 = V27
V35 = TABLES / "closed_loop_stress_summary.json"
V35_ROWS = TABLES / "closed_loop_stress_episode_rows.csv"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#8E5AA9"
GRAY = "#666666"
SKY = "#56B4E9"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def figure_v27(output: Path) -> None:
    document = load_json(V27)
    if "methods" in document:
        methods = document["methods"]

        def estimator_cell(arm: str, time: int) -> dict[str, Any]:
            return methods[arm]["checkpoints"][str(time)]

        def endpoint_p95(cell: dict[str, Any]) -> float:
            return float(cell["endpoint_position_error_m"]["p95"])

    else:
        summary = document["aggregate"]["by_arm_prefix"]
        mhe_summary = load_json(V34)["aggregate"]["by_checkpoint"]

        def estimator_cell(arm: str, time: int) -> dict[str, Any]:
            if arm == "mhe60_arrival_fej":
                return mhe_summary[str(time)]
            return summary[f"{arm}@{time}"]

        def endpoint_p95(cell: dict[str, Any]) -> float:
            return float(cell["endpoint_error_m"]["p95"])

    checkpoints = [30, 60, 120, 240, 440]
    arms = [
        ("global_full", "Full history + broad search", BLUE, "o"),
        ("local_nls6", "Six-start local NLS", GREEN, "s"),
        ("global_window60", "Last 60 s", ORANGE, "^"),
        ("pf_lw_16384", "Liu–West PF (16k)", PURPLE, "D"),
        ("ekf_static", "Static EKF", GRAY, "x"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), constrained_layout=True)
    for arm, label, color, marker in arms:
        cells = [estimator_cell(arm, time) for time in checkpoints]
        success = [100.0 * cell["success_le_7m_count"] / cell["count"] for cell in cells]
        p95 = [endpoint_p95(cell) for cell in cells]
        axes[0].plot(checkpoints, success, color=color, marker=marker, label=label)
        axes[1].plot(checkpoints, p95, color=color, marker=marker, label=label)
    mhe_cells = [estimator_cell("mhe60_arrival_fej", time) for time in checkpoints]
    mhe_success = [100.0 * cell["success_le_7m_count"] / cell["count"] for cell in mhe_cells]
    mhe_p95 = [endpoint_p95(cell) for cell in mhe_cells]
    axes[0].plot(
        checkpoints,
        mhe_success,
        color="#CC79A7",
        marker="P",
        linestyle="--",
        label="MHE-60 + arrival",
    )
    axes[1].plot(
        checkpoints,
        mhe_p95,
        color="#CC79A7",
        marker="P",
        linestyle="--",
        label="MHE-60 + arrival",
    )
    axes[0].axhline(95, color="#999999", linestyle="--", linewidth=0.8)
    axes[0].set(
        xlabel="Causal history [s]",
        ylabel=r"Success with $e_p\leq7$ m [%]",
        ylim=(-2, 104),
    )
    axes[0].set_xticks(checkpoints)
    axes[0].grid(True, alpha=0.22)
    axes[0].set_title("(a) Endpoint success")
    axes[1].axhline(7, color="#999999", linestyle="--", linewidth=0.8, label="7-m criterion")
    axes[1].set(
        xlabel="Causal history [s]",
        ylabel=r"Localization error $e_p$, p95 [m]",
        yscale="log",
    )
    axes[1].set_xticks(checkpoints)
    axes[1].grid(True, which="both", alpha=0.22)
    axes[1].set_title("(b) Error-tail evolution")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.20),
        columnspacing=1.25,
        handletextpad=0.55,
    )
    save(fig, output / "fig_estimator_history.pdf")


def _read_v22() -> list[dict[str, str]]:
    with V22.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _positive_count(value: str, *, field: str, row_number: int) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{V35_ROWS}:{row_number}: "
            f"invalid {field}={value!r}"
        ) from error
    if not np.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(
            f"{V35_ROWS}:{row_number}: "
            f"{field} must be a finite nonnegative integer, got {value!r}"
        )
    return numeric > 0.0


def _truth_invalid_track_scenarios(
    conditions: list[str],
    arm: str,
) -> list[int]:
    """Count the union of episodes invalid at TRACK start or TRACK end."""
    csv_path = V35_ROWS
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "episode_seed",
            "condition",
            "arm",
            "unsafe_track_start_count",
            "unsafe_track_end_count",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
        rows = list(reader)

    counts: list[int] = []
    for condition in conditions:
        selected = [
            (row_number, row)
            for row_number, row in enumerate(rows, start=2)
            if row["condition"] == condition and row["arm"] == arm
        ]
        if len(selected) != 100:
            raise ValueError(
                f"{csv_path}: expected 100 rows for {condition}@{arm}, "
                f"found {len(selected)}"
            )
        seeds = [row["episode_seed"] for _, row in selected]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{csv_path}: duplicate seeds for {condition}@{arm}")
        count = sum(
            _positive_count(
                row["unsafe_track_start_count"],
                field="unsafe_track_start_count",
                row_number=row_number,
            )
            or _positive_count(
                row["unsafe_track_end_count"],
                field="unsafe_track_end_count",
                row_number=row_number,
            )
            for row_number, row in selected
        )
        counts.append(count)
    return counts


def figure_v22(output: Path) -> None:
    rows = _read_v22()
    arm_specs = [
        ("s_turn_causal", "Fixed S-turn", ORANGE),
        ("belief_fim_causal", "Evidence seeking", BLUE),
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.35),
        gridspec_kw={"width_ratios": [1.7, 0.8, 0.8]},
        constrained_layout=True,
    )
    for arm, label, color in arm_specs:
        arm_rows = [row for row in rows if row["arm"] == arm]
        locked = sorted(
            float(row["gate_first_track_action_time_s"])
            for row in arm_rows
            if _bool(row["gate_ever_locked"]) and row["gate_first_track_action_time_s"]
        )
        x = [0.0] + locked + [440.0]
        y = [0.0] + [index / len(arm_rows) for index in range(1, len(locked) + 1)] + [
            len(locked) / len(arm_rows)
        ]
        axes[0].step(x, y, where="post", color=color, label=label)
    axes[0].set(
        xlabel="First evidence-qualified lock [s]",
        ylabel="Cumulative incidence",
        xlim=(0, 440),
        ylim=(0, 1.03),
    )
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_title("(a) Lock-time distribution")

    terminal = []
    effort = []
    labels = []
    colors = []
    for arm, label, color in arm_specs:
        arm_rows = [row for row in rows if row["arm"] == arm]
        labels.append(label.replace(" ", "\n"))
        colors.append(color)
        terminal.append(sum(_bool(row["terminal_joint_success"]) for row in arm_rows))
        effort.append(np.mean([float(row["mean_squared_action"]) for row in arm_rows]))
    axes[1].bar(labels, terminal, color=colors, width=0.65)
    axes[1].set(ylabel="Terminal joint success / 100", ylim=(0, 105))
    axes[1].set_title("(b) Task endpoint")
    axes[1].grid(True, axis="y", alpha=0.22)
    for index, value in enumerate(terminal):
        axes[1].text(index, value + 2, str(value), ha="center", va="bottom", fontsize=7)
    axes[2].bar(labels, effort, color=colors, width=0.65)
    axes[2].set(ylabel="Mean squared action", ylim=(0, 0.32))
    axes[2].set_title("(c) Acquisition effort")
    axes[2].grid(True, axis="y", alpha=0.22)
    for index, value in enumerate(effort):
        axes[2].text(index, value + 0.009, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    save(fig, output / "fig_active_acquisition.pdf")


def figure_v28_v33(output: Path) -> None:
    stress = load_json(V28)["aggregate"]["by_condition_arm_prefix"]
    ident = load_json(V29)["aggregate"]["by_model_prefix"]
    timeline = load_json(V32)["aggregate"]["by_condition_checkpoint"]
    v33 = load_json(V33)["aggregate"]["true_bias_headroom"]
    conditions = [
        ("nominal", "Nominal"),
        ("doppler_common_bias_p003", "Common bias"),
        ("doppler_differential_bias_003", "Differential bias"),
        ("doppler_scale_102", "Doppler scale"),
        ("colored_noise_rho09_sd003", "Colored noise"),
        ("dropout_iid10", "Dropout"),
        ("broadcast_delay_2s", "Broadcast delay"),
        ("broadcast_offset_2m", "Broadcast offset"),
        ("dead_reckoning_scale_101", "DR scale"),
        ("dead_reckoning_drift_001", "DR drift"),
    ]
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.15, 4.55), constrained_layout=True)
    axes = axes_grid.ravel()

    p95 = []
    coverage = []
    labels = []
    for condition, label in conditions:
        cell = stress[f"{condition}|global_full@440"]
        labels.append(label)
        p95.append(cell["endpoint_error_m"]["p95"])
        coverage.append(100.0 * cell["coverage_rate"])
    y = np.arange(len(labels))
    axes[0].barh(y, p95, color=SKY, height=0.68, label="p95 error")
    axes[0].axvline(7, color=ORANGE, linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("440-s error p95 [m]")
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=6.4)
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x", alpha=0.22)
    twin = axes[0].twiny()
    twin.plot(coverage, y, color=PURPLE, marker="o", linewidth=1.1, label="Coverage")
    twin.set_xlabel("Nominal-radius coverage [%]", color=PURPLE)
    twin.tick_params(axis="x", colors=PURPLE)
    twin.set_xlim(0, 108)
    axes[0].set_title("(a) Model mismatch")

    checkpoints = [30, 60, 120, 240, 440]
    models = [
        ("p3_b2", "Position + 2 bias", BLUE, "o"),
        ("p3_b2_zscale_drscale", "+ two scales", ORANGE, "s"),
        ("p3_b2_d2_zscale_drscale_drv3", "Rich 12-param.", GRAY, "^"),
    ]
    for model, label, color, marker in models:
        values = [
            ident[f"{model}@{time}"]["scaled_weakest_singular_value"]["p5"]
            for time in checkpoints
        ]
        axes[1].plot(checkpoints, values, color=color, marker=marker, label=label)
    axes[1].axhline(1.0, color="#999999", linestyle="--", linewidth=0.8)
    axes[1].set(
        xlabel="Causal history [s]",
        ylabel="p5 weakest scaled singular value",
        yscale="log",
    )
    axes[1].set_xticks(checkpoints)
    axes[1].grid(True, which="both", alpha=0.22)
    axes[1].legend(frameon=False, loc="lower right", fontsize=5.9)
    axes[1].set_title("(b) Nuisance identifiability")

    evidence_times = [120, 140, 160, 180, 200, 240, 300, 360, 440]
    timeline_specs = [
        ("doppler_common_bias_p003", "Common bias", BLUE, "o"),
        ("doppler_differential_bias_003", "Differential bias", GREEN, "s"),
        ("colored_noise_rho09_sd003", "Colored false activation", ORANGE, "^"),
    ]
    for condition, label, color, marker in timeline_specs:
        values = [
            timeline[f"{condition}@{time}"]["activation_count"] for time in evidence_times
        ]
        axes[2].plot(evidence_times, values, color=color, marker=marker, label=label)
    axes[2].axhline(90, color="#999999", linestyle="--", linewidth=0.8)
    axes[2].axvline(240, color=PURPLE, linestyle=":", linewidth=0.9)
    axes[2].axvline(300, color=GRAY, linestyle=":", linewidth=0.9)
    axes[2].set(
        xlabel="Causal decision time [s]",
        ylabel="Model activations / 100",
        ylim=(-3, 104),
    )
    axes[2].grid(True, alpha=0.22)
    axes[2].legend(frameon=False, loc="center right", fontsize=6.2)
    axes[2].set_title("(c) Time to model evidence")

    checks = [
        ("Score\nchange", 100.0 * v33["v22_vs_combined_disagreement_fraction"], 25.0),
        ("Separation\nregret", 100.0 * v33["median_v22_bias_gls_regret_fraction"], 10.0),
        ("Bias\ncapture", 100.0 * v33["median_combined_bias_gls_capture_fraction"], 90.0),
        ("Schur\ncapture", 100.0 * v33["median_combined_schur_reduction_capture_fraction"], 90.0),
    ]
    x = np.arange(len(checks))
    actual = [item[1] for item in checks]
    threshold = [item[2] for item in checks]
    colors = [BLUE if value >= limit else ORANGE for value, limit in zip(actual, threshold)]
    axes[3].bar(x - 0.18, actual, width=0.36, color=colors, label="Observed")
    axes[3].bar(
        x + 0.18,
        threshold,
        width=0.36,
        color="none",
        edgecolor=GRAY,
        hatch="////",
        linewidth=0.8,
        label="Prespecified threshold",
    )
    axes[3].set_xticks(x, [item[0] for item in checks])
    axes[3].tick_params(axis="x", labelsize=6.2)
    axes[3].set(ylabel="Observed metric [%]", ylim=(0, 108))
    axes[3].grid(True, axis="y", alpha=0.22)
    axes[3].legend(frameon=False, loc="upper left", fontsize=6.2)
    axes[3].set_title("(d) Offline action-score screen")
    for index, value in enumerate(actual):
        axes[3].text(index - 0.18, value + 2, f"{value:.1f}", ha="center", va="bottom", fontsize=6.2)
    save(fig, output / "fig_robustness_evidence.pdf")


def figure_closed_loop_stress(output: Path) -> None:
    summary = load_json(V35)
    cells = summary["by_condition_arm"]
    primary_arm = "v24_nominal_model"
    conditions = [
        ("nominal", "Nom"),
        ("doppler_common_bias_p003", "CB"),
        ("doppler_differential_bias_003", "DB"),
        ("doppler_scale_102", "DS"),
        ("colored_noise_rho09_sd003", "CN"),
        ("dropout_iid10", "DO"),
        ("broadcast_delay_2s", "Lag"),
        ("dead_reckoning_scale_101", "DR"),
    ]
    primary = [cells[f"{name}@{primary_arm}"] for name, _ in conditions]
    labels = [label for _, label in conditions]
    x = np.arange(len(conditions))

    fig, axes_grid = plt.subplots(2, 2, figsize=(7.15, 4.85), constrained_layout=True)
    axes = axes_grid.ravel()

    terminal = [cell["terminal_success_count"] for cell in primary]
    tail80 = [cell["tail80_success_count"] for cell in primary]
    width = 0.36
    axes[0].bar(x - width / 2, terminal, width=width, color=BLUE, label="Terminal")
    axes[0].bar(x + width / 2, tail80, width=width, color=GREEN, label="Tail80")
    axes[0].axhline(90, color=GRAY, linestyle="--", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].tick_params(axis="x", labelsize=6.8, rotation=0, pad=2.5)
    axes[0].set(ylabel="Successful scenarios / 100", ylim=(0, 125))
    axes[0].grid(True, axis="y", alpha=0.22)
    axes[0].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        fontsize=6.2,
        borderaxespad=0.0,
    )
    axes[0].set_title("(a) Integrated task endpoints")

    localization_p95 = [cell["terminal_localization_error_m"]["p95"] for cell in primary]
    formation_p95 = [cell["terminal_formation_error_m"]["p95"] for cell in primary]
    axes[1].plot(x, localization_p95, color=PURPLE, marker="o", label=r"$e_p$ p95")
    axes[1].plot(x, formation_p95, color=ORANGE, marker="s", label=r"$e_f$ p95")
    axes[1].axhline(7, color=GRAY, linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].tick_params(axis="x", labelsize=6.8, rotation=0, pad=2.5)
    axes[1].set(ylabel="Terminal p95 error [m]", yscale="log")
    axes[1].grid(True, which="both", alpha=0.22)
    axes[1].legend(frameon=False, loc="upper left", fontsize=6.2)
    axes[1].set_title("(b) Tail sensitivity to mismatch")

    unsafe_scenarios = _truth_invalid_track_scenarios(
        [name for name, _ in conditions],
        primary_arm,
    )
    ever_lock = [cell["ever_lock_count"] for cell in primary]
    unsafe_bars = axes[2].bar(x, unsafe_scenarios, color=ORANGE, width=0.62)
    axes[2].bar_label(
        unsafe_bars,
        labels=[str(value) if value else "" for value in unsafe_scenarios],
        padding=2,
        fontsize=6.2,
    )
    axes[2].set_xticks(x, labels)
    axes[2].tick_params(axis="x", labelsize=6.8, rotation=0, pad=2.5)
    axes[2].set(
        ylabel="Scenarios with any truth-invalid\nTRACK sample",
        ylim=(0, max(26, max(unsafe_scenarios) + 4)),
    )
    axes[2].grid(True, axis="y", alpha=0.22)
    twin = axes[2].twinx()
    twin.plot(x, ever_lock, color=BLUE, marker="o", linewidth=1.2)
    twin.set_ylabel("Ever-lock [scenarios/100]", color=BLUE)
    twin.tick_params(axis="y", colors=BLUE)
    twin.set_ylim(45, 103)
    axes[2].set_title("(c) Gate outcome under model mismatch")

    bias_names = [
        ("doppler_common_bias_p003", "CB"),
        ("doppler_differential_bias_003", "DB"),
    ]
    baseline_p95 = [
        cells[f"{name}@{primary_arm}"]["terminal_localization_error_m"]["p95"]
        for name, _ in bias_names
    ]
    switched_p95 = [
        cells[f"{name}@v35_bias_gate_300"]["terminal_localization_error_m"]["p95"]
        for name, _ in bias_names
    ]
    bx = np.arange(2)
    axes[3].bar(bx - width / 2, baseline_p95, width=width, color=GRAY, label="Nominal model")
    axes[3].bar(bx + width / 2, switched_p95, width=width, color=SKY, label="Profiled switch")
    axes[3].set_xticks(bx, [label for _, label in bias_names])
    axes[3].set(ylabel=r"Localization error $e_p$, p95 [m]", ylim=(0, 7.3))
    axes[3].grid(True, axis="y", alpha=0.22)
    axes[3].legend(
        frameon=False,
        loc="upper center",
        ncol=2,
        fontsize=6.2,
    )
    axes[3].axhline(3.0, color=PURPLE, linestyle=":", linewidth=0.8)
    axes[3].set_title("(d) Rejected late bias-model switch")
    save(fig, output / "fig_closed_loop_stress.pdf")


def authorize_closed_loop_stress(
    summary: dict[str, Any],
    *,
    allow_invalid_descriptive: bool,
) -> None:
    integrity_valid = summary.get("integrity_valid")
    if not isinstance(integrity_valid, bool):
        raise RuntimeError("closed-loop stress integrity_valid must be Boolean")
    integrity_checks = summary.get("integrity_checks")
    if not isinstance(integrity_checks, dict):
        raise RuntimeError("closed-loop stress integrity_checks must be an object")
    failed_checks = {
        str(name)
        for name, value in integrity_checks.items()
        if not isinstance(value, bool) or not value
    }
    if integrity_valid != (not failed_checks):
        raise RuntimeError(
            "closed-loop stress integrity_valid disagrees with integrity_checks"
        )
    if not integrity_valid and failed_checks != {"maximum_runtime_below_2s"}:
        raise RuntimeError(
            "the archived descriptive closed-loop-stress diagnostic is authorized only for the frozen V35 "
            "runtime-only integrity failure"
        )
    if not integrity_valid and not allow_invalid_descriptive:
        raise RuntimeError(
            "refusing integrity-invalid V35 evidence; pass "
            "--allow-invalid-descriptive to generate the labelled descriptive figure"
        )
    if not integrity_valid:
        print(
            "WARNING: generating archived descriptive closed-loop-stress diagnostic from integrity-invalid V35 "
            "data; the frozen runtime criterion failed.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/figures",
        help="Output directory (default: repository results/figures/).",
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=(
            "estimator-history",
            "closed-loop-stress",
            "active-acquisition",
            "robustness-evidence",
            "all",
        ),
        help=(
            "Figure to generate; repeat as needed. The default generates only "
            "the estimator-history figure used by the current manuscript."
        ),
    )
    parser.add_argument(
        "--estimator-summary",
        type=Path,
        default=TABLES / "estimator_benchmark_summary.json",
        help="Released compact estimator summary.",
    )
    parser.add_argument(
        "--stress-summary",
        type=Path,
        default=TABLES / "closed_loop_stress_summary.json",
        help="Released compact closed-loop stress summary.",
    )
    parser.add_argument(
        "--stress-rows",
        type=Path,
        default=TABLES / "closed_loop_stress_episode_rows.csv",
        help="Released row-level closed-loop stress CSV.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/raw",
        help=(
            "DOI-archive root for optional legacy figures not used by the "
            "current manuscript."
        ),
    )
    parser.add_argument(
        "--allow-invalid-descriptive",
        action="store_true",
        help=(
            "Acknowledge that the released V35 campaign is integrity-invalid "
            "and generate its archived diagnostic only as explicitly labelled "
            "descriptive output."
        ),
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    global V27, V22, V28, V29, V32, V33, V34, V35, V35_ROWS
    V27 = args.estimator_summary.expanduser().resolve()
    V34 = V27
    V35 = args.stress_summary.expanduser().resolve()
    V35_ROWS = args.stress_rows.expanduser().resolve()
    V22 = data_root / "experiments_v22_active_acquisition_dev100" / "episode_arm_summary.csv"
    V28 = data_root / "experiments_v28_estimator_stress_dev100" / "campaign_summary.json"
    V29 = data_root / "experiments_v29_nuisance_identifiability_dev100" / "campaign_summary.json"
    V32 = data_root / "experiments_v32_bias_time_to_evidence_dev100" / "campaign_summary.json"
    V33 = data_root / "experiments_v33_offline_nuisance_acquisition_dev100" / "campaign_summary.json"
    configure_style()
    requested = set(args.figure or ("estimator-history",))
    if "all" in requested:
        requested = {
            "estimator-history",
            "closed-loop-stress",
            "active-acquisition",
            "robustness-evidence",
        }
    if "closed-loop-stress" in requested:
        authorize_closed_loop_stress(
            load_json(V35),
            allow_invalid_descriptive=args.allow_invalid_descriptive,
        )
    if "estimator-history" in requested:
        figure_v27(args.output_dir)
    if "closed-loop-stress" in requested:
        figure_closed_loop_stress(args.output_dir)
    if "active-acquisition" in requested:
        figure_v22(args.output_dir)
    if "robustness-evidence" in requested:
        figure_v28_v33(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
