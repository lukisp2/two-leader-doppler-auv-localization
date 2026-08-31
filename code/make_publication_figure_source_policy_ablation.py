#!/usr/bin/env python3
"""Generate the publication figure for the Doppler-source/policy ablation.

The figure is built only from the frozen campaign summary and its per-episode
CSV.  It intentionally rejects smoke artifacts unless ``--allow-smoke`` is
passed, and marks smoke output so that it cannot be mistaken for evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data/tables"
DEFAULT_SUMMARY = TABLES / "leader_source_policy_summary.json"
DEFAULT_ROWS = TABLES / "leader_source_policy_episode_rows.csv"
DEFAULT_OUTPUT = ROOT / "results/figures/fig_source_policy_ablation.pdf"
FINAL_SEEDS = tuple(range(48600, 48700))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 38_200_100

SOURCE_ORDER = ("leader1_only", "leader2_only", "both_leaders")
POLICY_ORDER = ("fixed_s_turn", "belief_active")
SOURCE_LABELS = {
    "leader1_only": "Leader 1\nlink",
    "leader2_only": "Leader 2\nlink",
    "both_leaders": "Both\nlinks",
}
POLICY_LABELS = {
    "fixed_s_turn": "Fixed S-turn",
    "belief_active": "Information-guided",
}

FIXED_COLOR = "#D55E00"
ACTIVE_COLOR = "#0072B2"
TAIL_COLOR = "#F0E442"
GRID_COLOR = "#777777"

PUBLICATION_CONTRAST_SPECS = (
    (
        "active_vs_fixed@leader1_only",
        "leader1_only__fixed_s_turn",
        "leader1_only__belief_active",
        "terminal",
        BOOTSTRAP_SEED,
    ),
    (
        "active_vs_fixed@leader2_only",
        "leader2_only__fixed_s_turn",
        "leader2_only__belief_active",
        "terminal",
        BOOTSTRAP_SEED + 10,
    ),
    (
        "active_vs_fixed@both_leaders",
        "both_leaders__fixed_s_turn",
        "both_leaders__belief_active",
        "terminal",
        BOOTSTRAP_SEED + 20,
    ),
    (
        "both_vs_leader1_only@fixed_s_turn",
        "leader1_only__fixed_s_turn",
        "both_leaders__fixed_s_turn",
        "checkpoint_60s",
        BOOTSTRAP_SEED + 30,
    ),
    (
        "both_vs_leader2_only@fixed_s_turn",
        "leader2_only__fixed_s_turn",
        "both_leaders__fixed_s_turn",
        "checkpoint_60s",
        BOOTSTRAP_SEED + 40,
    ),
    (
        "both_vs_leader1_only@belief_active",
        "leader1_only__belief_active",
        "both_leaders__belief_active",
        "terminal",
        BOOTSTRAP_SEED + 50,
    ),
    (
        "both_vs_leader2_only@belief_active",
        "leader2_only__belief_active",
        "both_leaders__belief_active",
        "terminal",
        BOOTSTRAP_SEED + 60,
    ),
)


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
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Allow smoke data for layout testing; the output is visibly marked.",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def arm_name(source: str, policy: str) -> str:
    return f"{source}__{policy}"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample size")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def percentile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("percentiles require finite, non-empty data")
    return float(np.percentile(array, quantile))


def _distribution(values: Iterable[float]) -> dict[str, float] | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _require_close(
    label: str,
    saved: Any,
    recomputed: float,
    *,
    rel_tol: float = 1.0e-12,
    abs_tol: float = 1.0e-12,
) -> None:
    if not math.isclose(
        float(saved),
        float(recomputed),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        raise RuntimeError(
            f"{label} differs between campaign_summary.json and raw CSV"
        )


def _validate_distribution(
    label: str,
    saved: Any,
    recomputed: dict[str, float] | None,
) -> None:
    if saved is None or recomputed is None:
        if saved is not None or recomputed is not None:
            raise RuntimeError(
                f"{label} availability differs between summary and raw CSV"
            )
        return
    if not isinstance(saved, dict):
        raise RuntimeError(f"{label} is not a distribution object")
    for key in ("mean", "median", "p90", "p95", "max"):
        if key not in saved:
            raise RuntimeError(f"{label} is missing {key}")
        _require_close(f"{label}/{key}", saved[key], recomputed[key])


def validate_ablation_table_fields(
    summary: dict[str, Any],
    grouped: dict[str, list[dict[str, str]]],
) -> None:
    """Cross-check every summary-only field displayed by the ablation table."""

    for arm, arm_rows in grouped.items():
        lock_times: list[float] = []
        for row in arm_rows:
            ever_locked = parse_bool(row["ever_locked"])
            raw_time = row.get("first_track_action_time_s", "").strip()
            if ever_locked and not raw_time:
                raise RuntimeError(
                    f"{arm}: entered TRACK but first TRACK time is missing"
                )
            if not ever_locked and raw_time:
                raise RuntimeError(
                    f"{arm}: first TRACK time is present for a never-TRACK episode"
                )
            if raw_time:
                value = float(raw_time)
                if not math.isfinite(value) or value < 0.0:
                    raise RuntimeError(f"{arm}: invalid first TRACK time")
                lock_times.append(value)

        cell = summary["by_arm"][arm]
        _validate_distribution(
            f"{arm}/first_track_action_time_s",
            cell.get("first_track_action_time_s"),
            _distribution(lock_times),
        )
        for key in (
            "unsafe_transition_count",
            "unsafe_track_start_count",
            "unsafe_track_end_count",
        ):
            recomputed = sum(int(row[key]) for row in arm_rows)
            if int(cell[key]) != recomputed:
                raise RuntimeError(
                    f"{arm}: {key} differs between summary and raw CSV"
                )


def _discordance_p(left_only: int, right_only: int) -> float:
    count = int(left_only) + int(right_only)
    if count == 0:
        return 1.0
    lower = min(int(left_only), int(right_only))
    probability = sum(math.comb(count, k) for k in range(lower + 1)) / (
        2.0**count
    )
    return float(min(1.0, 2.0 * probability))


def _bootstrap_mean(values: Iterable[float], seed: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("paired bootstrap requires finite, non-empty data")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    output = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    chunk = 1000
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        end = min(start + chunk, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0,
            array.size,
            size=(end - start, array.size),
        )
        output[start:end] = np.mean(array[indices], axis=1)
    return {
        "pairs": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "bootstrap_mean_95": [
            float(np.percentile(output, 2.5)),
            float(np.percentile(output, 97.5)),
        ],
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": int(seed),
    }


def _validate_bootstrap(
    label: str,
    saved: Any,
    recomputed: dict[str, Any],
) -> None:
    if not isinstance(saved, dict):
        raise RuntimeError(f"{label} is not a bootstrap object")
    for key in ("pairs", "replicates", "seed"):
        if int(saved[key]) != int(recomputed[key]):
            raise RuntimeError(
                f"{label}/{key} differs between summary and raw CSV"
            )
    for key in ("mean", "median"):
        _require_close(f"{label}/{key}", saved[key], recomputed[key])
    saved_interval = saved.get("bootstrap_mean_95")
    if not isinstance(saved_interval, list) or len(saved_interval) != 2:
        raise RuntimeError(f"{label}/bootstrap_mean_95 is malformed")
    for index, value in enumerate(recomputed["bootstrap_mean_95"]):
        _require_close(
            f"{label}/bootstrap_mean_95[{index}]",
            saved_interval[index],
            value,
        )


def validate_publication_paired_contrasts(
    summary: dict[str, Any],
    grouped: dict[str, list[dict[str, str]]],
) -> None:
    """Recompute every contrast field displayed in the publication table."""

    contrasts = summary.get("paired_contrasts")
    if not isinstance(contrasts, dict):
        raise RuntimeError("campaign summary has no paired_contrasts object")
    expected_names = {spec[0] for spec in PUBLICATION_CONTRAST_SPECS}
    if set(contrasts) != expected_names:
        raise RuntimeError("campaign summary contains an unexpected contrast set")

    indexed: dict[str, dict[int, dict[str, str]]] = {}
    for arm, arm_rows in grouped.items():
        by_seed = {int(row["episode_seed"]): row for row in arm_rows}
        if len(by_seed) != len(arm_rows):
            raise RuntimeError(f"{arm}: duplicate seeds in paired contrast data")
        indexed[arm] = by_seed

    for name, left_arm, right_arm, endpoint, bootstrap_seed in (
        PUBLICATION_CONTRAST_SPECS
    ):
        saved = contrasts[name]
        if saved.get("left") != left_arm or saved.get("right") != right_arm:
            raise RuntimeError(f"{name}: paired contrast orientation differs")
        left = indexed[left_arm]
        right = indexed[right_arm]
        if set(left) != set(right):
            raise RuntimeError(f"{name}: paired contrast seed sets differ")
        seeds = sorted(left)
        if int(saved.get("pairs", -1)) != len(seeds):
            raise RuntimeError(f"{name}: paired contrast count differs")

        if endpoint == "terminal":
            success_column = "terminal_joint_success"
            difference_key = (
                "terminal_success_rate_difference_right_minus_left"
            )
            discordance_key = "terminal_discordance"
            error_column = "terminal_localization_error_m"
            improvement_key = "terminal_localization_improvement_m"
            improvement_seed = bootstrap_seed
        elif endpoint == "checkpoint_60s":
            success_column = "checkpoint_60s_localization_below_7m"
            difference_key = (
                "checkpoint_60s_localization_success_rate_difference_"
                "right_minus_left"
            )
            discordance_key = "checkpoint_60s_localization_discordance"
            error_column = "checkpoint_60s_localization_error_m"
            improvement_key = "checkpoint_60s_localization_improvement_m"
            improvement_seed = bootstrap_seed + 2
        else:
            raise RuntimeError(f"{name}: unsupported publication endpoint")

        left_success = [
            parse_bool(left[seed][success_column]) for seed in seeds
        ]
        right_success = [
            parse_bool(right[seed][success_column]) for seed in seeds
        ]
        left_only = sum(
            l_value and not r_value
            for l_value, r_value in zip(left_success, right_success)
        )
        right_only = sum(
            r_value and not l_value
            for l_value, r_value in zip(left_success, right_success)
        )
        success_difference = (
            sum(right_success) - sum(left_success)
        ) / len(seeds)
        _require_close(
            f"{name}/{difference_key}",
            saved[difference_key],
            success_difference,
        )

        saved_discordance = saved.get(discordance_key)
        if not isinstance(saved_discordance, dict):
            raise RuntimeError(f"{name}/{discordance_key} is malformed")
        if int(saved_discordance["left_only"]) != left_only:
            raise RuntimeError(f"{name}/{discordance_key}/left_only differs")
        if int(saved_discordance["right_only"]) != right_only:
            raise RuntimeError(f"{name}/{discordance_key}/right_only differs")
        _require_close(
            f"{name}/{discordance_key}/exact_mcnemar_p_two_sided",
            saved_discordance["exact_mcnemar_p_two_sided"],
            _discordance_p(left_only, right_only),
            rel_tol=1.0e-12,
            abs_tol=1.0e-30,
        )

        improvements = [
            float(left[seed][error_column])
            - float(right[seed][error_column])
            for seed in seeds
        ]
        _validate_bootstrap(
            f"{name}/{improvement_key}",
            saved.get(improvement_key),
            _bootstrap_mean(improvements, improvement_seed),
        )


def validate_and_group(
    summary: dict[str, Any],
    rows: list[dict[str, str]],
    allow_smoke: bool,
) -> tuple[dict[str, list[dict[str, str]]], bool]:
    is_smoke = bool(summary.get("smoke", False))
    if is_smoke and not allow_smoke:
        raise RuntimeError(
            "Refusing to plot smoke evidence. Re-run with --allow-smoke only for layout testing."
        )
    if summary.get("status") != "complete":
        raise RuntimeError(f"campaign is not complete: {summary.get('status')!r}")
    if not bool(summary.get("integrity_valid", False)):
        raise RuntimeError("campaign_summary.json reports failed integrity checks")
    if int(summary.get("row_count", -1)) != len(rows):
        raise RuntimeError("CSV row count disagrees with campaign_summary.json")

    expected_arms = {
        arm_name(source, policy)
        for source in SOURCE_ORDER
        for policy in POLICY_ORDER
    }
    by_arm_summary = summary.get("by_arm", {})
    if set(by_arm_summary) != expected_arms:
        raise RuntimeError(
            "unexpected arm set: "
            f"expected {sorted(expected_arms)}, found {sorted(by_arm_summary)}"
        )

    grouped: dict[str, list[dict[str, str]]] = {arm: [] for arm in expected_arms}
    for row in rows:
        arm = row.get("arm", "")
        if arm not in grouped:
            raise RuntimeError(f"CSV contains unexpected arm {arm!r}")
        grouped[arm].append(row)

    episode_counts = {len(arm_rows) for arm_rows in grouped.values()}
    if len(episode_counts) != 1:
        raise RuntimeError(f"arms are not balanced: {sorted(episode_counts)}")
    if not episode_counts or next(iter(episode_counts)) <= 0:
        raise RuntimeError("no episode rows were found")
    if not is_smoke:
        if episode_counts != {len(FINAL_SEEDS)}:
            raise RuntimeError(
                "publication output requires exactly 100 episodes per arm"
            )
        for arm, arm_rows in grouped.items():
            seeds = tuple(sorted(int(row["episode_seed"]) for row in arm_rows))
            if seeds != FINAL_SEEDS:
                raise RuntimeError(
                    f"{arm}: publication output requires seeds 48600--48699"
                )

    for arm, arm_rows in grouped.items():
        cell = by_arm_summary[arm]
        checks = {
            "episodes": len(arm_rows),
            "terminal_success_count": sum(
                parse_bool(row["terminal_joint_success"]) for row in arm_rows
            ),
            "tail80_success_count": sum(
                parse_bool(row["tail80_joint_success"]) for row in arm_rows
            ),
            "checkpoint_60s_localization_success_count": sum(
                parse_bool(row["checkpoint_60s_localization_below_7m"])
                for row in arm_rows
            ),
            "ever_lock_count": sum(parse_bool(row["ever_locked"]) for row in arm_rows),
        }
        for key, observed in checks.items():
            if int(cell[key]) != int(observed):
                raise RuntimeError(
                    f"{arm}: {key} differs between summary ({cell[key]}) and CSV ({observed})"
                )

        for csv_key, summary_key, label in (
            (
                "checkpoint_60s_localization_error_m",
                "checkpoint_60s_localization_error_m",
                "60-s localization",
            ),
            (
                "terminal_localization_error_m",
                "terminal_localization_error_m",
                "terminal localization",
            ),
            (
                "terminal_formation_error_m",
                "terminal_formation_error_m",
                "terminal formation",
            ),
        ):
            errors = [float(row[csv_key]) for row in arm_rows]
            expected = cell[summary_key]
            recomputed = {
                "median": percentile(errors, 50.0),
                "p95": percentile(errors, 95.0),
                "max": max(errors),
            }
            for key, observed in recomputed.items():
                if not math.isclose(
                    float(expected[key]), observed, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise RuntimeError(
                        f"{arm}: {label} {key} differs between summary and CSV"
                    )

    return grouped, is_smoke


def add_percentage_label(axis: mpl.axes.Axes, x: float, value: float) -> None:
    text = f"{value:.0f}"
    if value >= 22.0:
        axis.text(
            x,
            min(value - 8.0, 88.0),
            text,
            ha="center",
            va="top",
            color="white",
            fontsize=6.6,
            fontweight="bold",
        )
    else:
        axis.text(
            x,
            value + 2.0,
            text,
            ha="center",
            va="bottom",
            color="#222222",
            fontsize=6.6,
        )


def make_figure(
    summary: dict[str, Any],
    grouped: dict[str, list[dict[str, str]]],
    is_smoke: bool,
    output: Path,
) -> None:
    colors = {
        "fixed_s_turn": FIXED_COLOR,
        "belief_active": ACTIVE_COLOR,
    }
    source_x = np.arange(len(SOURCE_ORDER), dtype=float)
    bar_width = 0.32
    offsets = {
        "fixed_s_turn": -bar_width / 1.75,
        "belief_active": bar_width / 1.75,
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.10, 4.55),
        constrained_layout=True,
    )

    # Panel (a): the three fixed-policy arms have identical commands and
    # physical histories through 60 s.  This is therefore the cleanest
    # measurement-source comparison.
    checkpoint_rates: list[float] = []
    checkpoint_low: list[float] = []
    checkpoint_high: list[float] = []
    for source in SOURCE_ORDER:
        rows = grouped[arm_name(source, "fixed_s_turn")]
        total = len(rows)
        count = sum(
            parse_bool(row["checkpoint_60s_localization_below_7m"])
            for row in rows
        )
        rate = 100.0 * count / total
        lower, upper = wilson_interval(count, total)
        checkpoint_rates.append(rate)
        checkpoint_low.append(max(0.0, rate - 100.0 * lower))
        checkpoint_high.append(max(0.0, 100.0 * upper - rate))
    axes[0, 0].bar(
        source_x,
        checkpoint_rates,
        width=0.52,
        color=FIXED_COLOR,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    axes[0, 0].errorbar(
        source_x,
        checkpoint_rates,
        yerr=np.asarray([checkpoint_low, checkpoint_high]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        zorder=4,
    )
    for position, value in zip(source_x, checkpoint_rates):
        add_percentage_label(axes[0, 0], float(position), value)
    axes[0, 0].set(
        ylabel="Localization success [%]",
        ylim=(0.0, 122.0),
        xticks=source_x,
        xticklabels=[SOURCE_LABELS[source] for source in SOURCE_ORDER],
    )
    axes[0, 0].set_title("(a) Common trajectory at 60 s")
    axes[0, 0].grid(
        True,
        axis="y",
        alpha=0.22,
        color=GRID_COLOR,
        zorder=0,
    )
    axes[0, 0].text(
        0.02,
        0.96,
        r"Fixed S-turn; $e_p<7$ m",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#444444",
    )

    # Panel (b): terminal success is the primary system endpoint.  Tail80 is
    # overlaid as a distinct marker to expose late-horizon persistence.
    for policy in POLICY_ORDER:
        positions = source_x + offsets[policy]
        terminal_rates: list[float] = []
        tail_rates: list[float] = []
        error_low: list[float] = []
        error_high: list[float] = []
        tail_error_low: list[float] = []
        tail_error_high: list[float] = []
        for source in SOURCE_ORDER:
            rows = grouped[arm_name(source, policy)]
            total = len(rows)
            terminal_count = sum(
                parse_bool(row["terminal_joint_success"]) for row in rows
            )
            tail_count = sum(parse_bool(row["tail80_joint_success"]) for row in rows)
            rate = 100.0 * terminal_count / total
            lower, upper = wilson_interval(terminal_count, total)
            tail_rate = 100.0 * tail_count / total
            tail_lower, tail_upper = wilson_interval(tail_count, total)
            terminal_rates.append(rate)
            tail_rates.append(tail_rate)
            error_low.append(max(0.0, rate - 100.0 * lower))
            error_high.append(max(0.0, 100.0 * upper - rate))
            tail_error_low.append(max(0.0, tail_rate - 100.0 * tail_lower))
            tail_error_high.append(max(0.0, 100.0 * tail_upper - tail_rate))

        axes[0, 1].bar(
            positions,
            terminal_rates,
            width=bar_width,
            color=colors[policy],
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        axes[0, 1].errorbar(
            positions,
            terminal_rates,
            yerr=np.asarray([error_low, error_high]),
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            zorder=4,
        )
        axes[0, 1].errorbar(
            positions,
            tail_rates,
            yerr=np.asarray([tail_error_low, tail_error_high]),
            fmt="D",
            markersize=4.8,
            markerfacecolor=TAIL_COLOR,
            markeredgecolor="#222222",
            markeredgewidth=0.65,
            ecolor="#555555",
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            zorder=5,
        )
        for position, value, upper_error, tail_rate, tail_upper in zip(
            positions,
            terminal_rates,
            error_high,
            tail_rates,
            tail_error_high,
        ):
            if policy == "fixed_s_turn":
                label_y = max(
                    value + upper_error,
                    tail_rate + tail_upper,
                ) + 1.8
                axes[0, 1].text(
                    float(position),
                    label_y,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    color="#222222",
                    fontsize=6.6,
                )
            else:
                add_percentage_label(axes[0, 1], float(position), value)

    axes[0, 1].set(
        ylabel="Joint success [%]",
        ylim=(0.0, 132.0),
        xticks=source_x,
        xticklabels=[SOURCE_LABELS[source] for source in SOURCE_ORDER],
    )
    axes[0, 1].set_title("(b) Nominal task outcomes")
    axes[0, 1].grid(True, axis="y", alpha=0.22, color=GRID_COLOR, zorder=0)
    axes[0, 1].legend(
        handles=[
            Patch(facecolor=FIXED_COLOR, label=POLICY_LABELS["fixed_s_turn"]),
            Patch(facecolor=ACTIVE_COLOR, label=POLICY_LABELS["belief_active"]),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=TAIL_COLOR,
                markeredgecolor="#222222",
                markersize=5.2,
                label="Tail80 success",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                marker="|",
                markersize=7,
                linewidth=0.8,
                label="Wilson 95% CI",
            ),
        ],
        loc="upper center",
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        handlelength=1.25,
        bbox_to_anchor=(0.5, 0.98),
    )

    def draw_error_panel(
        axis: mpl.axes.Axes,
        csv_key: str,
        title: str,
        ylabel: str,
        limit: float,
    ) -> None:
        all_errors: list[float] = []
        for policy in POLICY_ORDER:
            positions = source_x + offsets[policy]
            for position, source in zip(positions, SOURCE_ORDER):
                rows = grouped[arm_name(source, policy)]
                errors = np.asarray(
                    [float(row[csv_key]) for row in rows],
                    dtype=float,
                )
                if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
                    raise RuntimeError(
                        f"{csv_key} values must be finite and strictly positive "
                        "for the logarithmic publication panel"
                    )
                median = float(np.percentile(errors, 50.0))
                p95 = float(np.percentile(errors, 95.0))
                maximum = float(np.max(errors))
                all_errors.extend(errors.tolist())
                axis.vlines(
                    position,
                    median,
                    maximum,
                    color=colors[policy],
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=2,
                )
                axis.vlines(
                    position,
                    median,
                    p95,
                    color=colors[policy],
                    linewidth=3.6,
                    alpha=0.95,
                    zorder=3,
                )
                axis.scatter(
                    position,
                    median,
                    color=colors[policy],
                    edgecolor="white",
                    linewidth=0.45,
                    s=28,
                    zorder=4,
                )
                axis.scatter(
                    position,
                    p95,
                    marker="_",
                    color="#222222",
                    linewidth=1.35,
                    s=58,
                    zorder=5,
                )
                axis.scatter(
                    position,
                    maximum,
                    marker="^",
                    facecolor="white",
                    edgecolor=colors[policy],
                    linewidth=0.9,
                    s=24,
                    zorder=5,
                )
        lower = 10.0 ** math.floor(math.log10(min(all_errors) * 0.75))
        upper = 10.0 ** math.ceil(math.log10(max(all_errors) * 1.25))
        axis.axhline(
            limit,
            color="#555555",
            linestyle="--",
            linewidth=0.85,
            zorder=1,
        )
        axis.set(
            ylabel=ylabel,
            yscale="log",
            ylim=(lower, upper),
            xticks=source_x,
            xticklabels=[SOURCE_LABELS[source] for source in SOURCE_ORDER],
        )
        axis.set_title(title)
        axis.grid(
            True,
            which="both",
            axis="y",
            alpha=0.20,
            color=GRID_COLOR,
            zorder=0,
        )
        axis.text(
            0.98,
            limit * 1.08,
            f"{limit:g}-m limit",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=6.4,
            color="#444444",
        )

    draw_error_panel(
        axes[1, 0],
        "terminal_localization_error_m",
        "(c) Terminal localization-error tails",
        r"Localization error $e_p$ [m]",
        7.0,
    )
    draw_error_panel(
        axes[1, 1],
        "terminal_formation_error_m",
        "(d) Terminal formation-error tails",
        r"Formation error $e_f$ [m]",
        8.0,
    )

    if is_smoke:
        fig.text(
            0.5,
            0.5,
            "SMOKE TEST - NOT FOR REPORTING",
            ha="center",
            va="center",
            fontsize=18.0,
            color="#8A1C1C",
            fontweight="bold",
            rotation=24,
            alpha=0.12,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    if args.campaign_dir is None:
        summary_path = args.summary.expanduser().resolve()
        csv_path = args.rows.expanduser().resolve()
    else:
        campaign_dir = args.campaign_dir.expanduser().resolve()
        summary_path = campaign_dir / "campaign_summary.json"
        csv_path = campaign_dir / "episode_arm_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    summary = load_json(summary_path)
    rows = load_csv(csv_path)
    grouped, is_smoke = validate_and_group(summary, rows, args.allow_smoke)
    output = args.output.expanduser().resolve()
    make_figure(summary, grouped, is_smoke, output)
    print(output)


if __name__ == "__main__":
    main()
