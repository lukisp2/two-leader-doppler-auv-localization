#!/usr/bin/env python3
"""Generate the full-campaign source/policy dynamics figure.

The publication path is deliberately restricted to a complete, integrity-valid
six-arm campaign over the full prespecified seed block 48600--48699.  The
``--allow-invalid-layout-test`` switch exists only to inspect layout with an
older campaign; it adds an unmistakable title and watermark.

All plotted time series are reconstructed from saved NPZ traces.  The campaign
summary and per-episode CSV are used only to verify completeness, integrity,
pairing, and agreement with the saved traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "data/raw/leader_source_ablation"
DEFAULT_OUTPUT = ROOT / "results/figures/fig_source_policy_dynamics.pdf"

FINAL_SEEDS = tuple(range(48_600, 48_700))
SOURCE_ORDER = ("leader1_only", "leader2_only", "both_leaders")
POLICY_ORDER = ("fixed_s_turn", "belief_active")
ARM_ORDER = tuple(
    f"{source}__{policy}"
    for policy in POLICY_ORDER
    for source in SOURCE_ORDER
)

SOURCE_LABEL = {
    "leader1_only": "Leader 1 link",
    "leader2_only": "Leader 2 link",
    "both_leaders": "Both links",
}
SOURCE_MARKER = {
    "leader1_only": "o",
    "leader2_only": "^",
    "both_leaders": "s",
}
SOURCE_MARKER_OFFSET = {
    "leader1_only": 3,
    "leader2_only": 9,
    "both_leaders": 15,
}
POLICY_LABEL = {
    "fixed_s_turn": "Fixed S-turn",
    "belief_active": "Information-guided",
}
POLICY_COLOR = {
    "fixed_s_turn": "#D55E00",
    "belief_active": "#0072B2",
}
POLICY_LINESTYLE = {
    "fixed_s_turn": (0, (4.0, 2.0)),
    "belief_active": "-",
}

LOCALIZATION_LIMIT_M = 7.0
FORMATION_LIMIT_M = 8.0
GRID_COLOR = "#777777"
THRESHOLD_COLOR = "#404040"
TEST_COLOR = "#8A1C1C"


@dataclass(frozen=True)
class ArmTrace:
    source: str
    policy: str
    seeds: tuple[int, ...]
    action_start_time_s: np.ndarray
    time_s: np.ndarray
    phase_track: np.ndarray
    localization_error_m: np.ndarray
    formation_error_m: np.ndarray

    @property
    def arm(self) -> str:
        return f"{self.source}__{self.policy}"

    @property
    def episode_count(self) -> int:
        return len(self.seeds)

    def first_track_times_s(self) -> np.ndarray:
        first = np.full(self.episode_count, np.nan, dtype=float)
        locked = self.phase_track > 0.5
        has_track = np.any(locked, axis=1)
        first_index = np.argmax(locked, axis=1)
        first[has_track] = self.action_start_time_s[first_index[has_track]]
        return first


@dataclass(frozen=True)
class CampaignData:
    campaign_dir: Path
    summary: Mapping[str, Any]
    seeds: tuple[int, ...]
    arms: Mapping[str, ArmTrace]
    layout_test: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=DEFAULT_CAMPAIGN,
        help="Campaign directory containing summary, CSV, and traces_npz/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF, PNG, SVG, or EPS path.",
    )
    parser.add_argument(
        "--allow-invalid-layout-test",
        action="store_true",
        help=(
            "Permit a complete but invalid/non-final campaign only for layout "
            "testing. The output receives a TEST title and watermark."
        ),
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
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean value {value!r}")


def _source_policy(arm: str) -> tuple[str, str]:
    parts = arm.split("__")
    if len(parts) != 2:
        raise ValueError(f"invalid arm name {arm!r}")
    source, policy = parts
    if source not in SOURCE_ORDER or policy not in POLICY_ORDER:
        raise ValueError(f"unexpected arm name {arm!r}")
    return source, policy


def _unique_trace(campaign_dir: Path, arm: str, seed: int) -> Path:
    directory = campaign_dir / "traces_npz" / arm
    matches = sorted(directory.glob(f"episode_*_seed_{seed}.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one trace for arm={arm}, seed={seed}; "
            f"found {len(matches)} in {directory}"
        )
    return matches[0]


def _validate_summary(
    summary: Mapping[str, Any],
    row_count: int,
    allow_invalid_layout_test: bool,
) -> bool:
    if summary.get("status") != "complete":
        raise RuntimeError(
            f"campaign is not complete: status={summary.get('status')!r}"
        )
    if int(summary.get("row_count", -1)) != row_count:
        raise RuntimeError(
            "campaign_summary.json row_count disagrees with episode_arm_summary.csv"
        )
    expected_arms = set(ARM_ORDER)
    observed_arms = set(summary.get("by_arm", {}))
    if observed_arms != expected_arms:
        raise RuntimeError(
            f"unexpected summary arms: {sorted(observed_arms)}; "
            f"expected {sorted(expected_arms)}"
        )

    integrity_valid = bool(summary.get("integrity_valid", False))
    if not integrity_valid and not allow_invalid_layout_test:
        raise RuntimeError(
            "campaign is integrity-invalid; use --allow-invalid-layout-test "
            "only to inspect a visibly watermarked layout"
        )
    return not integrity_valid


def _validate_rows(
    rows: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, str]]], tuple[int, ...]]:
    grouped: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARM_ORDER}
    for row in rows:
        arm = row.get("arm", "")
        if arm not in grouped:
            raise RuntimeError(f"CSV contains unexpected arm {arm!r}")
        grouped[arm].append(row)

    seed_sets: dict[str, tuple[int, ...]] = {}
    for arm in ARM_ORDER:
        arm_rows = grouped[arm]
        if not arm_rows:
            raise RuntimeError(f"CSV contains no rows for {arm}")
        seeds = tuple(sorted(int(row["episode_seed"]) for row in arm_rows))
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"CSV contains duplicate seeds for {arm}")
        seed_sets[arm] = seeds
        arm_rows.sort(key=lambda row: int(row["episode_seed"]))

    common = seed_sets[ARM_ORDER[0]]
    for arm in ARM_ORDER[1:]:
        if seed_sets[arm] != common:
            raise RuntimeError("six arms do not contain the same paired seed set")
    if len(common) != 100:
        raise RuntimeError(
            f"campaign must contain 100 paired seeds per arm, found {len(common)}"
        )
    return grouped, common


def _require_vector(
    archive: Mapping[str, np.ndarray],
    path: Path,
    key: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    if key not in archive:
        raise RuntimeError(f"{path}: missing trace array {key!r}")
    value = np.asarray(archive[key], dtype=float)
    if value.ndim != 1:
        raise RuntimeError(f"{path}: {key!r} is not one-dimensional")
    if length is not None and value.size != length:
        raise RuntimeError(
            f"{path}: {key!r} has {value.size} rows; expected {length}"
        )
    return value


def _load_arm_trace(
    campaign_dir: Path,
    arm: str,
    rows: list[dict[str, str]],
    reference_time_s: np.ndarray | None,
    reference_action_start_s: np.ndarray | None,
) -> ArmTrace:
    source, policy = _source_policy(arm)
    expected_mask = {
        "leader1_only": (1.0, 0.0),
        "leader2_only": (0.0, 1.0),
        "both_leaders": (1.0, 1.0),
    }[source]

    phase_rows: list[np.ndarray] = []
    localization_rows: list[np.ndarray] = []
    formation_rows: list[np.ndarray] = []
    seeds: list[int] = []
    arm_time_s: np.ndarray | None = None
    arm_action_start_s: np.ndarray | None = None

    for row in rows:
        seed = int(row["episode_seed"])
        path = _unique_trace(campaign_dir, arm, seed)
        with np.load(path, allow_pickle=False) as archive:
            time_s = _require_vector(archive, path, "time_s")
            action_start_s = _require_vector(
                archive, path, "action_start_time_s", length=time_s.size
            )
            phase = _require_vector(
                archive, path, "phase_track", length=time_s.size
            )
            localization = _require_vector(
                archive, path, "localization_error_m", length=time_s.size
            )
            formation = _require_vector(
                archive, path, "formation_error_truth_m", length=time_s.size
            )
            mask_l1 = _require_vector(
                archive, path, "source_mask_l1", length=time_s.size
            )
            mask_l2 = _require_vector(
                archive, path, "source_mask_l2", length=time_s.size
            )

        if (
            time_s.size < 2
            or not np.all(np.isfinite(time_s))
            or not np.all(np.diff(time_s) > 0.0)
        ):
            raise RuntimeError(f"{path}: invalid time_s")
        if (
            not np.all(np.isfinite(action_start_s))
            or not np.all(np.diff(action_start_s) > 0.0)
            or not np.all(action_start_s < time_s)
        ):
            raise RuntimeError(f"{path}: invalid action_start_time_s")
        if not np.all(np.isin(phase, (0.0, 1.0))):
            raise RuntimeError(f"{path}: phase_track is not binary")
        if not np.all(np.isfinite(formation)) or np.any(formation < 0.0):
            raise RuntimeError(f"{path}: invalid formation-error trace")
        finite_localization = localization[np.isfinite(localization)]
        if finite_localization.size == 0 or np.any(finite_localization < 0.0):
            raise RuntimeError(f"{path}: invalid localization-error trace")
        if not (
            np.all(mask_l1 == expected_mask[0])
            and np.all(mask_l2 == expected_mask[1])
        ):
            raise RuntimeError(f"{path}: source mask does not match {source}")

        if arm_time_s is None:
            arm_time_s = time_s.copy()
            arm_action_start_s = action_start_s.copy()
        elif not (
            np.array_equal(time_s, arm_time_s)
            and np.array_equal(action_start_s, arm_action_start_s)
        ):
            raise RuntimeError(f"{arm}: trace time grids differ across seeds")

        locked = phase > 0.5
        ever_locked = bool(np.any(locked))
        if ever_locked != _parse_bool(row["ever_locked"]):
            raise RuntimeError(f"{path}: trace ever-lock disagrees with CSV")
        csv_first = row["first_track_action_time_s"].strip()
        if ever_locked:
            first_track = float(action_start_s[int(np.argmax(locked))])
            if not csv_first or not math.isclose(
                float(csv_first), first_track, rel_tol=0.0, abs_tol=1.0e-8
            ):
                raise RuntimeError(f"{path}: first TRACK time disagrees with CSV")
        elif csv_first:
            raise RuntimeError(f"{path}: CSV reports TRACK but trace never tracks")

        if not math.isclose(
            float(localization[-1]),
            float(row["terminal_localization_error_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise RuntimeError(f"{path}: terminal localization error disagrees with CSV")
        if not math.isclose(
            float(formation[-1]),
            float(row["terminal_formation_error_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise RuntimeError(f"{path}: terminal formation error disagrees with CSV")

        seeds.append(seed)
        phase_rows.append(phase)
        localization_rows.append(localization)
        formation_rows.append(formation)

    assert arm_time_s is not None
    assert arm_action_start_s is not None
    if reference_time_s is not None and not np.array_equal(
        arm_time_s, reference_time_s
    ):
        raise RuntimeError(f"{arm}: decision-end time grid differs between arms")
    if reference_action_start_s is not None and not np.array_equal(
        arm_action_start_s, reference_action_start_s
    ):
        raise RuntimeError(f"{arm}: action-start time grid differs between arms")

    localization_matrix = np.stack(localization_rows, axis=0)
    # For an honest population percentile, an estimator sample must be present
    # for either every paired episode or none at a decision time.
    finite_counts = np.sum(np.isfinite(localization_matrix), axis=0)
    if np.any((finite_counts != 0) & (finite_counts != len(seeds))):
        raise RuntimeError(
            f"{arm}: localization availability differs across paired episodes"
        )
    available = finite_counts == len(seeds)
    if not np.any(available):
        raise RuntimeError(f"{arm}: no localization estimate was ever emitted")
    first_available = int(np.argmax(available))
    if not np.all(available[first_available:]):
        raise RuntimeError(
            f"{arm}: localization estimates disappear after becoming available"
        )

    return ArmTrace(
        source=source,
        policy=policy,
        seeds=tuple(seeds),
        action_start_time_s=arm_action_start_s,
        time_s=arm_time_s,
        phase_track=np.stack(phase_rows, axis=0),
        localization_error_m=localization_matrix,
        formation_error_m=np.stack(formation_rows, axis=0),
    )


def load_campaign(
    campaign_dir: Path,
    allow_invalid_layout_test: bool,
) -> CampaignData:
    campaign_dir = campaign_dir.expanduser().resolve()
    summary_path = campaign_dir / "campaign_summary.json"
    csv_path = campaign_dir / "episode_arm_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    summary = _load_json(summary_path)
    rows = _load_csv(csv_path)
    integrity_invalid = _validate_summary(
        summary, len(rows), allow_invalid_layout_test
    )
    grouped, seeds = _validate_rows(rows)

    nonfinal_seed_block = seeds != FINAL_SEEDS
    if nonfinal_seed_block and not allow_invalid_layout_test:
        raise RuntimeError(
            "publication output is restricted to the complete seed block "
            "48600--48699"
        )
    layout_test = bool(integrity_invalid or nonfinal_seed_block)

    arms: dict[str, ArmTrace] = {}
    reference_time_s: np.ndarray | None = None
    reference_action_start_s: np.ndarray | None = None
    for arm in ARM_ORDER:
        loaded = _load_arm_trace(
            campaign_dir,
            arm,
            grouped[arm],
            reference_time_s,
            reference_action_start_s,
        )
        if loaded.seeds != seeds:
            raise RuntimeError(f"{arm}: trace seeds do not match the CSV pairing")
        arms[arm] = loaded
        if reference_time_s is None:
            reference_time_s = loaded.time_s
            reference_action_start_s = loaded.action_start_time_s

    return CampaignData(
        campaign_dir=campaign_dir,
        summary=summary,
        seeds=seeds,
        arms=arms,
        layout_test=layout_test,
    )


def _arm(source: str, policy: str) -> str:
    return f"{source}__{policy}"


def _style_axis(axis: mpl.axes.Axes) -> None:
    axis.grid(
        True,
        which="major",
        color=GRID_COLOR,
        alpha=0.20,
        linewidth=0.55,
        zorder=0,
    )
    axis.tick_params(direction="out", length=2.8, width=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_arm_line(
    axis: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    source: str,
    policy: str,
    *,
    drawstyle: str = "default",
    linewidth: float | None = None,
) -> None:
    axis.plot(
        x,
        y,
        color=POLICY_COLOR[policy],
        linestyle=POLICY_LINESTYLE[policy],
        linewidth=linewidth
        if linewidth is not None
        else (1.35 if source == "both_leaders" else 1.0),
        alpha=1.0 if source == "both_leaders" else 0.82,
        marker=SOURCE_MARKER[source],
        markerfacecolor="white",
        markeredgecolor=POLICY_COLOR[policy],
        markeredgewidth=0.65,
        markersize=3.1,
        markevery=(SOURCE_MARKER_OFFSET[source], 24),
        drawstyle=drawstyle,
        zorder=3 if source == "both_leaders" else 2,
    )


def _time_grid(campaign: CampaignData) -> tuple[np.ndarray, np.ndarray]:
    reference = campaign.arms[ARM_ORDER[0]]
    outcome_time = reference.time_s
    cdf_time = np.unique(
        np.concatenate(
            (
                np.asarray([0.0]),
                reference.action_start_time_s,
                outcome_time,
            )
        )
    )
    return cdf_time, outcome_time


def _draw_time_to_track(
    axis: mpl.axes.Axes,
    campaign: CampaignData,
    cdf_time: np.ndarray,
) -> None:
    for policy in POLICY_ORDER:
        for source in SOURCE_ORDER:
            trace = campaign.arms[_arm(source, policy)]
            first = trace.first_track_times_s()
            finite = np.isfinite(first)
            cdf = 100.0 * np.mean(
                finite[:, np.newaxis]
                & (first[:, np.newaxis] <= cdf_time[np.newaxis, :]),
                axis=0,
            )
            _plot_arm_line(
                axis,
                cdf_time,
                cdf,
                source,
                policy,
                drawstyle="steps-post",
            )

    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("Episodes reaching TRACK [%]")
    axis.set_title("(a) Empirical time to first TRACK")
    _style_axis(axis)


def _draw_localization_success(
    axis: mpl.axes.Axes,
    campaign: CampaignData,
    outcome_time: np.ndarray,
) -> None:
    for policy in POLICY_ORDER:
        for source in SOURCE_ORDER:
            trace = campaign.arms[_arm(source, policy)]
            error = trace.localization_error_m
            success = np.isfinite(error) & (error < LOCALIZATION_LIMIT_M)
            rate = 100.0 * np.mean(success, axis=0)
            _plot_arm_line(axis, outcome_time, rate, source, policy)
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel(r"Episodes with $e_p<7$ m [%]")
    axis.set_title("(b) Localization success over time")
    axis.text(
        0.985,
        0.045,
        r"success: $e_p<7$ m",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#444444",
    )
    _style_axis(axis)


def _log_limits(values: Iterable[np.ndarray], threshold: float) -> tuple[float, float]:
    flattened = [
        np.asarray(value, dtype=float)[np.isfinite(value)]
        for value in values
    ]
    all_values = np.concatenate(
        [value for value in flattened if value.size]
        + [np.asarray([threshold], dtype=float)]
    )
    if np.any(all_values <= 0.0):
        raise RuntimeError("logarithmic panel requires strictly positive values")
    lower = 10.0 ** (
        math.floor(math.log10(float(np.min(all_values)) * 0.78) * 4.0)
        / 4.0
    )
    upper = 10.0 ** (
        math.ceil(math.log10(float(np.max(all_values)) * 1.25) * 4.0)
        / 4.0
    )
    return lower, upper


def _draw_localization_quantiles(
    axis: mpl.axes.Axes,
    campaign: CampaignData,
    outcome_time: np.ndarray,
) -> None:
    plotted: list[np.ndarray] = []
    for policy in POLICY_ORDER:
        trace = campaign.arms[_arm("both_leaders", policy)]
        availability = np.sum(np.isfinite(trace.localization_error_m), axis=0)
        valid = availability == trace.episode_count
        values = trace.localization_error_m[:, valid]
        time = outcome_time[valid]
        median = np.median(values, axis=0)
        p95 = np.percentile(values, 95.0, axis=0)
        plotted.extend((median, p95))
        axis.fill_between(
            time,
            median,
            p95,
            color=POLICY_COLOR[policy],
            alpha=0.12,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            time,
            p95,
            color=POLICY_COLOR[policy],
            linestyle=POLICY_LINESTYLE[policy],
            linewidth=0.75,
            alpha=0.78,
            zorder=2,
        )
        axis.plot(
            time,
            median,
            color=POLICY_COLOR[policy],
            linestyle=POLICY_LINESTYLE[policy],
            linewidth=1.45,
            zorder=3,
        )

    axis.axhline(
        LOCALIZATION_LIMIT_M,
        color=THRESHOLD_COLOR,
        linestyle=(0, (3.0, 2.0)),
        linewidth=0.8,
        zorder=1,
    )
    lower, upper = _log_limits(plotted, LOCALIZATION_LIMIT_M)
    axis.set_yscale("log")
    axis.set_ylim(lower, upper)
    axis.set_ylabel(r"Localization error $e_p$ [m]")
    axis.set_title("(c) Both links: median and p95 error")
    axis.text(
        0.985,
        LOCALIZATION_LIMIT_M * 1.06,
        "7-m localization limit",
        transform=axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=THRESHOLD_COLOR,
    )
    _style_axis(axis)


def _draw_joint_success(
    axis: mpl.axes.Axes,
    campaign: CampaignData,
    outcome_time: np.ndarray,
) -> None:
    for policy in POLICY_ORDER:
        for source in SOURCE_ORDER:
            trace = campaign.arms[_arm(source, policy)]
            localization_ok = np.isfinite(trace.localization_error_m) & (
                trace.localization_error_m < LOCALIZATION_LIMIT_M
            )
            formation_ok = trace.formation_error_m < FORMATION_LIMIT_M
            rate = 100.0 * np.mean(localization_ok & formation_ok, axis=0)
            _plot_arm_line(axis, outcome_time, rate, source, policy)
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("Jointly successful episodes [%]")
    axis.set_title("(d) Closed-loop joint success over time")
    axis.text(
        0.02,
        0.95,
        r"joint: $e_p<7$ m" "\n" r"and $e_f<8$ m",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color="#444444",
    )
    _style_axis(axis)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=POLICY_COLOR[policy],
            linestyle=POLICY_LINESTYLE[policy],
            linewidth=1.45,
            label=POLICY_LABEL[policy],
        )
        for policy in POLICY_ORDER
    ] + [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="none",
            marker=SOURCE_MARKER[source],
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=0.75,
            markersize=4.3,
            label=SOURCE_LABEL[source],
        )
        for source in SOURCE_ORDER
    ]


def caption_text(campaign: CampaignData) -> str:
    seed_text = f"{campaign.seeds[0]}--{campaign.seeds[-1]}"
    estimate_start_s = max(
        float(
            trace.time_s[
                int(
                    np.argmax(
                        np.all(np.isfinite(trace.localization_error_m), axis=0)
                    )
                )
            ]
        )
        for trace in campaign.arms.values()
    )
    prefix = (
        "LAYOUT TEST WITH AN INTEGRITY-INVALID OR NONFINAL CAMPAIGN. "
        if campaign.layout_test
        else ""
    )
    return (
        prefix
        + f"Full-campaign dynamics over the {len(campaign.seeds)} paired seeds "
        f"{seed_text}; no episode was selected for display. Line color and style "
        "encode the acquisition rule, while markers encode which Doppler links "
        "enter the localization, planning, and gate layers. Both physical "
        "leaders remain present in every arm. (a) Empirical cumulative incidence "
        "of the first ACQUIRE-to-TRACK action; curves ending below 100% correspond "
        "to episodes that never enter TRACK. "
        "(b) Fraction of episodes with offline truth-scored e_p below 7 m. "
        "(c) Median (thick) and p95 (thin boundary and ribbon) of e_p for the two "
        "both-link arms; the horizontal line is the 7-m limit. (d) Fraction "
        "simultaneously satisfying e_p below 7 m and e_f below 8 m. Missing "
        "pre-estimate localization "
        "values count as failures in (b) and (d), and are omitted from the error "
        f"quantiles in (c); all arms emit estimates from {estimate_start_s:g} s "
        "onward. Panels "
        "(b)--(d) use truth only for retrospective scoring, not online decisions."
    )


def make_figure(campaign: CampaignData, output: Path) -> None:
    cdf_time, outcome_time = _time_grid(campaign)
    fig, axes = plt.subplots(2, 2, figsize=(7.12, 4.82))
    fig.subplots_adjust(
        left=0.082,
        right=0.992,
        bottom=0.105,
        top=0.865 if campaign.layout_test else 0.895,
        wspace=0.285,
        hspace=0.42,
    )

    _draw_time_to_track(axes[0, 0], campaign, cdf_time)
    _draw_localization_success(axes[0, 1], campaign, outcome_time)
    _draw_localization_quantiles(axes[1, 0], campaign, outcome_time)
    _draw_joint_success(axes[1, 1], campaign, outcome_time)

    end_time = float(outcome_time[-1])
    tick_values = list(np.arange(0.0, end_time, 100.0))
    if not tick_values or abs(tick_values[-1] - end_time) > 20.0:
        tick_values.append(end_time)
    for axis in axes.flat:
        axis.set_xlim(0.0, end_time)
        axis.set_xticks(tick_values)
        axis.set_xlabel("Time [s]")

    legend_y = 0.935 if campaign.layout_test else 0.985
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=5,
        frameon=False,
        columnspacing=1.05,
        handlelength=2.1,
        handletextpad=0.48,
    )

    if campaign.layout_test:
        fig.suptitle(
            "INVALID/NONFINAL CAMPAIGN — LAYOUT TEST ONLY",
            x=0.5,
            y=0.995,
            ha="center",
            va="top",
            color=TEST_COLOR,
            fontsize=9.5,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.49,
            "LAYOUT TEST — NOT EVIDENCE",
            ha="center",
            va="center",
            rotation=23,
            color=TEST_COLOR,
            fontsize=22.0,
            fontweight="bold",
            alpha=0.09,
            zorder=20,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in {".pdf", ".png", ".svg", ".eps"}:
        raise ValueError("output suffix must be .pdf, .png, .svg, or .eps")
    kwargs: dict[str, Any] = {"format": suffix.lstrip(".")}
    if suffix == ".png":
        kwargs["dpi"] = 300
    fig.savefig(output, **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    campaign = load_campaign(
        args.campaign_dir,
        bool(args.allow_invalid_layout_test),
    )
    output = args.output.expanduser().resolve()
    make_figure(campaign, output)
    print(output)
    print("CAPTION:")
    print(caption_text(campaign))


if __name__ == "__main__":
    main()
