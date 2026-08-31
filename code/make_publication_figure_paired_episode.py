#!/usr/bin/env python3
"""Generate the publication paired-episode illustration.

The figure compares the fixed S-turn and information-guided acquisition arms with both
Doppler links admitted for one common episode seed. It reads only the saved
per-episode JSON and NPZ trace artifacts; it does not rerun or modify the
simulation.

Seed 48698 is the descriptive manuscript episode. It was selected after the
campaign by a deterministic robust-medoid rule within the modal paired outcome
stratum. The episode illustrates mechanism only; all quantitative claims use
the complete 100-seed paired campaign.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "data/raw/leader_source_ablation"
DEFAULT_OUTPUT = ROOT / "results/figures/fig_paired_episode_seed48698.pdf"
MANUSCRIPT_SEED = 48_698

FIXED_ARM = "both_leaders__fixed_s_turn"
ACTIVE_ARM = "both_leaders__belief_active"
ARM_ORDER = (FIXED_ARM, ACTIVE_ARM)

ARM_LABEL = {
    FIXED_ARM: "Fixed S-turn",
    ACTIVE_ARM: "Information-guided",
}
ARM_COLOR = {
    FIXED_ARM: "#D55E00",
    ACTIVE_ARM: "#0072B2",
}
ARM_LINESTYLE = {
    FIXED_ARM: (0, (4.0, 2.0)),
    ACTIVE_ARM: "-",
}

LEADER_COLOR = "#666666"
LEADER_LINESTYLES = ((0, (1.4, 1.4)), (0, (5.0, 1.8, 1.2, 1.8)))
THRESHOLD_COLOR = "#404040"
GRID_COLOR = "#777777"


@dataclass(frozen=True)
class EpisodeArm:
    arm: str
    result_path: Path
    trace_path: Path
    result: Mapping[str, Any]
    trace: Mapping[str, np.ndarray]

    @property
    def time_s(self) -> np.ndarray:
        return np.asarray(self.trace["time_s"], dtype=float)

    @property
    def first_track_time_s(self) -> float | None:
        gate = self.result.get("gate", {})
        value = gate.get("first_track_action_time_s")
        if value is None:
            transitions = gate.get("transitions", [])
            if transitions:
                value = transitions[0].get("time_s")
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=DEFAULT_CAMPAIGN,
        help="Campaign directory containing traces_npz/ and episode_results/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MANUSCRIPT_SEED,
        help="Common episode seed to illustrate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF, PNG, SVG, or EPS path.",
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
            "legend.fontsize": 7.0,
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


def _unique_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one artifact matching {directory / pattern}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_arm(campaign_dir: Path, seed: int, arm: str) -> EpisodeArm:
    trace_path = _unique_match(
        campaign_dir / "traces_npz" / arm,
        f"episode_*_seed_{seed}.npz",
    )
    result_path = _unique_match(
        campaign_dir / "episode_results",
        f"episode_*_seed_{seed}_{arm}.json",
    )
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: np.asarray(archive[key]) for key in archive.files}

    if int(result.get("episode_seed", -1)) != seed:
        raise RuntimeError(f"{result_path}: episode_seed does not match --seed")
    if result.get("arm") != arm:
        raise RuntimeError(f"{result_path}: arm does not match its filename")
    if list(result.get("source_mask", [])) != [True, True]:
        raise RuntimeError(f"{result_path}: both Doppler links are not admitted")
    return EpisodeArm(
        arm=arm,
        result_path=result_path,
        trace_path=trace_path,
        result=result,
        trace=trace,
    )


def _require_vector(
    episode: EpisodeArm,
    key: str,
    *,
    length: int | None = None,
    finite: bool = False,
) -> np.ndarray:
    if key not in episode.trace:
        raise RuntimeError(f"{episode.trace_path}: missing array {key!r}")
    value = np.asarray(episode.trace[key])
    if value.ndim != 1:
        raise RuntimeError(f"{episode.trace_path}: {key!r} is not one-dimensional")
    if length is not None and value.size != length:
        raise RuntimeError(
            f"{episode.trace_path}: {key!r} has {value.size} rows, expected {length}"
        )
    if finite and not np.all(np.isfinite(value)):
        raise RuntimeError(f"{episode.trace_path}: {key!r} contains non-finite values")
    return value


def _validate_pair(episodes: Mapping[str, EpisodeArm]) -> None:
    fixed = episodes[FIXED_ARM]
    active = episodes[ACTIVE_ARM]
    required_decision_vectors = (
        "phase_track",
        "gate_locked_after_update",
        "truth_x",
        "truth_y",
        "truth_z",
        "formation_error_truth_m",
        "localization_error_m",
        "batch_local_radius95_m",
        "source_mask_l1",
        "source_mask_l2",
    )
    for episode in episodes.values():
        time_s = _require_vector(episode, "time_s", finite=True)
        if time_s.size < 2 or not np.all(np.diff(time_s) > 0.0):
            raise RuntimeError(f"{episode.trace_path}: invalid decision time grid")
        for key in required_decision_vectors:
            _require_vector(episode, key, length=time_s.size)
        if not np.all(np.asarray(episode.trace["source_mask_l1"]) == 1.0):
            raise RuntimeError(f"{episode.trace_path}: Doppler link 1 is masked")
        if not np.all(np.asarray(episode.trace["source_mask_l2"]) == 1.0):
            raise RuntimeError(f"{episode.trace_path}: Doppler link 2 is masked")

        online_t_s = _require_vector(episode, "online_t_s", finite=True)
        leader_position = np.asarray(episode.trace.get("online_leader_position_m"))
        if leader_position.shape != (online_t_s.size, 2, 3):
            raise RuntimeError(
                f"{episode.trace_path}: invalid online_leader_position_m shape "
                f"{leader_position.shape}"
            )
        if not np.all(np.isfinite(leader_position)):
            raise RuntimeError(
                f"{episode.trace_path}: leader broadcasts contain non-finite values"
            )

    for key in ("time_s", "online_t_s", "online_leader_position_m"):
        left = np.asarray(fixed.trace[key], dtype=float)
        right = np.asarray(active.trace[key], dtype=float)
        if left.shape != right.shape or not np.allclose(
            left, right, rtol=0.0, atol=1.0e-10
        ):
            raise RuntimeError(
                f"paired arms do not share the same recorded {key!r}"
            )

    if fixed.result.get("noise_tape_sha256") != active.result.get("noise_tape_sha256"):
        raise RuntimeError("paired arms do not share the same noise tape")
    fixed_initial = np.asarray(fixed.result.get("initial_truth_m"), dtype=float)
    active_initial = np.asarray(active.result.get("initial_truth_m"), dtype=float)
    if fixed_initial.shape != (3,) or not np.allclose(
        fixed_initial, active_initial, rtol=0.0, atol=1.0e-10
    ):
        raise RuntimeError("paired arms do not share the same initial follower truth")


def load_pair(campaign_dir: Path, seed: int) -> dict[str, EpisodeArm]:
    if not campaign_dir.is_dir():
        raise FileNotFoundError(campaign_dir)
    episodes = {
        arm: _load_arm(campaign_dir, seed, arm)
        for arm in ARM_ORDER
    }
    _validate_pair(episodes)
    return episodes


def _decision_centroid(episode: EpisodeArm) -> np.ndarray:
    online_t_s = np.asarray(episode.trace["online_t_s"], dtype=float)
    leader_position = np.asarray(
        episode.trace["online_leader_position_m"], dtype=float
    )
    decision_t_s = episode.time_s
    indices = np.searchsorted(online_t_s, decision_t_s, side="left")
    indices = np.clip(indices, 0, online_t_s.size - 1)
    previous = np.maximum(indices - 1, 0)
    choose_previous = (
        np.abs(online_t_s[previous] - decision_t_s)
        < np.abs(online_t_s[indices] - decision_t_s)
    )
    indices = np.where(choose_previous, previous, indices)
    if np.max(np.abs(online_t_s[indices] - decision_t_s)) > 0.51:
        raise RuntimeError("could not align leader broadcasts to decision times")
    return np.mean(leader_position[indices], axis=1)


def _follower_relative_xy(episode: EpisodeArm) -> np.ndarray:
    truth = np.column_stack(
        [
            np.asarray(episode.trace["truth_x"], dtype=float),
            np.asarray(episode.trace["truth_y"], dtype=float),
            np.asarray(episode.trace["truth_z"], dtype=float),
        ]
    )
    if not np.all(np.isfinite(truth)):
        raise RuntimeError(f"{episode.trace_path}: follower truth is not finite")
    return (truth - _decision_centroid(episode))[:, :2]


def _leader_relative_xy(episode: EpisodeArm) -> np.ndarray:
    positions = np.asarray(
        episode.trace["online_leader_position_m"], dtype=float
    )
    centroid = np.mean(positions, axis=1, keepdims=True)
    return (positions - centroid)[:, :, :2]


def _desired_relative_xy(episode: EpisodeArm) -> np.ndarray:
    velocity = np.asarray(
        episode.trace["online_leader_velocity_mps"], dtype=float
    )
    if velocity.ndim != 3 or velocity.shape[1:] != (2, 3):
        raise RuntimeError(
            f"{episode.trace_path}: invalid online_leader_velocity_mps shape "
            f"{velocity.shape}"
        )
    centroid_velocity = np.mean(velocity, axis=(0, 1))
    forward = centroid_velocity[:2]
    forward_norm = float(np.linalg.norm(forward))
    if not math.isfinite(forward_norm) or forward_norm <= 0.0:
        raise RuntimeError("leader-centroid horizontal velocity is degenerate")
    return -120.0 * forward / forward_norm


def _finite_positive_series(
    episode: EpisodeArm,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = episode.time_s
    values = np.asarray(episode.trace[key], dtype=float)
    mask = np.isfinite(values)
    if not np.any(mask):
        raise RuntimeError(f"{episode.trace_path}: {key!r} has no finite samples")
    if np.any(values[mask] <= 0.0):
        raise RuntimeError(
            f"{episode.trace_path}: {key!r} must be positive for logarithmic plotting"
        )
    return time_s[mask], values[mask]


def _log_limits(
    series: list[np.ndarray],
    threshold: float,
    *,
    lower_padding: float = 0.72,
    upper_padding: float = 1.35,
) -> tuple[float, float]:
    values = np.concatenate(series + [np.asarray([threshold], dtype=float)])
    # Quarter-decade padding keeps the panels compact while preserving all data.
    lower = 10.0 ** (
        math.floor(math.log10(float(np.min(values)) * lower_padding) * 4.0)
        / 4.0
    )
    upper = 10.0 ** (
        math.ceil(math.log10(float(np.max(values)) * upper_padding) * 4.0)
        / 4.0
    )
    return lower, upper


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


def _draw_threshold(
    axis: mpl.axes.Axes,
    value: float,
    label: str,
) -> None:
    axis.axhline(
        value,
        color=THRESHOLD_COLOR,
        linestyle=(0, (3.0, 2.0)),
        linewidth=0.8,
        zorder=1,
    )


def _draw_first_track_lines(
    axis: mpl.axes.Axes,
    episodes: Mapping[str, EpisodeArm],
    *,
    label: bool,
) -> None:
    label_levels = {ACTIVE_ARM: 0.965, FIXED_ARM: 0.865}
    for arm in ARM_ORDER:
        track_time = episodes[arm].first_track_time_s
        if track_time is None:
            continue
        axis.axvline(
            track_time,
            color=ARM_COLOR[arm],
            linestyle=(0, (1.0, 1.5)),
            linewidth=0.9,
            alpha=0.92,
            zorder=2,
        )
        if label:
            late = track_time > 0.80 * float(episodes[arm].time_s[-1])
            axis.text(
                track_time - 4.0 if late else track_time + 4.0,
                label_levels[arm],
                f"TRACK {track_time:.0f} s",
                transform=axis.get_xaxis_transform(),
                ha="right" if late else "left",
                va="top",
                fontsize=6.3,
                color=ARM_COLOR[arm],
            )


def _plot_episode_series(
    axis: mpl.axes.Axes,
    episodes: Mapping[str, EpisodeArm],
    key: str,
    threshold: float,
    threshold_label: str,
    ylabel: str,
) -> None:
    plotted_values: list[np.ndarray] = []
    for arm in ARM_ORDER:
        time_s, values = _finite_positive_series(episodes[arm], key)
        plotted_values.append(values)
        axis.plot(
            time_s,
            values,
            color=ARM_COLOR[arm],
            linestyle=ARM_LINESTYLE[arm],
            linewidth=1.25,
            zorder=3,
        )
    lower, upper = _log_limits(plotted_values, threshold)
    axis.set_yscale("log")
    axis.set_ylim(lower, upper)
    axis.set_ylabel(ylabel)
    _draw_threshold(axis, threshold, threshold_label)
    _style_axis(axis)


def _plot_geometry(
    axis: mpl.axes.Axes,
    episodes: Mapping[str, EpisodeArm],
) -> None:
    reference = episodes[FIXED_ARM]
    leader_xy = _leader_relative_xy(reference)
    for leader_index in range(2):
        axis.plot(
            leader_xy[:, leader_index, 0],
            leader_xy[:, leader_index, 1],
            color=LEADER_COLOR,
            linestyle=LEADER_LINESTYLES[leader_index],
            linewidth=0.9,
            alpha=0.85,
            zorder=1,
        )
        end = leader_xy[-1, leader_index]
        offset = (5.0, 4.0) if leader_index == 0 else (-5.0, -4.0)
        axis.annotate(
            rf"$\mathbf{{p}}_{leader_index + 1}$",
            xy=end,
            xytext=offset,
            textcoords="offset points",
            ha="left" if leader_index == 0 else "right",
            va="bottom" if leader_index == 0 else "top",
            fontsize=6.5,
            color=LEADER_COLOR,
        )

    desired_xy = _desired_relative_xy(reference)
    axis.scatter(
        desired_xy[0],
        desired_xy[1],
        marker="D",
        s=25,
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=6,
    )
    axis.annotate(
        r"$\mathbf{p}_{F,\mathrm{des}}$",
        xy=desired_xy,
        xytext=(5.0, -6.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6.5,
        color="#111111",
    )

    for arm in ARM_ORDER:
        episode = episodes[arm]
        xy = _follower_relative_xy(episode)
        axis.plot(
            xy[:, 0],
            xy[:, 1],
            color=ARM_COLOR[arm],
            linestyle=ARM_LINESTYLE[arm],
            linewidth=1.45,
            zorder=3,
        )
        axis.scatter(
            xy[0, 0],
            xy[0, 1],
            marker="o",
            s=17,
            facecolor="white",
            edgecolor=ARM_COLOR[arm],
            linewidth=0.8,
            zorder=4,
        )
        axis.scatter(
            xy[-1, 0],
            xy[-1, 1],
            marker=">",
            s=22,
            facecolor=ARM_COLOR[arm],
            edgecolor="white",
            linewidth=0.45,
            zorder=8,
        )
        track_time = episode.first_track_time_s
        if track_time is not None:
            track_x = float(np.interp(track_time, episode.time_s, xy[:, 0]))
            track_y = float(np.interp(track_time, episode.time_s, xy[:, 1]))
            axis.scatter(
                track_x,
                track_y,
                marker="*",
                s=62,
                facecolor=ARM_COLOR[arm],
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )

    all_xy = [leader_xy.reshape(-1, 2)]
    all_xy.extend(_follower_relative_xy(episodes[arm]) for arm in ARM_ORDER)
    stacked = np.vstack(all_xy)
    x_min, y_min = np.min(stacked, axis=0)
    x_max, y_max = np.max(stacked, axis=0)
    span = max(x_max - x_min, y_max - y_min)
    padding = 0.075 * span
    axis.set_xlim(x_min - padding, x_max + padding)
    axis.set_ylim(y_min - padding, y_max + padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Centroid-relative $x$ [m]")
    axis.set_ylabel(r"Centroid-relative $y$ [m]")
    _style_axis(axis)


def _set_common_time_axis(
    axes: tuple[mpl.axes.Axes, ...],
    episodes: Mapping[str, EpisodeArm],
) -> None:
    end_time = max(float(episodes[arm].time_s[-1]) for arm in ARM_ORDER)
    ticks = list(np.arange(0.0, end_time, 100.0))
    if not ticks or abs(ticks[-1] - end_time) > 20.0:
        ticks.append(end_time)
    for axis in axes:
        axis.set_xlim(0.0, end_time)
        axis.set_xticks(ticks)
        axis.set_xlabel("Time [s]")


def caption_text(seed: int) -> str:
    common = (
        "Both physical leaders are present in both arms, and both Doppler links "
        "are admitted. (a) Horizontal follower paths in the translated "
        "leader-centroid frame; gray curves show the recorded leader broadcasts. "
        "(b,c) Offline truth-scored localization and formation errors; horizontal "
        "dashed lines mark the 7-m and 8-m task limits. (d) Nominal local radius "
        "used by the evidence gate; the dashed line marks its 7-m release "
        "criterion. Open circles, stars, and filled triangles mark the start, "
        "first ACQUIRE-to-TRACK action, and end. Truth is used only for "
        "retrospective plotting "
        "in (a)--(c); it was unavailable to the online estimator, planner, and "
        "gate. The episode is illustrative, not inferential; statistical claims "
        "use the full paired campaign."
    )
    if seed == MANUSCRIPT_SEED:
        return (
            "Illustrative paired episode for seed 48698. The seed was selected "
            "post hoc by a deterministic robust-medoid rule within the modal "
            "paired outcome stratum. "
            + common
        )
    return (
        f"Layout-test paired episode for seed {seed}; this seed is not the "
        "descriptive manuscript illustration. "
        + common
    )


def make_figure(
    episodes: Mapping[str, EpisodeArm],
    seed: int,
    output: Path,
) -> None:
    fig, axes_array = plt.subplots(
        2,
        2,
        figsize=(7.12, 4.75),
    )
    fig.subplots_adjust(
        left=0.078,
        right=0.992,
        bottom=0.105,
        top=0.895,
        wspace=0.285,
        hspace=0.42,
    )
    geometry_axis = axes_array[0, 0]
    localization_axis = axes_array[0, 1]
    formation_axis = axes_array[1, 0]
    evidence_axis = axes_array[1, 1]

    _plot_geometry(geometry_axis, episodes)
    geometry_axis.set_title("(a) Horizontal paths in the centroid frame")

    _plot_episode_series(
        localization_axis,
        episodes,
        "localization_error_m",
        threshold=7.0,
        threshold_label="7-m localization limit",
        ylabel=r"Localization error $e_p(t)$ [m]",
    )
    localization_axis.set_title(r"(b) Truth-scored $e_p(t)$")
    _draw_first_track_lines(localization_axis, episodes, label=False)

    _plot_episode_series(
        formation_axis,
        episodes,
        "formation_error_truth_m",
        threshold=8.0,
        threshold_label="8-m formation limit",
        ylabel=r"Formation error $e_f(t)$ [m]",
    )
    formation_axis.set_title(r"(c) Truth-scored $e_f(t)$")
    _draw_first_track_lines(formation_axis, episodes, label=False)

    _plot_episode_series(
        evidence_axis,
        episodes,
        "batch_local_radius95_m",
        threshold=7.0,
        threshold_label="7-m gate criterion",
        ylabel=r"Nominal local radius $r_{\mathrm{nom}}(t)$ [m]",
    )
    evidence_axis.set_title(r"(d) Online $r_{\mathrm{nom}}(t)$ and transition")
    _draw_first_track_lines(evidence_axis, episodes, label=True)

    _set_common_time_axis(
        (localization_axis, formation_axis, evidence_axis),
        episodes,
    )

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=ARM_COLOR[FIXED_ARM],
                linestyle=ARM_LINESTYLE[FIXED_ARM],
                linewidth=1.5,
                label=ARM_LABEL[FIXED_ARM],
            ),
            Line2D(
                [0],
                [0],
                color=ARM_COLOR[ACTIVE_ARM],
                linestyle=ARM_LINESTYLE[ACTIVE_ARM],
                linewidth=1.5,
                label=ARM_LABEL[ACTIVE_ARM],
            ),
            Line2D(
                [0],
                [0],
                color=LEADER_COLOR,
                linestyle=LEADER_LINESTYLES[0],
                linewidth=1.0,
                label="Leader broadcasts",
            ),
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="*",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=7.0,
                label="First TRACK",
            ),
            Line2D(
                [0],
                [0],
                color="#111111",
                marker="D",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#111111",
                markersize=5.2,
                label=r"$\mathbf{p}_{F,\mathrm{des}}$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=5,
        frameon=False,
        columnspacing=0.95,
        handlelength=2.25,
        handletextpad=0.55,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in {".pdf", ".png", ".svg", ".eps"}:
        raise ValueError("output suffix must be .pdf, .png, .svg, or .eps")
    save_kwargs: dict[str, Any] = {"format": suffix.lstrip(".")}
    if suffix == ".png":
        save_kwargs["dpi"] = 300
    fig.savefig(output, **save_kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    episodes = load_pair(campaign_dir, int(args.seed))
    make_figure(episodes, int(args.seed), output)
    print(output)
    print("CAPTION:")
    print(caption_text(int(args.seed)))


if __name__ == "__main__":
    main()
