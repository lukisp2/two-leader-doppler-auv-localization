#!/usr/bin/env python3
"""Plot the recorded formation-frame geometry of the manuscript episode."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from make_publication_figure_paired_episode import (
    ACTIVE_ARM,
    DEFAULT_CAMPAIGN,
    MANUSCRIPT_SEED,
    configure_style,
    load_pair,
)


BLUE = "#0072B2"
LIGHT_BLUE = "#78B7D8"
GREEN = "#009E73"
GRAY = "#666666"
BLACK = "#111111"
GRID = "#777777"
LEADER_LINESTYLES = ((0, (1.4, 1.4)), (0, (5.0, 1.8, 1.2, 1.8)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--seed", type=int, default=MANUSCRIPT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/figures/fig_mission_geometry_seed48698.pdf",
    )
    return parser.parse_args()


def _nearest_indices(source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_t, target_t, side="left")
    indices = np.clip(indices, 0, source_t.size - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = (
        np.abs(source_t[previous] - target_t)
        < np.abs(source_t[indices] - target_t)
    )
    indices = np.where(use_previous, previous, indices)
    if float(np.max(np.abs(source_t[indices] - target_t))) > 0.51:
        raise RuntimeError("could not align leader broadcasts to decision times")
    return indices


def _formation_coordinates(
    trace: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    time_s = np.asarray(trace["time_s"], dtype=float)
    online_t_s = np.asarray(trace["online_t_s"], dtype=float)
    leader_online = np.asarray(trace["online_leader_position_m"], dtype=float)
    velocity_online = np.asarray(trace["online_leader_velocity_mps"], dtype=float)
    indices = _nearest_indices(online_t_s, time_s)
    leaders = leader_online[indices]
    centroid = np.mean(leaders, axis=1)

    centroid_velocity = np.mean(velocity_online, axis=(0, 1))
    forward = centroid_velocity.copy()
    forward[2] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if not math.isfinite(forward_norm) or forward_norm <= 0.0:
        raise RuntimeError("leader-centroid horizontal velocity is degenerate")
    forward /= forward_norm
    vertical = np.asarray([0.0, 0.0, 1.0])
    side = np.cross(forward, vertical)
    basis = np.column_stack((forward, side, vertical))

    truth = np.column_stack(
        (
            np.asarray(trace["truth_x"], dtype=float),
            np.asarray(trace["truth_y"], dtype=float),
            np.asarray(trace["truth_z"], dtype=float),
        )
    )
    estimate = np.column_stack(
        (
            np.asarray(trace["estimate_x"], dtype=float),
            np.asarray(trace["estimate_y"], dtype=float),
            np.asarray(trace["estimate_z"], dtype=float),
        )
    )
    truth_frame = np.einsum("ni,ij->nj", truth - centroid, basis)
    estimate_frame = np.einsum("ni,ij->nj", estimate - centroid, basis)
    leader_frame = np.einsum(
        "nli,ij->nlj",
        leaders - centroid[:, np.newaxis, :],
        basis,
    )
    desired_frame = np.tile(np.asarray([-120.0, 0.0, 0.0]), (time_s.size, 1))
    return (
        time_s,
        truth_frame,
        estimate_frame,
        leader_frame,
        desired_frame,
        basis,
    )


def _style_axis(axis: mpl.axes.Axes) -> None:
    axis.grid(True, color=GRID, alpha=0.20, linewidth=0.55, zorder=0)
    axis.tick_params(direction="out", length=2.8, width=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_aspect("equal", adjustable="box")


def _plot_projection(
    axis: mpl.axes.Axes,
    time_s: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    leaders: np.ndarray,
    desired: np.ndarray,
    columns: tuple[int, int],
    track_time_s: float,
) -> None:
    x_index, y_index = columns
    track_index = int(np.argmin(np.abs(time_s - track_time_s)))
    acquire = time_s <= track_time_s
    track = time_s >= track_time_s

    for leader_index in range(2):
        axis.plot(
            leaders[:, leader_index, x_index],
            leaders[:, leader_index, y_index],
            color=GRAY,
            linestyle=LEADER_LINESTYLES[leader_index],
            linewidth=1.0,
            zorder=2,
        )
        endpoint = leaders[-1, leader_index, [x_index, y_index]]
        axis.annotate(
            rf"$\mathbf{{p}}_{leader_index + 1}$",
            xy=endpoint,
            xytext=(4.0, 4.0 if leader_index == 0 else -7.0),
            textcoords="offset points",
            fontsize=6.6,
            color=GRAY,
            ha="left",
            va="bottom" if leader_index == 0 else "top",
        )

    axis.plot(
        truth[acquire, x_index],
        truth[acquire, y_index],
        color=LIGHT_BLUE,
        linewidth=1.4,
        zorder=3,
    )
    axis.plot(
        truth[track, x_index],
        truth[track, y_index],
        color=BLUE,
        linewidth=1.8,
        zorder=4,
    )
    finite_estimate = np.all(np.isfinite(estimate), axis=1)
    axis.plot(
        estimate[finite_estimate, x_index],
        estimate[finite_estimate, y_index],
        color=GREEN,
        linestyle=(0, (3.0, 1.7)),
        linewidth=1.0,
        zorder=3,
    )

    follower_track = truth[track_index, [x_index, y_index]]
    for leader_index in range(2):
        leader_track = leaders[track_index, leader_index, [x_index, y_index]]
        axis.plot(
            [follower_track[0], leader_track[0]],
            [follower_track[1], leader_track[1]],
            color="#A0A0A0",
            linestyle=(0, (1.0, 1.8)),
            linewidth=0.65,
            zorder=1,
        )

    axis.scatter(
        desired[0, x_index],
        desired[0, y_index],
        marker="D",
        s=27,
        facecolor="white",
        edgecolor=BLACK,
        linewidth=0.9,
        zorder=6,
    )
    axis.annotate(
        r"$\mathbf{p}_{F,\mathrm{des}}$",
        xy=desired[0, [x_index, y_index]],
        xytext=(5.0, -7.0),
        textcoords="offset points",
        fontsize=6.6,
        color=BLACK,
        ha="left",
        va="top",
    )
    axis.scatter(
        truth[0, x_index],
        truth[0, y_index],
        marker="o",
        s=23,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=0.9,
        zorder=6,
    )
    axis.scatter(
        truth[track_index, x_index],
        truth[track_index, y_index],
        marker="*",
        s=68,
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.6,
        zorder=7,
    )
    axis.scatter(
        truth[-1, x_index],
        truth[-1, y_index],
        marker=">",
        s=28,
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.5,
        zorder=7,
    )
    _style_axis(axis)


def make_figure(trace: dict[str, np.ndarray], track_time_s: float, output: Path) -> None:
    time_s, truth, estimate, leaders, desired, _ = _formation_coordinates(trace)
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 3.03))
    fig.subplots_adjust(
        left=0.077,
        right=0.992,
        bottom=0.245,
        top=0.905,
        wspace=0.27,
    )
    _plot_projection(
        axes[0],
        time_s,
        truth,
        estimate,
        leaders,
        desired,
        (0, 1),
        track_time_s,
    )
    axes[0].set_title("(a) Horizontal formation-frame projection")
    axes[0].set_xlabel("Along-track offset from leader centroid [m]")
    axes[0].set_ylabel("Cross-track offset from leader centroid [m]")

    _plot_projection(
        axes[1],
        time_s,
        truth,
        estimate,
        leaders,
        desired,
        (0, 2),
        track_time_s,
    )
    axes[1].set_title("(b) Along-track--vertical projection")
    axes[1].set_xlabel("Along-track offset from leader centroid [m]")
    axes[1].set_ylabel("Vertical offset from leader centroid [m]")

    handles = [
        Line2D([0], [0], color=LIGHT_BLUE, linewidth=1.5, label="Follower truth: ACQUIRE"),
        Line2D([0], [0], color=BLUE, linewidth=1.8, label="Follower truth: TRACK"),
        Line2D(
            [0],
            [0],
            color=GREEN,
            linestyle=(0, (3.0, 1.7)),
            linewidth=1.1,
            label=r"Online estimate $\widehat{\mathbf{p}}_F$",
        ),
        Line2D(
            [0],
            [0],
            color=GRAY,
            linestyle=LEADER_LINESTYLES[0],
            linewidth=1.0,
            label=r"Leaders $\mathbf{p}_1,\mathbf{p}_2$",
        ),
        Line2D(
            [0],
            [0],
            color=BLACK,
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markersize=5.2,
            label=r"Reference $\mathbf{p}_{F,\mathrm{des}}$",
        ),
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="*",
            linestyle="none",
            markeredgecolor="white",
            markersize=7.5,
            label=f"First TRACK ({track_time_s:.0f} s)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        columnspacing=1.15,
        handlelength=2.2,
        handletextpad=0.5,
        fontsize=6.8,
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
    pair = load_pair(args.campaign_dir.expanduser().resolve(), int(args.seed))
    episode = pair[ACTIVE_ARM]
    track_time_s = episode.first_track_time_s
    if track_time_s is None:
        raise RuntimeError("the selected adaptive episode never enters TRACK")
    output = args.output.expanduser().resolve()
    make_figure(dict(episode.trace), float(track_time_s), output)
    print(output)
    print(
        "CAPTION: Recorded nominal geometry for the descriptive paired episode "
        f"(seed {int(args.seed)}; first TRACK at {track_time_s:.0f} s)."
    )


if __name__ == "__main__":
    main()
