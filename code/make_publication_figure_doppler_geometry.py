#!/usr/bin/env python3
"""Generate the Doppler-geometry schematic using the manuscript notation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
RED = "#CC3355"
PLANE = "#EAF4FB"
GRID = "#BBD7E8"
TEXT = "#222222"
LEADER_FILL = "#F9D423"
FOLLOWER_FILL = "#9ED9D0"
VEHICLE_EDGE = "#111827"
VEHICLE_CENTERLINE = "#718096"
PROJECTION_GRAY = "#6B7280"
UUV_SCALE = 0.62
UUV_LENGTH_SCALE = 1.36


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def arrow(
    axis: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    linestyle: str = "-",
    linewidth: float = 1.4,
    mutation_scale: float = 8.0,
    zorder: float = 5,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
    )


def base_panel(
    axis: mpl.axes.Axes,
    title: str,
    *,
    draw_ground_plane: bool = True,
) -> None:
    if draw_ground_plane:
        plane = np.asarray(
            [[-0.15, -0.10], [2.75, -0.10], [3.35, 0.52], [0.45, 0.52]]
        )
        axis.add_patch(
            Polygon(
                plane,
                closed=True,
                facecolor=PLANE,
                edgecolor=GRID,
                linewidth=0.7,
                zorder=0,
            )
        )
        for fraction in (0.25, 0.50, 0.75):
            left = plane[0] + fraction * (plane[3] - plane[0])
            right = plane[1] + fraction * (plane[2] - plane[1])
            axis.plot(
                [left[0], right[0]],
                [left[1], right[1]],
                color=GRID,
                linewidth=0.35,
                zorder=1,
            )
        for fraction in (0.25, 0.50, 0.75):
            bottom = plane[0] + fraction * (plane[1] - plane[0])
            top = plane[3] + fraction * (plane[2] - plane[3])
            axis.plot(
                [bottom[0], top[0]],
                [bottom[1], top[1]],
                color=GRID,
                linewidth=0.35,
                zorder=1,
            )
    axis.set_xlim(-0.25, 3.45)
    axis.set_ylim(-0.25, 2.15)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, loc="left", fontweight="bold", pad=1.5)


def vehicle(
    axis: mpl.axes.Axes,
    xy: tuple[float, float],
    label: str,
    fill_color: str,
    label_color: str,
    *,
    angle_deg: float,
    label_offset: tuple[float, float] = (0.0, -0.20),
    scale: float = UUV_SCALE,
    length_scale: float = UUV_LENGTH_SCALE,
) -> None:
    """Draw a compact torpedo-like UUV glyph in data coordinates."""
    transform = (
        transforms.Affine2D()
        .scale(scale * length_scale, scale)
        .rotate_deg(angle_deg)
        .translate(xy[0], xy[1])
        + axis.transData
    )
    tail = Rectangle(
        (-0.365, -0.064),
        0.095,
        0.128,
        facecolor=fill_color,
        edgecolor=VEHICLE_EDGE,
        linewidth=0.8,
        transform=transform,
        zorder=7,
    )
    body = FancyBboxPatch(
        (-0.29, -0.095),
        0.58,
        0.19,
        boxstyle="round,pad=0,rounding_size=0.095",
        facecolor=fill_color,
        edgecolor=VEHICLE_EDGE,
        linewidth=0.9,
        transform=transform,
        zorder=7,
    )
    viewport = Circle(
        (0.205, 0.0),
        0.057,
        facecolor="white",
        edgecolor=VEHICLE_CENTERLINE,
        linewidth=0.65,
        transform=transform,
        zorder=8,
    )
    center = Circle(
        (0.0, 0.0),
        0.034,
        facecolor=VEHICLE_EDGE,
        edgecolor="white",
        linewidth=0.45,
        transform=transform,
        zorder=9,
    )
    axis.add_patch(tail)
    axis.add_patch(body)
    axis.add_patch(viewport)
    axis.add_patch(center)
    axis.plot(
        [-0.20, 0.16],
        [0.0, 0.0],
        color=VEHICLE_CENTERLINE,
        linewidth=0.65,
        linestyle=(0, (3.0, 2.5)),
        transform=transform,
        zorder=8,
    )
    axis.text(
        xy[0] + label_offset[0],
        xy[1] + label_offset[1],
        label,
        ha="center",
        va="center",
        color=label_color,
        fontweight="bold",
        zorder=10,
    )


def los(axis: mpl.axes.Axes, follower: tuple[float, float], leader: tuple[float, float]) -> None:
    axis.plot(
        [follower[0], leader[0]],
        [follower[1], leader[1]],
        color="#222222",
        linewidth=1.0,
        zorder=3,
    )


def annotate_box(
    axis: mpl.axes.Axes,
    text: str,
    xy: tuple[float, float],
    color: str,
    *,
    fontsize: float = 7.2,
) -> None:
    axis.text(
        xy[0],
        xy[1],
        text,
        ha="left",
        va="top",
        color=color,
        fontsize=fontsize,
        bbox={
            "boxstyle": "square,pad=0.22",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 0.55,
            "alpha": 0.92,
        },
        zorder=10,
    )


def panel_radial(axis: mpl.axes.Axes) -> None:
    base_panel(axis, "(a) Radial relative velocity")
    follower = np.asarray((0.72, 0.22))
    leader = np.asarray((2.55, 1.31))
    unit_los = (leader - follower) / np.linalg.norm(leader - follower)
    relative_velocity = 0.92 * unit_los
    transverse_velocity = (
        np.eye(2) - np.outer(unit_los, unit_los)
    ) @ relative_velocity
    assert np.allclose(transverse_velocity, 0.0)
    heading_deg = float(np.degrees(np.arctan2(unit_los[1], unit_los[0])))
    los(axis, follower, leader)
    vehicle(
        axis,
        follower,
        r"$\mathbf{p}_F$",
        FOLLOWER_FILL,
        GREEN,
        angle_deg=heading_deg,
        label_offset=(-0.08, -0.30),
    )
    vehicle(
        axis,
        leader,
        r"$\mathbf{p}_i$",
        LEADER_FILL,
        BLUE,
        angle_deg=heading_deg,
        label_offset=(0.12, 0.30),
    )
    relative_tip = follower + relative_velocity
    arrow(
        axis,
        tuple(follower),
        tuple(relative_tip),
        ORANGE,
        linewidth=1.55,
        zorder=10,
    )
    u_start = follower + 1.04 * unit_los
    u_tip = follower + 1.48 * unit_los
    arrow(axis, tuple(u_start), tuple(u_tip), TEXT, linewidth=1.0)
    axis.text(
        *(follower + 1.18 * unit_los + np.asarray((0.06, -0.14))),
        r"$\mathbf{u}_{i,k}$",
        color=TEXT,
    )
    axis.text(
        1.30,
        1.00,
        r"$\widetilde{\mathbf{v}}_{r,i,k}\parallel\mathbf{u}_{i,k}$",
        color=ORANGE,
        ha="center",
    )
    axis.text(
        0.05,
        0.91,
        r"$\mathbf{v}_{\perp,i,k}=\mathbf{0}$",
        color=RED,
    )
    annotate_box(
        axis,
        r"$\mathbf{g}_{i,k}=\mathbf{0}$" "\n"
        r"$\mathbf{J}_{i,k}=\mathbf{0}$",
        (0.00, 1.88),
        RED,
    )


def panel_matched(axis: mpl.axes.Axes) -> None:
    base_panel(axis, "(b) Matched motion")
    follower = np.asarray((0.92, 0.22))
    leader = np.asarray((2.48, 1.25))
    follower_velocity = np.asarray((0.82, 0.28))
    leader_velocity = follower_velocity.copy()
    relative_velocity = leader_velocity - follower_velocity
    assert np.allclose(relative_velocity, 0.0)
    heading_deg = float(
        np.degrees(
            np.arctan2(follower_velocity[1], follower_velocity[0])
        )
    )
    los(axis, follower, leader)
    vehicle(
        axis,
        follower,
        r"$\mathbf{p}_F$",
        FOLLOWER_FILL,
        GREEN,
        angle_deg=heading_deg,
        label_offset=(-0.02, -0.30),
    )
    vehicle(
        axis,
        leader,
        r"$\mathbf{p}_i$",
        LEADER_FILL,
        BLUE,
        angle_deg=heading_deg,
        label_offset=(0.16, 0.30),
    )
    arrow(
        axis,
        tuple(follower),
        tuple(follower + follower_velocity),
        BLUE,
        zorder=10,
    )
    arrow(
        axis,
        tuple(leader),
        tuple(leader + leader_velocity),
        BLUE,
        zorder=10,
    )
    axis.text(1.20, 0.58, r"$\widetilde{\mathbf{v}}_{F,k}$", color=BLUE)
    axis.text(2.78, 1.60, r"$\mathbf{v}_{i,k}$", color=BLUE)
    annotate_box(
        axis,
        r"$\mathbf{v}_{i,k}=\widetilde{\mathbf{v}}_{F,k}$" "\n"
        r"$\widetilde{\mathbf{v}}_{r,i,k}=\mathbf{0}$" "\n"
        r"$\mathbf{v}_{\perp,i,k}=\mathbf{g}_{i,k}=\mathbf{0}$",
        (0.00, 1.88),
        RED,
    )


def spatial_vehicle_scene(
    axis: mpl.axes.Axes,
    *,
    origin_xy: tuple[float, float] = (1.65, 0.48),
    leader_1_position: tuple[float, float, float],
    leader_2_position: tuple[float, float, float],
    plane_x_limits: tuple[float, float] = (-1.35, 1.35),
    plane_y_limits: tuple[float, float] = (-0.70, 0.70),
    show_depth_guides: bool,
    show_lateral_baseline: bool,
    collinear_attitudes: bool,
):
    """Draw an oblique three-vehicle geometry relative to the plane Pi."""
    origin = np.asarray(origin_xy, dtype=float)
    projection = np.asarray(
        (
            (0.95, 0.00),
            (0.35, 0.32),
            (0.18, 0.75),
        )
    )

    def project(vector: np.ndarray) -> np.ndarray:
        return origin + np.asarray(vector) @ projection

    plane_coordinates = np.asarray(
        (
            (plane_x_limits[0], plane_y_limits[0], 0.0),
            (plane_x_limits[1], plane_y_limits[0], 0.0),
            (plane_x_limits[1], plane_y_limits[1], 0.0),
            (plane_x_limits[0], plane_y_limits[1], 0.0),
        )
    )
    plane = np.asarray(tuple(project(point) for point in plane_coordinates))
    axis.add_patch(
        Polygon(
            plane,
            closed=True,
            facecolor=PLANE,
            edgecolor=GRID,
            linewidth=0.8,
            zorder=0,
        )
    )
    for fraction in (0.33, 0.66):
        bottom = plane[0] + fraction * (plane[1] - plane[0])
        top = plane[3] + fraction * (plane[2] - plane[3])
        axis.plot(
            [bottom[0], top[0]],
            [bottom[1], top[1]],
            color=GRID,
            linewidth=0.35,
            zorder=1,
        )
    left = 0.5 * (plane[0] + plane[3])
    right = 0.5 * (plane[1] + plane[2])
    axis.plot(
        [left[0], right[0]],
        [left[1], right[1]],
        color=GRID,
        linewidth=0.35,
        zorder=1,
    )
    leader_1_position = np.asarray(leader_1_position, dtype=float)
    leader_2_position = np.asarray(leader_2_position, dtype=float)
    leader_1 = project(leader_1_position)
    leader_2 = project(leader_2_position)
    if show_depth_guides:
        for leader_position, leader in (
            (leader_1_position, leader_1),
            (leader_2_position, leader_2),
        ):
            projected_position = leader_position.copy()
            projected_position[2] = 0.0
            projected_leader = project(projected_position)
            axis.plot(
                [projected_leader[0], leader[0]],
                [projected_leader[1], leader[1]],
                color=PROJECTION_GRAY,
                linewidth=0.9,
                linestyle=(0, (2.0, 2.0)),
                zorder=3,
            )
        axis.text(
            2.76,
            1.24,
            r"$z_1\ne z_2$",
            color=PROJECTION_GRAY,
            fontsize=7.0,
            ha="center",
            va="center",
        )
    if show_lateral_baseline:
        baseline_midpoint_position = 0.5 * (
            leader_1_position + leader_2_position
        )
        baseline_midpoint = project(baseline_midpoint_position)
        axis.plot(
            [leader_1[0], leader_2[0]],
            [leader_1[1], leader_2[1]],
            color=BLUE,
            linewidth=0.8,
            linestyle=(0, (4.0, 2.0)),
            zorder=2,
        )
        axis.plot(
            [origin[0], baseline_midpoint[0]],
            [origin[1], baseline_midpoint[1]],
            color=PROJECTION_GRAY,
            linewidth=0.85,
            linestyle=(0, (3.0, 2.0)),
            zorder=3,
        )
        axis.text(
            *(0.5 * (origin + baseline_midpoint) + np.asarray((0.0, -0.28))),
            "lateral offset",
            color=PROJECTION_GRAY,
            fontsize=7.0,
            ha="center",
            va="center",
        )
    los(axis, origin, leader_1)
    los(axis, origin, leader_2)
    collinear_heading = float(
        np.degrees(
            np.arctan2(
                (leader_2 - origin)[1],
                (leader_2 - origin)[0],
            )
        )
    )
    vehicle(
        axis,
        origin,
        r"$\mathbf{p}_F$",
        FOLLOWER_FILL,
        GREEN,
        angle_deg=collinear_heading if collinear_attitudes else 8,
        label_offset=(0.0, -0.24),
    )
    vehicle(
        axis,
        leader_1,
        r"$\mathbf{p}_1$",
        LEADER_FILL,
        BLUE,
        angle_deg=collinear_heading if collinear_attitudes else 10,
        label_offset=(0.0, 0.25) if show_depth_guides else (0.0, -0.24),
    )
    vehicle(
        axis,
        leader_2,
        r"$\mathbf{p}_2$",
        LEADER_FILL,
        BLUE,
        angle_deg=collinear_heading if collinear_attitudes else 18,
        label_offset=(0.0, 0.25) if show_depth_guides else (0.0, -0.24),
    )

    if show_lateral_baseline:
        axis.text(
            0.75,
            0.46,
            r"$\mathbf{u}_{1,k}$",
            color=TEXT,
            fontsize=7.0,
            ha="center",
            va="center",
            zorder=6,
        )
        axis.text(
            2.45,
            1.00,
            r"$\mathbf{u}_{2,k}$",
            color=TEXT,
            fontsize=7.0,
            ha="center",
            va="center",
            zorder=6,
        )
        plane_label_xy = (3.17, 0.54)
    else:
        axis.text(
            1.10,
            0.41,
            r"$\mathbf{u}_{1,k}$",
            color=TEXT,
            fontsize=7.0,
            ha="center",
            va="center",
            zorder=6,
        )
        axis.text(
            2.25,
            0.88,
            r"$\mathbf{u}_{2,k}$",
            color=TEXT,
            fontsize=7.0,
            ha="center",
            va="center",
            zorder=6,
        )
        plane_label_xy = (3.20, 0.35)
    axis.text(*plane_label_xy, r"$\Pi$", color=BLUE, fontweight="bold")
    return origin, project, leader_1_position, leader_2_position


def panel_planar(axis: mpl.axes.Axes) -> None:
    base_panel(
        axis,
        "(c) Equal-depth near-collinear: rank deficient",
        draw_ground_plane=False,
    )
    origin, project, leader_1_position, leader_2_position = (
        spatial_vehicle_scene(
            axis,
            origin_xy=(0.35, 0.38),
            leader_1_position=(1.10, 0.75, 0.0),
            leader_2_position=(2.30, 1.55, 0.0),
            plane_x_limits=(-0.25, 2.65),
            plane_y_limits=(-0.55, 1.65),
            show_depth_guides=False,
            show_lateral_baseline=False,
            collinear_attitudes=True,
        )
    )
    sensitivity_1 = (
        1.20
        * np.asarray((-0.75, 1.10, 0.0))
        / np.linalg.norm((-0.75, 1.10))
    )
    sensitivity_2 = (
        1.05
        * np.asarray((-1.55, 2.30, 0.0))
        / np.linalg.norm((-1.55, 2.30))
    )
    sensitivity_3 = np.asarray((0.65, -0.25, 0.0))
    later_los = np.asarray((0.25, 0.65, 0.0))
    assert np.isclose(leader_1_position @ sensitivity_1, 0.0)
    assert np.isclose(leader_2_position @ sensitivity_2, 0.0)
    assert np.isclose(later_los @ sensitivity_3, 0.0)
    planar_sensitivities = np.asarray(
        (sensitivity_1, sensitivity_2, sensitivity_3)
    )
    normal = np.asarray((0.0, 0.0, 1.0))
    information = planar_sensitivities.T @ planar_sensitivities
    assert np.linalg.matrix_rank(planar_sensitivities) == 2
    assert np.allclose(planar_sensitivities @ normal, 0.0)
    assert np.allclose(information @ normal, 0.0)
    for sensitivity in planar_sensitivities:
        arrow(
            axis,
            tuple(origin),
            tuple(project(sensitivity)),
            GREEN,
            linewidth=1.5,
            zorder=10,
        )
    weak_tip = project(np.asarray((0.0, 0.0, 0.95)))
    arrow(
        axis,
        tuple(origin),
        tuple(weak_tip),
        RED,
        linestyle="--",
        linewidth=1.5,
        zorder=9,
    )
    axis.text(
        0.70,
        0.80,
        r"$\mathbf{v}_{\perp,1,k}\approx\mathbf{v}_{\perp,2,k}$",
        color=GREEN,
        fontsize=7.0,
        ha="left",
    )
    axis.text(0.75, 0.16, r"$\mathbf{v}_{\perp,i,k'}$", color=GREEN)
    axis.text(
        0.68,
        1.12,
        r"$\mathbf{d}\in\Pi^\perp$",
        color=RED,
        fontsize=7.0,
        ha="left",
    )
    annotate_box(
        axis,
        r"$p_{1,z}=p_{F,z}=p_{2,z}$" "\n"
        r"$\mathbf{u}_{1,k}\approx\mathbf{u}_{2,k}$" "\n"
        r"$\mathrm{span}\{\mathbf{v}_{\perp,i,k}\}_{\mathcal{K}}\subseteq\Pi$" "\n"
        r"$\mathbf{J}_{\mathcal{K}}\mathbf{d}=\mathbf{0},\quad"
        r"\mathrm{rank}(\mathbf{J}_{\mathcal{K}})\leq2$",
        (-0.18, 1.98),
        RED,
        fontsize=7.0,
    )


def panel_diverse(axis: mpl.axes.Axes) -> None:
    base_panel(
        axis,
        "(d) Depth-diverse lateral: full-rank history",
        draw_ground_plane=False,
    )
    origin, project, leader_1_position, leader_2_position = (
        spatial_vehicle_scene(
            axis,
            origin_xy=(1.65, 0.72),
            leader_1_position=(-1.20, -0.85, -0.55),
            leader_2_position=(1.65, -0.85, 0.55),
            plane_x_limits=(-1.25, 1.70),
            plane_y_limits=(-1.00, 0.75),
            show_depth_guides=True,
            show_lateral_baseline=True,
            collinear_attitudes=False,
        )
    )
    sensitivity_1 = 1.30 * np.asarray((-0.85, 1.20, 0.0))
    sensitivity_2 = 0.90 * np.asarray((2.0 / 11.0, 1.0, 1.0))
    sensitivity_3 = np.asarray((-0.90, 0.0, 0.0))
    later_los = np.asarray((0.0, 1.0, 0.0))
    assert np.isclose(leader_1_position @ sensitivity_1, 0.0)
    assert np.isclose(leader_2_position @ sensitivity_2, 0.0)
    assert np.isclose(later_los @ sensitivity_3, 0.0)
    spatial_sensitivities = np.asarray(
        (sensitivity_1, sensitivity_2, sensitivity_3)
    )
    assert np.linalg.matrix_rank(spatial_sensitivities) == 3
    assert np.linalg.eigvalsh(
        spatial_sensitivities.T @ spatial_sensitivities
    )[0] > 0.0
    tip_1 = project(sensitivity_1)
    tip_2 = project(sensitivity_2)
    tip_3 = project(sensitivity_3)
    arrow(axis, tuple(origin), tuple(tip_1), GREEN, linewidth=1.5, zorder=10)
    arrow(axis, tuple(origin), tuple(tip_3), GREEN, linewidth=1.5, zorder=10)
    arrow(axis, tuple(origin), tuple(tip_2), GREEN, linewidth=1.7, zorder=10)
    axis.text(
        1.02,
        1.27,
        r"$\mathbf{v}_{\perp,1,k}\in\Pi$",
        color=GREEN,
        ha="right",
    )
    axis.text(
        0.42,
        0.83,
        r"$\mathbf{v}_{\perp,i,k'}\in\Pi$",
        color=GREEN,
        ha="left",
    )
    axis.text(
        2.34,
        1.53,
        r"$\mathbf{v}_{\perp,2,k}\notin\Pi$",
        color=GREEN,
        ha="left",
    )
    annotate_box(
        axis,
        r"$\mathbf{u}_{2,k}^{\top}\mathbf{v}_{\perp,2,k}=0$" "\n"
        r"$\mathrm{span}\{\mathbf{v}_{\perp,i,k}\}_{\mathcal{K}}=\mathbb{R}^3$" "\n"
        r"$\mathbf{J}_{\mathcal{K}}\succ\mathbf{0},\quad"
        r"\lambda_{\min}(\mathbf{J}_{\mathcal{K}})>0$",
        (-0.18, 1.98),
        GREEN,
        fontsize=7.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/figures/figure3_doppler_degeneracies.pdf",
        help="Output PDF or SVG (default: repository results/figures/).",
    )
    args = parser.parse_args()
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.55))
    panel_radial(axes[0, 0])
    panel_matched(axes[0, 1])
    panel_planar(axes[1, 0])
    panel_diverse(axes[1, 1])
    handles = [
        Line2D([0], [0], color="#222222", linewidth=1.0, label="LOS / acoustic link"),
        Line2D([0], [0], color=GREEN, linewidth=1.5, label=r"$\mathbf{v}_{\perp,i,k}$"),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            linewidth=1.5,
            label=r"$\widetilde{\mathbf{v}}_{r,i,k}$",
        ),
        Line2D(
            [0],
            [0],
            color=RED,
            linestyle="--",
            linewidth=1.5,
            label=r"weak direction $\mathbf{d}$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=7.0,
        columnspacing=1.3,
        handlelength=2.2,
    )
    fig.subplots_adjust(
        left=0.015,
        right=0.99,
        top=0.97,
        bottom=0.105,
        wspace=0.10,
        hspace=0.22,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix not in {".pdf", ".svg"}:
        raise ValueError("output suffix must be .pdf or .svg")
    fig.savefig(args.output, format=suffix.lstrip("."))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
