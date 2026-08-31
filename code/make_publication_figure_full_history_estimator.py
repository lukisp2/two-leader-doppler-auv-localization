#!/usr/bin/env python3
"""Generate the explanatory full-history-estimator figure for the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GRAY = "#5F6368"
LIGHT = "#E9ECEF"
DARK = "#202124"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", pad=4, fontsize=7.6, linespacing=1.05)


def trajectory_panel(ax: plt.Axes) -> None:
    panel_title(
        ax,
        "(a) Each initial-position hypothesis\ngenerates one complete trajectory",
    )
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 2.65)
    ax.set_aspect("equal")
    ax.axis("off")

    t = np.linspace(0.0, 1.0, 120)
    dx = 2.55 * t + 0.14 * np.sin(2.0 * np.pi * t)
    dy = 0.38 * t + 0.18 * np.sin(np.pi * t)
    starts = [(0.55, 0.38), (0.55, 0.98), (0.55, 1.58)]
    colors = [GREEN, BLUE, ORANGE]
    line_styles = ["-", (0, (5, 2)), (0, (1.4, 1.4))]
    labels = [
        r"$\widehat{\mathbf{p}}_{F,0}^{(0)}$",
        r"$\widehat{\mathbf{p}}_{F,0}^{(1)}$",
        r"$\widehat{\mathbf{p}}_{F,0}^{(2)}$",
    ]

    for (x0, y0), color, line_style, label in zip(
        starts, colors, line_styles, labels
    ):
        end = (x0 + dx[-1], y0 + dy[-1])
        ax.plot(x0 + dx, y0 + dy, color=color, lw=1.55, ls=line_style)
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                end,
                arrowstyle="-|>",
                mutation_scale=7,
                lw=0.70,
                linestyle=(0, (3, 2)),
                color=GRAY,
                alpha=0.88,
                zorder=2,
            )
        )
        ax.scatter([x0], [y0], s=28, marker="o", facecolor="white",
                   edgecolor=color, linewidth=1.15, zorder=4)
        ax.scatter([end[0]], [end[1]], s=20, marker="s", facecolor=color,
                   edgecolor="white", linewidth=0.45, zorder=5)
        ax.text(x0 - 0.12, y0, label, color=color, ha="right", va="center")

    ax.text(
        1.38,
        0.08,
        r"shared $\Delta\mathbf{p}_{\mathrm{DR}}(t_k)$",
        color=GRAY,
        ha="center",
        va="bottom",
        fontsize=7.0,
        bbox={"boxstyle": "square,pad=0.05", "fc": "white", "ec": "none"},
    )
    ax.text(
        3.35,
        0.08,
        r"$\blacksquare$ modeled position at $t_k$",
        color=GRAY,
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    ax.text(
        2.25,
        2.51,
        r"$\mathbf{p}_F^{\mathrm{mod}}(t_k;\widehat{\mathbf{p}}_{F,0}^{(m)})"
        r"=\widehat{\mathbf{p}}_{F,0}^{(m)}+\Delta\mathbf{p}_{\mathrm{DR}}(t_k)$",
        ha="center",
        va="top",
        bbox={"boxstyle": "square,pad=0.28", "fc": "white", "ec": "#B8B8B8", "lw": 0.7},
        fontsize=7.2,
    )


def shell_points(n: int, shift: float) -> np.ndarray:
    golden = np.pi * (3.0 - np.sqrt(5.0))
    j = np.arange(n)
    radii = 0.67 + 0.57 * ((j + 0.5 + shift) % n) / n
    angles = golden * (j + shift)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def search_panel(ax: plt.Axes) -> None:
    panel_title(ax, r"(b) Broad support search and local refinement")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.47, 1.47)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Circle((0, 0), 1.27, fc="#F4F5F6", ec=GRAY, lw=0.8))
    ax.add_patch(Circle((0, 0), 0.62, fc="white", ec=GRAY, lw=0.8))
    for radius in (0.82, 1.03):
        ax.add_patch(Circle((0, 0), radius, fc="none", ec="#D5D8DC",
                            lw=0.45, ls=(0, (2, 2))))
    ax.scatter([0], [0], s=24, marker="+", color=DARK, lw=1.0, zorder=5)
    ax.text(0.08, -0.06, r"$\mathbf{c}_L(0)$", ha="left", va="top")

    inner_angle = np.deg2rad(145.0)
    outer_angle = np.deg2rad(0.0)
    inner_end = 0.62 * np.array([np.cos(inner_angle), np.sin(inner_angle)])
    outer_end = 1.27 * np.array([np.cos(outer_angle), np.sin(outer_angle)])
    for endpoint, label, offset in [
        (inner_end, r"$120\,\mathrm{m}$", (-0.08, 0.05)),
        (outer_end, r"$350\,\mathrm{m}$", (0.04, 0.04)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                (0.0, 0.0),
                endpoint,
                arrowstyle="-|>",
                mutation_scale=6,
                lw=0.65,
                color=GRAY,
                zorder=2,
            )
        )
        midpoint = 0.56 * endpoint
        ax.text(
            midpoint[0] + offset[0],
            midpoint[1] + offset[1],
            label,
            color=GRAY,
            ha="center",
            va="center",
            fontsize=6.5,
            bbox={"boxstyle": "square,pad=0.04", "fc": "white", "ec": "none"},
        )

    points_1 = shell_points(24, 0.0)
    points_2 = shell_points(24, 0.43)
    ax.scatter(points_1[:, 0], points_1[:, 1], s=11, facecolor="white",
               edgecolor=BLUE, linewidth=0.85, label="shift 1", zorder=3)
    ax.scatter(points_2[:, 0], points_2[:, 1], s=14, marker="x",
               color=ORANGE, linewidth=0.85, label="shift 2", zorder=3)

    minima = np.array([[-0.66, 0.63], [0.72, -0.53], [0.76, 0.50]])
    selected_sources = [points_1[4], points_2[12], points_1[19]]
    for source, target in zip(selected_sources, minima):
        ax.scatter(
            source[0],
            source[1],
            s=16,
            marker="s",
            facecolor="white",
            edgecolor=DARK,
            linewidth=0.75,
            zorder=5,
        )
        ax.add_patch(
            FancyArrowPatch(
                source,
                target,
                arrowstyle="-|>",
                mutation_scale=7,
                connectionstyle="arc3,rad=0.10",
                lw=0.9,
                color=GRAY,
            )
        )
        ax.add_patch(Ellipse(target, 0.26, 0.14, angle=28, fill=False,
                             ec=GRAY, lw=0.75, alpha=0.75))
        ax.scatter(*target, marker="*", s=42, color=DARK, edgecolor="white",
                   linewidth=0.4, zorder=6)

    ax.text(
        0,
        1.35,
        "○ sweep 1   × sweep 2   □ selected start   "
        r"$\star$ refined minimum",
        ha="center",
        va="bottom",
        color=DARK,
        fontsize=5.4,
    )
    ax.annotate(
        "constrained local refinement",
        xy=tuple((selected_sources[1] + minima[1]) / 2),
        xytext=(-0.16, -1.35),
        ha="center",
        va="center",
        color=GRAY,
        fontsize=6.2,
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.55},
    )


def diagnostic_panel(ax: plt.Axes) -> None:
    panel_title(ax, r"(c) Local diagnostic and a retained alternative")
    ax.set_xlim(0, 4.20)
    ax.set_ylim(0, 2.70)
    ax.set_aspect("equal")
    ax.axis("off")

    primary = np.array([0.72, 0.65])
    alternative = np.array([3.58, 1.48])
    for center, widths, angle, color in [
        (primary, [(1.34, 0.76), (0.92, 0.50), (0.54, 0.28)], 24, GREEN),
        (alternative, [(1.22, 0.70), (0.82, 0.44)], -18, ORANGE),
    ]:
        for w, h in widths:
            ax.add_patch(Ellipse(center, w, h, angle=angle, fill=False,
                                 ec=color, lw=0.72, alpha=0.55,
                                 ls="-" if color == GREEN else (0, (4, 2))))

    ax.scatter(*primary, marker="*", s=58, color=GREEN, edgecolor="white",
               linewidth=0.45, zorder=5)
    ax.scatter(*alternative, marker="*", s=58, facecolor="white", edgecolor=ORANGE,
               linewidth=1.0, zorder=5)
    ax.annotate(
        "primary\n" + r"$\widehat{\mathbf{p}}_{F,0}^{(0)},\ S_0$",
        xy=primary,
        xytext=(0.08, 0.08),
        color=GREEN,
        ha="left",
        va="center",
        linespacing=1.0,
        arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 0.55},
    )
    ax.annotate(
        "alternative\n" + r"$\widehat{\mathbf{p}}_{F,0}^{(m)},\ S_m$",
        xy=alternative,
        xytext=(3.62, 2.03),
        color=ORANGE,
        ha="center",
        va="center",
        linespacing=1.0,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.55},
    )

    separation = alternative - primary
    separation_unit = separation / np.linalg.norm(separation)
    dimension_normal = np.array([separation_unit[1], -separation_unit[0]])
    dimension_offset = 0.30 * dimension_normal
    dimension_start = primary + dimension_offset
    dimension_end = alternative + dimension_offset
    ax.plot(
        [primary[0], dimension_start[0]],
        [primary[1], dimension_start[1]],
        color=GRAY,
        lw=0.50,
        ls=(0, (2, 2)),
    )
    ax.plot(
        [alternative[0], dimension_end[0]],
        [alternative[1], dimension_end[1]],
        color=GRAY,
        lw=0.50,
        ls=(0, (2, 2)),
    )
    ax.add_patch(
        FancyArrowPatch(
            dimension_start,
            dimension_end,
            arrowstyle="<->",
            shrinkA=0,
            shrinkB=0,
            mutation_scale=8,
            lw=0.85,
            linestyle=(0, (4, 2)),
            color=GRAY,
        )
    )
    dimension_midpoint = (dimension_start + dimension_end) / 2
    ax.text(
        dimension_midpoint[0],
        dimension_midpoint[1] - 0.12,
        "spatial separation\n"
        r"$\|\widehat{\mathbf{p}}_{F,0}^{(m)}-\widehat{\mathbf{p}}_{F,0}^{(0)}\|"
        r"\geq7\,\mathrm{m}$",
        ha="center",
        va="top",
        color=GRAY,
        fontsize=7.0,
        linespacing=1.05,
        bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none"},
    )

    local_angle = np.deg2rad(24.0)
    local_radius_end = primary + 0.67 * np.array(
        [np.cos(local_angle), np.sin(local_angle)]
    )
    ax.add_patch(
        FancyArrowPatch(
            primary,
            local_radius_end,
            arrowstyle="-|>",
            shrinkA=0,
            shrinkB=0,
            mutation_scale=7,
            lw=0.9,
            color=GREEN,
        )
    )
    local_radius_midpoint = (primary + local_radius_end) / 2
    ax.text(
        local_radius_midpoint[0] + 0.02,
        local_radius_midpoint[1] + 0.11,
        r"$r_{\mathrm{nom}}$",
        color=GREEN,
        ha="center",
        va="bottom",
        bbox={"boxstyle": "square,pad=0.04", "fc": "white", "ec": "none"},
    )
    ax.text(
        1.58,
        2.40,
        r"$\Delta\chi_m^2=(S_m-S_0)/\sigma_0^2$",
        ha="center",
        va="top",
        bbox={"boxstyle": "square,pad=0.28", "fc": "white", "ec": "#B8B8B8", "lw": 0.7},
        fontsize=7.2,
    )
    ax.text(
        1.58,
        2.63,
        "residual-cost contrast",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=7.0,
    )
    ax.text(
        0.08,
        1.86,
        "schematic local-curvature\ncontour for the primary basin",
        color=GREEN,
        ha="left",
        va="center",
        fontsize=7.1,
        linespacing=1.0,
    )
    contour_parameter = np.deg2rad(100.0)
    contour_anchor = primary + np.array(
        [
            0.67 * np.cos(contour_parameter) * np.cos(local_angle)
            - 0.38 * np.sin(contour_parameter) * np.sin(local_angle),
            0.67 * np.cos(contour_parameter) * np.sin(local_angle)
            + 0.38 * np.sin(contour_parameter) * np.cos(local_angle),
        ]
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.84, 1.68),
            contour_anchor,
            arrowstyle="-",
            lw=0.55,
            color=GREEN,
        )
    )


def build(output_dir: Path) -> None:
    setup()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48), gridspec_kw={"wspace": 0.12})
    trajectory_panel(axes[0])
    search_panel(axes[1])
    diagnostic_panel(axes[2])
    fig.subplots_adjust(left=0.012, right=0.993, bottom=0.035, top=0.88)

    stem = output_dir / "figure4_full_history_estimator"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output_dir / "figure4_full_history_estimator_220dpi.png",
                dpi=220, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output_dir / "figure4_full_history_estimator_360dpi.png",
                dpi=360, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/figures",
        help="Output directory (default: repository results/figures/).",
    )
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
