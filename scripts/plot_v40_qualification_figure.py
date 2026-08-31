#!/usr/bin/env python3
"""Render the publication figure from the validated qualification JSON.

The script consumes only ``v40_publication_results.json`` produced by
``postprocess_v40_qualification.py``.  It does not open a campaign directory,
episode record, trace, campaign summary, or decision file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter


POLICY_FIXED = "fixed_s_turn"
POLICY_ACTIVE = "belief_active"
POLICY_ORDER = (POLICY_FIXED, POLICY_ACTIVE)

EXECUTION_KINEMATIC = "kinematic"
EXECUTION_DYNAMIC = "low_order_dynamic"
EXECUTION_ORDER = (EXECUTION_KINEMATIC, EXECUTION_DYNAMIC)

CURRENT_NONE = "no_current"
CURRENT_VISIBLE = "bottom_track_visible"
CURRENT_ORDER = (CURRENT_NONE, CURRENT_VISIBLE)

CELL_ORDER = tuple(
    (execution, current)
    for execution in EXECUTION_ORDER
    for current in CURRENT_ORDER
)
EXPECTED_CELLS = {
    (policy, execution, current)
    for policy in POLICY_ORDER
    for execution, current in CELL_ORDER
}

CELL_LABELS = {
    (EXECUTION_KINEMATIC, CURRENT_NONE): "Kinematic\nNo current",
    (EXECUTION_KINEMATIC, CURRENT_VISIBLE): "Kinematic\nVisible current",
    (EXECUTION_DYNAMIC, CURRENT_NONE): "Delayed first-order\nNo current",
    (EXECUTION_DYNAMIC, CURRENT_VISIBLE): "Delayed first-order\nVisible current",
}
CELL_X_LABELS = {
    (EXECUTION_KINEMATIC, CURRENT_NONE): "Kinematic\nNo current",
    (EXECUTION_KINEMATIC, CURRENT_VISIBLE): "Kinematic\nVisible current",
    (EXECUTION_DYNAMIC, CURRENT_NONE): "Delayed\nfirst-order\nNo current",
    (EXECUTION_DYNAMIC, CURRENT_VISIBLE): "Delayed\nfirst-order\nVisible current",
}
POLICY_LABELS = {
    POLICY_FIXED: "Fixed S-turn",
    POLICY_ACTIVE: "Information-guided acquisition",
}

N_QUALIFICATION_EPISODES = 100
DEFAULT_DPI = 600
MINIMUM_DPI = 300
FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 4.15
DEFAULT_BASENAME = "fig_policy_execution_current"
DEFAULT_INPUT_NAME = "v40_publication_results.json"


class FigureInputError(ValueError):
    """Raised when the compact postprocessor output fails validation."""


@dataclass(frozen=True)
class ArmRates:
    policy: str
    execution: str
    current: str
    episodes: int
    terminal_rate: float
    tail80_rate: float


@dataclass(frozen=True)
class BinaryEffect:
    risk_difference: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class CellEffect:
    execution: str
    current: str
    pairs: int
    terminal: BinaryEffect
    tail80: BinaryEffect


@dataclass(frozen=True)
class FigureData:
    arms: tuple[ArmRates, ...]
    effects: tuple[CellEffect, ...]

    def arm(self, policy: str, execution: str, current: str) -> ArmRates:
        matches = tuple(
            arm
            for arm in self.arms
            if (arm.policy, arm.execution, arm.current)
            == (policy, execution, current)
        )
        if len(matches) != 1:
            raise FigureInputError(
                "validated figure data lost its unique policy/execution/current cell"
            )
        return matches[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureInputError(message)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureInputError(f"{label} is not numeric") from exc
    _require(math.isfinite(number), f"{label} is not finite")
    return number


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} is not an integer",
    )
    return int(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    _require(total > 0, "Wilson interval requires a positive sample size")
    _require(0 <= successes <= total, "Wilson successes are outside support")
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


def paired_newcombe_method10_interval(
    both_success: int,
    treatment_only: int,
    reference_only: int,
    neither_success: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float, float]:
    """Return treatment-minus-reference RD and paired Newcombe method-10 CI."""

    cells = (both_success, treatment_only, reference_only, neither_success)
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in cells),
        "paired cells must be integer counts",
    )
    _require(all(value >= 0 for value in cells), "paired cells must be nonnegative")
    total = sum(cells)
    _require(total > 0, "paired interval requires at least one pair")

    treatment_success = both_success + treatment_only
    reference_success = both_success + reference_only
    treatment_rate = treatment_success / total
    reference_rate = reference_success / total
    difference = treatment_rate - reference_rate
    treatment_lower, treatment_upper = _wilson_interval(treatment_success, total, z)
    reference_lower, reference_upper = _wilson_interval(reference_success, total, z)

    phi_denominator = math.sqrt(
        (both_success + treatment_only)
        * (reference_only + neither_success)
        * (both_success + reference_only)
        * (treatment_only + neither_success)
    )
    phi_numerator = both_success * neither_success - treatment_only * reference_only
    if phi_numerator > 0:
        phi_numerator = max(phi_numerator - total / 2.0, 0.0)
    phi = phi_numerator / phi_denominator if phi_denominator > 0.0 else 0.0
    phi = min(1.0, max(-1.0, phi))

    treatment_lower_half = treatment_rate - treatment_lower
    treatment_upper_half = treatment_upper - treatment_rate
    reference_lower_half = reference_rate - reference_lower
    reference_upper_half = reference_upper - reference_rate
    lower = difference - math.sqrt(
        max(
            0.0,
            treatment_lower_half**2
            - 2.0 * phi * treatment_lower_half * reference_upper_half
            + reference_upper_half**2,
        )
    )
    upper = difference + math.sqrt(
        max(
            0.0,
            treatment_upper_half**2
            - 2.0 * phi * treatment_upper_half * reference_lower_half
            + reference_lower_half**2,
        )
    )
    return difference, max(-1.0, lower), min(1.0, upper)


def _parse_arm(raw: Any, index: int) -> ArmRates:
    item = _mapping(raw, f"arms[{index}]")
    policy = item.get("policy")
    execution = item.get("execution")
    current = item.get("current")
    _require(policy in POLICY_ORDER, f"arms[{index}] has unsupported policy")
    _require(execution in EXECUTION_ORDER, f"arms[{index}] has unsupported execution")
    _require(current in CURRENT_ORDER, f"arms[{index}] has unsupported current")

    episodes = _integer(item.get("N"), f"arms[{index}].N")
    _require(
        episodes == N_QUALIFICATION_EPISODES,
        f"arms[{index}] must contain exactly {N_QUALIFICATION_EPISODES} episodes",
    )
    terminal_count = _integer(
        item.get("terminal_success_n"),
        f"arms[{index}].terminal_success_n",
    )
    tail80_count = _integer(item.get("Tail80_n"), f"arms[{index}].Tail80_n")
    _require(
        0 <= terminal_count <= episodes and 0 <= tail80_count <= episodes,
        f"arms[{index}] success counts are outside [0, N]",
    )
    terminal_rate = _finite(
        item.get("terminal_success_rate"),
        f"arms[{index}].terminal_success_rate",
    )
    tail80_rate = _finite(item.get("Tail80_rate"), f"arms[{index}].Tail80_rate")
    _require(
        math.isclose(
            terminal_rate,
            terminal_count / episodes,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        f"arms[{index}] terminal rate disagrees with its count",
    )
    _require(
        math.isclose(
            tail80_rate,
            tail80_count / episodes,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        f"arms[{index}] Tail80 rate disagrees with its count",
    )
    return ArmRates(
        policy=str(policy),
        execution=str(execution),
        current=str(current),
        episodes=episodes,
        terminal_rate=terminal_rate,
        tail80_rate=tail80_rate,
    )


def _parse_binary_effect(
    raw: Any,
    *,
    pairs: int,
    label: str,
) -> BinaryEffect:
    item = _mapping(raw, label)
    cell_names = (
        "concordant_success",
        "treatment_only",
        "reference_only",
        "concordant_failure",
    )
    counts = tuple(_integer(item.get(name), f"{label}.{name}") for name in cell_names)
    _require(all(count >= 0 for count in counts), f"{label} contains a negative count")
    _require(sum(counts) == pairs, f"{label} paired counts do not sum to pairs")

    calculated = paired_newcombe_method10_interval(*counts)
    risk_difference = _finite(item.get("risk_difference"), f"{label}.risk_difference")
    interval = item.get("paired_newcombe_method10_95")
    _require(
        isinstance(interval, (list, tuple)) and len(interval) == 2,
        f"{label}.paired_newcombe_method10_95 is not a two-element interval",
    )
    ci_lower = _finite(interval[0], f"{label}.CI lower")
    ci_upper = _finite(interval[1], f"{label}.CI upper")
    _require(ci_lower <= risk_difference <= ci_upper, f"{label} CI excludes its estimate")
    for observed, expected, component in zip(
        (risk_difference, ci_lower, ci_upper),
        calculated,
        ("risk difference", "CI lower", "CI upper"),
    ):
        _require(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12),
            f"{label} stored {component} disagrees with paired counts",
        )
    return BinaryEffect(risk_difference, ci_lower, ci_upper)


def load_publication_results(path: Path) -> FigureData:
    """Load and strictly validate the compact postprocessor JSON."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FigureInputError(f"cannot read publication JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FigureInputError(f"publication JSON is malformed: {path}") from exc

    root = _mapping(document, "publication JSON")
    _require(root.get("schema_version") == 1, "unsupported publication JSON schema")
    _require(root.get("integrity_valid") is True, "postprocessor integrity is not valid")
    raw_arms = root.get("arms")
    _require(isinstance(raw_arms, list), "arms is not a JSON array")
    _require(len(raw_arms) == 8, "publication JSON must contain exactly eight arms")
    arms = tuple(_parse_arm(raw, index) for index, raw in enumerate(raw_arms))
    observed_cells = {(arm.policy, arm.execution, arm.current) for arm in arms}
    _require(
        observed_cells == EXPECTED_CELLS and len(observed_cells) == len(arms),
        "arms do not form the exact 2 x 2 x 2 policy/execution/current design",
    )

    main_effects = _mapping(root.get("paired_main_effects"), "paired_main_effects")
    raw_contrasts = _mapping(
        main_effects.get("active_minus_fixed_within_execution_current"),
        "active-minus-fixed contrasts",
    )
    expected_names = {f"{execution}__{current}" for execution, current in CELL_ORDER}
    _require(
        set(raw_contrasts) == expected_names,
        "active-minus-fixed contrasts do not contain the exact four cells",
    )

    effects: list[CellEffect] = []
    for execution, current in CELL_ORDER:
        name = f"{execution}__{current}"
        contrast = _mapping(raw_contrasts[name], f"contrast {name}")
        _require(
            contrast.get("direction") == "treatment_minus_reference",
            f"contrast {name} has the wrong direction",
        )
        pairs = _integer(contrast.get("pairs"), f"contrast {name}.pairs")
        _require(
            pairs == N_QUALIFICATION_EPISODES,
            f"contrast {name} must contain exactly {N_QUALIFICATION_EPISODES} pairs",
        )
        terminal = _parse_binary_effect(
            contrast.get("terminal_joint_success"),
            pairs=pairs,
            label=f"contrast {name}.terminal_joint_success",
        )
        tail80 = _parse_binary_effect(
            contrast.get("tail80_joint_success"),
            pairs=pairs,
            label=f"contrast {name}.tail80_joint_success",
        )

        fixed = next(
            arm
            for arm in arms
            if (arm.policy, arm.execution, arm.current)
            == (POLICY_FIXED, execution, current)
        )
        active = next(
            arm
            for arm in arms
            if (arm.policy, arm.execution, arm.current)
            == (POLICY_ACTIVE, execution, current)
        )
        _require(
            math.isclose(
                active.terminal_rate - fixed.terminal_rate,
                terminal.risk_difference,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"contrast {name} terminal RD disagrees with arm marginals",
        )
        _require(
            math.isclose(
                active.tail80_rate - fixed.tail80_rate,
                tail80.risk_difference,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"contrast {name} Tail80 RD disagrees with arm marginals",
        )
        effects.append(CellEffect(execution, current, pairs, terminal, tail80))

    return FigureData(arms=arms, effects=tuple(effects))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.3,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _effect_axis_limit(effects: Sequence[CellEffect]) -> tuple[float, float, float]:
    endpoints = [0.0]
    for effect in effects:
        endpoints.extend(
            (
                effect.terminal.ci_lower,
                effect.terminal.ci_upper,
                effect.tail80.ci_lower,
                effect.tail80.ci_upper,
            )
        )
    maximum = max(abs(value) for value in endpoints)
    limit = min(1.0, max(0.10, math.ceil((maximum + 0.025) / 0.10) * 0.10))
    if limit <= 0.40:
        step = 0.10
    elif limit <= 0.80:
        step = 0.20
    else:
        step = 0.25
    return -limit, limit, step


def build_figure(data: FigureData) -> Figure:
    """Build the two-panel monochrome figure without reading further files."""

    configure_style()
    figure, (absolute_axis, effect_axis) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        gridspec_kw={"width_ratios": (1.30, 1.00)},
    )
    figure.subplots_adjust(
        left=0.080,
        right=0.985,
        top=0.880,
        bottom=0.305,
        wspace=0.500,
    )

    cell_positions = tuple(float(index) for index in range(len(CELL_ORDER)))
    policy_offsets = {POLICY_FIXED: -0.17, POLICY_ACTIVE: 0.17}
    metric_offsets = {"terminal": -0.026, "tail80": 0.026}
    marker_specs = {
        "terminal": ("o", "Terminal success"),
        "tail80": ("s", "Tail80 success"),
    }
    for cell_position, (execution, current) in zip(cell_positions, CELL_ORDER):
        for policy in POLICY_ORDER:
            arm = data.arm(policy, execution, current)
            center = cell_position + policy_offsets[policy]
            values = {
                "terminal": arm.terminal_rate,
                "tail80": arm.tail80_rate,
            }
            absolute_axis.plot(
                [center + metric_offsets["terminal"], center + metric_offsets["tail80"]],
                [values["terminal"], values["tail80"]],
                color="0.55",
                linewidth=0.65,
                zorder=2,
            )
            for metric, value in values.items():
                marker, _ = marker_specs[metric]
                absolute_axis.plot(
                    center + metric_offsets[metric],
                    value,
                    linestyle="none",
                    marker=marker,
                    markersize=5.2,
                    markeredgewidth=0.85,
                    markeredgecolor="black",
                    markerfacecolor=("white" if policy == POLICY_FIXED else "black"),
                    zorder=4,
                )

    absolute_axis.set_title("(a) Absolute success rates (eight arms)", loc="left")
    absolute_axis.set_ylabel("Episode success rate")
    absolute_axis.set_xticks(cell_positions, [CELL_X_LABELS[cell] for cell in CELL_ORDER])
    absolute_axis.set_xlim(-0.52, len(CELL_ORDER) - 0.48)
    absolute_axis.set_ylim(-0.015, 1.035)
    absolute_axis.yaxis.set_major_locator(MultipleLocator(0.20))
    absolute_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    absolute_axis.grid(axis="y", color="0.86", linewidth=0.55, zorder=0)
    absolute_axis.tick_params(direction="out", length=2.5, width=0.65)

    effect_by_cell = {
        (effect.execution, effect.current): effect for effect in data.effects
    }
    y_positions = tuple(float(index) for index in range(len(CELL_ORDER)))
    endpoint_specs = (
        ("terminal", -0.12, "o", "black"),
        ("tail80", 0.12, "s", "0.45"),
    )
    for y_position, cell in zip(y_positions, CELL_ORDER):
        effect = effect_by_cell[cell]
        for endpoint, offset, marker, facecolor in endpoint_specs:
            value = getattr(effect, endpoint)
            effect_axis.errorbar(
                value.risk_difference,
                y_position + offset,
                xerr=(
                    [value.risk_difference - value.ci_lower],
                    [value.ci_upper - value.risk_difference],
                ),
                fmt=marker,
                markersize=5.0,
                markeredgewidth=0.80,
                markeredgecolor="black",
                markerfacecolor=facecolor,
                ecolor=("black" if endpoint == "terminal" else "0.45"),
                elinewidth=0.90,
                capsize=2.4,
                capthick=0.80,
                zorder=3,
            )

    effect_axis.axvline(0.0, color="0.25", linestyle=(0, (3.0, 2.0)), linewidth=0.75)
    effect_axis.set_title("(b) Information-guided − fixed S-turn", loc="left")
    effect_axis.set_xlabel("Risk difference [percentage points]")
    effect_axis.set_yticks(y_positions, [CELL_LABELS[cell] for cell in CELL_ORDER])
    effect_axis.set_ylim(len(CELL_ORDER) - 0.55, -0.45)
    lower, upper, step = _effect_axis_limit(data.effects)
    effect_axis.set_xlim(lower, upper)
    effect_axis.xaxis.set_major_locator(MultipleLocator(step))
    effect_axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{100.0 * value:.0f}")
    )
    effect_axis.grid(axis="x", color="0.86", linewidth=0.55, zorder=0)
    effect_axis.tick_params(direction="out", length=2.5, width=0.65)

    legend_handles = (
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=5.2,
            markeredgecolor="black",
            markerfacecolor="0.45",
            label="Terminal success (circles)",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=5.2,
            markeredgecolor="black",
            markerfacecolor="0.45",
            label="Tail80 success (squares)",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.7,
            markeredgecolor="black",
            markerfacecolor="white",
            label=f"{POLICY_LABELS[POLICY_FIXED]} (open markers)",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.7,
            markeredgecolor="black",
            markerfacecolor="black",
            label=f"{POLICY_LABELS[POLICY_ACTIVE]} (filled markers)",
        ),
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.15,
        borderaxespad=0.0,
    )
    return figure


def write_figure(
    data: FigureData,
    output_directory: Path,
    *,
    basename: str = DEFAULT_BASENAME,
    dpi: int = DEFAULT_DPI,
) -> dict[str, Path]:
    """Write a vector PDF and a publication-resolution monochrome PNG."""

    _require(
        isinstance(dpi, int) and not isinstance(dpi, bool) and dpi >= MINIMUM_DPI,
        f"PNG resolution must be at least {MINIMUM_DPI} dpi",
    )
    _require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", basename)),
        "basename must be a plain filename without a path separator",
    )
    destination = output_directory.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": destination / f"{basename}.pdf",
        "png": destination / f"{basename}_{dpi}dpi.png",
    }
    figure = build_figure(data)
    try:
        figure.savefig(
            outputs["pdf"],
            format="pdf",
            facecolor="white",
            metadata={
                "Title": "Policy, execution, and current qualification",
                "Subject": "Absolute success rates and paired risk differences",
                "Creator": "plot_v40_qualification_figure.py",
            },
        )
        figure.savefig(
            outputs["png"],
            format="png",
            dpi=dpi,
            facecolor="white",
            metadata={"Software": "plot_v40_qualification_figure.py"},
        )
    finally:
        plt.close(figure)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "publication_json",
        nargs="?",
        type=Path,
        default=repository / "data" / "tables" / DEFAULT_INPUT_NAME,
        help=(
            "Compact JSON written by postprocess_v40_qualification.py "
            f"(default: data/tables/{DEFAULT_INPUT_NAME})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "results" / "figures",
        help="Directory for the PDF and PNG (default: results/figures).",
    )
    parser.add_argument(
        "--basename",
        default=DEFAULT_BASENAME,
        help=f"Output basename without extension (default: {DEFAULT_BASENAME}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG resolution, at least {MINIMUM_DPI} dpi (default: {DEFAULT_DPI}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        data = load_publication_results(arguments.publication_json)
        outputs = write_figure(
            data,
            arguments.output_dir,
            basename=arguments.basename,
            dpi=arguments.dpi,
        )
    except FigureInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for kind, path in outputs.items():
        print(f"{kind.upper()}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
