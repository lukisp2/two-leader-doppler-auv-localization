from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
FIGURE_SCRIPTS = (
    "make_publication_figure_paired_episode.py",
    "make_publication_figure_mission_geometry.py",
    "make_publication_figure_doppler_geometry.py",
    "make_publication_figure_full_history_estimator.py",
    "make_publication_figure_source_policy_dynamics.py",
    "make_publication_figure_source_policy_ablation.py",
    "make_publication_figure_policy_execution_current.py",
    "make_publication_figures_v33.py",
)


@pytest.fixture(scope="module")
def figure_environment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path_factory.mktemp("matplotlib"))
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(CODE), environment.get("PYTHONPATH", "")))
    )
    return environment


def run_figure_script(
    script: str,
    *arguments: str,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CODE / script), *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("script", FIGURE_SCRIPTS)
def test_figure_script_help_is_data_independent(
    script: str,
    figure_environment: dict[str, str],
) -> None:
    completed = run_figure_script(script, "--help", environment=figure_environment)
    assert "usage:" in completed.stdout.lower()


def test_data_free_explanatory_figures_render(
    tmp_path: Path,
    figure_environment: dict[str, str],
) -> None:
    doppler = tmp_path / "doppler.pdf"
    run_figure_script(
        "make_publication_figure_doppler_geometry.py",
        "--output",
        str(doppler),
        environment=figure_environment,
    )
    assert doppler.stat().st_size > 10_000

    estimator_dir = tmp_path / "estimator"
    run_figure_script(
        "make_publication_figure_full_history_estimator.py",
        "--output-dir",
        str(estimator_dir),
        environment=figure_environment,
    )
    expected = (
        "figure4_full_history_estimator.pdf",
        "figure4_full_history_estimator.svg",
        "figure4_full_history_estimator_220dpi.png",
        "figure4_full_history_estimator_360dpi.png",
    )
    for name in expected:
        assert (estimator_dir / name).stat().st_size > 10_000


def test_compact_quantitative_inputs_render_current_figures(
    tmp_path: Path,
    figure_environment: dict[str, str],
) -> None:
    source_policy = tmp_path / "fig_source_policy_ablation.pdf"
    run_figure_script(
        "make_publication_figure_source_policy_ablation.py",
        "--output",
        str(source_policy),
        environment=figure_environment,
    )
    assert source_policy.stat().st_size > 10_000

    run_figure_script(
        "make_publication_figures_v33.py",
        "--output-dir",
        str(tmp_path),
        environment=figure_environment,
    )
    assert (tmp_path / "fig_estimator_history.pdf").stat().st_size > 10_000
    assert not (tmp_path / "fig_closed_loop_stress.pdf").exists()

    execution_current = tmp_path / "fig_policy_execution_current.pdf"
    run_figure_script(
        "make_publication_figure_policy_execution_current.py",
        "--output-pdf",
        str(execution_current),
        "--no-png",
        environment=figure_environment,
    )
    assert execution_current.stat().st_size > 10_000

    run_figure_script(
        "make_publication_figures_v33.py",
        "--figure",
        "closed-loop-stress",
        "--output-dir",
        str(tmp_path),
        "--allow-invalid-descriptive",
        environment=figure_environment,
    )
    assert (tmp_path / "fig_closed_loop_stress.pdf").stat().st_size > 10_000


def test_closed_loop_stress_figure_fails_closed_without_acknowledgement(
    tmp_path: Path,
    figure_environment: dict[str, str],
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CODE / "make_publication_figures_v33.py"),
            "--figure",
            "closed-loop-stress",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=figure_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode != 0
    assert "--allow-invalid-descriptive" in completed.stderr
    assert not (tmp_path / "fig_closed_loop_stress.pdf").exists()


def test_mission_geometry_frame_transform_on_synthetic_trace() -> None:
    sys.path.insert(0, str(CODE))
    try:
        mission = importlib.import_module("make_publication_figure_mission_geometry")
    finally:
        sys.path.pop(0)

    time_s = np.asarray([0.0, 1.0, 2.0])
    centroid = np.column_stack((time_s, np.zeros(3), np.zeros(3)))
    leader_positions = np.stack(
        (
            centroid + np.asarray([0.0, -10.0, -5.0]),
            centroid + np.asarray([0.0, 10.0, 5.0]),
        ),
        axis=1,
    )
    leader_velocities = np.broadcast_to(
        np.asarray([1.0, 0.0, 0.0]),
        leader_positions.shape,
    ).copy()
    truth = centroid + np.asarray([-120.0, 5.0, 2.0])
    estimate = centroid + np.asarray([-119.0, 4.0, 1.0])
    trace = {
        "time_s": time_s,
        "online_t_s": time_s,
        "online_leader_position_m": leader_positions,
        "online_leader_velocity_mps": leader_velocities,
        "truth_x": truth[:, 0],
        "truth_y": truth[:, 1],
        "truth_z": truth[:, 2],
        "estimate_x": estimate[:, 0],
        "estimate_y": estimate[:, 1],
        "estimate_z": estimate[:, 2],
    }

    _, truth_frame, estimate_frame, leader_frame, desired, basis = (
        mission._formation_coordinates(trace)
    )
    np.testing.assert_allclose(basis, np.diag([1.0, -1.0, 1.0]))
    np.testing.assert_allclose(truth_frame, [[-120.0, -5.0, 2.0]] * 3)
    np.testing.assert_allclose(estimate_frame, [[-119.0, -4.0, 1.0]] * 3)
    np.testing.assert_allclose(leader_frame[:, 0], [[0.0, 10.0, -5.0]] * 3)
    np.testing.assert_allclose(leader_frame[:, 1], [[0.0, -10.0, 5.0]] * 3)
    np.testing.assert_allclose(desired, [[-120.0, 0.0, 0.0]] * 3)
