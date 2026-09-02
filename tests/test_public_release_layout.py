from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"

MIRRORED_TESTS = (
    "test_delay_aware_formation_tracker.py",
    "test_run_v41_controller_repair.py",
    "test_uuv_v27_publication_baselines.py",
    "test_uuv_v28_estimator_stress.py",
    "test_uuv_v34_mhe60_baseline.py",
    "test_uuv_v35_closed_loop_stress.py",
    "test_uuv_v39_planner_component_ablation.py",
    "test_uuv_v40_dynamic_plant_stress.py",
    "test_uuv_v41_controller_repair.py",
)


@pytest.mark.parametrize("name", MIRRORED_TESTS)
def test_frozen_runner_test_mirror_is_byte_identical(name: str) -> None:
    canonical = ROOT / "tests" / name
    mirror = CODE / "tests" / name
    assert canonical.is_file()
    assert mirror.is_file()
    assert mirror.read_bytes() == canonical.read_bytes()


@pytest.mark.parametrize(
    ("module_name", "manifest_function", "expected_key"),
    (
        (
            "run_v27_publication_baselines",
            "_source_manifest",
            "tests/test_uuv_v27_publication_baselines.py",
        ),
        (
            "run_v28_estimator_stress",
            "_source_manifest",
            "tests/test_uuv_v28_estimator_stress.py",
        ),
        (
            "run_v34_mhe60_baseline",
            "_source_manifest",
            "tests/test_uuv_v34_mhe60_baseline.py",
        ),
        (
            "run_v35_closed_loop_stress",
            "_loaded_local_sources",
            "tests/test_uuv_v35_closed_loop_stress.py",
        ),
        (
            "run_v39_planner_component_ablation",
            "_loaded_local_sources",
            "tests/test_uuv_v39_planner_component_ablation.py",
        ),
        (
            "run_v40_dynamic_plant_stress",
            "_source_hashes",
            "tests/test_uuv_v40_dynamic_plant_stress.py",
        ),
    ),
)
def test_frozen_runner_source_preflight_accepts_public_layout(
    module_name: str,
    manifest_function: str,
    expected_key: str,
) -> None:
    sys.path.insert(0, str(CODE))
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, manifest_function)
        manifest = function(CODE, CODE) if module_name == "run_v40_dynamic_plant_stress" else function(CODE)
    finally:
        sys.path.remove(str(CODE))
    assert expected_key in manifest


def test_v41_runner_source_preflight_accepts_public_layout() -> None:
    sys.path.insert(0, str(CODE))
    try:
        module = importlib.import_module("run_v41_controller_repair")
        manifest = module._source_hashes(CODE, CODE)
    finally:
        sys.path.remove(str(CODE))
    assert "delay_aware_formation_tracker.py" in manifest
    assert "uuv_v41_controller_repair.py" in manifest
    assert "run_v41_controller_repair.py" in manifest
    assert "protocols/EXPERIMENT_PROTOCOL_V41_CONTROLLER_REPAIR.md" in manifest
