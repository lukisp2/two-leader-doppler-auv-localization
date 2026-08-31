#!/usr/bin/env python3
"""Tests for the independent V38 vertical-datum audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_v38_vertical_datum as audit
import run_v20_positioning_ablation as runner20
import uuv_v38_leader_source_ablation as v38


class VerticalDatumAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.cfg = runner20._load_environment_config(
            runner20._default_metadata(root)
        )

    def _synthetic_campaign(
        self,
        root: Path,
        *,
        follower_z: np.ndarray,
        leader_z: np.ndarray,
    ) -> Path:
        campaign = root / "campaign"
        arms = [v38.arm_name(source, policy) for source, policy in v38.arm_pairs()]
        contract = {
            "episode_start": 0,
            "episodes": 1,
            "expected_runs": 6,
            "seeds": [48_700],
            "arms": [{"arm": arm} for arm in arms],
        }
        progress = {
            "status": "complete",
            "completed_runs": 6,
            "total_runs": 6,
        }
        control = campaign / "control"
        control.mkdir(parents=True)
        (control / "campaign_contract.json").write_text(
            json.dumps(contract),
            encoding="utf-8",
        )
        (control / "progress.json").write_text(
            json.dumps(progress),
            encoding="utf-8",
        )
        for arm in arms:
            result_dir = campaign / "episode_results"
            trace_dir = campaign / "traces_npz" / arm
            result_dir.mkdir(parents=True, exist_ok=True)
            trace_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "episode_index": 0,
                "episode_seed": 48_700,
                "arm": arm,
                "initial_truth_m": [1.0, 2.0, float(follower_z[0])],
                "mission_support": {
                    "center_m": [0.0, 0.0, -50.0],
                    "radius_min_m": 120.0,
                    "radius_max_m": 350.0,
                },
            }
            (result_dir / f"episode_0000_seed_48700_{arm}.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            leaders = np.zeros((leader_z.shape[0], 2, 3), dtype=np.float64)
            leaders[:, :, 2] = leader_z
            np.savez_compressed(
                trace_dir / "episode_0000_seed_48700.npz",
                truth_z=np.asarray(follower_z[1:], dtype=np.float64),
                online_leader_position_m=leaders,
                time_s=2.0
                * np.arange(1, follower_z.size, dtype=np.float64),
            )
        return campaign

    def test_physical_z_preserves_relative_depth(self) -> None:
        local = np.asarray([-400.0, -50.0, 250.0])
        physical = audit.physical_z(local, -50.0)
        np.testing.assert_allclose(
            physical,
            np.asarray([-1350.0, -1000.0, -700.0]),
        )
        np.testing.assert_allclose(
            np.diff(physical),
            np.diff(local),
        )

    def test_complete_synthetic_campaign_passes_clearance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._synthetic_campaign(
                Path(tmp),
                follower_z=np.asarray([-350.0, -200.0, 200.0]),
                leader_z=np.asarray(
                    [[-40.0, -80.0], [-40.0, -80.0]],
                ),
            )
            result = audit.audit_campaign_clearance(
                campaign,
                expected_runs=6,
                expected_runs_per_arm=1,
            )
        self.assertTrue(result["valid"])
        self.assertEqual(result["run_count"], 6)
        self.assertEqual(set(result["per_arm"]), set(audit.EXPECTED_ARMS))
        overall = result["overall"]["all_vehicles"]
        self.assertAlmostEqual(overall["minimum_z_phys_m"], -1300.0)
        self.assertAlmostEqual(overall["maximum_z_phys_m"], -750.0)
        self.assertGreaterEqual(overall["minimum_clearance_m"], 100.0)

    def test_clearance_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._synthetic_campaign(
                Path(tmp),
                follower_z=np.asarray([-50.0, 920.0, 950.0]),
                leader_z=np.asarray(
                    [[-40.0, -80.0], [-40.0, -80.0]],
                ),
            )
            result = audit.audit_campaign_clearance(
                campaign,
                expected_runs=6,
                expected_runs_per_arm=1,
            )
        self.assertFalse(result["valid"])
        self.assertLess(
            result["overall"]["all_vehicles"]["surface_clearance_m"],
            100.0,
        )

    def test_incomplete_campaign_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._synthetic_campaign(
                Path(tmp),
                follower_z=np.asarray([-50.0, -40.0]),
                leader_z=np.asarray([[-40.0, -80.0]]),
            )
            progress = campaign / "control" / "progress.json"
            progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "completed_runs": 5,
                        "total_runs": 6,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "refused"):
                audit.audit_campaign_clearance(
                    campaign,
                    expected_runs=6,
                    expected_runs_per_arm=1,
                )

    def test_plant_translation_invariance(self) -> None:
        result = audit.run_plant_invariance_smoke(
            self.cfg,
            seed=48_982,
            steps=5,
        )
        self.assertTrue(result["valid"], result)

    def test_full_active_episode_translation_invariance_debug_settings(self) -> None:
        original_class = v38.UUVTwoLeader3DPFEnv
        result = audit.run_closed_loop_invariance_smoke(
            self.cfg,
            seed=48_983,
            coarse_candidates=64,
            coarse_sweeps=1,
            local_starts=6,
        )
        self.assertIs(v38.UUVTwoLeader3DPFEnv, original_class)
        self.assertTrue(result["full_horizon_completed"])
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["publication_estimator_settings"])


if __name__ == "__main__":
    unittest.main()
