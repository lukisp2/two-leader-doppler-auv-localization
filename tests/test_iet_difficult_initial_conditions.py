"""Small protocol and integrity tests; no simulated episodes are executed."""
import copy
from dataclasses import asdict
import unittest
import numpy as np
import run_iet_difficult_initial_conditions as run


class ProtocolTests(unittest.TestCase):
    def test_sealed_range_always_rejected(self):
        for seed in (50000,50500,50999):
            for smoke in (False,True):
                with self.assertRaises(PermissionError):
                    run.assert_allowed(seed,[seed],smoke)

    def test_new_seed_must_be_selected(self):
        run.assert_allowed(52001,[52001])
        for seed in (52002,51000,53000):
            with self.assertRaises(PermissionError):
                run.assert_allowed(seed,[52001])
        with self.assertRaises(PermissionError):
            run.assert_allowed(52001,[52001],True)
        run.assert_allowed(49566,[],True)

    def test_seed_scope_restores_historical_guard(self):
        original=run.inherited.v41._assert_seed_allowed_automatic
        with run.seed_scope([52001]):
            run.inherited.v41._assert_seed_allowed_automatic(52001)
            with self.assertRaises(PermissionError):
                run.inherited.v41._assert_seed_allowed_automatic(50000)
        self.assertIs(run.inherited.v41._assert_seed_allowed_automatic,original)
        with self.assertRaises(PermissionError):
            original(52001)

    def test_frozen_config_and_reproducible_initial_geometry(self):
        _,cfg=run.load_config()
        before=copy.deepcopy(asdict(cfg))
        self.assertEqual((cfg.start_rho_min,cfg.start_rho_max,cfg.max_steps,cfg.action_dt),
                         (120.0,350.0,220,2.0))
        env=run.inherited.v40.DynamicCurrentEnv(cfg=cfg,render_mode="none",
                dynamics=run.inherited.v40.DYNAMICS_LOW_ORDER,
                current=run.inherited.v40.CURRENT_NONE,episode_seed=52001)
        try:
            a=run.initial_record(env,52001)
            b=run.initial_record(env,52001)
        finally:
            env.close()
        self.assertEqual(a,b)
        self.assertEqual(before,asdict(cfg))
        self.assertEqual(a["eligible"],a["initial_formation_error_m"]>=350 and a["initial_heading_error_deg"]>=120)
        self.assertAlmostEqual(a["initial_formation_error_m"],float(np.linalg.norm(
            np.asarray(a["initial_position_m"])-np.asarray(a["desired_position_m"]))))

    def test_json_seed_audit_handles_lists(self):
        self.assertEqual(run._json_seeds({"seeds":[52001,52002],"nested":{"episode_seed":52003}},
                                        {52001,52003}),{52001,52003})


if __name__ == "__main__":
    unittest.main()
