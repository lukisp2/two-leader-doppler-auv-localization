from __future__ import annotations

import inspect
import unittest

import numpy as np

import uuv_v19_observability as v19
import uuv_v27_publication_baselines as v27
import uuv_v31_bias_evidence_gate as v31
from tests.test_uuv_v27_publication_baselines import synthetic_history
from tests.test_uuv_v30_profiled_bias import with_bias


class V31BiasEvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.history, _, _ = synthetic_history(440)
        self.config = v27.EvaluatorConfig(
            coarse_candidates=512,
            coarse_sweeps=1,
            local_starts=8,
            maximum_modes=8,
            pf_particles_small=256,
            pf_particles_large=512,
        )

    def test_gate_api_has_no_truth_or_seed_parameter(self):
        parameters = inspect.signature(v31.evaluate_evidence).parameters
        self.assertNotIn("truth", parameters)
        self.assertNotIn("episode_seed", parameters)
        self.assertNotIn("seed", parameters)

    def test_noiseless_nominal_does_not_activate(self):
        result = v31.evaluate_evidence(
            self.history, 120.0, 120.0, 350.0, self.config
        )
        self.assertFalse(result.activate_bias_model)
        self.assertFalse(result.material_training_bias)

    def test_exact_common_and_differential_biases_activate(self):
        for bias in (np.array([0.03, 0.03]), np.array([0.03, -0.03])):
            with self.subTest(bias=bias.tolist()):
                result = v31.evaluate_evidence(
                    with_bias(self.history, bias),
                    120.0,
                    120.0,
                    350.0,
                    self.config,
                )
                self.assertTrue(result.activate_bias_model)
                self.assertGreaterEqual(result.relative_sse_gain, 0.10)

    def test_future_measurements_cannot_change_120s_decision(self):
        first = v31.evaluate_evidence(
            with_bias(self.history, np.array([0.03, -0.03])),
            120.0,
            120.0,
            350.0,
            self.config,
        )
        changed_measurements = self.history.doppler_measured_mps.copy()
        changed_measurements[self.history.t_s > 120.0] += 100.0
        changed = v19.OnlineDopplerHistory(
            t_s=self.history.t_s.copy(),
            dead_reckoned_displacement_m=self.history.dead_reckoned_displacement_m.copy(),
            leader_position_m=self.history.leader_position_m.copy(),
            leader_velocity_mps=self.history.leader_velocity_mps.copy(),
            follower_velocity_measured_mps=self.history.follower_velocity_measured_mps.copy(),
            doppler_measured_mps=changed_measurements,
            historical_pf_gate_factor=self.history.historical_pf_gate_factor.copy(),
        )
        changed = with_bias(changed, np.array([0.03, -0.03]))
        second = v31.evaluate_evidence(
            changed, 120.0, 120.0, 350.0, self.config
        )
        self.assertEqual(
            v31.deterministic_evidence(first),
            v31.deterministic_evidence(second),
        )

    def test_evidence_is_deterministic(self):
        history = with_bias(self.history, np.array([0.03, 0.03]))
        first = v31.evaluate_evidence(
            history, 440.0, 120.0, 350.0, self.config
        )
        second = v31.evaluate_evidence(
            history, 440.0, 120.0, 350.0, self.config
        )
        self.assertEqual(
            v31.deterministic_evidence(first),
            v31.deterministic_evidence(second),
        )


if __name__ == "__main__":
    unittest.main()
