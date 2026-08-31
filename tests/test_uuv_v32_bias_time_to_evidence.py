from __future__ import annotations

import inspect
import unittest

import numpy as np

import uuv_v27_publication_baselines as v27
import uuv_v31_bias_evidence_gate as v31
import uuv_v32_bias_time_to_evidence as v32
from tests.test_uuv_v27_publication_baselines import synthetic_history
from tests.test_uuv_v30_profiled_bias import with_bias


class V32BiasTimeSweepTests(unittest.TestCase):
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

    def test_api_has_no_truth_or_seed(self):
        parameters = inspect.signature(v32.evaluate_checkpoint).parameters
        self.assertNotIn("truth", parameters)
        self.assertNotIn("seed", parameters)
        self.assertNotIn("episode_seed", parameters)

    def test_120s_evidence_exactly_reproduces_v31(self):
        history = with_bias(self.history, np.array([0.03, -0.03]))
        expected = v31.evaluate_evidence(
            history, 120.0, 120.0, 350.0, self.config
        )
        actual = v32.evaluate_checkpoint(
            history, 120.0, 120.0, 350.0, self.config
        ).evidence
        self.assertEqual(
            v31.deterministic_evidence(expected),
            v31.deterministic_evidence(actual),
        )

    def test_all_frozen_checkpoints_are_finite_and_causal(self):
        history = with_bias(self.history, np.array([0.03, 0.03]))
        for checkpoint in v32.CHECKPOINTS_S:
            with self.subTest(checkpoint=checkpoint):
                result = v32.evaluate_checkpoint(
                    history, checkpoint, 120.0, 350.0, self.config
                )
                self.assertEqual(result.evidence.checkpoint_s, checkpoint)
                self.assertGreater(result.evidence.validation_count, 2)
                self.assertTrue(np.isfinite(result.total_runtime_s))
                self.assertTrue(result.evidence.activate_bias_model)

    def test_nominal_synthetic_history_never_activates(self):
        for checkpoint in v32.CHECKPOINTS_S:
            result = v32.evaluate_checkpoint(
                self.history, checkpoint, 120.0, 350.0, self.config
            )
            self.assertFalse(result.evidence.activate_bias_model)


if __name__ == "__main__":
    unittest.main()
