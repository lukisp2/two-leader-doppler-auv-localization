import unittest

import numpy as np

from uuv_v18_resampling_guard import UUV3DConfig
import uuv_v28_estimator_stress as v28
import uuv_v30_profiled_bias as v30
import uuv_v35_closed_loop_stress as v35


class _FakePF:
    def __init__(self) -> None:
        self.predict_calls = []
        self.update_calls = []

    def predict(self, vF_meas, dt):
        self.predict_calls.append((np.asarray(vF_meas).copy(), float(dt)))
        return None

    def update_doppler(self, pL_list, vL_list, vF_meas, s_meas_list, **kwargs):
        self.update_calls.append((pL_list, vL_list, vF_meas, s_meas_list, kwargs))
        return None


class _FakeEnv:
    def __init__(self) -> None:
        self.cfg = UUV3DConfig()
        self.pf = _FakePF()
        self.next_s_time = 1.0
        self.pL1 = np.asarray([0.0, 50.0, -30.0])
        self.pL2 = np.asarray([0.0, -50.0, -70.0])
        self.vL1 = np.asarray([2.0, 0.0, 0.0])
        self.vL2 = np.asarray([2.0, 0.0, 0.0])
        self._v11_speed_meas = 2.0
        self._v11_yaw_meas = 10.0
        self._v11_pitch_meas = -2.0
        self.step_count = 0


def _capture_row(recorder, env, row, doppler=(1.0, -1.0)):
    env.next_s_time = float(row + 1)
    positions = [
        np.asarray([float(row + 1), 50.0, -30.0]),
        np.asarray([float(row + 1), -50.0, -70.0]),
    ]
    velocities = [np.asarray([2.0, 0.0, 0.0]), np.asarray([2.0, 0.0, 0.0])]
    env.pf.predict(np.asarray([1.0, 0.0, 0.0]), 1.0)
    env.pf.update_doppler(
        positions,
        velocities,
        np.asarray([1.0, 0.0, 0.0]),
        list(doppler),
        gate_factors=[1.0, 1.0],
    )


class V35StressTests(unittest.TestCase):
    def test_arm_condition_contract_has_1200_full_runs(self):
        self.assertEqual(len(v35.CONDITIONS), 8)
        self.assertEqual(len(v35.arm_condition_pairs()), 12)
        self.assertEqual(100 * len(v35.arm_condition_pairs()), 1200)

    def test_reserved_and_final_seeds_are_rejected(self):
        v35.assert_v35_seed_allowed(48_100)
        v35.assert_v35_seed_allowed(49_591)
        for seed in (49_900, 49_999, 50_000, 50_999):
            with self.assertRaises(PermissionError):
                v35.assert_v35_seed_allowed(seed)

    def test_stress_tape_is_deterministic_and_arm_independent(self):
        left = v35.make_stress_tape(v28.COLORED_NOISE, 17)
        right = v35.make_stress_tape(v28.COLORED_NOISE, 17)
        self.assertEqual(left.content_sha256(), right.content_sha256())
        np.testing.assert_array_equal(left.colored_noise_mps, right.colored_noise_mps)
        self.assertEqual(left.colored_noise_mps.shape, (440, 2))

    def test_common_and_differential_bias_transform(self):
        for condition, expected in (
            (v28.DOPPLER_COMMON_BIAS, np.asarray([1.03, -0.97])),
            (v28.DOPPLER_DIFFERENTIAL_BIAS, np.asarray([1.03, -1.03])),
        ):
            env = _FakeEnv()
            recorder = v35.CausalStressRecorder(
                env, v35.make_stress_tape(condition, 0)
            )
            try:
                _capture_row(recorder, env, 0)
                np.testing.assert_allclose(
                    recorder.history().doppler_measured_mps[0], expected
                )
                # The excluded legacy PF still receives the nominal row.
                self.assertEqual(env.pf.update_calls[0][3], [1.0, -1.0])
            finally:
                recorder.restore()

    def test_doppler_scale_and_dr_scale_transform(self):
        env = _FakeEnv()
        recorder = v35.CausalStressRecorder(
            env, v35.make_stress_tape(v28.DOPPLER_SCALE, 0)
        )
        try:
            _capture_row(recorder, env, 0)
            np.testing.assert_allclose(
                recorder.history().doppler_measured_mps[0], [1.02, -1.02]
            )
        finally:
            recorder.restore()

        env = _FakeEnv()
        recorder = v35.CausalStressRecorder(
            env, v35.make_stress_tape(v28.DR_SCALE, 0)
        )
        try:
            _capture_row(recorder, env, 0)
            np.testing.assert_allclose(
                recorder.accumulated_dead_reckoning_m, [1.01, 0.0, 0.0]
            )
            self.assertAlmostEqual(recorder.online_view()._v11_speed_meas, 2.02)
        finally:
            recorder.restore()

    def test_dropout_matches_forced_v28_rows(self):
        tape = v35.make_stress_tape(v28.DROPOUT, 4)
        self.assertTrue(tape.dropout_keep[0])
        self.assertTrue(tape.dropout_keep[119])
        self.assertTrue(tape.dropout_keep[-1])
        self.assertLess(int(np.sum(tape.dropout_keep)), 440)

    def test_two_second_broadcast_delay(self):
        env = _FakeEnv()
        recorder = v35.CausalStressRecorder(
            env, v35.make_stress_tape(v28.BROADCAST_DELAY, 0)
        )
        try:
            for row in range(4):
                _capture_row(recorder, env, row)
            history = recorder.history()
            # At t=4 the newest admissible broadcast is t=2.
            self.assertAlmostEqual(history.leader_position_m[-1, 0, 0], 2.0)
            self.assertAlmostEqual(recorder.online_view().pL1[0], 2.0)
        finally:
            recorder.restore()

    def test_profile_mode_adapter_preserves_a_valid_covariance(self):
        mode = v30.ProfiledBiasMode(
            initial_position_m=np.asarray([1.0, 2.0, 3.0]),
            link_bias_mps=np.asarray([0.03, -0.03]),
            residual_sse_mps2=0.5,
            residual_rmse_mps=0.04,
            iterations=4,
            converged=True,
            position_covariance_m2=np.diag([1.0, 2.0, 3.0]),
            nominal_radius95_m=4.0,
            link_bias_standard_error_mps=np.asarray([0.001, 0.001]),
        )
        adapted = v35._profile_mode_to_batch(mode)
        self.assertTrue(adapted.local_covariance_valid)
        self.assertEqual(adapted.hessian_rank, 3)
        self.assertAlmostEqual(adapted.local_radius95_m, 4.0)


if __name__ == "__main__":
    unittest.main()

