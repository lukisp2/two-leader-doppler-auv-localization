from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from uuv_v11_rng import (
    ALL_STREAM_NAMES,
    EpisodeRNGStreams,
    EpisodeSeedPlan,
    ExogenousNoiseCursor,
    ExogenousNoiseTape,
)


class EpisodeRNGStreamsTests(unittest.TestCase):
    def test_same_plan_replays_every_stream(self) -> None:
        plan = EpisodeSeedPlan(root_seed=42, episode_index=17, env_rank=3)
        first = EpisodeRNGStreams(plan, controller_id="rl")
        second = EpisodeRNGStreams(plan, controller_id="rl")

        for stream_name in ALL_STREAM_NAMES:
            np.testing.assert_array_equal(
                first.generator(stream_name).standard_normal(128),
                second.generator(stream_name).standard_normal(128),
            )

    def test_named_streams_do_not_share_the_same_sequence(self) -> None:
        streams = EpisodeRNGStreams(
            EpisodeSeedPlan(root_seed=42, episode_index=17), controller_id="rl"
        )
        draws = {
            name: streams.generator(name).bytes(64) for name in ALL_STREAM_NAMES
        }
        self.assertEqual(len(set(draws.values())), len(ALL_STREAM_NAMES))

    def test_streams_are_order_independent_and_isolated(self) -> None:
        plan = EpisodeSeedPlan(root_seed=991, episode_index=8)
        branchy = EpisodeRNGStreams(plan, controller_id="random")
        reference = EpisodeRNGStreams(plan, controller_id="random")

        # Simulate controller-dependent PF resampling/roughening consumption.
        branchy.pf.standard_normal((10_000, 3))
        branchy.pf.random(777)

        np.testing.assert_array_equal(
            branchy.sensor.standard_normal(64),
            reference.sensor.standard_normal(64),
        )
        np.testing.assert_array_equal(
            branchy.dead_reckoning.standard_normal((64, 3)),
            reference.dead_reckoning.standard_normal((64, 3)),
        )
        np.testing.assert_array_equal(
            branchy.scenario.random(64), reference.scenario.random(64)
        )

    def test_controller_id_only_changes_controller_stream(self) -> None:
        plan = EpisodeSeedPlan(root_seed=1234, episode_index=56)
        rl = EpisodeRNGStreams(plan, controller_id="rl")
        pid = EpisodeRNGStreams(plan, controller_id="pid-track")

        for name in ("scenario", "sensor", "dead_reckoning", "pf"):
            np.testing.assert_array_equal(
                rl.generator(name).random(32), pid.generator(name).random(32)
            )
        self.assertFalse(
            np.array_equal(rl.controller.random(32), pid.controller.random(32))
        )

    def test_checkpoint_restores_exact_generator_positions(self) -> None:
        plan = EpisodeSeedPlan(root_seed=7, episode_index=9, env_rank=2)
        streams = EpisodeRNGStreams(plan, controller_id="random")
        streams.scenario.random(11)
        streams.pf.standard_normal(23)
        streams.controller.uniform(-1.0, 1.0, 19)

        checkpoint = streams.checkpoint()
        # Explicitly verify it can be used as a JSON artifact.
        checkpoint = json.loads(json.dumps(checkpoint))
        expected = {
            name: streams.generator(name).standard_normal(40)
            for name in ALL_STREAM_NAMES
        }
        restored = EpisodeRNGStreams.from_checkpoint(checkpoint)
        for name in ALL_STREAM_NAMES:
            np.testing.assert_array_equal(
                expected[name], restored.generator(name).standard_normal(40)
            )

    def test_checkpoint_json_round_trip(self) -> None:
        streams = EpisodeRNGStreams(
            EpisodeSeedPlan(root_seed=81, episode_index=4), controller_id="rl"
        )
        streams.pf.random(100)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rng_checkpoint.json"
            streams.save_checkpoint_json(path)
            expected = streams.pf.random(20)
            restored = EpisodeRNGStreams.load_checkpoint_json(path)
            np.testing.assert_array_equal(expected, restored.pf.random(20))


class ExogenousNoiseTapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = EpisodeSeedPlan(root_seed=20260429, episode_index=860)

    def test_tape_is_controller_independent_and_reproducible(self) -> None:
        first = ExogenousNoiseTape.generate(
            self.plan, n_substeps=4400, n_doppler_measurements=440
        )
        second = ExogenousNoiseTape.generate(
            self.plan, n_substeps=4400, n_doppler_measurements=440
        )
        np.testing.assert_array_equal(
            first.dead_reckoning_z, second.dead_reckoning_z
        )
        np.testing.assert_array_equal(first.doppler_z, second.doppler_z)
        self.assertEqual(first.content_sha256(), second.content_sha256())

        # Different controllers obtain the same physical noise by using the tape.
        rl = ExogenousNoiseCursor(first)
        pid = ExogenousNoiseCursor(first)
        for _ in range(30):
            np.testing.assert_array_equal(
                rl.next_dead_reckoning([0.03, 0.2, 0.1]),
                pid.next_dead_reckoning([0.03, 0.2, 0.1]),
            )
        for _ in range(10):
            np.testing.assert_array_equal(
                rl.next_doppler(0.02), pid.next_doppler(0.02)
            )

    def test_tape_scaling_and_direct_time_indexing(self) -> None:
        tape = ExogenousNoiseTape.generate(
            self.plan, n_substeps=5, n_doppler_measurements=3
        )
        np.testing.assert_array_equal(
            tape.dead_reckoning_at(2, [1.0, 2.0, 3.0]),
            tape.dead_reckoning_z[2] * np.array([1.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            tape.doppler_at(1, 0.02), tape.doppler_z[1] * 0.02
        )

    def test_npz_round_trip_preserves_provenance_and_digest(self) -> None:
        tape = ExogenousNoiseTape.generate(
            self.plan, n_substeps=80, n_doppler_measurements=8
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episode_0860_noise.npz"
            tape.save_npz(path)
            restored = ExogenousNoiseTape.load_npz(path)

        self.assertEqual(tape.seed_plan, restored.seed_plan)
        self.assertEqual(tape.content_sha256(), restored.content_sha256())
        np.testing.assert_array_equal(
            tape.dead_reckoning_z, restored.dead_reckoning_z
        )
        np.testing.assert_array_equal(tape.doppler_z, restored.doppler_z)

    def test_npz_digest_detects_modified_noise(self) -> None:
        tape = ExogenousNoiseTape.generate(
            self.plan, n_substeps=8, n_doppler_measurements=2
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episode_noise.npz"
            tape.save_npz(path)
            with np.load(path, allow_pickle=False) as archive:
                dead = np.array(archive["dead_reckoning_z"], copy=True)
                doppler = np.array(archive["doppler_z"], copy=True)
                metadata_json = np.array(archive["metadata_json"], copy=True)
                old_digest = np.array(archive["content_sha256"], copy=True)
            dead[0, 0] += 1.0
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    dead_reckoning_z=dead,
                    doppler_z=doppler,
                    metadata_json=metadata_json,
                    content_sha256=old_digest,
                )
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                ExogenousNoiseTape.load_npz(path)

    def test_cursor_state_replays_from_exact_noise_index(self) -> None:
        tape = ExogenousNoiseTape.generate(
            self.plan, n_substeps=20, n_doppler_measurements=10
        )
        cursor = ExogenousNoiseCursor(tape)
        for _ in range(7):
            cursor.next_dead_reckoning([0.1, 0.2, 0.3])
        for _ in range(3):
            cursor.next_doppler([0.01, 0.02])
        state = cursor.state_dict()

        expected_dead = cursor.next_dead_reckoning([0.1, 0.2, 0.3])
        expected_doppler = cursor.next_doppler([0.01, 0.02])
        replay = ExogenousNoiseCursor(tape)
        replay.load_state_dict(state)
        np.testing.assert_array_equal(
            expected_dead, replay.next_dead_reckoning([0.1, 0.2, 0.3])
        )
        np.testing.assert_array_equal(
            expected_doppler, replay.next_doppler([0.01, 0.02])
        )

    def test_bounds_and_invalid_scales_fail_loudly(self) -> None:
        tape = ExogenousNoiseTape.generate(
            self.plan, n_substeps=2, n_doppler_measurements=1
        )
        with self.assertRaises(IndexError):
            tape.dead_reckoning_at(2, [1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            tape.dead_reckoning_at(0, [1.0, 2.0])
        with self.assertRaises(ValueError):
            tape.doppler_at(0, -0.02)


if __name__ == "__main__":
    unittest.main()
