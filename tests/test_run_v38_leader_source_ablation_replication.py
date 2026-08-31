import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import run_v38_leader_source_ablation_replication as replication


class V38ReplicationRunnerTests(unittest.TestCase):
    def _args(self, output, **values):
        defaults = {
            "output_dir": Path(output),
            "seed_start": 48_600,
            "preflight_seed": None,
            "resume": False,
            "progress_every": 2,
            "reference_contract": None,
            "reference_decision": None,
        }
        defaults.update(values)
        return argparse.Namespace(**defaults)

    def test_full_campaign_is_exact_authorized_block(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replication._settings(self._args(directory))
        self.assertFalse(settings["preflight"])
        self.assertEqual(settings["episodes"], 100)
        self.assertEqual(settings["seeds"], list(range(48_600, 48_700)))
        self.assertTrue(settings["publication_settings"])
        self.assertEqual(settings["coarse_candidates"], 4096)
        self.assertEqual(settings["coarse_sweeps"], 2)
        self.assertEqual(settings["local_starts"], 48)

    def test_other_campaign_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                replication._settings(
                    self._args(directory, seed_start=48_700)
                )

    def test_preflight_must_be_outside_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                replication._settings(
                    self._args(directory, preflight_seed=48_650)
                )
            settings = replication._settings(
                self._args(directory, preflight_seed=48_799)
            )
        self.assertTrue(settings["preflight"])
        self.assertEqual(settings["seeds"], [48_799])
        self.assertEqual(settings["campaign_seeds"], list(range(48_600, 48_700)))

    def test_freshness_audit_detects_selected_seed_and_ignores_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "experiments_old"
            old.mkdir()
            (old / "result_seed_48607.json").write_text(
                json.dumps({"episode_seed": 48_607}),
                encoding="utf-8",
            )
            output = root / "experiments_new"
            output.mkdir()
            (output / "result_seed_48608.json").write_text(
                json.dumps({"episode_seed": 48_608}),
                encoding="utf-8",
            )
            audit = replication._fresh_seed_audit(
                root,
                output,
                list(range(48_600, 48_700)),
            )
        self.assertFalse(audit["fresh"])
        self.assertEqual(audit["finding_count"], 1)
        self.assertEqual(audit["findings"][0]["seeds"], [48_607])
        self.assertTrue(audit["consecutive"])

    def test_runtime_threshold_is_frozen(self):
        self.assertEqual(replication.RUNTIME_THRESHOLD_S, 2.0)
        self.assertIn("No efficacy", replication.REPLICATION_REASON)

    def test_public_runner_uses_frozen_reference_hash_when_private_campaign_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "private_campaign" / "decision.json"
            digest, source = replication._reference_artifact_sha256(
                missing,
                replication.REFERENCE_DECISION_SHA256,
                label="reference decision",
                explicitly_provided=False,
            )
        self.assertEqual(digest, replication.REFERENCE_DECISION_SHA256)
        self.assertEqual(source, "frozen_expected_sha256_fallback")

    def test_public_release_source_manifest_resolves_sibling_tests(self):
        root = Path(replication.__file__).resolve().parent
        manifest = replication._loaded_local_sources(root)
        self.assertIn(
            "tests/test_uuv_v38_leader_source_ablation.py",
            manifest,
        )
        self.assertIn(
            "tests/test_run_v38_leader_source_ablation_replication.py",
            manifest,
        )

    def test_public_contract_builds_without_private_reference_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = replication._settings(
                self._args(directory, preflight_seed=48_799)
            )
            self.assertFalse(Path(settings["reference_contract"]).exists())
            self.assertFalse(Path(settings["reference_decision"]).exists())
            cfg = replication.runner20._load_environment_config(
                Path(settings["metadata"])
            )
            contract = replication._contract(
                settings,
                cfg,
                replication.v19.BatchEstimatorConfig(
                    coarse_candidates=4096,
                    coarse_sweeps=2,
                    local_starts=48,
                    gate_mode="raw",
                    candidate_radial_distribution="uniform_radius",
                ),
                replication.v22.ActivePlannerConfig(),
                replication.v24.AuditedLockConfig(),
                {"fresh": True, "range": [48_799, 48_799]},
                {"fresh": True, "range": [48_600, 48_699]},
            )
        amendment = contract["replication_amendment"]
        self.assertEqual(
            amendment["reference_campaign_contract_sha256"],
            replication.REFERENCE_CAMPAIGN_CONTRACT_SHA256,
        )
        self.assertEqual(
            amendment["reference_campaign_contract_sha256_source"],
            "frozen_expected_sha256_fallback",
        )
        self.assertEqual(
            amendment["reference_decision_sha256_source"],
            "frozen_expected_sha256_fallback",
        )

    def test_present_reference_is_verified_and_explicit_missing_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "reference.json"
            artifact.write_bytes(b"frozen reference\n")
            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            digest, source = replication._reference_artifact_sha256(
                artifact,
                expected,
                label="reference artifact",
                explicitly_provided=True,
            )
            self.assertEqual(digest, expected)
            self.assertEqual(source, "verified_file")
            with self.assertRaises(FileNotFoundError):
                replication._reference_artifact_sha256(
                    root / "missing.json",
                    expected,
                    label="reference artifact",
                    explicitly_provided=True,
                )
            with self.assertRaises(RuntimeError):
                replication._reference_artifact_sha256(
                    artifact,
                    "0" * 64,
                    label="reference artifact",
                    explicitly_provided=False,
                )


if __name__ == "__main__":
    unittest.main()
