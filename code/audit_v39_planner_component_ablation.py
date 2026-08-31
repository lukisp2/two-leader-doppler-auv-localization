#!/usr/bin/env python3
"""Independently audit V39 pairing, causal scores, RNG, and claim gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from uuv_v11_rng import ExogenousNoiseTape
import audit_v38_leader_source_ablation as audit38
import uuv_v19_observability as v19
import uuv_v39_planner_component_ablation as v39


AUDITOR_VERSION = "v39_independent_causal_auditor_1.0"
RUNTIME_THRESHOLD_S = 2.0
MATERIAL_RATE_DIFFERENCE = 0.05


def _parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(
    campaign: Path,
    episode_index: int,
    seed: int,
    arm: str,
) -> Tuple[Path, Path]:
    return (
        campaign
        / "episode_results"
        / f"episode_{episode_index:04d}_seed_{seed}_{arm}.json",
        campaign
        / "traces_npz"
        / arm
        / f"episode_{episode_index:04d}_seed_{seed}.npz",
    )


def _material_decision(
    full_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
    *,
    positive: str,
    negative: str,
    integrity_valid: bool,
) -> str:
    full = {int(row["episode_seed"]): row for row in full_rows}
    comparator = {
        int(row["episode_seed"]): row for row in comparator_rows
    }
    if set(full) != set(comparator):
        raise RuntimeError("material-decision seed sets differ")
    ordered = sorted(full)
    count = len(ordered)
    terminal_difference = (
        sum(bool(full[seed]["terminal_joint_success"]) for seed in ordered)
        - sum(
            bool(comparator[seed]["terminal_joint_success"])
            for seed in ordered
        )
    ) / max(count, 1)
    tail_difference = (
        sum(bool(full[seed]["tail80_joint_success"]) for seed in ordered)
        - sum(
            bool(comparator[seed]["tail80_joint_success"])
            for seed in ordered
        )
    ) / max(count, 1)
    safety_keys = (
        "unsafe_transition_count",
        "unsafe_track_start_count",
        "unsafe_track_end_count",
    )
    safety = all(
        sum(int(full[seed][key]) for seed in ordered)
        <= sum(int(comparator[seed][key]) for seed in ordered)
        for key in safety_keys
    )
    material = bool(
        terminal_difference >= MATERIAL_RATE_DIFFERENCE
        or tail_difference >= MATERIAL_RATE_DIFFERENCE
    )
    return positive if integrity_valid and safety and material else negative


def _audit_random_policy(
    summary: Mapping[str, Any],
    trace: Mapping[str, np.ndarray],
) -> None:
    metadata = summary["planner"]["random_policy"]
    if metadata is None:
        raise RuntimeError("random arm lacks policy metadata")
    seed = int(metadata["policy_seed"])
    if seed != v39.policy_seed_for_episode(int(summary["episode_seed"])):
        raise RuntimeError("random policy seed differs from frozen derivation")
    counts = [int(value) for value in metadata["candidate_counts"]]
    indices = [int(value) for value in metadata["candidate_indices"]]
    if len(counts) != len(indices) or len(counts) != int(
        summary["planner"]["decision_count"]
    ):
        raise RuntimeError("random policy stream length differs")
    rng = np.random.Generator(np.random.PCG64(seed))
    expected = [int(rng.integers(0, count)) for count in counts]
    if expected != indices:
        raise RuntimeError("random candidate indices do not match policy RNG")
    if any(count < 1 for count in counts) or any(
        index < 0 or index >= count
        for index, count in zip(indices, counts)
    ):
        raise RuntimeError("random candidate index is out of bounds")
    expected_hash = v39.policy_index_stream_sha256(seed, counts, indices)
    if expected_hash != metadata["policy_index_stream_sha256"]:
        raise RuntimeError("random policy-tape hash differs")
    trace_counts = np.asarray(
        trace["planner_candidate_count"],
        dtype=np.int64,
    )
    trace_indices = np.asarray(
        trace["planner_random_candidate_index"],
        dtype=np.int64,
    )
    decisions = trace_counts > 0
    if list(trace_counts[decisions]) != counts:
        raise RuntimeError("trace random candidate counts differ")
    if list(trace_indices[decisions]) != indices:
        raise RuntimeError("trace random candidate indices differ")
    if np.any(trace_indices[~decisions] != -1):
        raise RuntimeError("random candidate index exists without a decision")
    if np.any(
        np.asarray(trace["planner_hypothesis_count"], dtype=np.int64) != 0
    ):
        raise RuntimeError("random selector received belief hypotheses")


def _audit_active_pairing(
    traces: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Any]:
    first_decisions: Dict[str, int] = {}
    for arm in v39.ACTIVE_ARMS:
        indices = np.flatnonzero(
            np.asarray(
                traces[arm]["planner_candidate_count"],
                dtype=np.int64,
            )
            > 0
        )
        if indices.size == 0:
            raise RuntimeError(f"{arm} never called its planner")
        first_decisions[arm] = int(indices[0])
    if len(set(first_decisions.values())) != 1:
        raise RuntimeError("active arms have different first planner actions")
    first = next(iter(first_decisions.values()))
    fields = (
        "action_speed",
        "action_yaw",
        "action_pitch",
        "truth_x",
        "truth_y",
        "truth_z",
        "estimate_x",
        "estimate_y",
        "estimate_z",
    )
    reference = traces[v39.ARM_FULL]
    for arm in (v39.ARM_NO_PAIR, v39.ARM_BEST_ONLY):
        for field in fields:
            left = np.asarray(reference[field])[:first]
            right = np.asarray(traces[arm][field])[:first]
            if not np.array_equal(left, right, equal_nan=True):
                raise RuntimeError(
                    f"active common prefix differs: {arm}/{field}"
                )
    full_hash = str(
        np.asarray(reference["planner_hypothesis_sha256"])[first]
    )
    no_pair_hash = str(
        np.asarray(
            traces[v39.ARM_NO_PAIR]["planner_hypothesis_sha256"]
        )[first]
    )
    if not full_hash or full_hash != no_pair_hash:
        raise RuntimeError(
            "full/no-pair initial hypothesis sets are not identical"
        )
    if np.max(
        np.asarray(
            traces[v39.ARM_BEST_ONLY]["planner_hypothesis_count"],
            dtype=np.int64,
        ),
        initial=0,
    ) > 1:
        raise RuntimeError("best-only arm retained more than one hypothesis")
    return {
        "first_planner_action_index": first,
        "common_prefix_action_count": first,
        "full_no_pair_initial_hypothesis_sha256": full_hash,
    }


def _audit_snapshot(
    campaign: Path,
    contract: Mapping[str, Any],
) -> None:
    manifest = contract["source_sha256"]
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != contract[
        "source_manifest_sha256"
    ]:
        raise RuntimeError("source manifest digest differs")
    for name, expected in manifest.items():
        snapshot = campaign / "control" / "source_snapshot" / name
        if not snapshot.is_file():
            raise RuntimeError(f"source snapshot missing {name}")
        if _sha256(snapshot) != expected:
            raise RuntimeError(f"source snapshot digest differs: {name}")


def audit_campaign(campaign: Path) -> Dict[str, Any]:
    campaign = campaign.expanduser().resolve()
    contract_path = campaign / "control" / "campaign_contract.json"
    campaign_summary_path = campaign / "campaign_summary.json"
    if not contract_path.is_file() or not campaign_summary_path.is_file():
        raise FileNotFoundError("campaign contract or summary is missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    campaign_summary = json.loads(
        campaign_summary_path.read_text(encoding="utf-8")
    )
    _audit_snapshot(campaign, contract)
    expected_arms = [str(value["arm"]) for value in contract["arms"]]
    if expected_arms != list(v39.ARM_NAMES):
        raise RuntimeError("contract arm order differs from frozen V39")
    if contract["reserved_final_range_untouched"] != [
        v39.RESERVED_START,
        v39.FINAL_END,
    ]:
        raise RuntimeError("reserved/final range contract differs")
    if any(
        v39.RESERVED_START <= int(seed) <= v39.FINAL_END
        for seed in contract["seeds"]
    ):
        raise RuntimeError("campaign selected a reserved/final seed")

    rows: List[Dict[str, Any]] = []
    pair_audits: List[Dict[str, Any]] = []
    maximum_runtime = 0.0
    audit_release_violations = 0
    for local_index, seed_value in enumerate(contract["seeds"]):
        episode_index = int(contract["episode_start"]) + local_index
        seed = int(seed_value)
        tape_path = (
            campaign
            / "noise_tapes"
            / f"episode_{episode_index:04d}_seed_{seed}.npz"
        )
        tape = ExogenousNoiseTape.load_npz(tape_path)
        summaries: Dict[str, Mapping[str, Any]] = {}
        traces: Dict[str, Mapping[str, np.ndarray]] = {}
        for arm in v39.ARM_NAMES:
            result_path, trace_path = _paths(
                campaign,
                episode_index,
                seed,
                arm,
            )
            summary = json.loads(result_path.read_text(encoding="utf-8"))
            with np.load(trace_path, allow_pickle=False) as archive:
                trace = {key: archive[key].copy() for key in archive.files}
            if summary["arm"] != arm:
                raise RuntimeError("saved arm label differs")
            if int(summary["episode_seed"]) != seed:
                raise RuntimeError("saved episode seed differs")
            if summary["noise_tape_sha256"] != tape.content_sha256():
                raise RuntimeError("saved noise-tape hash differs")
            if tuple(summary["source_mask"]) != v39.SOURCE_MASK:
                raise RuntimeError("V39 source mask differs from two links")
            if int(summary["action_count"]) != int(
                contract["fixed_horizon_actions"]
            ):
                raise RuntimeError("fixed action horizon differs")
            if int(summary["active_scalar_measurement_count"]) != 2 * int(
                summary["measurement_row_count"]
            ):
                raise RuntimeError("two-link residual denominator differs")
            actions = np.column_stack(
                (
                    trace["action_speed"],
                    trace["action_yaw"],
                    trace["action_pitch"],
                )
            )
            if (
                not np.all(np.isfinite(actions))
                or np.any(actions < -1.0)
                or np.any(actions > 1.0)
            ):
                raise RuntimeError("trace contains an invalid action")
            score = audit38._trace_score(trace)
            audit38._check_summary(summary, score)
            full_history = audit38._full_history(trace)
            masked = v39.v38.MaskedDopplerHistory.from_full(
                full_history,
                v39.SOURCE_MASK,
            )
            audit38._check_masked_gate_metrics(trace, masked)
            if arm == v39.ARM_RANDOM:
                _audit_random_policy(summary, trace)
            else:
                config = v39.planner_config_for_arm(arm)
                if config is None or summary["planner"]["config"] != (
                    config.to_dict()
                ):
                    raise RuntimeError("active planner config differs")
                if summary["planner"]["kind"] != "belief_conditioned":
                    raise RuntimeError("active planner kind differs")
            exact = score["exact"]
            row = {
                "episode_seed": seed,
                "arm": arm,
                "terminal_joint_success": bool(
                    score["terminal_joint_success"]
                ),
                "tail80_joint_success": bool(score["tail80_joint_success"]),
                "unsafe_transition_count": int(
                    exact["false_transition_count"]
                ),
                "unsafe_track_start_count": int(
                    exact["false_locked_action_start_count"]
                ),
                "unsafe_track_end_count": int(
                    exact["false_locked_action_end_count"]
                ),
                "terminal_localization_error_m": float(
                    score["terminal_localization_error_m"]
                ),
            }
            rows.append(row)
            summaries[arm] = summary
            traces[arm] = trace
            maximum_runtime = max(
                maximum_runtime,
                float(summary["maximum_combined_decision_runtime_s"]),
            )
            audit_release_violations += int(
                summary["gate"]["audit_release_violation_count"]
            )
        hashes = {
            str(summary["noise_tape_sha256"])
            for summary in summaries.values()
        }
        supports = {
            json.dumps(summary["mission_support"], sort_keys=True)
            for summary in summaries.values()
        }
        cursors = {
            json.dumps(summary["noise_cursor"], sort_keys=True)
            for summary in summaries.values()
        }
        if len(hashes) != 1 or len(supports) != 1 or len(cursors) != 1:
            raise RuntimeError("paired arms differ in tape, support, or cursor")
        pair_audits.append(
            {
                "episode_seed": seed,
                **_audit_active_pairing(traces),
            }
        )

    expected_rows = int(contract["expected_runs"])
    integrity_checks = {
        "all_runs_complete": len(rows) == expected_rows,
        "all_four_arms_paired": len(pair_audits) == int(contract["episodes"]),
        "publication_settings": bool(contract["publication_settings"]),
        "audit_release_violations_zero": audit_release_violations == 0,
        "maximum_runtime_below_2s": maximum_runtime < RUNTIME_THRESHOLD_S,
        "fresh_seed_audit_pass": bool(
            contract["smoke"] or contract["fresh_seed_audit"]["fresh"]
        ),
        "reserved_final_untouched": True,
        "source_snapshot_valid": True,
    }
    integrity_valid = bool(all(integrity_checks.values()))
    indexed = {
        arm: [row for row in rows if row["arm"] == arm]
        for arm in v39.ARM_NAMES
    }
    if bool(contract["smoke"]):
        decisions = {
            "pair_term": "SMOKE_ONLY",
            "retained_hypotheses": "SMOKE_ONLY",
            "informed_selection": "SMOKE_ONLY",
        }
        decision = "SMOKE_PASS" if integrity_valid else "SMOKE_FAIL"
    else:
        decisions = {
            "pair_term": _material_decision(
                indexed[v39.ARM_FULL],
                indexed[v39.ARM_NO_PAIR],
                positive="MATERIAL_PAIR_TERM_EFFECT",
                negative="NO_MATERIAL_PAIR_TERM_EFFECT_DETECTED",
                integrity_valid=integrity_valid,
            ),
            "retained_hypotheses": _material_decision(
                indexed[v39.ARM_FULL],
                indexed[v39.ARM_BEST_ONLY],
                positive="MATERIAL_RETAINED_HYPOTHESIS_EFFECT",
                negative=(
                    "NO_MATERIAL_RETAINED_HYPOTHESIS_EFFECT_DETECTED"
                ),
                integrity_valid=integrity_valid,
            ),
            "informed_selection": _material_decision(
                indexed[v39.ARM_FULL],
                indexed[v39.ARM_RANDOM],
                positive="MATERIAL_INFORMED_SELECTION_EFFECT",
                negative="NO_MATERIAL_INFORMED_SELECTION_EFFECT_DETECTED",
                integrity_valid=integrity_valid,
            ),
        }
        decision = "V39_COMPLETE" if integrity_valid else "V39_INVALID"
    if decisions != campaign_summary["component_decisions"]:
        raise RuntimeError("independent component decisions differ from runner")
    if integrity_valid != bool(campaign_summary["integrity_valid"]):
        raise RuntimeError("independent integrity decision differs from runner")
    return {
        "schema_version": 1,
        "auditor_version": AUDITOR_VERSION,
        "campaign": str(campaign),
        "valid": integrity_valid,
        "decision": decision,
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "integrity_checks": integrity_checks,
        "component_decisions": decisions,
        "maximum_combined_decision_runtime_s": maximum_runtime,
        "audit_release_violation_count": audit_release_violations,
        "pair_audits": pair_audits,
        "issue_count": 0,
        "issues": [],
        "reserved_final_range_untouched": [
            v39.RESERVED_START,
            v39.FINAL_END,
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse(argv)
    result = audit_campaign(args.campaign)
    target = args.campaign.expanduser().resolve() / "independent_audit.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
