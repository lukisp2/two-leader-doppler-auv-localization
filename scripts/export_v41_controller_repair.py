#!/usr/bin/env python3
"""Validate and export the V41.1 controller-repair qualification.

The exporter treats the 400 episode-arm rows as the numerical source of
truth.  It independently reconstructs arm summaries and paired contrasts,
checks them against the frozen campaign summary, and writes compact Git-sized
publication artifacts.  The frozen campaign files are never modified.

Two inherited field names require an explicit semantic correction in the
public export:

* ``final_holdout_sealed`` and ``sealed_seed_range_untouched`` contain a
  reserved range, not evidence that no earlier process ever inspected it;
* ``CONTROLLER_REPAIR_REJECT_OR_INVALID`` conflates a valid qualification
  that missed one composite gate with an integrity-invalid campaign.

The original bytes and labels remain in provenance.  The public audit records
the unambiguous interpretation ``VALID_COMPLETE__COMPOSITE_ACCEPTANCE_NOT_MET``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
EXPECTED_SEEDS = tuple(range(51_000, 51_100))
RESERVED_RANGE = (50_000, 50_999)
CONTROLLERS = ("baseline_pid", "delay_aware")
CURRENTS = ("no_current", "bottom_track_visible")
EXPECTED_ROWS = len(EXPECTED_SEEDS) * len(CONTROLLERS) * len(CURRENTS)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 41_051_000
LOCALIZATION_MARGIN_M = 0.50
LEGACY_DECISION = "CONTROLLER_REPAIR_REJECT_OR_INVALID"
PUBLIC_DECISION = "VALID_COMPLETE__COMPOSITE_ACCEPTANCE_NOT_MET"
PRIVATE_ROOT_TOKEN = "${FROZEN_SOURCE_ROOT}"

BOOL_FIELDS = (
    "terminal_joint_success",
    "tail80_joint_success",
    "dwell15_joint_success",
    "ever_locked",
)
INT_FIELDS = (
    "episode_index",
    "episode_seed",
    "unsafe_transition_count",
    "unsafe_track_start_count",
    "unsafe_track_end_count",
    "audit_release_violation_count",
    "action_count",
    "post_track_action_count",
)
FLOAT_FIELDS = (
    "tail50_joint_occupancy",
    "terminal_localization_error_m",
    "terminal_formation_error_m",
    "first_track_time_s",
    "mean_squared_action",
    "maximum_combined_decision_runtime_s",
    "requested_delivered_action_rms",
    "post_track_saturation_fraction",
    "post_track_speed_saturation_fraction",
    "post_track_yaw_saturation_fraction",
    "post_track_pitch_saturation_fraction",
    "post_track_action_total_variation",
    "post_track_action_curvature_rms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _parse_bool(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise RuntimeError(f"invalid Boolean in {field}: {value!r}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        required = {
            "arm",
            "controller",
            "current",
            "noise_tape_sha256",
            "current_tape_sha256",
            *BOOL_FIELDS,
            *INT_FIELDS,
            *FLOAT_FIELDS,
        }
        missing = sorted(required - set(fieldnames))
        if missing:
            raise RuntimeError(f"row CSV lacks fields: {missing}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in BOOL_FIELDS:
                row[field] = _parse_bool(str(raw[field]), field)
            for field in INT_FIELDS:
                row[field] = int(str(raw[field]))
            for field in FLOAT_FIELDS:
                text = str(raw[field]).strip()
                row[field] = None if not text else float(text)
            rows.append(row)
    return rows


def _finite(values: Iterable[Any]) -> np.ndarray:
    parsed = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return np.asarray(parsed, dtype=np.float64)


def distribution(values: Iterable[Any]) -> dict[str, Any] | None:
    array = _finite(values)
    if not array.size:
        return None
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _mcnemar_exact(reference_only: int, treatment_only: int) -> float | None:
    discordant = int(reference_only) + int(treatment_only)
    if discordant == 0:
        return None
    lower = min(int(reference_only), int(treatment_only))
    probability = sum(math.comb(discordant, k) for k in range(lower + 1))
    probability /= 2.0**discordant
    return float(min(1.0, 2.0 * probability))


def _bootstrap_mean_interval(values: Sequence[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        stop = min(start + 1000, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        draws[start:stop] = np.mean(array[indices], axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    expected_cells = {
        (seed, controller, current)
        for seed in EXPECTED_SEEDS
        for controller in CONTROLLERS
        for current in CURRENTS
    }
    actual_cells = [
        (int(row["episode_seed"]), str(row["controller"]), str(row["current"]))
        for row in rows
    ]
    if len(set(actual_cells)) != len(actual_cells):
        raise RuntimeError("duplicate seed/controller/current row")
    if set(actual_cells) != expected_cells:
        missing = sorted(expected_cells - set(actual_cells))[:5]
        extra = sorted(set(actual_cells) - expected_cells)[:5]
        raise RuntimeError(f"qualification support mismatch; missing={missing}, extra={extra}")
    for row in rows:
        seed = int(row["episode_seed"])
        if int(row["episode_index"]) != seed - EXPECTED_SEEDS[0]:
            raise RuntimeError(f"seed/index mismatch for {seed}")
        if int(row["action_count"]) != 220:
            raise RuntimeError(f"non-frozen horizon for seed {seed}")
        for field in FLOAT_FIELDS:
            value = row[field]
            if value is not None and not math.isfinite(float(value)):
                raise RuntimeError(f"non-finite {field} for seed {seed}")

    by_seed_current: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed_current[(int(row["episode_seed"]), str(row["current"]))].append(row)
    for key, pair in by_seed_current.items():
        if len(pair) != 2:
            raise RuntimeError(f"incomplete controller pair: {key}")
        for hash_field in ("noise_tape_sha256", "current_tape_sha256"):
            if len({str(row[hash_field]) for row in pair}) != 1:
                raise RuntimeError(f"paired {hash_field} mismatch: {key}")
    selected = {int(row["episode_seed"]) for row in rows}
    reserved = set(range(RESERVED_RANGE[0], RESERVED_RANGE[1] + 1))
    return {
        "row_count": len(rows),
        "seed_count": len(selected),
        "seed_range": [min(selected), max(selected)],
        "complete_factorial_support": True,
        "paired_tape_hashes_equal": True,
        "all_fixed_horizon": True,
        "selected_seeds_disjoint_reserved_range": selected.isdisjoint(reserved),
        "reserved_range": list(RESERVED_RANGE),
    }


def aggregate_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    result: dict[str, Any] = {
        "episodes": n,
        "terminal_success_count": sum(bool(row["terminal_joint_success"]) for row in rows),
        "tail80_success_count": sum(bool(row["tail80_joint_success"]) for row in rows),
        "dwell15_success_count": sum(bool(row["dwell15_joint_success"]) for row in rows),
        "ever_lock_count": sum(bool(row["ever_locked"]) for row in rows),
        "unsafe_transition_count": sum(int(row["unsafe_transition_count"]) for row in rows),
        "unsafe_track_start_count": sum(int(row["unsafe_track_start_count"]) for row in rows),
        "unsafe_track_end_count": sum(int(row["unsafe_track_end_count"]) for row in rows),
        "audit_release_violation_count": sum(
            int(row["audit_release_violation_count"]) for row in rows
        ),
    }
    result["terminal_success_rate"] = result["terminal_success_count"] / n
    result["tail80_success_rate"] = result["tail80_success_count"] / n
    result["ever_lock_rate"] = result["ever_lock_count"] / n
    for field in (
        "first_track_time_s",
        "terminal_localization_error_m",
        "terminal_formation_error_m",
        "tail50_joint_occupancy",
        "mean_squared_action",
        "requested_delivered_action_rms",
        "post_track_saturation_fraction",
        "post_track_action_total_variation",
        "post_track_action_curvature_rms",
        "maximum_combined_decision_runtime_s",
    ):
        result[field] = distribution(row[field] for row in rows)
    return result


def paired_binary(
    reference: Mapping[int, Mapping[str, Any]],
    treatment: Mapping[int, Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    reference_only = treatment_only = 0
    differences: list[float] = []
    for seed in EXPECTED_SEEDS:
        left = bool(reference[seed][field])
        right = bool(treatment[seed][field])
        reference_only += int(left and not right)
        treatment_only += int(right and not left)
        differences.append(float(right) - float(left))
    return {
        "pairs": len(differences),
        "reference_only": reference_only,
        "treatment_only": treatment_only,
        "risk_difference": float(np.mean(differences)),
        "mcnemar_exact_two_sided_p": _mcnemar_exact(reference_only, treatment_only),
    }


def paired_numeric(
    reference: Mapping[int, Mapping[str, Any]],
    treatment: Mapping[int, Mapping[str, Any]],
    field: str,
    seed: int,
) -> dict[str, Any]:
    differences = [
        float(treatment[episode_seed][field]) - float(reference[episode_seed][field])
        for episode_seed in EXPECTED_SEEDS
        if treatment[episode_seed][field] is not None
        and reference[episode_seed][field] is not None
    ]
    return {
        "pairs": len(differences),
        "treatment_minus_reference": distribution(differences),
        "bootstrap_mean_95": _bootstrap_mean_interval(differences, seed),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def paired_contrast(rows: Sequence[Mapping[str, Any]], current: str) -> dict[str, Any]:
    maps = {
        controller: {
            int(row["episode_seed"]): row
            for row in rows
            if row["controller"] == controller and row["current"] == current
        }
        for controller in CONTROLLERS
    }
    if any(set(values) != set(EXPECTED_SEEDS) for values in maps.values()):
        raise RuntimeError(f"incomplete paired support for {current}")
    reference, treatment = maps[CONTROLLERS[0]], maps[CONTROLLERS[1]]
    result: dict[str, Any] = {
        "current": current,
        "reference": CONTROLLERS[0],
        "treatment": CONTROLLERS[1],
        "pairs": len(EXPECTED_SEEDS),
    }
    for field in (
        "terminal_joint_success",
        "tail80_joint_success",
        "dwell15_joint_success",
    ):
        result[field] = paired_binary(reference, treatment, field)
    for index, field in enumerate(
        (
            "terminal_localization_error_m",
            "terminal_formation_error_m",
            "tail50_joint_occupancy",
            "mean_squared_action",
            "post_track_saturation_fraction",
            "post_track_action_total_variation",
            "post_track_action_curvature_rms",
            "requested_delivered_action_rms",
        )
    ):
        result[field] = paired_numeric(
            reference,
            treatment,
            field,
            BOOTSTRAP_SEED + 100 * CURRENTS.index(current) + index,
        )
    return result


def _close_subset(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        missing = set(left) - set(right)
        if missing:
            raise RuntimeError(f"summary lacks derived keys at {path}: {sorted(missing)}")
        for key in left:
            _close_subset(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"summary lengths differ at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _close_subset(left_item, right_item, f"{path}[{index}]")
        return
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if right is None or not math.isclose(float(left), float(right), rel_tol=2e-12, abs_tol=2e-12):
            raise RuntimeError(f"summary value differs at {path}: {left!r} != {right!r}")
        return
    if left != right:
        raise RuntimeError(f"summary value differs at {path}: {left!r} != {right!r}")


def _public_arm_rows(by_arm: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for current in CURRENTS:
        for controller in CONTROLLERS:
            key = (
                "belief_active__low_order_dynamic__"
                f"{controller}__{current}"
            )
            metric = by_arm[key]
            output.append(
                {
                    "current": current,
                    "controller_internal": controller,
                    "controller": (
                        "delay-unaware proportional tracker"
                        if controller == "baseline_pid"
                        else "delay-aware tracker"
                    ),
                    "episodes": metric["episodes"],
                    "terminal_success_count": metric["terminal_success_count"],
                    "terminal_success_rate": metric["terminal_success_rate"],
                    "tail80_success_count": metric["tail80_success_count"],
                    "tail80_success_rate": metric["tail80_success_rate"],
                    "terminal_formation_error_m_median": metric["terminal_formation_error_m"]["median"],
                    "terminal_formation_error_m_p95": metric["terminal_formation_error_m"]["p95"],
                    "terminal_localization_error_m_median": metric["terminal_localization_error_m"]["median"],
                    "terminal_localization_error_m_p95": metric["terminal_localization_error_m"]["p95"],
                    "first_track_time_s_median": metric["first_track_time_s"]["median"],
                    "post_track_saturation_fraction_mean": metric["post_track_saturation_fraction"]["mean"],
                    "post_track_action_curvature_rms_mean": metric["post_track_action_curvature_rms"]["mean"],
                    "mean_squared_action_mean": metric["mean_squared_action"]["mean"],
                    "unsafe_transition_start_end_count": (
                        metric["unsafe_transition_count"]
                        + metric["unsafe_track_start_count"]
                        + metric["unsafe_track_end_count"]
                    ),
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sanitize(value: Any, source_root: str) -> tuple[Any, int]:
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        replacements = 0
        for key, child in value.items():
            converted_child, count = _sanitize(child, source_root)
            converted[str(key)] = converted_child
            replacements += count
        return converted, replacements
    if isinstance(value, list):
        converted_list: list[Any] = []
        replacements = 0
        for child in value:
            converted_child, count = _sanitize(child, source_root)
            converted_list.append(converted_child)
            replacements += count
        return converted_list, replacements
    if isinstance(value, str):
        count = value.count(source_root)
        return value.replace(source_root, PRIVATE_ROOT_TOKEN), count
    return value, 0


def export(
    campaign: Path,
    table_dir: Path,
    provenance_dir: Path,
    *,
    source_root: Path,
) -> dict[str, Any]:
    campaign = campaign.expanduser().resolve()
    table_dir = table_dir.expanduser().resolve()
    provenance_dir = provenance_dir.expanduser().resolve()
    source_root_text = str(source_root.expanduser().resolve())
    required = {
        "rows": campaign / "episode_arm_summary.csv",
        "summary": campaign / "campaign_summary.json",
        "decision": campaign / "decision.json",
        "contract": campaign / "control/campaign_contract.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = (
        table_dir / "controller_repair_episode_rows.csv",
        table_dir / "controller_repair_publication_arms.csv",
        table_dir / "controller_repair_results.json",
        provenance_dir,
    )
    if any(path.exists() for path in outputs):
        raise RuntimeError("V41 public export destination already exists")

    rows = load_rows(required["rows"])
    validation = validate_rows(rows)
    summary = read_json_object(required["summary"])
    decision = read_json_object(required["decision"])
    contract = read_json_object(required["contract"])
    if not bool(summary.get("integrity_valid")) or not bool(decision.get("integrity_valid")):
        raise RuntimeError("V41 qualification is not integrity-valid")
    if summary.get("status") != "complete" or not bool(summary.get("full_qualification")):
        raise RuntimeError("V41 qualification is not complete")
    if summary.get("decision") != LEGACY_DECISION or decision.get("decision") != LEGACY_DECISION:
        raise RuntimeError("unexpected frozen V41 decision label")
    if contract.get("seeds") != list(EXPECTED_SEEDS):
        raise RuntimeError("campaign contract has unexpected seed cohort")
    if contract.get("sealed_final_range") != list(RESERVED_RANGE):
        raise RuntimeError("campaign contract has unexpected reserved range")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    by_arm = {name: aggregate_arm(cell) for name, cell in sorted(grouped.items())}
    contrasts = {current: paired_contrast(rows, current) for current in CURRENTS}
    _close_subset(by_arm, summary["by_arm"], "by_arm")
    contrast_overlap = {
        current: {
            key: value
            for key, value in contrasts[current].items()
            if key in summary["paired_delay_aware_minus_baseline_by_current"][current]
        }
        for current in CURRENTS
    }
    _close_subset(
        contrast_overlap,
        summary["paired_delay_aware_minus_baseline_by_current"],
        "paired_contrasts",
    )

    baseline_p95 = by_arm[
        "belief_active__low_order_dynamic__baseline_pid__no_current"
    ]["terminal_localization_error_m"]["p95"]
    repaired_p95 = by_arm[
        "belief_active__low_order_dynamic__delay_aware__no_current"
    ]["terminal_localization_error_m"]["p95"]
    p95_difference = float(repaired_p95 - baseline_p95)
    excess = float(p95_difference - LOCALIZATION_MARGIN_M)
    semantic_audit = {
        "schema_version": SCHEMA_VERSION,
        "campaign_integrity": "valid",
        "campaign_completeness": "complete 100-seed paired qualification",
        "prespecified_composite_acceptance": "not_met",
        "legacy_decision_label": LEGACY_DECISION,
        "public_unambiguous_decision_label": PUBLIC_DECISION,
        "legacy_artifacts_modified": False,
        "decision_semantics_note": (
            "The frozen label joins REJECT and INVALID. This campaign is not invalid: "
            "all integrity checks passed. It is a valid complete qualification that "
            "did not meet every prespecified acceptance condition."
        ),
        "failed_prespecified_condition": {
            "current": "no_current",
            "criterion": "terminal localization median and p95 each no more than 0.50 m above reference",
            "reference_p95_m": baseline_p95,
            "treatment_p95_m": repaired_p95,
            "treatment_minus_reference_p95_m": p95_difference,
            "margin_m": LOCALIZATION_MARGIN_M,
            "excess_over_margin_m": excess,
        },
        "reserved_seed_marker_audit": {
            "legacy_fields": ["final_holdout_sealed", "sealed_seed_range_untouched"],
            "stored_value": list(RESERVED_RANGE),
            "correct_semantics": (
                "reserved range marker and runner exclusion rule; not proof that no "
                "earlier external process ever inspected the range"
            ),
            "verified_here": (
                "all selected qualification seeds are 51000--51099 and are disjoint "
                "from the reserved range 50000--50999"
            ),
            "global_never_opened_claim_made": False,
        },
    }

    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(required["rows"], outputs[0])
    arm_rows = _public_arm_rows(by_arm)
    _write_csv(outputs[1], arm_rows)
    public_results = {
        "schema_version": SCHEMA_VERSION,
        "campaign": campaign.name,
        "source_artifacts": {
            name: {
                "relative_path": path.relative_to(campaign).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
        "validation": validation,
        "controller_label_map": {
            "baseline_pid": "delay-unaware proportional tracker",
            "delay_aware": "delay-aware tracker",
        },
        "by_arm": by_arm,
        "paired_delay_aware_minus_baseline_by_current": contrasts,
        "frozen_acceptance_criteria": summary["acceptance_criteria"],
        "frozen_acceptance_screens_by_current": summary[
            "acceptance_screens_by_current"
        ],
        "semantic_audit": semantic_audit,
    }
    _write_json(outputs[2], public_results)

    provenance_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{provenance_dir.name}.tmp-", dir=provenance_dir.parent)
    )
    try:
        records: list[dict[str, Any]] = []
        for relative, source in (
            ("control/campaign_contract.json", required["contract"]),
            ("campaign_summary.json", required["summary"]),
            ("decision.json", required["decision"]),
        ):
            document = read_json_object(source)
            sanitized, replacements = _sanitize(document, source_root_text)
            destination = temporary / relative
            _write_json(destination, sanitized)
            if source_root_text in destination.read_text(encoding="utf-8"):
                raise RuntimeError(f"private source root remains in {relative}")
            records.append(
                {
                    "relative_path": relative,
                    "original_sha256": sha256_file(source),
                    "exported_sha256": sha256_file(destination),
                    "source_root_replacements": replacements,
                    "scientific_values_changed": False,
                }
            )
        audit_path = temporary / "public_semantic_audit.json"
        _write_json(audit_path, semantic_audit)
        records.append(
            {
                "relative_path": audit_path.name,
                "original_sha256": None,
                "exported_sha256": sha256_file(audit_path),
                "source_root_replacements": 0,
                "scientific_values_changed": False,
                "kind": "additive public interpretation; frozen artifacts unchanged",
            }
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "campaign_name": campaign.name,
            "source_root_token": PRIVATE_ROOT_TOKEN,
            "scientific_values_changed": False,
            "only_frozen_record_transformation": "literal source-root path replacement",
            "semantic_audit_is_additive": True,
            "files": records,
        }
        _write_json(temporary / "provenance_export_manifest.json", manifest)
        temporary.replace(provenance_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return public_results


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--table-dir", type=Path, default=Path("data/tables"))
    parser.add_argument(
        "--provenance-dir",
        type=Path,
        default=Path("data/provenance/controller_repair"),
    )
    parser.add_argument("--source-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    result = export(
        args.campaign,
        args.table_dir,
        args.provenance_dir,
        source_root=args.source_root,
    )
    audit = result["semantic_audit"]
    print(
        "V41 publication export PASS: "
        f"{result['validation']['row_count']} row-level records; "
        f"decision={audit['public_unambiguous_decision_label']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
