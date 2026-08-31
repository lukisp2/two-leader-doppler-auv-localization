#!/usr/bin/env python3
"""Export and verify the row-level estimator benchmark behind manuscript Table 8.

The public CSV is a compact normalization of the frozen V27 estimator campaign
and the append-only V34 moving-horizon baseline.  A source-free check can be
run from the released CSV.  Regeneration from the archived campaign JSON files
requires the two DOI-archive directories supplied with ``--v27-dir`` and
``--v34-dir``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "data/tables/estimator_checkpoint_rows.csv"
DEFAULT_SUMMARY = ROOT / "data/tables/estimator_benchmark_summary.json"

V27_CAMPAIGN = "experiments_v27_publication_baselines_dev100"
V34_CAMPAIGN = "experiments_v34_mhe60_baseline_dev100"
PUBLIC_ARCHIVE_ALIASES = {
    V27_CAMPAIGN: "v27_campaign",
    V34_CAMPAIGN: "v34_campaign",
}
V27_CONTRACT_SHA256 = (
    "06a4b969dd591004a91d81b5bab3e8b49a282283a3912ff10bad7998f63784cd"
)
V34_CONTRACT_SHA256 = (
    "f86842e86735cadc4d1e664746faabc98ffba383de53b2ad32d4475c6d291f2b"
)
V27_SUMMARY_SHA256 = (
    "de37497c82934762b08fb8c6185b4828e54e23fb480ed67b3d1820c9f10df83e"
)
V34_SUMMARY_SHA256 = (
    "c917d0b6df51e580d2caf67bf682baf94cadc36ff12701065f047973fe389ac5"
)
CHECKPOINTS_S = (30, 60, 120, 240, 440)
EPISODE_INDICES = tuple(range(100))
EPISODE_SEEDS = tuple(range(45000, 45100))

METHOD_ORDER = (
    "legacy_pf",
    "pf_lw_4096",
    "pf_lw_16384",
    "ekf_static",
    "local_nls6",
    "coarse_only_8192",
    "global_window60",
    "global_full",
    "mhe60_arrival_fej",
)

METHOD_LABELS = {
    "legacy_pf": "Original particle filter, 1024",
    "pf_lw_4096": "Liu-West particle filter, 4096",
    "pf_lw_16384": "Liu-West particle filter, 16384",
    "ekf_static": "Static extended Kalman filter",
    "local_nls6": "Local nonlinear least squares, six starts",
    "coarse_only_8192": "Coarse global, no refinement",
    "global_window60": "Truncated-history least squares, last 60 s",
    "global_full": "Broad-search/local, full history",
    "mhe60_arrival_fej": "Moving horizon, 60-s window with FEJ arrival cost",
}

EXPECTED_MANUSCRIPT_ROWS = {
    "legacy_pf": ((11, 28, 53, 80, 87), ("2.108", "22.724", "65.303")),
    "pf_lw_4096": ((60, 76, 90, 91, 92), ("0.914", "7.920", "42.138")),
    "pf_lw_16384": ((73, 92, 100, 100, 100), ("0.598", "1.869", "6.261")),
    "ekf_static": ((0, 3, 7, 10, 12), ("56.649", "339.365", "2189.860")),
    "local_nls6": ((77, 94, 99, 100, 100), ("0.527", "1.527", "2.560")),
    "coarse_only_8192": ((8, 8, 5, 16, 9), ("15.987", "31.910", "40.256")),
    "global_window60": ((77, 94, 93, 72, 27), ("13.854", "178.037", "319.661")),
    "global_full": ((77, 94, 100, 100, 100), ("0.527", "1.527", "2.560")),
    "mhe60_arrival_fej": ((77, 94, 100, 100, 100), ("0.526", "1.519", "2.563")),
}

CSV_FIELDS = (
    "source_campaign",
    "method_id",
    "method_label",
    "episode_index",
    "episode_seed",
    "checkpoint_s",
    "endpoint_position_error_m",
    "initial_position_error_m",
    "success_le_7m",
    "success_lt_7m",
    "nominal_radius95_m",
    "nominal_radius95_covers",
    "residual_rmse_mps",
    "reported_runtime_s",
    "reported_runtime_definition",
    "checkpoint_update_runtime_s",
    "online_inputs_sha256",
    "truth_labels_sha256",
    "unscored_sha256",
    "contract_sha256",
    "source_result_sha256",
)

BOOTSTRAP_SEED = 34045000
BOOTSTRAP_RESAMPLES = 50000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _finite_or_none(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite value in {field}")
    return number


def _bool_or_none(value: Any, *, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise RuntimeError(f"expected Boolean or null in {field}")
    return value


def _validate_campaign(directory: Path, *, campaign: str, cells: int) -> None:
    accepted_names = {campaign, PUBLIC_ARCHIVE_ALIASES.get(campaign)} - {None}
    if directory.name not in accepted_names:
        expected = " or ".join(sorted(accepted_names))
        raise RuntimeError(
            f"expected archived directory named {expected}, got {directory.name}"
        )
    summary = _read_json(directory / "campaign_summary.json")
    audit = _read_json(directory / "independent_audit.json")
    if summary.get("status") != "complete" or bool(summary.get("smoke", False)):
        raise RuntimeError(f"{campaign} is not a complete non-smoke campaign")
    if int(summary.get("episode_count", -1)) != 100:
        raise RuntimeError(f"{campaign} does not contain 100 episodes")
    if int(summary.get("cell_count", -1)) != cells:
        raise RuntimeError(f"{campaign} does not contain {cells} cells")
    if not bool(audit.get("valid", False)):
        raise RuntimeError(f"{campaign} independent audit is not valid")


def _normalize_source_result(
    path: Path,
    *,
    campaign: str,
    expected_contract: str,
) -> dict[str, Any]:
    wrapper = _read_json(path)
    score = wrapper.get("score")
    unscored = wrapper.get("unscored")
    if not isinstance(score, dict) or not isinstance(unscored, dict):
        raise RuntimeError(f"missing score/unscored objects: {path}")

    method = str(wrapper.get("arm"))
    if method not in METHOD_ORDER:
        raise RuntimeError(f"unexpected estimator arm {method!r}: {path}")
    checkpoint = int(
        round(float(wrapper.get("prefix_s", wrapper.get("checkpoint_s"))))
    )
    if checkpoint not in CHECKPOINTS_S:
        raise RuntimeError(f"unexpected checkpoint {checkpoint}: {path}")
    if not math.isclose(
        float(score.get("endpoint_s")), checkpoint, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"score endpoint does not match checkpoint: {path}")
    episode_index = int(wrapper.get("episode_index"))
    episode_seed = int(wrapper.get("episode_seed"))
    if episode_index not in EPISODE_INDICES or episode_seed != 45000 + episode_index:
        raise RuntimeError(f"unexpected episode identity: {path}")
    contract = str(wrapper.get("contract_sha256"))
    if contract != expected_contract:
        raise RuntimeError(f"unexpected contract hash: {path}")
    if unscored.get("arm") != method:
        raise RuntimeError(f"unscored arm mismatch: {path}")

    if campaign == V27_CAMPAIGN:
        runtime = _finite_or_none(unscored.get("runtime_s"), field="runtime_s")
        runtime_definition = (
            "not recorded for historical trace"
            if runtime is None
            else "single causal-prefix estimator evaluation"
        )
        checkpoint_runtime = None
        residual = _finite_or_none(
            unscored.get("residual_rmse_mps"), field="residual_rmse_mps"
        )
    elif campaign == V34_CAMPAIGN:
        if method != "mhe60_arrival_fej":
            raise RuntimeError(f"unexpected method in V34 archive: {path}")
        runtime = _finite_or_none(
            unscored.get("cumulative_runtime_s"), field="cumulative_runtime_s"
        )
        runtime_definition = "cumulative estimator CPU time from 30 s"
        checkpoint_runtime = _finite_or_none(
            unscored.get("checkpoint_update_runtime_s"),
            field="checkpoint_update_runtime_s",
        )
        residual = _finite_or_none(
            unscored.get("full_prefix_residual_rmse_mps"),
            field="full_prefix_residual_rmse_mps",
        )
    else:
        raise RuntimeError(f"unknown campaign {campaign}")

    return {
        "source_campaign": campaign,
        "method_id": method,
        "method_label": METHOD_LABELS[method],
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "checkpoint_s": checkpoint,
        "endpoint_position_error_m": _finite_or_none(
            score.get("endpoint_position_error_m"),
            field="endpoint_position_error_m",
        ),
        "initial_position_error_m": _finite_or_none(
            score.get("initial_position_error_m"),
            field="initial_position_error_m",
        ),
        "success_le_7m": _bool_or_none(
            score.get("success_le_7m"), field="success_le_7m"
        ),
        "success_lt_7m": _bool_or_none(
            score.get("success_lt_7m"), field="success_lt_7m"
        ),
        "nominal_radius95_m": _finite_or_none(
            unscored.get("nominal_radius95_m"), field="nominal_radius95_m"
        ),
        "nominal_radius95_covers": _bool_or_none(
            score.get("nominal_radius95_covers"),
            field="nominal_radius95_covers",
        ),
        "residual_rmse_mps": residual,
        "reported_runtime_s": runtime,
        "reported_runtime_definition": runtime_definition,
        "checkpoint_update_runtime_s": checkpoint_runtime,
        "online_inputs_sha256": str(wrapper.get("online_inputs_sha256")),
        "truth_labels_sha256": str(wrapper.get("truth_labels_sha256")),
        "unscored_sha256": str(wrapper.get("unscored_sha256")),
        "contract_sha256": contract,
        "source_result_sha256": sha256_file(path),
    }


def load_archived_rows(v27_dir: Path, v34_dir: Path) -> list[dict[str, Any]]:
    v27_dir = v27_dir.resolve()
    v34_dir = v34_dir.resolve()
    _validate_campaign(v27_dir, campaign=V27_CAMPAIGN, cells=4000)
    _validate_campaign(v34_dir, campaign=V34_CAMPAIGN, cells=500)

    rows: list[dict[str, Any]] = []
    for path in sorted((v27_dir / "episode_results").glob("*.json")):
        rows.append(
            _normalize_source_result(
                path,
                campaign=V27_CAMPAIGN,
                expected_contract=V27_CONTRACT_SHA256,
            )
        )
    for path in sorted((v34_dir / "episode_results").glob("*.json")):
        rows.append(
            _normalize_source_result(
                path,
                campaign=V34_CAMPAIGN,
                expected_contract=V34_CONTRACT_SHA256,
            )
        )
    validate_rows(rows)
    ordered = sort_rows(rows)
    cross_check_archived_campaign_summaries(ordered, v27_dir, v34_dir)
    return ordered


def sort_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            order[str(row["method_id"])],
            int(row["checkpoint_s"]),
            int(row["episode_index"]),
        ),
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in CSV_FIELDS})
    temporary.replace(path)


def _parse_optional_float(value: str, *, field: str) -> float | None:
    return None if value == "" else _finite_or_none(value, field=field)


def _parse_optional_bool(value: str, *, field: str) -> bool | None:
    if value == "":
        return None
    if value not in {"true", "false"}:
        raise RuntimeError(f"invalid Boolean in {field}: {value!r}")
    return value == "true"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RuntimeError(f"unexpected estimator CSV schema: {path}")
        for source in reader:
            rows.append(
                {
                    **source,
                    "episode_index": int(source["episode_index"]),
                    "episode_seed": int(source["episode_seed"]),
                    "checkpoint_s": int(source["checkpoint_s"]),
                    "endpoint_position_error_m": _parse_optional_float(
                        source["endpoint_position_error_m"],
                        field="endpoint_position_error_m",
                    ),
                    "initial_position_error_m": _parse_optional_float(
                        source["initial_position_error_m"],
                        field="initial_position_error_m",
                    ),
                    "success_le_7m": _parse_optional_bool(
                        source["success_le_7m"], field="success_le_7m"
                    ),
                    "success_lt_7m": _parse_optional_bool(
                        source["success_lt_7m"], field="success_lt_7m"
                    ),
                    "nominal_radius95_m": _parse_optional_float(
                        source["nominal_radius95_m"], field="nominal_radius95_m"
                    ),
                    "nominal_radius95_covers": _parse_optional_bool(
                        source["nominal_radius95_covers"],
                        field="nominal_radius95_covers",
                    ),
                    "residual_rmse_mps": _parse_optional_float(
                        source["residual_rmse_mps"], field="residual_rmse_mps"
                    ),
                    "reported_runtime_s": _parse_optional_float(
                        source["reported_runtime_s"], field="reported_runtime_s"
                    ),
                    "checkpoint_update_runtime_s": _parse_optional_float(
                        source["checkpoint_update_runtime_s"],
                        field="checkpoint_update_runtime_s",
                    ),
                }
            )
    validate_rows(rows)
    return sort_rows(rows)


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 4500:
        raise RuntimeError(f"expected exactly 4500 estimator rows, found {len(rows)}")

    expected = {
        (method, checkpoint, episode)
        for method in METHOD_ORDER
        for checkpoint in CHECKPOINTS_S
        for episode in EPISODE_INDICES
    }
    observed: set[tuple[str, int, int]] = set()
    for row in rows:
        method = str(row["method_id"])
        checkpoint = int(row["checkpoint_s"])
        episode = int(row["episode_index"])
        identity = (method, checkpoint, episode)
        if identity in observed:
            raise RuntimeError(f"duplicate estimator cell {identity}")
        observed.add(identity)
        if row["method_label"] != METHOD_LABELS.get(method):
            raise RuntimeError(f"method label mismatch for {method}")
        if int(row["episode_seed"]) != 45000 + episode:
            raise RuntimeError(f"seed/index mismatch for {identity}")
        expected_campaign = (
            V34_CAMPAIGN if method == "mhe60_arrival_fej" else V27_CAMPAIGN
        )
        expected_contract = (
            V34_CONTRACT_SHA256
            if method == "mhe60_arrival_fej"
            else V27_CONTRACT_SHA256
        )
        if row["source_campaign"] != expected_campaign:
            raise RuntimeError(f"campaign mismatch for {identity}")
        if row["contract_sha256"] != expected_contract:
            raise RuntimeError(f"contract mismatch for {identity}")
        error = _finite_or_none(
            row["endpoint_position_error_m"], field="endpoint_position_error_m"
        )
        if error is None:
            raise RuntimeError(f"missing endpoint error for {identity}")
        success_le = _bool_or_none(row["success_le_7m"], field="success_le_7m")
        success_lt = _bool_or_none(row["success_lt_7m"], field="success_lt_7m")
        if success_le is None or success_lt is None:
            raise RuntimeError(f"missing success label for {identity}")
        if success_le != (error <= 7.0) or success_lt != (error < 7.0):
            raise RuntimeError(f"success label/error mismatch for {identity}")
        for hash_field in (
            "online_inputs_sha256",
            "truth_labels_sha256",
            "unscored_sha256",
            "source_result_sha256",
        ):
            value = str(row[hash_field])
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise RuntimeError(f"invalid {hash_field} for {identity}")

    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise RuntimeError(
            f"estimator cell identity mismatch: missing={len(missing)}, extra={len(extra)}"
        )


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("statistics require a non-empty finite vector")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _bootstrap_mean_ci(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (100,) or not np.all(np.isfinite(values)):
        raise RuntimeError("paired MHE comparison requires 100 finite differences")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    chunks: list[np.ndarray] = []
    remaining = BOOTSTRAP_RESAMPLES
    while remaining:
        count = min(5000, remaining)
        indices = rng.integers(0, values.size, size=(count, values.size), endpoint=False)
        chunks.append(np.mean(values[indices], axis=1))
        remaining -= count
    samples = np.concatenate(chunks)
    return {
        "mean": float(np.mean(values)),
        "lower95": float(np.percentile(samples, 2.5)),
        "upper95": float(np.percentile(samples, 97.5)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _require_same_number(observed: Any, archived: Any, *, field: str) -> None:
    if isinstance(observed, int) and isinstance(archived, int):
        if observed != archived:
            raise RuntimeError(f"archived aggregate mismatch in {field}")
        return
    if not math.isclose(
        float(observed), float(archived), rel_tol=0.0, abs_tol=2.0e-12
    ):
        raise RuntimeError(
            f"archived aggregate mismatch in {field}: {observed} != {archived}"
        )


def cross_check_archived_campaign_summaries(
    rows: Sequence[Mapping[str, Any]], v27_dir: Path, v34_dir: Path
) -> None:
    """Prove that normalization preserves every Table-7 source aggregate."""

    if sha256_file(v27_dir / "campaign_summary.json") != V27_SUMMARY_SHA256:
        raise RuntimeError("unexpected V27 campaign-summary hash")
    if sha256_file(v34_dir / "campaign_summary.json") != V34_SUMMARY_SHA256:
        raise RuntimeError("unexpected V34 campaign-summary hash")
    archived_v27 = _read_json(v27_dir / "campaign_summary.json")["aggregate"]
    archived_v34 = _read_json(v34_dir / "campaign_summary.json")["aggregate"]
    normalized = aggregate_rows(rows)

    for method in METHOD_ORDER[:-1]:
        for checkpoint in CHECKPOINTS_S:
            observed = normalized["methods"][method]["checkpoints"][str(checkpoint)]
            archived = archived_v27["by_arm_prefix"][f"{method}@{checkpoint}"]
            _require_same_number(
                observed["count"], archived["count"], field=f"{method}@{checkpoint}.count"
            )
            _require_same_number(
                observed["success_le_7m_count"],
                archived["success_le_7m_count"],
                field=f"{method}@{checkpoint}.success_le_7m_count",
            )
            for statistic in ("count", "mean", "median", "p95", "max"):
                _require_same_number(
                    observed["endpoint_position_error_m"][statistic],
                    archived["endpoint_error_m"][statistic],
                    field=f"{method}@{checkpoint}.endpoint_error_m.{statistic}",
                )

    method = "mhe60_arrival_fej"
    for checkpoint in CHECKPOINTS_S:
        observed = normalized["methods"][method]["checkpoints"][str(checkpoint)]
        archived = archived_v34["by_checkpoint"][str(checkpoint)]
        _require_same_number(
            observed["count"], archived["count"], field=f"{method}@{checkpoint}.count"
        )
        _require_same_number(
            observed["success_le_7m_count"],
            archived["success_le_7m_count"],
            field=f"{method}@{checkpoint}.success_le_7m_count",
        )
        for statistic in ("count", "mean", "median", "p95", "max"):
            _require_same_number(
                observed["endpoint_position_error_m"][statistic],
                archived["endpoint_error_m"][statistic],
                field=f"{method}@{checkpoint}.endpoint_error_m.{statistic}",
            )

    observed_paired = normalized[
        "paired_mhe_minus_full_history_error_m_at_440s"
    ]
    archived_paired = archived_v34[
        "paired_mhe_minus_v27_global_full_error_m"
    ]["440"]
    for statistic in ("mean", "lower95", "upper95", "resamples", "seed"):
        _require_same_number(
            observed_paired[statistic],
            archived_paired[statistic],
            field=f"paired_440.{statistic}",
        )


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_rows(rows)
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method_id"]), int(row["checkpoint_s"]))].append(row)

    methods: dict[str, Any] = {}
    for method in METHOD_ORDER:
        checkpoints: dict[str, Any] = {}
        for checkpoint in CHECKPOINTS_S:
            subset = grouped[(method, checkpoint)]
            checkpoints[str(checkpoint)] = {
                "count": len(subset),
                "success_le_7m_count": sum(
                    bool(row["success_le_7m"]) for row in subset
                ),
                "endpoint_position_error_m": _stats(
                    float(row["endpoint_position_error_m"]) for row in subset
                ),
            }
        methods[method] = {
            "method_label": METHOD_LABELS[method],
            "checkpoints": checkpoints,
        }

    full_440 = {
        int(row["episode_index"]): float(row["endpoint_position_error_m"])
        for row in grouped[("global_full", 440)]
    }
    mhe_440 = {
        int(row["episode_index"]): float(row["endpoint_position_error_m"])
        for row in grouped[("mhe60_arrival_fej", 440)]
    }
    if set(full_440) != set(mhe_440):
        raise RuntimeError("MHE/full-history terminal pairing is incomplete")
    differences = np.asarray(
        [mhe_440[index] - full_440[index] for index in sorted(full_440)],
        dtype=np.float64,
    )

    return {
        "schema_version": 1,
        "row_count": len(rows),
        "episode_count": 100,
        "checkpoints_s": list(CHECKPOINTS_S),
        "criterion": "endpoint_position_error_m <= 7.0",
        "source_campaigns": {
            V27_CAMPAIGN: {
                "row_count": 4000,
                "contract_sha256": V27_CONTRACT_SHA256,
                "campaign_summary_sha256": V27_SUMMARY_SHA256,
            },
            V34_CAMPAIGN: {
                "row_count": 500,
                "contract_sha256": V34_CONTRACT_SHA256,
                "campaign_summary_sha256": V34_SUMMARY_SHA256,
            },
        },
        "methods": methods,
        "paired_mhe_minus_full_history_error_m_at_440s": _bootstrap_mean_ci(
            differences
        ),
    }


def cross_check_manuscript(summary: Mapping[str, Any]) -> None:
    methods = summary["methods"]
    for method in METHOD_ORDER:
        expected_counts, expected_tail = EXPECTED_MANUSCRIPT_ROWS[method]
        observed_counts = tuple(
            int(methods[method]["checkpoints"][str(checkpoint)]["success_le_7m_count"])
            for checkpoint in CHECKPOINTS_S
        )
        if observed_counts != expected_counts:
            raise RuntimeError(
                f"manuscript success-count mismatch for {method}: "
                f"{observed_counts} != {expected_counts}"
            )
        terminal = methods[method]["checkpoints"]["440"][
            "endpoint_position_error_m"
        ]
        observed_tail = (
            f"{float(terminal['median']):.3f}",
            f"{float(terminal['p95']):.3f}",
            f"{float(terminal['max']):.3f}",
        )
        if observed_tail != expected_tail:
            raise RuntimeError(
                f"manuscript terminal-tail mismatch for {method}: "
                f"{observed_tail} != {expected_tail}"
            )

    paired = summary["paired_mhe_minus_full_history_error_m_at_440s"]
    observed_paired = (
        f"{float(paired['mean']):.4f}",
        f"{float(paired['lower95']):.4f}",
        f"{float(paired['upper95']):.4f}",
    )
    expected_paired = ("-0.0022", "-0.0061", "0.0008")
    if observed_paired != expected_paired:
        raise RuntimeError(
            "manuscript paired MHE/full-history mismatch: "
            f"{observed_paired} != {expected_paired}"
        )


def write_summary(path: Path, summary: Mapping[str, Any], rows_path: Path) -> None:
    output = dict(summary)
    output["row_file"] = rows_path.name
    output["row_file_sha256"] = sha256_file(rows_path)
    output["manuscript_cross_check"] = "PASS"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v27-dir", type=Path)
    parser.add_argument("--v34-dir", type=Path)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate released rows and manuscript values without writing files",
    )
    args = parser.parse_args()

    if (args.v27_dir is None) != (args.v34_dir is None):
        parser.error("--v27-dir and --v34-dir must be supplied together")
    if args.check_only and args.v27_dir is not None:
        parser.error("--check-only cannot be combined with archived source directories")

    if args.v27_dir is not None:
        rows = load_archived_rows(args.v27_dir, args.v34_dir)
        write_rows(args.rows, rows)
    else:
        rows = load_rows(args.rows)

    summary = aggregate_rows(rows)
    cross_check_manuscript(summary)
    if args.v27_dir is not None:
        summary["archived_campaign_aggregate_cross_check"] = "PASS"
    if not args.check_only:
        write_summary(args.summary, summary, args.rows)
    print(
        "Estimator benchmark PASS: 4500 rows, 45 complete method-checkpoint "
        "cells, and all manuscript counts/tails match."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
