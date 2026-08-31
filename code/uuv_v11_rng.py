# -*- coding: utf-8 -*-
"""Deterministic, separated random streams for ``v11_online``.

The v8/v10 environment uses one ``self.rng`` for four different jobs:

* drawing the initial scenario,
* dead-reckoning measurement errors,
* Doppler measurement errors, and
* particle-filter propagation, resampling, roughening and injection.

Consequently, a controller-dependent PF resampling decision changes the random
numbers later used as *physical* measurement noise.  That invalidates paired
controller comparisons.  This module is the deliberately small RNG boundary
for v11; it does not import or modify v8/v10.

Intended environment integration
--------------------------------

At episode reset, construct one :class:`EpisodeSeedPlan` from the experiment
seed, episode index and vector-environment rank.  Build
:class:`EpisodeRNGStreams` from it, then use the streams as follows::

    self.rng_scenario = streams.scenario
    self.rng_sensor = streams.sensor
    self.rng_dead_reckoning = streams.dead_reckoning
    self.rng_pf = streams.pf
    self.rng_controller = streams.controller
    self.pf.rng = self.rng_pf

All stochastic initial-condition draws must use ``rng_scenario``.  In training,
the sensor/dead-reckoning generators may be consumed directly because streams
remain isolated.  In paired evaluation, generate and save one
:class:`ExogenousNoiseTape` per episode, and index it by simulation substep and
Doppler measurement number for every controller.  PF/controller randomness is
never taken from that tape.

Named streams are derived independently rather than with ``SeedSequence.spawn``.
Adding or requesting another stream therefore cannot shift existing streams.
The explicit PCG64 bit generator and schema version are recorded in replay
artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np


RNG_SCHEMA_VERSION = 1
TAPE_SCHEMA_VERSION = 1
BIT_GENERATOR_NAME = "PCG64"

SHARED_STREAM_NAMES = (
    "scenario",
    "sensor",
    "dead_reckoning",
    "pf",
)
ALL_STREAM_NAMES = SHARED_STREAM_NAMES + ("controller",)

_DOMAIN = "uuv-v11-online-rng-v1"


PathLike = Union[str, Path]


def _require_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if value > (1 << 64) - 1:
        raise ValueError(f"{name} must fit in uint64, got {value}")
    return value


def _uint64_words(value: int) -> Sequence[int]:
    value = _require_nonnegative_int("seed component", value)
    return (value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF)


def _stable_words(label: str) -> Sequence[int]:
    """Map a label to four stable uint32 words (unlike Python's ``hash``)."""

    if not isinstance(label, str) or not label:
        raise ValueError("stream/controller label must be a non-empty string")
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return tuple(
        int.from_bytes(digest[offset : offset + 4], "little", signed=False)
        for offset in range(0, 16, 4)
    )


@dataclass(frozen=True)
class EpisodeSeedPlan:
    """Stable key for all random streams belonging to one environment episode.

    ``controller_id`` is intentionally *not* part of this plan.  It is used only
    while deriving the controller stream, so scenario/sensor/dead-reckoning/PF
    draws are paired across controllers.
    """

    root_seed: int
    episode_index: int
    env_rank: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "root_seed", _require_nonnegative_int("root_seed", self.root_seed)
        )
        object.__setattr__(
            self,
            "episode_index",
            _require_nonnegative_int("episode_index", self.episode_index),
        )
        object.__setattr__(
            self, "env_rank", _require_nonnegative_int("env_rank", self.env_rank)
        )

    def seed_sequence(
        self, stream_name: str, *, controller_id: Optional[str] = None
    ) -> np.random.SeedSequence:
        """Return an independently keyed seed sequence for a named stream."""

        if stream_name not in ALL_STREAM_NAMES:
            allowed = ", ".join(ALL_STREAM_NAMES)
            raise ValueError(f"unknown stream {stream_name!r}; expected one of: {allowed}")

        entropy = list(_stable_words(_DOMAIN))
        entropy.extend(_uint64_words(self.root_seed))
        entropy.extend(_uint64_words(self.episode_index))
        entropy.extend(_uint64_words(self.env_rank))
        entropy.extend(_stable_words(stream_name))

        if stream_name == "controller":
            if controller_id is None:
                raise ValueError("controller_id is required for the controller stream")
            entropy.extend(_stable_words(str(controller_id)))
        elif controller_id is not None:
            raise ValueError(
                "controller_id may only key the controller stream; shared streams "
                "must remain controller-independent"
            )

        return np.random.SeedSequence(entropy)

    def generator(
        self, stream_name: str, *, controller_id: Optional[str] = None
    ) -> np.random.Generator:
        """Construct a fresh PCG64 generator at the start of a named stream."""

        seed_sequence = self.seed_sequence(
            stream_name, controller_id=controller_id
        )
        return np.random.Generator(np.random.PCG64(seed_sequence))

    def to_dict(self) -> Dict[str, int]:
        return {
            "root_seed": self.root_seed,
            "episode_index": self.episode_index,
            "env_rank": self.env_rank,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeSeedPlan":
        return cls(
            root_seed=int(value["root_seed"]),
            episode_index=int(value["episode_index"]),
            env_rank=int(value.get("env_rank", 0)),
        )


class EpisodeRNGStreams:
    """Owned generator instances for one episode.

    Reconstructing this object with the same plan and controller ID replays all
    streams from their beginnings.  ``checkpoint``/``from_checkpoint`` restores
    their exact current positions (the environment state still has to be logged
    separately for a mid-episode resume).
    """

    def __init__(self, plan: EpisodeSeedPlan, controller_id: str) -> None:
        if not isinstance(plan, EpisodeSeedPlan):
            raise TypeError("plan must be an EpisodeSeedPlan")
        if not isinstance(controller_id, str) or not controller_id:
            raise ValueError("controller_id must be a non-empty string")
        self.plan = plan
        self.controller_id = controller_id
        self._generators: Dict[str, np.random.Generator] = {
            name: plan.generator(name) for name in SHARED_STREAM_NAMES
        }
        self._generators["controller"] = plan.generator(
            "controller", controller_id=controller_id
        )

    def generator(self, stream_name: str) -> np.random.Generator:
        try:
            return self._generators[stream_name]
        except KeyError as exc:
            allowed = ", ".join(ALL_STREAM_NAMES)
            raise ValueError(
                f"unknown stream {stream_name!r}; expected one of: {allowed}"
            ) from exc

    @property
    def scenario(self) -> np.random.Generator:
        return self._generators["scenario"]

    @property
    def sensor(self) -> np.random.Generator:
        return self._generators["sensor"]

    @property
    def dead_reckoning(self) -> np.random.Generator:
        return self._generators["dead_reckoning"]

    @property
    def pf(self) -> np.random.Generator:
        return self._generators["pf"]

    @property
    def controller(self) -> np.random.Generator:
        return self._generators["controller"]

    def checkpoint(self) -> Dict[str, Any]:
        """Return a JSON-serializable checkpoint of every generator position."""

        return {
            "schema_version": RNG_SCHEMA_VERSION,
            "bit_generator": BIT_GENERATOR_NAME,
            "seed_plan": self.plan.to_dict(),
            "controller_id": self.controller_id,
            "states": {
                name: copy.deepcopy(generator.bit_generator.state)
                for name, generator in self._generators.items()
            },
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "EpisodeRNGStreams":
        if int(checkpoint.get("schema_version", -1)) != RNG_SCHEMA_VERSION:
            raise ValueError("unsupported RNG checkpoint schema")
        if str(checkpoint.get("bit_generator", "")) != BIT_GENERATOR_NAME:
            raise ValueError("unsupported RNG checkpoint bit generator")

        streams = cls(
            EpisodeSeedPlan.from_dict(checkpoint["seed_plan"]),
            str(checkpoint["controller_id"]),
        )
        states = checkpoint.get("states", {})
        if set(states) != set(ALL_STREAM_NAMES):
            raise ValueError("RNG checkpoint does not contain exactly all named streams")
        for name in ALL_STREAM_NAMES:
            state = copy.deepcopy(states[name])
            if str(state.get("bit_generator", "")) != BIT_GENERATOR_NAME:
                raise ValueError(f"invalid bit generator state for stream {name!r}")
            streams._generators[name].bit_generator.state = state
        return streams

    def save_checkpoint_json(self, path: PathLike) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.checkpoint(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load_checkpoint_json(cls, path: PathLike) -> "EpisodeRNGStreams":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        return cls.from_checkpoint(checkpoint)


@dataclass(frozen=True)
class ExogenousNoiseTape:
    """Controller-independent standard-normal errors for paired evaluation.

    ``dead_reckoning_z[k]`` contains speed/yaw/pitch errors for simulation
    substep ``k``.  ``doppler_z[m]`` contains leader-1/leader-2 Doppler errors
    for measurement epoch ``m``.  Values are standard normal; applying physical
    standard deviations at use time keeps the tape reusable and auditable.
    """

    seed_plan: EpisodeSeedPlan
    dead_reckoning_z: np.ndarray
    doppler_z: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.seed_plan, EpisodeSeedPlan):
            raise TypeError("seed_plan must be an EpisodeSeedPlan")

        # Own the arrays so making the tape read-only never changes flags on an
        # array supplied by the caller.
        dead = np.array(self.dead_reckoning_z, dtype="<f8", order="C", copy=True)
        doppler = np.array(self.doppler_z, dtype="<f8", order="C", copy=True)
        if dead.ndim != 2 or dead.shape[1] != 3:
            raise ValueError(
                "dead_reckoning_z must have shape (n_substeps, 3) for "
                "speed/yaw/pitch"
            )
        if doppler.ndim != 2 or doppler.shape[1] != 2:
            raise ValueError(
                "doppler_z must have shape (n_measurements, 2) for the two leaders"
            )
        if not np.all(np.isfinite(dead)) or not np.all(np.isfinite(doppler)):
            raise ValueError("noise tape contains non-finite values")

        dead.setflags(write=False)
        doppler.setflags(write=False)
        object.__setattr__(self, "dead_reckoning_z", dead)
        object.__setattr__(self, "doppler_z", doppler)

    @classmethod
    def generate(
        cls,
        seed_plan: EpisodeSeedPlan,
        *,
        n_substeps: int,
        n_doppler_measurements: int,
    ) -> "ExogenousNoiseTape":
        n_substeps = _require_nonnegative_int("n_substeps", n_substeps)
        n_doppler_measurements = _require_nonnegative_int(
            "n_doppler_measurements", n_doppler_measurements
        )
        dead_rng = seed_plan.generator("dead_reckoning")
        sensor_rng = seed_plan.generator("sensor")
        return cls(
            seed_plan=seed_plan,
            dead_reckoning_z=dead_rng.standard_normal((n_substeps, 3)),
            doppler_z=sensor_rng.standard_normal((n_doppler_measurements, 2)),
        )

    @property
    def n_substeps(self) -> int:
        return int(self.dead_reckoning_z.shape[0])

    @property
    def n_doppler_measurements(self) -> int:
        return int(self.doppler_z.shape[0])

    def dead_reckoning_at(
        self, substep_index: int, sigma_speed_yaw_pitch: Sequence[float]
    ) -> np.ndarray:
        index = _checked_index("substep_index", substep_index, self.n_substeps)
        scale = _checked_scale("sigma_speed_yaw_pitch", sigma_speed_yaw_pitch, 3)
        return self.dead_reckoning_z[index] * scale

    def doppler_at(
        self, measurement_index: int, sigma_leader_1_2: Union[float, Sequence[float]]
    ) -> np.ndarray:
        index = _checked_index(
            "measurement_index", measurement_index, self.n_doppler_measurements
        )
        scale = _checked_scale("sigma_leader_1_2", sigma_leader_1_2, 2)
        return self.doppler_z[index] * scale

    def metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": TAPE_SCHEMA_VERSION,
            "rng_schema_version": RNG_SCHEMA_VERSION,
            "bit_generator": BIT_GENERATOR_NAME,
            "numpy_version_when_written": np.__version__,
            "seed_plan": self.seed_plan.to_dict(),
            "n_substeps": self.n_substeps,
            "n_doppler_measurements": self.n_doppler_measurements,
            "dead_reckoning_columns": ["speed", "yaw", "pitch"],
            "doppler_columns": ["leader_1", "leader_2"],
            "distribution": "standard_normal",
        }

    def content_sha256(self) -> str:
        # Deliberately exclude the NumPy writer version from the digest.  It is
        # useful provenance, but the identity of a loaded tape must not change
        # merely because it is inspected on another machine/version.
        metadata_dict = self.metadata()
        metadata_dict.pop("numpy_version_when_written", None)
        metadata_json = json.dumps(
            metadata_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(b"uuv-v11-exogenous-noise-tape\x00")
        digest.update(metadata_json)
        for name, array in (
            ("dead_reckoning_z", self.dead_reckoning_z),
            ("doppler_z", self.doppler_z),
        ):
            canonical = np.asarray(array, dtype="<f8", order="C")
            digest.update(name.encode("ascii") + b"\x00")
            digest.update(str(canonical.shape).encode("ascii") + b"\x00")
            digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()

    def save_npz(self, path: PathLike) -> None:
        """Save values, provenance and a content digest to an NPZ artifact."""

        path = Path(path)
        metadata = self.metadata()
        metadata_json = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        content_sha256 = self.content_sha256()
        # Pass a file handle so NumPy does not silently append another suffix.
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                dead_reckoning_z=self.dead_reckoning_z,
                doppler_z=self.doppler_z,
                metadata_json=np.asarray(metadata_json),
                content_sha256=np.asarray(content_sha256),
            )

    @classmethod
    def load_npz(
        cls, path: PathLike, *, verify_digest: bool = True
    ) -> "ExogenousNoiseTape":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "dead_reckoning_z",
                "doppler_z",
                "metadata_json",
                "content_sha256",
            }
            if set(archive.files) != required:
                raise ValueError("noise tape NPZ has missing or unexpected fields")
            metadata_json = str(archive["metadata_json"].item())
            expected_digest = str(archive["content_sha256"].item())
            dead = np.array(archive["dead_reckoning_z"], dtype="<f8", copy=True)
            doppler = np.array(archive["doppler_z"], dtype="<f8", copy=True)

        metadata = json.loads(metadata_json)
        if int(metadata.get("schema_version", -1)) != TAPE_SCHEMA_VERSION:
            raise ValueError("unsupported noise tape schema")
        if int(metadata.get("rng_schema_version", -1)) != RNG_SCHEMA_VERSION:
            raise ValueError("unsupported RNG schema in noise tape")
        if str(metadata.get("bit_generator", "")) != BIT_GENERATOR_NAME:
            raise ValueError("unsupported bit generator in noise tape")

        tape = cls(
            seed_plan=EpisodeSeedPlan.from_dict(metadata["seed_plan"]),
            dead_reckoning_z=dead,
            doppler_z=doppler,
        )
        if tape.n_substeps != int(metadata.get("n_substeps", -1)):
            raise ValueError("noise tape substep count does not match metadata")
        if tape.n_doppler_measurements != int(
            metadata.get("n_doppler_measurements", -1)
        ):
            raise ValueError("noise tape measurement count does not match metadata")
        if verify_digest and tape.content_sha256() != expected_digest:
            raise ValueError("noise tape content digest mismatch")
        return tape


class ExogenousNoiseCursor:
    """Sequential facade over a tape, with replayable integer positions."""

    def __init__(self, tape: ExogenousNoiseTape) -> None:
        if not isinstance(tape, ExogenousNoiseTape):
            raise TypeError("tape must be an ExogenousNoiseTape")
        self.tape = tape
        self.substep_index = 0
        self.measurement_index = 0

    def next_dead_reckoning(
        self, sigma_speed_yaw_pitch: Sequence[float]
    ) -> np.ndarray:
        value = self.tape.dead_reckoning_at(
            self.substep_index, sigma_speed_yaw_pitch
        )
        self.substep_index += 1
        return value

    def next_doppler(
        self, sigma_leader_1_2: Union[float, Sequence[float]]
    ) -> np.ndarray:
        value = self.tape.doppler_at(
            self.measurement_index, sigma_leader_1_2
        )
        self.measurement_index += 1
        return value

    def state_dict(self) -> Dict[str, int]:
        return {
            "substep_index": self.substep_index,
            "measurement_index": self.measurement_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        substep_index = _require_nonnegative_int(
            "substep_index", int(state["substep_index"])
        )
        measurement_index = _require_nonnegative_int(
            "measurement_index", int(state["measurement_index"])
        )
        if substep_index > self.tape.n_substeps:
            raise ValueError("cursor substep position is beyond the tape")
        if measurement_index > self.tape.n_doppler_measurements:
            raise ValueError("cursor measurement position is beyond the tape")
        self.substep_index = substep_index
        self.measurement_index = measurement_index


def _checked_index(name: str, value: int, length: int) -> int:
    value = _require_nonnegative_int(name, value)
    if value >= length:
        raise IndexError(f"{name}={value} is outside a tape of length {length}")
    return value


def _checked_scale(
    name: str, value: Union[float, Sequence[float]], expected_size: int
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(expected_size, float(array), dtype=float)
    else:
        array = array.reshape(-1)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must be scalar or have {expected_size} values")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    return array


__all__ = [
    "ALL_STREAM_NAMES",
    "BIT_GENERATOR_NAME",
    "EpisodeRNGStreams",
    "EpisodeSeedPlan",
    "ExogenousNoiseCursor",
    "ExogenousNoiseTape",
    "RNG_SCHEMA_VERSION",
    "SHARED_STREAM_NAMES",
    "TAPE_SCHEMA_VERSION",
]
