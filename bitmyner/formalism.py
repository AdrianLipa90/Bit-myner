from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Iterable

import numpy as np

DIM = 36
KAPPA = math.log(2.0) / (24.0 * math.pi)
Q28_MODE_QUBITS = 6
Q28_ADDRESS_CAPACITY = 1 << Q28_MODE_QUBITS
GREMLIN_ROLES = ("SPIDER:correctness", "ANT:parallelism", "MANTIS:tuning", "BELZEBUB:promotion")


def canonical_json_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def commitment(obj: object) -> str:
    return hashlib.blake2b(canonical_json_bytes(obj), digest_size=32).hexdigest()


@dataclass(frozen=True, slots=True)
class SchedulerCandidate:
    address6: int
    kernel_variant: str
    threads: int
    batch_size: int
    phase_coordinate: int
    genome_id: str
    route_commitment: str


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    candidate: SchedulerCandidate
    hps_samples: tuple[float, ...]
    median_hps: float
    min_hps: float
    max_hps: float

    @property
    def receipt(self) -> dict:
        payload = {
            "schema": "BITMYNER_GREMLIN_CANDIDATE_MEASUREMENT_V1",
            "candidate": asdict(self.candidate),
            "hps_samples": list(self.hps_samples),
            "median_hps": self.median_hps,
            "min_hps": self.min_hps,
            "max_hps": self.max_hps,
            "promotion_authority": "MEASURED_THROUGHPUT_ONLY",
        }
        payload["receipt_sha256"] = commitment(payload)
        return payload


def phase_seed(phase36: Iterable[float], task: str) -> bytes:
    phi = np.asarray(tuple(phase36), dtype="<f8")
    if phi.shape != (DIM,) or not np.isfinite(phi).all():
        raise ValueError("PhaseNav seed requires exactly 36 finite coordinates")
    return hashlib.sha256(phi.tobytes() + task.encode("utf-8")).digest()


def _dncrypt_permutation(seed: bytes, n: int) -> list[int]:
    if n <= 0:
        return []
    raw = hashlib.shake_256(seed + b"/DNCrypt/scheduler-order/v1").digest(n * 8)
    keys = np.frombuffer(raw, dtype="<u8")
    return np.argsort(keys, kind="stable").astype(int).tolist()


def generate_candidates(phase36: Iterable[float], cpu_count: int, task: str = "Bitcoin SHA256d throughput") -> tuple[SchedulerCandidate, ...]:
    """Build at most 64 measured scheduler candidates.

    PhaseNav supplies the 36D live state and coordinate labels. DNCrypt supplies
    deterministic mutation/order semantics. Q28 contributes only its exact six-bit
    addressing capacity: it indexes candidates, it is not placed in the SHA256d hot
    path. GREMLIN role commitments bind evaluation provenance. No formalism is used
    to claim a change in SHA256d success probability.
    """
    cpu = max(1, int(cpu_count))
    seed = phase_seed(phase36, task)
    variants = ("avx512x32", "avx512x16")
    batches = (1 << 20, 1 << 22, 1 << 24, 1 << 25)
    raw: list[tuple[str, int, int]] = []
    for variant in variants:
        for threads in range(1, cpu + 1):
            for batch in batches:
                raw.append((variant, threads, batch))
    if len(raw) > Q28_ADDRESS_CAPACITY:
        raw = raw[:Q28_ADDRESS_CAPACITY]
    order = _dncrypt_permutation(seed, len(raw))
    out: list[SchedulerCandidate] = []
    for address6, raw_idx in enumerate(order):
        variant, threads, batch = raw[raw_idx]
        phase_coordinate = raw_idx % DIM
        genome_payload = {
            "schema": "BITMYNER_DNCRYPT_SCHEDULER_GENOME_V1",
            "seed_commitment": hashlib.sha256(seed).hexdigest(),
            "raw_index": raw_idx,
            "phase_coordinate": phase_coordinate,
            "variant": variant,
            "threads": threads,
            "batch_size": batch,
            "kappa": KAPPA,
        }
        genome_id = commitment(genome_payload)
        route_payload = {
            "schema": "BITMYNER_GREMLIN_ROUTE_V1",
            "roles": GREMLIN_ROLES,
            "genome_id": genome_id,
            "address6": address6,
            "q28_role": "SCHEDULER_ADDRESS_ONLY",
        }
        out.append(SchedulerCandidate(address6, variant, threads, batch, phase_coordinate, genome_id, commitment(route_payload)))
    return tuple(out)


def promote_measurements(measurements: Iterable[CandidateMeasurement]) -> CandidateMeasurement:
    rows = tuple(measurements)
    if not rows:
        raise ValueError("no scheduler measurements")
    return max(rows, key=lambda r: (r.median_hps, r.min_hps, -r.candidate.threads, -r.candidate.batch_size))
