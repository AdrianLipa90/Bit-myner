from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import os
from pathlib import Path
import statistics
import time

from .core import (
    Candidate,
    NativeBackend,
    NoemaStatus,
    available_backend_names,
    backend_selftest,
    build_backend,
    candidate_from_phase,
    digest_obj,
    read_cpu_flags,
)


@dataclass(frozen=True)
class TauModulation:
    median_latency_ns_num: int
    median_latency_ns_den: int
    mean_abs_jitter_ns_num: int
    mean_abs_jitter_ns_den: int
    stale_budget_ns_num: int
    stale_budget_ns_den: int
    tau_jitter_num: int
    tau_jitter_den: int
    tau_latency_num: int
    tau_latency_den: int
    tau_scalar_num: int
    tau_scalar_den: int

    @property
    def tau_jitter(self) -> Fraction:
        return Fraction(self.tau_jitter_num, self.tau_jitter_den)

    @property
    def tau_latency(self) -> Fraction:
        return Fraction(self.tau_latency_num, self.tau_latency_den)

    @property
    def tau_scalar(self) -> Fraction:
        return Fraction(self.tau_scalar_num, self.tau_scalar_den)

    @property
    def median_latency_ns(self) -> Fraction:
        return Fraction(self.median_latency_ns_num, self.median_latency_ns_den)

    @property
    def mean_abs_jitter_ns(self) -> Fraction:
        return Fraction(self.mean_abs_jitter_ns_num, self.mean_abs_jitter_ns_den)


def _median_fraction(values: list[int] | tuple[int, ...]) -> Fraction:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(int(v) for v in values)
    if any(v <= 0 for v in ordered):
        raise ValueError("latencies must be positive integer nanoseconds")
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return Fraction(ordered[mid], 1)
    return Fraction(ordered[mid - 1] + ordered[mid], 2)


def tau_modulation_from_latencies_ns(latencies_ns: list[int] | tuple[int, ...], stale_budget_s: float) -> TauModulation:
    """Map measured scheduling jitter into exact PNCS scalar-tau factors.

    Bit-myner project definition:

        m_tau = median(dt_i)
        d_tau = mean_i |dt_i - m_tau|
        s_tau,jitter = m_tau / (m_tau + d_tau)
        s_tau,latency = min(1, B_tau / m_tau)
        s_tau = s_tau,jitter * s_tau,latency

    All transforms are positive reduced rationals and therefore compose under
    the existing Natural Coding TRANSFORM -> COMPOSITION rule.  This changes
    only computational scheduling/ranking; it does not modify laboratory time,
    SHA256d, targets, nonce semantics, or physical time.
    """
    vals = tuple(int(v) for v in latencies_ns)
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("latencies_ns must contain positive integer nanoseconds")
    if stale_budget_s <= 0:
        raise ValueError("stale_budget_s must be positive")

    med = _median_fraction(vals)
    dispersion = sum((abs(Fraction(v, 1) - med) for v in vals), Fraction(0, 1)) / len(vals)
    budget = Fraction(str(stale_budget_s)) * 1_000_000_000
    if budget <= 0:
        raise ValueError("stale budget must be positive")

    s_jitter = med / (med + dispersion)
    s_latency = min(Fraction(1, 1), budget / med)
    s_tau = s_jitter * s_latency

    return TauModulation(
        med.numerator,
        med.denominator,
        dispersion.numerator,
        dispersion.denominator,
        budget.numerator,
        budget.denominator,
        s_jitter.numerator,
        s_jitter.denominator,
        s_latency.numerator,
        s_latency.denominator,
        s_tau.numerator,
        s_tau.denominator,
    )


@dataclass(frozen=True)
class TauBenchRecord:
    backend: str
    threads: int
    batch_hashes: int
    trials_hps: tuple[float, ...]
    trials_latency_ns: tuple[int, ...]
    median_hps: float
    max_hps: float
    jitter_cv_telemetry: float
    median_batch_latency_s: float
    tau_latency_ratio_telemetry: float
    tau_jitter_num: int
    tau_jitter_den: int
    tau_latency_num: int
    tau_latency_den: int
    tau_scalar_num: int
    tau_scalar_den: int
    tau_modulated_hps: float
    promotion_eligible: bool
    score: float
    genome_id: str
    route_commitment: str
    q28_schedule_id: int


def benchmark_candidate_tau(
    backend: NativeBackend,
    candidate: Candidate,
    *,
    repeats: int = 5,
    stale_budget_s: float = 0.20,
) -> TauBenchRecord:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    count = candidate.batch_hashes - (candidate.batch_hashes % backend.lanes)
    if count <= 0:
        raise ValueError("batch_hashes smaller than backend lane width")

    backend.bench(
        __import__("bitmyner.core", fromlist=["GENESIS_HEADER76"]).GENESIS_HEADER76,
        0,
        max(backend.lanes * 4096, min(count, backend.lanes * 20000)),
        candidate.threads,
    )

    from .core import GENESIS_HEADER76

    latencies_ns: list[int] = []
    samples_hps: list[float] = []
    for r in range(repeats):
        start = (r * count) & 0xFFFFFFFF
        t0 = time.perf_counter_ns()
        backend.bench(GENESIS_HEADER76, start, count, candidate.threads)
        dt_ns = max(1, time.perf_counter_ns() - t0)
        latencies_ns.append(dt_ns)
        samples_hps.append((count * 1_000_000_000.0) / dt_ns)

    med_hps = statistics.median(samples_hps)
    mean_hps = statistics.fmean(samples_hps)
    cv = statistics.pstdev(samples_hps) / mean_hps if len(samples_hps) > 1 and mean_hps else 0.0
    tau = tau_modulation_from_latencies_ns(latencies_ns, stale_budget_s)
    median_latency_s = float(tau.median_latency_ns) / 1_000_000_000.0
    tau_latency_ratio = median_latency_s / stale_budget_s

    # Jitter no longer has an arbitrary hard CV cut. It is represented by the
    # exact scalar-tau transform. The stale-work latency budget remains a hard
    # operational admission boundary.
    eligible = tau.tau_latency == 1
    tau_modulated_hps = med_hps * float(tau.tau_scalar)
    score = tau_modulated_hps

    return TauBenchRecord(
        candidate.backend,
        candidate.threads,
        count,
        tuple(samples_hps),
        tuple(latencies_ns),
        med_hps,
        max(samples_hps),
        cv,
        median_latency_s,
        tau_latency_ratio,
        tau.tau_jitter_num,
        tau.tau_jitter_den,
        tau.tau_latency_num,
        tau.tau_latency_den,
        tau.tau_scalar_num,
        tau.tau_scalar_den,
        tau_modulated_hps,
        eligible,
        score,
        candidate.genome_id,
        candidate.route_commitment,
        candidate.q28_schedule_id,
    )


def autotune_tau(
    noema: NoemaStatus,
    *,
    repeats: int = 5,
    batch_hashes: int = 6_400_000,
    stale_budget_s: float = 0.20,
    build_dir: Path | None = None,
) -> dict[str, object]:
    flags = read_cpu_flags()
    names = available_backend_names(flags)
    cpu_count = max(1, os.cpu_count() or 1)
    records: list[TauBenchRecord] = []
    selftests: list[dict[str, object]] = []
    backend_errors: dict[str, str] = {}

    for name in names:
        try:
            backend = build_backend(name, build_dir)
            selftests.append(backend_selftest(backend))
        except Exception as exc:
            backend_errors[name] = repr(exc)
            continue

        if name in {"avx512x2", "avx512"}:
            batch_grid = sorted({max(backend.lanes * 4096, batch_hashes // 2), batch_hashes, batch_hashes * 2})
        elif name == "avx2":
            batch_grid = sorted({max(backend.lanes * 4096, batch_hashes // 2), batch_hashes})
        else:
            batch_grid = [max(backend.lanes * 4096, batch_hashes // 2)]

        for threads in range(1, cpu_count + 1):
            for bh in batch_grid:
                candidate = candidate_from_phase(noema.phase36, name, threads, bh)
                records.append(
                    benchmark_candidate_tau(
                        backend,
                        candidate,
                        repeats=repeats,
                        stale_budget_s=stale_budget_s,
                    )
                )

    if not records:
        raise RuntimeError(f"NO_BACKEND_SURVIVED:{backend_errors}")

    eligible = [r for r in records if r.promotion_eligible]
    pool = eligible if eligible else records
    winner = max(pool, key=lambda r: (r.score, r.median_hps, r.tau_scalar_num / r.tau_scalar_den))

    payload: dict[str, object] = {
        "schema": "BITMYNER_EXTREME_AUTOTUNE_V0_3_TAU_JITTER",
        "created_unix_ns": time.time_ns(),
        "noema": asdict(noema),
        "cpu_count": cpu_count,
        "cpu_flags_used": sorted(flags.intersection({"avx2", "avx512f", "sha_ni", "sha"})),
        "tau_stale_budget_s": stale_budget_s,
        "tau_jitter_definition": {
            "status": "BITMYNER_PROJECT_EXACT_DEFINITION",
            "median": "m_tau=median(dt_i)",
            "dispersion": "d_tau=mean_i(abs(dt_i-m_tau))",
            "jitter_transform": "s_tau_jitter=m_tau/(m_tau+d_tau)",
            "latency_transform": "s_tau_latency=min(1,B_tau/m_tau)",
            "composition": "s_tau=s_tau_jitter*s_tau_latency",
            "ranking": "score=median_hps*s_tau",
            "jitter_cv_role": "TELEMETRY_ONLY_NO_PROMOTION_AUTHORITY",
            "laboratory_time_modified": False,
            "hydrodynamic_lapse_modified": False,
            "physical_time_claim": False,
        },
        "gremlin_policy": {
            "candidate_generation": "broad",
            "promotion": "narrow_typed_deterministic_receipt_bearing",
            "jitter_authority": "EXACT_SCALAR_TAU_MODULATION",
            "roles": ["SPIDER", "ANT", "MANTIS", "HOUND", "OWL", "BELZEBUB"],
        },
        "q28_semantics": "CLASSICAL_EXACT_28_BIT_CONFIGURATION_ADDRESS_ONLY_NO_PHYSICAL_QPU_CLAIM",
        "selftests": selftests,
        "backend_errors": backend_errors,
        "records": [asdict(r) for r in records],
        "winner": asdict(winner),
    }
    payload["receipt_blake2b256"] = digest_obj(payload)
    return payload
