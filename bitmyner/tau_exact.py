from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import os
from pathlib import Path
from typing import Any

from .core import (
    available_backend_names,
    backend_selftest,
    build_backend,
    candidate_from_phase,
    digest_obj,
    read_cpu_flags,
)
from .tau_jitter import benchmark_candidate_tau, tau_modulation_from_latencies_ns


def exact_tau_score_hps(
    batch_hashes: int,
    latencies_ns: list[int] | tuple[int, ...],
    stale_budget_s: float,
) -> tuple[Fraction, dict[str, int]]:
    """Return the exact tau-modulated service rate in hashes/second.

    Let

        m = median latency
        d = mean absolute latency deviation about m
        s_j = m/(m+d)
        s_l = min(1, B/m)

    Then the tau-modulated service rate is evaluated directly as

        H_tau = batch_hashes * 1e9 / (m+d) * s_l.

    This is the time-domain form of the median service rate modulated by tau.
    All quantities participating in promotion are reduced rational numbers.
    """
    if batch_hashes <= 0:
        raise ValueError("batch_hashes must be positive")
    tau = tau_modulation_from_latencies_ns(latencies_ns, stale_budget_s)
    m = tau.median_latency_ns
    d = tau.mean_abs_jitter_ns
    effective_latency = (m + d) / tau.tau_latency
    score = Fraction(batch_hashes * 1_000_000_000, 1) / effective_latency
    return score, {
        "effective_latency_ns_num": effective_latency.numerator,
        "effective_latency_ns_den": effective_latency.denominator,
        "score_hps_num": score.numerator,
        "score_hps_den": score.denominator,
    }


def _aligned_batch(value: int, lanes: int) -> int:
    floor = lanes * 4096
    value = max(floor, int(value))
    return value - (value % lanes)


def _batch_grid(name: str, lanes: int, center: int) -> tuple[int, ...]:
    """Broad tau lattice.

    AVX-512 candidates intentionally include center/8 and center/4. Shorter
    batches matter because scheduler jitter can dominate before raw throughput
    saturates; the previous center/2..2*center lattice could miss that regime.
    """
    if name in {"avx512x2", "avx512"}:
        raw = (center // 8, center // 4, center // 2, center, center * 2)
    elif name == "avx2":
        raw = (center // 4, center // 2, center)
    else:
        raw = (center // 4, center // 2)
    return tuple(sorted({_aligned_batch(v, lanes) for v in raw}))


def _enrich(record: dict[str, Any], stale_budget_s: float) -> tuple[Fraction, dict[str, Any]]:
    exact, evidence = exact_tau_score_hps(
        int(record["batch_hashes"]),
        tuple(int(x) for x in record["trials_latency_ns"]),
        stale_budget_s,
    )
    out = dict(record)
    out.update(evidence)
    out["tau_exact_score_hps"] = float(exact)
    out["promotion_basis"] = "EXACT_TAU_TIME_DOMAIN_SERVICE_RATE"
    return exact, out


def autotune_tau_exact(
    noema,
    *,
    repeats: int = 5,
    batch_hashes: int = 6_400_000,
    stale_budget_s: float = 0.20,
    build_dir: Path | None = None,
    refine_top_k: int = 6,
) -> dict[str, Any]:
    """GREMLIN broad-probe -> narrow-refine autotuner with exact tau authority.

    Stage A probes a broad backend/thread/batch lattice with at most three
    timing samples per candidate. Stage B remeasures only the strongest
    latency-eligible candidates with the requested repeat count. Promotion is
    based on exact rational H_tau, never on jitter CV or a transient peak.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if batch_hashes <= 0:
        raise ValueError("batch_hashes must be positive")
    if stale_budget_s <= 0:
        raise ValueError("stale_budget_s must be positive")
    if refine_top_k <= 0:
        raise ValueError("refine_top_k must be positive")

    flags = read_cpu_flags()
    names = available_backend_names(flags)
    cpu_count = max(1, os.cpu_count() or 1)
    coarse_repeats = min(3, repeats)
    refine_repeats = repeats

    selftests: list[dict[str, Any]] = []
    backend_errors: dict[str, str] = {}
    candidate_errors: list[dict[str, str]] = []
    coarse_records: list[dict[str, Any]] = []
    coarse_eligible: list[tuple[Fraction, object, dict[str, Any]]] = []
    lattice_size = 0

    for name in names:
        try:
            backend = build_backend(name, build_dir)
            selftests.append(backend_selftest(backend))
        except Exception as exc:
            backend_errors[name] = repr(exc)
            continue

        grid = _batch_grid(name, backend.lanes, batch_hashes)
        for threads in range(1, cpu_count + 1):
            for bh in grid:
                lattice_size += 1
                candidate = candidate_from_phase(noema.phase36, name, threads, bh)
                try:
                    raw = asdict(
                        benchmark_candidate_tau(
                            backend,
                            candidate,
                            repeats=coarse_repeats,
                            stale_budget_s=stale_budget_s,
                        )
                    )
                    exact, record = _enrich(raw, stale_budget_s)
                    record["stage"] = "COARSE"
                    coarse_records.append(record)
                    if record.get("promotion_eligible") is True:
                        coarse_eligible.append((exact, candidate, record))
                except Exception as exc:
                    candidate_errors.append(
                        {
                            "backend": name,
                            "threads": str(threads),
                            "batch_hashes": str(bh),
                            "error": repr(exc),
                        }
                    )

    if not coarse_records:
        raise RuntimeError(
            f"NO_BENCHMARK_RECORDS:backend_errors={backend_errors}:candidate_errors={candidate_errors}"
        )

    coarse_eligible.sort(
        key=lambda item: (
            item[0],
            Fraction(
                int(item[2]["tau_scalar_num"]),
                int(item[2]["tau_scalar_den"]),
            ),
            int(item[2]["median_hps"]),
        ),
        reverse=True,
    )
    finalists = coarse_eligible[:refine_top_k]

    refined_records: list[dict[str, Any]] = []
    refined_eligible: list[tuple[Fraction, dict[str, Any]]] = []
    backend_cache: dict[str, Any] = {}

    for _coarse_score, candidate, coarse_record in finalists:
        name = str(coarse_record["backend"])
        try:
            backend = backend_cache.get(name)
            if backend is None:
                backend = build_backend(name, build_dir)
                backend_cache[name] = backend
            raw = asdict(
                benchmark_candidate_tau(
                    backend,
                    candidate,
                    repeats=refine_repeats,
                    stale_budget_s=stale_budget_s,
                )
            )
            exact, record = _enrich(raw, stale_budget_s)
            record["stage"] = "REFINED"
            record["coarse_score_hps_num"] = _coarse_score.numerator
            record["coarse_score_hps_den"] = _coarse_score.denominator
            refined_records.append(record)
            if record.get("promotion_eligible") is True:
                refined_eligible.append((exact, record))
        except Exception as exc:
            candidate_errors.append(
                {
                    "backend": name,
                    "threads": str(coarse_record["threads"]),
                    "batch_hashes": str(coarse_record["batch_hashes"]),
                    "error": f"REFINE:{exc!r}",
                }
            )

    if refined_eligible:
        exact_winner, winner = max(
            refined_eligible,
            key=lambda item: (
                item[0],
                Fraction(
                    int(item[1]["tau_scalar_num"]),
                    int(item[1]["tau_scalar_den"]),
                ),
                int(item[1]["batch_hashes"]),
                -int(item[1]["threads"]),
            ),
        )
        status = "PASS"
        blocked_reason = None
    else:
        exact_winner = Fraction(0, 1)
        winner = None
        status = "BLOCKED"
        blocked_reason = (
            "NO_REFINED_LATENCY_ELIGIBLE_CANDIDATE"
            if finalists
            else "NO_COARSE_LATENCY_ELIGIBLE_CANDIDATE"
        )

    payload: dict[str, Any] = {
        "schema": "BITMYNER_EXTREME_AUTOTUNE_V0_5_GREMLIN_EXACT_TAU",
        "status": status,
        "blocked_reason": blocked_reason,
        "noema": asdict(noema),
        "cpu_count": cpu_count,
        "cpu_flags_used": sorted(
            flags.intersection({"avx2", "avx512f", "sha_ni", "sha"})
        ),
        "tau_stale_budget_s": stale_budget_s,
        "lattice": {
            "center_batch_hashes": batch_hashes,
            "candidate_count": lattice_size,
            "coarse_repeats": coarse_repeats,
            "refine_repeats": refine_repeats,
            "refine_top_k": refine_top_k,
            "avx512_batch_factors": ["1/8", "1/4", "1/2", "1", "2"],
        },
        "tau_promotion": {
            "jitter_cv_role": "TELEMETRY_ONLY_NO_PROMOTION_AUTHORITY",
            "median": "m_tau=median(dt_i)",
            "dispersion": "d_tau=mean(abs(dt_i-m_tau))",
            "jitter_transform": "s_tau_jitter=m_tau/(m_tau+d_tau)",
            "latency_transform": "s_tau_latency=min(1,B_tau/m_tau)",
            "composition": "s_tau=s_tau_jitter*s_tau_latency",
            "exact_score": "H_tau=batch_hashes*1e9/(m_tau+d_tau)*s_tau_latency",
            "promotion_boundary": "median_latency<=stale_budget",
            "laboratory_time_modified": False,
            "hydrodynamic_lapse_modified": False,
            "physical_time_claim": False,
        },
        "gremlin_policy": {
            "candidate_generation": "BROAD_COARSE_LATTICE",
            "promotion": "TOP_K_NARROW_REFINEMENT_THEN_EXACT_TAU",
            "roles": [
                "SPIDER",
                "ANT",
                "MANTIS",
                "HOUND",
                "OWL",
                "BELZEBUB",
            ],
        },
        "q28_semantics": (
            "CLASSICAL_EXACT_28_BIT_CONFIGURATION_ADDRESS_ONLY_"
            "NO_PHYSICAL_QPU_CLAIM"
        ),
        "selftests": selftests,
        "backend_errors": backend_errors,
        "candidate_errors": candidate_errors,
        "coarse_records": coarse_records,
        "records": refined_records,
        "winner": winner,
        "winner_exact_score_hps_num": exact_winner.numerator,
        "winner_exact_score_hps_den": exact_winner.denominator,
    }
    payload["receipt_blake2b256"] = digest_obj(payload)
    return payload
