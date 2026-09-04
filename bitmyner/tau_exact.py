from __future__ import annotations

from fractions import Fraction
from typing import Any

from .core import digest_obj
from .tau_jitter import autotune_tau, tau_modulation_from_latencies_ns


def exact_tau_score_hps(
    batch_hashes: int,
    latencies_ns: list[int] | tuple[int, ...],
    stale_budget_s: float,
) -> tuple[Fraction, dict[str, int]]:
    """Return the exact tau-modulated service rate in hashes/second.

    With

        m = median latency
        d = mean absolute latency deviation about m
        s_j = m/(m+d)
        s_l = min(1, B/m)

    the tau-modulated median service rate is exactly

        H_tau = batch_hashes * 1e9 / (m+d) * s_l.

    This is the time-domain form of H_median*s_j*s_l, avoiding a floating
    jitter penalty in promotion authority.  All quantities participating in
    the ranking are reduced rational numbers.
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


def autotune_tau_exact(
    noema,
    *,
    repeats: int = 5,
    batch_hashes: int = 6_400_000,
    stale_budget_s: float = 0.20,
    build_dir=None,
) -> dict[str, Any]:
    """Run the native lattice and promote by exact scalar-tau service rate.

    The underlying benchmark still records conventional floating telemetry.
    Promotion authority is replaced here by an exact rational time-domain
    score.  A candidate whose median batch latency exceeds the stale-work
    budget remains in evidence but cannot be promoted.
    """
    base = autotune_tau(
        noema,
        repeats=repeats,
        batch_hashes=batch_hashes,
        stale_budget_s=stale_budget_s,
        build_dir=build_dir,
    )

    enriched: list[dict[str, Any]] = []
    eligible: list[tuple[Fraction, dict[str, Any]]] = []
    for raw in base["records"]:
        record = dict(raw)
        exact, evidence = exact_tau_score_hps(
            int(record["batch_hashes"]),
            tuple(int(x) for x in record["trials_latency_ns"]),
            stale_budget_s,
        )
        record.update(evidence)
        record["tau_exact_score_hps"] = float(exact)
        record["promotion_basis"] = "EXACT_TAU_TIME_DOMAIN_SERVICE_RATE"
        enriched.append(record)
        if record.get("promotion_eligible") is True:
            eligible.append((exact, record))

    if eligible:
        exact_winner, winner = max(
            eligible,
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
        blocked_reason = "NO_LATENCY_ELIGIBLE_CANDIDATE"

    payload: dict[str, Any] = {
        "schema": "BITMYNER_EXTREME_AUTOTUNE_V0_4_EXACT_TAU",
        "status": status,
        "blocked_reason": blocked_reason,
        "source_receipt_blake2b256": base.get("receipt_blake2b256"),
        "noema": base.get("noema"),
        "cpu_count": base.get("cpu_count"),
        "cpu_flags_used": base.get("cpu_flags_used"),
        "tau_stale_budget_s": stale_budget_s,
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
        "gremlin_policy": base.get("gremlin_policy"),
        "q28_semantics": base.get("q28_semantics"),
        "selftests": base.get("selftests"),
        "backend_errors": base.get("backend_errors"),
        "records": enriched,
        "winner": winner,
        "winner_exact_score_hps_num": exact_winner.numerator,
        "winner_exact_score_hps_den": exact_winner.denominator,
    }
    payload["receipt_blake2b256"] = digest_obj(payload)
    return payload
