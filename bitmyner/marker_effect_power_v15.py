from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist

SCHEMA = 'BITMYNER_MARKER_EFFECT_POWER_V0_15'


def _wilson(successes: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * sqrt((phat * (1.0 - phat) / n) + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def expected_gamma_lcb_v15(
    sample_count: int,
    *,
    event_probability: float,
    marker_rate: float,
    priority_work_fraction: float,
    target_gamma: float,
    alpha: float,
) -> dict:
    """Expected-count design calculation for a preregistered no-skip priority effect.

    `target_gamma` is the design point, not a measured claim. The implied marker lift
    follows from Gamma = marker_rate * lift / priority_work_fraction.
    """
    n = int(sample_count)
    if n < 1:
        raise ValueError('sample_count must be positive')
    p = float(event_probability)
    q = float(marker_rate)
    c = float(priority_work_fraction)
    g = float(target_gamma)
    a = float(alpha)
    if not 0.0 < p < 1.0:
        raise ValueError('event_probability must be in (0,1)')
    if not 0.0 < q <= 1.0:
        raise ValueError('marker_rate must be in (0,1]')
    if not q <= c <= 1.0:
        raise ValueError('priority_work_fraction must be in [marker_rate,1]')
    if g <= 1.0:
        raise ValueError('target_gamma must be > 1')
    if not 0.0 < a < 1.0:
        raise ValueError('alpha must be in (0,1)')

    implied_lift = g * c / q
    conditional_event_probability = p * implied_lift
    if conditional_event_probability >= 1.0:
        raise ValueError('design implies conditional event probability >= 1')

    marker_n = max(1, min(n, int(round(n * q))))
    event_n = max(1, min(n, int(round(n * p))))
    marker_event_n = max(
        0,
        min(marker_n, event_n, int(round(marker_n * conditional_event_probability))),
    )
    cond_lo, _ = _wilson(marker_event_n, marker_n, a)
    _, base_hi = _wilson(event_n, n, a)
    lift_lcb = cond_lo / base_hi if base_hi else 0.0
    gamma_lcb = q * lift_lcb / c
    return {
        'sample_count': n,
        'event_probability': p,
        'marker_rate': q,
        'priority_work_fraction': c,
        'target_gamma': g,
        'implied_lift': implied_lift,
        'expected_event_n_rounded': event_n,
        'expected_marker_n_rounded': marker_n,
        'expected_marker_event_n_rounded': marker_event_n,
        'alpha': a,
        'expected_lift_lcb': lift_lcb,
        'expected_gamma_lcb': gamma_lcb,
    }


def plan_effect_power_v15(
    *,
    hardest_k: int,
    marker_rate: float,
    priority_work_fraction: float,
    target_gamma: float,
    desired_gamma_lcb: float = 1.0,
    familywise_alpha: float = 0.05,
    candidate_count: int = 8,
    threshold_count: int = 3,
    min_samples: int = 1024,
    max_samples: int = 1 << 30,
) -> dict:
    """Find the minimum expected-count sample size meeting a preregistered LCB target.

    This planner never observes event labels. Its assumptions must be frozen before the
    confirmation surface is opened. It is a design/power calculation, not evidence of
    an effect.
    """
    k = int(hardest_k)
    if not 1 <= k <= 32:
        raise ValueError('hardest_k must be in [1,32]')
    if float(desired_gamma_lcb) < 1.0:
        raise ValueError('desired_gamma_lcb must be >= 1')
    if candidate_count < 1 or threshold_count < 1:
        raise ValueError('candidate_count and threshold_count must be positive')
    if min_samples < 1 or max_samples < min_samples:
        raise ValueError('invalid sample bounds')

    tests = int(candidate_count) * int(threshold_count)
    alpha = float(familywise_alpha) / tests
    p = 2.0 ** (-k)

    def evaluate(n: int) -> dict:
        return expected_gamma_lcb_v15(
            n,
            event_probability=p,
            marker_rate=marker_rate,
            priority_work_fraction=priority_work_fraction,
            target_gamma=target_gamma,
            alpha=alpha,
        )

    lo = int(min_samples)
    hi = lo
    row = evaluate(hi)
    while row['expected_gamma_lcb'] < float(desired_gamma_lcb) and hi < int(max_samples):
        lo = hi + 1
        hi = min(int(max_samples), hi * 2)
        row = evaluate(hi)

    met = row['expected_gamma_lcb'] >= float(desired_gamma_lcb)
    if met:
        left = int(min_samples)
        right = hi
        while left < right:
            mid = (left + right) // 2
            if evaluate(mid)['expected_gamma_lcb'] >= float(desired_gamma_lcb):
                right = mid
            else:
                left = mid + 1
        row = evaluate(left)
        required = left
    else:
        required = hi

    return {
        'schema': SCHEMA,
        'status': 'POWER_TARGET_MET' if met else 'POWER_TARGET_NOT_MET_WITHIN_CAP',
        'hardest_k': k,
        'event_probability': p,
        'marker_rate_assumption': float(marker_rate),
        'priority_work_fraction_assumption': float(priority_work_fraction),
        'target_gamma_design_point': float(target_gamma),
        'desired_gamma_lcb': float(desired_gamma_lcb),
        'familywise_alpha': float(familywise_alpha),
        'candidate_count': int(candidate_count),
        'threshold_count': int(threshold_count),
        'per_test_alpha': alpha,
        'required_samples': int(required),
        'required_expected_events': int(required) * p,
        'design_at_required_samples': row,
        'max_samples': int(max_samples),
        'label_access': False,
        'effect_is_assumed_for_power_design_not_observed': True,
        'production_policy_modified': False,
    }
