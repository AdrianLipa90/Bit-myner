from __future__ import annotations

import math
from typing import Iterable


def _validate_k(k: int) -> int:
    k = int(k)
    if not 1 <= k <= 256:
        raise ValueError('k must be in [1, 256]')
    return k


def expected_threshold_events(samples: int, k: int) -> float:
    samples = int(samples)
    k = _validate_k(k)
    if samples <= 0:
        raise ValueError('samples must be positive')
    return samples / float(1 << k)


def samples_for_expected_events(k: int, desired_events: int) -> int:
    k = _validate_k(k)
    desired_events = int(desired_events)
    if desired_events <= 0:
        raise ValueError('desired_events must be positive')
    return desired_events * (1 << k)


def plan_threshold_power(thresholds: Iterable[int], *, desired_events: int = 200) -> dict:
    levels = tuple(sorted(set(_validate_k(k) for k in thresholds)))
    if not levels:
        raise ValueError('at least one threshold is required')
    desired_events = int(desired_events)
    if desired_events <= 0:
        raise ValueError('desired_events must be positive')
    per = {
        str(k): {
            'k': k,
            'baseline_probability': 1.0 / float(1 << k),
            'desired_expected_events': desired_events,
            'samples': samples_for_expected_events(k, desired_events),
        }
        for k in levels
    }
    n = max(row['samples'] for row in per.values())
    return {
        'schema': 'BITMYNER_MARKER_POWER_PLAN_V0_1',
        'thresholds': list(levels),
        'desired_expected_events': desired_events,
        'per_threshold': per,
        'minimum_common_samples': n,
        'expected_events_at_common_samples': {
            str(k): expected_threshold_events(n, k) for k in levels
        },
        'semantics': 'EXPECTED_EVENT_COUNT_UNDER_SHA256D_RANDOM_ORACLE_BASELINE_NOT_A_POWER_GUARANTEE',
    }
