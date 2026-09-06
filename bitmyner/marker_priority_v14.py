from __future__ import annotations

from typing import Any

SCHEMA = 'BITMYNER_MARKER_PRIORITY_PHASE_V0_14'


def annotate_priority_phase(metric: dict[str, Any]) -> dict[str, Any]:
    """Re-express the existing cost-aware marker metric as an exact no-skip priority phase.

    The underlying `Gamma` calculation is the event density among candidates completed
    during the marker-positive phase after spending checkpoint work on deferred
    candidates. Deferred candidates are *not* assumed impossible and are not discarded.
    Completing them afterwards restores exhaustive SHA256d coverage.
    """
    n = int(metric.get('n', 0))
    marker_n = int(metric.get('marker_n', 0))
    event_n = int(metric.get('event_n', 0))
    marker_event_n = int(metric.get('marker_event_n', 0))
    candidate_fraction = (marker_n / n) if n else 0.0
    event_recall = (marker_event_n / event_n) if event_n else 0.0
    return {
        'schema': SCHEMA,
        'priority_candidate_fraction': candidate_fraction,
        'deferred_candidate_fraction': (1.0 - candidate_fraction) if n else 0.0,
        'priority_event_recall': event_recall,
        'priority_phase_work_fraction': float(metric.get('compute_fraction', 1.0)),
        'priority_event_density_gain': float(metric.get('gamma', 0.0)),
        'priority_event_density_gain_lcb': float(metric.get('gamma_lcb', 0.0)),
        'priority_lift': float(metric.get('lift', 0.0)),
        'priority_lift_lcb': float(metric.get('lift_lcb', 0.0)),
        'exhaustive_total_work_fraction_after_deferred_completion': 1.0 if n else 0.0,
        'deferred_candidates_discarded': False,
        'hard_rejection_authorized': False,
        'exact_sha256d_coverage_preserved_if_deferred_completed': True,
        'production_filter_claim': False,
    }
