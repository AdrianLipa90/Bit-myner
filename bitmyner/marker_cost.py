from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Sequence

from .dncrypt_marker import MarkerGenome
from .marker_eval import TraceSample

FULL_SHA256D_ROUND_COST = 128


def _checkpoint_cost(stage: int, round_index: int) -> int:
    if stage == 1:
        return round_index + 1
    if stage == 2:
        return 64 + round_index + 1
    raise ValueError('stage must be 1 or 2')


@dataclass(frozen=True, slots=True)
class CostObservation:
    marker: bool
    event: bool
    cost_units: int

    def __post_init__(self) -> None:
        if self.cost_units <= 0:
            raise ValueError('cost_units must be positive')


@dataclass(frozen=True, slots=True)
class GammaReport:
    n: int
    marker_n: int
    event_n: int
    marker_event_n: int
    total_cost_units: int
    full_cost_units: int
    pass_rate: Fraction
    event_recall: Fraction
    compute_fraction: Fraction
    gamma: Fraction

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ('pass_rate', 'event_recall', 'compute_fraction', 'gamma'):
            value = d[key]
            if isinstance(value, Fraction):
                d[key] = {
                    'num': value.numerator,
                    'den': value.denominator,
                    'float': float(value),
                }
        return d


def gamma_from_cost_observations(
    rows: Sequence[CostObservation],
    *,
    full_cost_units: int = FULL_SHA256D_ROUND_COST,
) -> GammaReport:
    if not rows:
        raise ValueError('rows may not be empty')
    if full_cost_units <= 0:
        raise ValueError('full_cost_units must be positive')
    n = len(rows)
    marker_n = sum(r.marker for r in rows)
    event_n = sum(r.event for r in rows)
    marker_event_n = sum(r.marker and r.event for r in rows)
    total_cost = sum(r.cost_units for r in rows)
    pass_rate = Fraction(marker_n, n)
    event_recall = Fraction(marker_event_n, event_n) if event_n else Fraction(0, 1)
    compute_fraction = Fraction(total_cost, n * full_cost_units)
    # eta_policy / eta_baseline where eta = retained events / round cost.
    gamma = (
        Fraction(marker_event_n, total_cost) / Fraction(event_n, n * full_cost_units)
        if event_n and marker_event_n
        else Fraction(0, 1)
    )
    return GammaReport(
        n=n,
        marker_n=marker_n,
        event_n=event_n,
        marker_event_n=marker_event_n,
        total_cost_units=total_cost,
        full_cost_units=full_cost_units,
        pass_rate=pass_rate,
        event_recall=event_recall,
        compute_fraction=compute_fraction,
        gamma=gamma,
    )


def tau_adjusted_gamma(
    gamma: Fraction,
    marker_tau: Fraction,
    baseline_tau: Fraction,
) -> Fraction:
    """Adjust a compute-domain gain by exact positive scalar-tau factors."""
    if marker_tau <= 0 or baseline_tau <= 0:
        raise ValueError('tau factors must be positive')
    return gamma * marker_tau / baseline_tau


def policy_cost_observation(genome: MarkerGenome, sample: TraceSample) -> CostObservation:
    """Evaluate one marker as an early-pass filter in exact SHA round units.

    `all`: reject at the first failed predicate checkpoint; if all predicates
    pass, finish SHA256d to know whether the candidate is an event.

    `any`: once any predicate passes, the candidate is retained and SHA256d is
    completed; if every predicate fails, reject after the last checkpoint.

    This models only SHA round work. Bit-test, branch, cache and scheduler costs
    require a measured runtime receipt and are not silently folded in.
    """
    checkpoints = sorted(
        genome.predicates,
        key=lambda p: (_checkpoint_cost(p.stage, p.round), p.source, p.mask, p.value),
    )
    if genome.mode == 'all':
        for predicate in checkpoints:
            cost = _checkpoint_cost(predicate.stage, predicate.round)
            if not predicate.matches(sample.trace):
                return CostObservation(False, sample.event, cost)
        return CostObservation(True, sample.event, FULL_SHA256D_ROUND_COST)

    if genome.mode == 'any':
        last_cost = 1
        for predicate in checkpoints:
            last_cost = _checkpoint_cost(predicate.stage, predicate.round)
            if predicate.matches(sample.trace):
                return CostObservation(True, sample.event, FULL_SHA256D_ROUND_COST)
        return CostObservation(False, sample.event, last_cost)

    raise ValueError(f'unsupported genome mode: {genome.mode}')


def gamma_for_genome(genome: MarkerGenome, samples: Sequence[TraceSample]) -> GammaReport:
    return gamma_from_cost_observations(
        [policy_cost_observation(genome, sample) for sample in samples],
        full_cost_units=FULL_SHA256D_ROUND_COST,
    )


def gamma_lcb_from_lift(
    report: GammaReport,
    *,
    lift_lcb: float,
) -> float:
    """Project a conservative lift bound onto the observed cost envelope.

    This is not presented as a full confidence interval for cost. The marker
    selection fraction and deterministic round costs are held at their observed
    values while only the event enrichment term is replaced by its conservative
    lift lower bound.
    """
    if lift_lcb < 0:
        raise ValueError('lift_lcb may not be negative')
    avg_cost = report.total_cost_units / report.n
    if avg_cost <= 0:
        return 0.0
    return FULL_SHA256D_ROUND_COST * float(report.pass_rate) * lift_lcb / avg_cost
