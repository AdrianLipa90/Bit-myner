from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
from math import sqrt
from statistics import NormalDist
from typing import Iterable, Sequence

from .dncrypt_marker import MarkerGenome
from .sha256_trace import Sha256dTrace


@dataclass(frozen=True, slots=True)
class MarkerObservation:
    header_id: str
    marker: bool
    event: bool


@dataclass(frozen=True, slots=True)
class LiftReport:
    n: int
    marker_n: int
    event_n: int
    marker_event_n: int
    p_event: float
    p_event_given_marker: float
    lift: float
    lift_lcb: float
    alpha: float

    def to_dict(self) -> dict:
        return asdict(self)


def _wilson(successes: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    z = NormalDist().inv_cdf(1 - alpha/2)
    phat = successes / n
    z2 = z*z
    denom = 1 + z2/n
    center = (phat + z2/(2*n)) / denom
    half = z * sqrt((phat*(1-phat)/n) + z2/(4*n*n)) / denom
    return max(0.0, center-half), min(1.0, center+half)


def lift_report(observations: Sequence[MarkerObservation], *, alpha: float = 0.05) -> LiftReport:
    n = len(observations)
    if n == 0:
        raise ValueError('observations may not be empty')
    marker_n = sum(o.marker for o in observations)
    event_n = sum(o.event for o in observations)
    marker_event_n = sum(o.marker and o.event for o in observations)
    p_event = event_n / n
    p_cond = marker_event_n / marker_n if marker_n else 0.0
    lift = p_cond / p_event if p_event > 0 else 0.0
    cond_lo, _ = _wilson(marker_event_n, marker_n, alpha)
    _, base_hi = _wilson(event_n, n, alpha)
    lift_lcb = cond_lo / base_hi if base_hi > 0 else 0.0
    return LiftReport(n, marker_n, event_n, marker_event_n, p_event, p_cond, lift, lift_lcb, alpha)


def split_name(header_id: str, *, seed: str = 'bitmyner-marker-v0.1') -> str:
    h = hashlib.blake2b((seed + ':' + header_id).encode('utf-8'), digest_size=8).digest()
    bucket = int.from_bytes(h, 'big') % 10000
    if bucket < 6000:
        return 'train'
    if bucket < 8000:
        return 'validation'
    return 'holdout'


def partition(observations: Iterable[MarkerObservation], *, seed: str = 'bitmyner-marker-v0.1') -> dict[str, list[MarkerObservation]]:
    out = {'train': [], 'validation': [], 'holdout': []}
    for o in observations:
        out[split_name(o.header_id, seed=seed)].append(o)
    return out


@dataclass(frozen=True, slots=True)
class TraceSample:
    header_id: str
    trace: Sha256dTrace
    event_k: int

    @property
    def event(self) -> bool:
        return self.trace.leading_zero_bits >= self.event_k


def observations_for_genome(genome: MarkerGenome, samples: Sequence[TraceSample]) -> list[MarkerObservation]:
    return [MarkerObservation(s.header_id, genome.matches(s.trace), s.event) for s in samples]


def evaluate_genome(
    genome: MarkerGenome,
    samples: Sequence[TraceSample],
    *,
    seed: str = 'bitmyner-marker-v0.1',
    alpha: float = 0.05,
) -> dict:
    obs = observations_for_genome(genome, samples)
    parts = partition(obs, seed=seed)
    reports = {}
    for name, rows in parts.items():
        reports[name] = lift_report(rows, alpha=alpha).to_dict() if rows else None
    return {
        'genome_id': genome.genome_id,
        'reports': reports,
        'header_split': 'BLAKE2B_HEADER_LEVEL_60_20_20',
        'alpha': alpha,
    }


def samples_for_split(samples: Sequence[TraceSample], split: str, *, seed: str = 'bitmyner-marker-v0.1') -> list[TraceSample]:
    if split not in {'train', 'validation', 'holdout'}:
        raise ValueError('split must be train, validation, or holdout')
    return [s for s in samples if split_name(s.header_id, seed=seed) == split]


def raven_rank_train(
    genomes: Sequence[MarkerGenome],
    samples: Sequence[TraceSample],
    *,
    seed: str = 'bitmyner-marker-v0.1',
    alpha: float = 0.05,
    min_marker_n: int = 32,
    top_k: int = 8,
) -> list[tuple[MarkerGenome, dict]]:
    """Rank candidates using training headers only.

    Validation and holdout rows are deliberately not evaluated here. HOUND
    receives only the already-selected top-K genomes in a separate phase.
    """
    train_samples = samples_for_split(samples, 'train', seed=seed)
    ranked: list[tuple[float, float, MarkerGenome, dict]] = []
    for genome in genomes:
        rows = observations_for_genome(genome, train_samples)
        if not rows:
            continue
        report = lift_report(rows, alpha=alpha).to_dict()
        if report['marker_n'] < min_marker_n or report['event_n'] == 0:
            continue
        ranked.append((report['lift_lcb'], report['lift'], genome, report))
    ranked.sort(key=lambda x: (x[0], x[1], x[2].genome_id), reverse=True)
    return [(g, r) for _, _, g, r in ranked[:top_k]]


def belzebub_promotion_verdict(
    evaluation: dict,
    *,
    delta: float = 0.0,
    min_holdout_marker_n: int = 32,
    require_validation: bool = True,
) -> dict:
    reports = evaluation['reports']
    holdout = reports.get('holdout')
    validation = reports.get('validation')
    reasons: list[str] = []
    if holdout is None:
        reasons.append('NO_HOLDOUT_ROWS')
    else:
        if holdout['marker_n'] < min_holdout_marker_n:
            reasons.append('INSUFFICIENT_HOLDOUT_MARKER_SUPPORT')
        if holdout['event_n'] == 0:
            reasons.append('NO_HOLDOUT_EVENTS')
        if holdout['lift_lcb'] <= 1.0 + delta:
            reasons.append('HOLDOUT_LCB_NOT_ABOVE_PROMOTION_THRESHOLD')
    if require_validation:
        if validation is None:
            reasons.append('NO_VALIDATION_ROWS')
        elif validation['lift_lcb'] <= 1.0:
            reasons.append('VALIDATION_LCB_NOT_ABOVE_ONE')
    return {
        'promote': not reasons,
        'reasons': reasons,
        'rule': 'HOLDOUT_LCB>1+delta AND VALIDATION_LCB>1',
        'delta': delta,
    }
