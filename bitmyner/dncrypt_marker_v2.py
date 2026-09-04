from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

from .dncrypt_marker import FeaturePredicate, MarkerGenome

SCHEMA = 'BITMYNER_DNCRYPT_MARKER_GENOME_V0_2'


def _predicate_key(p: FeaturePredicate) -> tuple[int, int, str, int, int]:
    return (p.stage, p.round, p.source, p.mask, p.value)


def composite_genome(
    predicates: Iterable[FeaturePredicate],
    *,
    mode: str = 'all',
    policy: str = 'pass',
) -> MarkerGenome:
    unique = {_predicate_key(p): p for p in predicates}
    if not unique:
        raise ValueError('composite genome requires at least one predicate')
    ordered = tuple(unique[key] for key in sorted(unique))
    return MarkerGenome(ordered, mode=mode, policy=policy, schema=SCHEMA)


def raven_pair_population(
    seed_genomes: Sequence[MarkerGenome],
    *,
    max_pairs: int = 128,
    mode: str = 'all',
) -> tuple[MarkerGenome, ...]:
    """Generate deterministic unique two-predicate genomes from seed motifs.

    RAVEN passes only training-selected seeds here. Predicates are flattened,
    canonicalized and paired without replacement; validation/holdout data never
    participates in candidate construction.
    """
    if max_pairs < 0:
        raise ValueError('max_pairs must be non-negative')
    predicates: dict[tuple[int, int, str, int, int], FeaturePredicate] = {}
    for genome in seed_genomes:
        for predicate in genome.predicates:
            predicates[_predicate_key(predicate)] = predicate
    ordered = [predicates[key] for key in sorted(predicates)]
    out: list[MarkerGenome] = []
    seen: set[str] = set()
    for p1, p2 in combinations(ordered, 2):
        g = composite_genome((p1, p2), mode=mode, policy='pass')
        if g.genome_id in seen:
            continue
        seen.add(g.genome_id)
        out.append(g)
        if len(out) >= max_pairs:
            break
    return tuple(out)
