from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
import hashlib
import json
import random
from typing import Iterable, Sequence

from .dncrypt_marker import FeaturePredicate, raven_single_bit_population
from .gremlin_marker import deterministic_headers, collect_samples
from .marker_eval import TraceSample, split_name

FULL_COST = 128
SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_1'


def _checkpoint_cost(p: FeaturePredicate) -> int:
    return p.round + 1 if p.stage == 1 else 64 + p.round + 1


def _wilson(successes: int, n: int, alpha: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    phat = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = z * sqrt((phat * (1 - phat) / n) + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass(frozen=True, slots=True)
class GenomeScore:
    genome: tuple[int, ...]
    score: float
    complexity: int
    per_level: tuple[dict, ...]


class CompiledCorpus:
    def __init__(self, samples: Sequence[TraceSample], predicates: Sequence[FeaturePredicate], thresholds: Sequence[int], split_seed: str):
        self.predicates = tuple(predicates)
        self.thresholds = tuple(thresholds)
        self.split_seed = split_seed
        self.samples_by_split: dict[str, list[TraceSample]] = {'train': [], 'validation': [], 'holdout': []}
        for s in samples:
            self.samples_by_split[split_name(s.header_id, seed=split_seed)].append(s)
        self.pred_masks: dict[str, tuple[int, ...]] = {}
        self.event_masks: dict[str, dict[int, int]] = {}
        self.universe: dict[str, int] = {}
        self.header_masks: dict[str, dict[str, int]] = {}
        for split, rows in self.samples_by_split.items():
            n = len(rows)
            self.universe[split] = (1 << n) - 1 if n else 0
            hmap: dict[str, int] = {}
            for i, s in enumerate(rows):
                hmap[s.header_id] = hmap.get(s.header_id, 0) | (1 << i)
            self.header_masks[split] = hmap
            em: dict[int, int] = {}
            for k in thresholds:
                mask = 0
                for i, s in enumerate(rows):
                    if s.trace.leading_zero_bits >= k:
                        mask |= 1 << i
                em[int(k)] = mask
            self.event_masks[split] = em
            masks = [0] * len(predicates)
            by_feature: dict[tuple[int, int, str], list[tuple[int, int, int]]] = {}
            for pi, p in enumerate(predicates):
                bit_index = p.mask.bit_length() - 1
                expected = 1 if p.value else 0
                by_feature.setdefault((p.stage, p.round, p.source), []).append((pi, bit_index, expected))
            for i, sample in enumerate(rows):
                sample_bit = 1 << i
                for (stage, round_index, source), routes in by_feature.items():
                    value = sample.trace.record(stage, round_index).source(source)
                    for pi, bit_index, expected in routes:
                        if ((value >> bit_index) & 1) == expected:
                            masks[pi] |= sample_bit
            self.pred_masks[split] = tuple(masks)

    def metric(self, genome: tuple[int, ...], split: str, k: int, *, sample_mask: int | None = None, alpha: float = 0.05) -> dict:
        universe = self.universe[split] if sample_mask is None else (self.universe[split] & sample_mask)
        n = universe.bit_count()
        if n == 0:
            return {'n': 0, 'marker_n': 0, 'event_n': 0, 'marker_event_n': 0, 'lift': 0.0, 'lift_lcb': 0.0, 'gamma': 0.0, 'gamma_lcb': 0.0, 'compute_fraction': 1.0}
        active = universe
        total_cost = 0
        ordered = sorted(genome, key=lambda idx: (_checkpoint_cost(self.predicates[idx]), idx))
        for idx in ordered:
            p = self.predicates[idx]
            match = self.pred_masks[split][idx] & universe
            fail = active & ~match
            total_cost += fail.bit_count() * _checkpoint_cost(p)
            active &= match
        total_cost += active.bit_count() * FULL_COST
        marker_n = active.bit_count()
        event_mask = self.event_masks[split][k] & universe
        event_n = event_mask.bit_count()
        marker_event_n = (active & event_mask).bit_count()
        p_event = event_n / n if n else 0.0
        p_cond = marker_event_n / marker_n if marker_n else 0.0
        lift = p_cond / p_event if p_event else 0.0
        cond_lo, _ = _wilson(marker_event_n, marker_n, alpha) if marker_n else (0.0, 1.0)
        _, base_hi = _wilson(event_n, n, alpha) if event_n else (0.0, 1.0)
        lift_lcb = cond_lo / base_hi if base_hi else 0.0
        gamma = ((marker_event_n / total_cost) / (event_n / (n * FULL_COST))) if event_n and marker_event_n and total_cost else 0.0
        pass_rate = marker_n / n
        avg_cost = total_cost / n
        gamma_lcb = FULL_COST * pass_rate * lift_lcb / avg_cost if avg_cost else 0.0
        return {
            'n': n,
            'marker_n': marker_n,
            'event_n': event_n,
            'marker_event_n': marker_event_n,
            'lift': lift,
            'lift_lcb': lift_lcb,
            'gamma': gamma,
            'gamma_lcb': gamma_lcb,
            'compute_fraction': total_cost / (n * FULL_COST),
        }


def _canonical(genome: Iterable[int], max_predicates: int) -> tuple[int, ...]:
    return tuple(sorted(set(int(x) for x in genome)))[:max_predicates]


def _score_genome(corpus: CompiledCorpus, genome: tuple[int, ...], *, train_mask: int, thresholds: Sequence[int], alpha: float = 0.05) -> GenomeScore:
    reports = []
    qualified = []
    for k in thresholds:
        r = corpus.metric(genome, 'train', k, sample_mask=train_mask, alpha=alpha)
        reports.append({'k': int(k), **r})
        if r['event_n'] >= 8 and r['marker_n'] >= 32 and r['marker_event_n'] >= 2:
            qualified.append(r['gamma_lcb'])
    score = 0.0 if not qualified else min(qualified) - 0.01 * max(0, len(genome) - 1)
    return GenomeScore(genome, score, len(genome), tuple(reports))


def run_training_1000(
    *,
    iterations: int = 1000,
    header_count: int = 120,
    nonces_per_header: int = 192,
    thresholds: Sequence[int] = (4, 6, 8),
    population_size: int = 64,
    offspring_per_iteration: int = 8,
    max_predicates: int = 3,
    seed: int = 616,
    split_seed: str = 'bitmyner-marker-training-v0.1',
    checkpoint_every: int = 100,
) -> dict:
    if iterations < 1 or header_count < 3 or nonces_per_header < 1:
        raise ValueError('invalid training dimensions')
    rng = random.Random(seed)
    headers = deterministic_headers(header_count, seed=seed)
    samples = collect_samples(headers, nonces_per_header=nonces_per_header, event_k=min(thresholds))
    genomes = raven_single_bit_population(stages=(1, 2), rounds=(0, 7, 15, 31, 47, 63), sources=('a', 'e', 't1', 'w'), bits=(0, 7, 15, 23, 31))
    predicates = tuple(g.predicates[0] for g in genomes)
    corpus = CompiledCorpus(samples, predicates, thresholds, split_seed)
    train_headers = sorted(corpus.header_masks['train'])
    if len(train_headers) < 6:
        raise RuntimeError('insufficient training headers')
    rng.shuffle(train_headers)
    warm_headers = min(5, len(train_headers))
    seen_headers = list(train_headers[:warm_headers])
    next_header = warm_headers
    train_mask = 0
    for hid in seen_headers:
        train_mask |= corpus.header_masks['train'][hid]
    add_interval = max(1, iterations // max(1, len(train_headers) - warm_headers))

    universe_ids = tuple(range(len(predicates)))
    initial = [_canonical((rng.choice(universe_ids),), max_predicates) for _ in range(population_size * 2)]
    archive: set[tuple[int, ...]] = set()
    pop_scores: dict[tuple[int, ...], GenomeScore] = {}
    evaluated = 0
    duplicates = 0
    for g in initial:
        if not g or g in archive:
            continue
        archive.add(g)
        pop_scores[g] = _score_genome(corpus, g, train_mask=train_mask, thresholds=thresholds)
        evaluated += 1
        if len(pop_scores) >= population_size:
            break

    def rerank() -> None:
        nonlocal pop_scores
        rescored = {g: _score_genome(corpus, g, train_mask=train_mask, thresholds=thresholds) for g in pop_scores}
        ordered = sorted(rescored.values(), key=lambda x: (x.score, -x.complexity, x.genome), reverse=True)[:population_size]
        pop_scores = {x.genome: x for x in ordered}

    def propose(parent_keys: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
        if len(parent_keys) >= 2 and rng.random() < 0.20:
            a, b = rng.sample(list(parent_keys), 2)
            combined = list(set(a) | set(b))
            rng.shuffle(combined)
            return _canonical(combined[:max_predicates], max_predicates)
        p = list(rng.choice(parent_keys))
        u = rng.random()
        if u < 0.35 and len(p) < max_predicates:
            p.append(rng.choice(universe_ids))
        elif u < 0.50 and len(p) > 1:
            p.pop(rng.randrange(len(p)))
        else:
            p[rng.randrange(len(p))] = rng.choice(universe_ids)
        return _canonical(p, max_predicates)

    checkpoints: list[dict] = []
    champion_history: list[str] = []

    def checkpoint(iteration: int) -> None:
        rerank()
        champion = max(pop_scores.values(), key=lambda x: (x.score, -x.complexity, x.genome))
        champion_history.append(hashlib.blake2b(str(champion.genome).encode(), digest_size=8).hexdigest())
        validation = [corpus.metric(champion.genome, 'validation', k, alpha=0.05) for k in thresholds]
        checkpoints.append({
            'iteration': iteration,
            'train_headers_seen': len(seen_headers),
            'train_samples_seen': train_mask.bit_count(),
            'unique_genomes_evaluated': evaluated,
            'duplicate_proposals': duplicates,
            'archive_size': len(archive),
            'population_size': len(pop_scores),
            'champion_genome': list(champion.genome),
            'champion_complexity': champion.complexity,
            'train_score': champion.score,
            'train_levels': list(champion.per_level),
            'validation_levels_monitor_only': [{'k': int(k), **r} for k, r in zip(thresholds, validation)],
        })

    checkpoint(0)
    for it in range(1, iterations + 1):
        if next_header < len(train_headers) and it % add_interval == 0:
            hid = train_headers[next_header]
            next_header += 1
            seen_headers.append(hid)
            train_mask |= corpus.header_masks['train'][hid]
            rerank()
        ranked = sorted(pop_scores.values(), key=lambda x: (x.score, -x.complexity, x.genome), reverse=True)
        parent_keys = [x.genome for x in ranked[:max(8, population_size // 4)]]
        children: list[GenomeScore] = []
        attempts = 0
        while len(children) < offspring_per_iteration and attempts < offspring_per_iteration * 40:
            attempts += 1
            g = propose(parent_keys)
            if not g or g in archive:
                duplicates += 1
                continue
            archive.add(g)
            children.append(_score_genome(corpus, g, train_mask=train_mask, thresholds=thresholds))
            evaluated += 1
        for s in children:
            pop_scores[s.genome] = s
        ordered = sorted(pop_scores.values(), key=lambda x: (x.score, -x.complexity, x.genome), reverse=True)[:population_size]
        pop_scores = {x.genome: x for x in ordered}
        if it % checkpoint_every == 0 or it == iterations:
            checkpoint(it)

    final_ranked = sorted(pop_scores.values(), key=lambda x: (x.score, -x.complexity, x.genome), reverse=True)[:8]
    alpha_family = 0.05
    alpha_per_test = alpha_family / max(1, len(final_ranked) * len(thresholds) * 2)
    finals = []
    promoted = []
    for s in final_ranked:
        ladder = []
        qualified = []
        for k in thresholds:
            vr = corpus.metric(s.genome, 'validation', k, alpha=alpha_per_test)
            hr = corpus.metric(s.genome, 'holdout', k, alpha=alpha_per_test)
            q = bool(vr['event_n'] and hr['event_n'] and vr['marker_n'] >= 32 and hr['marker_n'] >= 32 and vr['gamma_lcb'] > 1.0 and hr['gamma_lcb'] > 1.0)
            qualified.append(q)
            ladder.append({'k': int(k), 'validation': vr, 'holdout': hr, 'qualified': q})
        longest = cur = 0
        for q in qualified:
            cur = cur + 1 if q else 0
            longest = max(longest, cur)
        verdict = longest >= min(2, len(thresholds))
        row = {'genome': list(s.genome), 'train_score': s.score, 'ladder': ladder, 'longest_qualified_run': longest, 'promote': verdict}
        finals.append(row)
        if verdict:
            promoted.append(row)

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_TRAINING_COMPLETE',
        'iterations': iterations,
        'fixture': {
            'synthetic_headers': True,
            'header_count': header_count,
            'nonces_per_header': nonces_per_header,
            'sample_count': len(samples),
            'split_counts': {k: len(v) for k, v in corpus.samples_by_split.items()},
            'thresholds': list(thresholds),
            'seed': seed,
            'split_seed': split_seed,
        },
        'training_policy': {
            'selection_surface': 'TRAIN_ONLY',
            'validation_role': 'MONITOR_ONLY_NO_SELECTION_AUTHORITY',
            'holdout_role': 'FROZEN_UNTIL_ITERATION_1000_COMPLETE',
            'population_size': population_size,
            'offspring_per_iteration': offspring_per_iteration,
            'max_predicates': max_predicates,
            'evidence_growth': 'HEADER_LEVEL_INCREMENTAL',
            'objective': 'MIN_SUPPORTED_TRAIN_GAMMA_LCB_MINUS_COMPLEXITY_PENALTY',
        },
        'checkpoints': checkpoints,
        'search': {
            'unique_genomes_evaluated': evaluated,
            'duplicate_proposals': duplicates,
            'archive_size': len(archive),
            'champion_changes_at_checkpoints': sum(a != b for a, b in zip(champion_history, champion_history[1:])),
        },
        'final_top8': finals,
        'promoted_count': len(promoted),
        'multiple_testing': {
            'familywise_alpha': alpha_family,
            'per_test_alpha': alpha_per_test,
            'correction': 'BONFERRONI_TOP8_X_LEVELS_X_VALIDATION_HOLDOUT',
        },
        'claims': {
            'cryptographic_advantage_assumed': False,
            'production_policy_changed': False,
            'holdout_used_during_training': False,
            'physical_time_claim': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32).hexdigest()
    return payload
