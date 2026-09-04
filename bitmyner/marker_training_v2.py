from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import random
import statistics
from typing import Iterable, Sequence

from .dncrypt_marker import FeaturePredicate, raven_single_bit_population
from .gremlin_marker import deterministic_headers, collect_samples
from .marker_training import CompiledCorpus, _canonical

FULL_COST = 128
SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_2'


def _genome_distance(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard distance over canonical predicate-index sets."""
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / len(sa | sb))


def _novelty_score(genome: tuple[int, ...], references: Sequence[tuple[int, ...]], *, k: int = 8) -> float:
    refs = [tuple(r) for r in references if tuple(r) != tuple(genome)]
    if not refs:
        return 1.0
    distances = sorted(_genome_distance(genome, r) for r in refs)
    take = distances[: max(1, min(int(k), len(distances)))]
    return sum(take) / len(take)


class NegativeMemory:
    """Training-only memory of repeatedly unstable/weak predicate families."""

    def __init__(self) -> None:
        self._counts: Counter[int] = Counter()
        self.observations = 0
        self.total_weight = 0

    def observe(self, genome: Iterable[int], *, severity: float) -> None:
        sev = max(0.0, min(1.0, float(severity)))
        if sev <= 0.0:
            return
        weight = max(1, int(round(1.0 + 4.0 * sev)))
        g = tuple(sorted(set(int(x) for x in genome)))
        for idx in g:
            self._counts[idx] += weight
        self.observations += 1
        self.total_weight += weight

    def penalty(self, genome: Iterable[int]) -> float:
        g = tuple(sorted(set(int(x) for x in genome)))
        if not g or not self._counts:
            return 0.0
        m = max(self._counts.values())
        if m <= 0:
            return 0.0
        return sum(self._counts.get(idx, 0) / m for idx in g) / len(g)

    def snapshot(self, *, top_n: int = 16) -> dict:
        return {
            'observations': self.observations,
            'total_weight': self.total_weight,
            'tracked_predicates': len(self._counts),
            'top_predicates': [[int(k), int(v)] for k, v in self._counts.most_common(top_n)],
        }


@dataclass(frozen=True, slots=True)
class AdaptiveGenomeScore:
    genome: tuple[int, ...]
    score: float
    robust_core: float
    novelty: float
    negative_penalty: float
    genealogy_penalty: float
    instability: float
    complexity: int
    per_level: tuple[dict, ...]
    lineage_roots: tuple[int, ...]


def _cohort_masks(
    corpus: CompiledCorpus,
    seen_headers: Sequence[str],
    *,
    cohort_count: int,
    epoch: int,
    seed: str,
) -> tuple[int, ...]:
    headers = list(dict.fromkeys(seen_headers))
    if not headers:
        return tuple()
    c = max(2, min(int(cohort_count), max(2, len(headers) // 2), len(headers)))
    keyed = []
    for hid in headers:
        h = hashlib.blake2b(f'{seed}|{epoch}|{hid}'.encode(), digest_size=8).digest()
        keyed.append((h, hid))
    keyed.sort()
    groups = [0] * c
    for i, (_, hid) in enumerate(keyed):
        groups[i % c] |= corpus.header_masks['train'][hid]
    return tuple(mask for mask in groups if mask)


def _lower_quartile(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(float(x) for x in values)
    return s[int(math.floor((len(s) - 1) * 0.25))]


def _genealogy_penalty(roots: tuple[int, ...], descendant_counts: Counter[int]) -> float:
    if not roots or not descendant_counts:
        return 0.0
    m = max(descendant_counts.values(), default=0)
    if m <= 0:
        return 0.0
    return sum(descendant_counts.get(r, 0) / m for r in roots) / len(roots)


def _score_genome_v2(
    corpus: CompiledCorpus,
    genome: tuple[int, ...],
    *,
    train_mask: int,
    cohort_masks: Sequence[int],
    thresholds: Sequence[int],
    novelty_refs: Sequence[tuple[int, ...]],
    negative_memory: NegativeMemory,
    genealogy_penalty: float,
    lineage_roots: tuple[int, ...],
    alpha: float = 0.05,
) -> AdaptiveGenomeScore:
    reports: list[dict] = []
    robust_levels: list[float] = []
    instability_levels: list[float] = []

    for k in thresholds:
        full = corpus.metric(genome, 'train', int(k), sample_mask=train_mask, alpha=alpha)
        cohort_rows = [corpus.metric(genome, 'train', int(k), sample_mask=m, alpha=alpha) for m in cohort_masks]
        supported = [r for r in cohort_rows if r['event_n'] >= 2 and r['marker_n'] >= 8]
        cohort_gammas = [float(r['gamma']) for r in supported]
        q25 = _lower_quartile(cohort_gammas)
        med = statistics.median(cohort_gammas) if cohort_gammas else 0.0
        spread = 0.0
        if len(cohort_gammas) >= 2 and med > 0:
            spread = min(2.0, (max(cohort_gammas) - min(cohort_gammas)) / med) / 2.0

        full_supported = bool(full['event_n'] >= 8 and full['marker_n'] >= 32 and full['marker_event_n'] >= 2)
        cohort_supported = len(supported) >= min(2, max(1, len(cohort_masks)))
        robust = min(float(full['gamma_lcb']), q25) if full_supported and cohort_supported else 0.0
        if full_supported and cohort_supported:
            robust_levels.append(robust)
            instability_levels.append(spread)
        reports.append({
            'k': int(k),
            'full_train': full,
            'cohort_count': len(cohort_rows),
            'supported_cohorts': len(supported),
            'cohort_gamma': cohort_gammas,
            'cohort_gamma_q25': q25,
            'cohort_gamma_median': med,
            'cohort_instability': spread,
            'robust_gamma': robust,
        })

    robust_core = min(robust_levels) if robust_levels else 0.0
    instability = statistics.mean(instability_levels) if instability_levels else 1.0
    novelty = _novelty_score(genome, novelty_refs, k=8)
    neg = negative_memory.penalty(genome)
    complexity_penalty = 0.0125 * max(0, len(genome) - 1)
    novelty_bonus = (0.04 if robust_core > 0 else 0.005) * novelty
    score = robust_core + novelty_bonus - 0.08 * instability - 0.05 * neg - 0.03 * genealogy_penalty - complexity_penalty
    return AdaptiveGenomeScore(
        genome=genome,
        score=score,
        robust_core=robust_core,
        novelty=novelty,
        negative_penalty=neg,
        genealogy_penalty=genealogy_penalty,
        instability=instability,
        complexity=len(genome),
        per_level=tuple(reports),
        lineage_roots=lineage_roots,
    )


def _severity_for_negative_memory(score: AdaptiveGenomeScore) -> float:
    weakness = max(0.0, min(1.0, 0.90 - score.robust_core))
    instability = max(0.0, min(1.0, score.instability))
    return max(weakness, instability - 0.20)


def run_training_1000_v2(
    *,
    iterations: int = 1000,
    header_count: int = 100,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4, 6, 8),
    population_size: int = 64,
    offspring_per_iteration: int = 8,
    max_predicates: int = 3,
    seed: int = 616,
    split_seed: str = 'bitmyner-marker-training-v0.1',
    checkpoint_every: int = 100,
    cohort_count: int = 4,
    cohort_refresh_every: int = 25,
) -> dict:
    if iterations < 1 or header_count < 3 or nonces_per_header < 1:
        raise ValueError('invalid training dimensions')
    if population_size < 2 or offspring_per_iteration < 1 or max_predicates < 1:
        raise ValueError('invalid evolution dimensions')
    if cohort_count < 2 or cohort_refresh_every < 1:
        raise ValueError('invalid cohort configuration')

    rng = random.Random(seed)
    headers = deterministic_headers(header_count, seed=seed)
    samples = collect_samples(headers, nonces_per_header=nonces_per_header, event_k=min(thresholds))
    single_genomes = raven_single_bit_population(
        stages=(1, 2),
        rounds=(0, 7, 15, 31, 47, 63),
        sources=('a', 'e', 't1', 'w'),
        bits=(0, 7, 15, 23, 31),
    )
    predicates: tuple[FeaturePredicate, ...] = tuple(g.predicates[0] for g in single_genomes)
    corpus = CompiledCorpus(samples, predicates, thresholds, split_seed)
    train_headers = sorted(corpus.header_masks['train'])
    if len(train_headers) < 6:
        raise RuntimeError('insufficient training headers')
    rng.shuffle(train_headers)

    warm_headers = min(len(train_headers), max(8, cohort_count * 3))
    seen_headers = list(train_headers[:warm_headers])
    next_header = warm_headers
    train_mask = 0
    for hid in seen_headers:
        train_mask |= corpus.header_masks['train'][hid]
    remaining = max(0, len(train_headers) - warm_headers)
    add_interval = max(1, iterations // max(1, remaining)) if remaining else iterations + 1

    universe_ids = tuple(range(len(predicates)))
    archive: set[tuple[int, ...]] = set()
    population: dict[tuple[int, ...], AdaptiveGenomeScore] = {}
    lineage_by_genome: dict[tuple[int, ...], tuple[int, ...]] = {}
    descendant_counts: Counter[int] = Counter()
    negative_memory = NegativeMemory()
    next_root = 0
    evaluated = 0
    duplicates = 0
    cohort_epoch = 0
    cohort_refreshes = 0
    operator_stats = {
        'mutate': {'proposed': 0, 'accepted_unique': 0, 'duplicates': 0},
        'crossover': {'proposed': 0, 'accepted_unique': 0, 'duplicates': 0},
        'restart': {'proposed': 0, 'accepted_unique': 0, 'duplicates': 0},
    }

    cohort_masks = _cohort_masks(corpus, seen_headers, cohort_count=cohort_count, epoch=cohort_epoch, seed=split_seed)

    def score_candidate(g: tuple[int, ...], roots: tuple[int, ...], refs: Sequence[tuple[int, ...]]) -> AdaptiveGenomeScore:
        gp = _genealogy_penalty(roots, descendant_counts)
        return _score_genome_v2(
            corpus,
            g,
            train_mask=train_mask,
            cohort_masks=cohort_masks,
            thresholds=thresholds,
            novelty_refs=refs,
            negative_memory=negative_memory,
            genealogy_penalty=gp,
            lineage_roots=roots,
        )

    attempts = 0
    while len(population) < population_size and attempts < population_size * 100:
        attempts += 1
        size = 1 if rng.random() < 0.75 else min(max_predicates, 2)
        g = _canonical(rng.sample(universe_ids, size), max_predicates)
        if not g or g in archive:
            duplicates += 1
            continue
        root = next_root
        next_root += 1
        roots = (root,)
        archive.add(g)
        lineage_by_genome[g] = roots
        s = score_candidate(g, roots, tuple(population))
        population[g] = s
        descendant_counts[root] += 1
        evaluated += 1
    if len(population) < 2:
        raise RuntimeError('failed to initialize diverse population')

    def rank_key(s: AdaptiveGenomeScore):
        return (s.score, s.robust_core, s.novelty, -s.negative_penalty, -s.genealogy_penalty, -s.complexity, tuple(-x for x in s.genome))

    def rerank() -> None:
        nonlocal population
        refs = tuple(population)
        rescored: dict[tuple[int, ...], AdaptiveGenomeScore] = {}
        for g in population:
            roots = lineage_by_genome[g]
            rescored[g] = score_candidate(g, roots, refs)
        ordered = sorted(rescored.values(), key=rank_key, reverse=True)[:population_size]
        population = {s.genome: s for s in ordered}

    def choose_parent(ranked: Sequence[AdaptiveGenomeScore]) -> AdaptiveGenomeScore:
        pool = list(ranked[: max(8, len(ranked) // 2)])
        weights = [1.0 / (1.0 + 3.0 * s.genealogy_penalty + 2.0 * s.negative_penalty) for s in pool]
        return rng.choices(pool, weights=weights, k=1)[0]

    def propose(ranked: Sequence[AdaptiveGenomeScore]) -> tuple[tuple[int, ...], tuple[int, ...], str]:
        nonlocal next_root
        u = rng.random()
        if u < 0.18 and len(ranked) >= 2:
            a = choose_parent(ranked)
            b = choose_parent(ranked)
            guard = 0
            while b.genome == a.genome and guard < 8:
                b = choose_parent(ranked)
                guard += 1
            merged = list(set(a.genome) | set(b.genome))
            rng.shuffle(merged)
            g = _canonical(merged[:max_predicates], max_predicates)
            roots = tuple(sorted(set(a.lineage_roots) | set(b.lineage_roots)))[:4]
            return g, roots, 'crossover'
        if u < 0.35:
            size = rng.randint(1, max_predicates)
            g = _canonical(rng.sample(universe_ids, size), max_predicates)
            root = next_root
            next_root += 1
            return g, (root,), 'restart'

        p = choose_parent(ranked)
        genes = list(p.genome)
        m = rng.random()
        if m < 0.35 and len(genes) < max_predicates:
            genes.append(rng.choice(universe_ids))
        elif m < 0.50 and len(genes) > 1:
            genes.pop(rng.randrange(len(genes)))
        else:
            genes[rng.randrange(len(genes))] = rng.choice(universe_ids)
        return _canonical(genes, max_predicates), p.lineage_roots, 'mutate'

    checkpoints: list[dict] = []
    champion_history: list[str] = []

    def checkpoint(iteration: int) -> None:
        rerank()
        champion = max(population.values(), key=rank_key)
        champion_history.append(hashlib.blake2b(str(champion.genome).encode(), digest_size=8).hexdigest())
        checkpoints.append({
            'iteration': iteration,
            'train_headers_seen': len(seen_headers),
            'train_samples_seen': train_mask.bit_count(),
            'cohort_epoch': cohort_epoch,
            'cohort_sample_counts': [m.bit_count() for m in cohort_masks],
            'unique_genomes_evaluated': evaluated,
            'duplicate_proposals': duplicates,
            'archive_size': len(archive),
            'population_size': len(population),
            'champion_genome': list(champion.genome),
            'champion_complexity': champion.complexity,
            'train_score': champion.score,
            'robust_core': champion.robust_core,
            'novelty': champion.novelty,
            'negative_penalty': champion.negative_penalty,
            'genealogy_penalty': champion.genealogy_penalty,
            'cohort_instability': champion.instability,
            'train_levels': list(champion.per_level),
            'validation_observed': False,
            'holdout_observed': False,
        })

    checkpoint(0)
    for it in range(1, iterations + 1):
        evidence_changed = False
        if next_header < len(train_headers) and it % add_interval == 0:
            hid = train_headers[next_header]
            next_header += 1
            seen_headers.append(hid)
            train_mask |= corpus.header_masks['train'][hid]
            evidence_changed = True

        if it % cohort_refresh_every == 0:
            cohort_epoch += 1
            cohort_masks = _cohort_masks(corpus, seen_headers, cohort_count=cohort_count, epoch=cohort_epoch, seed=split_seed)
            cohort_refreshes += 1
            evidence_changed = True

        if evidence_changed:
            rerank()

        ranked = sorted(population.values(), key=rank_key, reverse=True)
        children: list[AdaptiveGenomeScore] = []
        attempts = 0
        max_attempts = offspring_per_iteration * 80
        while len(children) < offspring_per_iteration and attempts < max_attempts:
            attempts += 1
            g, roots, operator = propose(ranked)
            operator_stats[operator]['proposed'] += 1
            if not g or g in archive:
                duplicates += 1
                operator_stats[operator]['duplicates'] += 1
                continue

            refs = tuple(population) + tuple(s.genome for s in children)
            s = score_candidate(g, roots, refs)
            archive.add(g)
            lineage_by_genome[g] = roots
            for root in roots:
                descendant_counts[root] += 1
            evaluated += 1
            operator_stats[operator]['accepted_unique'] += 1
            children.append(s)

            if train_mask.bit_count() >= 512:
                severity = _severity_for_negative_memory(s)
                if severity >= 0.20:
                    negative_memory.observe(g, severity=severity)

        for s in children:
            population[s.genome] = s
        rerank()
        ordered = sorted(population.values(), key=rank_key, reverse=True)[:population_size]
        population = {s.genome: s for s in ordered}

        if it % checkpoint_every == 0 or it == iterations:
            checkpoint(it)

    final_ranked = sorted(population.values(), key=rank_key, reverse=True)[:8]
    alpha_family = 0.05
    alpha_per_test = alpha_family / max(1, len(final_ranked) * len(thresholds) * 2)
    finals: list[dict] = []
    promoted: list[dict] = []
    for s in final_ranked:
        ladder = []
        qualified = []
        for k in thresholds:
            vr = corpus.metric(s.genome, 'validation', int(k), alpha=alpha_per_test)
            hr = corpus.metric(s.genome, 'holdout', int(k), alpha=alpha_per_test)
            q = bool(
                vr['event_n'] and hr['event_n']
                and vr['marker_n'] >= 32 and hr['marker_n'] >= 32
                and vr['gamma_lcb'] > 1.0 and hr['gamma_lcb'] > 1.0
            )
            qualified.append(q)
            ladder.append({'k': int(k), 'validation': vr, 'holdout': hr, 'qualified': q})
        longest = cur = 0
        for q in qualified:
            cur = cur + 1 if q else 0
            longest = max(longest, cur)
        verdict = longest >= min(2, len(thresholds))
        row = {
            'genome': list(s.genome),
            'train_score': s.score,
            'robust_core': s.robust_core,
            'novelty': s.novelty,
            'negative_penalty': s.negative_penalty,
            'genealogy_penalty': s.genealogy_penalty,
            'cohort_instability': s.instability,
            'ladder': ladder,
            'longest_qualified_run': longest,
            'promote': verdict,
        }
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
            'selection_surface': 'TRAIN_COHORTS_ONLY',
            'validation_role': 'FROZEN_UNTIL_TRAINING_COMPLETE_NO_SELECTION_AUTHORITY',
            'holdout_role': 'FROZEN_UNTIL_TRAINING_COMPLETE',
            'population_size': population_size,
            'offspring_per_iteration': offspring_per_iteration,
            'max_predicates': max_predicates,
            'evidence_growth': 'HEADER_LEVEL_INCREMENTAL',
            'rolling_header_cohorts': True,
            'cohort_count': cohort_count,
            'cohort_refresh_every': cohort_refresh_every,
            'novelty_search': True,
            'negative_memory': True,
            'genealogy_penalty': True,
            'objective': 'ROBUST_TRAIN_COHORT_GAMMA_WITH_NOVELTY_NEGATIVE_MEMORY_GENEALOGY_AND_INSTABILITY_TERMS',
        },
        'checkpoints': checkpoints,
        'search': {
            'unique_genomes_evaluated': evaluated,
            'duplicate_proposals': duplicates,
            'duplicate_proposal_fraction': duplicates / max(1, duplicates + evaluated),
            'archive_size': len(archive),
            'cohort_refreshes': cohort_refreshes,
            'root_lineages_created': next_root,
            'negative_memory': negative_memory.snapshot(),
            'operator_stats': operator_stats,
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
            'validation_used_during_training': False,
            'holdout_used_during_training': False,
            'physical_time_claim': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
