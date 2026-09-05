from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import random
import statistics
from typing import Sequence

from .gremlin_marker import deterministic_headers, collect_samples
from .marker_training import _canonical
from .marker_training_v2 import NegativeMemory, _genome_distance, _novelty_score
from .structural_marker import StructuralCompiledCorpus, StructuralPredicate, structural_population, nonce_variation_filter

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_4_STRUCTURAL_CAUSAL'


def _cohort_masks(corpus: StructuralCompiledCorpus, headers: Sequence[str], *, count: int, epoch: int, seed: str) -> tuple[int, ...]:
    hs = list(dict.fromkeys(headers))
    if not hs:
        return tuple()
    c = min(max(2, count), len(hs))
    keyed = sorted((hashlib.blake2b(f'{seed}|{epoch}|{h}'.encode(), digest_size=8).digest(), h) for h in hs)
    groups = [0] * c
    for i, (_, h) in enumerate(keyed):
        groups[i % c] |= corpus.header_masks['train'][h]
    return tuple(x for x in groups if x)


def _q25(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(float(x) for x in values)
    return s[int(math.floor((len(s)-1)*0.25))]


@dataclass(frozen=True, slots=True)
class Score:
    genome: tuple[int, ...]
    score: float
    robust_core: float
    novelty: float
    instability: float
    negative_penalty: float
    complexity: int
    per_level: tuple[dict, ...]


def _score(corpus: StructuralCompiledCorpus, genome: tuple[int, ...], *, train_mask: int, cohorts: Sequence[int], thresholds: Sequence[int], refs: Sequence[tuple[int,...]], neg: NegativeMemory) -> Score:
    rows = []
    robust = []
    instability = []
    for k in thresholds:
        full = corpus.metric(genome, 'train', int(k), sample_mask=train_mask)
        cms = [corpus.metric(genome, 'train', int(k), sample_mask=m) for m in cohorts]
        supported = [r for r in cms if r['event_n'] >= 2 and r['marker_n'] >= 8 and r['marker_event_n'] >= 1]
        gammas = [r['gamma'] for r in supported]
        q = _q25(gammas)
        med = statistics.median(gammas) if gammas else 0.0
        spread = 1.0
        if len(gammas) >= 2 and med > 0:
            spread = min(1.0, (max(gammas)-min(gammas)) / (2*med))
        okay = full['event_n'] >= 8 and full['marker_n'] >= 32 and full['marker_event_n'] >= 2 and len(supported) >= min(2, len(cohorts))
        rg = min(full['gamma_lcb'], q) if okay else 0.0
        if okay:
            robust.append(rg)
            instability.append(spread)
        rows.append({'k':int(k),'full_train':full,'cohort_gamma':gammas,'robust_gamma':rg,'cohort_instability':spread})
    core = min(robust) if robust else 0.0
    inst = statistics.mean(instability) if instability else 1.0
    novelty = _novelty_score(genome, refs, k=8)
    negp = neg.penalty(genome)
    complexity_penalty = 0.0125 * max(0, len(genome)-1)
    s = core + 0.035*novelty - 0.075*inst - 0.045*negp - complexity_penalty
    return Score(genome, s, core, novelty, inst, negp, len(genome), tuple(rows))


def _consistent_genome(genome: tuple[int, ...], predicates: Sequence[StructuralPredicate]) -> bool:
    seen: dict[tuple[int,int,str,str,str|None,int], int] = {}
    for idx in genome:
        p = predicates[idx]
        key = (p.stage, p.round, p.op, p.left, p.right, p.bit)
        prev = seen.get(key)
        if prev is not None and prev != p.value:
            return False
        seen[key] = p.value
    return True


def _family_signature(genome: tuple[int,...], predicates: Sequence[StructuralPredicate]) -> tuple[str, ...]:
    return tuple(sorted(predicates[i].family for i in genome))


def run_training_1000_v4(
    *,
    iterations: int = 1000,
    header_count: int = 100,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4,6,8),
    population_size: int = 64,
    offspring_per_iteration: int = 8,
    max_predicates: int = 3,
    seed: int = 616,
    split_seed: str = 'bitmyner-marker-training-v0.4',
    checkpoint_every: int = 100,
    cohort_count: int = 4,
    cohort_refresh_every: int = 25,
    family_whitelist: Sequence[str] | None = None,
    family_prior: dict[str, float] | None = None,
    family_prior_strength: float = 0.0,
    final_evaluate: bool = True,
) -> dict:
    rng = random.Random(seed)
    headers = deterministic_headers(header_count, seed=seed)
    samples = collect_samples(headers, nonces_per_header=nonces_per_header, event_k=min(thresholds), nonce_seed=seed ^ 0x5EED)
    raw_predicates = structural_population()
    predicates, filter_receipt = nonce_variation_filter(samples, raw_predicates, split_seed=split_seed)
    if family_whitelist is not None:
        allowed = set(str(x) for x in family_whitelist)
        before = len(predicates)
        predicates = tuple(p for p in predicates if p.family in allowed)
        filter_receipt = dict(filter_receipt)
        filter_receipt['family_whitelist'] = sorted(allowed)
        filter_receipt['pre_whitelist_kept_predicates'] = before
        filter_receipt['post_whitelist_kept_predicates'] = len(predicates)
    if len(predicates) < 2:
        raise RuntimeError('too few nonce-varying structural predicates')
    requested_population_size = int(population_size)
    population_size = min(requested_population_size, len(predicates))
    corpus = StructuralCompiledCorpus(samples, predicates, thresholds, split_seed)
    train_headers = sorted(corpus.header_masks['train'])
    if len(train_headers) < 8:
        raise RuntimeError('insufficient train headers')
    train_mask = corpus.universe['train']
    cohort_epoch = 0
    cohorts = _cohort_masks(corpus, train_headers, count=cohort_count, epoch=cohort_epoch, seed=split_seed)
    universe = tuple(range(len(predicates)))
    prior_map = {str(k): max(0.0, float(v)) for k,v in (family_prior or {}).items()}
    prior_strength = max(0.0, float(family_prior_strength))
    prior_indices = tuple(idx for idx in universe if prior_map.get(predicates[idx].family, 0.0) > 0.0)
    prior_weights = tuple(prior_map[predicates[idx].family] for idx in prior_indices)
    prior_probability = 0.0 if not prior_indices or prior_strength <= 0.0 else min(0.90, prior_strength / (1.0 + prior_strength))

    def pick_gene(excluded: set[int] | None = None) -> int:
        blocked = excluded or set()
        if prior_indices and rng.random() < prior_probability:
            for _ in range(16):
                picked = rng.choices(prior_indices, weights=prior_weights, k=1)[0]
                if picked not in blocked:
                    return picked
        for _ in range(64):
            picked = rng.choice(universe)
            if picked not in blocked:
                return picked
        remaining = [idx for idx in universe if idx not in blocked]
        if not remaining:
            raise RuntimeError('empty structural predicate pool')
        return rng.choice(remaining)

    def weighted_sample(k: int) -> list[int]:
        out: list[int] = []
        blocked: set[int] = set()
        for _ in range(min(int(k), len(universe))):
            picked = pick_gene(blocked)
            out.append(picked)
            blocked.add(picked)
        return out
    archive: set[tuple[int,...]] = set()
    neg = NegativeMemory()
    population: dict[tuple[int,...], Score] = {}
    evaluated = duplicates = refreshes = 0

    while len(population) < population_size:
        size = 1 if rng.random() < 0.75 else min(2, max_predicates)
        g = _canonical(weighted_sample(size), max_predicates)
        if g in archive or not _consistent_genome(g, predicates):
            duplicates += 1
            continue
        archive.add(g)
        population[g] = _score(corpus, g, train_mask=train_mask, cohorts=cohorts, thresholds=thresholds, refs=tuple(population), neg=neg)
        evaluated += 1

    def key(s: Score):
        return (s.score, s.robust_core, s.novelty, -s.instability, -s.negative_penalty, -s.complexity, tuple(-x for x in s.genome))

    def rerank() -> None:
        nonlocal population
        refs = tuple(population)
        rescored = {g:_score(corpus,g,train_mask=train_mask,cohorts=cohorts,thresholds=thresholds,refs=refs,neg=neg) for g in population}
        population = {s.genome:s for s in sorted(rescored.values(), key=key, reverse=True)[:population_size]}

    def propose(ranked: Sequence[Score]) -> tuple[int,...]:
        u = rng.random()
        if u < 0.18 and len(ranked) >= 2:
            a,b = rng.sample(list(ranked[:max(8,len(ranked)//2)]),2)
            genes = list(set(a.genome)|set(b.genome)); rng.shuffle(genes)
            return _canonical(genes[:max_predicates], max_predicates)
        if u < 0.36:
            return _canonical(weighted_sample(rng.randint(1,min(max_predicates,len(universe)))), max_predicates)
        p = list(rng.choice(list(ranked[:max(8,len(ranked)//2)])).genome)
        m = rng.random()
        if m < 0.35 and len(p) < max_predicates:
            p.append(pick_gene(set(p)))
        elif m < 0.50 and len(p) > 1:
            p.pop(rng.randrange(len(p)))
        else:
            pos = rng.randrange(len(p))
            blocked = set(p)
            blocked.discard(p[pos])
            replacement = pick_gene(blocked)
            guard = 0
            while replacement == p[pos] and len(universe) > len(blocked) + 1 and guard < 8:
                replacement = pick_gene(blocked); guard += 1
            p[pos] = replacement
        return _canonical(p, max_predicates)

    checkpoints = []
    def checkpoint(it: int) -> None:
        rerank()
        c = max(population.values(), key=key)
        checkpoints.append({
            'iteration':it,'unique_genomes_evaluated':evaluated,'duplicate_proposals':duplicates,
            'champion_genome':list(c.genome),'champion_family':list(_family_signature(c.genome,predicates)),
            'train_score':c.score,'robust_core':c.robust_core,'novelty':c.novelty,
            'instability':c.instability,'negative_penalty':c.negative_penalty,
            'validation_observed':False,'holdout_observed':False,
        })

    checkpoint(0)
    for it in range(1, iterations+1):
        if it % cohort_refresh_every == 0:
            cohort_epoch += 1
            cohorts = _cohort_masks(corpus, train_headers, count=cohort_count, epoch=cohort_epoch, seed=split_seed)
            refreshes += 1
            rerank()
        ranked = sorted(population.values(), key=key, reverse=True)
        children: list[Score] = []
        attempts = 0
        while len(children) < offspring_per_iteration and attempts < offspring_per_iteration*100:
            attempts += 1
            g = propose(ranked)
            if not g or g in archive or not _consistent_genome(g, predicates):
                duplicates += 1
                continue
            refs = tuple(population) + tuple(x.genome for x in children)
            s = _score(corpus,g,train_mask=train_mask,cohorts=cohorts,thresholds=thresholds,refs=refs,neg=neg)
            archive.add(g); children.append(s); evaluated += 1
            sev = max(0.0, min(1.0, 0.90-s.robust_core, s.instability-0.2))
            if sev > 0:
                neg.observe(g, severity=sev)
        for s in children:
            population[s.genome] = s
        population = {s.genome:s for s in sorted(population.values(), key=key, reverse=True)[:population_size]}
        if it % checkpoint_every == 0 or it == iterations:
            checkpoint(it)

    top = sorted(population.values(), key=key, reverse=True)[:8]
    finals=[]; promoted=[]
    if final_evaluate:
        alpha = 0.05 / max(1, len(top)*len(thresholds)*2)
        for s in top:
            ladder=[]; qs=[]
            for k in thresholds:
                vr=corpus.metric(s.genome,'validation',int(k),alpha=alpha)
                hr=corpus.metric(s.genome,'holdout',int(k),alpha=alpha)
                q=bool(vr['marker_n']>=32 and hr['marker_n']>=32 and vr['event_n']>0 and hr['event_n']>0 and vr['gamma_lcb']>1.0 and hr['gamma_lcb']>1.0)
                qs.append(q); ladder.append({'k':int(k),'validation':vr,'holdout':hr,'qualified':q})
            longest=cur=0
            for q in qs:
                cur=cur+1 if q else 0; longest=max(longest,cur)
            row={'genome':list(s.genome),'family_signature':list(_family_signature(s.genome,predicates)),'train_score':s.score,'robust_core':s.robust_core,'ladder':ladder,'longest_qualified_run':longest,'promote':longest>=min(2,len(thresholds))}
            finals.append(row)
            if row['promote']: promoted.append(row)
    else:
        for s in top:
            finals.append({
                'genome':list(s.genome),
                'family_signature':list(_family_signature(s.genome,predicates)),
                'train_score':s.score,
                'robust_core':s.robust_core,
                'ladder':[],
                'longest_qualified_run':0,
                'promote':False,
                'post_training_evaluation_performed':False,
            })

    payload={
        'schema':SCHEMA,'status':'PASS_TRAINING_COMPLETE','iterations':iterations,
        'fixture':{'header_count':header_count,'nonces_per_header':nonces_per_header,'sample_count':len(samples),'split_counts':{k:len(v) for k,v in corpus.samples_by_split.items()},'thresholds':list(thresholds),'seed':seed,'split_seed':split_seed},
        'causal_filter':filter_receipt,
        'training_policy':{'selection_surface':'TRAIN_ONLY','validation_role':'FROZEN_UNTIL_TRAINING_COMPLETE','holdout_role':'FROZEN_UNTIL_TRAINING_COMPLETE','rolling_header_cohorts':True,'structural_predicates':True,'unlabelled_nonce_variation_filter':True,'requested_population_size':requested_population_size,'effective_population_size':population_size,'family_prior':prior_map,'family_prior_strength':prior_strength,'family_prior_probability':prior_probability},
        'search':{'unique_genomes_evaluated':evaluated,'duplicate_proposals':duplicates,'duplicate_proposal_fraction':duplicates/max(1,duplicates+evaluated),'cohort_refreshes':refreshes,'negative_memory':neg.snapshot()},
        'checkpoints':checkpoints,'final_top8':finals,'promoted_count':len(promoted),
        'predicate_catalog':[{'stage':p.stage,'round':p.round,'op':p.op,'left':p.left,'right':p.right,'bit':p.bit,'value':p.value,'family':p.family,'predicate_id':p.predicate_id} for p in predicates],
        'claims':{'cryptographic_advantage_assumed':False,'validation_used_during_training':False,'holdout_used_during_training':False,'post_training_evaluation_performed':bool(final_evaluate),'production_policy_modified':False},
    }
    payload['receipt_blake2b256']=hashlib.blake2b(json.dumps(payload,sort_keys=True,separators=(',',':')).encode(),digest_size=32).hexdigest()
    return payload
