from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Iterable, Sequence

from .marker_eval import TraceSample, split_name
from .marker_training import _wilson

FULL_COST = 128
SCHEMA = 'BITMYNER_STRUCTURAL_MARKER_V0_4'

_ALLOWED_SOURCES = {'a','b','c','d','e','f','g','h','t1','t2','w'}
_ALLOWED_OPS = {'direct','xor','xor_prev','carry'}


@dataclass(frozen=True, slots=True)
class StructuralPredicate:
    stage: int
    round: int
    op: str
    left: str
    bit: int
    value: int
    right: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError('stage must be 1 or 2')
        if not 0 <= self.round < 64:
            raise ValueError('round must be in [0,63]')
        if self.op not in _ALLOWED_OPS:
            raise ValueError('unsupported op')
        if self.left not in _ALLOWED_SOURCES:
            raise ValueError('unsupported left source')
        if self.op == 'direct':
            if self.right is not None:
                raise ValueError('direct predicate may not have right source')
        elif self.op == 'xor':
            if self.right not in _ALLOWED_SOURCES or self.right == self.left:
                raise ValueError('xor requires distinct supported right source')
        elif self.op == 'xor_prev':
            if self.round == 0:
                raise ValueError('xor_prev requires round >= 1')
            if self.right not in _ALLOWED_SOURCES:
                raise ValueError('xor_prev requires a supported previous-round source')
        elif self.op == 'carry':
            if self.right not in _ALLOWED_SOURCES or self.right == self.left:
                raise ValueError('carry requires distinct supported right source')
        if not 0 <= self.bit < 32:
            raise ValueError('bit must be in [0,31]')
        if self.value not in (0, 1):
            raise ValueError('value must be 0 or 1')

    def expression_value(self, sample: TraceSample) -> int:
        r = sample.trace.record(self.stage, self.round)
        x = r.source(self.left)
        if self.op == 'direct':
            return x
        if self.op == 'xor':
            return x ^ r.source(self.right or '')
        if self.op == 'xor_prev':
            prev = sample.trace.record(self.stage, self.round - 1)
            return x ^ prev.source(self.right or '')
        y = r.source(self.right or '')
        summed = (x + y) & 0xFFFFFFFF
        return ((x & y) | ((x | y) & (~summed & 0xFFFFFFFF))) & 0xFFFFFFFF

    def matches(self, sample: TraceSample) -> bool:
        return ((self.expression_value(sample) >> self.bit) & 1) == self.value

    @property
    def checkpoint_cost(self) -> int:
        return self.round + 1 if self.stage == 1 else 64 + self.round + 1

    @property
    def family(self) -> str:
        rhs = '' if self.right is None else f'_{self.right}'
        return f's{self.stage}:r{self.round}:{self.op}:{self.left}{rhs}'

    @property
    def predicate_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.blake2b(raw, digest_size=16).hexdigest()


def structural_population(
    *,
    stages: Iterable[int] = (1, 2),
    rounds: Iterable[int] = (0, 3, 7, 15, 31, 47, 63),
    direct_sources: Iterable[str] = ('a','e','t1','w'),
    xor_pairs: Iterable[tuple[str, str]] = (
        ('a','e'), ('a','t1'), ('e','t1'), ('t1','w'),
        ('t1','t2'), ('e','w'), ('a','w'),
    ),
    bits: Iterable[int] = (0, 7, 15, 23, 31),
) -> tuple[StructuralPredicate, ...]:
    out: list[StructuralPredicate] = []
    for stage in stages:
        for r in rounds:
            for source in direct_sources:
                for bit in bits:
                    for value in (0, 1):
                        out.append(StructuralPredicate(stage, r, 'direct', source, bit, value))
            for left, right in xor_pairs:
                for bit in bits:
                    for value in (0, 1):
                        out.append(StructuralPredicate(stage, r, 'xor', left, bit, value, right))
    return tuple(out)


def augmented_structural_population(
    *,
    stages: Iterable[int] = (1, 2),
    rounds: Iterable[int] = (0, 3, 7, 15, 31, 47, 63),
    direct_sources: Iterable[str] = ('a','e','t1','w'),
    xor_pairs: Iterable[tuple[str, str]] = (
        ('a','e'), ('a','t1'), ('e','t1'), ('t1','w'),
        ('t1','t2'), ('e','w'), ('a','w'),
    ),
    temporal_sources: Iterable[str] = ('a','e','t1','w'),
    carry_pairs: Iterable[tuple[str, str]] = (('t1','t2'),),
    bits: Iterable[int] = (0, 7, 15, 23, 31),
) -> tuple[StructuralPredicate, ...]:
    """Base v0.4 predicates plus cross-round XOR and addition-carry features.

    v0.13 keeps the original `structural_population()` unchanged for exact
    reproduction of older receipts. Temporal predicates compare the selected
    current-round source with a source from the immediately previous round. Carry
    predicates expose the carry-out mask of a two-word uint32 addition at the same
    checkpoint.
    """
    stages = tuple(int(x) for x in stages)
    rounds = tuple(int(x) for x in rounds)
    bits = tuple(int(x) for x in bits)
    out = list(structural_population(
        stages=stages,
        rounds=rounds,
        direct_sources=direct_sources,
        xor_pairs=xor_pairs,
        bits=bits,
    ))
    for stage in stages:
        for r in rounds:
            if r > 0:
                for source in temporal_sources:
                    for bit in bits:
                        for value in (0, 1):
                            out.append(StructuralPredicate(stage, r, 'xor_prev', source, bit, value, source))
            for left, right in carry_pairs:
                for bit in bits:
                    for value in (0, 1):
                        out.append(StructuralPredicate(stage, r, 'carry', left, bit, value, right))
    seen: set[str] = set()
    unique: list[StructuralPredicate] = []
    for predicate in out:
        if predicate.predicate_id in seen:
            continue
        seen.add(predicate.predicate_id)
        unique.append(predicate)
    return tuple(unique)


def nonce_variation_filter(
    samples: Sequence[TraceSample],
    predicates: Sequence[StructuralPredicate],
    *,
    split_seed: str,
    min_variable_header_fraction: float = 0.20,
    min_global_fraction: float = 0.01,
    max_global_fraction: float = 0.99,
) -> tuple[tuple[StructuralPredicate, ...], dict]:
    """Unlabelled TRAIN-only filter for header selectors and algebraic invariants.

    A predicate survives only if its underlying expression bit varies within a
    sufficient fraction of TRAIN headers. Low-hash/event labels are never read.
    """
    train = [s for s in samples if split_name(s.header_id, seed=split_seed) == 'train']
    by_header: dict[str, list[TraceSample]] = {}
    for s in train:
        by_header.setdefault(s.header_id, []).append(s)
    if not by_header:
        raise RuntimeError('no training headers')

    kept: list[StructuralPredicate] = []
    removed_constant = 0
    removed_imbalanced = 0
    diagnostics: list[dict] = []
    n = len(train)

    for p in predicates:
        ones = 0
        variable_headers = 0
        for rows in by_header.values():
            seen0 = seen1 = False
            for s in rows:
                bit = (p.expression_value(s) >> p.bit) & 1
                ones += bit
                if bit:
                    seen1 = True
                else:
                    seen0 = True
            if seen0 and seen1:
                variable_headers += 1
        frac_variable = variable_headers / len(by_header)
        bit_one_fraction = ones / n if n else 0.0
        if frac_variable < min_variable_header_fraction:
            removed_constant += 1
            continue
        pass_fraction = bit_one_fraction if p.value else 1.0 - bit_one_fraction
        if not min_global_fraction <= pass_fraction <= max_global_fraction:
            removed_imbalanced += 1
            continue
        kept.append(p)
        if len(diagnostics) < 16:
            diagnostics.append({
                'predicate_id': p.predicate_id,
                'family': p.family,
                'bit': p.bit,
                'value': p.value,
                'variable_header_fraction': frac_variable,
                'pass_fraction': pass_fraction,
            })

    return tuple(kept), {
        'input_predicates': len(predicates),
        'kept_predicates': len(kept),
        'removed_nonce_invariant_or_header_selector': removed_constant,
        'removed_imbalanced': removed_imbalanced,
        'train_headers': len(by_header),
        'train_samples': n,
        'min_variable_header_fraction': min_variable_header_fraction,
        'label_access': False,
        'diagnostic_examples': diagnostics,
    }


class StructuralCompiledCorpus:
    def __init__(self, samples: Sequence[TraceSample], predicates: Sequence[StructuralPredicate], thresholds: Sequence[int], split_seed: str):
        self.predicates = tuple(predicates)
        self.thresholds = tuple(int(k) for k in thresholds)
        self.split_seed = split_seed
        self.samples_by_split: dict[str, list[TraceSample]] = {'train': [], 'validation': [], 'holdout': []}
        for s in samples:
            self.samples_by_split[split_name(s.header_id, seed=split_seed)].append(s)
        self.pred_masks: dict[str, tuple[int, ...]] = {}
        self.event_masks: dict[str, dict[int, int]] = {}
        self.universe: dict[str, int] = {}
        self.header_masks: dict[str, dict[str, int]] = {}
        for split, rows in self.samples_by_split.items():
            self.universe[split] = (1 << len(rows)) - 1 if rows else 0
            hmap: dict[str, int] = {}
            for i, s in enumerate(rows):
                hmap[s.header_id] = hmap.get(s.header_id, 0) | (1 << i)
            self.header_masks[split] = hmap
            self.event_masks[split] = {
                k: sum((1 << i) for i, s in enumerate(rows) if s.trace.leading_zero_bits >= k)
                for k in self.thresholds
            }
            masks = [0] * len(self.predicates)
            for i, s in enumerate(rows):
                b = 1 << i
                cache: dict[tuple[int,int,str,str,str|None], int] = {}
                for pi, p in enumerate(self.predicates):
                    key = (p.stage, p.round, p.op, p.left, p.right)
                    value = cache.get(key)
                    if value is None:
                        value = p.expression_value(s)
                        cache[key] = value
                    if ((value >> p.bit) & 1) == p.value:
                        masks[pi] |= b
            self.pred_masks[split] = tuple(masks)

    def metric(self, genome: tuple[int, ...], split: str, k: int, *, sample_mask: int | None = None, alpha: float = 0.05) -> dict:
        universe = self.universe[split] if sample_mask is None else (self.universe[split] & sample_mask)
        n = universe.bit_count()
        if n == 0:
            return {'n':0,'marker_n':0,'event_n':0,'marker_event_n':0,'lift':0.0,'lift_lcb':0.0,'gamma':0.0,'gamma_lcb':0.0,'compute_fraction':1.0}
        active = universe
        total_cost = 0
        for idx in sorted(genome, key=lambda i: (self.predicates[i].checkpoint_cost, i)):
            p = self.predicates[idx]
            match = self.pred_masks[split][idx] & universe
            failed = active & ~match
            total_cost += failed.bit_count() * p.checkpoint_cost
            active &= match
        total_cost += active.bit_count() * FULL_COST
        marker_n = active.bit_count()
        event_mask = self.event_masks[split][int(k)] & universe
        event_n = event_mask.bit_count()
        marker_event_n = (active & event_mask).bit_count()
        p_event = event_n / n
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
