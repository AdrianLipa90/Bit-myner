from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Iterable

from .sha256_trace import Sha256dTrace

SOURCES = {'a','b','c','d','e','f','g','h','t1','t2','w'}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


@dataclass(frozen=True, slots=True)
class FeaturePredicate:
    stage: int
    round: int
    source: str
    mask: int
    value: int

    def __post_init__(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError('stage must be 1 or 2')
        if not 0 <= self.round < 64:
            raise ValueError('round must be in [0, 63]')
        if self.source not in SOURCES:
            raise ValueError(f'unsupported source: {self.source}')
        if not 0 < self.mask <= 0xFFFFFFFF:
            raise ValueError('mask must be a non-zero uint32')
        if not 0 <= self.value <= 0xFFFFFFFF:
            raise ValueError('value must fit uint32')
        if self.value & ~self.mask:
            raise ValueError('value may only set bits selected by mask')

    def matches(self, trace: Sha256dTrace) -> bool:
        value = trace.record(self.stage, self.round).source(self.source)
        return (value & self.mask) == self.value


@dataclass(frozen=True, slots=True)
class MarkerGenome:
    predicates: tuple[FeaturePredicate, ...]
    mode: str = 'all'
    policy: str = 'rank'
    schema: str = 'BITMYNER_DNCRYPT_MARKER_GENOME_V0_1'

    def __post_init__(self) -> None:
        if not self.predicates:
            raise ValueError('marker genome requires at least one predicate')
        if self.mode not in {'all', 'any'}:
            raise ValueError('mode must be all or any')
        if self.policy not in {'rank', 'pass', 'reject'}:
            raise ValueError('policy must be rank, pass, or reject')

    @property
    def genome_id(self) -> str:
        return hashlib.blake2b(canonical_json(self.to_dict(include_id=False)), digest_size=32).hexdigest()

    def matches(self, trace: Sha256dTrace) -> bool:
        values = (p.matches(trace) for p in self.predicates)
        return all(values) if self.mode == 'all' else any(values)

    def to_dict(self, *, include_id: bool = True) -> dict:
        payload = {
            'schema': self.schema,
            'mode': self.mode,
            'policy': self.policy,
            'predicates': [asdict(p) for p in self.predicates],
        }
        if include_id:
            payload['genome_id'] = self.genome_id
        return payload


def single_bit_genome(stage: int, round_index: int, source: str, bit: int, value: int) -> MarkerGenome:
    if not 0 <= bit < 32:
        raise ValueError('bit must be in [0,31]')
    if value not in (0, 1):
        raise ValueError('value must be 0 or 1')
    mask = 1 << bit
    return MarkerGenome((FeaturePredicate(stage, round_index, source, mask, mask if value else 0),))


def raven_single_bit_population(
    *,
    stages: Iterable[int] = (1, 2),
    rounds: Iterable[int] = (0, 7, 15, 31, 47, 63),
    sources: Iterable[str] = ('a', 'e', 't1', 'w'),
    bits: Iterable[int] = (0, 3, 7, 11, 15, 19, 23, 27, 31),
) -> tuple[MarkerGenome, ...]:
    out: list[MarkerGenome] = []
    for stage in stages:
        for round_index in rounds:
            for source in sources:
                for bit in bits:
                    out.append(single_bit_genome(stage, round_index, source, bit, 0))
                    out.append(single_bit_genome(stage, round_index, source, bit, 1))
    return tuple(out)
