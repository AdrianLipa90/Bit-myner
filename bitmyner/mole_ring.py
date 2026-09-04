from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import hashlib
from typing import Iterable


def bitcoin_leading_zero_bits(digest: bytes) -> int:
    """Count leading zero bits in Bitcoin display/PoW integer order.

    SHA256d returns 32 raw bytes. Bitcoin interprets that digest as a
    little-endian integer for the PoW target comparison, so display order is
    digest[::-1].
    """
    if len(digest) != 32:
        raise ValueError('digest must be exactly 32 bytes')
    n = 0
    for byte in digest[::-1]:
        if byte == 0:
            n += 8
            continue
        n += 8 - byte.bit_length()
        break
    return n


@dataclass(frozen=True, slots=True)
class CapturedWork:
    header_id: str
    header76_hex: str
    nonce: int
    digest_hex: str
    leading_zero_bits: int

    def to_dict(self) -> dict:
        return asdict(self)


class MoleEventRing:
    """Bounded hot-path capture surface for low-hash events and controls.

    This object never performs SHA round tracing. It accepts an already
    computed digest and retains only compact tuples suitable for later MANTIS
    replay. Each threshold owns an independent bounded ring.
    """

    schema = 'BITMYNER_MOLE_EVENT_RING_V0_2'

    def __init__(
        self,
        *,
        thresholds: Iterable[int] = (8, 12, 16, 20),
        event_capacity_per_level: int = 64,
        control_capacity: int = 64,
        control_modulus: int = 257,
    ) -> None:
        levels = tuple(sorted(set(int(k) for k in thresholds)))
        if not levels or any(k < 1 or k > 256 for k in levels):
            raise ValueError('thresholds must be unique integers in [1,256]')
        if event_capacity_per_level < 1 or control_capacity < 0:
            raise ValueError('capacities must be positive (control may be zero)')
        if control_modulus < 1:
            raise ValueError('control_modulus must be positive')
        self.thresholds = levels
        self.event_capacity_per_level = int(event_capacity_per_level)
        self.control_capacity = int(control_capacity)
        self.control_modulus = int(control_modulus)
        self._events = {k: deque(maxlen=self.event_capacity_per_level) for k in levels}
        self._controls = deque(maxlen=self.control_capacity or 1)
        self.observed = 0

    @staticmethod
    def _header_id(header76: bytes) -> str:
        return hashlib.blake2b(header76, digest_size=16).hexdigest()

    def _control_selected(self, header_id: str, nonce: int) -> bool:
        if self.control_capacity == 0:
            return False
        token = hashlib.blake2b(
            f'{header_id}:{nonce}'.encode('ascii'), digest_size=8
        ).digest()
        return int.from_bytes(token, 'big') % self.control_modulus == 0

    def observe(self, header76: bytes, nonce: int, digest: bytes) -> None:
        if len(header76) != 76:
            raise ValueError('header76 must be exactly 76 bytes')
        if not 0 <= nonce <= 0xFFFFFFFF:
            raise ValueError('nonce must fit uint32')
        lz = bitcoin_leading_zero_bits(digest)
        hid = self._header_id(header76)
        row = CapturedWork(hid, header76.hex(), int(nonce), digest.hex(), lz)
        hit_any = False
        for k in self.thresholds:
            if lz >= k:
                self._events[k].append(row)
                hit_any = True
        if not hit_any and self._control_selected(hid, nonce):
            self._controls.append(row)
        self.observed += 1

    def replay_rows(self) -> tuple[CapturedWork, ...]:
        """Return a de-duplicated deterministic replay set for MANTIS."""
        by_key: dict[tuple[str, int], CapturedWork] = {}
        for k in self.thresholds:
            for row in self._events[k]:
                by_key[(row.header_id, row.nonce)] = row
        for row in self._controls:
            by_key[(row.header_id, row.nonce)] = row
        return tuple(by_key[key] for key in sorted(by_key))

    def snapshot(self) -> dict:
        return {
            'schema': self.schema,
            'thresholds': list(self.thresholds),
            'observed': self.observed,
            'hot_path_round_tracing': False,
            'event_capacity_per_level': self.event_capacity_per_level,
            'control_capacity': self.control_capacity,
            'control_modulus': self.control_modulus,
            'event_counts': {str(k): len(self._events[k]) for k in self.thresholds},
            'events': {str(k): [row.to_dict() for row in self._events[k]] for k in self.thresholds},
            'control_count': 0 if self.control_capacity == 0 else len(self._controls),
            'controls': [] if self.control_capacity == 0 else [row.to_dict() for row in self._controls],
            'replay_count': len(self.replay_rows()),
        }
