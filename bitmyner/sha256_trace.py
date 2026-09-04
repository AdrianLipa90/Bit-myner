from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

MASK32 = 0xFFFFFFFF
K = (
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2,
)
IV = (0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19)


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK32


def _compress(state: tuple[int, ...], block: bytes, *, stage: int | None = None) -> tuple[tuple[int, ...], tuple['RoundRecord', ...]]:
    if len(block) != 64:
        raise ValueError('SHA-256 block must be exactly 64 bytes')
    w = [int.from_bytes(block[i:i+4], 'big') for i in range(0, 64, 4)]
    for i in range(16, 64):
        x = w[i-15]
        y = w[i-2]
        s0 = _rotr(x, 7) ^ _rotr(x, 18) ^ (x >> 3)
        s1 = _rotr(y, 17) ^ _rotr(y, 19) ^ (y >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & MASK32)

    a,b,c,d,e,f,g,h = state
    records: list[RoundRecord] = []
    for i in range(64):
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        t1 = (h + S1 + ch + K[i] + w[i]) & MASK32
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + maj) & MASK32
        h,g,f,e,d,c,b,a = g,f,e,(d+t1)&MASK32,c,b,a,(t1+t2)&MASK32
        if stage is not None:
            records.append(RoundRecord(stage, i, a,b,c,d,e,f,g,h,t1,t2,w[i]))

    out = tuple((s + v) & MASK32 for s, v in zip(state, (a,b,c,d,e,f,g,h)))
    return out, tuple(records)


@dataclass(frozen=True, slots=True)
class RoundRecord:
    stage: int
    round: int
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int
    g: int
    h: int
    t1: int
    t2: int
    w: int

    def source(self, name: str) -> int:
        if name not in {'a','b','c','d','e','f','g','h','t1','t2','w'}:
            raise KeyError(name)
        return int(getattr(self, name))


@dataclass(frozen=True, slots=True)
class Sha256dTrace:
    header76: bytes
    nonce: int
    first_digest: bytes
    digest: bytes
    stage1: tuple[RoundRecord, ...]
    stage2: tuple[RoundRecord, ...]

    def record(self, stage: int, round_index: int) -> RoundRecord:
        if stage == 1:
            return self.stage1[round_index]
        if stage == 2:
            return self.stage2[round_index]
        raise ValueError('stage must be 1 or 2')

    @property
    def bitcoin_display_hash(self) -> str:
        return self.digest[::-1].hex()

    @property
    def leading_zero_bits(self) -> int:
        n = 0
        for byte in self.digest[::-1]:
            if byte == 0:
                n += 8
                continue
            n += 8 - byte.bit_length()
            break
        return n


def sha256d_trace(header76: bytes, nonce: int) -> Sha256dTrace:
    if len(header76) != 76:
        raise ValueError('header76 must be exactly 76 bytes')
    if not 0 <= nonce <= 0xFFFFFFFF:
        raise ValueError('nonce must fit uint32')

    mid, _ = _compress(IV, header76[:64])

    block1 = bytearray(64)
    block1[:12] = header76[64:76]
    block1[12:16] = nonce.to_bytes(4, 'little')
    block1[16] = 0x80
    block1[62:64] = (80 * 8).to_bytes(2, 'big')
    first_state, stage1 = _compress(mid, bytes(block1), stage=1)
    first_digest = b''.join(x.to_bytes(4, 'big') for x in first_state)

    block2 = bytearray(64)
    block2[:32] = first_digest
    block2[32] = 0x80
    block2[62:64] = (32 * 8).to_bytes(2, 'big')
    second_state, stage2 = _compress(IV, bytes(block2), stage=2)
    digest = b''.join(x.to_bytes(4, 'big') for x in second_state)
    return Sha256dTrace(header76, nonce, first_digest, digest, stage1, stage2)


def sha256d_reference(header76: bytes, nonce: int) -> bytes:
    header80 = header76 + nonce.to_bytes(4, 'little')
    return hashlib.sha256(hashlib.sha256(header80).digest()).digest()


def surrogate_event(digest: bytes, k: int) -> bool:
    if not 0 <= k <= 256:
        raise ValueError('k must be in [0, 256]')
    if len(digest) != 32:
        raise ValueError('digest must be 32 bytes')
    return int.from_bytes(digest, 'little') < (1 << (256-k)) if k else True
