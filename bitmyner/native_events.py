from __future__ import annotations

import ctypes
import struct
from typing import TYPE_CHECKING

from .core import sha256d
from .mole_ring import bitcoin_leading_zero_bits

if TYPE_CHECKING:
    from .core import NativeBackend

UINT64_MAX = (1 << 64) - 1


def _event_function(backend: 'NativeBackend'):
    if backend.name != 'avx512x2':
        raise ValueError('native event capture is currently implemented only for avx512x2')
    try:
        fn = backend.lib.bm_avx512x2_scan_events
    except AttributeError as exc:
        raise RuntimeError('AVX512X2_EVENT_CAPTURE_SYMBOL_MISSING') from exc
    u8 = ctypes.c_ubyte
    fn.argtypes = [
        ctypes.POINTER(u8),       # header76
        ctypes.c_uint32,          # start
        ctypes.c_uint64,          # aligned count
        ctypes.c_int,             # threshold bits
        ctypes.c_int,             # threads
        ctypes.c_uint32,          # capacity
        ctypes.POINTER(ctypes.c_uint32),  # captured count
        ctypes.POINTER(ctypes.c_uint32),  # captured nonces
        ctypes.POINTER(u8),       # captured raw digests, 32 bytes each
    ]
    fn.restype = ctypes.c_uint64
    return fn


def scan_events(
    backend: 'NativeBackend',
    header76: bytes,
    start: int,
    count: int,
    *,
    threshold_bits: int,
    threads: int,
    capacity: int = 64,
) -> tuple[tuple[tuple[int, bytes], ...], int]:
    """Scan an AVX512x2 nonce interval and capture bounded low-hash events.

    The native kernel counts every event with Bitcoin-order leading-zero bits
    >= `threshold_bits` while retaining only the earliest `capacity` offsets.
    It emits the exact raw SHA256d digest for each retained event. Any final
    non-vector tail (at most 31 hashes) is evaluated exactly in Python.

    Returns `(events, total_hits)`, where `events` is an offset-ordered tuple of
    `(nonce, raw_digest)` and `total_hits` is independent of capture capacity.
    """
    if len(header76) != 76:
        raise ValueError('header76 must be exactly 76 bytes')
    if not 0 <= start <= 0xFFFFFFFF:
        raise ValueError('start must fit uint32')
    if count < 0:
        raise ValueError('count must be non-negative')
    if not 1 <= threshold_bits <= 32:
        raise ValueError('threshold_bits must be in [1,32]')
    if threads < 1:
        raise ValueError('threads must be positive')
    if capacity < 0:
        raise ValueError('capacity must be non-negative')

    fn = _event_function(backend)
    u8 = ctypes.c_ubyte
    header = (u8 * 76).from_buffer_copy(header76)
    aligned = count - (count % backend.lanes)

    array_n = max(1, capacity)
    out_count = ctypes.c_uint32(0)
    out_nonces = (ctypes.c_uint32 * array_n)()
    out_hashes = (u8 * (32 * array_n))()

    total_native = int(
        fn(
            header,
            start,
            aligned,
            threshold_bits,
            threads,
            capacity,
            ctypes.byref(out_count),
            out_nonces,
            out_hashes,
        )
    )
    if total_native == UINT64_MAX:
        raise RuntimeError('AVX512X2_EVENT_CAPTURE_FAILED')
    if out_count.value > capacity:
        raise RuntimeError('AVX512X2_EVENT_CAPTURE_COUNT_OVERFLOW')

    events: list[tuple[int, bytes]] = []
    for i in range(out_count.value):
        digest = bytes(out_hashes[i * 32 : (i + 1) * 32])
        events.append((int(out_nonces[i]), digest))

    total_hits = total_native
    for offset in range(aligned, count):
        nonce = (start + offset) & 0xFFFFFFFF
        digest = sha256d(header76 + struct.pack('<I', nonce))
        if bitcoin_leading_zero_bits(digest) >= threshold_bits:
            total_hits += 1
            if len(events) < capacity:
                events.append((nonce, digest))

    return tuple(events), int(total_hits)
