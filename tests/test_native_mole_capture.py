from __future__ import annotations

import struct

import pytest

from bitmyner.core import GENESIS_HEADER76, build_backend, read_cpu_flags, sha256d
from bitmyner.mole_ring import bitcoin_leading_zero_bits
from bitmyner.native_events import scan_events


def _backend_or_skip(tmp_path):
    if 'avx512f' not in read_cpu_flags():
        pytest.skip('AVX512F unavailable on CI host')
    try:
        return build_backend('avx512x2', tmp_path)
    except Exception as exc:
        pytest.skip(f'AVX512x2 backend unavailable: {exc!r}')


def test_native_mole_capture_matches_scalar_threshold_events(tmp_path):
    backend = _backend_or_skip(tmp_path)
    start = 0x12340000
    count = 4096
    threshold = 4
    events, total_hits = scan_events(
        backend,
        GENESIS_HEADER76,
        start,
        count,
        threshold_bits=threshold,
        threads=2,
        capacity=count,
    )
    expected = []
    for offset in range(count):
        nonce = (start + offset) & 0xFFFFFFFF
        digest = sha256d(GENESIS_HEADER76 + struct.pack('<I', nonce))
        if bitcoin_leading_zero_bits(digest) >= threshold:
            expected.append((nonce, digest))
    assert total_hits == len(expected)
    assert list(events) == expected


def test_native_mole_capture_is_bounded_but_counts_all_hits(tmp_path):
    backend = _backend_or_skip(tmp_path)
    events, total_hits = scan_events(
        backend,
        GENESIS_HEADER76,
        0,
        8192,
        threshold_bits=4,
        threads=2,
        capacity=7,
    )
    assert len(events) == 7
    assert total_hits >= len(events)
    assert [nonce for nonce, _ in events] == sorted(nonce for nonce, _ in events)
    assert all(bitmyner_lz >= 4 for _, digest in events for bitmyner_lz in [bitcoin_leading_zero_bits(digest)])


def test_native_mole_capture_rejects_invalid_threshold(tmp_path):
    backend = _backend_or_skip(tmp_path)
    with pytest.raises(ValueError):
        scan_events(backend, GENESIS_HEADER76, 0, 32, threshold_bits=0, threads=1, capacity=4)
    with pytest.raises(ValueError):
        scan_events(backend, GENESIS_HEADER76, 0, 32, threshold_bits=33, threads=1, capacity=4)
