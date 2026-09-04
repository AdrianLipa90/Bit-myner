from __future__ import annotations

import hashlib
import struct

import pytest

from bitmyner.core import (
    GENESIS_HASH,
    GENESIS_HEADER76,
    GENESIS_NONCE,
    available_backend_names,
    backend_selftest,
    build_backend,
    sha256d,
)


def _build_or_skip(name: str, tmp_path):
    try:
        return build_backend(name, tmp_path)
    except Exception as exc:  # platform/toolchain capability boundary
        pytest.skip(f"backend {name} unavailable: {exc!r}")


def test_all_detected_backends_recover_genesis(tmp_path):
    for name in available_backend_names():
        backend = _build_or_skip(name, tmp_path)
        receipt = backend_selftest(backend)
        assert receipt["status"] == "PASS"
        assert receipt["nonce"] == GENESIS_NONCE
        assert receipt["hash"] == GENESIS_HASH


def test_all_detected_backends_match_scalar_sha256d(tmp_path):
    header76 = bytes.fromhex(
        "02000000"
        + "11" * 32
        + "22" * 32
        + "78563412"
        + "ffff001d"
    )
    start = 0x13579BDF
    target = (1 << 256) - 1
    expected = sha256d(header76 + struct.pack("<I", start))

    for name in available_backend_names():
        backend = _build_or_skip(name, tmp_path)
        result = backend.scan_target(
            header76,
            start,
            backend.lanes,
            target,
            1,
        )
        assert result is not None
        nonce, digest = result
        assert nonce == start
        assert digest == expected
        assert digest == hashlib.sha256(
            hashlib.sha256(header76 + struct.pack("<I", start)).digest()
        ).digest()
