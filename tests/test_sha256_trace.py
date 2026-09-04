import random

from bitmyner.sha256_trace import sha256d_trace, sha256d_reference, surrogate_event


def genesis_header76():
    return bytes.fromhex(
        '01000000' +
        '00'*32 +
        '3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a' +
        '29ab5f49' +
        'ffff001d'
    )


def test_genesis_recovery_and_trace_shape():
    t = sha256d_trace(genesis_header76(), 2083236893)
    assert t.bitcoin_display_hash == '000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f'
    assert t.digest == sha256d_reference(genesis_header76(), 2083236893)
    assert len(t.stage1) == 64
    assert len(t.stage2) == 64
    assert t.stage1[0].stage == 1 and t.stage1[63].round == 63
    assert t.stage2[0].stage == 2 and t.stage2[63].round == 63


def test_100_random_scalar_crosschecks():
    rng = random.Random(616)
    for _ in range(100):
        h = bytes(rng.getrandbits(8) for _ in range(76))
        n = rng.getrandbits(32)
        assert sha256d_trace(h, n).digest == sha256d_reference(h, n)


def test_surrogate_event_definition():
    assert surrogate_event(bytes(32), 256)
    assert surrogate_event(bytes.fromhex('00' + 'ff'*31), 8)
    assert not surrogate_event(bytes.fromhex('01' + '00'*31), 8)
