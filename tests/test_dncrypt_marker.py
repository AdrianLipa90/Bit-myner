from bitmyner.dncrypt_marker import single_bit_genome
from bitmyner.sha256_trace import sha256d_trace


def test_genome_id_is_content_addressed_and_stable():
    a = single_bit_genome(1, 7, 'a', 3, 1)
    b = single_bit_genome(1, 7, 'a', 3, 1)
    c = single_bit_genome(1, 7, 'a', 3, 0)
    assert a.genome_id == b.genome_id
    assert a.genome_id != c.genome_id
    assert len(a.genome_id) == 64


def test_marker_matches_exact_selected_bit():
    h = bytes(range(76))
    t = sha256d_trace(h, 123)
    word = t.record(1, 7).a
    bit = 3
    g = single_bit_genome(1, 7, 'a', bit, (word >> bit) & 1)
    assert g.matches(t)
    opposite = single_bit_genome(1, 7, 'a', bit, 1 - ((word >> bit) & 1))
    assert not opposite.matches(t)
