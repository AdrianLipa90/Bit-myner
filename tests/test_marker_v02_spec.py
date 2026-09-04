from fractions import Fraction

from bitmyner.dncrypt_marker import FeaturePredicate
from bitmyner.dncrypt_marker_v2 import composite_genome, raven_pair_population
from bitmyner.marker_cost import CostObservation, gamma_from_cost_observations, tau_adjusted_gamma
from bitmyner.mole_ring import MoleEventRing, bitcoin_leading_zero_bits
from bitmyner.gremlin_marker_v2 import run_experiment_v2


def test_mole_ring_is_bounded_and_multilevel_without_round_traces():
    ring = MoleEventRing(thresholds=(4, 8, 12), event_capacity_per_level=2, control_capacity=2, control_modulus=1)
    header = bytes(range(76))
    # Raw SHA256d digest is interpreted little-endian for Bitcoin PoW.
    d4 = b'\xff' * 31 + b'\x0f'   # 4 leading zero bits in display order
    d8 = b'\xff' * 31 + b'\x00'   # 8 leading zero bits
    d12 = b'\xff' * 30 + b'\xff' + b'\x00'  # exactly 8, control sanity
    assert bitcoin_leading_zero_bits(d4) == 4
    assert bitcoin_leading_zero_bits(d8) == 8
    assert bitcoin_leading_zero_bits(d12) == 8
    for nonce, digest in enumerate((d4, d8, d8, d8)):
        ring.observe(header, nonce, digest)
    snap = ring.snapshot()
    assert snap['event_counts']['4'] == 2
    assert snap['event_counts']['8'] == 2
    assert snap['event_counts']['12'] == 0
    assert snap['control_count'] <= 2
    assert all('trace' not in row for rows in snap['events'].values() for row in rows)


def test_composite_genome_is_canonical_and_pair_population_unique():
    p1 = FeaturePredicate(1, 7, 'a', 1 << 3, 0)
    p2 = FeaturePredicate(2, 15, 'e', 1 << 9, 1 << 9)
    g12 = composite_genome((p1, p2), mode='all')
    g21 = composite_genome((p2, p1), mode='all')
    assert g12.genome_id == g21.genome_id
    pairs = raven_pair_population((g12, composite_genome((p1,), mode='all'), composite_genome((p2,), mode='all')), max_pairs=10)
    assert len({g.genome_id for g in pairs}) == len(pairs)
    assert all(len(g.predicates) == 2 for g in pairs)


def test_cost_gamma_and_tau_adjustment_are_exact():
    # 100 baseline samples, 10 events. Marker keeps 20 samples and 5 events.
    # Total marker-policy round cost = 3200 versus 12800 baseline.
    rows = [CostObservation(marker=(i < 20), event=(i < 5 or 20 <= i < 25), cost_units=32) for i in range(100)]
    report = gamma_from_cost_observations(rows, full_cost_units=128)
    assert report.event_n == 10
    assert report.marker_n == 20
    assert report.marker_event_n == 5
    assert report.gamma == Fraction(2, 1)
    assert tau_adjusted_gamma(report.gamma, Fraction(9, 10), Fraction(3, 4)) == Fraction(12, 5)


def test_v02_end_to_end_is_falsification_first_and_holdout_blind_for_raven():
    out = run_experiment_v2(
        header_count=15,
        nonces_per_header=128,
        thresholds=(4, 6, 8),
        seed_top_k=8,
        pair_limit=24,
        final_top_k=4,
        seed=616,
    )
    assert out['status'] == 'PASS_EXPERIMENT_COMPLETE'
    assert out['thresholds'] == [4, 6, 8]
    assert out['candidate_generation']['composite_candidates'] > 0
    assert out['selection']['holdout_used_for_ranking'] is False
    assert out['promotion_semantics']['production_policy_changed'] is False
    assert out['promotion_semantics']['cryptographic_advantage_assumed'] is False
    assert out['mole_ring']['hot_path_round_tracing'] is False
    assert out['tau_cost']['runtime_tau_measurement_available'] is False
