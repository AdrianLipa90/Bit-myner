from bitmyner.gremlin_marker import deterministic_headers, collect_samples, run_experiment
from bitmyner.dncrypt_marker import raven_single_bit_population
from bitmyner.marker_eval import raven_rank, belzebub_promotion_verdict


def test_real_sha_negative_control_does_not_auto_promote_training_winner():
    headers = deterministic_headers(15, seed=616)
    samples = collect_samples(headers, nonces_per_header=256, event_k=7)
    candidates = raven_single_bit_population(
        stages=(1, 2), rounds=(0, 15, 31, 63), sources=('a', 'e'), bits=(0, 7, 15, 23, 31)
    )
    ranked = raven_rank(candidates, samples, top_k=4, min_marker_n=24)
    assert ranked
    assert all(not belzebub_promotion_verdict(ev, min_holdout_marker_n=24)['promote'] for _, ev in ranked)


def test_end_to_end_marker_graph_experiment_receipt():
    out = run_experiment(header_count=15, nonces_per_header=128, event_k=6, top_k=4, seed=616)
    assert out['status'] == 'PASS_EXPERIMENT_COMPLETE'
    assert out['fixture']['sample_count'] == 1920
    assert out['graph']['node_count'] > 0
    assert out['graph']['edge_count'] > 0
    assert out['promotion_semantics']['cryptographic_advantage_assumed'] is False
    assert out['promotion_semantics']['production_policy_changed'] is False
