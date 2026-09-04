from bitmyner.gremlin_marker import deterministic_headers, collect_samples, run_experiment
from bitmyner.dncrypt_marker import raven_single_bit_population
from bitmyner.marker_eval import raven_rank_train, evaluate_genome, belzebub_promotion_verdict


def test_real_sha_negative_control_does_not_auto_promote_training_winner():
    headers = deterministic_headers(15, seed=616)
    samples = collect_samples(headers, nonces_per_header=256, event_k=7)
    candidates = raven_single_bit_population(
        stages=(1, 2), rounds=(0, 15, 31, 63), sources=('a', 'e'), bits=(0, 7, 15, 23, 31)
    )
    ranked = raven_rank_train(candidates, samples, top_k=4, min_marker_n=24)
    assert ranked
    alpha = 0.05 / len(ranked)
    evaluations = [evaluate_genome(g, samples, alpha=alpha) for g, _ in ranked]
    assert all(not belzebub_promotion_verdict(ev, min_holdout_marker_n=24)['promote'] for ev in evaluations)


def test_end_to_end_marker_graph_experiment_receipt():
    out = run_experiment(header_count=15, nonces_per_header=128, event_k=6, top_k=4, seed=616)
    assert out['status'] == 'PASS_EXPERIMENT_COMPLETE'
    assert out['fixture']['sample_count'] == 1920
    assert out['graph']['node_count'] > 0
    assert out['graph']['edge_count'] > 0
    assert out['promotion_semantics']['cryptographic_advantage_assumed'] is False
    assert out['promotion_semantics']['production_policy_changed'] is False
    assert out['multiple_testing']['holdout_used_for_ranking'] is False
    assert out['multiple_testing']['correction'] == 'BONFERRONI_OVER_SELECTED_TOP_K'
