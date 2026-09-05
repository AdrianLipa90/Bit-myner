from bitmyner.marker_bridge_v9 import run_bridge_recombination_v9


def _meta():
    return {
        'status': 'PASS_META_COMPLETE',
        'consensus_component_families': [
            {'family': 's1:r7:direct:a', 'seed_support': 3},
            {'family': 's1:r15:xor:t1_w', 'seed_support': 2},
            {'family': 's2:r0:xor:a_w', 'seed_support': 2},
        ],
    }


def test_bridge_v9_uses_train_only_ranking_and_disjoint_fixture():
    r = run_bridge_recombination_v9(
        _meta(),
        bridge_seed=909,
        excluded_seeds=(101, 202, 303),
        excluded_header_count=24,
        header_count=24,
        nonces_per_header=24,
        thresholds=(2, 3),
        top_singles_per_family=2,
        pair_limit=12,
        final_top_k=4,
        minimum_seed_support=2,
    )
    assert r['status'] == 'PASS_BRIDGE_RECOMBINATION_COMPLETE'
    assert r['fixture']['header_overlap_with_excluded_fixtures'] == 0
    assert r['policy']['ranking_surface'] == 'BRIDGE_TRAIN_ONLY'
    assert r['policy']['validation_used_for_ranking'] is False
    assert r['policy']['holdout_used_for_ranking'] is False
    assert r['candidate_generation']['recurrent_family_count'] >= 2
    assert r['candidate_generation']['train_selected_single_count'] > 0
    assert r['candidate_generation']['pair_candidate_count'] >= 0
    assert isinstance(r['external_evaluation_authorized_candidates'], list)


def test_bridge_v9_is_deterministic():
    kwargs = dict(
        bridge_seed=919,
        excluded_seeds=(111, 222, 333),
        excluded_header_count=20,
        header_count=20,
        nonces_per_header=20,
        thresholds=(2, 3),
        top_singles_per_family=2,
        pair_limit=8,
        final_top_k=3,
        minimum_seed_support=2,
    )
    a = run_bridge_recombination_v9(_meta(), **kwargs)
    b = run_bridge_recombination_v9(_meta(), **kwargs)
    assert a['receipt_blake2b256'] == b['receipt_blake2b256']
    assert a['selected_top_k'] == b['selected_top_k']
