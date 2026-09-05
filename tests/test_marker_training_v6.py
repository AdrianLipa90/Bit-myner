from bitmyner.marker_training_v6 import run_island_training_v6, summarize_island_runs_v6


def _meta():
    return {
        'status':'PASS_META_COMPLETE',
        'consensus_component_families':[
            {'family':'s1:r7:direct:a','seed_support':3},
            {'family':'s1:r15:xor:t1_w','seed_support':2},
        ],
    }


def test_v6_uses_soft_prior_and_keeps_escape_islands_unrestricted():
    r=run_island_training_v6(
        _meta(),iterations=4,seeds=(1,2,3),header_count=20,nonces_per_header=20,
        thresholds=(2,3),minimum_seed_support=2,focused_prior_strength=2.0,
    )
    assert r['status']=='PASS_ISLAND_TRAINING_COMPLETE'
    assert r['policies'][0]['family_whitelist'] is None
    assert r['policies'][0]['family_prior']
    assert r['policies'][0]['mode']=='SOFT_RECURRENT_FAMILY_PRIOR'
    assert r['policies'][1]['family_prior']=={}
    assert r['policies'][2]['family_prior']=={}
    assert r['claims']['hard_whitelist_applied_to_any_island'] is False
    assert r['claims']['focused_island_preserves_global_support'] is True
    assert r['claims']['escape_islands_preserve_global_exploration'] is True


def test_v6_can_resume_from_precomputed_island_runs():
    tiny=run_island_training_v6(
        _meta(),iterations=2,seeds=(11,12,13),header_count=30,nonces_per_header=20,
        thresholds=(2,3),minimum_seed_support=2,focused_prior_strength=1.0,
    )
    resumed=summarize_island_runs_v6(
        _meta(), tiny['runs'], tiny['policies'], iterations=2, minimum_seed_support=2
    )
    assert resumed['status']=='PASS_ISLAND_TRAINING_COMPLETE'
    assert resumed['total_iterations']==6
    assert len(resumed['runs'])==3
