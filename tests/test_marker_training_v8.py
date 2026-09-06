from bitmyner.marker_training_v8 import calibrate_family_prior_v8, run_multi_pilot_island_training_v8


def _meta():
    return {
        'status':'PASS_META_COMPLETE',
        'consensus_component_families':[
            {'family':'s1:r7:direct:a','seed_support':3},
            {'family':'s1:r15:xor:t1_w','seed_support':2},
        ],
    }


def test_v8_calibration_uses_multiple_train_only_pilots():
    r=calibrate_family_prior_v8(
        _meta(),strengths=(0.0,0.5),pilot_iterations=3,pilot_seeds=(11,22,33),
        header_count=20,nonces_per_header=20,thresholds=(2,3),
    )
    assert r['status']=='PASS_MULTI_PILOT_CALIBRATION_COMPLETE'
    assert r['chosen_strength'] in (0.0,0.5)
    assert len(r['candidates'])==2
    assert all(len(x['pilot_runs'])==3 for x in r['candidates'])
    assert all(not p['post_training_evaluation_performed'] for x in r['candidates'] for p in x['pilot_runs'])
    assert r['claims']['validation_opened_during_calibration'] is False


def test_v8_full_fixtures_are_disjoint_and_escape_islands_remain():
    r=run_multi_pilot_island_training_v8(
        _meta(),iterations=2,full_seeds=(101,202,303),header_count=20,nonces_per_header=20,
        thresholds=(2,3),strengths=(0.0,0.5),pilot_iterations=2,pilot_seeds=(404,505,606),
    )
    assert r['status']=='PASS_MULTI_PILOT_ISLAND_TRAINING_COMPLETE'
    assert r['pilot_full_header_overlap']==0
    assert r['policies'][1]['mode']=='UNRESTRICTED_ESCAPE'
    assert r['policies'][2]['mode']=='UNRESTRICTED_ESCAPE'
    assert r['claims']['prior_strength_selected_from_training_only_multi_seed_pilots'] is True
