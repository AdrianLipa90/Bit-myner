from bitmyner.marker_training_v7 import calibrate_family_prior_v7, run_adaptive_island_training_v7


def _meta():
    return {
        'status':'PASS_META_COMPLETE',
        'consensus_component_families':[
            {'family':'s1:r7:direct:a','seed_support':3},
            {'family':'s1:r15:xor:t1_w','seed_support':2},
            {'family':'s1:r3:direct:w','seed_support':1},
        ],
    }


def test_v7_calibration_is_train_only_and_deterministic():
    kw=dict(meta_summary=_meta(),strengths=(0.0,0.5),pilot_iterations=6,seed=77,header_count=20,nonces_per_header=20,thresholds=(2,3))
    a=calibrate_family_prior_v7(**kw)
    b=calibrate_family_prior_v7(**kw)
    assert a['receipt_blake2b256']==b['receipt_blake2b256']
    assert a['chosen_strength'] in (0.0,0.5)
    assert a['claims']['validation_opened_during_calibration'] is False
    assert all(x['post_training_evaluation_performed'] is False for x in a['candidates'])


def test_v7_full_run_keeps_escape_islands_and_fixture_separation():
    r=run_adaptive_island_training_v7(
        _meta(),iterations=3,seeds=(101,202,303),header_count=20,nonces_per_header=20,
        thresholds=(2,3),strengths=(0.0,0.5),pilot_iterations=3,calibration_seed=404,
    )
    assert r['status']=='PASS_ADAPTIVE_ISLAND_TRAINING_COMPLETE'
    assert r['calibration_full_header_overlap']==0
    assert r['policies'][0]['mode']=='CALIBRATED_SOFT_PRIOR'
    assert r['policies'][1]['mode']=='UNRESTRICTED_ESCAPE'
    assert r['policies'][2]['mode']=='UNRESTRICTED_ESCAPE'
    assert r['claims']['prior_strength_selected_without_validation_or_holdout'] is True
