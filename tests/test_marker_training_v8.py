from bitmyner.marker_training_v8 import (
    checkpoint_duplicate_trajectory_v8,
    predict_duplicate_fraction_v8,
    select_prior_strength_v8,
    summarize_horizon_adaptive_islands_v8,
)


def _run(points):
    checkpoints=[]
    for it, unique, dup in points:
        checkpoints.append({'iteration':it,'unique_genomes_evaluated':unique,'duplicate_proposals':dup})
    return {'checkpoints':checkpoints}


def test_horizon_prediction_extrapolates_growth_and_never_below_observed():
    run=_run([(0,80,10),(20,200,40),(40,320,80)])
    traj=checkpoint_duplicate_trajectory_v8(run)
    p=predict_duplicate_fraction_v8(traj,horizon_iterations=100)
    assert p['observed_final_fraction'] == 0.2
    assert p['predicted_horizon_fraction'] >= p['observed_final_fraction']
    assert p['trend_slope_per_iteration'] > 0


def test_horizon_selector_rejects_pilot_that_looks_ok_now_but_projects_over_cap():
    pilots=[
        {'strength':0.0,'duplicate_fraction':0.05,'predicted_horizon_fraction':0.08,'robust_core':0.65},
        {'strength':0.35,'duplicate_fraction':0.08,'predicted_horizon_fraction':0.18,'robust_core':0.68},
        {'strength':1.0,'duplicate_fraction':0.17,'predicted_horizon_fraction':0.28,'robust_core':0.75},
    ]
    s=select_prior_strength_v8(pilots,duplicate_cap=0.20,minimum_robust_ratio=0.90)
    assert s['selected_strength'] == 0.35
    assert 1.0 in s['rejected_strengths']
    assert s['selection_uses_horizon_prediction'] is True


def test_v8_summary_keeps_crypto_claim_false():
    meta={'status':'PASS_META_COMPLETE','receipt_blake2b256':'abc'}
    runs=[
        {'search':{'duplicate_proposal_fraction':0.1},'promoted_count':0},
        {'search':{'duplicate_proposal_fraction':0.11},'promoted_count':0},
        {'search':{'duplicate_proposal_fraction':0.09},'promoted_count':0},
    ]
    policies=[
        {'mode':'HORIZON_ADAPTIVE_FOCUSED'},
        {'mode':'UNRESTRICTED_ESCAPE'},
        {'mode':'UNRESTRICTED_ESCAPE'},
    ]
    sel={'selected_strength':0.35,'validation_or_holdout_used_for_selection':False}
    r=summarize_horizon_adaptive_islands_v8(meta,runs,policies,pilot_selection=sel,iterations=100)
    assert r['status']=='PASS_HORIZON_ADAPTIVE_ISLAND_TRAINING_COMPLETE'
    assert r['claims']['cryptographic_advantage_assumed'] is False
    assert r['claims']['validation_or_holdout_used_to_select_prior_strength'] is False
