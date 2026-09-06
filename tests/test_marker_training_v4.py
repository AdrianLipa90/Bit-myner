from bitmyner.marker_training_v4 import run_training_1000_v4
from bitmyner.marker_meta_v5 import summarize_multi_seed


def test_small_v4_training_is_deterministic_and_frozen():
    kw=dict(iterations=18,header_count=20,nonces_per_header=20,thresholds=(2,3),population_size=12,offspring_per_iteration=2,max_predicates=2,seed=919,checkpoint_every=9,cohort_count=3,cohort_refresh_every=3)
    a=run_training_1000_v4(**kw); b=run_training_1000_v4(**kw)
    assert a['receipt_blake2b256']==b['receipt_blake2b256']
    assert a['claims']['validation_used_during_training'] is False
    assert a['claims']['holdout_used_during_training'] is False
    assert a['causal_filter']['label_access'] is False


def test_meta_requires_recurrence_across_runs():
    def r(seed,fam,gen):
        return {'status':'PASS_TRAINING_COMPLETE','fixture':{'seed':seed},'promoted_count':0,'final_top8':[{'family_signature':[fam],'genome':[gen]}]}
    m=summarize_multi_seed([r(1,'A',1),r(2,'A',2),r(3,'B',3)])
    assert [x['family_signature'] for x in m['consensus_families']]==[['A']]
    assert m['consensus_exact_genomes']==[]
    assert m['consensus_component_families'][0]['family']=='A'
    assert m['external_evaluation_authorized_candidates']==[]


def test_v4_adapts_population_to_small_whitelist_without_leaking_labels():
    r=run_training_1000_v4(iterations=4,header_count=20,nonces_per_header=20,thresholds=(2,3),population_size=64,offspring_per_iteration=2,max_predicates=2,seed=321,checkpoint_every=2,cohort_count=3,cohort_refresh_every=2,family_whitelist=('s1:r7:direct:a','s1:r15:xor:t1_w'))
    assert r['training_policy']['requested_population_size']==64
    assert 2 <= r['training_policy']['effective_population_size'] <= 64
    assert r['causal_filter']['label_access'] is False


def test_v4_can_finish_train_only_without_opening_post_training_splits():
    r=run_training_1000_v4(
        iterations=6,header_count=20,nonces_per_header=20,thresholds=(2,3),
        population_size=10,offspring_per_iteration=2,max_predicates=2,seed=4242,
        checkpoint_every=3,cohort_count=3,cohort_refresh_every=3,final_evaluate=False,
    )
    assert r['claims']['post_training_evaluation_performed'] is False
    assert r['promoted_count']==0
    assert all(row['ladder']==[] for row in r['final_top8'])
    assert all(row['post_training_evaluation_performed'] is False for row in r['final_top8'])


def test_v4_nested_train_confirmation_keeps_validation_holdout_closed_for_failures():
    r = run_training_1000_v4(
        iterations=8,
        header_count=30,
        nonces_per_header=24,
        thresholds=(2,3),
        population_size=12,
        offspring_per_iteration=2,
        max_predicates=2,
        seed=991,
        split_seed='v4-nested-test',
        checkpoint_every=4,
        cohort_count=3,
        cohort_refresh_every=4,
        train_discovery_fraction=0.70,
    )
    assert r['training_policy']['selection_surface'] == 'TRAIN_DISCOVERY_ONLY'
    assert r['training_policy']['train_confirmation_headers'] > 0
    assert r['claims']['nested_train_confirmation_enabled'] is True
    for row in r['final_top8']:
        assert 'confirmation_pass' in row
        if not row['confirmation_pass']:
            assert row['ladder'] == []
            assert row['promote'] is False
