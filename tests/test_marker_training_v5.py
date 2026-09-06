from bitmyner.marker_training_v5 import run_meta_training_v5


def test_meta_training_uses_only_recurrent_family_whitelist():
    meta={'status':'PASS_META_COMPLETE','receipt_blake2b256':'abc','consensus_component_families':[{'family':'s1:r7:direct:a','seed_support':3},{'family':'s1:r15:xor:t1_w','seed_support':2},{'family':'s1:r3:direct:w','seed_support':1}]}
    r=run_meta_training_v5(meta,iterations=12,seed=123,header_count=20,nonces_per_header=20,thresholds=(2,3),minimum_seed_support=2)
    assert r['status']=='PASS_META_TRAINING_COMPLETE'
    assert r['whitelisted_families']==['s1:r15:xor:t1_w','s1:r7:direct:a']
    assert r['claims']['external_data_used'] is False
    assert set(r['training_run']['causal_filter']['family_whitelist'])==set(r['whitelisted_families'])

from bitmyner.marker_meta_v5 import summarize_multi_seed

def test_meta_exact_consensus_uses_predicate_ids_not_local_indices():
    def run(seed, pid):
        return {'status':'PASS_TRAINING_COMPLETE','fixture':{'seed':seed},'promoted_count':0,'predicate_catalog':[{'predicate_id':pid}], 'final_top8':[{'genome':[0],'family_signature':['fam']}]}
    r=summarize_multi_seed([run(1,'A'),run(2,'B'),run(3,'A')],minimum_seed_support=2)
    assert r['consensus_exact_genomes']==[{'predicate_ids':['A'],'seed_support':2,'run_indices':[0,2]}]
