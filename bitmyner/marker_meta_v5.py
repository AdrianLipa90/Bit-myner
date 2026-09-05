from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Sequence

SCHEMA='BITMYNER_MARKER_META_V0_5_MULTI_SEED'


def summarize_multi_seed(runs: Sequence[dict], *, minimum_seed_support: int = 2) -> dict:
    if minimum_seed_support < 2:
        raise ValueError('minimum_seed_support must be >=2')
    family_runs: dict[tuple[str,...], set[int]] = {}
    exact_runs: dict[tuple[int,...], set[int]] = {}
    component_family_runs: dict[str, set[int]] = {}
    rows=[]
    for ri, run in enumerate(runs):
        if run.get('status') != 'PASS_TRAINING_COMPLETE':
            raise ValueError('all runs must be complete')
        top=run.get('final_top8',[])
        seen_family=set(); seen_exact=set(); seen_components=set()
        for row in top:
            fam=tuple(row.get('family_signature',[])); gen=tuple(row.get('genome',[]))
            if fam and fam not in seen_family:
                family_runs.setdefault(fam,set()).add(ri); seen_family.add(fam)
            for component in fam:
                if component not in seen_components:
                    component_family_runs.setdefault(component,set()).add(ri); seen_components.add(component)
            catalog=run.get('predicate_catalog',[])
            exact_signature=tuple(sorted(str(catalog[idx].get('predicate_id')) for idx in gen if 0 <= idx < len(catalog))) if catalog else tuple()
            if exact_signature and len(exact_signature)==len(gen) and exact_signature not in seen_exact:
                exact_runs.setdefault(exact_signature,set()).add(ri); seen_exact.add(exact_signature)
        rows.append({'run_index':ri,'seed':run.get('fixture',{}).get('seed'),'top8_count':len(top),'promoted_count':run.get('promoted_count',0)})
    fam_cons=[{'family_signature':list(k),'seed_support':len(v),'run_indices':sorted(v)} for k,v in family_runs.items() if len(v)>=minimum_seed_support]
    fam_cons.sort(key=lambda x:(x['seed_support'],x['family_signature']),reverse=True)
    exact_cons=[{'predicate_ids':list(k),'seed_support':len(v),'run_indices':sorted(v)} for k,v in exact_runs.items() if len(v)>=minimum_seed_support]
    exact_cons.sort(key=lambda x:(x['seed_support'],x['predicate_ids']),reverse=True)
    component_cons=[{'family':k,'seed_support':len(v),'run_indices':sorted(v)} for k,v in component_family_runs.items() if len(v)>=minimum_seed_support]
    component_cons.sort(key=lambda x:(x['seed_support'],x['family']),reverse=True)
    payload={'schema':SCHEMA,'status':'PASS_META_COMPLETE','run_count':len(runs),'minimum_seed_support':minimum_seed_support,'runs':rows,'consensus_families':fam_cons,'consensus_component_families':component_cons,'consensus_exact_genomes':exact_cons,'external_evaluation_authorized_candidates':exact_cons[:8],'claims':{'consensus_is_not_cryptographic_proof':True,'no_candidate_promoted_by_meta_alone':True}}
    payload['receipt_blake2b256']=hashlib.blake2b(json.dumps(payload,sort_keys=True,separators=(',',':')).encode(),digest_size=32).hexdigest()
    return payload
