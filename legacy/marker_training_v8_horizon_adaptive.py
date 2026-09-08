from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .marker_meta_v5 import summarize_multi_seed
from .marker_training_v4 import run_training_1000_v4

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_8_HORIZON_ADAPTIVE_PRIOR'


def checkpoint_duplicate_trajectory_v8(run: dict) -> list[dict]:
    rows=[]
    last_iteration=-1
    for cp in run.get('checkpoints', []):
        it=int(cp['iteration'])
        if it < last_iteration:
            raise ValueError('checkpoint iterations must be monotone')
        last_iteration=it
        unique=int(cp['unique_genomes_evaluated'])
        duplicate=int(cp['duplicate_proposals'])
        denom=unique+duplicate
        frac=(duplicate/denom) if denom else 0.0
        rows.append({
            'iteration':it,
            'unique_genomes_evaluated':unique,
            'duplicate_proposals':duplicate,
            'duplicate_fraction':frac,
        })
    if not rows:
        raise ValueError('run has no checkpoints')
    return rows


def predict_duplicate_fraction_v8(
    trajectory: Sequence[dict],
    *,
    horizon_iterations: int,
    safety_margin: float = 0.0,
) -> dict:
    if not trajectory:
        raise ValueError('trajectory is required')
    if horizon_iterations < int(trajectory[-1]['iteration']):
        raise ValueError('horizon must be >= last checkpoint iteration')
    if safety_margin < 0.0:
        raise ValueError('safety_margin must be non-negative')

    xs=[float(r['iteration']) for r in trajectory]
    ys=[float(r['duplicate_fraction']) for r in trajectory]
    last=ys[-1]
    if len(xs) < 2 or max(xs) == min(xs):
        slope=0.0
        intercept=last
    else:
        mx=sum(xs)/len(xs)
        my=sum(ys)/len(ys)
        denom=sum((x-mx)**2 for x in xs)
        slope=(sum((x-mx)*(y-my) for x,y in zip(xs,ys))/denom) if denom else 0.0
        slope=max(0.0, slope)
        intercept=my-slope*mx
    projected=intercept+slope*float(horizon_iterations)
    predicted=min(1.0,max(last, projected)+float(safety_margin))
    return {
        'observed_final_fraction':last,
        'trend_slope_per_iteration':slope,
        'ols_intercept':intercept,
        'horizon_iterations':int(horizon_iterations),
        'safety_margin':float(safety_margin),
        'predicted_horizon_fraction':predicted,
        'method':'OLS_NONNEGATIVE_SLOPE_MAX_OBSERVED_PLUS_MARGIN',
        'trajectory':[dict(r) for r in trajectory],
    }


def select_prior_strength_v8(
    pilots: Sequence[dict],
    *,
    duplicate_cap: float = 0.20,
    minimum_robust_ratio: float = 0.90,
) -> dict:
    if not pilots:
        raise ValueError('at least one pilot is required')
    if not 0.0 <= duplicate_cap < 1.0:
        raise ValueError('duplicate_cap must be in [0,1)')
    if not 0.0 <= minimum_robust_ratio <= 1.0:
        raise ValueError('minimum_robust_ratio must be in [0,1]')
    normalized=sorted(({
        'strength':float(p['strength']),
        'duplicate_fraction':float(p['duplicate_fraction']),
        'predicted_horizon_fraction':float(p['predicted_horizon_fraction']),
        'robust_core':float(p['robust_core']),
        'receipt_blake2b256':p.get('receipt_blake2b256'),
        'prediction':p.get('prediction'),
    } for p in pilots),key=lambda x:x['strength'])
    baseline_rows=[p for p in normalized if p['strength']==0.0]
    if not baseline_rows:
        raise ValueError('pilot set must contain strength=0 baseline')
    baseline=baseline_rows[0]
    robust_floor=baseline['robust_core']*minimum_robust_ratio
    eligible=[p for p in normalized if p['strength']>0.0 and p['predicted_horizon_fraction']<=duplicate_cap and p['robust_core']>=robust_floor]
    if eligible:
        selected=max(eligible,key=lambda p:(p['robust_core'],-p['predicted_horizon_fraction'],-p['strength']))
        fallback=False
    else:
        selected=baseline
        fallback=True
    rejected=[p['strength'] for p in normalized if p['strength']>0.0 and p not in eligible]
    return {
        'selected_strength':selected['strength'],
        'selected_duplicate_fraction':selected['duplicate_fraction'],
        'selected_predicted_horizon_fraction':selected['predicted_horizon_fraction'],
        'selected_robust_core':selected['robust_core'],
        'baseline_robust_core':baseline['robust_core'],
        'baseline_duplicate_fraction':baseline['duplicate_fraction'],
        'baseline_predicted_horizon_fraction':baseline['predicted_horizon_fraction'],
        'duplicate_cap':float(duplicate_cap),
        'minimum_robust_ratio':float(minimum_robust_ratio),
        'robust_floor':robust_floor,
        'rejected_strengths':rejected,
        'fallback_to_unrestricted':fallback,
        'selection_uses_horizon_prediction':True,
        'selected_from_training_only':True,
        'validation_or_holdout_used_for_selection':False,
        'pilots':normalized,
    }


def summarize_horizon_adaptive_islands_v8(
    meta_summary: dict,
    runs: Sequence[dict],
    policies: Sequence[dict],
    *,
    pilot_selection: dict,
    iterations: int,
) -> dict:
    if meta_summary.get('status')!='PASS_META_COMPLETE':
        raise ValueError('meta summary must be complete')
    if len(runs)<3 or len(runs)!=len(policies):
        raise ValueError('three or more runs with matching policies required')
    duplicates=[float(r['search']['duplicate_proposal_fraction']) for r in runs]
    focused=[d for d,p in zip(duplicates,policies) if p.get('mode')=='HORIZON_ADAPTIVE_FOCUSED']
    escape=[d for d,p in zip(duplicates,policies) if p.get('mode')=='UNRESTRICTED_ESCAPE']
    payload={
        'schema':SCHEMA,
        'status':'PASS_HORIZON_ADAPTIVE_ISLAND_TRAINING_COMPLETE',
        'iterations_per_island':int(iterations),
        'total_iterations':int(iterations)*len(runs),
        'pilot_selection':dict(pilot_selection),
        'prior_source_receipt':meta_summary.get('receipt_blake2b256'),
        'policies':list(policies),
        'runs':list(runs),
        'mean_duplicate_fraction':sum(duplicates)/len(duplicates),
        'focused_duplicate_fraction':(sum(focused)/len(focused)) if focused else None,
        'escape_mean_duplicate_fraction':(sum(escape)/len(escape)) if escape else None,
        'promoted_total':sum(int(r.get('promoted_count',0)) for r in runs),
        'claims':{
            'pilot_selection_uses_training_only':True,
            'validation_or_holdout_used_to_select_prior_strength':False,
            'pilot_strengths_share_common_fixture_split':True,
            'horizon_duplicate_prediction_used':True,
            'escape_islands_preserve_global_exploration':True,
            'hard_whitelist_applied':False,
            'production_policy_modified':False,
            'cryptographic_advantage_assumed':False,
        },
    }
    payload['receipt_blake2b256']=hashlib.blake2b(json.dumps(payload,sort_keys=True,separators=(',',':')).encode(),digest_size=32).hexdigest()
    return payload


def run_horizon_adaptive_island_training_v8(
    meta_summary: dict,
    *,
    iterations: int = 1000,
    pilot_iterations: int = 100,
    pilot_strengths: Sequence[float] = (0.0, 0.2, 0.25, 0.35, 0.5, 1.0, 2.0),
    duplicate_cap: float = 0.20,
    minimum_robust_ratio: float = 0.90,
    prediction_safety_margin: float = 0.0,
    seeds: Sequence[int] = (1717, 1818, 1919),
    pilot_seed: int = 1601,
    header_count: int = 100,
    nonces_per_header: int = 128,
    pilot_header_count: int = 60,
    pilot_nonces_per_header: int = 64,
    thresholds: Sequence[int] = (4,6,8),
    pilot_thresholds: Sequence[int] = (3,4,5),
    minimum_seed_support: int = 2,
) -> dict:
    if meta_summary.get('status')!='PASS_META_COMPLETE':
        raise ValueError('meta summary must be complete')
    if len(seeds)<3:
        raise ValueError('at least three island seeds required')
    recurrent=[r for r in meta_summary.get('consensus_component_families',[]) if int(r.get('seed_support',0))>=int(minimum_seed_support)]
    family_prior={str(r['family']):float(r['seed_support']) for r in recurrent}
    if not family_prior:
        raise RuntimeError('no recurrent families available for adaptive prior')

    common_pilot_split='bitmyner-marker-v0.8-pilot-common'
    pilots=[]
    for strength in pilot_strengths:
        run=run_training_1000_v4(
            iterations=int(pilot_iterations),header_count=int(pilot_header_count),nonces_per_header=int(pilot_nonces_per_header),
            thresholds=pilot_thresholds,population_size=48,offspring_per_iteration=6,max_predicates=3,seed=int(pilot_seed),
            split_seed=common_pilot_split,checkpoint_every=max(1,int(pilot_iterations)//5),cohort_count=4,
            cohort_refresh_every=max(5,int(pilot_iterations)//10),family_whitelist=None,
            family_prior=(family_prior if float(strength)>0 else None),family_prior_strength=float(strength),final_evaluate=False,
        )
        trajectory=checkpoint_duplicate_trajectory_v8(run)
        pred=predict_duplicate_fraction_v8(trajectory,horizon_iterations=int(iterations),safety_margin=prediction_safety_margin)
        last=run['checkpoints'][-1]
        pilots.append({
            'strength':float(strength),'duplicate_fraction':float(run['search']['duplicate_proposal_fraction']),
            'predicted_horizon_fraction':float(pred['predicted_horizon_fraction']),'robust_core':float(last['robust_core']),
            'receipt_blake2b256':run['receipt_blake2b256'],'unique_genomes_evaluated':int(run['search']['unique_genomes_evaluated']),
            'prediction':pred,
        })
    selection=select_prior_strength_v8(pilots,duplicate_cap=duplicate_cap,minimum_robust_ratio=minimum_robust_ratio)
    selected_strength=float(selection['selected_strength'])

    runs=[]; policies=[]
    for i,seed in enumerate(seeds):
        focused=i==0 and selected_strength>0.0
        strength=selected_strength if focused else 0.0
        run=run_training_1000_v4(
            iterations=int(iterations),header_count=int(header_count),nonces_per_header=int(nonces_per_header),thresholds=thresholds,
            population_size=64,offspring_per_iteration=8,max_predicates=3,seed=int(seed),split_seed=f'bitmyner-marker-v0.8-island-{i}',
            checkpoint_every=max(1,int(iterations)//10),cohort_count=4,cohort_refresh_every=max(10,min(25,int(iterations)//4)),
            family_whitelist=None,family_prior=(family_prior if focused else None),family_prior_strength=strength,final_evaluate=True,
        )
        runs.append(run)
        policies.append({'island':i,'seed':int(seed),'mode':'HORIZON_ADAPTIVE_FOCUSED' if focused else 'UNRESTRICTED_ESCAPE','strength':strength,'family_prior':family_prior if focused else {}})

    result=summarize_horizon_adaptive_islands_v8(meta_summary,runs,policies,pilot_selection=selection,iterations=iterations)
    result['recurrent_family_prior']=sorted(family_prior)
    result['cross_island_meta']=summarize_multi_seed(runs,minimum_seed_support=minimum_seed_support)
    result['receipt_blake2b256']=hashlib.blake2b(json.dumps({k:v for k,v in result.items() if k!='receipt_blake2b256'},sort_keys=True,separators=(',',':')).encode(),digest_size=32).hexdigest()
    return result
