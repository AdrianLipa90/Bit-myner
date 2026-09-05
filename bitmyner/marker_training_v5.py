from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .marker_training_v4 import run_training_1000_v4

SCHEMA='BITMYNER_MARKER_TRAINING_V0_5_META_FAMILY'


def run_meta_training_v5(
    meta_summary: dict,
    *,
    iterations: int = 1000,
    seed: int = 919,
    header_count: int = 120,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4,6,8),
    minimum_seed_support: int = 2,
) -> dict:
    if meta_summary.get('status') != 'PASS_META_COMPLETE':
        raise ValueError('meta summary must be complete')
    families = [
        str(row['family']) for row in meta_summary.get('consensus_component_families', [])
        if int(row.get('seed_support',0)) >= int(minimum_seed_support)
    ]
    families = sorted(set(families))
    if not families:
        raise RuntimeError('no recurrent component families available')
    run = run_training_1000_v4(
        iterations=iterations,
        header_count=header_count,
        nonces_per_header=nonces_per_header,
        thresholds=thresholds,
        population_size=16,
        offspring_per_iteration=8,
        max_predicates=3,
        seed=seed,
        split_seed='bitmyner-marker-meta-v0.5',
        checkpoint_every=max(1, iterations//10),
        cohort_count=4,
        cohort_refresh_every=25,
        family_whitelist=families,
    )
    payload={
        'schema':SCHEMA,
        'status':'PASS_META_TRAINING_COMPLETE',
        'source_meta_receipt':meta_summary.get('receipt_blake2b256'),
        'minimum_seed_support':minimum_seed_support,
        'whitelisted_families':families,
        'training_run':run,
        'claims':{
            'prior_validation_or_holdout_used_as_labels':False,
            'family_whitelist_derived_from_training_topologies_only':True,
            'external_data_used':False,
            'production_policy_modified':False,
        },
    }
    payload['receipt_blake2b256']=hashlib.blake2b(json.dumps(payload,sort_keys=True,separators=(',',':')).encode(),digest_size=32).hexdigest()
    return payload
