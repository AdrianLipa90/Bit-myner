# Marker training v0.9–v0.10

This document records the bridge-recombination and nested cross-fit layers. Neither layer asserts a cryptographic advantage over SHA256d.

## v0.9 cross-seed bridge recombination

The v0.8 multi-pilot calibration selected prior strength `0.0`; therefore recurrent families are no longer injected as a forced proposal prior. v0.9 instead uses recurrent component families only to define a bounded search space on a new bridge fixture.

The bridge policy is:

```text
meta recurrence -> fresh bridge fixture -> TRAIN-only single ranking
               -> cross-family pair recombination -> selected TOP-K
               -> bridge validation + bridge holdout -> bridge pass/reject
```

Validation and holdout have no ranking authority. Bonferroni correction is applied over selected candidates × thresholds × validation/holdout. A bridge pass only authorizes a new preregistered external evaluation.

The first bounded bridge run used 80 headers × 96 nonces = 7,680 samples. It retained 46 nonce-varying predicates from five recurrent families, selected 15 training singles, tested 32 pair candidates and advanced 6 candidates to the bridge gate. Zero candidates passed. Receipt: `c28c6280b5b5ff5e53301d7d36f8de22a75f20188dbbcf10bc68f57f6a8ef29c`.

## v0.10 nested TRAIN cross-fit

v0.10 adds an outer split inside the ordinary TRAIN partition:

```text
TRAIN -> discovery headers + confirmation headers
```

Evolution, cohort refresh, novelty and negative memory operate only on discovery headers. The final TOP-8 is then tested on the previously unseen TRAIN-confirmation headers. A candidate that fails confirmation does not open validation or holdout.

For candidates that pass confirmation, the existing validation/holdout gate is applied unchanged. Multi-seed meta recurrence in v0.10 is computed only over confirmation-passed candidates.

This reduces selection leakage from repeated evolutionary search without spending the external-evaluation dataset.

The first completed 3×1000 nested run used seeds `3030,3131,3232` and produced:

- mean duplicate proposal fraction: `0.3742401890275418`;
- mean final robust core: `0.705439998069812`;
- TRAIN-confirmation pass counts: `[0,0,0]`;
- validation/holdout open counts: `[0,0,0]`;
- promoted counts: `[0,0,0]`;
- exact confirmation-passed cross-seed consensus: none;
- compact receipt: `da69032a5ebe24e168535a172f865ce5f1bdb93061bb34a5cefea3600f110510`.

This is a stronger negative result than the earlier validation failures because the external validation/holdout surfaces were not spent at all: the candidates failed one layer earlier on previously unseen TRAIN-confirmation headers.

## Evidence boundary

Bridge recurrence, discovery fitness, confirmation success, validation success and holdout success are all research evidence layers. Production promotion still requires the configured independent external-evidence policy.
