# Marker Training 1000 v0.1

Status: `PASS_TRAINING_COMPLETE`  
Receipt: `8f4b4206f788eda3ffa13f0d123383436ce6eb8833b45d6ecde754c848c9375c`

This is an offline falsification-first training experiment over deterministic synthetic Bitcoin block-header fixtures. It does not alter production mining policy and does not assume a cryptographic advantage. Training selection used the training split only. Validation was monitor-only; the holdout was frozen until all 1000 cognitive iterations completed.

## Configuration

- cognitive iterations: 1000
- headers: 100
- nonces/header: 128
- samples: 12800
- train / validation / holdout: 7552 / 3200 / 2048
- surrogate thresholds: [4, 6, 8] leading-zero bits
- population: 64
- offspring/iteration: 8
- maximum marker predicates: 3

## Search evolution

The cognitive training score rose from `0.390607` at iteration 0 to `0.869711` at iteration 1000, a relative increase of `122.66%`. Champion complexity grew from 1 predicate to 3 predicates. Across checkpoint snapshots the champion changed 6 times.

The search evaluated 8064 unique genomes and rejected 11545 duplicate proposals, giving a duplicate-proposal fraction of 58.88%. This is evidence of search-space saturation/novelty pressure, not evidence of SHA256d predictability.

Final champion predicate indices: `[18, 31, 51]`. In the fixed v0.1 predicate lattice they correspond to:

1. `stage1 round0 e bit31 = 0`
2. `stage1 round0 w bit0 = 1`
3. `stage1 round7 e bit0 = 1`

On the full training split the final champion had exact cost-domain gamma values:

- k=4: 1.208331, conservative gamma LCB 0.889711
- k=6: 1.618557, conservative gamma LCB 0.915175
- k=8: 3.233877, conservative gamma LCB 1.104134

The strongest apparent training enrichment was therefore at k=8, but that signal did not generalize.

## Frozen final validation / holdout

After training completed, the top eight genomes were tested using Bonferroni correction over top-8 × thresholds × validation/holdout. The final champion produced:

| k | validation gamma | validation gamma LCB | holdout gamma | holdout gamma LCB |
|---:|---:|---:|---:|---:|
| 4 | 0.953806 | 0.445761 | 1.172841 | 0.482425 |
| 6 | 0.749783 | 0.151109 | 0.697459 | 0.069224 |
| 8 | 0.420797 | 0.014245 | 0.755580 | 0.019916 |

Promoted genomes: **0**.

The k=4 holdout point estimate exceeded 1 (`gamma=1.172841`), but its corrected lower confidence bound was only `0.482425`. At k=6 and k=8 both the point estimates and lower bounds were below 1. Therefore the 1000-iteration run provides no validated evidence of a predictive SHA256d marker.

## Operational conclusion

The run demonstrates measurable cognitive adaptation of the optimizer on its training evidence, but also clear overfitting: the training objective improved while frozen validation/holdout did not follow it. The next optimization target is therefore not a stronger claim, but better generalization discipline: novelty-aware mutation, genealogy/negative-memory penalties, rolling header cohorts, and selection based on cross-header stability before the final holdout gate.
