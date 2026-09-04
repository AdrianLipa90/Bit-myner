# Marker Training 1000 v0.2

Status: `PASS_TRAINING_COMPLETE`

Exact source commit: `3fcf366cf07fdd5b07de30431a3265a6017ba181`  
GitHub Actions run: `33917235010`  
Raw receipt BLAKE2b-256: `37e3e452dbf6c62b160b87da54ec55570f16dc49c46646b70f82b00fce998750`

This experiment keeps the v0.1 deterministic 12,800-sample fixture and frozen train/validation/holdout split while changing the evolutionary selection discipline. Validation and holdout have no selection authority during the 1000 cognitive iterations. Final validation/holdout are opened only after the evolutionary run is complete.

## v0.2 selection discipline

The v0.2 optimizer adds four mechanisms requested after the v0.1 overfit diagnosis:

- novelty search over Jaccard distance between predicate-index genomes;
- negative memory for repeatedly weak or unstable predicate families;
- genealogy crowding penalty so one successful lineage cannot cheaply occupy the search;
- deterministic rolling train-header cohorts, refreshed every 25 iterations.

The train objective is no longer the strongest fixed-training gamma. Each threshold requires support on the full visible train set and across multiple train-only cohorts. The robust term is bounded by both the full-train gamma lower confidence bound and the lower quartile of cohort gamma. Instability, negative-memory load, genealogy crowding and excess complexity reduce the final score; novelty contributes only a bounded bonus.

## Search behavior

Both v0.1 and v0.2 evaluated 8,064 unique genomes. Duplicate proposals changed from:

- v0.1: `58.8760%`
- v0.2: `42.7963%`

This is an absolute reduction of `16.0797` percentage points and a relative reduction of `27.3111%`. v0.2 therefore spent materially less of its proposal budget revisiting an already evaluated genome.

The v0.2 run refreshed train cohorts 40 times, created 2,397 root lineages and recorded 7,927 negative-memory observations. The checkpoint champion changed nine times. The composite train score is intentionally non-monotonic because visible train evidence and cohort partitions change during the run.

## Final champion

Final predicate indices: `[12, 14, 118]`, corresponding in the fixed predicate lattice to:

1. stage 1, round 0, `e`, bit 7 = 0;
2. stage 1, round 0, `e`, bit 15 = 0;
3. stage 1, round 15, `w`, bit 31 = 0.

Final robust core: `0.7193256305`.

Full-train gamma:

| k | gamma | gamma LCB | robust gamma |
|---:|---:|---:|---:|
| 4 | 1.054735 | 0.817518 | 0.793061 |
| 6 | 1.399115 | 0.866102 | 0.866102 |
| 8 | 1.986418 | 0.719326 | 0.719326 |

Frozen external splits:

| k | validation gamma | validation LCB | holdout gamma | holdout LCB |
|---:|---:|---:|---:|---:|
| 4 | 0.992785 | 0.492378 | 0.679270 | 0.209506 |
| 6 | 1.322314 | 0.378085 | 0.875214 | 0.086889 |
| 8 | 1.385281 | 0.133236 | 1.896296 | 0.087570 |

Promoted genomes: **0**.

The k=8 holdout point estimate is above 1, but the corrected lower confidence bound is only `0.08757`. It is therefore evidence of statistical uncertainty, not evidence of a predictive SHA256d marker.

## v0.1 -> v0.2 generalization diagnostics

The mean absolute gap between the final champion's full-train gamma and holdout gamma across k=4,6,8 fell from:

`1.144962 -> 0.329829`

which is a `71.19%` reduction. This supports the narrower claim that v0.2 reduced train-only inflation on this fixture.

However, the mean absolute validation-to-holdout gamma gap changed from:

`0.202048 -> 0.423877`.

Independent-split stability therefore did **not** improve uniformly. Generalization is not established.

## Current bottleneck

At k=8, a 2,048-sample holdout contains only about eight events in expectation. The confidence interval is consequently too broad to distinguish a moderate effect from chance. Further evolutionary iterations on the same evidence are lower priority than a larger preregistered final-evaluation corpus.

For a Bernoulli baseline p=2^-k, the expected sample requirement for E threshold events is:

`N = E * 2^k`.

Thus at k=8:

- 100 expected events require about 25,600 fresh samples;
- 200 expected events require about 51,200 fresh samples.

The next stage should freeze the selected genomes first, generate a much larger set of fresh headers/nonces, and evaluate those frozen genomes without further tuning. Native MOLE can provide cheap low-hash event capture, while MANTIS replays only the checkpoint state required by each frozen marker.

No cryptographic advantage is claimed, and production mining policy remains unchanged.
