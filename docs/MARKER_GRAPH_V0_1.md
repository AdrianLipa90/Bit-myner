# Bit-myner Marker Graph v0.1

Status: experimental research layer. No cryptographic advantage is assumed.

## Roles

`MOLE -> MANTIS -> RAVEN -> DNCrypt -> HOUND -> OWL -> BELZEBUB`

- MOLE: SHA256d work and threshold-event capture.
- MANTIS: deterministic replay of selected nonce values with exact round-state tracing.
- RAVEN: motif/candidate search over training-header traces only.
- DNCrypt: content-addressed marker genomes.
- HOUND: validation/holdout evaluation of the already-selected top-K genomes.
- OWL: lift, support, confidence and cost telemetry.
- BELZEBUB: the only marker-promotion gate.

## Surrogate event

For SHA256d digest `H` and difficulty surrogate `k`:

`Y_k = 1[H < 2^(256-k)]`.

Bitcoin compares the SHA256d digest as a little-endian numerical value. The implementation therefore evaluates the raw 32-byte SHA256d digest as little-endian, equivalently counting leading zero bits on the conventional reversed/display hash. The genesis block fixture has 43 such leading zero bits.

The implementation uses these leading-zero thresholds as dense research labels. A surrogate event is not a Bitcoin share or block.

## Marker genome

A v0.1 genome is a Boolean composition of exact 32-bit predicates over nonce-dependent SHA256d round-state observables:

`(stage, round, source, mask, value)`

where `source` is one of `a..h`, `t1`, `t2`, or message-schedule word `w`. Genome identity is BLAKE2b-256 over canonical JSON.

## Evidence graph

Computational SHA relations are stored only as `COMPUTATIONAL_EDGE`. Statistical/motif relations are stored only as `ASSOCIATION_EDGE`. The two edge classes are intentionally non-interchangeable.

The exact first-SHA -> second-SHA transition is represented explicitly. Stage-1 compression is finalized and serialized into a `SHA256_FIRST` hash node; that 32-byte digest becomes the message input for stage 2. The final digest is stored separately as `SHA256D_FINAL`.

## Lift

`L_k(g) = P(Y_k=1 | M_g=1) / P(Y_k=1)`.

The evaluator reports a conservative lower confidence bound obtained from a Wilson lower bound on the conditional event probability divided by a Wilson upper bound on the baseline event probability.

Train/validation/holdout assignment is deterministic and performed by header identity, not by nonce, to prevent nonce-level leakage across the same header.

RAVEN never evaluates validation or holdout rows while ranking the candidate population. It selects top-K genomes using training headers only. HOUND then evaluates those selected genomes on validation and holdout. The familywise alpha is Bonferroni-corrected over the selected top-K before BELZEBUB sees the evidence.

## Promotion rule

Training rank does not authorize promotion. BELZEBUB requires:

- sufficient holdout marker support,
- non-zero holdout event support,
- holdout lift lower-confidence-bound > `1 + delta`, and
- validation lift lower-confidence-bound > `1`.

The v0.1 experiment never changes the production mining policy automatically.

## Runtime boundary

The tracer is an offline/replay instrument. The AVX512x2 mining hot path is not instrumented per hash. MOLE should retain only selected event/control `(header, nonce, digest)` tuples and MANTIS reconstructs their round states out of band.
