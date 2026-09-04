# Bit-myner Marker Graph v0.2

Status: experimental research layer. No cryptographic advantage is assumed or claimed.

## Objective

Marker Graph v0.2 tests whether internal SHA256d state predicates can improve the rate of low-hash events per unit of computation after the cost of the marker policy is included. Candidate discovery and production promotion remain separate.

The GREMLIN research loop is:

```text
MOLE -> MANTIS -> RAVEN -> DNCrypt -> HOUND -> OWL -> BELZEBUB
```

RAVEN may generate and rank candidates only from training headers. Validation and holdout data are unavailable to candidate construction/ranking. BELZEBUB alone evaluates the final promotion rule.

## Multilevel low-hash ladder

For raw SHA256d digest `H`, interpreted as the Bitcoin little-endian PoW integer, define

```text
Y_k = 1[H < 2^(256-k)]
```

for an ordered threshold ladder

```text
K = {k_1, ..., k_m}.
```

A low-hash event at a stricter level is therefore also an event at every weaker level. Surrogate events are research labels; they are not Bitcoin shares or blocks.

## MOLE bounded capture ring

The online capture surface accepts only an already-computed tuple

```text
(header76, nonce, digest)
```

and retains bounded event rings plus a bounded deterministic control sample. It does not compute or retain SHA round traces in the hot path.

For a retained row MANTIS later performs deterministic replay:

```text
(header76, nonce) -> full SHA256d round trace
```

and rejects the replay if its digest differs from the captured digest.

## DNCrypt composite marker genome

A predicate is

```text
(stage, round, source, mask, value)
```

with `source in {a,b,c,d,e,f,g,h,t1,t2,w}`.

A v0.2 genome is a canonical ordered set of predicates combined with Boolean `all` or `any`. Predicate order in the input does not affect the genome ID. Identity remains BLAKE2b-256 over canonical JSON.

RAVEN first ranks single-predicate seeds using training headers only, then generates deterministic two-predicate genomes from those training-selected seeds. The v0.2 implementation deliberately keeps this search bounded; larger genomes require a later explicit search-budget gate.

## Exact computational cost

A complete SHA256d evaluation costs

```text
C_0 = 128
```

SHA-256 round units: 64 rounds in the first hash and 64 in the second.

For an `all` genome, a sample may be rejected at the first failed predicate checkpoint. If every predicate passes, computation continues to the complete SHA256d result. For an `any` genome, a passing predicate retains the sample and completes SHA256d; if all fail, the sample can be rejected at the last evaluated checkpoint.

Let `c_i(g)` be the resulting exact round cost for sample `i`. For threshold `k`, define

```text
N       = number of samples
N_Y     = number with Y_k = 1
N_MY    = number with M_g = 1 and Y_k = 1
C_g     = sum_i c_i(g)
```

The observed compute-domain gain is

```text
Gamma_k(g)
  = (N_MY / C_g) / (N_Y / (128 N))
  = 128 N N_MY / (N_Y C_g).
```

`Gamma > 1` means more retained low-hash events per SHA round unit than the full-SHA baseline on that evaluation set. It does not by itself establish a cryptographic probability advantage.

The cost model currently includes SHA round work only. Predicate bit tests, branches, cache effects, SIMD packing and scheduler latency are not silently estimated.

## Conservative cost-adjusted gate

The statistical evaluator still measures

```text
L_k(g) = P(Y_k = 1 | M_g = 1) / P(Y_k = 1).
```

For validation and holdout, the event-enrichment term is replaced by its conservative lift lower confidence bound while the observed deterministic cost envelope is held fixed. This yields a conservative cost-adjusted quantity `Gamma_LCB`. It is explicitly not presented as a complete confidence interval for execution cost.

Multiple testing uses a familywise alpha of 0.05 with Bonferroni correction over

```text
selected candidates x threshold levels x {validation, holdout}.
```

## Persistence requirement

A candidate is research-promotable only if both validation and holdout satisfy

```text
Gamma_LCB > 1 + delta
```

at multiple consecutive levels of the low-hash ladder. v0.2 requires at least two consecutive levels when two or more levels are supplied.

This persistence requirement is intended to reject a marker that is attractive only at one loose surrogate threshold.

Even a research-promotable marker does not automatically alter the production mining policy in v0.2.

## Tau integration

Let `s_tau,g` be the measured exact scalar-tau factor for the marker policy and `s_tau,0` the corresponding baseline factor. Runtime-adjusted gain is defined as

```text
Gamma_k^(tau)(g) = Gamma_k(g) * s_tau,g / s_tau,0.
```

The offline Marker Graph v0.2 experiment does not contain a measured marker-policy latency receipt. Therefore its tau ratio is fixed to identity:

```text
s_tau,g / s_tau,0 = 1
```

until a measured runtime experiment exists. No laboratory-time, hydrodynamic-lapse, or physical-time modification is claimed.

## Epistemic boundary

The following statements are distinct:

- a marker has high training lift;
- a marker generalizes to validation;
- a marker survives frozen holdout;
- a marker gives `Gamma > 1` under the exact round-cost model;
- a marker gives `Gamma^(tau) > 1` under measured runtime latency;
- a marker changes the probability distribution of SHA256d outputs.

Marker Graph v0.2 may establish the earlier statements experimentally. It does not infer the final statement from them.
