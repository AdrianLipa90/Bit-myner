# Bit-myner v0.10

High-throughput Bitcoin SHA256d mining kernel with NOEMA fail-closed admission, DNCrypt/PhaseNav 36D scheduler state, GREMLIN two-stage autotuning, exact scalar-tau jitter modulation, non-overlapping nonce sharding, and an experimental marker-graph research layer.

## Hot path

The native CPU path contains:

- AVX-512 ×2: 32 nonces per batch, two interleaved 16-lane states;
- rolling 16-word SHA-256 message schedules in the AVX-512 ×2 kernel, avoiding full 64-word schedule arrays;
- AVX-512: 16 lanes;
- AVX2: 8 lanes;
- OpenSSL SHA transform fallback;
- OpenMP thread fan-out;
- Bitcoin first-64-byte midstate reuse plus fixed first-SHA round-0..2 prehash;
- exact Bitcoin-target fast path: for the normal zero high target word, reject lanes after round 60 of the second SHA when final word 7 is already non-zero;
- full second-SHA completion and exact 256-bit comparison for surviving lanes, with a full exact fallback for general targets.

The scheduler never assumes that nonce ordering changes SHA256d success probability. Optimization authority is limited to measured throughput, latency, jitter, duplicate-work avoidance, scale-out, and experimentally validated marker policies.

## Formalism bindings

`T^36 / PhaseNav` supplies live 36D scheduler state and a bijective worker-to-shard assignment. `DNCrypt` supplies deterministic 36D candidate/genome identities. `GREMLIN` uses broad candidate generation followed by narrow promotion. `Q28` remains an exact classical 28-bit configuration address only; no physical-QPU performance claim is made.

`κ = ln(2)/(24π)` remains candidate metadata. It no longer controls jitter promotion.

## Exact tau modulation

For measured batch latencies `dt_i`:

```text
m_tau = median(dt_i)
d_tau = mean_i |dt_i - m_tau|

s_tau,jitter  = m_tau / (m_tau + d_tau)
s_tau,latency = min(1, B_tau / m_tau)
s_tau         = s_tau,jitter * s_tau,latency
```

The promotion score is evaluated directly in the time domain:

```text
H_tau = batch_hashes * 1e9 / (m_tau + d_tau) * s_tau,latency
```

All quantities participating in promotion are represented as reduced rationals. `jitter_cv` is telemetry only. A candidate whose median latency exceeds the stale-work budget remains in evidence but cannot be promoted.

Receipt boundary:

```text
laboratory_time_modified = false
hydrodynamic_lapse_modified = false
physical_time_claim = false
nonce_probability_modified = false
sha256d_semantics_modified = false
```

## GREMLIN two-stage autotuner

`bitmyner/tau_exact.py` performs:

```text
BROAD COARSE LATTICE
    backend × threads × batch-size
            ↓
exact tau score
            ↓
TOP-K REFINEMENT
            ↓
exact tau promotion
```

For AVX-512 backends the default batch lattice spans `1/8, 1/4, 1/2, 1, 2` of the requested center batch. A smaller batch can therefore win after tau modulation even when a larger batch has a higher transient raw peak.

## Marker Graph v0.1

The research loop is:

```text
MOLE -> MANTIS -> RAVEN -> DNCrypt -> HOUND -> OWL -> BELZEBUB
```

MOLE captures selected low-hash events and controls. MANTIS deterministically replays those exact `(header, nonce)` pairs with a full nonce-dependent SHA256d round trace. RAVEN searches candidate motifs on training headers only. DNCrypt stores marker candidates as content-addressed genomes. HOUND evaluates only the already-selected top-K candidates on header-separated validation and holdout sets. OWL reports lift/support/confidence. BELZEBUB is the only marker-promotion gate.

For surrogate threshold `k`:

```text
Y_k = 1[H < 2^(256-k)]
L_k(g) = P(Y_k=1 | M_g=1) / P(Y_k=1)
```

The raw SHA256d digest is interpreted with Bitcoin's little-endian PoW numerical convention, equivalently counting leading zero bits on the reversed/display hash. Train/validation/holdout splitting is performed by header identity, not nonce. RAVEN never sees validation or holdout during ranking; HOUND applies a Bonferroni familywise correction across the selected top-K before BELZEBUB receives the evidence. A training correlation is never stored as a computational SHA dependency: `COMPUTATIONAL_EDGE` and `ASSOCIATION_EDGE` are separate graph types.

The v0.1 evidence graph also represents the first-SHA -> second-SHA transition explicitly through the serialized first digest, rather than treating the two compression stages as one uninterrupted round chain.

The v0.1 implementation does not instrument every AVX512x2 hash. It replays selected events out of band, preserving the mining hot path. No cryptographic advantage is assumed and no marker modifies production policy until independent validation and holdout lower-confidence-bound gates pass.

Run the bounded v0.1 benchmark with:

```text
python -m bitmyner marker-benchmark \
  --headers 15 \
  --nonces-per-header 512 \
  --event-k 8 \
  --top-k 8 \
  --output evidence/marker_graph_v01.json
```

## Marker Graph v0.2

v0.2 adds four things to the falsification loop:

1. a bounded multilevel MOLE event ring that stores only `(header, nonce, digest)` tuples in the capture path;
2. deterministic composite two-predicate DNCrypt genomes generated only from training-selected RAVEN seeds;
3. a threshold ladder `K={k_1,...,k_m}` instead of a single surrogate target;
4. an exact SHA-round cost model for deciding whether a marker can increase retained low-hash events per unit of computation.

For a genome `g`, threshold `k`, and exact marker-policy round cost `c_i(g)`:

```text
C0 = 128 SHA rounds
Gamma_k(g)
  = (N_marker,event / sum_i c_i(g))
    / (N_event / (128 N))
```

so `Gamma > 1` means more retained threshold events per SHA-round unit than full SHA256d evaluation on that dataset. The metric includes event loss as well as early-rejection savings.

HOUND evaluates validation and frozen holdout separately at each level. The familywise alpha is Bonferroni-corrected across selected candidates × threshold levels × validation/holdout. BELZEBUB requires conservative cost-adjusted gain above `1 + delta` on both validation and holdout for multiple consecutive threshold levels before a research candidate can pass the v0.2 gate.

Runtime tau coupling is formally

```text
Gamma_tau = Gamma * s_tau,marker / s_tau,baseline
```

but the offline v0.2 experiment has no measured marker-policy latency receipt, so the ratio remains exactly 1 until a real runtime measurement exists. No runtime tau gain is invented from round counts.

Run the bounded v0.2 benchmark with:

```text
python -m bitmyner marker-benchmark-v2 \
  --headers 21 \
  --nonces-per-header 512 \
  --thresholds 6,8,10,12 \
  --seed-top-k 16 \
  --pair-limit 64 \
  --top-k 8 \
  --output evidence/marker_graph_v02.json
```

Formal v0.1 semantics are documented in `docs/MARKER_GRAPH_V0_1.md`; v0.2 multilevel/cost semantics are in `docs/MARKER_GRAPH_V0_2.md`. The machine-readable v0.1 graph schema remains `schemas/MARKER_GRAPH_V0_1.schema.json` because v0.2 extends experiment/promotion semantics without changing the stored graph node/edge contract.

## Correctness boundary

Every admitted native backend must recover the Bitcoin genesis nonce `2083236893` and hash:

```text
000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
```

The optimized AVX-512 ×2 target fast path is additionally cross-checked against scalar exact SHA256d first-hit semantics on arbitrary targets. Current measured receipt: `evidence/SHA256D_TARGET_FASTPATH_V1.json`.

## Build and autotune

Requirements: Linux, Python 3.11+, GCC, OpenMP, OpenSSL development headers/libcrypto.

```text
python -m pip install -e '.[test]'
python -m bitmyner status
python -m bitmyner autotune --output evidence/autotune.json
```

The native libraries are built locally into `.bitmyner-build/`. CI performs Python compile checks and the native/tau/marker-graph test suite.

## Mining a supplied block-header prefix

Actual work remains fail-closed. `mine-header` requires the live NOEMA supervisor to report `execution_authorized=true`.

```text
python -m bitmyner mine-header \
  --backend avx512x2 \
  --threads 4 \
  --header76 <152 hex chars> \
  --nbits 1d00ffff \
  --start 0 \
  --count 100000000
```

A Stratum adapter remains separate from the hashing kernel. Pool/network effects require their own scoped NOEMA authorization.

## Non-overlapping scale-out

```text
python -m bitmyner shard --worker-id 17 --workers 100000
```

For `N` workers, PhaseNav permutes worker IDs and then partitions `[0, 2^32)` exactly. Shards do not overlap and their union covers the nonce space.
