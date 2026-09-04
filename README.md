# Bit-myner v0.6

High-throughput Bitcoin SHA256d mining kernel with NOEMA fail-closed admission, DNCrypt/PhaseNav 36D scheduler state, GREMLIN two-stage autotuning, exact scalar-tau jitter modulation, non-overlapping nonce sharding, and an experimental marker-graph research layer.

## Hot path

The native CPU path contains:

- AVX-512 ×2: 32 nonces per batch, two interleaved 16-lane states;
- rolling 16-word SHA-256 message schedules in the AVX-512 ×2 kernel, avoiding full 64-word schedule arrays;
- AVX-512: 16 lanes;
- AVX2: 8 lanes;
- OpenSSL SHA transform fallback;
- OpenMP thread fan-out;
- Bitcoin first-64-byte midstate reuse;
- exact full-256-bit target comparison.

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

MOLE captures selected low-hash events and controls. MANTIS deterministically replays those exact `(header, nonce)` pairs with a full nonce-dependent SHA256d round trace. RAVEN searches candidate motifs. DNCrypt stores marker candidates as content-addressed genomes. HOUND performs header-level validation/holdout falsification. OWL reports lift/support/confidence. BELZEBUB is the only marker-promotion gate.

For surrogate threshold `k`:

```text
Y_k = 1[H < 2^(256-k)]
L_k(g) = P(Y_k=1 | M_g=1) / P(Y_k=1)
```

Train/validation/holdout splitting is performed by header identity, not nonce. A training correlation is never stored as a computational SHA dependency: `COMPUTATIONAL_EDGE` and `ASSOCIATION_EDGE` are separate graph types.

The v0.1 implementation does not instrument every AVX512x2 hash. It replays selected events out of band, preserving the mining hot path. No cryptographic advantage is assumed and no marker modifies production policy until independent validation and holdout lower-confidence-bound gates pass.

Run the bounded research benchmark with:

```text
python -m bitmyner marker-benchmark \
  --headers 15 \
  --nonces-per-header 512 \
  --event-k 8 \
  --top-k 8 \
  --output evidence/marker_graph.json
```

Formal graph semantics are documented in `docs/MARKER_GRAPH_V0_1.md`; the machine-readable schema is `schemas/MARKER_GRAPH_V0_1.schema.json`.

## Correctness boundary

Every admitted native backend must recover the Bitcoin genesis nonce `2083236893` and hash:

```text
000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
```

The rolling AVX-512 ×2 kernel was additionally checked against scalar Python SHA256d on 100 independent nonce cases with 4 native threads: `100/100 PASS` on the validation host. Target-hit aggregation was made race-free by restricting shared best-offset mutation to an OpenMP critical section.

Marker Graph adds a separate reference tracer. It is tested against `hashlib` on the genesis header and 100 deterministic random header/nonce pairs before any marker evaluation is trusted.

## Fresh v0.5 host evidence

`evidence/TAU_NATIVE_VALIDATION_V0_5.json` is the current native validation receipt. Scope is explicitly the current chat-container host, not the user's physical machine, H200 hardware, or a physical QPU.

With a 200 ms stale-work budget and five timing trials per configuration, the fresh bounded rolling-kernel lattice selected:

```text
backend          AVX512X2 rolling 16-word schedule
threads          4
batch_hashes     800,000
median raw       101.123507 MH/s
exact tau score   94.918056 MH/s
jitter CV          0.074690  (telemetry only)
s_tau             39555590 / 42141613
```

The full measured 2–5 thread × 0.8–6.4M batch lattice and nanosecond timings are recorded in the v0.5 evidence file. The raw benchmark receipt is:

```text
dde33d486918d65074cfee8c4a4d321d8fc888f14ec7242abdf8cc6806b39651
```

Host scheduling varied materially between runs, which is why promotion uses tau-modulated repeated timings rather than a peak-hashrate winner.

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
