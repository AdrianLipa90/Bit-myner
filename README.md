# Bit-myner v0.4

High-throughput Bitcoin SHA256d mining kernel with NOEMA fail-closed admission, DNCrypt/PhaseNav 36D scheduler state, GREMLIN-style audited autotuning, exact scalar-tau jitter modulation, and exact non-overlapping nonce sharding.

## What is optimized

The hot path uses native CPU vector backends:

- AVX-512: 16 independent SHA256d lanes per vector;
- AVX-512 ×2: two interleaved 16-lane states (32 nonces per batch) for instruction-level parallelism;
- AVX2: 8 independent SHA256d lanes per vector;
- OpenSSL SHA transform fallback, using the platform's optimized SHA implementation when available;
- OpenMP thread fan-out;
- Bitcoin 64-byte first-block midstate reuse;
- exact full-256-bit target comparison;
- runtime autotuning over backend × thread count × batch size.

The scheduler never claims that nonce ordering changes the probability of satisfying SHA256d. Performance claims are limited to hashes/second, latency, duplicate-work avoidance, jitter handling, and scale-out.

## Formalism bindings

`T^36 / PhaseNav` is the scheduler-state surface. The live 36D `phi` vector is used to derive deterministic candidate identities and a bijective worker-to-shard assignment.

`DNCrypt` supplies the 36D genome/route pattern used for candidate generation. Candidate promotion is benchmark-gated, not assumed from the formalism.

`GREMLIN` is represented by auditable roles: SPIDER (correctness), ANT (parallelism), MANTIS (tuning), HOUND (regression), OWL (telemetry), BELZEBUB (promotion). Candidate generation can be broad; promotion is narrow and receipt-bearing.

`Q28` is used only as an exact 28-bit configuration address (`q28_schedule_id`). This is a classical exact schedule identifier, not a physical-QPU speed claim.

`κ = ln(2)/(24π)` remains part of candidate metadata, but jitter promotion authority is no longer a κ-weighted CV heuristic.

## Exact tau modulation of scheduler jitter

For measured batch latencies `dt_i`, Bit-myner defines

```text
m_tau = median(dt_i)
d_tau = mean_i |dt_i - m_tau|

s_tau,jitter  = m_tau / (m_tau + d_tau)
s_tau,latency = min(1, B_tau / m_tau)
s_tau         = s_tau,jitter * s_tau,latency
```

where `B_tau` is the stale-work latency budget. Nanosecond measurements are converted to reduced rational factors.

The promotion score is evaluated directly in the time domain:

```text
H_tau = batch_hashes * 1e9 / (m_tau + d_tau) * s_tau,latency
```

For latency-eligible candidates this is the exact rational form of median service rate modulated by `s_tau,jitter`, without a floating-point jitter coefficient in promotion authority. `jitter_cv` remains telemetry only. A candidate whose median latency exceeds the stale-work budget remains in evidence but cannot be promoted.

The receipt explicitly records:

```text
laboratory_time_modified = false
hydrodynamic_lapse_modified = false
physical_time_claim = false
```

This is computational phase-proper-time/scheduler modulation only. It does not change SHA256d, Bitcoin target semantics, nonce probability, or laboratory time.

## Native correctness boundary

Every native backend must recover the Bitcoin genesis nonce `2083236893` and hash

```text
000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
```

before it is admitted to autotuning.

The current v0.4 host-validation receipt additionally records 50 scalar-vs-native random-header checks per backend. The target-path OpenMP best-hit reduction is race-free: candidate hits are compared only inside the named critical section and the lowest offset is returned after the parallel barrier.

## Current host evidence

`evidence/TAU_NATIVE_VALIDATION_V0_4.json` records the current chat-container validation. It is not a performance claim about the user's physical machine.

On that host, with a 200 ms stale-work budget, the best measured configuration in the bounded validation lattice was:

```text
backend          avx512x2
threads          5
batch_hashes     1,600,000
median           119.570848 MH/s
tau-modulated    111.343660 MH/s
```

A lower-jitter alternative was:

```text
backend          avx512x2
threads          4
batch_hashes     3,200,000
median            97.496007 MH/s
tau-modulated     93.936953 MH/s
```

These measurements are host-specific and must be re-run on the deployment machine.

## Build requirements

Linux, Python 3.11+, GCC, OpenMP, and OpenSSL development headers/libcrypto.

```text
python -m pip install -e .
python -m bitmyner status
python -m bitmyner autotune --output evidence/autotune.json
```

The native library is compiled locally into `.bitmyner-build/` for the detected backend. AVX-512 and AVX2 implementations are separate source files so unsupported instructions are never executed by a selected fallback backend.

## Mining a supplied block-header prefix

Actual work execution is deliberately fail-closed: `mine-header` requires the live NOEMA supervisor to report `execution_authorized=true`.

```text
python -m bitmyner mine-header \
  --backend avx512x2 \
  --threads 4 \
  --header76 <152 hex chars> \
  --nbits 1d00ffff \
  --start 0 \
  --count 100000000
```

A Stratum adapter is intentionally separate from the SHA256d kernel. Pool network effects require their own scoped NOEMA `AuthorizationGrant`; the hashing engine does not mint that authority for itself.

## Non-overlapping scale-out

```text
python -m bitmyner shard --worker-id 17 --workers 100000
```

For `N` workers, the PhaseNav mapping is a permutation of worker IDs followed by exact integer interval partitioning of `[0, 2^32)`. No two workers receive overlapping nonce intervals, and the union covers the full nonce space.
