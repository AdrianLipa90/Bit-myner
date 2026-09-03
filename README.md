# Bit-myner v0.2

High-throughput Bitcoin SHA256d mining kernel with NOEMA fail-closed admission, DNCrypt/PhaseNav 36D scheduler state, GREMLIN-style audited autotuning, and exact non-overlapping nonce sharding.

## What is optimized

The hot path is no longer Python/Numba. It uses native CPU vector backends:

- AVX-512: 16 independent SHA256d lanes per vector;
- AVX-512 ×2: two interleaved 16-lane states (32 nonces per batch) for additional instruction-level parallelism;
- AVX2: 8 independent SHA256d lanes per vector;
- OpenSSL SHA transform backend, using SHA-NI when the platform exposes it;
- OpenMP thread fan-out;
- Bitcoin 64-byte first-block midstate reuse;
- exact full-256-bit target comparison;
- runtime autotuning over backend × thread count.

The scheduler never claims that a nonce ordering changes the probability of satisfying SHA256d. Performance claims are only hashes/second, latency, duplicate-work avoidance, and scale-out.

## Formalism bindings

`T^36 / PhaseNav` is the scheduler-state surface. The live 36D `phi` vector is used to derive deterministic candidate identities and a bijective worker-to-shard assignment.

`DNCrypt` supplies the 36D genome/route pattern used for candidate generation. Candidate promotion is benchmark-gated, not assumed from the formalism.

`GREMLIN` is represented by auditable roles: SPIDER (correctness), ANT (parallelism), MANTIS (tuning), HOUND (regression), OWL (telemetry), BELZEBUB (promotion). Candidate generation can be broad; promotion is narrow and receipt-bearing.

`Q28` is used only as an exact 28-bit configuration address (`q28_schedule_id`). This is a classical exact schedule identifier, not a physical-QPU speed claim.

`κ = ln(2)/(24π)` is used only as a conservative jitter penalty in autotuning. It does not alter SHA256d or target semantics.

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

## Correctness boundary

Every backend must recover the Bitcoin genesis nonce `2083236893` and hash

```text
000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
```

before it is admitted to autotuning.
