# Bit-myner

Bit-myner is a NOEMA-gated Bitcoin SHA256d miner focused on measurable throughput rather than heuristic claims about hash success probability. The current v0.2 branch combines an exact native SHA256d hot path with PhaseNav, DNCrypt, GREMLIN and Q28 as scheduler/orchestration layers.

## v0.2 performance core

The hot path is deliberately narrow:

- Bitcoin-correct double SHA-256 over the 80-byte block header;
- first 64-byte SHA256 midstate precomputation;
- 32 nonces processed per inner iteration as two interleaved 16-lane AVX-512 vectors;
- OpenMP across empirically tuned CPU threads;
- vectorized high-64-bit target rejection;
- full 256-bit Python reference verification only for a surviving candidate;
- exact non-overlapping `2^32` nonce sharding, including 100,000-worker partitioning;
- no silent scalar fallback: unsupported hardware fails closed.

The two independent 16-lane SHA256 chains are interleaved to expose instruction-level parallelism while remaining byte-for-byte identical to Bitcoin SHA256d.

## NOEMA / PhaseNav / DNCrypt / GREMLIN / Q28

These layers optimize scheduling and provenance, not SHA256d probability:

- **NOEMA**: live `/dev/shm/ciel_noema` admission, 36D vector integrity, required tether/socket/service checks, fresh batch-level fail-closed gate.
- **PhaseNav 36D**: current live phase state seeds the scheduler candidate ordering and binds the tuning run to the current runtime state.
- **DNCrypt**: deterministic SHAKE-derived candidate ordering/genome commitments over kernel variant, thread count and batch parameters.
- **GREMLIN**: explicit `SPIDER:correctness -> ANT:parallelism -> MANTIS:tuning -> BELZEBUB:promotion` evaluation lineage. Promotion is based only on measured holdout throughput.
- **Q28**: the 6 address qubits provide a 64-candidate scheduler address space. Q28 is intentionally **not** inserted into the SHA256d hot loop: the current Q28 backend is an exact classical simulator and would reduce mining throughput.

Canonical project constant `kappa = ln(2)/(24*pi)` is recorded in scheduler genome provenance but is not used to bias nonce acceptance or alter SHA256d.

## Measured result on the current NOEMA host

Host during the 3 September 2026 benchmark: AMD EPYC 9V74, 5 visible CPUs, AVX-512 available.

A direct v0.1/v0.2 comparison produced these median rates:

| path | median |
|---|---:|
| scalar Python SHA256d | 1.114 MH/s |
| 5-process `hashlib` | 3.198 MH/s |
| previous Numba v0.1, 5 threads | 4.747 MH/s |
| v0.2 AVX512x32 native kernel, 5 threads | **87.604 MH/s** |

Measured median speedups from that comparison:

- **78.67x** vs scalar Python;
- **27.39x** vs same-host 5-process `hashlib`;
- **18.46x** vs the previous Numba mining kernel.

The independent NOEMA/DNCrypt/GREMLIN autotune run selected `avx512x32`; its three-sample holdout was 118.33, 111.64 and 78.00 MH/s, median **111.64 MH/s**. Shared-host scheduling causes visible run-to-run variance, so the project stores raw samples rather than claiming a single fixed rate.

Evidence files:

- `benchmarks/NOEMA_AVX512_20260903.json`
- `benchmarks/V01_V02_COMPARISON_20260903.json`

## Correctness

The native implementation reproduces the Bitcoin genesis block exactly:

```text
nonce = 2083236893
hash  = 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
```

Tests also verify a bounded native scan around the genesis nonce and non-overlapping shard boundaries.

## Build and test

```bash
python -m pip install -e '.[test]'
pytest -q
bitmyner benchmark
```

The native backend builds with GCC using `-O3 -funroll-loops -frename-registers -march=native -mavx512f -mavx512bw -fopenmp`.

## Live mining gate

`mine-header` requires NOEMA supervisor `execution_authorized=true`. Benchmarking and correctness tests do not bypass this gate. If live authority is false, mining fails with `NOEMA_EXECUTION_NOT_AUTHORIZED`.

```bash
bitmyner mine-header --header76 <152 hex chars> --nbits <compact-target-hex>
```

## Claim boundary

No scheduler, phase state, DNCrypt genome, GREMLIN route or Q28 address makes a random SHA256d digest more likely to satisfy the Bitcoin target. Their legitimate performance surface is throughput, hardware utilization, duplicate-work avoidance, stale-work latency, deterministic sharding and auditable promotion.
