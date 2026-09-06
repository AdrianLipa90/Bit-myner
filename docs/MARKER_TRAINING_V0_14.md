# Marker Training v0.14 — Power-Balanced No-Skip Crossfit

Status: experimental research layer. No production mining policy is modified by this document or by the v0.14 trainer.

## Purpose

v0.13 expanded the structural predicate space, but its nested TRAIN-confirmation partitions were statistically weak at the hardest surrogate threshold. v0.14 changes the allocation policy before training so that the unseen TRAIN-confirmation partition has a preregistered expected-event target.

The split is label-free. It uses only:

- deterministic header identity and outer TRAIN membership;
- `header_count` and `nonces_per_header`;
- the hardest surrogate threshold `k`;
- the requested expected confirmation-event count;
- a minimum discovery-header budget.

No observed low-hash event is inspected when choosing the confirmation allocation.

For the hardest threshold `k`, the planner uses

```text
P(Y_k = 1) = 2^-k
required_confirmation_samples = ceil(E_target * 2^k)
required_confirmation_headers = ceil(required_confirmation_samples / nonces_per_header)
```

subject to retaining the configured minimum number of discovery headers.

## Nested evidence boundary

The training evidence flow remains:

```text
outer TRAIN
    -> discovery headers: evolution/search only
    -> confirmation headers: frozen until evolution completes
validation: frozen until a selected candidate passes TRAIN confirmation
holdout:    frozen until a selected candidate passes TRAIN confirmation
```

The existing v0.11 label-free adaptive cohort planner and LCB-Q25 discovery robustness are retained.

## Exact no-skip priority semantics

The existing cost-aware marker metric is interpreted as a priority-phase metric, not an exact rejection proof.

For a marker `M`, the measured checkpoint work is spent on every candidate. Candidates that fail the marker are deferred. Candidates that pass the marker are completed first through the full SHA256d path. During that priority phase,

```text
Gamma_k
  = (marker_event_count / priority_phase_work)
    / (baseline_event_count / baseline_full_work)
```

therefore describes observed low-hash event density per SHA-round unit during the marker-positive priority phase.

A marker-negative candidate is **not** declared impossible and is **not** discarded. If all deferred candidates are completed afterwards, exhaustive SHA256d nonce coverage is restored and total full-search work returns to the baseline work budget apart from scheduling/instrumentation overhead.

Consequently:

```text
hard_rejection_authorized = false
deferred_candidates_discarded = false
exact_sha256d_coverage_preserved_if_deferred_completed = true
production_filter_claim = false
```

An empirical `Gamma > 1` can justify further investigation of candidate ordering under stale-job/time constraints only after the nested statistical gates pass. It does not by itself prove a deterministic SHA256d shortcut.

## v0.14 preregistered large run

The dedicated workflow runs three independent islands:

```text
seeds                  6060, 6161, 6262
iterations/island      1000
headers/island         320
nonces/header          256
thresholds             4, 6, 8
population             64
offspring/iteration    8
max predicates/genome  3
expected k=8 confirmation events target  64
minimum discovery headers               32
```

The augmented v0.13 structural space is retained, including `direct`, `xor`, `carry`, and `xor_prev` operators and the unlabeled within-header nonce-variation filter.

Promotion semantics are unchanged: no validation/holdout opening without a nested TRAIN-confirmation pass, no external-evaluation candidate without cross-island evidence, and no production-policy mutation from this experiment.
