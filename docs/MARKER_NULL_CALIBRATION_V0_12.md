# Marker Null Calibration v0.12

## Purpose

v0.12 measures the conditional false-positive behaviour of the nested TRAIN-confirmation gate without changing candidate genomes, opening validation/holdout, or modifying production mining policy.

For each completed v0.11 island, the final TOP-8 candidate genomes are frozen. The confirmation surface is reconstructed exactly from the deterministic header fixture and nested discovery/confirmation split. Low-hash labels are then permuted **within each confirmation header** as exact leading-zero-bit values. This preserves, for every header, the event count at each threshold and preserves threshold nesting.

The null experiment therefore keeps fixed:

- header identity;
- nonce identity;
- all structural SHA features;
- candidate genomes and their ordering;
- confirmation sample size;
- per-header event counts at every `k`;
- the Bonferroni confirmation gate.

Only the association between confirmation features and low-hash labels is destroyed.

## Gate

For every frozen candidate and threshold, v0.12 reuses the existing cost-aware metric `Gamma_k` and its lower confidence bound `Gamma_k^LCB`. A threshold qualifies only when:

```text
marker_n >= 32
and event_n > 0
and Gamma_k^LCB > 1
```

A candidate requires at least two consecutive qualifying threshold levels. The familywise alpha remains 0.05 with Bonferroni correction over candidate × threshold tests.

The null-calibrated research threshold is:

```text
max(1, q95(max Gamma_k^LCB under the within-header null))
```

so null calibration may keep or raise the substantive `Gamma^LCB > 1` boundary, but never lower it below one.

## 3 x 1000 result

GitHub Actions run `34033519338` executed 3,000 v0.11 training iterations plus 3,000 within-header null permutations. The base v0.11 confirmation pass counts remained `[0,0,0]`.

Observed maximum confirmation `Gamma^LCB` values were:

```text
seed 4040: 0.6335722849
seed 4141: 0.6115649692
seed 4242: 0.6765480192
```

The corresponding 95th-percentile null maxima were:

```text
seed 4040: 0.6339007003
seed 4141: 0.6366926525
seed 4242: 0.6765480192
```

Empirical one-sided p-values for the observed maximum were:

```text
0.0699300699
0.2077922078
0.1158841159
```

Across 1,000 permutations per island, the familywise false-pass count was zero for all three islands. The recommended substantive gate therefore remains exactly:

```text
Gamma^LCB > 1.0
```

This does **not** establish a marker advantage. It shows that the present confirmation gate is not behaving as an overly liberal false-positive gate on these frozen candidate sets. The observed candidates are not significant against the within-header permutation null at alpha 0.05.

## Current bottleneck

The hardest confirmation threshold remains power-limited. The v0.11 islands contain 2,816–3,584 confirmation samples, corresponding to only about 11–14 expected `k=8` events under the `2^-8` surrogate model. The next improvement should increase independent confirmation information or expand the feature family while retaining the frozen nested-confirmation boundary. It should not weaken `Gamma^LCB > 1`.

## Evidence

Compact receipt:

`evidence/MARKER_NULL_CALIBRATION_V0_12_3X1000_SUMMARY.json`

Full run receipt BLAKE2b-256:

`2a8b847a26ddf0d3a270a6e01a5502ad65fabdf250493c2f1b8e5cf01a450849`

Artifact ZIP SHA-256:

`672eb290f49c3f0ca5a6bdffab1ee0ca2c22aad5ebb7f28e67d4955146715c2e`

No cryptographic advantage is asserted. No production marker policy was changed.
