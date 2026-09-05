# Marker training v0.4–v0.8

This document records the implemented research-training path above Marker Graph v0.2. It does not assert a cryptographic advantage over SHA256d.

## v0.3 frozen external evaluation

The v0.2 champion was frozen before a fresh 51,200-sample evaluation. Header overlap with the training fixture was zero. Results were:

| k | events | marker events | lift | lift LCB | Gamma | Gamma LCB | external pass |
|---|---:|---:|---:|---:|---:|---:|---|
| 4 | 3181 | 770 | 1.030052 | 0.909274 | 1.004505 | 0.886723 | false |
| 6 | 822 | 200 | 1.035357 | 0.805826 | 1.009679 | 0.785841 | false |
| 8 | 216 | 54 | 1.063830 | 0.654251 | 1.037445 | 0.638024 | false |

No external-evidence gate passed. Receipt: `5287d248cc8d2ed714471576d5cfc4a18faee18f0e300fcaa0dd1e246e3fa6a3`.

## v0.4 structural / causal search

v0.4 extends the predicate catalog from direct word bits to structural relations between SHA round values. Implemented operators include direct values and selected XOR relations. A label-free nonce-variation filter removes predicates that are nonce-invariant, behave primarily as header selectors, or are too globally imbalanced before threshold labels are used for evolutionary selection.

Training remains header-separated. Rolling train cohorts are used for the robust objective. Validation and holdout are not used during evolution. `final_evaluate=False` additionally permits hyperparameter pilots to finish without opening either post-training split.

## v0.5 multi-seed meta layer

The meta layer compares independent completed training runs by stable predicate identity rather than local catalog index. It separately records exact-genome recurrence and component-family recurrence. Meta recurrence alone cannot promote a production marker.

The corrected v0.5 three-run meta receipt is `c5c9a45d6f5994ddde64e140a7fa2d5272a928d65c8dc0ed9d5842924bc44c42`.

## v0.6 soft-prior islands

v0.6 keeps one focused island and two unrestricted escape islands. The focused island receives only a soft family prior; no hard family whitelist is required. Two escape islands preserve global exploration.

The current 3×1000 comparison produced:

- mean duplicate proposal fraction: `0.4270972792800589`;
- mean final robust core: `0.721543398578332`;
- exact multi-island genome consensus: none;
- promoted production markers: none.

## v0.7 adaptive single-pilot prior

v0.7 calibrates prior strength using train-only pilot runs and then performs three full islands on a disjoint fixture. The calibration objective combines robust core, duplicate fraction, instability, and negative-memory penalty. The single-pilot calibration selected strength `2.0`.

The subsequent 3×1000 run produced:

- mean duplicate proposal fraction: `0.4618219235219559`;
- mean final robust core: `0.7232938942834451`;
- exact multi-island genome consensus: none;
- promoted production markers: none.

The focused full island had duplicate fraction `0.6035008358737339`, so v0.8 replaces single-fixture prior calibration with multiple independent pilot fixtures.

## v0.8 multi-pilot prior calibration

v0.8 evaluates each candidate prior strength across at least three independent, train-only pilot fixtures. The objective is:

```text
median_robust
- 0.35 * mean_duplicate_fraction
- 0.10 * robust_core_MAD
- 0.03 * mean_instability
- 0.01 * mean_negative_penalty
```

Pilot headers and full-training headers are required to be disjoint. The tested strengths were `0, 0.25, 0.5, 1, 2`.

The calibration selected:

```text
chosen prior strength = 0.0
```

so the recurrent-family prior was disabled rather than forced into the full training stage. The three full 1000-iteration islands then produced:

- mean duplicate proposal fraction: `0.37545664604817286`;
- mean final robust core: `0.702978611095001`;
- exact multi-island genome consensus: none;
- promoted production markers: none;
- receipt: `daf2da8268edd1a00575fd215755fd69d07cfde99ab000912a6e46ef7d01b033`.

The compact evolution receipt is `evidence/MARKER_TRAINING_EVOLUTION_V0_3_TO_V0_8.json`.

## Promotion boundary

A marker is not a production optimization because it has high training lift, high training Gamma, family recurrence, or meta recurrence. Production promotion still requires independent evidence under the configured validation/holdout or preregistered external-evaluation policy. No marker in v0.4–v0.8 has passed that boundary.
