# Phase 5B recommendations

## Evidence from the validation diagnostic

Run: `run-20261107T081541.101804Z`, 200 CTC-feasible validation samples,
stratified across six languages with one crop per source board.

- Arm A (original image, native contrast `2.0`) had CER `0.062317` and WER
  `0.191844`.
- CLAHE followed by the normal native contrast step (B) had CER `0.096694`.
  Its paired mean CER delta (`CER_A - CER_B`) was `-0.032119`; 8 samples
  improved, 134 were unchanged, and 58 degraded.
- CLAHE with native contrast disabled (D) had CER `0.280866`. Relative to
  Arm F, its paired mean delta was `-0.042862`; 34 improved, 79 were
  unchanged, and 87 degraded.
- The descriptive CLAHE interaction contrast `(F-D) - (A-B)` was
  `-0.010744`. The paired pilot does not establish a meaningful interaction.
- Histogram equalization was severely disruptive in either setting: CER
  `0.878730` with native contrast enabled (C), and `0.669982` with it disabled
  (E). Disabling native contrast after equalization reduced the mean paired
  error relative to C, but E still degraded versus its original-image control.
- With no external operation, disabling native contrast (F) increased CER to
  `0.233177`, compared with `0.062317` in A. This supports retaining `2.0` as
  the model's frozen default for the original-image baseline.
- Per-language Arm A CER ranged from `0.024582` (English) to `0.107173`
  (Tamil). Both external contrast operations were worse than A in aggregate
  across each of the six sampled language strata; these are small per-language
  samples.

## Recommendations

1. Keep the production CRNN adapter and default native contrast factor at
   `2.0`. Do not change the default based on this validation diagnostic.
2. Do not select histogram equalization as a default preprocessing operation
   for this checkpoint based on the observed large regressions.
3. Do not infer that CLAHE's degradation is caused solely by its interaction
   with native contrast: it degraded in both native-contrast settings, and the
   interaction contrast was descriptive and small relative to uncertainty in
   this sample.
4. Preserve all six ablation definitions and current operation parameters if
   the study advances. Do not tune thresholds or parameters using test-set OCR
   results.
5. Before any final test evaluation, freeze the protocol, use the same paired
   sample set and checkpoint across conditions, and report paired outcomes,
   per-language results, and uncertainty. Treat the full test split as held
   out; do not make further choices based on its results.
6. Review saved four-panel comparisons for likely failure patterns such as
   matra/component loss, thin-stroke changes, stroke merging, texture, and
   amplified noise. These examples support diagnosis, not parameter tuning or
   population-level conclusions.

## Limitations

This is a fixed-seed, 200-image validation-split diagnostic. No saved Phase 4D
validation prediction set was available; Arm A was instead verified against
the frozen adapter on these same images. No confidence intervals, hypothesis
tests, or causal claims are provided. No test images were evaluated, no model
was retrained, and no selector settings were changed.
