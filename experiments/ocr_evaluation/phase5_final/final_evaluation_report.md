# Phase 5C frozen held-out test evaluation

- **Run:** `run-20261107T085016.947924Z`
- **Status:** passed
- **Final held-out population:** 7894 CTC-feasible test samples.
- **Phase 5A test samples excluded:** 200.
- **Population policy:** all remaining feasible test samples, original Phase 4B order.
- **Language counts:** `{"bn": 1360, "en": 1244, "hi": 1351, "mr": 1521, "ta": 1168, "te": 1250}`
- **Source boards:** 612 unique; 7282 extra crops across repeated source IDs.

## Overall OCR results

| Arm | N | CER | WER | Exact match | Aggregate Delta CER | Mean paired sample Delta CER | Median paired Delta CER | Improved / unchanged / degraded | Paired mean 95% cluster-bootstrap CI | Preprocess ms/sample | OCR ms/sample | Total ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A — Original | 7894 | 0.072892 | 0.223541 | 11.64% | +0.000000 | +0.000000 | +0.000000 | 0 / 7894 / 0 | [+0.000000, +0.000000] | 0.00 | 98.56 | 98.56 |
| B — Fixed histogram equalization | 7894 | 0.878428 | 0.998332 | 1.20% | -0.805536 | -0.778974 | -0.848771 | 27 / 256 / 7611 | [-0.795358, -0.761857] | 0.86 | 119.04 | 119.91 |
| C — Scene-text CLAHE | 7894 | 0.118020 | 0.325509 | 10.96% | -0.045128 | -0.045907 | +0.000000 | 516 / 4816 / 2562 | [-0.054993, -0.036920] | 0.94 | 93.55 | 94.49 |
| D — Automatic adaptive | 7894 | 0.129472 | 0.318136 | 11.41% | -0.056580 | -0.041602 | +0.000000 | 26 / 6786 / 1082 | [-0.048126, -0.035521] | 4.21 | 85.15 | 89.36 |
| E — User-guided adaptive | 7894 | 0.129472 | 0.318136 | 11.41% | -0.056580 | -0.041602 | +0.000000 | 26 / 6786 / 1082 | [-0.048126, -0.035521] | 4.22 | 86.52 | 90.74 |

Positive Delta CER indicates improvement over Arm A; negative indicates degradation. Aggregate Delta CER is the difference of corpus CER values. The paired mean is the mean of per-sample CER differences; they need not be identical.

## Per-language results

| Arm | Language | N | CER | WER | Exact match |
|---|---|---:|---:|---:|---:|
| A — Original | bn | 1360 | 0.086349 | 0.262933 | 14.71% |
| A — Original | en | 1244 | 0.062191 | 0.187145 | 2.57% |
| A — Original | hi | 1351 | 0.051120 | 0.169253 | 15.99% |
| A — Original | mr | 1521 | 0.071174 | 0.201419 | 14.14% |
| A — Original | ta | 1168 | 0.085031 | 0.260970 | 13.53% |
| A — Original | te | 1250 | 0.082259 | 0.275008 | 7.84% |
| B — Fixed histogram equalization | bn | 1360 | 0.756848 | 0.986241 | 1.84% |
| B — Fixed histogram equalization | en | 1244 | 0.980411 | 1.014371 | 0.08% |
| B — Fixed histogram equalization | hi | 1351 | 0.883621 | 0.985534 | 1.70% |
| B — Fixed histogram equalization | mr | 1521 | 0.894744 | 0.990822 | 2.04% |
| B — Fixed histogram equalization | ta | 1168 | 0.957383 | 1.024062 | 0.43% |
| B — Fixed histogram equalization | te | 1250 | 0.815793 | 0.998264 | 0.80% |
| C — Scene-text CLAHE | bn | 1360 | 0.133524 | 0.351541 | 14.19% |
| C — Scene-text CLAHE | en | 1244 | 0.108615 | 0.294694 | 2.49% |
| C — Scene-text CLAHE | hi | 1351 | 0.079994 | 0.249079 | 16.43% |
| C — Scene-text CLAHE | mr | 1521 | 0.110208 | 0.277288 | 13.74% |
| C — Scene-text CLAHE | ta | 1168 | 0.155629 | 0.420205 | 10.36% |
| C — Scene-text CLAHE | te | 1250 | 0.124726 | 0.395866 | 7.12% |
| D — Automatic adaptive | bn | 1360 | 0.140251 | 0.352091 | 14.49% |
| D — Automatic adaptive | en | 1244 | 0.159013 | 0.321226 | 2.33% |
| D — Automatic adaptive | hi | 1351 | 0.080568 | 0.219884 | 15.99% |
| D — Automatic adaptive | mr | 1521 | 0.090742 | 0.243147 | 14.07% |
| D — Automatic adaptive | ta | 1168 | 0.197412 | 0.461253 | 12.93% |
| D — Automatic adaptive | te | 1250 | 0.124150 | 0.363048 | 7.52% |
| E — User-guided adaptive | bn | 1360 | 0.140251 | 0.352091 | 14.49% |
| E — User-guided adaptive | en | 1244 | 0.159013 | 0.321226 | 2.33% |
| E — User-guided adaptive | hi | 1351 | 0.080568 | 0.219884 | 15.99% |
| E — User-guided adaptive | mr | 1521 | 0.090742 | 0.243147 | 14.07% |
| E — User-guided adaptive | ta | 1168 | 0.197412 | 0.461253 | 12.93% |
| E — User-guided adaptive | te | 1250 | 0.124150 | 0.363048 | 7.52% |

## Adaptive selector behavior

- Selection counts and paired recognition outcomes conditioned on selected operation are in `operation_selection.csv` and `metrics.json`.

| Adaptive arm | Keep original | CLAHE | Deskew |
|---|---:|---:|---:|
| D — Automatic adaptive | 6695 (84.81%) | 608 (7.70%) | 591 (7.49%) |
| E — User-guided adaptive | 6695 (84.81%) | 608 (7.70%) | 591 (7.49%) |

| Adaptive operation | N | CER | WER | Mean paired Delta CER | Improved / unchanged / degraded |
|---|---:|---:|---:|---:|---:|
| clahe | 608 | 0.321025 | 0.731869 | -0.226876 | 14 / 31 / 563 |
| deskew | 591 | 0.397840 | 0.752285 | -0.322279 | 12 / 60 / 519 |
| keep_original | 6695 | 0.068020 | 0.206276 | +0.000000 | 0 / 6695 / 0 |

- Automatic/user-guided exact operation agreement: 7894/7894 (100.00%).
- Transformation failures with original-image fallback: 0.
- Conditional outcome differences are observational and do not show that an operation caused an OCR change.

## Failure analysis

- Transformation failures: 0; the predeclared original-image fallback was not needed.
- Fixed global histogram equalization produced the largest aggregate regression: CER 0.878428 versus 0.072892 for the original arm.
- The `visual_comparisons/failure_analysis/` panels show the worst observed adaptive CLAHE and deskew regressions, selected after evaluation for diagnosis only. They were not used to change any parameter or threshold.
- A sampled Bengali histogram-equalization panel shows a visibly more mottled background and altered stroke edges; this is an illustrative failure, not a population-wide visual classification.

## Statistical uncertainty

- 95% percentile confidence intervals use 5,000 paired bootstrap replicates, resampling source-image IDs as clusters to account for multiple crops per board.
- The reported statistic is the mean per-sample CER difference. The bootstrap seed is 7.
- The run is a census of the remaining finite test population, so the reported test metrics are exact descriptive scores for that population. The cluster-bootstrap intervals describe hypothetical source-board sampling variability, not uncertainty in those finite-population scores.
- One held-out dataset and one checkpoint do not establish broad generalization.

## Implementation and integrity

- Original arm matched Phase 4B predictions exactly: 7894/7894.
- Every arm has 7894 predictions with identical filenames and ground truth.
- Source files unchanged: 7894/7894. Annotation, checkpoint, and vocabulary hashes were unchanged.
- Train/validation source-ID overlaps with test: {'train_vs_val': 0, 'train_vs_test': 0, 'val_vs_test': 0}.
- The complete remainder necessarily includes multiple crops from source boards; one-crop-per-board sampling would not represent the required population because it has only 612 source boards.

## Limitations and supported claims

- The 200 Phase 5A test samples were excluded. This evaluation used the remaining 7,894 CTC-feasible held-out test samples.
- Results apply to this synth6k test remainder, the recovered CRNN checkpoint, its vocabulary/decoder, and the frozen preprocessing configuration.
- No universal improvement, model-family generalization, or causal effect is claimed.
- No test-set tuning, retraining, or changes to prior experiment artifacts were performed.

## Visual comparisons

Fixed-seed, language-stratified examples are saved under `visual_comparisons/`; these examples were selected independently of OCR outcome. The separate `failure_analysis/` examples were selected post hoc for error diagnosis only.

Run artifacts are preserved in `run-20261107T085016.947924Z/`.
