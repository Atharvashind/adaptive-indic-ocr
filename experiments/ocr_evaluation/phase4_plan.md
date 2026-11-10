# Phase 4: controlled OCR evaluation plan

## Notebook audit update

The original notebook `notebooks/crnn_tes (2).ipynb` now supplies the exact
cell-94 CRNN architecture and the cell-57 native input preparation used by the
primary training/evaluation blocks. Static comparison shows both checkpoints'
keys and shapes match, including the 280-class output and 6,095,128 parameter
count. See
[`recovery/notebook_implementation_audit.md`](./recovery/notebook_implementation_audit.md).
The implementation is sufficiently recovered to proceed to an approved,
safe CPU checkpoint-load test, but no load or inference has been performed.

The baseline is still not reproduced and the five conditions are not ready:
the environment lacks PyTorch, JiWER is unpinned, the archived loss-history
CSV conflicts with the training-history JSON, and the checkpoint load has not
been runtime-verified. Do not start Phase 4 conditions yet.

## Archive-only audit: blocked before the notebook was located

The archive at
`experiments/recovered_baseline/Multilingual_CRNN_OCR-20261105T161607Z-1-001.zip`
is intact and contains both checkpoints, model/training/preprocessing configs,
vocabulary, documentation, historical test predictions, and a metric report.
It does **not** contain the executable CRNN architecture, dataset/collate code,
inference script, evaluation/metric implementation, test exclusion manifest,
or environment/dependency lockfile. The existing model adapter and evaluation
modules in this project remain stubs, and PyTorch is not installed.

The historical CER 7.29% and WER 22.37% are documented in the archive but have
not been reproduced. The checkpoints were statically inspected without loading
their tensor payloads. Both are raw state dictionaries with 42 matching keys
and shapes; their parameter shapes sum to 6,095,128 and the output dimension
matches the 280-index vocabulary. This does not establish strict architecture
compatibility. Recovery is stopped before adapter implementation because the
documentation conflicts on CNN layer count and preprocessing order, and the
original feasibility filter and metric behavior are absent.

## Dataset facts verified

- `synth6k/metadata.csv` contains 82,568 rows: 66,303 train, 8,108 validation,
  and 8,157 test.
- `synth6k/annotations/test.txt` contains 8,157 unique filenames. Its filename
  set exactly matches the metadata test split; every transcription matches
  exactly, and all referenced test image files exist.
- The archived historical prediction log contains 8,094 unique current-test
  samples with exact filename, transcription, and language matches. It omits
  63 of the current 8,157 test rows. The archive README separately claims 746
  CTC-infeasible exclusions, which cannot be reconciled with the saved log;
  no exclusion manifest or feasibility implementation is present.
- Dataset transcriptions are already NFC and NFKC normalized in this test
  split; this does not establish what normalization the original baseline
  applied before training, decoding, or metric calculation.

## Baseline compatibility audit

| Requirement | Finding |
|---|---|
| CRNN model definition and architecture | Missing; documentation conflicts with checkpoint key evidence |
| Input height 32 and width/resize policy | Documented, not executable; 32 px, rounded aspect ratio, bilinear |
| Normalization and padding | Partially documented in README snippet; exact neutral-gray value/collate absent |
| Six-language vocabulary and class-index mapping | Available; 280 contiguous indices, blank 0, output dimension matches |
| CTC decoding and Unicode/whitespace normalization | Greedy decoding example only; normalization and evaluator absent |
| Best epoch-4 checkpoint | Available; raw state dict, static keys/shapes checked, not loaded |
| Test manifest used by reported metrics | Historical predictions cover 8,094 rows; 63 current test rows have no reasoned exclusion record |
| Baseline inference/evaluation command | Missing |
| Recorded CER/WER evidence | Historical report available; not reproduced |
| Required runtime | PyTorch absent and not declared in requirements |

The Phase 2 operation registry has independent `clahe`, `bilateral_filter`,
and `deskew` operations, with defaults in `configs/preprocessing.yaml`.
`AdaptivePreprocessorSelector` can recommend one operation without executing
it. These pieces do not supply a CRNN-native tensor adapter, checkpoint
inference, CTC decoder, or a controlled evaluation runner.

## Experiment protocol after baseline restoration

Do not begin the five-condition comparison until the baseline artifacts and
reproduction gate below are satisfied.

1. Obtain the missing executable CRNN model source, dataset/collate code,
   inference/evaluation code, exact CTC feasibility/exclusion logic, and
   original metric/normalization implementation. Record their hashes and the
   runtime/library versions. The archive already supplies the epoch-4
   checkpoint, vocabulary, and partial preprocessing/model documentation.
2. Run the existing baseline evaluation without changing its input or metric
   configuration. Verify that checkpoint loading succeeds and predictions are
   produced for the exact expected sample set. Compare CER/WER to 7.29%/22.37%
   using the baseline's original decoder, Unicode normalization, and metric
   rules. Do not proceed if the sample IDs or decoding contract are unknown or
   the results cannot be reconciled.
3. Freeze and save one canonical manifest containing the sample ID, source
   filename/path, original split, language, and transcription. Every condition
   must use exactly this manifest and preserve its image-to-transcription
   mapping. Resolve the 63 missing prediction-log rows against the README's
   unsupported claim of 746 exclusions; do not infer exclusions from the
   historical output count.
4. For each canonical original image, create separate condition inputs:
   - **E0:** original pixels.
   - **E1:** one CLAHE operation using the recorded fixed parameters.
   - **E2:** one bilateral filter using the recorded fixed parameters.
   - **E3:** one deskew operation using the recorded fixed parameters; record
     its estimated angle, confidence, and any no-op/failure result.
   - **E4:** evaluate the selector on the original image, execute at most its
     single recommended operation with its recorded parameters, or retain the
     original for `keep_original`.
5. Send each condition's pixels through the unchanged, verified CRNN-native
   input adapter. Do not change the height, width policy, normalization,
   padding, architecture, vocabulary, checkpoint, decoder, or metric rules
   between conditions.
6. Save each condition's transformed images (when applicable), operation and
   selector metadata, predictions, per-image metrics, aggregate CER/WER, and
   hashes under a unique
   `experiments/ocr_evaluation/<dataset>/run-<timestamp>/`. Never overwrite
   source images, checkpoints, annotations, or prior runs.
7. Check sample-ID equality and annotation equality across all five prediction
   sets before comparing metrics. Report failures and excluded samples
   explicitly. Make no improvement claim without these verified OCR metrics.

## Artifacts required to unblock

Provide or point to the following missing baseline components:

- Exact CRNN model implementation and inference entry point, including
  resolution of the documented seven-convolution versus six checkpoint
  convolution-layer discrepancy.
- Original dataset loader/collate and executable image-preprocessing order;
  the README and preprocessing source disagree about whether resize precedes
  contrast enhancement.
- CTC feasibility/exclusion code and the original test manifest that accounts
  for 8,094 predictions versus the current 8,157-row test split.
- Exact decoder, Unicode/whitespace normalization, and CER/WER metric code.
- Runtime/dependency versions and the command/configuration used for the
  historical evaluation.
