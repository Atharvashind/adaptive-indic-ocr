# Adaptive Indic OCR

Phase 1 provides read-only dataset ingestion, annotation profiling, image-quality
indicators, and reproducible reports. It does not run OCR, classify domain,
language, or script, or modify source images.

## Setup

On Windows, create and activate a virtual environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --only-binary=:all:
python -m pip install -r requirements.txt
```

The explicit PyTorch command installs the verified CPU-only build required by
the recovered CRNN adapter. `requirements.txt` pins its compatible PyTorch
release and JiWER 4.0.0.

Run the unit tests:

```powershell
python -m pytest
```

## Configure a dataset

Edit `configs/datasets.yaml`. The locally verified `synth6k` dataset is enabled
using `synth6k/metadata.csv` as its structured annotation source and the three
split text files as validation references. Its metadata rows preserve the
original image filename, source board ID, bbox, language, board type,
transcription, split, and crop dimensions. The text files are parsed as
`<filename> <transcription>` with the transcription preserved after the first
whitespace separator. Source-board split overlap and metadata/reference
consistency are reported; existing split assignments are never redistributed.

The PMGSY, IIIT-INDIC-HW-WORDS, and Mozhi entries remain disabled until their
actual paths and annotation schemas are verified. Relative paths are resolved
from the project root. Image discovery supports PNG, JPG/JPEG, TIFF, BMP, and
WEBP.

Implemented annotation adapters:

- `csv`: configure an explicit `columns` mapping from normalized fields
  (`source_image_id`, `image_path`, `annotation_id`, `parent_annotation_id`,
  `granularity`, `transcription`, `language`, and `script`) to actual CSV column
  names. Set `bbox_columns` to four actual columns for x/y/width/height,
  `bbox_field` for a stringified four-coordinate sequence, and `polygon_field`
  to a JSON list of points when present. Do not configure fields that are not
  in the source. Optional `reference_files` can identify text reference files
  using the `text_reference` format.
- `coco_instances`: reads the documented COCO instances `images`, `annotations`,
  `id`, `image_id`, `file_name`, `bbox`, and polygon segmentation structure.
  Transcription, language, script, and granularity are read only when their
  actual custom field names are explicitly configured.

Other annotation formats are reported as unsupported rather than guessed. When
an annotation file is absent, images are still profiled and annotation coverage
is reported as unavailable. User-provided domain labels are kept distinct from
image-based predictions; no domain classifier is included.

## Run profiling

After enabling and configuring a dataset:

```powershell
python -m src.run_profiling --dataset pmgsy
python -m src.run_profiling --dataset pmgsy --sample-limit 200
python -m src.run_profiling --dataset pmgsy --domain scene_text
```

`--sample-limit` uses a reproducible, split- and language-stratified random
sample (`--seed 0` by default). The report records its method, seed, and sample
limit.
Profiles retain the discovered total image count and separately report how many
images were assessed for that run.
`--domain` records a user-specified metadata label; quality metrics are still
calculated independently. Indicator thresholds are configurable under `quality`
in `configs/datasets.yaml`.

Each invocation creates a new run directory under
`experiments/profiling/<dataset>/run-<UTC timestamp>/`, preserving previous
results. It contains `dataset_profile.json`, `image_quality.csv`,
`annotation_profile.csv`, `dataset_summary.md`, a dataset comparison CSV, and
PNG charts for dimensions, aspect ratio, brightness, contrast, blur, and
representative source images. Empty or unreadable datasets are represented
explicitly. Noise uses a high-frequency median-filter residual estimator and
may include text strokes, texture, or compression artifacts; it is not a
reliable noise-only classification. Skew estimates are omitted when line
orientation evidence is insufficient.

## Phase 2: preprocessing strategy library

Phase 2 provides independent operations; it does not automatically chain them,
select operations adaptively, or run OCR. Operations accept a `uint8` NumPy
grayscale or RGB/RGBA image and optional metadata. Each result includes a new
image, operation name, effective parameters, input/output dimensions, elapsed
time, status, warnings, details, and the passed metadata. The input array is
never mutated. Source identifiers and Unicode annotation metadata remain
outside the pixel transformation.

Available operation names:

- Contrast: `clahe`, `histogram_equalization`
- Denoising: `gaussian_filter`, `median_filter`, `bilateral_filter`
- Thresholding: `otsu_threshold`, `adaptive_gaussian_threshold`,
  `adaptive_mean_threshold`
- Sharpening: `unsharp_mask`
- Geometry: `deskew`, `aspect_ratio_padding`

Default parameters and validated ranges are documented in
`configs/preprocessing.yaml`. Defaults are starting points, not universally
optimal settings or evidence of improved OCR. CLAHE/histogram equalization
operate on luminance for color inputs; thresholding produces grayscale binary
output. Deskew rotates only when line evidence is sufficient and reports its
angle/confidence; otherwise it returns an unchanged copy and a warning.
`aspect_ratio_padding` pads to a configured dimension multiple without
resizing source pixels. Geometric operations may change output dimensions.
Color arrays use RGB channel order (not OpenCV's BGR convention); convert BGR
inputs before calling the operations. Geometric results record an input-to-output
affine transform. Annotation boxes are transformed only when explicitly declared
as `xyxy` in input-image coordinates; other coordinate spaces remain unchanged
and are identified in the operation details. In synth6k, `bbox` is retained in
source-board coordinates and is not treated as a line-crop-local box.

Run the deterministic 10-image demonstration on synth6k:

```powershell
python -m src.run_preprocessing_demo --dataset synth6k --sample-limit 10 --seed 7
```

Override `--preprocessing-config`, `--dataset-config`, or `--output-root` as
needed. Each invocation creates a new, non-overwriting run directory under
`experiments/preprocessing/<dataset>/run-<UTC timestamp>/`. Every sample keeps
its source image in the dataset and receives one separate output variant per
configured operation, with before/after comparison images. The run includes
`operation_metadata.json`, per-sample metadata, `transformations.csv`,
`transformations.json`, and `source_hash_verification.json`. Saved output and
comparison paths are recorded per operation, including null paths for failures.
All operations run independently on the
original source pixels, not on another operation's output. The demo hashes
sampled source files before and after and fails if any hash changes.

## Phase 3: adaptive preprocessing selection

The initial selector is a deterministic, rule-based recommender. It reuses the
Phase 1 `QualityAnalyzer` measurements and returns **one** primary operation or
`keep_original`; it does not apply the recommendation, chain operations, run
OCR, or infer a domain with a classifier.

Modes:

- `automatic`: assesses the image and reads a domain label only from optional
  metadata (`domain`, `source_domain`, or `domain_labels`). If no supported
  label is supplied, the recorded domain is `unknown`.
- `user_guided`: requires a domain supplied by the user. Image quality is still
  measured independently. The selected domain's configurable candidate
  priorities only break ties among candidates already supported by measured
  signals.

Rules and provisional thresholds are in `configs/preprocessing.yaml` under
`adaptive_selection`. Current starting values are: global grayscale standard
deviation below 30 for a CLAHE candidate; high-frequency residual at or above
10 and confidence at least 0.2 for median/bilateral candidates; variance of
Laplacian below 100 to flag possible blur (Gaussian smoothing is explicitly
rejected and unsharp masking is disabled); spatial illumination variation at
or above 18 is reported but not corrected; and deskew is considered only at an
estimated angle of at least 1 degree and confidence at least 0.7. These values
are configurable, provisional indicators—not universal cutoffs—and should be
calibrated using representative data. The noise estimator can respond to text
strokes, texture, and compression artifacts, so its denoising recommendations
remain low-confidence. Thresholding operations are explicitly excluded from
automatic selection because visual validation showed speckling and possible
foreground merging.

Example CLI calls (emit JSON to stdout; optionally save to a new `--output-json`
path):

```powershell
python -m src.run_adaptive_selection --image synth6k/images/line_003728.png
python -m src.run_adaptive_selection --image synth6k/images/line_003728.png --mode user_guided --domain scene_text
```

For automatic mode with known sidecar metadata, pass a JSON object using
`--metadata-json path/to/image_metadata.json`; the selector uses only supported
domain labels present there and still measures image quality itself.

The decision records image identifier, mode and domain metadata, measured
quality values, selected operation and its configured parameters, rejected
candidates, rule reasons, uncertainty/confidence, versions, and selection
execution time. Because this command only evaluates a recommendation,
`output_path` is null and no transformation execution time is claimed. Original
image bytes are never overwritten; an optional decision JSON destination must
be separate from the input and must not already exist. Domain priorities are
provisional tie-breaks, not claims of domain-specific OCR benefit.

## Phase 3.1: batch selector validation

Run a reproducible two-mode recommendation evaluation without executing any
preprocessing operation:

```powershell
.\.venv\Scripts\python.exe -m src.run_adaptive_validation --dataset synth6k --sample-limit 200 --seed 7 --user-domain scene_text
```

The sampler stratifies by the original split and language, selects at most one
line crop per `source_image_id` to reduce source-board overlap, and preserves
the original split labels. The sample manifest records paths, identifiers,
available language/script and board metadata, and before/after source hashes.
Missing or unreadable files remain in the evaluation outputs as failures.
Automatic mode uses the configured dataset domain metadata; user-guided mode
uses the supplied domain, while both modes use the same independently measured
image-quality profile.

Each invocation creates a unique
`experiments/adaptive_selection/<dataset>/run-<timestamp>/` directory containing
the sample manifest, per-mode decision JSONL, JSON/CSV summaries, quality
measurements, visualizations, and `validation_report.md`. The configuration
hash and sample seed are recorded. Recommendations are experimental outputs
from provisional rules, not OCR performance results; timings may vary between
runs, and no source image or annotation is modified.

## Phase 4: CRNN OCR evaluation adapter

The recovered multilingual CRNN is available through the shared adapter
registry. Configuration in `configs/models.yaml` points to the immutable
recovered ZIP and names the checkpoint/vocabulary members and their SHA-256
hashes. The loader verifies both hashes, deserializes weights on CPU with
`weights_only=True`, and uses strict state-dictionary loading. A missing,
modified, or incompatible artifact fails closed.

Example:

```python
from pathlib import Path
from src.models import load_model_adapter

adapter = load_model_adapter("crnn", project_root=Path.cwd())
predictions = adapter.predict([Path("synth6k/images/line_000001.png")])
```

The shared interface accepts image paths and returns predictions in input
order. Model-native grayscale conversion, height-32 resizing, contrast
enhancement, dynamic-width batching/padding, and CTC greedy decoding remain
inside the CRNN adapter; the generic evaluation runner and CER/WER metric
function do not depend on CRNN tensor shapes or CTC. The recovered adapter is
for this verified checkpoint only; no Transformer/VLM adapter or family-wide
generalization is claimed.

Validate exact parity against the Phase 4B CPU predictions:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_crnn_adapter.py
.\.venv\Scripts\python.exe .\experiments\ocr_evaluation\recovery\run_crnn_adapter_validation.py
```

The full validation checks all 8,094 feasible synth6k test samples, metadata,
prediction order, exact prediction strings, CER, and WER. It writes the
integration report under `experiments/ocr_evaluation/recovery/`. It performs
inference only: no training, source-image changes, adaptive preprocessing, or
modification of recovered artifacts.

## Phase 5A: controlled preprocessing pilot

The frozen pilot configuration is in `configs/ocr_evaluation.yaml`. It selects
200 CTC-feasible synth6k test crops using seed 7, stratified by language and
limited to one crop per source board. It verifies source-board separation
between dataset splits before sampling. The five arms are original images,
fixed global histogram equalization, predeclared scene-text CLAHE, automatic
adaptive selection, and user-guided `scene_text` adaptive selection.

Run the pilot with:

```powershell
.\.venv\Scripts\python.exe .\experiments\ocr_evaluation\run_phase5_pilot.py
```

Each run gets a new directory under
`experiments/ocr_evaluation/phase5_pilot/`. Original-arm predictions are checked
against Phase 4B before transformed-arm inference begins. The runner preserves
the recorded Phase 4B padded-width context, applies transformations only to
separate output images, records quality/dimension/hash/latency information,
and verifies that source images, annotations, and the recovered checkpoint
archive remain unchanged. No source pixels or annotations are modified.

The 200-sample results are exploratory and not statistically conclusive. They
must not be used by themselves to claim OCR improvement or tune operations for
the full experiment.

## Phase 5B: CRNN contrast interaction ablation

The validation-only, 200-sample diagnostic is configured separately in
`configs/ocr_evaluation_phase5b.yaml` and is run with:

```powershell
.\.venv\Scripts\python.exe .\experiments\ocr_evaluation\phase5b_ablation\run_ablation.py
```

Its diagnostic adapter delegates checkpoint loading, architecture, batching,
normalization, and decoding to the frozen CRNN adapter. It can skip only the
native PIL contrast enhancement for the designated ablation arms; the
production adapter's contrast factor remains 2.0. CLAHE and histogram
equalization use their existing preprocessing-library configurations. The
run rejects train/test leakage, selects only CTC-feasible validation samples,
and records paired per-sample CER interactions and image comparisons. It does
not modify the checkpoint, source images, annotations, Phase 5A outputs, or
selector configuration.

## Phase 5C: frozen held-out evaluation

Run the final five-arm evaluation with:

```powershell
.\.venv\Scripts\python.exe .\experiments\ocr_evaluation\run_phase5_final.py
```

The runner freezes the Phase 5A dataset, preprocessing, selector, model, and
metric configuration before inference. It excludes every filename in the
Phase 5A pilot manifest and evaluates the remaining 7,894 CTC-feasible test
samples in Phase 4B order. It stops unless the population reconciles and the
original arm exactly matches the Phase 4B predictions. Every run gets a new
directory under `experiments/ocr_evaluation/phase5_final/`; the final report
is written to `final_evaluation_report.md`. Paired CER intervals use a
source-board cluster bootstrap. The complete population has multiple crops
per source board, so one-crop-per-board sampling is not feasible for this
full-remainder evaluation.

## Project layout

- `configs/`: dataset, preprocessing, and model configuration.
- `data/`: raw inputs, processed data, and dataset manifests.
- `src/`: ingestion, profiling, quality metrics, preprocessing, models,
  evaluation, and reporting modules.
- `experiments/`: experiment definitions and profiling outputs.
- `notebooks/`: exploratory analysis.
- `tests/`: automated tests.
