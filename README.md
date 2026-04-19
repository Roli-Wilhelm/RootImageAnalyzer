# RootImageAnalyzer

A multi-stage pipeline for segmenting and analyzing root systems from rhizotron photographs. Detects roots as lighter filaments against darker soil, extracts morphological traits (diameter, length, lateral count), matches comparable regions of interest across multiple plants, and supports training a custom classifier from your own annotations.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Workflow Overview](#workflow-overview)
- [Usage](#usage)
  - [Analyze Images](#analyze-images)
  - [Annotate Images](#annotate-images)
  - [Corrective Annotation](#corrective-annotation)
  - [Train the Classifier](#train-the-classifier)
  - [Benchmark the Classifier](#benchmark-the-classifier)
  - [PRMI Training Data](#prmi-training-data)
- [All Flags Reference](#all-flags-reference)
- [Output Files](#output-files)
- [Tuning Guide](#tuning-guide)

---

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

```
opencv-python-headless
scikit-image
scikit-learn
scipy
numpy
matplotlib
pandas
pillow
joblib
```

---

## Installation

```bash
git clone https://github.com/Roli-Wilhelm/RootImageAnalyzer.git
cd RootImageAnalyzer
pip install -r requirements.txt
```

---

## Workflow Overview

```
testimages/          ← your rhizotron JPEGs/PNGs/TIFFs
      │
      ▼
1. --annotate        ← click-label points on images (root / pore_edge / background)
      │
      ▼
2. --train           ← fit Random Forest classifier on your annotations
      │                 (optionally augmented + weighted with PRMI external data)
      ▼
3. --correct         ← brush-paint corrections on top of segmentation results
      │
      ▼
4. [run pipeline]    ← segment roots, extract morphology, match ROIs across plants
      │
      ▼
output/
  roots_summary.csv  ← per-image root traits
  roi_matches.csv    ← cross-plant region-of-interest matches
  debug/             ← intermediate segmentation overlays (with --debug)
```

---

## Usage

### Analyze Images

Run the full segmentation and ROI-matching pipeline on a directory of images:

```bash
python rhizotron_analyzer.py --images testimages/
```

Specify your camera resolution (critical for diameter and length accuracy):

```bash
python rhizotron_analyzer.py --images testimages/ --scale 13.0
```

Use multiple CPU cores and save debug overlays:

```bash
python rhizotron_analyzer.py --images testimages/ --n-jobs 8 --debug
```

---

### Annotate Images

Launch an interactive window to label points on each image. Click to place labels; labels are saved automatically.

```bash
python rhizotron_analyzer.py --images testimages/ --annotate
```

**Controls:**

| Key | Action |
|-----|--------|
| `1` | Label point as **root** |
| `2` | Label point as **pore edge** |
| `3` | Label point as **background** |
| `z` | Undo last point |
| `n` | Next image |
| `q` | Save and quit |

Annotations are saved to `annotations/<stem>.json`. Re-running `--annotate` on the same image appends to existing labels.

---

### Corrective Annotation

After running the pipeline, paint corrections directly on the segmentation overlay:

```bash
python rhizotron_analyzer.py --images testimages/ --correct
```

**Controls:**

| Action | Result |
|--------|--------|
| Left-drag | Paint **root** |
| Middle-drag | Paint **pore edge** |
| Right-drag | Paint **background** |
| `+` / `-` | Increase / decrease brush radius |
| `z` | Undo last stroke |
| `c` | Clear all corrections |
| `q` | Save and quit |

Set the initial brush radius:

```bash
python rhizotron_analyzer.py --images testimages/ --correct --brush-radius 12
```

---

### Train the Classifier

Train a Random Forest classifier from your annotations:

```bash
python rhizotron_analyzer.py --images testimages/ --train
```

**With data augmentation** (on by default — applies H-flip, V-flip, 90° rotation, Gaussian blur):

```bash
python rhizotron_analyzer.py --images testimages/ --train
# disable augmentation:
python rhizotron_analyzer.py --images testimages/ --train --no-augment
```

**With external PRMI training data** (see [PRMI Training Data](#prmi-training-data)):

```bash
python rhizotron_analyzer.py --images testimages/ --train \
    --external-features external_features/prmi \
    --source-weight 0.5
```

`--source-weight` controls how much PRMI samples count relative to your own annotations. Your annotations always have weight `1.0`; set values below `1.0` to downweight domain-shifted external data.

**Pool annotations from the persistent library** across multiple sessions:

```bash
python rhizotron_analyzer.py --images testimages/ --train --use-library
```

The trained model is saved to `models/root_classifier.joblib`.

---

### Benchmark the Classifier

Evaluate the trained classifier against your annotated points:

```bash
python rhizotron_analyzer.py --images testimages/ --benchmark
```

Prints a precision / recall / F1 report per class, plus **precision at 80% recall** for the root class. Change the recall target:

```bash
python rhizotron_analyzer.py --images testimages/ --benchmark --target-recall 0.90
```

---

### PRMI Training Data

The [PRMI dataset](https://datadryad.org/dataset/doi:10.5061/dryad.2v6wwpzp4) provides 72,000 minirhizotron images with pixel-level masks across six plant species. Use it to pre-train the classifier before annotating your own data.

**Step 1 — Check what's available (no download):**

```bash
python download_training_data.py --list-only
```

**Step 2 — Download and convert** (~9.3 GB, resumable):

```bash
python download_training_data.py
```

> **Note:** Dryad requires a free account to download files. If you get a `401` error, download `PRMI_official.zip` manually from the Dryad page, place it in `data/external/prmi/`, and re-run with:
>
> ```bash
> python download_training_data.py --skip-download
> ```

**Quick test with a subset of images:**

```bash
python download_training_data.py --skip-download --max-images 500 --n-jobs 8
```

**Step 3 — Train with the converted features:**

```bash
python rhizotron_analyzer.py --images testimages/ --train \
    --external-features external_features/prmi \
    --source-weight 0.5
```

---

## All Flags Reference

### Core Pipeline

| Flag | Default | Description |
|------|---------|-------------|
| `--images DIR` | *(required)* | Directory of input images (JPEG / PNG / TIFF) |
| `--scale PX_PER_MM` | `10.0` | Image resolution — calibrate from a ruler in-frame |
| `--bins MM [MM ...]` | `0.5 1.0 2.0` | Diameter class boundaries in mm |
| `--metric` | `cosine` | Similarity metric for ROI matching (`cosine` or `euclidean`) |
| `--roi-size PX` | `300` | Sliding-window ROI side length |
| `--roi-stride PX` | `roi-size // 2` | Stride between consecutive ROIs |
| `--downsample N` | `1` | Process at 1/N resolution (2 = 4× faster) |
| `--n-jobs N` | `cpu_count - 1` | Parallel worker processes |
| `--output DIR` | `output` | Directory for all output files |
| `--debug` | off | Save intermediate segmentation overlays to `output/debug/` |

### Segmentation Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--tophat-radius MM` | `2.5` | Top-hat filter radius — set to ~1.5× your thickest root |
| `--min-primary-diameter MM` | `0.5` | Diameter threshold separating primary from lateral roots |
| `--min-segment-length PX` | `30` | Discard skeleton segments shorter than this |
| `--min-aspect-ratio RATIO` | `3.0` | Remove round blobs below this major/minor axis ratio |
| `--max-root-diameter MM` | `3.0` | Remove blobs wider than this (soil aggregates) |
| `--vesselness-threshold SCORE` | `0.01` | Frangi filter cutoff; set to `0` to disable |
| `--min-roi-density FRAC` | `0.02` | Minimum skeleton density for ROI to enter matching |
| `--border-margin PX` | `100` | Suppress frame-edge artifacts in ROI matching |
| `--min-component-area PX²` | `500` | Remove small mask components before skeletonization |
| `--pre-skeleton-threshold PROB` | `0.65` | Classifier gate applied to mask before skeletonization |
| `--remove-loops` | off | Strip closed-ring mask components (useful only for sparse images) |

### Lateral Root Controls

| Flag | Default | Description |
|------|---------|-------------|
| `--prune-length PX` | `50` | Remove terminal skeleton stubs shorter than this |
| `--prune-passes N` | `3` | Rounds of stub pruning before lateral counting |
| `--min-lateral-length PX` | `60` | Minimum length to count a branch as a lateral root |
| `--min-lateral-angle DEG` | `30` | Minimum emergence angle from parent segment |
| `--max-lateral-angle DEG` | `150` | Maximum emergence angle from parent segment |
| `--min-lateral-persistence PX` | `40` | Lateral must travel this far before re-branching |
| `--max-diameter-cv CV` | `0.4` | Max coefficient of variation of diameter along a lateral |
| `--lateral-classifier-threshold PROB` | `0.7` | Stricter P(root) gate applied only to lateral candidates |
| `--max-lateral-density PER_CM` | `2.0` | Cap on laterals per cm of parent root length |
| `--conservative` | off | Apply all precision-over-recall lateral parameters at once |

### Classifier

| Flag | Default | Description |
|------|---------|-------------|
| `--train` | off | Fit classifier from annotations; save to `models/root_classifier.joblib` |
| `--no-classifier` | off | Disable the RF post-filter even if a model file exists |
| `--classifier-path PATH` | `models/root_classifier.joblib` | Path to saved model |
| `--classifier-threshold PROB` | `0.6` | Minimum P(root) to retain a skeleton component |
| `--max-loop-size PX` | `200` | Remove closed skeleton loops shorter than this unconditionally |
| `--no-augment` | off | Disable training augmentation (H-flip, V-flip, rotation, blur) |
| `--external-features DIR` | — | Directory of `.npz` feature arrays from PRMI conversion |
| `--source-weight W` | `1.0` | Weight multiplier for external features (own annotations = 1.0) |

### Annotation

| Flag | Default | Description |
|------|---------|-------------|
| `--annotate` | off | Launch interactive point-label annotation window |
| `--correct` | off | Launch brush-based corrective annotation on segmentation overlay |
| `--brush-radius PX` | `8` | Initial brush radius for `--correct` mode |
| `--benchmark` | off | Print precision/recall/F1 report against annotated points |
| `--target-recall FRAC` | `0.80` | Recall level for the precision@recall metric in benchmark |
| `--annotation-dir DIR` | `annotations` | Directory for annotation JSON files |
| `--patch-size PX` | *(see source)* | Annotated patch size in `--annotate` mode |

### Annotation Library

| Flag | Default | Description |
|------|---------|-------------|
| `--use-library` | off | Pool all archived annotation sessions during `--train` |
| `--list-library` | off | Print summary of all archived sessions and exit |
| `--library-path DIR` | *(see source)* | Root directory of the persistent annotation library |
| `--operator NAME` | `unknown` | Your name, recorded in library metadata |
| `--notes TEXT` | — | Free-text note stored in library metadata |

---

## Output Files

```
output/
├── roots_summary.csv      # Per-image: total length, diameter classes, lateral count, etc.
├── roi_matches.csv        # Cross-plant ROI similarity scores and matched coordinates
└── debug/                 # (--debug only)
    ├── <stem>_tophat.png      # Raw top-hat enhancement
    ├── <stem>_filtered.png    # After shape filtering
    └── <stem>_skeleton.png    # Final mask with skeleton overlay
```

---

## Tuning Guide

**Scale is the most important parameter.** Measure a known object (ruler, frame dimension) in pixels to get the correct `--scale` value.

```
Typical values:
  CI-600 minirhizotron scanner   ≈ 13 px/mm
  DSLR at 1:1 macro              ≈ varies
```

**If the pipeline retains too much noise** (soil texture, pore edges):
- Raise `--classifier-threshold` (try 0.7–0.8)
- Raise `--pre-skeleton-threshold` (try 0.7)
- Raise `--min-aspect-ratio` (try 4.0–5.0)
- Raise `--min-segment-length`

**If the pipeline misses too many roots:**
- Lower `--classifier-threshold` (try 0.4–0.5)
- Lower `--tophat-radius` toward your thinnest root diameter
- Lower `--min-segment-length`

**For lateral root counting precision:**
- Use `--conservative` as a starting point
- Increase `--min-lateral-length` and `--min-lateral-persistence` for stricter laterals
- Adjust `--min-lateral-angle` / `--max-lateral-angle` to your species' architecture

**Performance:**
- `--n-jobs` defaults to `cpu_count - 1`; each worker uses ~40 MB at full resolution
- `--downsample 2` gives ~4× speedup with moderate accuracy trade-off
