# RootImageAnalyzer

A multi-stage pipeline for segmenting and analyzing root systems from rhizotron photographs. Detects roots as lighter filaments against darker soil, extracts morphological traits (diameter, length, lateral count), matches comparable regions of interest across multiple plants, and supports training a custom classifier from your own annotations.

The pipeline works in two complementary detection modes — a **single-pass primary pipeline** (intensity-based) and an **ensemble parameter-sweep with pixel voting** (more robust to image variation) — both gated by a **chromaticity-based color filter** that suppresses brown soil texture. A **manual curation GUI** lets you fix any remaining errors before downstream ROI matching.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start for new image sets](#quick-start-for-new-image-sets)
- [Workflow Overview](#workflow-overview)
- [Usage](#usage)
  - [Analyze Images](#analyze-images)
  - [Ensemble mode](#ensemble-mode)
  - [Color-based root probability gate](#color-based-root-probability-gate)
  - [Manual curation GUI](#manual-curation-gui)
  - [Annotate Images](#annotate-images)
  - [Corrective Annotation](#corrective-annotation)
  - [Train the Classifier](#train-the-classifier)
  - [Benchmark the Classifier](#benchmark-the-classifier)
  - [Annotation library](#annotation-library)
  - [PRMI Training Data](#prmi-training-data)
- [Configuration file (`rhizobox_config.json`)](#configuration-file-rhizobox_configjson)
- [All Flags Reference](#all-flags-reference)
- [Output Files](#output-files)
- [Tuning Guide](#tuning-guide)
- [Troubleshooting Guide](#troubleshooting-guide)

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

The manual curation GUI (`curate_roots.py`) additionally needs an interactive matplotlib backend (TkAgg or Qt5Agg) and a working display (X11 / VNC / local).

---

## Installation

```bash
git clone https://github.com/Roli-Wilhelm/RootImageAnalyzer.git
cd RootImageAnalyzer
pip install -r requirements.txt
```

---

## Quick start for new image sets

Use this sequence the first time you point the pipeline at a new image type, soil color, or plant species. Each step uses debug outputs to validate the previous step before moving on.

1. **Inspect a single image with `--primary-only`** (it always saves the per-step debug PNGs):
   ```bash
   python rhizotron_analyzer.py --images sample_dir/ --primary-only --output output/probe
   ```
   Open `output/probe/primary_only/<stem>_s0_cropped.png`. The white frame should be cropped out and the soil interior should fill the image.
   - If real roots near the edge are missing → reduce `--frame-margin` (default 150).
   - If the frame is still visible → increase `--frame-margin`.

2. **Check the binary threshold**: open `<stem>_s3_binary.png`.
   - Should look like roots-plus-some-noise. If the entire image is white, raise `--tophat-percentile` (try 88-92).
   - If almost nothing is detected, lower it (try 75-80) or check `--tophat-radius` matches your roots (≈ 1.5× thickest root in mm).

3. **Check the color mask**: open `<stem>_color_mask.png`.
   - Roots should appear bright; soil dark. (See [Color-based root probability gate](#color-based-root-probability-gate) below.)
   - If roots are dark, lower `--color-gate` to 0.4-0.5 — or pass `--no-color-gate` if soil and roots are too similar in color.

4. **Check the gated binary**: open `<stem>_color_gated.png`.
   - Most soil texture should be removed; roots should still be present.
   - If roots have been punched into fragments, raise `--close-radius` to 4 to bridge the gaps.

5. **Run the ensemble** for production-quality skeletons:
   ```bash
   python rhizotron_analyzer.py --images sample_dir/ --ensemble --ensemble-runs 10 --output output/run1 --n-jobs 27
   ```
   Inspect `output/run1/ensemble/<stem>_votes_A.png`. Bright = consistently detected; dark = missed by all runs.

6. **Curate any remaining errors** with the GUI:
   ```bash
   python curate_roots.py --image sample_dir/<stem>.JPG --skeleton output/run1/ensemble/<stem>_ensemble_skeleton.png --output-dir curated_skeletons --operator yourname
   ```

7. **Re-run with the curated skeletons** so Stage 4 ROI matching uses your hand edits:
   ```bash
   python rhizotron_analyzer.py --images sample_dir/ --ensemble --curated-skeletons curated_skeletons --output output/run2 --n-jobs 27
   ```

The four Stage 4 outputs (`roi_coordinates.csv`, `similarity_matrix.csv`, `matched_rois_detail.csv`, `comparison_panel.png`) appear in `output/run2/`.

---

## Workflow Overview

```
testimages/          ← your rhizotron JPEGs/PNGs/TIFFs
      │
      ▼
1. (optional) --annotate    ← click-label points (root / pore_edge / background)
      │
      ▼
2. (optional) --train       ← Random Forest classifier on annotations
      │
      ▼
3. analysis pipeline        ← three modes available:
      │                         (a) --primary-only        — single fast pass + debug PNGs
      │                         (b) --ensemble            — N-run sweep + pixel voting
      │                         (c) full pipeline (default) — RF-gated, lateral counting
      │
      ▼
4. (optional) curate_roots.py    ← hand-fix individual skeletons
      │
      ▼
5. re-run with --curated-skeletons    ← Stage 4 uses your hand edits
      │
      ▼
output/
  ensemble/                       ← per-image debug PNGs and skeletons (--ensemble)
  ensemble_roots.csv              ← per-segment metrics (--ensemble)
  roi_coordinates.csv             ← Stage 4: every detected ROI's coords + features
  similarity_matrix.csv           ← Stage 4: ROI × ROI cross-plant similarity
  matched_rois_detail.csv         ← Stage 4: best matches per ROI
  comparison_panel.png            ← Stage 4: side-by-side visualization
```

---

## Usage

### Analyze Images

The default invocation runs the full pipeline (frame detection → segmentation → RF classifier gate → skeletonization → lateral root identification → ROI matching):

```bash
python rhizotron_analyzer.py --images testimages/
```

Specify image resolution explicitly (or supply a config file with physical box dimensions, see below):

```bash
python rhizotron_analyzer.py --images testimages/ --scale 13.0
```

Multi-core + debug overlays:

```bash
python rhizotron_analyzer.py --images testimages/ --n-jobs 8 --debug
```

For faster iteration on a new dataset, use `--primary-only` to bypass the classifier and lateral-counting stages and produce the per-step debug PNGs unconditionally:

```bash
python rhizotron_analyzer.py --images testimages/ --primary-only --output output/probe
```

---

### Ensemble mode

`--ensemble` runs the primary segmentation N times with different parameter combinations, then merges the results by **pixel-level voting**: a pixel appears in the final skeleton only if at least `--vote-threshold` (default 0.3) of the runs detect it. This is the recommended mode for production runs because it is robust to single-parameter sensitivity.

```bash
python rhizotron_analyzer.py --images testimages/ \
    --ensemble \
    --ensemble-runs 10 \
    --vote-threshold 0.3 \
    --n-jobs 27 \
    --output output/ensemble_run
```

**Channel A** (large roots, default): sweeps `tophat_percentile ∈ [75, 92]`, cycles `blur_sigma ∈ {1.0, 1.5, 2.0}` and `close_radius ∈ {1, 2, 3}`.

**Channel B** (fine roots, opt-in via `--fine-roots`): a separate sweep using small-kernel tophat + adaptive thresholding + Frangi vesselness. Slower but recovers thin roots that Channel A misses.

```bash
python rhizotron_analyzer.py --images testimages/ \
    --ensemble --fine-roots --fine-ensemble-runs 15 --n-jobs 27
```

After the per-image sweep finishes, the pipeline automatically extracts ROIs and runs cross-plant matching (Stage 4 of the full pipeline). The four outputs land in your `--output` directory: `roi_coordinates.csv`, `similarity_matrix.csv`, `matched_rois_detail.csv`, `comparison_panel.png`.

---

### Color-based root probability gate

The pipeline ships with a **chromaticity-suppression color gate** turned on by default at threshold 0.7. It targets roots that are *less brown* than the surrounding soil (roots are nearly neutral in LAB chromaticity; brown soil has strong positive a* and b*).

The mask is the weighted sum of two terms:

1. **Chromaticity suppression** — `local_mean(soil_score) − soil_score`, where `soil_score = 0.6·a* + 0.4·b*`. Pixels less brown than their local neighbourhood score high.
2. **Local lightness contrast** — `L − local_mean(L)`. Bright spots relative to local soil score high.

A specular-glare hard-exclusion zeroes pixels where R, G, AND B all exceed the 98th percentile of interior values.

The mask is multiplied with the binary tophat output as an AND gate: a pixel is kept only if it passes both the intensity threshold AND the color check.

**Disable for grayscale or color-similar images:**
```bash
python rhizotron_analyzer.py --images testimages/ --ensemble --no-color-gate
```

**Tune the threshold:**
```bash
python rhizotron_analyzer.py --images testimages/ --ensemble --color-gate 0.5
```

Diagnostic outputs (always produced in `--primary-only`):

| File | What it shows |
|---|---|
| `<stem>_lab_a_channel.png` | Raw LAB a* — soil should be bright, roots dark |
| `<stem>_lab_b_channel.png` | Raw LAB b* — should look similar to a* |
| `<stem>_color_prob_raw.png` | Combined probability before glare exclusion |
| `<stem>_color_mask.png` | Final probability after glare suppression |
| `<stem>_color_gated.png` | Binary mask after `binary AND (color_mask > color_gate)` |

---

### Manual curation GUI

`curate_roots.py` opens a full-screen matplotlib window showing the original color image with the automated skeleton overlaid in cyan. You can erase / draw / box-select to fix any remaining errors before ROI matching.

```bash
python curate_roots.py \
    --image realimages/DSC_0028.JPG \
    --skeleton output/realrun_ensemble/ensemble/DSC_0028_ensemble_skeleton.png \
    --output-dir curated_skeletons \
    --operator yourname
```

**Keybindings:**

| Key | Action |
|-----|--------|
| `e` | **Erase** mode (default) — circular brush; left-click-drag to erase |
| `b` | **Box-erase** mode — drag a rectangle, everything inside is erased on release |
| `d` | **Draw** mode — circular brush; new strokes are auto-thinned to 1-px width |
| `[` / `]` | Shrink / grow brush radius (e/d modes) |
| `u` | Undo last stroke (history depth ≥ 20) |
| `+` / `-` | Zoom in / out at the cursor |
| `0` | Reset zoom |
| `g` | Toggle ROI grid overlay |
| `c` | Toggle skeleton overlay (peek at the underlying image) |
| `s` | Save curated PNG + timestamped backup + JSON log; banner confirms on screen |
| `q` | Save (with confirmation prompt) and quit |
| `h` | Print full keybindings to terminal |

The cursor changes per mode: solid green ring + pencil tip icon for draw, dashed red ring + eraser block icon for erase, dotted red ring for box.

**Saved artifacts:**

- `<stem>_curated_skeleton.png` — binary PNG, same H×W as the cropped interior, drop-in compatible with `--curated-skeletons`.
- `<stem>_curation_log.json` — operator, timestamp, pixels erased, pixels drawn, undos, original skeleton path, session duration.
- `curation_backups/<stem>_curated_skeleton_<timestamp>.png` — every save also writes a timestamped backup.

**Use the curated skeletons in the analyzer:**

```bash
python rhizotron_analyzer.py --images realimages/ \
    --ensemble \
    --curated-skeletons curated_skeletons/ \
    --output output/curated_run
```

At startup the analyzer prints `Curated skeletons: ON — using curated skeleton for K/N images`. For images with a curated skeleton, the parameter sweep is bypassed entirely (fast); for the rest, the normal ensemble runs. Stage 4 ROI matching then runs on the merged set.

**Notes:**
- Works over X11 forwarding (`ssh -X` / RealVNC). Pass `--backend Qt5Agg` if TkAgg is unavailable.
- Live brush feedback is throttled to ~5 fps over remote displays; commits update fully on mouse release.
- The cursor circle always shows the brush coverage; the small icon to the upper-right of the cursor reinforces the mode.

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

### Annotation library

The library is a persistent store of annotation sessions across runs. Every `--annotate`, `--correct`, and `--train` invocation can append to or read from it.

| Flag | Default | Effect |
|---|---|---|
| `--use-library` | off | During `--train`, pool every archived session in addition to the current `annotations/` directory. |
| `--list-library` | off | Print a summary of all archived sessions and exit. Useful audit. |
| `--library-path DIR` | `library/` (relative to the script) | Override the default library location. |
| `--operator NAME` | `unknown` | Saved into the session metadata. |
| `--notes TEXT` | — | Free-text note saved into the session metadata. |

Library entries are immutable once written; archive growth is append-only.

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

## Configuration file (`rhizobox_config.json`)

A small JSON file lets you keep per-rhizotron physical dimensions out of the command line. Pass it with `--config rhizobox_config.json`. CLI flags override config values when both are provided.

**Supported fields:**

```json
{
  "box_width_mm":   305.0,
  "box_height_mm":  205.0,
  "roi_width_mm":   30.0,
  "roi_height_mm":  30.0,
  "scale_px_per_mm": null,
  "roi_size_px":    300,
  "frame_margin":   150
}
```

| Field | Type | Default | Effect |
|---|---|---|---|
| `box_width_mm` | float | — | Physical width of the soil-viewable rhizobox window in mm. Used with `box_height_mm` to derive `scale_px_per_mm` from the detected interior pixel dimensions. |
| `box_height_mm` | float | — | Physical height. |
| `scale_px_per_mm` | float / null | `10.0` | Direct override; when null and box dimensions are present, the scale is derived. |
| `roi_width_mm` | float | — | ROI window width in mm. Converted to pixels using the derived scale; overrides `--roi-size`. |
| `roi_height_mm` | float | — | ROI window height in mm. Pair with `roi_width_mm`. |
| `roi_size_px` | int | `300` | Used by `curate_roots.py`'s ROI-grid overlay. |
| `frame_margin` | int | `150` | Pixels cropped from each edge before analysis. |

YAML format is also accepted if PyYAML is installed.

**Example invocation:**

```bash
python rhizotron_analyzer.py --images testimages/ --ensemble --config rhizobox_config.json
```

---

## All Flags Reference

### Core Pipeline

| Flag | Default | Description |
|------|---------|-------------|
| `--images DIR` | *(required)* | Directory of input images (JPEG / PNG / TIFF) |
| `--scale PX_PER_MM` | `10.0` | Image resolution — calibrate from a ruler in-frame, or use `--config` |
| `--config FILE` | — | JSON / YAML with physical box dimensions; CLI flags override |
| `--bins MM [MM ...]` | `0.5 1.0 2.0` | Diameter class boundaries in mm |
| `--metric` | `cosine` | Similarity metric for ROI matching (`cosine` or `euclidean`) |
| `--roi-size PX` | `300` | Sliding-window ROI side length |
| `--roi-stride PX` | `roi-size // 2` | Stride between consecutive ROIs |
| `--downsample N` | `1` | Process at 1/N resolution (2 = 4× faster) |
| `--n-jobs N` | `cpu_count - 1` | Parallel worker processes |
| `--output DIR` | `output` | Directory for all output files |
| `--debug` | off | Save intermediate segmentation overlays (`output/debug/`) |

### Mode selection

| Flag | Default | Description |
|------|---------|-------------|
| `--primary-only` / `--complexity 1` | off | Fast single-pass tophat → percentile threshold → skeleton; saves all per-step debug PNGs |
| `--ensemble` | off | N-run parameter sweep with pixel voting (recommended for production) |
| `--complexity {1,2,3,4}` | `4` | 1 = primary only; 2 = + diameter; 3 = + conservative laterals; 4 = full pipeline |
| `--curated-skeletons DIR` | — | Use hand-curated skeletons from `curate_roots.py` instead of running detection |

### Color gate (chromaticity-suppression)

| Flag | Default | Description |
|------|---------|-------------|
| `--color-gate PROB` | `0.7` | **ON by default.** Probability threshold; pixel kept only if `color_mask > PROB` |
| `--no-color-gate` | off | Disable the color gate entirely (grayscale / color-similar images) |
| `--local-soil-radius SIGMA` | `20.0` | Gaussian sigma for the local soil-chromaticity neighbourhood |
| `--local-lightness-radius SIGMA` | `15.0` | Gaussian sigma for the secondary lightness contrast |
| `--chroma-weight W` | `0.7` | Weight of the chromaticity term (lightness gets `1−W`) |
| `--glare-percentile PCT` | `98.0` | Hard-exclude pixels where R, G, AND B exceed this percentile |

### Ensemble mode

| Flag | Default | Description |
|------|---------|-------------|
| `--ensemble-runs N` | `10` | Channel A parameter-sweep runs |
| `--vote-threshold FRAC` | `0.3` | Fraction of runs that must agree for a pixel to appear in the merged skeleton |
| `--ensemble-seed N` | `42` | Reproducibility |
| `--save-individual-runs` | off | Save each sweep run's skeleton as a separate PNG |
| `--fine-roots` | off | Enable Channel B (fine-root detection: small tophat + Frangi) |
| `--fine-ensemble-runs N` | `15` (when on) | Channel B sweep runs |
| `--vote-threshold-b FRAC` | `0.4` | Channel B vote threshold (held stricter than A) |
| `--fine-root-weight W` | `0.7` | Channel B weight in the combined heatmap display |
| `--ensemble-min-roi-density FRAC` | `0.002` | Stage 4 — minimum dilated-skeleton fraction in ROI window |
| `--ensemble-min-skeleton-density FRAC` | `0.005` | Stage 4 — minimum skeleton density after per-ROI re-thinning |
| `--ensemble-dilate-skeleton PX` | `2` | Stage 4 — dilation radius before density pre-filter |

### Segmentation tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--tophat-radius MM` | `2.5` | Top-hat structuring element radius — set to ~1.5× thickest root |
| `--tophat-percentile PCT` | `85.0` | Threshold percentile in `--primary-only` / Channel A baseline |
| `--blur-sigma SIGMA` | `1.5` | Gaussian sigma applied between tophat and threshold |
| `--close-radius PX` | `2` | Morphological closing disk radius |
| `--frame-margin PX` | `150` | Hard inset from each edge to suppress the frame |
| `--min-segment-length PX` | `30` | Discard skeleton segments shorter than this |
| `--min-aspect-ratio RATIO` | `3.0` | Remove round blobs below this major/minor axis ratio |
| `--max-root-diameter MM` | `3.0` | Remove blobs wider than this (soil aggregates) |
| `--vesselness-threshold SCORE` | `0.01` | Frangi filter cutoff; set to `0` to disable |
| `--min-roi-density FRAC` | `0.02` | Minimum skeleton density for ROI to enter matching |
| `--border-margin PX` | `100` | Suppress frame-edge artifacts in ROI matching |
| `--min-component-area PX²` | `500` | Remove small mask components before skeletonization |
| `--pre-skeleton-threshold PROB` | `0.65` | Classifier gate applied to mask before skeletonization |
| `--remove-loops` | off | Strip closed-ring mask components |

### Lateral root controls

| Flag | Default | Description |
|------|---------|-------------|
| `--prune-length PX` | `50` | Remove terminal skeleton stubs shorter than this |
| `--prune-passes N` | `3` | Rounds of stub pruning |
| `--min-lateral-length PX` | `60` | Minimum length to count a branch as a lateral |
| `--min-lateral-angle DEG` | `30` | Minimum emergence angle from parent segment |
| `--max-lateral-angle DEG` | `150` | Maximum emergence angle from parent segment |
| `--min-lateral-persistence PX` | `40` | Lateral must travel this far before re-branching |
| `--max-diameter-cv CV` | `0.4` | Max coefficient of variation of diameter along a lateral |
| `--lateral-classifier-threshold PROB` | `0.7` | Stricter P(root) gate applied only to lateral candidates |
| `--max-lateral-density PER_CM` | `2.0` | Cap on laterals per cm of parent root length |
| `--conservative` | off | Apply all precision-over-recall lateral parameters at once |
| `--primary-prune-length PX` | `20` | Stub pruning length in `--primary-only` / `--ensemble` |
| `--primary-prune-passes N` | `2` | Stub pruning passes in `--primary-only` / `--ensemble` |

### Classifier

| Flag | Default | Description |
|------|---------|-------------|
| `--train` | off | Fit classifier from annotations; save to `models/root_classifier.joblib` |
| `--no-classifier` | off | Disable the RF post-filter even if a model file exists |
| `--classifier-path PATH` | `models/root_classifier.joblib` | Path to saved model |
| `--classifier-threshold PROB` | `0.6` | Minimum P(root) to retain a skeleton component |
| `--max-loop-size PX` | `200` | Remove closed skeleton loops shorter than this unconditionally |
| `--no-augment` | off | Disable training augmentation |
| `--external-features DIR` | — | Directory of `.npz` feature arrays from PRMI conversion |
| `--source-weight W` | `1.0` | Weight multiplier for external features |

### Annotation

| Flag | Default | Description |
|------|---------|-------------|
| `--annotate` | off | Launch interactive point-label annotation window |
| `--correct` | off | Launch brush-based corrective annotation |
| `--brush-radius PX` | `8` | Initial brush radius for `--correct` mode |
| `--benchmark` | off | Print precision/recall/F1 report against annotated points |
| `--target-recall FRAC` | `0.80` | Recall level for the precision@recall metric |
| `--annotation-dir DIR` | `annotations` | Directory for annotation JSON files |
| `--patch-size PX` | *(see source)* | Annotated patch size in `--annotate` mode |

### Annotation library

| Flag | Default | Description |
|------|---------|-------------|
| `--use-library` | off | Pool all archived annotation sessions during `--train` |
| `--list-library` | off | Print summary of all archived sessions and exit |
| `--library-path DIR` | *(see source)* | Root directory of the persistent annotation library |
| `--operator NAME` | `unknown` | Recorded in library metadata |
| `--notes TEXT` | — | Free-text note stored in library metadata |

---

## Output Files

```
output/
├── ensemble/                      # (--ensemble) per-image debug + skeletons
│   ├── <stem>_ensemble_skeleton.png  # Final binary skeleton (curate_roots input)
│   ├── <stem>_votes_A.png            # Channel A vote heatmap
│   ├── <stem>_votes_B.png            # Channel B (--fine-roots only)
│   ├── <stem>_votes_combined.png     # Combined heatmap (--fine-roots only)
│   └── <stem>_final_primary.png      # Color-coded skeleton overlay
├── primary_only/                  # (--primary-only) per-step debug PNGs
│   ├── <stem>_s0_cropped.png         # After frame crop
│   ├── <stem>_s1_tophat.png          # Top-hat enhancement
│   ├── <stem>_s2_blurred.png         # After Gaussian blur
│   ├── <stem>_s3_binary.png          # After percentile threshold
│   ├── <stem>_lab_a_channel.png      # LAB a* channel (color-mask diagnostic)
│   ├── <stem>_lab_b_channel.png      # LAB b* channel
│   ├── <stem>_color_prob_raw.png     # Combined color probability before glare
│   ├── <stem>_color_mask.png         # Final color probability
│   ├── <stem>_color_gated.png        # Binary AND color_mask (when gate enabled)
│   ├── <stem>_s4_closed.png          # After morphological closing
│   ├── <stem>_s6_skel_raw.png        # Raw skeleton
│   ├── <stem>_s7_skel_pruned.png     # After stub pruning + length filter
│   └── <stem>_primary.png            # Cyan-skeleton-on-grayscale final overlay
├── ensemble_roots.csv             # Per-segment metrics (ensemble mode)
├── primary_roots.csv              # Per-segment metrics (primary-only mode)
├── roots_summary.csv              # Per-image traits (full pipeline)
├── roi_coordinates.csv            # Stage 4: every ROI's coords + features
├── similarity_matrix.csv          # Stage 4: ROI × ROI cross-plant similarity
├── matched_rois_detail.csv        # Stage 4: best-match per ROI with scores
├── comparison_panel.png           # Stage 4: side-by-side visualization
└── debug/                         # (--debug only, full pipeline)
```

---

## Tuning Guide

**Scale is the most important parameter.** Measure a known object in pixels (ruler, frame dimension) for `--scale`, or supply `box_width_mm` / `box_height_mm` via `--config` for automatic derivation.

```
Typical values:
  CI-600 minirhizotron scanner   ≈ 13 px/mm
  Pixel 6a at ~30 cm distance    ≈ 10 px/mm
  DSLR at 1:1 macro              varies — calibrate per setup
```

**If the pipeline retains too much noise** (soil texture, pore edges):
- Raise `--color-gate` (try 0.8-0.9)
- Raise `--classifier-threshold` (try 0.7-0.8)
- Raise `--pre-skeleton-threshold` (try 0.7)
- Raise `--min-aspect-ratio` (try 4.0-5.0)
- Raise `--min-segment-length`
- Raise `--vote-threshold` in ensemble mode (try 0.5)

**If the pipeline misses too many roots:**
- Lower `--color-gate` (try 0.5) or pass `--no-color-gate`
- Lower `--classifier-threshold` (try 0.4-0.5)
- Lower `--tophat-radius` toward your thinnest root diameter
- Lower `--min-segment-length`
- Add `--fine-roots` for thin-root detection in ensemble mode

**Color mask too aggressive on certain soils** (sandy / very dry / non-iron-oxide soils):
- Try `--no-color-gate` first to confirm the gate is the cause
- If the gate is needed but too strict, lower `--color-gate` to 0.4-0.5
- Increase `--local-soil-radius` to 30-40 if your soil color is uniform across wide patches

**For lateral root counting precision:**
- Use `--conservative` as a starting point
- Increase `--min-lateral-length` and `--min-lateral-persistence` for stricter laterals
- Adjust `--min-lateral-angle` / `--max-lateral-angle` to your species' architecture

**Performance:**
- `--n-jobs` defaults to `cpu_count - 1`; each worker uses ~40 MB at full resolution
- `--downsample 2` gives ~4× speedup with moderate accuracy trade-off
- Ensemble runtime is roughly linear in `--ensemble-runs`; 10 runs at full resolution on a 17-image set ≈ 4 hours on 27 cores
- For iteration on a new dataset, start with `--primary-only` (single pass) and only switch to `--ensemble` once the single-pass parameters look right

All `# TUNE:` comments in the source code contain authoritative guidance for individual parameters. Search for `# TUNE:` in `rhizotron_analyzer.py` to find the in-source documentation.

---

## Troubleshooting Guide

This section is written to help both human users and AI coding assistants (e.g. Claude Code, Copilot) diagnose and fix common problems. Each symptom includes the most likely cause, which parameter to change, and how to verify the fix using debug outputs.

### Quick diagnosis workflow

Always run with `--primary-only --debug` first on a single image. The per-step debug images reveal exactly where the pipeline succeeds or fails:

| Debug image | Diagnostic value |
|---|---|
| `_s0_cropped.png` | If real roots near the edge are missing here, `--frame-margin` is too large |
| `_s1_tophat.png` | If roots are dark / invisible, `--tophat-radius` doesn't match root width |
| `_s3_binary.png` | If the entire image is white, `--tophat-percentile` is too low; if mostly black, too high |
| `_lab_a_channel.png` | Brown soil should be bright, roots dark. If they look identical, the LAB conversion is fine but soil and root colors are too similar — disable the color gate |
| `_color_mask.png` | Roots should appear bright, soil dark. If roots are dark here, the color gate will remove them — lower `--color-gate` |
| `_color_gated.png` | If roots disappear between `_s3_binary` and here, `--color-gate` is too strict |
| `_s4_closed.png` | If skeleton is fragmented, raise `--close-radius` (especially after color gating) |
| `_s6_skel_raw.png` vs `_s7_skel_pruned.png` | If real roots vanish between these, pruning is too aggressive |
| `_votes_A.png` (ensemble) | Bright = consistently detected; dark = missed by all runs. Adjust `--tophat-radius` / `--blur-sigma` ranges if mostly dark |

### Symptom table

| Symptom | Most likely cause | First thing to try |
|---|---|---|
| Traces cover entire image in dense network | Threshold too low | Raise `--tophat-percentile` to 88-92 |
| Large primary roots missing entirely | Frame crop removing them OR size filter too strict | Check `_s0_cropped.png`; lower `--frame-margin` or `--min-segment-length` |
| Color gate removes real roots | Color gate too strict for this soil/root type | Lower `--color-gate` to 0.5 or use `--no-color-gate` |
| Color mask shows roots and soil identically bright/dark | Soil and root colors are too similar (e.g. dark or stained roots in dark soil) | Use `--no-color-gate` |
| Ensemble votes heatmap is all dark | Tophat kernel wrong size for image resolution | Adjust `--tophat-radius` relative to root width in pixels |
| Skeleton fragmented into many short segments after color gate | Gate punches holes that closing can't bridge | Raise `--close-radius` to 4 |
| Central glass-smear / glare zone detected as dense roots | Diffuse glare not caught by per-channel percentile rule | Lower `--glare-percentile` to 92, or curate manually with `curate_roots.py` |
| ROI matching returns "No match" | ROI density filter too strict OR all ROIs are similar | Lower `--ensemble-min-roi-density` to 0.001; check `similarity_matrix.csv` for the actual scores |
| Stage 4 "Skipped — need ≥2 successfully processed images" | Only one image was processed, or all but one failed | Run with `--primary-only` first to verify each image processes individually |
| Curated skeleton not being used | Wrong directory passed to `--curated-skeletons` | Verify filename matches `<imagename>_curated_skeleton.png` exactly |
| Curated skeleton shape mismatch warning | `--scale` differs between curation and analysis | Use the same `--scale` (or `--config`) for both runs |
| Script runs but produces no output files | Output directory permissions OR ROI step silently skipped | Check terminal output for warnings; verify `--output` directory is writable |
| Very different results across ensemble runs | Parameter sweep range too wide for this image type | Run `--primary-only` first to identify a sensible center value, then narrow the sweep |
| `curate_roots.py` mouse drag does nothing | Matplotlib navigation toolbar intercepting events | Already disabled in current version; if it returns, restart and verify TkAgg/Qt5Agg backend |
| `curate_roots.py` window doesn't appear over SSH | Missing X11 forwarding | `ssh -X your-host`; `echo $DISPLAY` should be non-empty |
| `curate_roots.py` slow over VNC | Full-resolution overlay redraw on every motion event | Already throttled to ~5 fps in current version; further drop with smaller display window |

### For AI coding assistants troubleshooting this codebase

If you are an AI assistant helping a user debug this pipeline, start here:

1. **Always start with `--primary-only`.** It saves all per-step debug PNGs unconditionally. Ask the user to share `_s3_binary.png` and `_color_mask.png` first — these two diagnose ~80% of detection problems.

2. **The pipeline has these distinct failure modes, in order of frequency:**
   1. **Color gate too strict** — symptom: `_color_gated.png` is much sparser than `_s3_binary.png`. Fix: lower `--color-gate` or pass `--no-color-gate`.
   2. **Tophat percentile wrong for image brightness** — symptom: `_s3_binary.png` is either entirely white or mostly black. Fix: adjust `--tophat-percentile` 75-92.
   3. **Frame crop eating real roots** — symptom: `_s0_cropped.png` shows a black border crossing real roots. Fix: lower `--frame-margin`.
   4. **Pruning too aggressive** — symptom: `_s7_skel_pruned.png` has many fewer segments than `_s6_skel_raw.png`. Fix: lower `--prune-length` / `--prune-passes` (or `--primary-prune-length` / `--primary-prune-passes` in `--primary-only` / `--ensemble`).
   5. **ROI density threshold rejecting all candidates** — symptom: `Stage 4 ... Aborted — not enough ROIs across images`. Fix: lower `--ensemble-min-roi-density` (try `0.001`).

3. **Parameters that interact (tune together):**
   - `tophat_percentile` and `close_radius` — stricter threshold needs larger closing.
   - `color_gate` and `close_radius` — stricter gate fragments skeletons; raise closing to compensate.
   - `vote_threshold` and `ensemble_runs` — more runs allow finer-grained voting.
   - `prune_length` and `prune_passes` — multiplicative effect on total stub removal.
   - `local_soil_radius` and `chroma_weight` — both affect color-mask aggressiveness.

4. **Independent parameters** (tune in isolation):
   - `frame_margin` — purely geometric.
   - `min_segment_length` — final length filter.
   - `glare_percentile` — only affects pixels at the very top of the brightness distribution.

5. **The `rhizobox_config.json` file sets physical dimensions** — if ROI sizes seem wrong, check this file first. CLI flags override config values when both are provided.

6. **All `# TUNE:` comments in the source code contain authoritative per-parameter guidance.** Each one specifies symptoms for "increase if" / "decrease if" / "values to avoid" / "interacts with". `grep "# TUNE:" rhizotron_analyzer.py` lists them all.

7. **The pipeline supports three independent skeleton-detection paths** that can be combined via `--curated-skeletons`:
   - Automated: `--ensemble` produces `_ensemble_skeleton.png` per image.
   - Manual: `curate_roots.py` produces `_curated_skeleton.png` per image.
   - Override: pass `--curated-skeletons DIR` to the analyzer; for any image with a curated file, the automated detection is bypassed.
   This means a user can hand-fix only the few images that fail automated detection, and the rest still run through the full pipeline automatically.

8. **Key code entry points:**
   - Single-pass primary segmentation: `PrimaryOnlyPipeline._process_image` in `rhizotron_analyzer.py`.
   - Ensemble worker (Channel A): `_run_primary_for_ensemble` (module-level function for multiprocessing).
   - Ensemble worker (Channel B): `_run_fine_for_ensemble`.
   - Ensemble orchestrator: `EnsemblePipeline.run` and `EnsemblePipeline._process_image`.
   - Color mask: `compute_color_mask` (module-level).
   - Stage 4 (cross-image ROI matching): `EnsemblePipeline._run_stage4` and `match_rois_across_plants`.
   - Curation GUI: `CurationApp` class in `curate_roots.py`.
