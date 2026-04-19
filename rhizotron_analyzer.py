#!/usr/bin/env python3
"""
rhizotron_analyzer.py — Multi-stage pipeline for rhizotron image analysis.

Segments root systems from rhizotron photographs, extracts per-root morphological
traits, and identifies comparable regions of interest (ROIs) across multiple plants
to guide physical rhizosphere sampling and mineral analysis.

Root appearance in these images: lighter/beige filaments against darker soil,
photographed against the glass face of a rectangular rhizotron frame.

Usage:
    python rhizotron_analyzer.py --images testimages/ --bins 0.5 1 2 --metric cosine
    python rhizotron_analyzer.py --images testimages/ --scale 10.5 --roi-size 300 --metric euclidean
    python rhizotron_analyzer.py --images testimages/ --n-jobs 8 --debug
    python rhizotron_analyzer.py --help
"""

import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
# Use interactive backend only for --annotate; Agg for all other modes.
# Must be called before pyplot is imported.
if "--annotate" not in sys.argv and "--correct" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, label as nd_label
from scipy.spatial.distance import cdist
from skimage.filters import frangi
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import (
    closing,
    disk,
    remove_small_objects,
    skeletonize,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Default parameters — all exposed on the CLI or documented with # TUNE
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BINS_MM: List[float] = [0.5, 1.0, 2.0]
DEFAULT_METRIC: str = "cosine"
DEFAULT_ROI_SIZE_PX: int = 300       # sliding-window side length
DEFAULT_SCALE_PX_PER_MM: float = 10.0  # TUNE: calibrate from a ruler in frame
DEFAULT_DOWNSAMPLE: int = 1          # 1 = full resolution; 2 = half resolution etc.
DEFAULT_N_JOBS: int = max(1, (os.cpu_count() or 1) - 1)
ANNOTATION_DIR: str = "annotations"
DEFAULT_MODEL_PATH: str = "models/root_classifier.joblib"
DEFAULT_CLASSIFIER_THRESHOLD: float = 0.6
DEFAULT_MAX_LOOP_SIZE: int = 200     # px — closed loops shorter than this are auto-removed
DEFAULT_PRUNE_LENGTH: int = 50       # px — terminal skeleton stubs shorter than this are pruned
DEFAULT_PRUNE_PASSES: int = 3        # iterations of stub pruning before lateral counting
DEFAULT_MIN_LATERAL_LENGTH: int = 60 # px — minimum skeleton length to count as a lateral
DEFAULT_MIN_LATERAL_ANGLE: float = 30.0   # degrees — minimum emergence angle from parent
DEFAULT_MAX_LATERAL_ANGLE: float = 150.0  # degrees — maximum emergence angle from parent
DEFAULT_MIN_LATERAL_PERSISTENCE: int = 40 # px — must travel this far before re-branching
DEFAULT_MAX_DIAMETER_CV: float = 0.4      # coefficient of variation of diameter along lateral
DEFAULT_LATERAL_CLF_THRESHOLD: float = 0.7  # stricter P(root) gate applied to laterals only
DEFAULT_MAX_LATERAL_DENSITY: float = 2.0    # max laterals per cm of parent root length
DEFAULT_PRE_SKELETON_THRESHOLD: float = 0.65  # classifier P(root) gate applied to mask components before skeletonization
DEFAULT_MIN_COMPONENT_AREA: int = 2000        # px² — mask components smaller than this removed before skeletonization
DEFAULT_MIN_STRAIGHTNESS: float = 0.5         # end-to-end / path-length; below this → curly fragment, removed
DEFAULT_LARGE_ROOT_LENGTH: int = 200          # px — segments longer than this use clf_threshold; shorter use small_root_threshold
DEFAULT_SMALL_ROOT_THRESHOLD: float = 0.80    # stricter P(root) gate for short skeleton segments
# Memory estimate per image worker (for --n-jobs tuning):
#   Full-res 4032×3024 image arrays  ≈  37 MB
#   ROI patch buffers (400 patches)  ≈   3 MB
#   Total per worker                 ≈  40 MB
# With 24 workers → ~1 GB.  With 64 GB RAM, up to ~1000 workers fit in theory,
# but I/O and CPU saturation dominate well before that.  n_jobs = n_cpu - 1 is safe.


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 0 — Image loading & rhizotron frame detection
# ─────────────────────────────────────────────────────────────────────────────

class RhizotronImage:
    """
    Loads a rhizotron photograph and locates the soil-filled interior region.

    The white/metal frame is detected as the largest bright connected region;
    a small inset clips the frame itself so downstream analysis sees only soil
    and roots.
    """

    def __init__(self, path: str, scale_px_per_mm: float = DEFAULT_SCALE_PX_PER_MM):
        self.path = path
        self.name = Path(path).stem
        self.scale = scale_px_per_mm    # pixels per millimetre

        raw = cv2.imread(path)
        if raw is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        # Standardise to landscape so top-of-frame ≈ crown zone in all images
        h, w = raw.shape[:2]
        if h > w:
            raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)

        self.image_bgr = raw
        self.image_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self.image_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        self.shape = self.image_gray.shape  # (H, W)

        # Bounding box of the soil interior: (y1, x1, y2, x2) in full-image px
        self.interior_bbox = self._detect_interior()

    # ── Frame detection ───────────────────────────────────────────────────────

    def _detect_interior(self) -> Tuple[int, int, int, int]:
        """
        Identify the rectangular soil-filled interior of the rhizotron frame.

        Two-step strategy:
          1. Coarse: threshold for bright (white/metal) frame pixels, find the
             largest connected region's bounding box.
          2. Refine: scan inward from each edge of that bbox until brightness
             drops to soil levels, giving the true inner soil boundary.

        # TUNE: FRAME_THRESH — raise if bright soil patches trigger false detection
        # TUNE: SOIL_THRESH — lower if roots near the frame edges are excluded
        """
        gray = self.image_gray
        FRAME_THRESH = 175   # TUNE: brightness that identifies the metal frame
        SOIL_THRESH = 130    # TUNE: brightness below which a stripe is "soil, not frame"

        _, bright = cv2.threshold(gray, FRAME_THRESH, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=3)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = gray.shape
        fallback_margin = int(0.05 * min(h, w))

        if not contours:
            return (fallback_margin, fallback_margin, h - fallback_margin, w - fallback_margin)

        bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # Scan inward from each coarse edge to find the true soil boundary
        cx1 = self._scan_to_soil(gray, bx,      bx + bw // 2,  axis="x", step=+1, thresh=SOIL_THRESH)
        cx2 = self._scan_to_soil(gray, bx + bw, bx + bw // 2,  axis="x", step=-1, thresh=SOIL_THRESH)
        cy1 = self._scan_to_soil(gray, by,      by + bh // 2,  axis="y", step=+1, thresh=SOIL_THRESH)
        cy2 = self._scan_to_soil(gray, by + bh, by + bh // 2,  axis="y", step=-1, thresh=SOIL_THRESH)

        # Safety clamp
        y1 = max(0, min(cy1, h - 1))
        x1 = max(0, min(cx1, w - 1))
        y2 = max(y1 + 1, min(cy2, h))
        x2 = max(x1 + 1, min(cx2, w))

        return (y1, x1, y2, x2)

    @staticmethod
    def _scan_to_soil(
        gray: np.ndarray,
        start: int,
        end: int,
        axis: str,
        step: int,
        thresh: float,
        stripe_w: int = 10,
    ) -> int:
        """
        Walk from *start* toward *end* in steps of *step*, returning the first
        position where the mean brightness of a thin stripe drops below *thresh*.
        Falls back to *end* if no such position is found.
        """
        positions = range(start, end, step)
        for pos in positions:
            if pos < 0 or pos >= (gray.shape[1] if axis == "x" else gray.shape[0]):
                continue
            if axis == "x":
                stripe = gray[:, pos : pos + stripe_w]
            else:
                stripe = gray[pos : pos + stripe_w, :]
            if stripe.size > 0 and stripe.mean() < thresh:
                return pos
        return end

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def interior_crop(self) -> np.ndarray:
        """BGR interior crop."""
        y1, x1, y2, x2 = self.interior_bbox
        return self.image_bgr[y1:y2, x1:x2]

    @property
    def interior_rgb(self) -> np.ndarray:
        """RGB interior crop — ready for matplotlib display."""
        y1, x1, y2, x2 = self.interior_bbox
        return self.image_rgb[y1:y2, x1:x2]

    @property
    def interior_gray(self) -> np.ndarray:
        """Grayscale interior crop."""
        y1, x1, y2, x2 = self.interior_bbox
        return self.image_gray[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
#  Debug helpers (module-level so they are accessible from worker processes)
# ─────────────────────────────────────────────────────────────────────────────

def _save_debug_img(
    binary_mask: np.ndarray,
    gray_bg: np.ndarray,
    debug_dir: str,
    stem: str,
    suffix: str,
) -> None:
    """Overlay *binary_mask* (green) on *gray_bg* and write a scaled PNG."""
    out_path = Path(debug_dir) / f"{stem}_{suffix}.png"
    h_m, w_m = binary_mask.shape
    bg = cv2.resize(gray_bg, (w_m, h_m))
    display = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    display[binary_mask] = (0, 180, 0)
    # Cap at 800 px on the long edge so debug images stay manageable
    sc = min(1.0, 800 / max(h_m, w_m))
    if sc < 1.0:
        display = cv2.resize(display, (int(w_m * sc), int(h_m * sc)))
    cv2.imwrite(str(out_path), display)


def _save_debug_skeleton(
    mask: np.ndarray,
    gray_bg: np.ndarray,
    debug_dir: str,
    stem: str,
) -> None:
    """Skeletonize a display-scale copy of *mask* and save with orange overlay."""
    h, w = mask.shape
    sc = min(1.0, 800 / max(h, w))
    dw, dh = max(1, int(w * sc)), max(1, int(h * sc))
    mask_small = cv2.resize(
        mask.astype(np.uint8) * 255, (dw, dh), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    skel = skeletonize(mask_small)
    bg = cv2.resize(gray_bg, (dw, dh))
    display = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    display[skel] = (0, 100, 255)   # orange skeleton
    out_path = Path(debug_dir) / f"{stem}_03_skeleton.png"
    cv2.imwrite(str(out_path), display)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — Root segmentation & skeletonisation
# ─────────────────────────────────────────────────────────────────────────────

class RootSegmenter:
    """
    Segments root pixels from soil using a white morphological top-hat transform,
    then filters by shape to suppress false positives from soil texture and debris.

    Why top-hat instead of Frangi as the primary detector:
    These images are very high-resolution (≥3000 px per side) and contain a
    dense fibrous root mat.  The Frangi/Hessian vesselness filter requires roots
    to occupy only a few pixels in width to work well; at full resolution the
    roots span 5-50 px and the Hessian response collapses to near zero.
    The white top-hat — which extracts bright features *smaller* than the
    structuring element — is both faster and more reliable at these scales.

    Frangi is used here as a *secondary gate* only: it suppresses regions of
    soil texture that the top-hat detects but that have no tubular structure.
    If Frangi gives near-zero response even for confirmed root pixels (which
    can happen in very dense mats where the tubular assumption breaks down),
    lower vesselness_threshold toward 0 to disable the gate.

    After top-hat + Frangi gating, three shape filters remove objects that do
    not look like roots:
      - Aspect ratio: roots are elongated; roundish blobs are likely debris
      - Diameter ceiling: blobs thicker than any real root are soil aggregates
      - Min skeleton length: tiny skeleton fragments are texture noise
    """

    def __init__(
        self,
        tophat_se_radius_mm: float = 2.5,
        min_root_area_px: int = 50,
        downsample: int = DEFAULT_DOWNSAMPLE,
        min_segment_length_px: int = 30,
        min_aspect_ratio: float = 3.0,
        max_root_diameter_mm: float = 3.0,
        vesselness_threshold: float = 0.01,
    ):
        """
        Parameters
        ----------
        tophat_se_radius_mm : float
            Radius of the morphological structuring element in mm.
            Should be ≈ 1.5 × max_root_radius_mm.
            # TUNE: increase for species with thick primary roots; decrease to
            #       preserve fine lateral roots in the mask.
        min_root_area_px : int
            Remove connected blobs smaller than this area (soil texture noise).
            # TUNE: set to the cross-section area of the finest real root you
            #       care about, in pixels² at the working resolution.
        downsample : int
            Process at 1/downsample linear scale. 1 = full; 2 = half (4× faster).
            # TUNE: 2 is a good balance for 4032×3024 images.
        min_segment_length_px : int
            Skeleton segments shorter than this are discarded as texture noise.
            Applied per connected component of the skeleton inside each ROI patch.
            # TUNE: raise to 50–100 to remove more short spurious traces;
            #       lower to 10 to keep short root stubs at patch edges.
        min_aspect_ratio : float
            Blobs whose major-axis / minor-axis ratio is below this value are
            removed (they look round, not elongated like roots).
            # TUNE: lower toward 1.5 if short root segments are being discarded;
            #       raise toward 5 to be stricter about elongation.
        max_root_diameter_mm : float
            Blobs whose maximum inscribed-circle diameter exceeds this value are
            flagged as soil aggregates, not roots, and removed.
            # TUNE: set to the widest individual root diameter you'd expect (not
            #       a merged bundle).  For grass: 1.5–3 mm is reasonable.
        vesselness_threshold : float
            Frangi vesselness score below which pixels are treated as non-root.
            Set to 0 to disable the vesselness gate entirely.
            # TUNE: lower (e.g. 0.005) if true roots are being suppressed in
            #       dense-mat zones; raise (e.g. 0.05) to be stricter about
            #       tubular structure.  Check --debug output to calibrate.
        """
        self.se_radius_mm = tophat_se_radius_mm
        self.min_area = min_root_area_px
        self.downsample = downsample
        self.min_seg_len = min_segment_length_px
        self.min_aspect_ratio = min_aspect_ratio
        self.max_root_diameter_mm = max_root_diameter_mm
        self.vesselness_threshold = vesselness_threshold

    def segment(
        self,
        gray: np.ndarray,
        scale_px_per_mm: float,
        debug_dir: Optional[str] = None,
        debug_stem: str = "img",
    ) -> np.ndarray:
        """
        Produce a binary root mask from a grayscale interior crop.

        Pipeline
        --------
        1. Optional downsample for speed.
        2. CLAHE local contrast equalisation.
        3. White top-hat transform (bright structures smaller than SE).
        4. Otsu thresholding → raw binary mask.
        5. Frangi vesselness gate: AND with regions that look tubular.
        6. remove_small_objects.
        7. Morphological closing to bridge thin gaps.
        8. Shape filter: remove blobs with bad aspect ratio or excessive diameter.
        9. Optional upsample back to original resolution.

        Parameters
        ----------
        gray : (H, W) uint8 — grayscale interior crop at full resolution
        scale_px_per_mm : float — spatial resolution for computing SE size
        debug_dir : str or None — if set, save intermediate images here
        debug_stem : str — filename prefix for debug images

        Returns
        -------
        mask : (H, W) bool — True at root pixels (full-resolution shape)
        """
        full_h, full_w = gray.shape
        ds = self.downsample

        if ds > 1:
            work = cv2.resize(gray, (full_w // ds, full_h // ds),
                              interpolation=cv2.INTER_AREA)
            work_scale = scale_px_per_mm / ds
        else:
            work = gray
            work_scale = scale_px_per_mm

        # CLAHE: equalise local contrast (mitigates uneven lighting in rhizotron)
        # TUNE: clipLimit — higher = more local contrast; may amplify noise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(work)

        # White top-hat: reveals bright structures smaller than the SE.
        se_radius_px = max(3, int(self.se_radius_mm * work_scale))
        se_diameter = 2 * se_radius_px + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_diameter, se_diameter))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, se)

        # Otsu on top-hat response
        _, binary_u8 = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary_u8.astype(bool)

        # ── Frangi vesselness gate ────────────────────────────────────────────
        # Suppresses soil texture / organic debris that passes the top-hat but
        # lacks the elongated tubular structure of a root strand.
        if self.vesselness_threshold > 0:
            binary = binary & self._vesselness_gate(enhanced, work_scale)

        # Debug (a): raw mask before shape filtering
        if debug_dir:
            _save_debug_img(binary, work, debug_dir, debug_stem, "01_raw_mask")

        # Remove tiny blobs (soil texture grains, noise specks)
        min_area_scaled = max(1, self.min_area // (ds * ds))
        cleaned = remove_small_objects(binary, max_size=min_area_scaled - 1)

        # Close thin gaps that break continuous root segments
        closed = closing(cleaned, disk(2))

        # ── Shape filtering ───────────────────────────────────────────────────
        shape_filtered = self._filter_by_shape(closed, work_scale)

        # Debug (b): after shape filtering
        if debug_dir:
            _save_debug_img(shape_filtered, work, debug_dir, debug_stem, "02_shape_filtered")

        # Upsample mask back to original resolution
        if ds > 1:
            sf_u8 = shape_filtered.astype(np.uint8) * 255
            upsampled = cv2.resize(sf_u8, (full_w, full_h),
                                   interpolation=cv2.INTER_NEAREST)
            final = upsampled > 0
        else:
            final = shape_filtered.astype(bool)

        # Debug (c): final mask + display-scale skeleton
        if debug_dir:
            _save_debug_img(final, gray, debug_dir, debug_stem, "03_final_mask")
            _save_debug_skeleton(final, gray, debug_dir, debug_stem)

        return final

    def _vesselness_gate(
        self, enhanced: np.ndarray, scale_px_per_mm: float
    ) -> np.ndarray:
        """
        Return a binary gate mask: True where Frangi vesselness ≥ threshold.

        Roots are bright tubular structures (black_ridges=False).  Frangi sigmas
        are set to span the expected root *radius* range in pixels so the
        Hessian scale-space covers fine laterals through thick primary roots.

        In very dense mats the tubular assumption can break down (many overlapping
        roots → no clean second-derivative trough), causing suppression of real
        root pixels.  Lower vesselness_threshold if this occurs.
        """
        # Sigmas = expected root radius in pixels at the working resolution.
        # TUNE: widen the range if root sizes in your images differ from defaults.
        min_sigma = max(0.5, 0.05 * scale_px_per_mm)   # ≈ 0.05 mm minimum root radius
        max_sigma = max(2.0, 1.0 * scale_px_per_mm)    # ≈ 1 mm maximum single-root radius
        sigmas = np.linspace(min_sigma, max_sigma, 6)

        img_f = enhanced.astype(np.float64) / 255.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vessel = frangi(img_f, sigmas=sigmas, black_ridges=False)

        return vessel >= self.vesselness_threshold

    def _filter_by_shape(
        self, mask: np.ndarray, scale_px_per_mm: float
    ) -> np.ndarray:
        """
        Remove connected components that do not look like roots.

        Two size regimes are handled differently because the morphological closing
        step merges touching root strands into large networks that are no longer
        elongated as a whole — applying an aspect-ratio test to those networks
        would incorrectly discard real roots.

        Small blobs  (area ≤ small_blob_area_px)
            These are isolated — a single strand segment or a piece of debris.
            Apply BOTH tests:
            • Aspect ratio: elongated strands pass; round specks fail.
            • Diameter ceiling (max DT): a blob thicker than any real root is
              a soil aggregate, not a strand.

        Large blobs  (area > small_blob_area_px)
            These are merged root networks crossing at many angles — their
            bounding ellipse is nearly round regardless of root shape.
            Apply ONLY the diameter ceiling, but use the MEDIAN DT rather than
            the maximum:
            • Median DT ≈ typical root radius for a true network (many thin
              pixels dominate the median even though crossing points are thick).
            • Median DT ≈ aggregate radius for a soil clump (uniformly thick).
            This distinguishes aggregates from networks even when both are large.

        # TUNE: small_blob_area_px (derived below) — blobs above this size skip
        #       the aspect-ratio test.  Raise the multiplier if isolated long
        #       root segments are incorrectly skipping the aspect-ratio filter.
        """
        labeled = sk_label(mask)
        props = regionprops(labeled)

        max_radius_px = (self.max_root_diameter_mm / 2.0) * scale_px_per_mm
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

        # Area threshold separating isolated segments from merged networks.
        # Derived as: (min segment length) × (max root diameter in px) × 3.
        # At default scale 10 px/mm: 30 × 30 × 3 = 2700 px².
        # TUNE: multiply factor (3) — increase if long isolated segments are
        #       bypassing the aspect-ratio filter.
        small_blob_area_px = int(self.min_seg_len * max_radius_px * 2 * 3)
        small_blob_area_px = max(200, small_blob_area_px)

        keep = np.zeros_like(mask, dtype=bool)
        for prop in props:
            coords = prop.coords   # (N, 2) of (row, col)
            blob_dt = dt[coords[:, 0], coords[:, 1]]

            if prop.area <= small_blob_area_px:
                # Isolated blob — apply full shape tests
                minor = prop.axis_minor_length
                if minor < 1e-6:
                    continue   # degenerate single-pixel run

                # TUNE: min_aspect_ratio — raise to reject shorter root stubs
                ar = prop.axis_major_length / minor
                if ar < self.min_aspect_ratio:
                    continue

                # TUNE: max_root_diameter_mm — diameter ceiling for small blobs
                if float(blob_dt.max()) > max_radius_px:
                    continue
            else:
                # Merged root network — skip aspect ratio, use 25th-percentile DT.
                # Crossing points inflate the median in dense grass mats; the 25th
                # percentile is still dominated by thin root pixels (DT ≈ 1-3 px)
                # even when crossing zones push the median above max_radius_px.
                # TUNE: percentile (25) — raise → stricter diameter ceiling.
                if float(np.percentile(blob_dt, 25)) > max_radius_px:
                    continue

            keep[labeled == prop.label] = True

        return keep

    @staticmethod
    def skeletonize_and_measure(
        mask: np.ndarray,
        scale_px_per_mm: float,
        max_single_root_radius_mm: float = 1.0,
        min_segment_length_px: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Skeletonize a binary root mask and measure the perpendicular diameter
        at every skeleton pixel, then flag coalesced ("dense mat") regions.

        Why distance transform = perpendicular diameter
        -----------------------------------------------
        The skeleton (medial axis) is the locus of centers of *maximal
        inscribed circles* — circles that fit inside the root mask and touch
        the mask boundary on at least two sides.  The distance-transform value
        at each skeleton pixel is precisely the radius of that maximal inscribed
        circle, which by construction is tangent to the root boundary
        perpendicularly on both sides.

        Min-segment-length filtering
        ----------------------------
        After Lee skeletonization, connected skeleton components shorter than
        `min_segment_length_px` are removed.  Soil texture that survives the
        mask-level filters often produces short, isolated skeleton fragments;
        this step discards them before feature extraction.

        Dense-mat detection
        -------------------
        When two roots are so close that the binary mask merges them into one
        blob, the skeleton runs through the merged blob and the inscribed-circle
        radius becomes larger than any single root could produce.  Skeleton pixels
        where DT > `max_single_root_radius_mm × scale` are flagged as dense-mat.

        Parameters
        ----------
        mask : (H, W) bool
        scale_px_per_mm : float
        max_single_root_radius_mm : float
            # TUNE: lower to be more aggressive about flagging merged regions
        min_segment_length_px : int
            # TUNE: raise to discard longer spurious traces; lower to retain
            #       short root stubs at ROI boundaries.

        Returns
        -------
        skeleton : (H, W) bool — 1-pixel-wide centerline network
        skel_diam_px : (H, W) float32 — diameter (px) at skeleton pixels; 0 elsewhere
        dense_mat : (H, W) bool — True where strands cannot be individually resolved
        """
        skeleton = skeletonize(mask)

        # ── Remove short skeleton segments ────────────────────────────────────
        # TUNE: min_segment_length_px — raise to reject longer spurious fragments
        if min_segment_length_px > 1 and skeleton.any():
            # 8-connectivity required: skeletons use diagonal links that 4-conn splits
            _conn8 = np.ones((3, 3), dtype=int)
            labeled_skel, n_skel = nd_label(skeleton, structure=_conn8)
            for comp_id in range(1, n_skel + 1):
                comp_mask = labeled_skel == comp_id
                if comp_mask.sum() < min_segment_length_px:
                    skeleton[comp_mask] = False

        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        skel_diam_px = np.where(skeleton, dt * 2.0, 0.0).astype(np.float32)

        max_radius_px = max_single_root_radius_mm * scale_px_per_mm
        dense_mat = skeleton & (dt > max_radius_px)

        return skeleton, skel_diam_px, dense_mat


def _apply_pre_skeleton_gate(
    mask: np.ndarray,
    gray: np.ndarray,
    scale: float,
    classifier: "SkeletonClassifier",
    threshold: float,
    min_component_area: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each connected component of the binary mask before skeletonization.

    Each component's centroid is scored with the RF classifier; components with
    P(root) < threshold are removed.  Components smaller than min_component_area
    are always removed regardless of classifier score.

    Skeleton-dependent features (on_skeleton, curvature) are set to zero here
    because no skeleton exists yet — the remaining 28 features are sufficient to
    distinguish elongated root blobs from closed pore-boundary rings.

    Returns
    -------
    refined_mask : bool array — components that passed the gate
    prob_map     : float32 array — per-component P(root); 0 where mask was False
    """
    conn8    = np.ones((3, 3), dtype=int)
    labeled, n = nd_label(mask, structure=conn8)
    prob_map = np.zeros(mask.shape, dtype=np.float32)

    if n == 0:
        return mask.copy(), prob_map

    # Precompute DT and gradient once for the whole image
    dt   = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_g = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy_g ** 2)

    # Empty skeleton placeholder — skeleton-based features default to 0
    empty_skel = np.zeros_like(mask, dtype=bool)

    clf_active = classifier is not None and classifier.is_loaded()
    root_idx: Optional[int] = None
    if clf_active:
        root_idx = classifier._classes.index(0) if 0 in classifier._classes else None

    h, w    = mask.shape
    refined = mask.copy()

    for comp_id in range(1, n + 1):
        comp      = labeled == comp_id
        comp_size = int(comp.sum())

        # Minimum component area gate (always applied)
        if min_component_area > 0 and comp_size < min_component_area:
            refined[comp] = False
            continue

        # Classifier gate — sample several interior points and take the max
        # P(root) so one bad centroid doesn't condemn a whole root strand.
        # (Skeleton features are unavailable here, so centroid-only scoring
        # is systematically biased low for real roots.)
        if clf_active and root_idx is not None:
            ys, xs = np.where(comp)
            n_pts  = min(5, len(ys))
            idx    = np.linspace(0, len(ys) - 1, n_pts, dtype=int)
            p_root = 0.0
            for i in idx:
                cy    = int(np.clip(ys[i], 0, h - 1))
                cx    = int(np.clip(xs[i], 0, w - 1))
                feats = _extract_features_at(
                    gray, empty_skel, mask, cy, cx, scale, _dt=dt, _grad=grad
                )
                p     = float(classifier.model.predict_proba([feats])[0][root_idx])
                if p > p_root:
                    p_root = p
            prob_map[comp] = p_root
            if p_root < threshold:
                refined[comp] = False
        else:
            # No classifier — mark all kept components with probability 1.0
            prob_map[comp] = 1.0

    return refined, prob_map


def _remove_loop_components(
    mask: np.ndarray, min_hole_area: int = 200
) -> Tuple[np.ndarray, int]:
    """
    Remove mask components that form closed rings around enclosed dark regions.

    Uses binary_fill_holes to detect interior holes.  Only holes larger than
    min_hole_area pixels are considered (avoids removing roots near small gaps).

    Returns (refined_mask, n_pixels_removed).
    """
    filled = binary_fill_holes(mask)
    holes  = filled & ~mask
    if not holes.any():
        return mask, 0

    h_labeled, n_holes = nd_label(holes, structure=np.ones((3, 3), dtype=int))
    big_holes = np.zeros_like(holes, dtype=bool)
    for hid in range(1, n_holes + 1):
        if int((h_labeled == hid).sum()) >= min_hole_area:
            big_holes |= h_labeled == hid
    if not big_holes.any():
        return mask, 0

    # Dilate holes by 1 px to find the mask components that surround them
    big_holes_d = cv2.dilate(big_holes.astype(np.uint8),
                              np.ones((3, 3), np.uint8)).astype(bool)
    labeled, n = nd_label(mask, structure=np.ones((3, 3), dtype=int))
    result = mask.copy()
    for comp_id in range(1, n + 1):
        comp = labeled == comp_id
        if (comp & big_holes_d).any():
            result[comp] = False

    return result, int(mask.sum()) - int(result.sum())


def _save_pre_skeleton_debug(
    prob_map: np.ndarray,
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
    gray: np.ndarray,
    debug_dir: str,
    stem: str,
) -> None:
    """
    Save two diagnostic images into debug_dir:
      <stem>_04_proba.png        — P(root) colourmap overlaid on the grayscale image
      <stem>_05_refined_mask.png — classifier-gated binary mask overlaid on grayscale
    """
    sc = min(1.0, 800 / max(gray.shape))
    dw = max(1, int(gray.shape[1] * sc))
    dh = max(1, int(gray.shape[0] * sc))

    # ── Probability map (RdYlGn: red = low P(root), green = high P(root)) ──
    import matplotlib
    cmap    = matplotlib.colormaps["RdYlGn"]
    prob_s  = cv2.resize(prob_map, (dw, dh), interpolation=cv2.INTER_NEAREST)
    gray_s  = cv2.resize(gray, (dw, dh))
    bg_rgb  = cv2.cvtColor(gray_s, cv2.COLOR_GRAY2RGB)
    prob_rgb = (cmap(prob_s)[:, :, :3] * 255).astype(np.uint8)
    any_mask = prob_s > 0
    display  = bg_rgb.copy()
    display[any_mask] = (
        bg_rgb[any_mask].astype(np.float32) * 0.35
        + prob_rgb[any_mask].astype(np.float32) * 0.65
    ).astype(np.uint8)
    cv2.imwrite(
        str(Path(debug_dir) / f"{stem}_04_proba.png"),
        cv2.cvtColor(display, cv2.COLOR_RGB2BGR),
    )

    # ── Refined mask (green overlay on grayscale) ──
    ref_s   = cv2.resize(refined_mask.astype(np.uint8) * 255,
                          (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    mask_display = cv2.cvtColor(gray_s, cv2.COLOR_GRAY2BGR)
    mask_display[ref_s] = np.clip(
        mask_display[ref_s].astype(np.int32) // 2 + [0, 80, 0], 0, 255
    ).astype(np.uint8)
    cv2.imwrite(
        str(Path(debug_dir) / f"{stem}_05_refined_mask.png"),
        mask_display,
    )


def _prune_one_pass(skeleton: np.ndarray, prune_length: int) -> np.ndarray:
    """Remove terminal skeleton branches shorter than *prune_length* pixels."""
    skel_u8 = skeleton.astype(np.uint8)
    kern = np.ones((3, 3), np.float32)
    kern[1, 1] = 0.0
    nbrs = (cv2.filter2D(skel_u8, -1, kern) * skel_u8).astype(np.uint8)
    branch_map = (nbrs >= 3) & skeleton
    tip_map    = (nbrs == 1) & skeleton
    if not branch_map.any():
        return skeleton  # no junctions → nothing to prune
    # Label segments between junction pixels
    seg_skel = skeleton & ~branch_map
    labeled, n = nd_label(seg_skel, structure=np.ones((3, 3), dtype=int))
    remove = np.zeros_like(skeleton, dtype=bool)
    for seg_id in range(1, n + 1):
        seg = labeled == seg_id
        if (seg & tip_map).any() and int(seg.sum()) < prune_length:
            remove |= seg
    return skeleton & ~remove


def prune_skeleton(
    skeleton: np.ndarray, prune_length: int, prune_passes: int
) -> np.ndarray:
    """Iteratively remove terminal stubs shorter than *prune_length* for up to
    *prune_passes* rounds (stops early if skeleton stops changing)."""
    if prune_length <= 0 or prune_passes <= 0:
        return skeleton
    result = skeleton.copy()
    for _ in range(prune_passes):
        nxt = _prune_one_pass(result, prune_length)
        if np.array_equal(nxt, result):
            break
        result = nxt
    return result


def _local_direction(
    mask: np.ndarray, jy: int, jx: int, radius: int = 20
) -> Optional[np.ndarray]:
    """
    Estimate the local orientation of *mask* pixels near (jy, jx) using PCA.
    Returns a unit vector [dy, dx] (unsigned direction), or None if too few pixels.
    """
    ys, xs = np.where(mask)
    if len(ys) < 2:
        return None
    dists = np.sqrt((ys - jy) ** 2 + (xs - jx) ** 2)
    for r in (radius, radius * 2, radius * 4):
        close = dists <= r
        if int(close.sum()) >= 3:
            break
    if int(close.sum()) < 2:
        return None
    pts = np.stack(
        [ys[close].astype(float) - jy, xs[close].astype(float) - jx], axis=1
    )
    cov = np.cov(pts.T)
    if cov.ndim < 2:
        return None
    _, evecs = np.linalg.eigh(cov)
    return evecs[:, -1]   # principal direction (unit vector)


def _dist_to_first_branch(
    lat_comp: np.ndarray, start_yx: Tuple[int, int], branch_map: np.ndarray
) -> int:
    """
    BFS from *start_yx* along *lat_comp* skeleton pixels; return the number of
    steps to the first branch point, or the component length if none exists.
    """
    h, w = lat_comp.shape
    sy, sx = start_yx
    if not lat_comp[sy, sx]:
        ys, xs = np.where(lat_comp)
        if len(ys) == 0:
            return 0
        idx = int(np.argmin(np.abs(ys - sy) + np.abs(xs - sx)))
        sy, sx = int(ys[idx]), int(xs[idx])

    visited = np.zeros_like(lat_comp, dtype=bool)
    queue: List[Tuple[int, int, int]] = [(sy, sx, 0)]
    visited[sy, sx] = True
    while queue:
        cy, cx, d = queue.pop(0)
        if branch_map[cy, cx] and d > 0:
            return d
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and lat_comp[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx, d + 1))
    return int(lat_comp.sum())   # no branch point → persistence = full length


def classify_primary_lateral(
    skeleton: np.ndarray,
    skel_diam_px: np.ndarray,
    scale_px_per_mm: float,
    min_primary_diameter_mm: float = 0.5,
    min_lateral_length_px: int = DEFAULT_MIN_LATERAL_LENGTH,
    min_lateral_angle_deg: float = DEFAULT_MIN_LATERAL_ANGLE,
    max_lateral_angle_deg: float = DEFAULT_MAX_LATERAL_ANGLE,
    min_lateral_persistence_px: int = DEFAULT_MIN_LATERAL_PERSISTENCE,
    max_diameter_cv: float = DEFAULT_MAX_DIAMETER_CV,
    max_lateral_density_per_cm: float = DEFAULT_MAX_LATERAL_DENSITY,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Classify skeleton components as primary or lateral root.

    A component is counted as a lateral only if it passes ALL of:
      1. Length ≥ min_lateral_length_px
      2. Attached to a primary segment (not a floating fragment)
      3. Emergence angle at junction within [min_lateral_angle_deg, max_lateral_angle_deg]
      4. Persists ≥ min_lateral_persistence_px before re-branching
      5. Diameter coefficient of variation < max_diameter_cv

    After qualification, if the inferred lateral density per cm of primary root
    exceeds max_lateral_density_per_cm, the longest candidates are kept and the
    rest discarded.  The third return value flags whether this cap fired.

    Returns
    -------
    primary_mask, lateral_mask, laterals_capped
    """
    labeled, n_comp = nd_label(skeleton, structure=np.ones((3, 3), dtype=int))
    primary_mask = np.zeros_like(skeleton, dtype=bool)

    threshold_px = min_primary_diameter_mm * scale_px_per_mm
    candidate_ids: List[int] = []

    # Pass 1 — split by mean diameter into primary / candidate lateral
    for comp_id in range(1, n_comp + 1):
        comp = labeled == comp_id
        diams = skel_diam_px[comp]
        mean_diam = float(diams.mean()) if diams.size > 0 else 0.0
        if mean_diam >= threshold_px:
            primary_mask |= comp
        else:
            candidate_ids.append(comp_id)

    # Pre-compute branch map for angle / persistence checks
    skel_u8 = skeleton.astype(np.uint8)
    kern = np.ones((3, 3), np.float32)
    kern[1, 1] = 0.0
    nbrs = (cv2.filter2D(skel_u8, -1, kern) * skel_u8).astype(np.uint8)
    branch_map = (nbrs >= 3) & skeleton

    # Unsigned angle threshold: [min_deg, 180-max_deg] → minimum unsigned angle
    min_unsigned_deg = min(min_lateral_angle_deg, 180.0 - max_lateral_angle_deg)
    kern3 = np.ones((3, 3), np.uint8)
    DIRECTION_RADIUS = 20

    # Pass 2 — apply all five qualification criteria
    qualified: List[Tuple[int, int]] = []   # (comp_id, length)
    for comp_id in candidate_ids:
        comp = labeled == comp_id
        comp_len = int(comp.sum())

        # Criterion 1: minimum length
        if comp_len < min_lateral_length_px:
            continue

        # Criterion 5: diameter coefficient of variation
        diams = skel_diam_px[comp]
        valid_d = diams[diams > 0]
        if valid_d.size > 1 and float(valid_d.mean()) > 0:
            if float(valid_d.std() / valid_d.mean()) > max_diameter_cv:
                continue

        # Criteria 2–4: require attachment to a primary root at a junction
        lat_dilated = cv2.dilate(comp.astype(np.uint8), kern3).astype(bool)
        junction_zone = lat_dilated & primary_mask
        if not junction_zone.any():
            continue   # floating fragment — not attached to any primary

        jys, jxs = np.where(junction_zone)
        jy, jx = int(jys.mean()), int(jxs.mean())

        # Criterion 3: emergence angle
        prim_dir = _local_direction(primary_mask, jy, jx, DIRECTION_RADIUS)
        lat_dir  = _local_direction(comp,         jy, jx, DIRECTION_RADIUS)
        if prim_dir is not None and lat_dir is not None:
            cos_a = float(np.clip(abs(np.dot(prim_dir, lat_dir)), 0.0, 1.0))
            unsigned_angle = float(np.degrees(np.arccos(cos_a)))
            if unsigned_angle < min_unsigned_deg:
                continue

        # Criterion 4: persistence before re-branching
        if _dist_to_first_branch(comp, (jy, jx), branch_map) < min_lateral_persistence_px:
            continue

        qualified.append((comp_id, comp_len))

    # Density cap — if too many laterals per cm of primary root, keep the longest
    laterals_capped = False
    primary_cm = float(primary_mask.sum()) / (scale_px_per_mm * 10.0)
    if max_lateral_density_per_cm > 0 and primary_cm > 0:
        max_allowed = int(np.ceil(primary_cm * max_lateral_density_per_cm))
        if len(qualified) > max_allowed:
            qualified.sort(key=lambda t: t[1], reverse=True)
            qualified = qualified[:max_allowed]
            laterals_capped = True

    lateral_mask = np.zeros_like(skeleton, dtype=bool)
    kept_ids = {comp_id for comp_id, _ in qualified}
    for comp_id in kept_ids:
        lateral_mask |= (labeled == comp_id)

    return primary_mask, lateral_mask, laterals_capped


def find_tips_and_branches(
    skeleton: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Locate tip pixels (degree 1) and branch-point pixels (degree ≥ 3) in a skeleton.

    Returns
    -------
    tip_map, branch_map : (H, W) bool
    """
    skel_u8 = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0.0
    n_neighbours = cv2.filter2D(skel_u8, -1, kernel).astype(np.uint8)
    n_neighbours *= skel_u8

    tip_map = (n_neighbours == 1) & skeleton
    branch_map = (n_neighbours >= 3) & skeleton

    return tip_map, branch_map


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 — Per-root trait extraction
# ─────────────────────────────────────────────────────────────────────────────

def _diameter_bin_labels(bins_mm: List[float]) -> List[str]:
    """Generate labels like ['lt0.5mm', '0.5to1.0mm', '1.0to2.0mm', 'gt2.0mm']."""
    labels = [f"lt{bins_mm[0]}mm"]
    for lo, hi in zip(bins_mm[:-1], bins_mm[1:]):
        labels.append(f"{lo}to{hi}mm")
    labels.append(f"gt{bins_mm[-1]}mm")
    return labels


def extract_roi_features(
    mask: np.ndarray,
    skeleton: np.ndarray,
    skel_diam_px: np.ndarray,
    dense_mat: np.ndarray,
    primary_mask: np.ndarray,
    lateral_mask: np.ndarray,
    scale_px_per_mm: float,
    diameter_bins_mm: List[float],
) -> Dict[str, float]:
    """
    Compute a feature vector describing root architecture within one ROI.

    All diameter measurements use `skel_diam_px`, which is non-zero only at
    skeleton (centerline) pixels, ensuring measurements reflect inscribed-circle
    radius at the strand midline rather than arbitrary interior blob points.

    Features
    --------
    root_density          fraction of ROI area covered by roots
    total_length_mm       total skeleton length in mm
    primary_length_mm     primary-root skeleton length
    lateral_length_mm     lateral-root skeleton length
    lateral_count         number of discrete lateral-root components
    branching_density     lateral root segments per cm of primary root
    mean_diameter_mm      mean perpendicular diameter at skeleton pixels
    std_diameter_mm       std of perpendicular diameters
    diam_frac_<bin>       fraction of skeleton length in each diameter class
    dense_mat_fraction    fraction of skeleton pixels in unresolvable merged zones
    tip_zone_frac         fraction of skeleton near root tips (young-tissue proxy)
    branch_zone_frac      fraction of skeleton near branch points (older-tissue proxy)
    """
    mm_per_px = 1.0 / scale_px_per_mm
    h, w = mask.shape
    feats: Dict[str, float] = {}

    feats["root_density"] = float(mask.sum()) / (h * w)

    skel_px = int(skeleton.sum())
    feats["total_length_mm"] = skel_px * mm_per_px

    prim_skel = skeleton & primary_mask
    lat_skel  = skeleton & lateral_mask
    feats["primary_length_mm"] = float(prim_skel.sum()) * mm_per_px
    feats["lateral_length_mm"] = float(lat_skel.sum()) * mm_per_px

    _, n_laterals = nd_label(lat_skel, structure=np.ones((3, 3), dtype=int))
    feats["lateral_count"] = float(n_laterals)
    primary_cm = feats["primary_length_mm"] / 10.0
    feats["branching_density"] = n_laterals / primary_cm if primary_cm > 0 else 0.0

    safe_skel = max(skel_px, 1)
    feats["dense_mat_fraction"] = float(dense_mat.sum()) / safe_skel

    resolvable = skeleton & (skel_diam_px > 0) & ~dense_mat
    diam_px = skel_diam_px[resolvable]
    if diam_px.size > 0:
        diam_mm = diam_px * mm_per_px
        feats["mean_diameter_mm"] = float(diam_mm.mean())
        feats["std_diameter_mm"] = float(diam_mm.std())
    else:
        all_skel_diam = skel_diam_px[skeleton & (skel_diam_px > 0)]
        if all_skel_diam.size > 0:
            diam_mm = all_skel_diam * mm_per_px
            feats["mean_diameter_mm"] = float(diam_mm.mean())
            feats["std_diameter_mm"] = float(diam_mm.std())
        else:
            feats["mean_diameter_mm"] = 0.0
            feats["std_diameter_mm"] = 0.0

    bin_edges_px = [0.0] + [b * scale_px_per_mm for b in diameter_bins_mm] + [np.inf]
    bin_labels   = _diameter_bin_labels(diameter_bins_mm)

    for label, lo, hi in zip(bin_labels, bin_edges_px[:-1], bin_edges_px[1:]):
        in_bin = skeleton & ~dense_mat & (skel_diam_px >= lo) & (skel_diam_px < hi)
        feats[f"diam_frac_{label}"] = float(in_bin.sum()) / safe_skel

    tip_map, branch_map = find_tips_and_branches(skeleton)
    tip_zone    = closing(tip_map,    disk(10))
    branch_zone = closing(branch_map, disk(10))
    feats["tip_zone_frac"]    = float((skeleton & tip_zone).sum())    / safe_skel
    feats["branch_zone_frac"] = float((skeleton & branch_zone).sum()) / safe_skel

    return feats


def _filter_laterals_by_classifier(
    lateral_mask: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    scale: float,
    classifier: "SkeletonClassifier",
    threshold: float,
) -> np.ndarray:
    """
    Remove lateral skeleton components whose P(root) < *threshold*.
    Applied after primary/lateral classification so primaries are unaffected.
    """
    if not classifier.is_loaded():
        return lateral_mask
    labeled, n = nd_label(lateral_mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return lateral_mask

    root_idx: Optional[int] = (
        classifier._classes.index(0) if 0 in classifier._classes else None
    )
    if root_idx is None:
        return lateral_mask

    dt   = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_g = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy_g ** 2)

    h, w     = gray.shape
    filtered = lateral_mask.copy()
    for comp_id in range(1, n + 1):
        comp = labeled == comp_id
        ys, xs = np.where(comp)
        cy = int(np.clip(ys.mean(), 0, h - 1))
        cx = int(np.clip(xs.mean(), 0, w - 1))
        feats  = _extract_features_at(gray, skeleton, mask, cy, cx, scale, _dt=dt, _grad=grad)
        proba  = classifier.model.predict_proba([feats])[0]
        if float(proba[root_idx]) < threshold:
            filtered[comp] = False
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
#  ROI-level parallel worker (module-level so ProcessPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _process_single_roi(
    roi_mask: np.ndarray,
    roi_gray: np.ndarray,
    scale: float,
    min_prim_diam: float,
    bins: List[float],
    y: int,
    x: int,
    size: int,
    iy1: int,
    ix1: int,
    name: str,
    min_seg_len: int,
    min_skeleton_density: float,
    crop_h: int,
    crop_w: int,
    classifier: Optional["SkeletonClassifier"] = None,
    clf_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
    max_loop_size: int = DEFAULT_MAX_LOOP_SIZE,
    prune_length: int = DEFAULT_PRUNE_LENGTH,
    prune_passes: int = DEFAULT_PRUNE_PASSES,
    min_lateral_length: int = DEFAULT_MIN_LATERAL_LENGTH,
    min_lateral_angle: float = DEFAULT_MIN_LATERAL_ANGLE,
    max_lateral_angle: float = DEFAULT_MAX_LATERAL_ANGLE,
    min_lateral_persistence: int = DEFAULT_MIN_LATERAL_PERSISTENCE,
    max_diameter_cv: float = DEFAULT_MAX_DIAMETER_CV,
    lateral_clf_threshold: float = DEFAULT_LATERAL_CLF_THRESHOLD,
    max_lateral_density: float = DEFAULT_MAX_LATERAL_DENSITY,
    min_straightness: float = DEFAULT_MIN_STRAIGHTNESS,
    large_root_length: int = DEFAULT_LARGE_ROOT_LENGTH,
    small_root_threshold: float = DEFAULT_SMALL_ROOT_THRESHOLD,
    collect_stats: bool = False,
) -> Optional[Dict]:
    """
    Process one sliding-window patch and return its feature dict, or None if
    the patch fails the skeleton-density pre-filter.
    """
    # Cap min_seg_len for per-patch use (boundary-crossing stubs)
    # TUNE: the divisor (10) — lower → keep shorter stubs; raise → stricter.
    effective_min_seg = min(min_seg_len, max(5, size // 10))
    roi_skel, roi_skel_diam, roi_dense = RootSegmenter.skeletonize_and_measure(
        roi_mask, scale, min_segment_length_px=effective_min_seg
    )

    # ── Skeleton pruning ──────────────────────────────────────────────────────
    # Remove terminal stubs before lateral counting so "hairy" branching
    # artifacts don't inflate lateral counts.
    if prune_length > 0 and prune_passes > 0:
        pruned = prune_skeleton(roi_skel, prune_length, prune_passes)
        removed = roi_skel & ~pruned
        roi_skel_diam[removed] = 0.0
        roi_dense[removed]     = False
        roi_skel = pruned

    # ── Classifier post-filter (full skeleton) ────────────────────────────────
    if classifier is not None and classifier.is_loaded():
        filtered_skel, flt_stats = classifier.filter_skeleton(
            roi_skel, roi_gray, roi_mask, scale, clf_threshold, max_loop_size,
            min_straightness=min_straightness,
            large_root_length=large_root_length,
            small_root_threshold=small_root_threshold,
            collect_stats=collect_stats,
        )
        removed = roi_skel & ~filtered_skel
        roi_skel_diam[removed] = 0.0
        roi_dense[removed]     = False
        roi_skel = filtered_skel

    # ── Skeleton-density pre-filter ───────────────────────────────────────────
    skel_density = float(roi_skel.sum()) / (size * size)
    if skel_density < min_skeleton_density:
        return None

    roi_prim, roi_lat, laterals_capped = classify_primary_lateral(
        roi_skel, roi_skel_diam, scale, min_prim_diam,
        min_lateral_length_px=min_lateral_length,
        min_lateral_angle_deg=min_lateral_angle,
        max_lateral_angle_deg=max_lateral_angle,
        min_lateral_persistence_px=min_lateral_persistence,
        max_diameter_cv=max_diameter_cv,
        max_lateral_density_per_cm=max_lateral_density,
    )

    # ── Per-lateral classifier gate (stricter threshold than full skeleton) ───
    if (classifier is not None and classifier.is_loaded()
            and lateral_clf_threshold > clf_threshold):
        roi_lat = _filter_laterals_by_classifier(
            roi_lat, roi_gray, roi_mask, roi_skel, scale,
            classifier, lateral_clf_threshold,
        )

    feats = extract_roi_features(
        roi_mask, roi_skel, roi_skel_diam, roi_dense,
        roi_prim, roi_lat, scale, bins,
    )
    feats["laterals_capped"] = 1.0 if laterals_capped else 0.0
    feats["image_name"] = name
    feats["y1"] = y;        feats["x1"] = x
    feats["y2"] = y + size; feats["x2"] = x + size
    feats["abs_y1"] = iy1 + y;        feats["abs_x1"] = ix1 + x
    feats["abs_y2"] = iy1 + y + size; feats["abs_x2"] = ix1 + x + size
    feats["interior_h"] = crop_h
    feats["interior_w"] = crop_w
    return feats


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3 — ROI definition & cross-plant matching
# ─────────────────────────────────────────────────────────────────────────────

class ROIExtractor:
    """
    Tiles each image's interior crop with a sliding window and computes
    a feature vector for every window that contains enough root material.
    """

    def __init__(
        self,
        roi_size_px: int = DEFAULT_ROI_SIZE_PX,
        stride_px: Optional[int] = None,
        min_root_density: float = 0.01,
        min_segment_length_px: int = 30,
        min_skeleton_density: float = 0.02,
    ):
        """
        Parameters
        ----------
        roi_size_px : int
            # TUNE: larger ROIs average more variation; smaller are noisier.
        stride_px : int or None
            Defaults to roi_size_px // 2 (50 % overlap).
            # TUNE: reduce to increase candidate count; increase for speed.
        min_root_density : float
            Fast pre-filter: skip windows where mask coverage < this value.
            # TUNE: increase (e.g. 0.05) to focus on root-dense areas only.
        min_segment_length_px : int
            Passed through to skeletonize_and_measure() for per-ROI patches.
            Important: the effective value is capped at roi_size // 6 inside
            each patch so that root segments that cross the patch boundary
            (and are therefore shorter than the full segment length) are not
            discarded.  Set --min-segment-length for full-image filtering
            stringency; the per-patch cap prevents over-filtering at boundaries.
        min_skeleton_density : float
            Post-skeletonisation filter: discard patches whose skeleton pixel
            fraction is below this threshold.  Applied after full skeleton
            computation so it reflects actual traced-root density, not raw
            mask coverage.  Eliminates empty-soil and near-border artifact
            patches before they enter cross-plant matching.
            # TUNE: raise to 0.05–0.10 for stricter root-density requirement;
            #       the --debug density map shows actual per-tile values.
        """
        self.roi_size = roi_size_px
        self.stride = stride_px if stride_px is not None else roi_size_px // 2
        self.min_density = min_root_density
        self.min_seg_len = min_segment_length_px
        self.min_skeleton_density = min_skeleton_density

    def extract_rois(
        self,
        rh: "RhizotronImage",
        mask: np.ndarray,
        _unused_diameter_map,
        diameter_bins_mm: List[float],
        min_primary_diameter_mm: float = 0.5,
        n_workers: int = 1,
        classifier: Optional["SkeletonClassifier"] = None,
        clf_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
        max_loop_size: int = DEFAULT_MAX_LOOP_SIZE,
        prune_length: int = DEFAULT_PRUNE_LENGTH,
        prune_passes: int = DEFAULT_PRUNE_PASSES,
        min_lateral_length: int = DEFAULT_MIN_LATERAL_LENGTH,
        min_lateral_angle: float = DEFAULT_MIN_LATERAL_ANGLE,
        max_lateral_angle: float = DEFAULT_MAX_LATERAL_ANGLE,
        min_lateral_persistence: int = DEFAULT_MIN_LATERAL_PERSISTENCE,
        max_diameter_cv: float = DEFAULT_MAX_DIAMETER_CV,
        lateral_clf_threshold: float = DEFAULT_LATERAL_CLF_THRESHOLD,
        max_lateral_density: float = DEFAULT_MAX_LATERAL_DENSITY,
        min_straightness: float = DEFAULT_MIN_STRAIGHTNESS,
        large_root_length: int = DEFAULT_LARGE_ROOT_LENGTH,
        small_root_threshold: float = DEFAULT_SMALL_ROOT_THRESHOLD,
    ) -> List[Dict]:
        """
        Slide a window across the interior crop and build one feature dict per ROI.

        Skeletonisation, stub pruning, primary/lateral classification, and the
        per-lateral classifier gate are all performed per-ROI on each 300×300 patch.
        """
        h, w = mask.shape
        iy1, ix1, iy2, ix2 = rh.interior_bbox
        size = self.roi_size
        gray = rh.interior_gray

        roi_positions = []
        for y in range(0, h - size + 1, self.stride):
            for x in range(0, w - size + 1, self.stride):
                patch = mask[y : y + size, x : x + size]
                if patch.mean() >= self.min_density:
                    roi_positions.append((
                        y, x,
                        patch.copy(),
                        gray[y : y + size, x : x + size].copy(),
                    ))

        if not roi_positions:
            return []

        def _proc(item: Tuple) -> Optional[Dict]:
            y, x, roi_mask, roi_gray = item
            return _process_single_roi(
                roi_mask, roi_gray, rh.scale, min_primary_diameter_mm,
                diameter_bins_mm, y, x, size, iy1, ix1, rh.name, self.min_seg_len,
                self.min_skeleton_density, h, w,
                classifier=classifier,
                clf_threshold=clf_threshold,
                max_loop_size=max_loop_size,
                prune_length=prune_length,
                prune_passes=prune_passes,
                min_lateral_length=min_lateral_length,
                min_lateral_angle=min_lateral_angle,
                max_lateral_angle=max_lateral_angle,
                min_lateral_persistence=min_lateral_persistence,
                max_diameter_cv=max_diameter_cv,
                lateral_clf_threshold=lateral_clf_threshold,
                max_lateral_density=max_lateral_density,
                min_straightness=min_straightness,
                large_root_length=large_root_length,
                small_root_threshold=small_root_threshold,
            )

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                rois = list(pool.map(_proc, roi_positions))
        else:
            rois = [_proc(item) for item in roi_positions]

        return [r for r in rois if r is not None]


# ── Feature matrix helpers ────────────────────────────────────────────────────

_COORD_KEYS = frozenset([
    "image_name", "y1", "x1", "y2", "x2",
    "abs_y1", "abs_x1", "abs_y2", "abs_x2",
    "interior_h", "interior_w",   # crop dims for border-penalty; not a root feature
])


def _feature_keys(rois: List[Dict]) -> List[str]:
    """Return numeric feature key names (excluding coordinate/name fields)."""
    if not rois:
        return []
    return [k for k in rois[0] if k not in _COORD_KEYS]


def _build_feature_matrix(
    rois: List[Dict], keys: List[str]
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Stack ROI feature dicts into a z-score normalised numeric matrix.

    Normalisation ensures no single feature dominates distance calculations
    purely due to scale differences (e.g. length in mm vs. fractional density).
    """
    rows, valid = [], []
    for roi in rois:
        row = [roi.get(k, np.nan) for k in keys]
        if all(np.isfinite(v) for v in row):
            rows.append(row)
            valid.append(roi)

    if not rows:
        return np.zeros((0, len(keys))), []

    mat = np.array(rows, dtype=np.float64)
    means = mat.mean(axis=0)
    stds = mat.std(axis=0)
    stds[stds == 0] = 1.0
    return (mat - means) / stds, valid


# ── Cross-plant matching ──────────────────────────────────────────────────────

def _border_penalty_factor(roi: Dict, border_margin: int) -> float:
    """
    Soft score multiplier that down-weights ROIs near the interior-crop boundary.

    ROIs whose nearest edge is ≥ border_margin pixels from any crop boundary
    receive factor 1.0 (no penalty).  Closer ROIs decay linearly to 0.0 at
    the crop edge.  Frame hardware and soil artifacts concentrate near edges,
    so this prevents border-adjacent false structure from winning matches.

    Uses interior-crop coordinates (y1, x1, y2, x2) and the stored crop dims.
    # TUNE: controlled by --border-margin; set to 0 to disable.
    """
    if border_margin <= 0:
        return 1.0
    y1 = roi.get("y1", 0)
    x1 = roi.get("x1", 0)
    y2 = roi.get("y2", 0)
    x2 = roi.get("x2", 0)
    ih = roi.get("interior_h", 99999)
    iw = roi.get("interior_w", 99999)
    min_dist = min(y1, x1, ih - y2, iw - x2)
    if min_dist >= border_margin:
        return 1.0
    return max(0.0, float(min_dist) / border_margin)


def match_rois_across_plants(
    all_rois: List[Dict],
    metric: str = "cosine",
    top_k: int = 3,
    border_margin: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every pair of images, find the top-K most similar ROIs.

    Parameters
    ----------
    metric : 'cosine' or 'euclidean'
    top_k : int — number of best matches per image pair
    border_margin : int
        ROIs within this many pixels of the interior-crop boundary have their
        similarity score multiplied by a linear decay factor (1.0 at margin,
        0.0 at edge).  Suppresses frame-adjacent artifacts from winning matches
        without hard-excluding them.  Set to 0 to disable.
        # TUNE: controlled by --border-margin.
    """
    keys = _feature_keys(all_rois)
    matrix, valid = _build_feature_matrix(all_rois, keys)

    if len(valid) == 0:
        return pd.DataFrame(), pd.DataFrame()

    image_names = sorted({r["image_name"] for r in valid})
    n = len(image_names)
    idx_by_img = {
        name: [i for i, r in enumerate(valid) if r["image_name"] == name]
        for name in image_names
    }

    sim_mat = np.eye(n)
    records: List[Dict] = []

    for i, img_a in enumerate(image_names):
        for j, img_b in enumerate(image_names):
            if i >= j:
                continue

            idx_a = idx_by_img[img_a]
            idx_b = idx_by_img[img_b]
            if not idx_a or not idx_b:
                continue

            mat_a = matrix[idx_a]
            mat_b = matrix[idx_b]

            if metric == "cosine":
                dists = cdist(mat_a, mat_b, metric="cosine")
                sims = 1.0 - dists
            else:
                dists = cdist(mat_a, mat_b, metric="euclidean")
                sims = 1.0 / (1.0 + dists)

            # ── Border penalty: down-weight edge-adjacent ROIs ────────────────
            if border_margin > 0:
                pen_a = np.array([_border_penalty_factor(valid[k], border_margin)
                                  for k in idx_a])
                pen_b = np.array([_border_penalty_factor(valid[k], border_margin)
                                  for k in idx_b])
                sims = sims * np.outer(pen_a, pen_b)

            best_sim = float(sims.max())
            sim_mat[i, j] = sim_mat[j, i] = best_sim

            flat_sorted = np.argsort(sims.ravel())[::-1][:top_k]
            for rank, flat_idx in enumerate(flat_sorted):
                ai, bi = np.unravel_index(flat_idx, sims.shape)
                ra = valid[idx_a[ai]]
                rb = valid[idx_b[bi]]
                records.append(dict(
                    image_a=img_a, image_b=img_b,
                    rank=rank + 1,
                    similarity=float(sims[ai, bi]),
                    roi_a_x1=ra["abs_x1"], roi_a_y1=ra["abs_y1"],
                    roi_a_x2=ra["abs_x2"], roi_a_y2=ra["abs_y2"],
                    roi_b_x1=rb["abs_x1"], roi_b_y1=rb["abs_y1"],
                    roi_b_x2=rb["abs_x2"], roi_b_y2=rb["abs_y2"],
                ))

    matches_df = (
        pd.DataFrame(records).sort_values("similarity", ascending=False)
        if records else pd.DataFrame()
    )
    sim_matrix_df = pd.DataFrame(sim_mat, index=image_names, columns=image_names)
    return matches_df, sim_matrix_df


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 4 — Output generation
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_density_map(
    mask: np.ndarray,
    gray_bg: np.ndarray,
    roi_size: int,
    stride: int,
    min_skeleton_density: float,
    debug_dir: str,
    stem: str,
) -> None:
    """
    Save a heatmap of skeleton pixel density per sliding-window tile.

    Computed from a display-scale skeleton (fast — one skeletonisation at ≤800 px).
    Tiles are coloured green (high density) through red (low density).
    Tiles below `min_skeleton_density` are outlined in red so you can see exactly
    which windows the pre-filter is discarding.  Use this to verify that the
    high-density zones align with visually obvious root regions before adjusting
    --min-roi-density.
    """
    h, w = mask.shape
    sc = min(1.0, 800 / max(h, w))
    dh = max(1, int(h * sc))
    dw = max(1, int(w * sc))

    # One skeletonisation at display scale — fast (~0.1 s)
    mask_small = cv2.resize(
        mask.astype(np.uint8) * 255, (dw, dh), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    skel_small = skeletonize(mask_small)

    ds_roi    = max(4, int(roi_size * sc))
    ds_stride = max(2, int(stride  * sc))

    bg = cv2.resize(gray_bg, (dw, dh))
    canvas = cv2.cvtColor((bg * 0.35).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cmap_fn = plt.colormaps["RdYlGn"]
    # Normalise colour so min_skeleton_density maps to ~mid-green
    norm_scale = max(min_skeleton_density * 3.0, 0.10)

    # First pass: fill tile colour
    for y in range(0, dh - ds_roi + 1, ds_stride):
        for x in range(0, dw - ds_roi + 1, ds_stride):
            dens = float(skel_small[y : y + ds_roi, x : x + ds_roi].sum()) / (ds_roi * ds_roi)
            rgba = cmap_fn(min(1.0, dens / norm_scale))
            color_bgr = (int(rgba[2] * 200), int(rgba[1] * 200), int(rgba[0] * 200))
            cv2.rectangle(canvas, (x, y), (x + ds_roi, y + ds_roi), color_bgr, -1)
            if ds_roi >= 20:
                cv2.putText(canvas, f"{dens:.3f}", (x + 2, y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 255), 1)

    # Second pass: outline below-threshold tiles in red
    for y in range(0, dh - ds_roi + 1, ds_stride):
        for x in range(0, dw - ds_roi + 1, ds_stride):
            dens = float(skel_small[y : y + ds_roi, x : x + ds_roi].sum()) / (ds_roi * ds_roi)
            if dens < min_skeleton_density:
                cv2.rectangle(canvas, (x, y), (x + ds_roi, y + ds_roi), (0, 0, 220), 1)

    out_path = Path(debug_dir) / f"{stem}_04_density_map.png"
    cv2.imwrite(str(out_path), canvas)


def save_roi_coordinates(all_rois: List[Dict], out_path: str) -> None:
    """Write every candidate ROI's coordinates and traits to CSV."""
    exclude = {"y1", "x1", "y2", "x2"}
    rows = [{k: v for k, v in roi.items() if k not in exclude} for roi in all_rois]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved ROI candidates   → {out_path}  ({len(rows)} ROIs)")


def save_similarity_matrix(sim_df: pd.DataFrame, out_path: str) -> None:
    """Write the pairwise inter-plant similarity matrix to CSV."""
    sim_df.to_csv(out_path)
    print(f"  Saved similarity matrix → {out_path}")


def save_match_details(
    all_rois: List[Dict],
    matches_df: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Write a fully exploded view of every matched ROI pair to CSV.

    Includes coordinates and all feature values for both matched ROIs side-by-side
    so you can see exactly which morphological traits drove each similarity score.
    """
    if matches_df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        print(f"  Saved match details     → {out_path}  (no matches)")
        return

    roi_index: Dict[Tuple, Dict] = {}
    for roi in all_rois:
        key = (roi["image_name"], roi["abs_x1"], roi["abs_y1"])
        roi_index[key] = roi

    feature_cols = [k for k in (all_rois[0].keys() if all_rois else [])
                    if k not in {"image_name", "y1", "x1", "y2", "x2",
                                 "abs_y1", "abs_x1", "abs_y2", "abs_x2"}]

    records = []
    for _, mrow in matches_df.iterrows():
        rec: Dict = {
            "image_a":    mrow["image_a"],
            "image_b":    mrow["image_b"],
            "rank":       int(mrow["rank"]),
            "similarity": float(mrow["similarity"]),
            "roi_a_x1": int(mrow["roi_a_x1"]), "roi_a_y1": int(mrow["roi_a_y1"]),
            "roi_a_x2": int(mrow["roi_a_x2"]), "roi_a_y2": int(mrow["roi_a_y2"]),
            "roi_b_x1": int(mrow["roi_b_x1"]), "roi_b_y1": int(mrow["roi_b_y1"]),
            "roi_b_x2": int(mrow["roi_b_x2"]), "roi_b_y2": int(mrow["roi_b_y2"]),
        }
        for prefix, img, x1, y1 in [
            ("roi_a", mrow["image_a"], mrow["roi_a_x1"], mrow["roi_a_y1"]),
            ("roi_b", mrow["image_b"], mrow["roi_b_x1"], mrow["roi_b_y1"]),
        ]:
            roi_data = roi_index.get((img, int(x1), int(y1)), {})
            for fc in feature_cols:
                rec[f"{prefix}_{fc}"] = roi_data.get(fc, np.nan)
        records.append(rec)

    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Saved match details     → {out_path}  ({len(records)} matched pairs)")


def _draw_skeleton_strands(
    gray_bg: np.ndarray,
    skeleton: np.ndarray,
    skel_diam_px: np.ndarray,
    dense_mat: np.ndarray,
    cmap,
    bins_px: List[float],
    display_size: int,
    max_display_radius: int = 4,
) -> Tuple[np.ndarray, float]:
    """
    Render the skeleton as colored strands on a grayscale background.

    Each skeleton pixel is drawn as a filled circle whose color encodes its
    diameter class (hue) and whose radius is proportional to the measured root
    radius (capped at `max_display_radius` for readability).  Dense-mat pixels
    are drawn in grey.

    # TUNE: max_display_radius — increase to show actual root width more literally
    """
    crop_h, crop_w = gray_bg.shape
    sc = display_size / max(crop_h, crop_w)
    dw = int(crop_w * sc)
    dh = int(crop_h * sc)

    bg = cv2.cvtColor(cv2.resize(gray_bg, (dw, dh)), cv2.COLOR_GRAY2BGR)
    canvas = (bg * 0.55).astype(np.uint8)

    n_bins = len(bins_px) - 1
    skel_ys, skel_xs = np.where(skeleton)

    bin_colors_bgr = []
    for bi in range(n_bins):
        rgba = cmap(bi / max(n_bins - 1, 1))
        bin_colors_bgr.append((int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255)))
    dense_color_bgr = (160, 160, 160)

    for y, x in zip(skel_ys, skel_xs):
        diam = skel_diam_px[y, x]
        if diam <= 0:
            continue
        cx = int(round(x * sc))
        cy = int(round(y * sc))
        r_display = min(max_display_radius, max(1, int(round(diam / 2 * sc))))
        if dense_mat[y, x]:
            color = dense_color_bgr
        else:
            bi = int(np.searchsorted(bins_px[1:-1], diam, side="right"))
            bi = min(bi, n_bins - 1)
            color = bin_colors_bgr[bi]
        cv2.circle(canvas, (cx, cy), r_display, color, -1)

    return canvas, sc


def save_visual_panel(
    images: List[RhizotronImage],
    masks: Dict[str, np.ndarray],
    matches_df: pd.DataFrame,
    out_path: str,
    scale_px_per_mm: float,
    diameter_bins_mm: List[float],
) -> None:
    """
    Produce a three-row comparison figure.

    Row 1 — Full image with cyan interior box and red matched-ROI box.
    Row 2 — Interior skeleton strand map (colors = diameter class, grey = dense mat)
             with yellow box showing the matched ROI location.
    Row 3 — Zoomed strand map of the matched ROI with feature annotations.
    """
    n = len(images)
    cmap = plt.colormaps["RdYlGn"].resampled(len(diameter_bins_mm) + 1)
    bin_labels = _diameter_bin_labels(diameter_bins_mm)
    bins_px = [0.0] + [b * scale_px_per_mm for b in diameter_bins_mm] + [np.inf]

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 13), squeeze=False)

    for col, rh in enumerate(images):
        iy1, ix1, iy2, ix2 = rh.interior_bbox
        h_full, w_full = rh.shape
        mask     = masks[rh.name]
        crop_h, crop_w = mask.shape

        best_row = _best_match_row(rh.name, matches_df)

        roi_on_crop: Optional[Tuple[int, int, int, int]] = None
        rx1_abs = ry1_abs = rx2_abs = ry2_abs = 0
        if best_row is not None:
            pfx = "roi_a" if best_row["image_a"] == rh.name else "roi_b"
            rx1_abs = int(best_row[f"{pfx}_x1"])
            ry1_abs = int(best_row[f"{pfx}_y1"])
            rx2_abs = int(best_row[f"{pfx}_x2"])
            ry2_abs = int(best_row[f"{pfx}_y2"])
            roi_on_crop = (
                ry1_abs - iy1, rx1_abs - ix1,
                ry2_abs - iy1, rx2_abs - ix1,
            )

        # ── Row 0: full image context ─────────────────────────────────────────
        ax0 = axes[0, col]
        display = rh.image_rgb.copy()
        cv2.rectangle(display, (ix1, iy1), (ix2, iy2), (0, 200, 255), 3)
        if best_row is not None:
            cv2.rectangle(display, (rx1_abs, ry1_abs), (rx2_abs, ry2_abs),
                          (255, 50, 50), 5)
            sim_val = best_row["similarity"]
            partner = (best_row["image_b"] if best_row["image_a"] == rh.name
                       else best_row["image_a"])
            ax0.set_title(
                f"{rh.name[-16:]}\nsim={sim_val:.3f} vs {partner[-14:]}",
                fontsize=6,
            )
        else:
            ax0.set_title(rh.name[-20:], fontsize=6)

        sc0 = 500 / max(h_full, w_full)
        ax0.imshow(cv2.resize(display, (int(w_full * sc0), int(h_full * sc0))))
        ax0.axis("off")

        # ── Row 1: skeleton strand map ────────────────────────────────────────
        ax1 = axes[1, col]
        DISP = 400
        ds_sc = DISP / max(crop_h, crop_w)
        dw_c, dh_c = max(1, int(crop_w * ds_sc)), max(1, int(crop_h * ds_sc))
        mask_small = cv2.resize(mask.astype(np.uint8) * 255, (dw_c, dh_c),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        skel_s, sdiam_s, dm_s = RootSegmenter.skeletonize_and_measure(
            mask_small, scale_px_per_mm * ds_sc, min_segment_length_px=0
        )

        strand_rgb, sc1 = _draw_skeleton_strands(
            cv2.resize(rh.interior_gray, (dw_c, dh_c)),
            skel_s, sdiam_s, dm_s, cmap, bins_px, display_size=DISP,
        )
        ax1.imshow(cv2.cvtColor(strand_rgb, cv2.COLOR_BGR2RGB))

        if roi_on_crop is not None:
            ry1c, rx1c, ry2c, rx2c = roi_on_crop
            rect = mpatches.Rectangle(
                (rx1c * ds_sc * sc1, ry1c * ds_sc * sc1),
                (rx2c - rx1c) * ds_sc * sc1,
                (ry2c - ry1c) * ds_sc * sc1,
                linewidth=2, edgecolor="yellow", facecolor="none",
            )
            ax1.add_patch(rect)

        ax1.set_title(
            "Skeleton strands  [color=diam class, grey=dense mat]\n"
            "yellow = matched ROI",
            fontsize=6,
        )
        ax1.axis("off")

        # ── Row 2: zoomed matched ROI ─────────────────────────────────────────
        ax2 = axes[2, col]
        if roi_on_crop is not None:
            ry1c, rx1c, ry2c, rx2c = roi_on_crop
            ry1c = max(0, ry1c); rx1c = max(0, rx1c)
            ry2c = min(crop_h, ry2c); rx2c = min(crop_w, rx2c)

            roi_gray = rh.interior_gray[ry1c:ry2c, rx1c:rx2c]
            roi_mask = mask[ry1c:ry2c, rx1c:rx2c]

            if roi_gray.size > 0:
                roi_skel, roi_sdiam, roi_dm = RootSegmenter.skeletonize_and_measure(
                    roi_mask, scale_px_per_mm, min_segment_length_px=0
                )
                zoom_rgb, _ = _draw_skeleton_strands(
                    roi_gray, roi_skel, roi_sdiam, roi_dm,
                    cmap, bins_px, display_size=300,
                )
                ax2.imshow(cv2.cvtColor(zoom_rgb, cv2.COLOR_BGR2RGB))

                roi_prim, roi_lat, _ = classify_primary_lateral(
                    roi_skel, roi_sdiam, scale_px_per_mm
                )
                feats = extract_roi_features(
                    roi_mask, roi_skel, roi_sdiam, roi_dm,
                    roi_prim, roi_lat, scale_px_per_mm, diameter_bins_mm,
                )
                dm_pct = feats["dense_mat_fraction"] * 100
                annotation = (
                    f"density={feats['root_density']:.2f}\n"
                    f"length={feats['total_length_mm']:.0f}mm\n"
                    f"laterals={feats['lateral_count']:.0f}\n"
                    f"branch_dens={feats['branching_density']:.1f}/cm\n"
                    f"mean_diam={feats['mean_diameter_mm']:.2f}mm\n"
                    f"dense_mat={dm_pct:.0f}%"
                )
                ax2.text(
                    0.02, 0.98, annotation,
                    transform=ax2.transAxes,
                    fontsize=5.5, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                )
            ax2.set_title("Matched ROI — skeleton strands", fontsize=6)
        else:
            ax2.set_title("No match", fontsize=6)
            ax2.text(0.5, 0.5, "—", transform=ax2.transAxes,
                     ha="center", va="center")
        ax2.axis("off")

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=cmap(i / max(len(bin_labels) - 1, 1)), label=lbl)
        for i, lbl in enumerate(bin_labels)
    ]
    legend_patches.append(
        mpatches.Patch(color=(0.63, 0.63, 0.63), label="dense mat (merged)")
    )
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=len(legend_patches), fontsize=8,
        title="Diameter class  (color = skeleton centerline only)",
    )

    plt.suptitle("Rhizotron ROI Comparison Panel", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved visual panel      → {out_path}")


def _best_match_row(
    img_name: str, matches_df: pd.DataFrame
) -> Optional[Dict]:
    """Return the single highest-similarity match record involving this image."""
    if matches_df.empty:
        return None
    relevant = matches_df[
        (matches_df["image_a"] == img_name) | (matches_df["image_b"] == img_name)
    ]
    if relevant.empty:
        return None
    return relevant.nlargest(1, "similarity").iloc[0].to_dict()


# ─────────────────────────────────────────────────────────────────────────────
#  Annotation tool (OpenCV-based; avoids matplotlib backend conflicts)
# ─────────────────────────────────────────────────────────────────────────────

# RGB colours and names for the three annotation classes (matplotlib uses RGB)
ANNOT_CLASSES: Dict[int, Tuple[str, Tuple[float, float, float]]] = {
    0: ("root",        (0.20, 0.85, 0.20)),   # green
    1: ("pore_edge",   (0.95, 0.20, 0.20)),   # red
    2: ("background",  (0.25, 0.60, 0.95)),   # blue
}
DEFAULT_PATCH_SIZE: int = 200
DEFAULT_LIBRARY_PATH: str = "~/rhizotron_annotation_library"


def _show_plt(fig: "plt.Figure") -> None:
    """Open a matplotlib figure and block until closed; give clear error if no display."""
    try:
        plt.show()
    except Exception as exc:
        print(
            f"\nERROR: cannot open window: {exc}\n"
            "Ensure a graphical display is available (DISPLAY env var set).\n"
            "Over SSH use:  ssh -X user@host",
            file=sys.stderr,
        )
        plt.close(fig)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Annotation library — persistent archive across sessions / projects
# ─────────────────────────────────────────────────────────────────────────────

def _archive_to_library(
    ann_dir: Path,
    library_path: Path,
    operator: str,
    notes: str,
    image_names: List[str],
) -> Optional[Path]:
    """
    Copy all JSON files from *ann_dir* into a new timestamped subfolder of
    *library_path* and write a meta.json alongside them.  Returns the path of
    the new subfolder, or None if there were no annotations to archive.
    The library is append-only; nothing is ever overwritten.
    """
    import shutil, datetime

    json_files = [ann_dir / f"{n}.json" for n in image_names
                  if (ann_dir / f"{n}.json").exists()]
    if not json_files:
        return None

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = library_path / ts
    dest.mkdir(parents=True, exist_ok=True)

    point_counts: Dict[str, int] = {}
    for jf in json_files:
        shutil.copy2(jf, dest / jf.name)
        with open(jf) as f:
            for ann in json.load(f):
                k = str(ann.get("cls", "?"))
                point_counts[k] = point_counts.get(k, 0) + 1

    meta = {
        "date":         datetime.datetime.now().isoformat(),
        "operator":     operator,
        "notes":        notes,
        "session_dir":  str(ann_dir.resolve()),
        "images":       image_names,
        "point_counts": point_counts,
    }
    with open(dest / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Archived {len(json_files)} annotation file(s) → {dest}")
    return dest


def _list_library(library_path: Path) -> None:
    """Print a summary table of all archived annotation sessions."""
    library_path = library_path.expanduser()
    if not library_path.exists():
        print(f"Library not found: {library_path}")
        return

    sessions = sorted(
        [d for d in library_path.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not sessions:
        print(f"No sessions in library: {library_path}")
        return

    cls_names = {0: "root", 1: "pore", 2: "bg"}
    header = f"{'Date/Time':<22} {'Operator':<14} {'Images':>6} {'Root':>6} {'Pore':>6} {'BG':>6}  Notes"
    print(f"\nAnnotation library: {library_path}")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for sess in sessions:
        mf = sess / "meta.json"
        if not mf.exists():
            print(f"{sess.name:<22}  (no meta.json)")
            continue
        with open(mf) as f:
            m = json.load(f)
        pc = m.get("point_counts", {})
        root = pc.get("0", 0)
        pore = pc.get("1", 0)
        bg   = pc.get("2", 0)
        imgs = len(m.get("images", []))
        op   = m.get("operator", "?")[:14]
        note = (m.get("notes", "") or "")[:40]
        print(f"{sess.name:<22} {op:<14} {imgs:>6} {root:>6} {pore:>6} {bg:>6}  {note}")
    print("─" * len(header))


def _load_library_annotations(library_path: Path) -> List[Dict]:
    """
    Return a flat list of all annotation dicts from every session in the
    library.  Each dict retains its original keys (y, x, cls, …).
    """
    library_path = library_path.expanduser()
    if not library_path.exists():
        return []
    all_anns: List[Dict] = []
    for sess in sorted(library_path.iterdir()):
        if not sess.is_dir():
            continue
        for jf in sorted(sess.glob("*.json")):
            if jf.name == "meta.json":
                continue
            try:
                with open(jf) as f:
                    all_anns.extend(json.load(f))
            except Exception:
                pass
    return all_anns


class AnnotationTool:
    """
    Two-stage patch-based annotation interface.

    Stage 1 — Patch picker (per image)
    ------------------------------------
    A scaled-down overview of the interior crop is shown with the segmentation
    mask overlay.  Click anywhere to drop a patch-centre marker.  Markers are
    numbered in order.  Press Enter (or Return) to accept and move to Stage 2.
    Press 's' to skip this image entirely.  Press 'q' to quit.

    Stage 2 — Patch annotator (per patch)
    ---------------------------------------
    Each selected patch is shown zoomed in with the segmentation overlay.
    Label individual points:
      Left-click   → class 1  (real root)
      Right-click  → class 2  (soil pore edge / false positive)
      'b'          → class 3  (background soil)  — places point at last cursor pos
      Ctrl+Z / u   → undo last point
      Enter        → confirm patch and continue
      'q'          → quit immediately (partial patches already saved)

    A running class tally is shown in the window title.

    Persistence
    -----------
    Annotations are saved after every confirmed patch so an interrupted session
    loses nothing.  On restart, images that already have annotations are shown
    with their existing point counts; you can add more patches or skip them.

    Coordinates in JSON files are interior-crop pixel space (same as segmenter).
    """

    def __init__(
        self,
        image_paths: List[str],
        annotation_dir: str,
        scale_px_per_mm: float,
        seg_kwargs: dict,
        patch_size: int = DEFAULT_PATCH_SIZE,
        operator: str = "unknown",
        notes: str = "",
        library_path: str = DEFAULT_LIBRARY_PATH,
    ) -> None:
        self.image_paths = image_paths
        self.ann_dir = Path(annotation_dir)
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale_px_per_mm
        self.patch_size = patch_size
        self.operator = operator
        self.notes = notes
        self.library_path = Path(library_path).expanduser()

        print("Segmenting images for annotation preview (Frangi gate disabled)...")
        self._crops:        Dict[str, np.ndarray] = {}   # grayscale (for segmenter)
        self._color_crops:  Dict[str, np.ndarray] = {}   # RGB (for display)
        self._masks:        Dict[str, np.ndarray] = {}
        self._skels:        Dict[str, np.ndarray] = {}   # display-scale skeleton
        self._sc:           Dict[str, float]       = {}  # display scale factor
        for path in image_paths:
            rh   = RhizotronImage(path, scale_px_per_mm)
            gray = rh.interior_gray.copy()
            rgb  = rh.interior_rgb.copy()
            kw   = dict(seg_kwargs, vesselness_threshold=0.0)
            mask = RootSegmenter(**kw).segment(gray, scale_px_per_mm)
            h, w = gray.shape
            sc   = min(1.0, 900 / max(h, w))
            dh, dw = max(1, int(h * sc)), max(1, int(w * sc))
            mask_s = cv2.resize(mask.astype(np.uint8) * 255, (dw, dh),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            self._crops[rh.name]       = gray
            self._color_crops[rh.name] = rgb
            self._masks[rh.name]       = mask_s
            self._skels[rh.name]       = skeletonize(mask_s)
            self._sc[rh.name]          = sc
            print(f"  {rh.name}  {w}×{h} → display {dw}×{dh}")
        self._names = [Path(p).stem for p in image_paths]

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _ann_path(self, name: str) -> Path:
        return self.ann_dir / f"{name}.json"

    def _load(self, name: str) -> List[Dict]:
        fp = self._ann_path(name)
        if fp.exists():
            with open(fp) as f:
                return json.load(f)
        return []

    def _save(self, name: str, anns: List[Dict]) -> None:
        with open(self._ann_path(name), "w") as f:
            json.dump(anns, f)

    # ── Composite image builder ───────────────────────────────────────────────

    def _make_overview(self, name: str) -> np.ndarray:
        """Grayscale crop + green mask tint + orange skeleton, as uint8 RGB."""
        gray   = self._crops[name]
        mask_s = self._masks[name]
        skel_s = self._skels[name]
        sc     = self._sc[name]
        dh, dw = mask_s.shape
        bg      = cv2.resize(gray, (dw, dh))
        display = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
        display[mask_s] = np.clip(
            display[mask_s].astype(np.int32) // 2 + [0, 60, 0], 0, 255
        ).astype(np.uint8)
        display[skel_s] = [255, 130, 0]
        return display

    def _make_patch_display(
        self, name: str, cy: int, cx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Return (raw_rgb, skel_bool, y0, x0) at full interior-crop resolution,
        centred on (cy, cx), clamped to image bounds.
        raw_rgb  — original photograph pixels, no overlay
        skel_bool — boolean skeleton mask at patch resolution
        y0, x0   — top-left of patch in interior-crop pixel coordinates
        """
        gray  = self._crops[name]
        color = self._color_crops[name]
        h, w  = gray.shape
        ps    = self.patch_size
        y0    = int(np.clip(cy - ps // 2, 0, h - ps))
        x0    = int(np.clip(cx - ps // 2, 0, w - ps))
        y1, x1 = y0 + ps, x0 + ps

        raw_rgb = color[y0:y1, x0:x1].copy()
        assert raw_rgb.ndim == 3 and raw_rgb.shape[2] == 3, (
            f"_make_patch_display: expected (H,W,3) RGB array, "
            f"got shape {raw_rgb.shape} — check _color_crops loading"
        )

        # Upsample display-scale mask → patch resolution for skeleton overlay
        sc    = self._sc[name]
        ms    = self._masks[name]
        ds_y0 = int(y0 * sc); ds_x0 = int(x0 * sc)
        ds_y1 = min(int(y1 * sc), ms.shape[0])
        ds_x1 = min(int(x1 * sc), ms.shape[1])
        mask_crop = ms[ds_y0:ds_y1, ds_x0:ds_x1]
        mask_full = cv2.resize(
            mask_crop.astype(np.uint8) * 255, (ps, ps),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        skel_full = skeletonize(mask_full)

        return raw_rgb, skel_full, y0, x0

    # ── Stage 1: Patch picker ─────────────────────────────────────────────────

    def _pick_patches(self, name: str) -> Optional[List[Tuple[int, int]]]:
        """
        Show overview; user clicks patch centres.  Returns list of (cy, cx) in
        interior-crop pixel space, or None if the user chose to quit.
        's' → skip image (returns empty list).
        """
        overview = self._make_overview(name)
        sc       = self._sc[name]
        ps_d     = max(4, int(self.patch_size * sc))  # patch size in display px
        existing = self._load(name)
        n_exist  = len(existing)

        centers: List[Tuple[int, int]] = []   # (cy, cx) in crop coords
        done     = [False]
        skipped  = [False]
        quitting = [False]

        fig, ax = plt.subplots(figsize=(13, 8))
        fig.patch.set_facecolor("#111111")
        ax.imshow(overview, aspect="auto", interpolation="nearest")
        ax.axis("off")

        patch_rects: List = []

        def _redraw():
            for r in patch_rects:
                r.remove()
            patch_rects.clear()
            for i, (cy, cx) in enumerate(centers):
                dy_d = cy * sc;  dx_d = cx * sc
                rect = plt.Rectangle(
                    (dx_d - ps_d / 2, dy_d - ps_d / 2), ps_d, ps_d,
                    linewidth=1.5, edgecolor="yellow", facecolor="none", zorder=4,
                )
                ax.add_patch(rect)
                patch_rects.append(rect)
                ax.text(dx_d, dy_d - ps_d / 2 - 4, str(i + 1),
                        color="yellow", fontsize=7, ha="center", va="bottom",
                        zorder=5)
            counts = [sum(1 for a in existing if a["cls"] == c) for c in range(3)]
            n_new  = len(centers)
            ax.set_title(
                f"PATCH PICKER — {name}   ({n_exist} existing pts: "
                f"root={counts[0]} pore={counts[1]} bg={counts[2]})\n"
                f"{n_new} patch{'es' if n_new != 1 else ''} selected   "
                f"Left-click=add   Right-click=remove last   "
                f"Enter=confirm   s=skip image   q=quit",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111", alpha=0.9),
            )
            fig.canvas.draw_idle()

        def _on_press(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                h_c, w_c = self._crops[name].shape
                cy = int(np.clip(event.ydata / sc, 0, h_c - 1))
                cx = int(np.clip(event.xdata / sc, 0, w_c - 1))
                centers.append((cy, cx))
                _redraw()
            elif event.button == 3 and centers:
                centers.pop()
                _redraw()

        def _on_key(event):
            if event.key in ("enter", "return"):
                done[0] = True
                plt.close(fig)
            elif event.key == "s":
                skipped[0] = True
                plt.close(fig)
            elif event.key == "q":
                quitting[0] = True
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", _on_press)
        fig.canvas.mpl_connect("key_press_event",    _on_key)
        _redraw()
        plt.tight_layout()
        _show_plt(fig)

        if quitting[0]:
            return None
        if skipped[0]:
            return []
        return centers

    # ── Stage 2: Patch annotator ─────────────────────────────────────────────

    # Overlay mode names shown in the title bar
    _OVERLAY_MODES = ["raw", "raw + skeleton (40%)", "skeleton only"]

    def _annotate_patch(
        self,
        name: str,
        cy: int,
        cx: int,
        patch_idx: int,
        n_patches: int,
    ) -> Optional[List[Dict]]:
        """
        Zoomed-in annotation of one patch.  Returns list of new annotation dicts
        (in interior-crop coordinates), or None if the user quit.
        Press 'o' to cycle overlay: raw → raw+skeleton → skeleton only.
        """
        raw_rgb, skel_full, y0, x0 = self._make_patch_display(name, cy, cx)
        ps = self.patch_size

        _SKEL_COLOR = np.array([255, 130, 0], dtype=np.float32)

        def _compose(mode: int) -> np.ndarray:
            if mode == 0:
                return raw_rgb
            if mode == 1:
                out = raw_rgb.copy().astype(np.float32)
                out[skel_full] = out[skel_full] * 0.60 + _SKEL_COLOR * 0.40
                return out.clip(0, 255).astype(np.uint8)
            # mode == 2: skeleton on black
            out = np.zeros((ps, ps, 3), dtype=np.uint8)
            out[skel_full] = [255, 130, 0]
            return out

        new_anns:    List[Dict] = []
        done         = [False]
        quitting     = [False]
        last_xy      = [None, None]
        overlay_mode = [0]

        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor("#111111")
        im = ax.imshow(_compose(0), aspect="equal", interpolation="nearest",
                       extent=[0, ps, ps, 0])
        ax.set_xlim(0, ps); ax.set_ylim(ps, 0)
        ax.axis("off")
        dots: List = []

        def _counts():
            return [sum(1 for a in new_anns if a["cls"] == c) for c in range(3)]

        def _redraw_title():
            c    = _counts()
            mode = self._OVERLAY_MODES[overlay_mode[0]]
            ax.set_title(
                f"PATCH {patch_idx}/{n_patches} — {name}   "
                f"(centre {cx},{cy})   [{mode}]\n"
                f"root={c[0]}  pore={c[1]}  bg={c[2]}   "
                f"Left=root   Right=pore   b=bg   "
                f"o=toggle overlay   Ctrl+Z/u=undo   Enter=confirm   q=quit",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111", alpha=0.9),
            )
            fig.canvas.draw_idle()

        def _place(px_d, py_d, cls):
            px_d  = float(np.clip(px_d, 0, ps - 1))
            py_d  = float(np.clip(py_d, 0, ps - 1))
            iy_ic = y0 + int(py_d)
            ix_ic = x0 + int(px_d)
            new_anns.append({"y": iy_ic, "x": ix_ic, "cls": cls,
                             "patch_y": y0, "patch_x": x0, "patch_size": ps})
            r, g_c, b_c = ANNOT_CLASSES[cls][1]
            dot, = ax.plot(px_d, py_d, "o",
                           color=(r, g_c, b_c), markersize=10,
                           markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            dots.append(dot)
            _redraw_title()

        def _on_press(event):
            if event.inaxes != ax:
                return
            last_xy[0], last_xy[1] = event.xdata, event.ydata
            if event.button == 1:
                _place(event.xdata, event.ydata, 0)
            elif event.button == 3:
                _place(event.xdata, event.ydata, 1)

        def _on_motion(event):
            if event.inaxes == ax:
                last_xy[0], last_xy[1] = event.xdata, event.ydata

        def _on_key(event):
            if event.key == "b":
                if last_xy[0] is not None:
                    _place(last_xy[0], last_xy[1], 2)
            elif event.key == "o":
                overlay_mode[0] = (overlay_mode[0] + 1) % 3
                im.set_data(_compose(overlay_mode[0]))
                _redraw_title()
            elif event.key in ("ctrl+z", "u"):
                if new_anns:
                    new_anns.pop()
                    if dots:
                        dots[-1].remove()
                        dots.pop()
                    _redraw_title()
            elif event.key in ("enter", "return"):
                done[0] = True
                plt.close(fig)
            elif event.key == "q":
                quitting[0] = True
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event",  _on_press)
        fig.canvas.mpl_connect("motion_notify_event", _on_motion)
        fig.canvas.mpl_connect("key_press_event",     _on_key)
        _redraw_title()
        plt.tight_layout()
        _show_plt(fig)

        if quitting[0]:
            return None
        return new_anns

    # ── Top-level orchestrator ────────────────────────────────────────────────

    def run(self) -> None:
        """
        Cycle through all images.  For each image:
          1. Show patch picker and collect 0–N patch centres.
          2. For each patch centre, run the zoomed annotator.
          3. Append new points to the image's JSON file immediately.
        """
        for path in self.image_paths:
            name     = self._names[self.image_paths.index(path)]
            existing = self._load(name)
            n_exist  = len(existing)

            # Print per-image status so the user can see what already exists
            if n_exist:
                counts = [sum(1 for a in existing if a["cls"] == c)
                          for c in range(3)]
                print(
                    f"\n[{name}]  {n_exist} existing annotations "
                    f"(root={counts[0]} pore={counts[1]} bg={counts[2]})"
                )
            else:
                print(f"\n[{name}]  no existing annotations")

            # Stage 1: pick patch centres
            centers = self._pick_patches(name)
            if centers is None:      # user pressed q
                print("Quitting annotation session.")
                return
            if not centers:          # user pressed s
                print(f"  Skipped {name}.")
                continue

            # Stage 2: annotate each patch, saving incrementally
            for i, (cy, cx) in enumerate(centers):
                print(f"  Patch {i+1}/{len(centers)} centre=({cx},{cy})")
                new_anns = self._annotate_patch(name, cy, cx, i + 1, len(centers))
                if new_anns is None:     # user pressed q inside a patch
                    print("Quitting annotation session.")
                    return

                # Merge with existing and save immediately
                existing = self._load(name)
                existing.extend(new_anns)
                self._save(name, existing)
                print(
                    f"    +{len(new_anns)} points  "
                    f"(total {len(existing)} for this image)"
                )

        print("\nAnnotation session complete.")
        _archive_to_library(
            ann_dir=self.ann_dir,
            library_path=self.library_path,
            operator=self.operator,
            notes=self.notes,
            image_names=self._names,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Corrective annotation tool (brush-based corrections on segmentation output)
# ─────────────────────────────────────────────────────────────────────────────

class CorrectionTool:
    """
    Interactive brush tool that lets the user paint corrections directly on
    top of the current segmentation result.

    Workflow
    --------
    1.  Run the normal pipeline once to generate masks and skeletons.
    2.  Launch this tool (--correct).  Each image opens in a matplotlib window
        showing the colour photograph with a segmentation overlay.
    3.  Paint corrections with the mouse:
          Left-drag   → root      (class 0, green dots)
          Middle-drag → pore_edge (class 1, yellow dots)
          Right-drag  → background (class 2, red dots)
        Mouse-wheel (or +/-) adjusts brush radius.
    4.  Press 'n' / Enter to save and move to the next image.
        Press 'u' to undo the last stroke.
        Press 'c' to clear all corrections on the current image.
        Press 'q' to quit and save everything so far.
    5.  Corrections are appended to the existing annotations/<stem>.json files
        and archived to the library on exit.

    This is equivalent to RootPainter-style corrective annotation but uses the
    existing annotation format so the same --train workflow applies.
    """

    _CLS_COLOR = {0: (0.0, 0.9, 0.0, 0.8), 1: (1.0, 0.9, 0.0, 0.8), 2: (0.9, 0.1, 0.1, 0.8)}
    _CLS_LABEL = {0: "root (L-drag)", 1: "pore_edge (M-drag)", 2: "bg (R-drag)"}
    _BUTTON_TO_CLS = {1: 0, 2: 1, 3: 2}

    def __init__(
        self,
        image_paths: List[str],
        annotation_dir: str,
        scale_px_per_mm: float,
        seg_kwargs: dict,
        operator: str = "unknown",
        notes: str = "",
        library_path: str = DEFAULT_LIBRARY_PATH,
        brush_radius: int = 8,
    ) -> None:
        self.image_paths = image_paths
        self.ann_dir = Path(annotation_dir)
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale_px_per_mm
        self.operator = operator
        self.notes = notes
        self.library_path = Path(library_path).expanduser()
        self.brush_radius = brush_radius

        print("Segmenting images for correction overlay...")
        self._crops:       Dict[str, np.ndarray] = {}   # grayscale (interior crop)
        self._color_crops: Dict[str, np.ndarray] = {}   # RGB
        self._masks:       Dict[str, np.ndarray] = {}   # display-scale bool mask
        self._skels:       Dict[str, np.ndarray] = {}   # display-scale bool skeleton
        self._sc:          Dict[str, float]       = {}  # display scale factor
        for path in image_paths:
            rh   = RhizotronImage(path, scale_px_per_mm)
            gray = rh.interior_gray.copy()
            rgb  = rh.interior_rgb.copy()
            kw   = dict(seg_kwargs, vesselness_threshold=0.0)
            mask = RootSegmenter(**kw).segment(gray, scale_px_per_mm)
            h, w = gray.shape
            sc   = min(1.0, 900 / max(h, w))
            dh, dw = max(1, int(h * sc)), max(1, int(w * sc))
            mask_s = cv2.resize(mask.astype(np.uint8) * 255, (dw, dh),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            self._crops[rh.name]       = gray
            self._color_crops[rh.name] = rgb
            self._masks[rh.name]       = mask_s
            self._skels[rh.name]       = skeletonize(mask_s)
            self._sc[rh.name]          = sc
            print(f"  {rh.name}  {w}×{h} → display {dw}×{dh}")
        self._names = [Path(p).stem for p in image_paths]

    def _ann_path(self, name: str) -> Path:
        return self.ann_dir / f"{name}.json"

    def _load(self, name: str) -> List[Dict]:
        fp = self._ann_path(name)
        if fp.exists():
            with open(fp) as f:
                return json.load(f)
        return []

    def _save(self, name: str, anns: List[Dict]) -> None:
        with open(self._ann_path(name), "w") as f:
            json.dump(anns, f)

    def _make_bg(self, name: str) -> np.ndarray:
        """Return an RGB display image with mask + skeleton overlay."""
        gray   = self._crops[name]
        mask_s = self._masks[name]
        skel_s = self._skels[name]
        dh, dw = mask_s.shape
        bg = cv2.resize(gray, (dw, dh))
        display = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
        display[mask_s] = np.clip(
            display[mask_s].astype(np.int32) // 2 + [0, 60, 0], 0, 255
        ).astype(np.uint8)
        display[skel_s] = [255, 130, 0]
        return display

    def _correct_image(self, name: str) -> bool:
        """Open the correction window for one image. Returns False if user quit."""
        sc = self._sc[name]
        h_c, w_c = self._crops[name].shape
        bg = self._make_bg(name)

        existing = self._load(name)
        new_anns: List[Dict] = []      # corrections added in this session
        history:  List[List[Dict]] = []  # for undo (each entry = one stroke)

        fig, ax = plt.subplots(figsize=(13, 8))
        fig.patch.set_facecolor("#111111")
        im_handle = ax.imshow(bg, aspect="auto", interpolation="nearest")
        ax.axis("off")

        dot_artists: List = []        # scatter artists for each class
        scatter_data: Dict[int, List[Tuple[float, float]]] = {0: [], 1: [], 2: []}
        scatters: Dict[int, object] = {}
        for cls, color in self._CLS_COLOR.items():
            scatters[cls] = ax.scatter([], [], c=[color], s=20, zorder=5)

        brush_state = {"radius": self.brush_radius}
        painting   = [False]
        cur_cls    = [0]
        cur_stroke: List[Dict] = []
        quitting   = [False]

        def _update_title():
            n_total = len(existing) + len(new_anns)
            counts = {c: sum(1 for a in (existing + new_anns) if a["cls"] == c) for c in range(3)}
            ax.set_title(
                f"CORRECT — {name}   "
                f"root={counts[0]}  pore={counts[1]}  bg={counts[2]}  total={n_total}\n"
                f"L=root  M=pore  R=bg  |  brush r={brush_state['radius']} px  "
                f"|  +/- = resize  u=undo  c=clear  n/Enter=save+next  q=quit",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111", alpha=0.9),
            )

        def _refresh_dots():
            for cls, sc_art in scatters.items():
                all_anns = [a for a in (existing + new_anns) if a["cls"] == cls]
                if all_anns:
                    xs = [a["x"] * sc for a in all_anns]
                    ys = [a["y"] * sc for a in all_anns]
                else:
                    xs, ys = [], []
                sc_art.set_offsets(np.c_[xs, ys] if xs else np.empty((0, 2)))
            _update_title()
            fig.canvas.draw_idle()

        def _paint_at(display_x: float, display_y: float, cls: int):
            r = brush_state["radius"]
            # Sample pixels within brush radius
            iy_d = int(round(display_y)); ix_d = int(round(display_x))
            dh, dw = self._masks[name].shape
            pts_added = []
            for dy in range(-r, r + 1, max(1, r // 4)):
                for dx in range(-r, r + 1, max(1, r // 4)):
                    if dy * dy + dx * dx > r * r:
                        continue
                    py_d = np.clip(iy_d + dy, 0, dh - 1)
                    px_d = np.clip(ix_d + dx, 0, dw - 1)
                    # Convert back to full-res crop coordinates
                    py_c = int(np.clip(py_d / sc, 0, h_c - 1))
                    px_c = int(np.clip(px_d / sc, 0, w_c - 1))
                    ann = {"y": py_c, "x": px_c, "cls": cls}
                    pts_added.append(ann)
            new_anns.extend(pts_added)
            cur_stroke.extend(pts_added)

        def _on_press(event):
            if event.inaxes != ax:
                return
            btn = event.button
            if btn not in self._BUTTON_TO_CLS:
                return
            cur_cls[0] = self._BUTTON_TO_CLS[btn]
            painting[0] = True
            cur_stroke.clear()
            _paint_at(event.xdata, event.ydata, cur_cls[0])
            _refresh_dots()

        def _on_motion(event):
            if not painting[0] or event.inaxes != ax:
                return
            _paint_at(event.xdata, event.ydata, cur_cls[0])
            _refresh_dots()

        def _on_release(event):
            if painting[0] and cur_stroke:
                history.append(list(cur_stroke))
                cur_stroke.clear()
            painting[0] = False

        def _on_scroll(event):
            delta = 1 if event.button == "up" else -1
            brush_state["radius"] = max(1, brush_state["radius"] + delta)
            _update_title()
            fig.canvas.draw_idle()

        done   = [False]
        def _on_key(event):
            if event.key in ("n", "enter", "return"):
                done[0] = True
                plt.close(fig)
            elif event.key == "q":
                quitting[0] = True
                plt.close(fig)
            elif event.key == "u" and history:
                stroke = history.pop()
                for ann in stroke:
                    if ann in new_anns:
                        new_anns.remove(ann)
                _refresh_dots()
            elif event.key == "c":
                new_anns.clear()
                history.clear()
                _refresh_dots()
            elif event.key in ("+", "="):
                brush_state["radius"] = min(50, brush_state["radius"] + 2)
                _update_title(); fig.canvas.draw_idle()
            elif event.key in ("-", "_"):
                brush_state["radius"] = max(1, brush_state["radius"] - 2)
                _update_title(); fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event",   _on_press)
        fig.canvas.mpl_connect("motion_notify_event",  _on_motion)
        fig.canvas.mpl_connect("button_release_event", _on_release)
        fig.canvas.mpl_connect("scroll_event",         _on_scroll)
        fig.canvas.mpl_connect("key_press_event",      _on_key)

        _refresh_dots()
        plt.tight_layout()
        _show_plt(fig)

        if new_anns:
            merged = existing + new_anns
            self._save(name, merged)
            print(f"  Saved {len(new_anns)} new annotations for {name} "
                  f"(total {len(merged)})")

        return not quitting[0]

    def run(self) -> None:
        for name in self._names:
            print(f"\n── Correcting {name} ──")
            if not self._correct_image(name):
                print("  Quit.")
                break
        _archive_to_library(
            ann_dir=self.ann_dir,
            library_path=self.library_path,
            operator=self.operator,
            notes=self.notes,
            image_names=self._names,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Classifier feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

_N_GABOR_ORIENTATIONS = 4
_GABOR_LAMBDAS: Tuple[float, float] = (6.0, 14.0)   # sinusoidal wavelengths (fine, coarse) in px
_CLASSIFIER_PATCH = 32


def _gabor_features(patch: np.ndarray) -> np.ndarray:
    """
    16 features: (mean |response|, energy) × 4 orientations × 2 wavelengths.

    Root strands produce strong Gabor response at ONE orientation (the strand
    direction) and weak response orthogonally.  Soil pore edges are curved, so
    their response is spread across multiple orientations.  The
    orientation-continuity ratio (max orient energy / total) is the key
    discriminant: roots ≈ 0.7–0.9, pore edges ≈ 0.3–0.5.
    """
    patch_f = patch.astype(np.float32) / 255.0
    feats: List[float] = []
    for i in range(_N_GABOR_ORIENTATIONS):
        theta = i * np.pi / _N_GABOR_ORIENTATIONS
        for lam in _GABOR_LAMBDAS:
            kern = cv2.getGaborKernel(
                (19, 19), sigma=4.0, theta=theta,
                lambd=lam, gamma=0.5, psi=0.0, ktype=cv2.CV_32F,
            )
            resp = cv2.filter2D(patch_f, cv2.CV_32F, kern)
            feats.append(float(np.abs(resp).mean()))
            feats.append(float((resp ** 2).mean()))
    return np.array(feats, dtype=np.float32)   # 16 values


def _skeleton_curvature(skel_patch: np.ndarray) -> float:
    """
    Estimate curvature magnitude from skeleton pixels in a small patch.

    Fits a 2nd-degree polynomial to the skeleton pixel coordinates.  The
    leading coefficient |2a| from y = ax² + bx + c gives curvature:
    low → straight strand (real root); high → sharply bent loop (pore edge).

    Returns 0.0 if fewer than 3 skeleton pixels are present.
    """
    ys, xs = np.where(skel_patch)
    if len(xs) < 3:
        return 0.0
    try:
        # Fit along the longer axis to avoid vertical-line degeneracy.
        # RankWarning is expected for nearly-straight segments and is harmless.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if xs.std() >= ys.std():
                a = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), 2)[0]
            else:
                a = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 2)[0]
        return float(min(abs(2.0 * a), 5.0))
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def _comp_loop_and_length(skeleton: np.ndarray, cy: int, cx: int) -> Tuple[bool, int]:
    """
    Return (is_loop, component_length_px) for the skeleton component at (cy, cx).

    A component is a closed loop if it has no degree-1 (tip) pixels.
    Real root strands always terminate at tips; soil pore boundaries often close
    into complete rings with no tips.
    """
    conn8 = np.ones((3, 3), dtype=int)
    labeled, _ = nd_label(skeleton, structure=conn8)
    comp_id = int(labeled[cy, cx])
    if comp_id == 0:
        return False, 0
    comp_u8 = (labeled == comp_id).astype(np.uint8)
    comp_len = int(comp_u8.sum())
    kern = np.ones((3, 3), dtype=np.float32)
    kern[1, 1] = 0.0
    n_nbrs = cv2.filter2D(comp_u8, cv2.CV_32F, kern).astype(np.uint8) * comp_u8
    return (not bool((n_nbrs == 1).any())), comp_len


def _skeleton_straightness(comp: np.ndarray) -> float:
    """
    Return end-to-end Euclidean distance / skeleton path length.

    1.0 = perfectly straight; values near 0 = tightly curled fragment.
    Returns 0.0 for loop components (no tip pixels) and single-pixel components.
    """
    ys, xs = np.where(comp)
    if len(ys) < 2:
        return 1.0
    kern = np.ones((3, 3), dtype=np.float32)
    kern[1, 1] = 0.0
    comp_u8 = comp.astype(np.uint8)
    n_nbrs  = cv2.filter2D(comp_u8, cv2.CV_32F, kern).astype(np.uint8) * comp_u8
    tips    = np.argwhere(n_nbrs == 1)  # degree-1 pixels
    if len(tips) < 2:
        return 0.0  # closed loop — no tips
    # Pick the most-distant tip pair for the end-to-end measurement
    t0, t1 = tips[0], tips[-1]
    if len(tips) > 2:
        tip_f = tips.astype(np.float64)
        D = cdist(tip_f, tip_f)
        i, j = np.unravel_index(D.argmax(), D.shape)
        t0, t1 = tips[i], tips[j]
    end_to_end = float(np.linalg.norm(t0.astype(float) - t1.astype(float)))
    path_len   = float(len(ys))
    return float(min(end_to_end / path_len, 1.0))


# ── Training augmentation transforms (module-level for picklability) ──────────

def _aug_gray_hflip(g):   return g[:, ::-1].copy()
def _aug_mask_hflip(m):   return m[:, ::-1].copy()
def _aug_coord_hflip(y, x, H, W):  return (y, W - 1 - x)

def _aug_gray_vflip(g):   return g[::-1, :].copy()
def _aug_mask_vflip(m):   return m[::-1, :].copy()
def _aug_coord_vflip(y, x, H, W):  return (H - 1 - y, x)

def _aug_gray_rot90cw(g): return np.rot90(g, k=-1).copy()
def _aug_mask_rot90cw(m): return np.rot90(m, k=-1).copy()
def _aug_coord_rot90cw(y, x, H, W): return (x, H - 1 - y)

def _aug_gray_blur(g):    return cv2.GaussianBlur(g, (0, 0), sigmaX=1.0)
def _aug_mask_identity(m): return m
def _aug_coord_identity(y, x, H, W): return (y, x)

_AUG_TRANSFORMS = [
    ("hflip",   _aug_gray_hflip,   _aug_mask_hflip,     _aug_coord_hflip),
    ("vflip",   _aug_gray_vflip,   _aug_mask_vflip,     _aug_coord_vflip),
    ("rot90cw", _aug_gray_rot90cw, _aug_mask_rot90cw,   _aug_coord_rot90cw),
    ("blur",    _aug_gray_blur,    _aug_mask_identity,  _aug_coord_identity),
]


def _train_one_image(args: tuple):
    """
    Worker: segment one annotated image and return feature rows.
    Returns (name, X float32, y int, w float32).
    """
    path, anns, scale, seg_kwargs, augment = args
    name = Path(path).stem

    rh = RhizotronImage(path, scale)
    gray = rh.interior_gray
    kw = dict(seg_kwargs, vesselness_threshold=0.0)
    mask = RootSegmenter(**kw).segment(gray, scale)
    skel, _, _ = RootSegmenter.skeletonize_and_measure(mask, scale)
    dt   = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_g = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy_g ** 2)

    H, W = gray.shape
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    w_rows: List[float] = []

    def _collect(g, sk, mk, d, gr, ann_list, coord_fn=None):
        fH, fW = g.shape
        for ann in ann_list:
            iy = int(ann["y"]); ix = int(ann["x"])
            if coord_fn is not None:
                iy, ix = coord_fn(iy, ix, fH, fW)
            iy = int(np.clip(iy, 0, fH - 1))
            ix = int(np.clip(ix, 0, fW - 1))
            X_rows.append(_extract_features_at(g, sk, mk, iy, ix, scale, _dt=d, _grad=gr))
            y_rows.append(int(ann["cls"]))
            w_rows.append(1.0)

    _collect(gray, skel, mask, dt, grad, anns)

    if augment:
        for _aug_name, gray_fn, mask_fn, coord_fn in _AUG_TRANSFORMS:
            g_a  = gray_fn(gray)
            m_a  = mask_fn(mask)
            sk_a, _, _ = RootSegmenter.skeletonize_and_measure(m_a, scale)
            dt_a = cv2.distanceTransform(m_a.astype(np.uint8), cv2.DIST_L2, 5)
            gx_a = cv2.Sobel(g_a, cv2.CV_32F, 1, 0, ksize=3)
            gy_a = cv2.Sobel(g_a, cv2.CV_32F, 0, 1, ksize=3)
            grad_a = np.sqrt(gx_a ** 2 + gy_a ** 2)
            _collect(g_a, sk_a, m_a, dt_a, grad_a, anns, coord_fn)

    return (
        name,
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=int),
        np.array(w_rows, dtype=np.float32),
    )


def _extract_features_at(
    gray: np.ndarray,
    skeleton: np.ndarray,
    mask: np.ndarray,
    cy: int,
    cx: int,
    scale: float,
    patch_size: int = _CLASSIFIER_PATCH,
    _dt: Optional[np.ndarray] = None,
    _grad: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract a 30-element feature vector at interior-crop position (cy, cx).

    Feature layout
    --------------
     0     Frangi vesselness at (cy, cx) — computed on local 32×32 patch
     1–2   Patch gray mean, std (normalised 0–1)
     3–4   8×8 central crop mean, std
     5–6   Patch vesselness mean, std
     7–22  Gabor: (|resp| mean, energy) × 4 orientations × 2 wavelengths
    23     Gabor orientation-continuity ratio  (max orient energy / total)
    24     On-skeleton flag (0 or 1)
    25     Skeleton curvature in 16-px neighbourhood
    26     Closed-loop flag (0 or 1)
    27     log(1 + component length)
    28     Distance-transform value at (cy, cx)
    29     Gradient magnitude at (cy, cx) (normalised 0–1)

    Parameters
    ----------
    _dt, _grad : optional precomputed arrays (same shape as gray) for speed
        when calling this function many times on the same image.
    """
    h, w = gray.shape
    half = patch_size // 2

    # Reflection-padded 32×32 gray patch
    y0 = max(0, cy - half); y1 = min(h, cy + half)
    x0 = max(0, cx - half); x1 = min(w, cx + half)
    raw = gray[y0:y1, x0:x1]
    pt = max(0, half - cy); pb = max(0, (cy + half) - h)
    pl = max(0, half - cx); pr = max(0, (cx + half) - w)
    gray_patch = np.pad(raw, ((pt, pb), (pl, pr)), mode="reflect")

    # Vesselness on the 32×32 patch (fast — small image)
    min_sig = max(0.5, 0.05 * scale)
    max_sig = max(2.0, 1.0 * scale)
    sigs = np.linspace(min_sig, max_sig, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vess_patch = frangi(
            gray_patch.astype(np.float64) / 255.0,
            sigmas=sigs, black_ridges=False,
        ).astype(np.float32)

    gray_f = gray_patch.astype(np.float32) / 255.0
    c4 = max(1, half // 4)
    centre = gray_f[half - c4: half + c4, half - c4: half + c4]

    feats: List[float] = [
        float(vess_patch[half, half]),     # 0  vesselness at point
        float(gray_f.mean()),              # 1  patch mean
        float(gray_f.std()),               # 2  patch std
        float(centre.mean()),              # 3  local mean (8-px centre)
        float(centre.std()),               # 4  local std
        float(vess_patch.mean()),          # 5  patch vessel mean
        float(vess_patch.std()),           # 6  patch vessel std
    ]

    # Gabor features: 16 values + orientation continuity ratio
    gab = _gabor_features(gray_patch)
    n_l = len(_GABOR_LAMBDAS)
    orient_energy = np.array([
        sum(gab[(i * n_l + j) * 2 + 1] for j in range(n_l))
        for i in range(_N_GABOR_ORIENTATIONS)
    ])
    orient_ratio = float(orient_energy.max() / (orient_energy.sum() + 1e-8))
    feats.extend(gab.tolist())             # 7–22
    feats.append(orient_ratio)             # 23

    # Skeleton features
    on_skel = int(skeleton[cy, cx]) if 0 <= cy < h and 0 <= cx < w else 0
    feats.append(float(on_skel))           # 24

    sk_r = 8
    skel_patch = skeleton[
        max(0, cy - sk_r): min(h, cy + sk_r),
        max(0, cx - sk_r): min(w, cx + sk_r),
    ]
    feats.append(_skeleton_curvature(skel_patch))   # 25

    is_loop, comp_len = _comp_loop_and_length(skeleton, cy, cx)
    feats.append(float(is_loop))                    # 26
    feats.append(float(np.log1p(comp_len)))         # 27

    # Distance transform at point
    if _dt is None:
        _dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    feats.append(float(_dt[cy, cx]))                # 28

    # Gradient magnitude at point
    if _grad is None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        _grad = np.sqrt(gx ** 2 + gy ** 2)
    feats.append(float(_grad[cy, cx]) / 255.0)     # 29

    return np.array(feats, dtype=np.float32)        # 30 features


# ─────────────────────────────────────────────────────────────────────────────
#  Skeleton classifier (Random Forest post-filter)
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFIER_FEAT_NAMES: List[str] = (
    ["vesselness", "gray_mean", "gray_std", "local_mean", "local_std",
     "vessel_mean", "vessel_std"]
    + [f"gabor_o{o}_{'fine' if j==0 else 'coarse'}_{s}"
       for o in range(_N_GABOR_ORIENTATIONS)
       for j in range(len(_GABOR_LAMBDAS))
       for s in ("mag", "nrg")]
    + ["orient_ratio", "on_skel", "curvature", "is_loop",
       "log_comp_len", "dt_val", "gradient"]
)


class SkeletonClassifier:
    """
    Random Forest classifier that distinguishes real root skeleton segments
    from soil pore boundary artifacts and background.

    Three classes
    -------------
    0  root        — actual root strand (thick or fine)
    1  pore_edge   — edge of a soil aggregate (key impostor class)
    2  background  — empty soil / no structure

    Filtering: a skeleton connected component is kept iff P(root) ≥ threshold.
    An additional hard rule removes small closed loops (no tip pixels, length ≤
    max_loop_size) without consulting the RF — these are almost always pore rings.

    Workflow
    --------
    1.  python rhizotron_analyzer.py --images DIR --annotate
            → place labelled points, saved to annotations/<stem>.json
    2.  python rhizotron_analyzer.py --images DIR --train
            → trains RF, saves to models/root_classifier.joblib
    3.  python rhizotron_analyzer.py --images DIR
            → auto-loads model if present, applies as post-filter
    4.  python rhizotron_analyzer.py --images DIR --no-classifier
            → run original pipeline without the RF (for comparison)
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self._classes: List[int] = []

    # ── Persistence ──────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            import joblib
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self._classes = bundle["classes"]
            print(
                f"  Loaded RF classifier  "
                f"({len(self.model.estimators_)} trees, classes {self._classes})"
                f"  ← {self.model_path}"
            )
            return True
        except Exception as exc:
            print(f"  WARNING: could not load classifier: {exc}", file=sys.stderr)
            return False

    def save(self) -> None:
        try:
            import joblib
        except ImportError:
            print(
                "WARNING: joblib not installed — cannot save classifier.\n"
                "Install with:  pip install scikit-learn joblib",
                file=sys.stderr,
            )
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "classes": self._classes}, self.model_path)
        print(f"  Saved classifier → {self.model_path}")

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        image_paths: List[str],
        annotation_dir: str,
        scale: float,
        seg_kwargs: dict,
        use_library: bool = False,
        library_path: str = DEFAULT_LIBRARY_PATH,
        external_features: Optional[str] = None,
        source_weight: float = 1.0,
        augment: bool = True,
        n_jobs: int = 1,
    ) -> None:
        """
        Build a feature matrix from annotated points and fit a Random Forest.

        When *use_library* is True, annotation dicts from all sessions in the
        library are pooled with the current-directory annotations.  Library
        annotations are matched to images by filename stem — only points whose
        image is present in *image_paths* contribute features.

        Segmentation during training skips the Frangi gate for speed; the
        classifier's own vesselness feature captures the same information at
        each annotated point.
        """
        try:
            import joblib  # noqa: F401
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.utils.class_weight import compute_class_weight
        except ImportError as exc:
            raise ImportError(
                f"scikit-learn and joblib are required for --train: {exc}\n"
                "Install with:  pip install scikit-learn joblib"
            ) from exc

        ann_dir = Path(annotation_dir)

        # Build a per-name annotation map from the current ann_dir
        name_to_anns: Dict[str, List[Dict]] = {}
        for path in image_paths:
            name = Path(path).stem
            ann_file = ann_dir / f"{name}.json"
            if ann_file.exists():
                with open(ann_file) as f:
                    name_to_anns.setdefault(name, []).extend(json.load(f))

        # Optionally pool annotations from the library
        if use_library:
            lib = Path(library_path).expanduser()
            lib_anns = _load_library_annotations(lib)
            # Group by image stem using the stored patch_y/patch_x + image origin
            # Library dicts don't carry an image stem, but each session folder has
            # per-image JSON files — reload them directly by stem.
            name_set = {Path(p).stem for p in image_paths}
            lib_p = Path(library_path).expanduser()
            for sess in sorted(lib_p.iterdir()) if lib_p.exists() else []:
                if not sess.is_dir():
                    continue
                for jf in sess.glob("*.json"):
                    if jf.name == "meta.json":
                        continue
                    stem = jf.stem
                    if stem in name_set:
                        try:
                            with open(jf) as f:
                                name_to_anns.setdefault(stem, []).extend(json.load(f))
                        except Exception:
                            pass
            print(
                f"  Library pooling enabled ({lib_p}): "
                f"{sum(len(v) for v in name_to_anns.values())} total annotations"
            )

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        w_list: List[float] = []  # per-sample weights for source weighting

        work = [
            (path, name_to_anns[Path(path).stem], scale, seg_kwargs, augment)
            for path in image_paths
            if Path(path).stem in name_to_anns
        ]

        print(f"  Extracting features from {len(work)} annotated image(s) "
              f"using {n_jobs} worker(s)...")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_train_one_image, w): w for w in work}
            for fut in as_completed(futures):
                try:
                    name, X_img, y_img, w_img = fut.result()
                except Exception as exc:
                    path = futures[fut][0]
                    print(f"  WARNING: failed on {Path(path).stem}: {exc}")
                    continue
                n_pts = len(y_img) // (1 + (4 if augment else 0))
                print(f"  [{name}]  {n_pts} annotations → {len(y_img)} samples")
                X_list.extend(X_img.tolist())
                y_list.extend(y_img.tolist())
                w_list.extend(w_img.tolist())

        own_n = len(X_list)
        if own_n < 10:
            raise ValueError(
                f"Only {own_n} labeled points found.  "
                "Use --annotate to label at least 10 points across the three classes first."
            )

        # Load pre-computed external feature arrays (.npz with keys X, y)
        ext_n = 0
        if external_features:
            ext_dir = Path(external_features).expanduser()
            if ext_dir.exists():
                for npz_path in sorted(ext_dir.glob("*.npz")):
                    try:
                        data = np.load(npz_path)
                        X_ext = data["X"].astype(np.float32)
                        y_ext = data["y"].astype(int)
                        if X_ext.shape[1] != 30:
                            print(f"  WARNING: skipping {npz_path.name} — expected 30 features, got {X_ext.shape[1]}")
                            continue
                        X_list.extend(X_ext.tolist())
                        y_list.extend(y_ext.tolist())
                        w_list.extend([source_weight] * len(y_ext))
                        ext_n += len(y_ext)
                    except Exception as exc:
                        print(f"  WARNING: could not load {npz_path.name}: {exc}")
                print(f"  External features: {ext_n} samples from {ext_dir} (weight={source_weight})")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=int)
        w = np.array(w_list, dtype=np.float32)
        present = sorted(np.unique(y).tolist())

        aug_label = f" + {(len(X_list) - own_n)} augmented" if augment else ""
        ext_label = f" + {ext_n} external" if ext_n else ""
        print(f"\n  Training RF on {own_n} own{aug_label}{ext_label} samples, classes {present}")
        print(f"  Class counts: { {c: int((y == c).sum()) for c in present} }")

        weights = compute_class_weight("balanced", classes=np.array(present), y=y)
        weight_map = dict(zip(present, weights.tolist()))
        # Combine class balance weights with source weights
        sample_weight = np.array([weight_map[cls] for cls in y], dtype=np.float32) * w

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            oob_score=True,
        )
        clf.fit(X, y, sample_weight=sample_weight)
        print(f"  OOB accuracy: {clf.oob_score_:.3f}")

        # Cross-validated macro-F1
        min_cls_count = min((y == c).sum() for c in present)
        cv_k = min(5, int(min_cls_count))
        if cv_k >= 3:
            cv_scores = cross_val_score(
                clf, X, y,
                cv=StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=0),
                scoring="f1_macro",
            )
            print(f"  {cv_k}-fold macro-F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Feature importance summary
        top5 = np.argsort(clf.feature_importances_)[::-1][:5]
        named = [
            _CLASSIFIER_FEAT_NAMES[i] if i < len(_CLASSIFIER_FEAT_NAMES) else f"feat{i}"
            for i in top5
        ]
        print(f"  Top-5 features: {named}")

        self.model = clf
        self._classes = present
        self.save()

    # ── Inference ─────────────────────────────────────────────────────────────

    def filter_skeleton(
        self,
        skeleton: np.ndarray,
        gray: np.ndarray,
        mask: np.ndarray,
        scale: float,
        threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
        max_loop_size: int = DEFAULT_MAX_LOOP_SIZE,
        min_straightness: float = DEFAULT_MIN_STRAIGHTNESS,
        large_root_length: int = DEFAULT_LARGE_ROOT_LENGTH,
        small_root_threshold: float = DEFAULT_SMALL_ROOT_THRESHOLD,
        collect_stats: bool = False,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Remove skeleton components that fail any of three sequential gates:

          1. Hard loop rule — no-tip component ≤ max_loop_size → removed.
          2. Straightness — end-to-end / path-length < min_straightness → removed.
          3. Two-tier RF rule — short segments (< large_root_length) use
             small_root_threshold; longer segments use threshold.

        Returns
        -------
        filtered : (H, W) bool skeleton
        stats : dict with per-stage counts (only populated when collect_stats=True)
        """
        if not self.is_loaded():
            return skeleton, None

        conn8 = np.ones((3, 3), dtype=int)
        labeled, n = nd_label(skeleton, structure=conn8)
        if n == 0:
            return skeleton, None

        dt   = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_g = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy_g ** 2)

        kern_nbr = np.ones((3, 3), dtype=np.float32)
        kern_nbr[1, 1] = 0.0

        root_idx: Optional[int] = (
            self._classes.index(0) if 0 in self._classes else None
        )

        filtered = skeleton.copy()

        n_loop = n_straight = n_rf_short = n_rf_long = 0
        lens_before: List[int] = []
        lens_after:  List[int] = []
        scores_kept:    List[float] = []
        scores_removed: List[float] = []

        for comp_id in range(1, n + 1):
            comp     = (labeled == comp_id)
            comp_len = int(comp.sum())
            if collect_stats:
                lens_before.append(comp_len)

            # ── 1. Hard loop rule ─────────────────────────────────────────────
            comp_u8 = comp.astype(np.uint8)
            n_nbrs  = (
                cv2.filter2D(comp_u8, cv2.CV_32F, kern_nbr).astype(np.uint8)
                * comp_u8
            )
            has_tip = bool((n_nbrs == 1).any())
            if not has_tip and comp_len <= max_loop_size:
                filtered[comp] = False
                n_loop += 1
                continue

            # ── 2. Straightness gate ──────────────────────────────────────────
            if min_straightness > 0:
                s = _skeleton_straightness(comp)
                if s < min_straightness:
                    filtered[comp] = False
                    n_straight += 1
                    continue

            # ── 3. Two-tier RF gate ───────────────────────────────────────────
            if root_idx is None:
                if collect_stats:
                    lens_after.append(comp_len)
                continue

            ys, xs = np.where(comp)
            cy = int(np.clip(ys.mean(), 0, gray.shape[0] - 1))
            cx = int(np.clip(xs.mean(), 0, gray.shape[1] - 1))
            feats = _extract_features_at(
                gray, skeleton, mask, cy, cx, scale,
                _dt=dt, _grad=grad,
            )
            p_root = float(self.model.predict_proba([feats])[0][root_idx])

            thr = threshold if comp_len >= large_root_length else small_root_threshold
            if p_root < thr:
                filtered[comp] = False
                if comp_len >= large_root_length:
                    n_rf_long += 1
                else:
                    n_rf_short += 1
                if collect_stats:
                    scores_removed.append(p_root)
            else:
                if collect_stats:
                    lens_after.append(comp_len)
                    scores_kept.append(p_root)

        stats = None
        if collect_stats:
            stats = {
                "n_total":       n,
                "n_loop":        n_loop,
                "n_straight":    n_straight,
                "n_rf_short":    n_rf_short,
                "n_rf_long":     n_rf_long,
                "n_kept":        len(lens_after),
                "lens_before":   lens_before,
                "lens_after":    lens_after,
                "scores_kept":   scores_kept,
                "scores_removed": scores_removed,
            }
        return filtered, stats

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(
        self,
        image_paths: List[str],
        annotation_dir: str,
        scale: float,
        seg_kwargs: dict,
        use_library: bool = False,
        library_path: str = DEFAULT_LIBRARY_PATH,
        target_recall: float = 0.80,
    ) -> None:
        """
        Evaluate the loaded classifier on annotated points and print a report.

        Reports per-class precision/recall/F1 (macro average), plus the
        precision achievable at *target_recall* for the root class by sweeping
        the P(root) threshold.  This metric is useful when you want to control
        false-negative rate: "at 80% recall, what fraction of kept segments are
        actually roots?"

        Parameters
        ----------
        target_recall : float — recall level at which to report precision@recall
        """
        if not self.is_loaded():
            print("ERROR: no classifier loaded — run --train first.", file=sys.stderr)
            return

        try:
            from sklearn.metrics import (
                classification_report, precision_recall_curve,
            )
        except ImportError as exc:
            print(f"ERROR: scikit-learn required for --benchmark: {exc}", file=sys.stderr)
            return

        ann_dir = Path(annotation_dir)
        name_to_anns: Dict[str, List[Dict]] = {}
        for path in image_paths:
            name = Path(path).stem
            ann_file = ann_dir / f"{name}.json"
            if ann_file.exists():
                with open(ann_file) as f:
                    name_to_anns.setdefault(name, []).extend(json.load(f))

        if use_library:
            lib_p = Path(library_path).expanduser()
            name_set = {Path(p).stem for p in image_paths}
            for sess in (sorted(lib_p.iterdir()) if lib_p.exists() else []):
                if not sess.is_dir():
                    continue
                for jf in sess.glob("*.json"):
                    if jf.name == "meta.json":
                        continue
                    stem = jf.stem
                    if stem in name_set:
                        try:
                            with open(jf) as f:
                                name_to_anns.setdefault(stem, []).extend(json.load(f))
                        except Exception:
                            pass

        X_list: List[np.ndarray] = []
        y_true: List[int] = []

        for path in image_paths:
            name = Path(path).stem
            anns = name_to_anns.get(name)
            if not anns:
                continue
            print(f"  [{name}]  {len(anns)} annotations — extracting features...")
            rh = RhizotronImage(path, scale)
            gray = rh.interior_gray
            kw = dict(seg_kwargs, vesselness_threshold=0.0)
            mask = RootSegmenter(**kw).segment(gray, scale)
            skel, _, _ = RootSegmenter.skeletonize_and_measure(mask, scale)
            dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy_g = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx ** 2 + gy_g ** 2)
            for ann in anns:
                iy = int(np.clip(ann["y"], 0, gray.shape[0] - 1))
                ix = int(np.clip(ann["x"], 0, gray.shape[1] - 1))
                feats = _extract_features_at(gray, skel, mask, iy, ix, scale, _dt=dt, _grad=grad)
                X_list.append(feats)
                y_true.append(int(ann["cls"]))

        if len(X_list) < 5:
            print("ERROR: fewer than 5 labeled points found — cannot benchmark.", file=sys.stderr)
            return

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_true, dtype=int)
        classes = sorted(self._classes)
        root_cls_idx = classes.index(0) if 0 in classes else None

        proba = self.model.predict_proba(X)  # (N, n_classes)
        y_pred = np.array([classes[i] for i in proba.argmax(axis=1)])

        cls_names = {0: "root", 1: "pore_edge", 2: "background"}
        target_names = [cls_names.get(c, str(c)) for c in classes]

        print("\n" + "=" * 60)
        print("  Classifier Benchmark")
        print("=" * 60)
        print(f"  Annotated points: {len(y)}")
        print(f"  Classes present:  { {c: int((y == c).sum()) for c in np.unique(y)} }")
        print()
        print(classification_report(
            y, y_pred,
            labels=classes,
            target_names=target_names,
            zero_division=0,
        ))

        # Precision@target_recall for root class
        if root_cls_idx is not None:
            p_root = proba[:, root_cls_idx]
            binary_y = (y == 0).astype(int)
            if binary_y.sum() > 0:
                precisions, recalls, thresholds = precision_recall_curve(binary_y, p_root)
                # Find the highest threshold where recall ≥ target_recall
                idx = np.where(recalls >= target_recall)[0]
                if len(idx) > 0:
                    best_i = idx[np.argmax(precisions[idx])]
                    print(
                        f"  Precision @ ≥{target_recall:.0%} recall (root):  "
                        f"{precisions[best_i]:.3f}  "
                        f"(threshold={thresholds[best_i - 1]:.3f})"
                    )
                else:
                    print(
                        f"  Precision @ ≥{target_recall:.0%} recall (root):  "
                        f"not achievable with current classifier"
                    )
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Image-level parallel worker (module-level so ProcessPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _process_image_worker(args: Tuple) -> Tuple[str, Tuple, np.ndarray, List[Dict]]:
    """
    Load, segment, and extract ROIs for one rhizotron image.

    This is a module-level function (not a method) so it can be pickled and
    dispatched by ProcessPoolExecutor on Linux (fork) and Windows (spawn).

    Parameters (packed as a tuple for executor.map compatibility)
    ----------
    args : (path, scale, seg_kw, ext_kw, bins, min_prim_diam, n_roi_workers,
            debug_dir, classifier, clf_threshold, max_loop_size, lateral_kw,
            pre_skeleton_threshold, min_component_area, remove_loops,
            min_straightness, large_root_length, small_root_threshold)

    Returns
    -------
    (name, interior_bbox, mask, rois)
    """
    (path, scale, seg_kw, ext_kw, bins, min_prim_diam, n_roi_workers,
     debug_dir, classifier, clf_threshold, max_loop_size, lateral_kw,
     pre_skeleton_threshold, min_component_area, remove_loops,
     min_straightness, large_root_length, small_root_threshold) = args

    rh   = RhizotronImage(path, scale)
    gray = rh.interior_gray

    segmenter = RootSegmenter(**seg_kw)
    mask = segmenter.segment(
        gray, rh.scale,
        debug_dir=debug_dir,
        debug_stem=rh.name,
    )

    # ── Pre-skeleton gate ─────────────────────────────────────────────────────
    # Apply the classifier at the mask-component level before skeletonization so
    # that soil pore boundaries never reach the skeleton tracer.
    prob_map: Optional[np.ndarray] = None
    initial_mask_px = int(mask.sum())

    if classifier is not None and classifier.is_loaded() and pre_skeleton_threshold > 0:
        gated_mask, prob_map = _apply_pre_skeleton_gate(
            mask, gray, scale, classifier,
            pre_skeleton_threshold, min_component_area,
        )
        # Safety check: if the gate removed >70% of the mask the classifier is
        # almost certainly under-trained on this image type.  Fall back to the
        # raw mask and warn rather than producing 0 ROIs.
        gated_px   = int(gated_mask.sum())
        gate_frac  = 1.0 - gated_px / max(initial_mask_px, 1)
        if gate_frac > 0.70:
            print(
                f"\n  WARNING: pre-skeleton gate removed {gate_frac*100:.0f}% of mask"
                f" in {rh.name} — classifier likely needs more training examples.\n"
                f"  Falling back to unfiltered mask for this image.\n"
                f"  Run --annotate to add more root / pore-edge annotations"
                f" covering this image, then re-run --train.",
                flush=True,
            )
            # Keep raw mask; still use prob_map for debug visualisation
        else:
            mask = gated_mask
            # Warn if pore-edge recall seems poor (many components rejected but
            # prob_map shows them near-threshold → borderline classifier)
            if initial_mask_px > 0:
                removed_px  = initial_mask_px - gated_px
                removed_frac = removed_px / initial_mask_px
                if removed_frac > 0.30:
                    print(
                        f"  NOTE: pre-skeleton gate removed {removed_frac*100:.0f}% of mask"
                        f" in {rh.name} — if results look under-segmented, add more"
                        f" class 0 (root) annotations and retrain.",
                        flush=True,
                    )
    elif min_component_area > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=min_component_area)

    # Loop removal is opt-in (--remove-loops) because binary_fill_holes treats every
    # background gap in a dense root mat as an "enclosed hole," which would remove
    # the entire mask.  Only enable for sparse, isolated pore outlines.
    loop_px_removed = 0
    if remove_loops:
        mask_before_loop = mask.copy()
        mask, loop_px_removed = _remove_loop_components(mask)
        loop_frac = loop_px_removed / max(initial_mask_px, 1)
        if loop_frac > 0.20:
            print(
                f"  WARNING: --remove-loops stripped {loop_frac*100:.0f}% of mask"
                f" in {rh.name} — this flag is not suitable for dense root mats."
                f"  Disabling for this image.",
                flush=True,
            )
            mask = mask_before_loop   # restore

    # ── Classifier-quality warning ────────────────────────────────────────────
    if (remove_loops and loop_px_removed > 0 and initial_mask_px > 0
            and classifier is not None and classifier.is_loaded()):
        loop_frac = loop_px_removed / initial_mask_px
        if loop_frac > 0.05:
            print(
                f"\n  WARNING: classifier may need additional pore-edge annotations"
                f" — {loop_frac*100:.1f}% of {rh.name} mask removed as closed rings"
                f" AFTER the classifier gate.\n"
                f"  Run --annotate and add more class 2 (soil pore edge) examples"
                f" from this image.",
                flush=True,
            )

    # ── Debug: save probability map; print per-stage filter stats ────────────
    if debug_dir and prob_map is not None:
        _save_pre_skeleton_debug(prob_map, mask, mask, gray, debug_dir, rh.name)

    if debug_dir and classifier is not None and classifier.is_loaded():
        full_skel, _, _ = RootSegmenter.skeletonize_and_measure(mask, scale)
        _, dbg_stats = classifier.filter_skeleton(
            full_skel, gray, mask, scale,
            clf_threshold, max_loop_size,
            min_straightness=min_straightness,
            large_root_length=large_root_length,
            small_root_threshold=small_root_threshold,
            collect_stats=True,
        )
        if dbg_stats and dbg_stats["n_total"] > 0:
            lb  = dbg_stats["lens_before"]
            la  = dbg_stats["lens_after"]
            sk  = dbg_stats["scores_kept"]
            sr  = dbg_stats["scores_removed"]
            print(f"\n  [{rh.name}] skeleton filter stats:")
            print(f"    Segments total:             {dbg_stats['n_total']}")
            print(f"    Removed — loop heuristic:   {dbg_stats['n_loop']}")
            print(f"    Removed — straightness:     {dbg_stats['n_straight']}")
            print(f"    Removed — RF (short <{large_root_length}px): {dbg_stats['n_rf_short']}")
            print(f"    Removed — RF (long ≥{large_root_length}px):  {dbg_stats['n_rf_long']}")
            print(f"    Kept:                       {dbg_stats['n_kept']}")
            if lb:
                print(f"    Length before  — median {int(np.median(lb))} px  "
                      f"mean {int(np.mean(lb))} px  "
                      f"max {max(lb)} px")
            if la:
                print(f"    Length after   — median {int(np.median(la))} px  "
                      f"mean {int(np.mean(la))} px  "
                      f"max {max(la)} px")
            if sk:
                print(f"    P(root) kept   — mean {np.mean(sk):.3f}  "
                      f"min {min(sk):.3f}")
            if sr:
                print(f"    P(root) removed— mean {np.mean(sr):.3f}  "
                      f"max {max(sr):.3f}")

    extractor = ROIExtractor(**ext_kw)
    rois = extractor.extract_rois(
        rh, mask, None, bins,
        min_primary_diameter_mm=min_prim_diam,
        n_workers=n_roi_workers,
        classifier=classifier,
        clf_threshold=clf_threshold,
        max_loop_size=max_loop_size,
        min_straightness=min_straightness,
        large_root_length=large_root_length,
        small_root_threshold=small_root_threshold,
        **lateral_kw,
    )

    return rh.name, rh.interior_bbox, mask, rois


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RhizotronPipeline:
    """
    Top-level orchestrator: runs all four analysis stages across a set of images
    and writes the output artefacts.
    """

    def __init__(
        self,
        image_paths: List[str],
        output_dir: str,
        scale_px_per_mm: float = DEFAULT_SCALE_PX_PER_MM,
        diameter_bins_mm: Optional[List[float]] = None,
        metric: str = DEFAULT_METRIC,
        roi_size_px: int = DEFAULT_ROI_SIZE_PX,
        roi_stride_px: Optional[int] = None,
        tophat_se_radius_mm: float = 2.5,
        min_primary_diameter_mm: float = 0.5,
        downsample: int = DEFAULT_DOWNSAMPLE,
        min_segment_length_px: int = 120,
        min_aspect_ratio: float = 3.0,
        max_root_diameter_mm: float = 3.0,
        vesselness_threshold: float = 0.05,
        min_skeleton_density: float = 0.02,
        border_margin: int = 100,
        n_jobs: int = DEFAULT_N_JOBS,
        debug: bool = False,
        classifier_path: str = DEFAULT_MODEL_PATH,
        use_classifier: bool = True,
        clf_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
        max_loop_size: int = DEFAULT_MAX_LOOP_SIZE,
        prune_length: int = DEFAULT_PRUNE_LENGTH,
        prune_passes: int = DEFAULT_PRUNE_PASSES,
        min_lateral_length: int = DEFAULT_MIN_LATERAL_LENGTH,
        min_lateral_angle: float = DEFAULT_MIN_LATERAL_ANGLE,
        max_lateral_angle: float = DEFAULT_MAX_LATERAL_ANGLE,
        min_lateral_persistence: int = DEFAULT_MIN_LATERAL_PERSISTENCE,
        max_diameter_cv: float = DEFAULT_MAX_DIAMETER_CV,
        lateral_clf_threshold: float = DEFAULT_LATERAL_CLF_THRESHOLD,
        max_lateral_density: float = DEFAULT_MAX_LATERAL_DENSITY,
        pre_skeleton_threshold: float = DEFAULT_PRE_SKELETON_THRESHOLD,
        min_component_area: int = DEFAULT_MIN_COMPONENT_AREA,
        remove_loops: bool = False,
        min_straightness: float = DEFAULT_MIN_STRAIGHTNESS,
        large_root_length: int = DEFAULT_LARGE_ROOT_LENGTH,
        small_root_threshold: float = DEFAULT_SMALL_ROOT_THRESHOLD,
    ):
        self.image_paths = image_paths
        self.output_dir = Path(output_dir)
        self.scale = scale_px_per_mm
        self.bins = sorted(diameter_bins_mm or DEFAULT_BINS_MM)
        self.metric = metric
        self.roi_size = roi_size_px
        self.roi_stride = roi_stride_px
        self.min_primary_diam_mm = min_primary_diameter_mm
        self.downsample = downsample
        self.border_margin = border_margin
        self.n_jobs = max(1, n_jobs)
        self.debug = debug
        self.clf_threshold = clf_threshold
        self.max_loop_size = max_loop_size
        self.prune_length = prune_length
        self.prune_passes = prune_passes
        self.min_lateral_length = min_lateral_length
        self.min_lateral_angle = min_lateral_angle
        self.max_lateral_angle = max_lateral_angle
        self.min_lateral_persistence = min_lateral_persistence
        self.max_diameter_cv = max_diameter_cv
        self.lateral_clf_threshold = lateral_clf_threshold
        self.max_lateral_density = max_lateral_density
        self.pre_skeleton_threshold = pre_skeleton_threshold
        self.min_component_area = min_component_area
        self.remove_loops = remove_loops
        self.min_straightness = min_straightness
        self.large_root_length = large_root_length
        self.small_root_threshold = small_root_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.segmenter = RootSegmenter(
            tophat_se_radius_mm=tophat_se_radius_mm,
            downsample=downsample,
            min_segment_length_px=min_segment_length_px,
            min_aspect_ratio=min_aspect_ratio,
            max_root_diameter_mm=max_root_diameter_mm,
            vesselness_threshold=vesselness_threshold,
        )
        self.roi_extractor = ROIExtractor(
            roi_size_px, roi_stride_px,
            min_segment_length_px=min_segment_length_px,
            min_skeleton_density=min_skeleton_density,
        )

        # Try to load the classifier; will be None if not found or disabled
        self.classifier: Optional[SkeletonClassifier] = None
        if use_classifier:
            clf = SkeletonClassifier(model_path=classifier_path)
            if clf.load():
                self.classifier = clf

    def run(self) -> None:
        """Execute the full pipeline and write all outputs."""
        self._print_header()

        debug_dir: Optional[str] = None
        if self.debug:
            debug_dir = str(self.output_dir / "debug")
            Path(debug_dir).mkdir(exist_ok=True)
            print(f"  Debug images → {debug_dir}/")

        # Distribute workers: image-level processes × ROI-level threads each
        n_imgs = len(self.image_paths)
        n_image_workers = min(self.n_jobs, n_imgs)
        n_roi_workers = max(1, self.n_jobs // n_image_workers)

        seg_kw = dict(
            tophat_se_radius_mm=self.segmenter.se_radius_mm,
            min_root_area_px=self.segmenter.min_area,
            downsample=self.segmenter.downsample,
            min_segment_length_px=self.segmenter.min_seg_len,
            min_aspect_ratio=self.segmenter.min_aspect_ratio,
            max_root_diameter_mm=self.segmenter.max_root_diameter_mm,
            vesselness_threshold=self.segmenter.vesselness_threshold,
        )
        ext_kw = dict(
            roi_size_px=self.roi_size,
            stride_px=self.roi_stride,
            min_root_density=self.roi_extractor.min_density,
            min_segment_length_px=self.roi_extractor.min_seg_len,
            min_skeleton_density=self.roi_extractor.min_skeleton_density,
        )

        lateral_kw = dict(
            prune_length=self.prune_length,
            prune_passes=self.prune_passes,
            min_lateral_length=self.min_lateral_length,
            min_lateral_angle=self.min_lateral_angle,
            max_lateral_angle=self.max_lateral_angle,
            min_lateral_persistence=self.min_lateral_persistence,
            max_diameter_cv=self.max_diameter_cv,
            lateral_clf_threshold=self.lateral_clf_threshold,
            max_lateral_density=self.max_lateral_density,
        )
        worker_args = [
            (path, self.scale, seg_kw, ext_kw, self.bins,
             self.min_primary_diam_mm, n_roi_workers, debug_dir,
             self.classifier, self.clf_threshold, self.max_loop_size, lateral_kw,
             self.pre_skeleton_threshold, self.min_component_area, self.remove_loops,
             self.min_straightness, self.large_root_length, self.small_root_threshold)
            for path in self.image_paths
        ]

        print(
            f"\n[Stage 1+2]  Segmenting & extracting ROIs"
            f"  ({n_image_workers} image worker{'s' if n_image_workers > 1 else ''}"
            f" × {n_roi_workers} ROI thread{'s' if n_roi_workers > 1 else ''} each)..."
        )

        if n_image_workers > 1:
            # Memory per worker ≈ 40 MB at full resolution.
            # With 24 workers and 64 GB RAM this is comfortable.
            # Reduce --n-jobs if memory pressure is observed.
            with ProcessPoolExecutor(max_workers=n_image_workers) as pool:
                results = list(pool.map(_process_image_worker, worker_args))
        else:
            results = [_process_image_worker(a) for a in worker_args]

        # Collect results; recreate RhizotronImage in main process for display
        images: List[RhizotronImage] = []
        masks: Dict[str, np.ndarray] = {}
        all_rois: List[Dict] = []

        for (name, _bbox, mask, rois), path in zip(results, self.image_paths):
            rh = RhizotronImage(path, self.scale)
            images.append(rh)
            masks[rh.name] = mask
            all_rois.extend(rois)
            root_frac = mask.mean() * 100
            print(f"           {name}: {len(rois)} ROIs  |  coverage {root_frac:.1f}%")

            # Debug density map per image (uses display-scale skeleton; fast)
            if debug_dir:
                stride = self.roi_extractor.stride
                save_debug_density_map(
                    mask, rh.interior_gray,
                    self.roi_size, stride,
                    self.roi_extractor.min_skeleton_density,
                    debug_dir, rh.name,
                )

        print(f"           Total ROIs: {len(all_rois)}")

        # ── Stage 3: Cross-plant ROI matching ─────────────────────────────────
        print(f"\n[Stage 3]  Matching ROIs  (metric={self.metric},"
              f" border_margin={self.border_margin}px)...")
        matches_df, sim_matrix_df = match_rois_across_plants(
            all_rois, self.metric, top_k=3, border_margin=self.border_margin,
        )
        print(f"           Match pairs generated: {len(matches_df)}")

        if not sim_matrix_df.empty:
            upper = sim_matrix_df.values[np.triu_indices(len(sim_matrix_df), k=1)]
            if upper.size > 0:
                print(
                    f"           Inter-plant similarity — "
                    f"mean {upper.mean():.3f}  min {upper.min():.3f}  "
                    f"max {upper.max():.3f}"
                )
            print(f"\n{sim_matrix_df.round(3).to_string()}\n")

        # ── Stage 4: Write outputs ─────────────────────────────────────────────
        print("[Stage 4]  Writing outputs...")
        save_roi_coordinates(all_rois, str(self.output_dir / "roi_coordinates.csv"))
        save_similarity_matrix(sim_matrix_df, str(self.output_dir / "similarity_matrix.csv"))
        save_match_details(all_rois, matches_df, str(self.output_dir / "matched_rois_detail.csv"))
        save_visual_panel(
            images, masks, matches_df,
            str(self.output_dir / "comparison_panel.png"),
            self.scale, self.bins,
        )

        print(f"\n  All outputs written to: {self.output_dir}/\n")

    def _print_header(self) -> None:
        w = 66
        print("\n" + "=" * w)
        print("  Rhizotron Image Analyzer")
        print("=" * w)
        print(f"  Images              : {len(self.image_paths)}")
        print(f"  Scale               : {self.scale:.2f} px/mm")
        print(f"  Diameter bins       : {self.bins} mm")
        print(f"  Similarity metric   : {self.metric}")
        print(f"  ROI size            : {self.roi_size} px")
        print(f"  Downsample          : {self.downsample}×")
        print(f"  Min segment length  : {self.segmenter.min_seg_len} px")
        print(f"  Min aspect ratio    : {self.segmenter.min_aspect_ratio:.1f}")
        print(f"  Max root diameter   : {self.segmenter.max_root_diameter_mm:.1f} mm")
        print(f"  Vesselness threshold: {self.segmenter.vesselness_threshold:.4f}")
        print(f"  Min skeleton density: {self.roi_extractor.min_skeleton_density:.3f}")
        print(f"  Border margin       : {self.border_margin} px")
        print(f"  Parallel jobs       : {self.n_jobs}")
        print(f"  Debug mode          : {'ON → output/debug/' if self.debug else 'off'}")
        clf_status = (
            f"ON  (threshold={self.clf_threshold:.2f}, max_loop={self.max_loop_size}px)"
            if self.classifier else "off (no model loaded)"
        )
        print(f"  RF classifier       : {clf_status}")
        print(f"  Output directory    : {self.output_dir}")
        print(f"  ── Pre-skeleton mask filtering ─────────────────────────────")
        clf_gate = (
            f"ON  (P(root) ≥ {self.pre_skeleton_threshold:.2f})"
            if self.classifier else "off (no classifier loaded)"
        )
        print(f"  Pre-skeleton gate   : {clf_gate}")
        print(f"  Min component area  : {self.min_component_area} px²")
        print(f"  Remove loop masks   : {'yes (--remove-loops)' if self.remove_loops else 'no (default)'}")
        print(f"  ── Lateral root precision ──────────────────────────────────")
        print(f"  Prune stubs         : {self.prune_length} px × {self.prune_passes} passes")
        print(f"  Min lateral length  : {self.min_lateral_length} px")
        print(f"  Lateral angle range : {self.min_lateral_angle:.0f}°–{self.max_lateral_angle:.0f}°")
        print(f"  Min persistence     : {self.min_lateral_persistence} px")
        print(f"  Max diameter CV     : {self.max_diameter_cv:.2f}")
        print(f"  Lateral clf thresh  : {self.lateral_clf_threshold:.2f}")
        print(f"  Max lateral density : {self.max_lateral_density:.1f} /cm")
        print("=" * w)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _find_images(directory: str) -> List[str]:
    """Return sorted list of JPEG/PNG/TIFF paths in *directory*."""
    supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    p = Path(directory)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    paths = sorted(f for f in p.iterdir() if f.suffix.lower() in supported)
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {directory}")
    return [str(f) for f in paths]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rhizotron_analyzer.py",
        description=(
            "Segment roots, extract morphological traits, and identify "
            "comparable ROIs across rhizotron images to guide sampling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Annotation + classifier workflow
---------------------------------
  Step 1 — label points interactively (OpenCV window):
    python rhizotron_analyzer.py --images testimages/ --annotate

  Step 2 — train the Random Forest (requires scikit-learn + joblib):
    python rhizotron_analyzer.py --images testimages/ --train

  Step 3 — run pipeline with classifier (auto-loaded from models/):
    python rhizotron_analyzer.py --images testimages/

  Step 4 — compare without classifier:
    python rhizotron_analyzer.py --images testimages/ --no-classifier

Examples
--------
  # Default settings on all images in testimages/
  python rhizotron_analyzer.py --images testimages/

  # Custom diameter bins, cosine similarity, parallel processing
  python rhizotron_analyzer.py --images testimages/ --bins 0.5 1 2 --metric cosine --n-jobs 8

  # Tighter false-positive filtering + debug to inspect intermediate masks
  python rhizotron_analyzer.py --images testimages/ --min-aspect-ratio 4.0 \\
      --max-root-diameter 2.0 --min-segment-length 50 --debug

  # Disable the Frangi vesselness gate (useful if dense mat suppresses real roots)
  python rhizotron_analyzer.py --images testimages/ --vesselness-threshold 0

  # Specify pixel scale and output directory
  python rhizotron_analyzer.py --images testimages/ --scale 12.5 --output results/

Parameter-tuning notes
----------------------
  --scale                   Measure a ruler visible in the image.
  --tophat-radius           Increase for thick-rooted species.
  --min-aspect-ratio        Raise to be stricter about elongation (roots vs debris).
  --max-root-diameter       Set to widest single root expected; blobs wider are removed.
  --min-segment-length      Raise to discard more short skeleton fragments.
  --vesselness-threshold    Lower toward 0 if true roots are suppressed in dense mats.
  --min-roi-density         Minimum skeleton fraction for an ROI to enter matching.
  --border-margin           Pixels from interior edge within which ROIs are penalised.
  --n-jobs                  See header comment for memory-per-worker estimate.
  --classifier-threshold    Lower → keep more skeleton; raise → stricter pore removal.
  --max-loop-size           Closed skeleton loops shorter than this removed automatically.
        """,
    )
    parser.add_argument(
        "--images", required=False, default=None, metavar="DIR",
        help="Directory containing rhizotron images (JPEG/PNG/TIFF).  "
             "Required for all modes except --list-library.",
    )
    parser.add_argument(
        "--bins", nargs="+", type=float, default=DEFAULT_BINS_MM, metavar="MM",
        help=(
            "Diameter class boundaries in mm.  E.g. --bins 0.5 1 2 creates "
            "classes <0.5, 0.5–1, 1–2, >2 mm.  (default: 0.5 1.0 2.0)"
        ),
    )
    parser.add_argument(
        "--metric", choices=["cosine", "euclidean"], default=DEFAULT_METRIC,
        help="Similarity metric for cross-plant ROI matching.  (default: cosine)",
    )
    parser.add_argument(
        "--scale", type=float, default=DEFAULT_SCALE_PX_PER_MM, metavar="PX_PER_MM",
        help=(
            "Image resolution in pixels per millimetre.  Calibrate by measuring "
            f"a known object in your images.  (default: {DEFAULT_SCALE_PX_PER_MM})"
        ),
    )
    parser.add_argument(
        "--roi-size", type=int, default=DEFAULT_ROI_SIZE_PX, metavar="PX",
        help=f"Sliding-window ROI side length in pixels.  (default: {DEFAULT_ROI_SIZE_PX})",
    )
    parser.add_argument(
        "--roi-stride", type=int, default=None, metavar="PX",
        help="Stride between consecutive ROIs.  (default: roi-size // 2)",
    )
    parser.add_argument(
        "--tophat-radius", type=float, default=2.5, metavar="MM",
        help=(
            "Top-hat structuring element radius in mm.  ≈ 1.5× thickest root. "
            "(default: 2.5)  # TUNE"
        ),
    )
    parser.add_argument(
        "--downsample", type=int, default=DEFAULT_DOWNSAMPLE, metavar="N",
        help="Process at 1/N linear resolution.  2 = 4× faster.  (default: 1)",
    )
    parser.add_argument(
        "--min-primary-diameter", type=float, default=0.5, metavar="MM",
        help="Diameter threshold (mm) separating primary from lateral roots.  (default: 0.5)  # TUNE",
    )
    parser.add_argument(
        "--min-segment-length", type=int, default=30, metavar="PX",
        help=(
            "Minimum skeleton segment length in pixels.  Shorter segments are "
            "discarded as texture noise.  (default: 30)  # TUNE"
        ),
    )
    parser.add_argument(
        "--min-aspect-ratio", type=float, default=3.0, metavar="RATIO",
        help=(
            "Minimum major/minor axis ratio for a blob to be kept as a root.  "
            "Round blobs (ratio < threshold) are removed.  (default: 3.0)  # TUNE"
        ),
    )
    parser.add_argument(
        "--max-root-diameter", type=float, default=3.0, metavar="MM",
        help=(
            "Maximum plausible single-root diameter in mm.  Blobs wider than "
            "this are removed as soil aggregates.  (default: 3.0)  # TUNE"
        ),
    )
    parser.add_argument(
        "--vesselness-threshold", type=float, default=0.01, metavar="SCORE",
        help=(
            "Frangi vesselness score threshold.  Pixels below this value are "
            "treated as non-root.  Set to 0 to disable.  (default: 0.01)  # TUNE"
        ),
    )
    parser.add_argument(
        "--min-roi-density", type=float, default=0.02, metavar="FRAC",
        help=(
            "Minimum skeleton pixel density (skeleton px / ROI px) for a window "
            "to enter cross-plant matching.  Eliminates empty-soil and border-edge "
            "candidates.  (default: 0.05)  # TUNE"
        ),
    )
    parser.add_argument(
        "--border-margin", type=int, default=100, metavar="PX",
        help=(
            "Interior-crop pixels from any edge within which ROI similarity scores "
            "are soft-penalised (linear decay to 0 at the edge).  Suppresses frame "
            "hardware artifacts from winning matches.  Set to 0 to disable.  "
            "(default: 100)  # TUNE"
        ),
    )
    parser.add_argument(
        "--n-jobs", type=int, default=DEFAULT_N_JOBS, metavar="N",
        help=(
            f"Parallel worker processes.  Each handles one image; remaining "
            f"workers become per-image ROI threads.  Memory ≈ 40 MB/worker at "
            f"full resolution.  (default: {DEFAULT_N_JOBS} = cpu_count - 1)"
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help=(
            "Save intermediate segmentation images to output/debug/.  "
            "Three images per input: (a) raw top-hat mask, (b) after shape "
            "filtering, (c) final mask with skeleton overlay."
        ),
    )
    parser.add_argument(
        "--output", default="output", metavar="DIR",
        help="Directory for output files.  (default: ./output)",
    )

    # ── Annotation and classifier flags ──────────────────────────────────────
    parser.add_argument(
        "--annotate", action="store_true",
        help=(
            "Launch an interactive OpenCV annotation window.  Click to place "
            "labelled points (1=root, 2=pore_edge, 3=background) on each image.  "
            "Annotations saved to annotations/<stem>.json.  Skips normal pipeline."
        ),
    )
    parser.add_argument(
        "--correct", action="store_true",
        help=(
            "Launch a brush-based corrective annotation window.  Paints root/pore/bg "
            "corrections directly on the segmentation overlay.  Left-drag=root, "
            "Middle-drag=pore_edge, Right-drag=background.  +/- adjusts brush radius.  "
            "Appends corrections to annotations/<stem>.json.  Skips normal pipeline."
        ),
    )
    parser.add_argument(
        "--brush-radius", type=int, default=8, metavar="PX",
        dest="brush_radius",
        help="Initial brush radius in display pixels for --correct mode.  (default: 8)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help=(
            "Evaluate the loaded classifier on all annotated points and print a "
            "precision/recall/F1 report.  Also reports precision at --target-recall "
            "for the root class.  Requires a trained model and existing annotations."
        ),
    )
    parser.add_argument(
        "--target-recall", type=float, default=0.80, metavar="FRAC",
        dest="target_recall",
        help=(
            "Recall level for the precision@recall metric in --benchmark mode.  "
            "(default: 0.80)"
        ),
    )
    parser.add_argument(
        "--train", action="store_true",
        help=(
            "Train a Random Forest classifier from existing annotations "
            "(created with --annotate).  Saves the model to models/root_classifier.joblib.  "
            "Requires scikit-learn and joblib.  Skips normal pipeline."
        ),
    )
    parser.add_argument(
        "--no-classifier", action="store_true", dest="no_classifier",
        help=(
            "Disable the RF post-filter even if models/root_classifier.joblib exists.  "
            "Useful for comparing output with and without the learned discriminator."
        ),
    )
    parser.add_argument(
        "--classifier-path", default=DEFAULT_MODEL_PATH, metavar="PATH",
        help=f"Path to the saved RF model file.  (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--classifier-threshold", type=float,
        default=DEFAULT_CLASSIFIER_THRESHOLD, metavar="PROB",
        help=(
            "Minimum P(root) for a skeleton component to be retained.  "
            f"Lower → permissive; higher → strict pore removal.  "
            f"(default: {DEFAULT_CLASSIFIER_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--max-loop-size", type=int, default=DEFAULT_MAX_LOOP_SIZE, metavar="PX",
        help=(
            "Closed skeleton loops (no tip pixels) shorter than this length in "
            "pixels are removed unconditionally as soil pore rings, before the "
            f"RF is consulted.  (default: {DEFAULT_MAX_LOOP_SIZE})"
        ),
    )

    # ── Lateral root precision controls ──────────────────────────────────────
    parser.add_argument(
        "--prune-length", type=int, default=DEFAULT_PRUNE_LENGTH, metavar="PX",
        dest="prune_length",
        help=(
            "Remove terminal skeleton stubs shorter than this before lateral "
            f"counting (iterative).  (default: {DEFAULT_PRUNE_LENGTH})"
        ),
    )
    parser.add_argument(
        "--prune-passes", type=int, default=DEFAULT_PRUNE_PASSES, metavar="N",
        dest="prune_passes",
        help=(
            f"How many rounds of stub pruning to run.  (default: {DEFAULT_PRUNE_PASSES})"
        ),
    )
    parser.add_argument(
        "--min-lateral-length", type=int, default=DEFAULT_MIN_LATERAL_LENGTH, metavar="PX",
        dest="min_lateral_length",
        help=(
            "Minimum skeleton length (px) to count a branch as a lateral root.  "
            f"(default: {DEFAULT_MIN_LATERAL_LENGTH})"
        ),
    )
    parser.add_argument(
        "--min-lateral-angle", type=float, default=DEFAULT_MIN_LATERAL_ANGLE, metavar="DEG",
        dest="min_lateral_angle",
        help=(
            "Minimum emergence angle (degrees) from parent segment.  Branches "
            f"more parallel than this are rejected.  (default: {DEFAULT_MIN_LATERAL_ANGLE})"
        ),
    )
    parser.add_argument(
        "--max-lateral-angle", type=float, default=DEFAULT_MAX_LATERAL_ANGLE, metavar="DEG",
        dest="max_lateral_angle",
        help=(
            "Maximum emergence angle (degrees) from parent segment.  "
            f"(default: {DEFAULT_MAX_LATERAL_ANGLE})"
        ),
    )
    parser.add_argument(
        "--min-lateral-persistence", type=int,
        default=DEFAULT_MIN_LATERAL_PERSISTENCE, metavar="PX",
        dest="min_lateral_persistence",
        help=(
            "A lateral must travel this many pixels before re-branching.  "
            "Very short stubs that immediately fork are skeleton artifacts.  "
            f"(default: {DEFAULT_MIN_LATERAL_PERSISTENCE})"
        ),
    )
    parser.add_argument(
        "--max-diameter-cv", type=float, default=DEFAULT_MAX_DIAMETER_CV, metavar="CV",
        dest="max_diameter_cv",
        help=(
            "Maximum coefficient of variation of diameter along a lateral.  "
            "Real roots taper gradually; noisy fragments have erratic width.  "
            f"(default: {DEFAULT_MAX_DIAMETER_CV})"
        ),
    )
    parser.add_argument(
        "--lateral-classifier-threshold", type=float,
        default=DEFAULT_LATERAL_CLF_THRESHOLD, metavar="PROB",
        dest="lateral_clf_threshold",
        help=(
            "Stricter P(root) threshold applied only to lateral candidates after "
            "primary/lateral classification.  Laterals are held to a higher bar "
            f"than primaries.  (default: {DEFAULT_LATERAL_CLF_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--max-lateral-density", type=float,
        default=DEFAULT_MAX_LATERAL_DENSITY, metavar="PER_CM",
        dest="max_lateral_density",
        help=(
            "Cap on laterals per cm of parent root length.  Excess candidates "
            "are ranked by length and the shortest are discarded.  Segments "
            "that hit this cap are flagged laterals_capped=1 in the CSV.  "
            f"(default: {DEFAULT_MAX_LATERAL_DENSITY})"
        ),
    )
    # ── Pre-skeleton mask filtering ───────────────────────────────────────────
    parser.add_argument(
        "--pre-skeleton-threshold", type=float,
        default=DEFAULT_PRE_SKELETON_THRESHOLD, metavar="PROB",
        dest="pre_skeleton_threshold",
        help=(
            "Classifier P(root) gate applied to binary mask components BEFORE "
            "skeletonization.  Components scoring below this are removed so soil "
            "pore boundaries never reach the skeleton tracer.  Requires a trained "
            f"classifier.  Set to 0 to disable.  (default: {DEFAULT_PRE_SKELETON_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--min-component-area", type=int,
        default=DEFAULT_MIN_COMPONENT_AREA, metavar="PX2",
        dest="min_component_area",
        help=(
            "Remove binary mask components smaller than this area (px²) before "
            "skeletonization.  Real roots form continuous elongated blobs; small "
            f"isolated specks are noise.  (default: {DEFAULT_MIN_COMPONENT_AREA})"
        ),
    )
    parser.add_argument(
        "--remove-loops", action="store_true", dest="remove_loops",
        help=(
            "Enable morphological loop removal: strip mask components that form "
            "closed rings around enclosed dark regions before skeletonization.  "
            "Disabled by default because binary_fill_holes treats every background "
            "gap in a dense root mat as an enclosed hole, removing the entire mask.  "
            "Only useful for very sparse, isolated pore-boundary components."
        ),
    )
    parser.set_defaults(remove_loops=False)

    parser.add_argument(
        "--min-straightness", type=float,
        default=DEFAULT_MIN_STRAIGHTNESS, metavar="RATIO",
        dest="min_straightness",
        help=(
            "Minimum skeleton straightness (end-to-end distance / path length).  "
            "1.0 = perfectly straight; values near 0 = tightly curled fragment.  "
            "Segments below this threshold are removed before the RF gate.  "
            f"Set to 0 to disable.  (default: {DEFAULT_MIN_STRAIGHTNESS})"
        ),
    )
    parser.add_argument(
        "--large-root-length", type=int,
        default=DEFAULT_LARGE_ROOT_LENGTH, metavar="PX",
        dest="large_root_length",
        help=(
            "Skeleton length threshold (px) for the two-tier RF gate.  "
            "Segments ≥ this length use --classifier-threshold; shorter segments "
            f"use --small-root-threshold.  (default: {DEFAULT_LARGE_ROOT_LENGTH})"
        ),
    )
    parser.add_argument(
        "--small-root-threshold", type=float,
        default=DEFAULT_SMALL_ROOT_THRESHOLD, metavar="PROB",
        dest="small_root_threshold",
        help=(
            "Stricter P(root) threshold applied to skeleton segments shorter than "
            "--large-root-length.  Short segments are far more likely to be false "
            f"positives.  (default: {DEFAULT_SMALL_ROOT_THRESHOLD})"
        ),
    )

    parser.add_argument(
        "--conservative", action="store_true",
        help=(
            "Apply all precision-over-recall lateral parameters simultaneously: "
            "--min-lateral-length 80 --min-lateral-persistence 60 "
            "--prune-length 60 --prune-passes 5 "
            "--lateral-classifier-threshold 0.75 --max-lateral-density 1.5"
        ),
    )
    parser.add_argument(
        "--annotation-dir", default=ANNOTATION_DIR, metavar="DIR",
        help=f"Directory for annotation JSON files.  (default: {ANNOTATION_DIR})",
    )
    parser.add_argument(
        "--patch-size", type=int, default=DEFAULT_PATCH_SIZE, metavar="PX",
        dest="patch_size",
        help=(
            "Size of annotated patch in pixels (height and width) during --annotate "
            f"mode.  (default: {DEFAULT_PATCH_SIZE})"
        ),
    )

    # ── Annotation library flags ──────────────────────────────────────────────
    parser.add_argument(
        "--operator", default="unknown", metavar="NAME",
        help=(
            "Your name, recorded in library metadata for each annotation session.  "
            "(default: unknown)"
        ),
    )
    parser.add_argument(
        "--notes", default="", metavar="TEXT",
        help="Free-text note stored in library metadata for this session.",
    )
    parser.add_argument(
        "--library-path", default=DEFAULT_LIBRARY_PATH, metavar="DIR",
        dest="library_path",
        help=(
            "Root directory of the persistent annotation library.  Stored outside "
            "the project so it survives across experiment runs.  "
            f"(default: {DEFAULT_LIBRARY_PATH})"
        ),
    )
    parser.add_argument(
        "--use-library", action="store_true", dest="use_library",
        help=(
            "When training (--train), pool annotations from every archived session "
            "in the library in addition to the current --annotation-dir."
        ),
    )
    parser.add_argument(
        "--list-library", action="store_true", dest="list_library",
        help=(
            "Print a summary table of all archived annotation sessions in the "
            "library and exit."
        ),
    )
    parser.add_argument(
        "--external-features", default=None, metavar="DIR",
        dest="external_features",
        help=(
            "Directory of pre-computed feature arrays (.npz files, each with keys "
            "'X' (N×30 float32) and 'y' (N,) int) generated by convert_*_to_library.py "
            "scripts.  Combined with own annotations during --train."
        ),
    )
    parser.add_argument(
        "--source-weight", type=float, default=1.0, metavar="W",
        dest="source_weight",
        help=(
            "Sample weight multiplier applied to external features loaded via "
            "--external-features.  Own annotations always receive weight 1.0.  "
            "Use values < 1.0 to downweight domain-shifted external data.  "
            "(default: 1.0)"
        ),
    )
    parser.add_argument(
        "--no-augment", action="store_false", dest="augment",
        help=(
            "Disable training data augmentation (H-flip, V-flip, 90° rotation, "
            "Gaussian blur).  Augmentation is ON by default when --train is used."
        ),
    )
    parser.set_defaults(augment=True)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    # ── --list-library mode (no images needed) ────────────────────────────────
    if args.list_library:
        _list_library(Path(args.library_path).expanduser())
        return

    if not args.images:
        print("ERROR: --images DIR is required.", file=sys.stderr)
        sys.exit(1)

    try:
        image_paths = _find_images(args.images)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images in '{args.images}'")

    # ── --conservative preset overrides individual lateral flags ──────────────
    if args.conservative:
        args.min_lateral_length    = 80
        args.min_lateral_persistence = 60
        args.prune_length          = 60
        args.prune_passes          = 5
        args.lateral_clf_threshold = 0.75
        args.max_lateral_density   = 1.5
        print(
            "  [conservative] min_lateral_length=80  min_lateral_persistence=60  "
            "prune_length=60  prune_passes=5  "
            "lateral_clf_threshold=0.75  max_lateral_density=1.5"
        )

    # Shared segmenter kwargs (used by annotation tool and trainer for preview masks)
    seg_kwargs = dict(
        tophat_se_radius_mm=args.tophat_radius,
        downsample=args.downsample,
        min_segment_length_px=args.min_segment_length,
        min_aspect_ratio=args.min_aspect_ratio,
        max_root_diameter_mm=args.max_root_diameter,
        vesselness_threshold=args.vesselness_threshold,
    )

    # ── --annotate mode ───────────────────────────────────────────────────────
    if args.annotate:
        tool = AnnotationTool(
            image_paths=image_paths,
            annotation_dir=args.annotation_dir,
            scale_px_per_mm=args.scale,
            seg_kwargs=seg_kwargs,
            patch_size=args.patch_size,
            operator=args.operator,
            notes=args.notes,
            library_path=args.library_path,
        )
        tool.run()
        return

    # ── --correct mode ────────────────────────────────────────────────────────
    if args.correct:
        tool = CorrectionTool(
            image_paths=image_paths,
            annotation_dir=args.annotation_dir,
            scale_px_per_mm=args.scale,
            seg_kwargs=seg_kwargs,
            operator=args.operator,
            notes=args.notes,
            library_path=args.library_path,
            brush_radius=args.brush_radius,
        )
        tool.run()
        return

    # ── --benchmark mode ──────────────────────────────────────────────────────
    if args.benchmark:
        clf = SkeletonClassifier(model_path=args.classifier_path)
        if not clf.load():
            print(
                f"ERROR: no classifier found at {args.classifier_path}.  "
                "Run --train first.",
                file=sys.stderr,
            )
            sys.exit(1)
        clf.benchmark(
            image_paths=image_paths,
            annotation_dir=args.annotation_dir,
            scale=args.scale,
            seg_kwargs=seg_kwargs,
            use_library=args.use_library,
            library_path=args.library_path,
            target_recall=args.target_recall,
        )
        return

    # ── --train mode ──────────────────────────────────────────────────────────
    if args.train:
        clf = SkeletonClassifier(model_path=args.classifier_path)
        try:
            clf.train(
                image_paths=image_paths,
                annotation_dir=args.annotation_dir,
                scale=args.scale,
                seg_kwargs=seg_kwargs,
                use_library=args.use_library,
                library_path=args.library_path,
                external_features=args.external_features,
                source_weight=args.source_weight,
                augment=args.augment,
                n_jobs=args.n_jobs,
            )
        except (ValueError, ImportError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # ── Normal analysis pipeline ──────────────────────────────────────────────
    pipeline = RhizotronPipeline(
        image_paths=image_paths,
        output_dir=args.output,
        scale_px_per_mm=args.scale,
        diameter_bins_mm=args.bins,
        metric=args.metric,
        roi_size_px=args.roi_size,
        roi_stride_px=args.roi_stride,
        tophat_se_radius_mm=args.tophat_radius,
        min_primary_diameter_mm=args.min_primary_diameter,
        downsample=args.downsample,
        min_segment_length_px=args.min_segment_length,
        min_aspect_ratio=args.min_aspect_ratio,
        max_root_diameter_mm=args.max_root_diameter,
        vesselness_threshold=args.vesselness_threshold,
        min_skeleton_density=args.min_roi_density,
        border_margin=args.border_margin,
        n_jobs=args.n_jobs,
        debug=args.debug,
        classifier_path=args.classifier_path,
        use_classifier=not args.no_classifier,
        clf_threshold=args.classifier_threshold,
        max_loop_size=args.max_loop_size,
        prune_length=args.prune_length,
        prune_passes=args.prune_passes,
        min_lateral_length=args.min_lateral_length,
        min_lateral_angle=args.min_lateral_angle,
        max_lateral_angle=args.max_lateral_angle,
        min_lateral_persistence=args.min_lateral_persistence,
        max_diameter_cv=args.max_diameter_cv,
        lateral_clf_threshold=args.lateral_clf_threshold,
        max_lateral_density=args.max_lateral_density,
        pre_skeleton_threshold=args.pre_skeleton_threshold,
        min_component_area=args.min_component_area,
        remove_loops=args.remove_loops,
        min_straightness=args.min_straightness,
        large_root_length=args.large_root_length,
        small_root_threshold=args.small_root_threshold,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
