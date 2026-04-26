#!/usr/bin/env python3
"""
regen_stage4.py — Re-generate Stage 4 outputs from saved ensemble skeleton PNGs.

After an ensemble run the following files exist in --ensemble-dir:
    {image_stem}_ensemble_skeleton.png   (binary, 1-px-wide skeleton)

This script loads those skeletons, runs ROI extraction + cross-plant matching,
and writes the four Stage 4 output files — without re-running the ensemble.

Usage
-----
  python regen_stage4.py \\
      --images        <original-image-dir> \\
      --ensemble-dir  ens/ensemble \\
      --output        ens/ \\
      --scale         10.0

The --scale value must match whatever was used for the original ensemble run.

Output files
------------
  <output>/roi_coordinates.csv
  <output>/similarity_matrix.csv
  <output>/matched_rois_detail.csv
  <output>/comparison_panel.png
"""

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from skimage.morphology import disk, dilation as sk_dilation

# Import shared infrastructure from the main analyzer (same directory).
sys.path.insert(0, str(Path(__file__).parent))
from rhizotron_analyzer import (
    DEFAULT_BINS_MM,
    DEFAULT_ROI_SIZE_PX,
    DEFAULT_SCALE_PX_PER_MM,
    RhizotronImage,
    ROIExtractor,
    match_rois_across_plants,
    save_match_details,
    save_roi_coordinates,
    save_similarity_matrix,
    save_visual_panel,
)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="regen_stage4.py",
        description="Re-generate Stage 4 outputs from saved ensemble skeleton PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--images", required=True, metavar="DIR",
        help="Directory containing the original rhizotron images.",
    )
    p.add_argument(
        "--ensemble-dir", default="ens/ensemble", metavar="DIR", dest="ensemble_dir",
        help="Directory containing *_ensemble_skeleton.png files.  (default: ens/ensemble)",
    )
    p.add_argument(
        "--output", default="ens", metavar="DIR",
        help="Directory for Stage 4 output files.  (default: ens/)",
    )
    p.add_argument(
        "--scale", type=float, default=DEFAULT_SCALE_PX_PER_MM, metavar="PX_PER_MM",
        help=f"Pixels per mm — must match the ensemble run.  (default: {DEFAULT_SCALE_PX_PER_MM})",
    )
    p.add_argument(
        "--bins", nargs="+", type=float, default=DEFAULT_BINS_MM, metavar="MM",
        help=f"Diameter bin edges in mm.  (default: {DEFAULT_BINS_MM})",
    )
    p.add_argument(
        "--roi-size", type=int, default=DEFAULT_ROI_SIZE_PX, metavar="PX", dest="roi_size",
        help=f"ROI sliding-window side length in pixels.  (default: {DEFAULT_ROI_SIZE_PX})",
    )
    p.add_argument(
        "--roi-stride", type=int, default=None, metavar="PX", dest="roi_stride",
        help="Sliding-window stride.  Defaults to roi_size // 2.",
    )
    p.add_argument(
        "--min-roi-density", type=float, default=0.002, metavar="F", dest="min_roi_density",
        help=(
            "Minimum skeleton-pixel fraction in a window to qualify as an ROI candidate.  "
            "Lower than the full-pipeline default because the input is a 1-px-wide skeleton.  "
            "(default: 0.002)"
        ),
    )
    p.add_argument(
        "--min-skeleton-density", type=float, default=0.005, metavar="F",
        dest="min_skeleton_density",
        help="Minimum skeleton density after per-ROI re-skeletonisation.  (default: 0.005)",
    )
    p.add_argument(
        "--dilate-skeleton", type=int, default=2, metavar="PX", dest="dilate_skeleton",
        help=(
            "Dilate the loaded skeleton by this radius (px) before density pre-filtering.  "
            "Gives the 1-px-wide skeleton enough coverage for the density filter to find "
            "root-containing windows.  Set to 0 to skip dilation.  (default: 2)"
        ),
    )
    p.add_argument(
        "--border-margin", type=int, default=100, metavar="PX", dest="border_margin",
        help="Border-penalty margin for cross-plant ROI matching.  (default: 100)",
    )
    p.add_argument(
        "--n-jobs", type=int, default=1, metavar="N", dest="n_jobs",
        help="Parallel threads for ROI extraction per image.  (default: 1)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Locate skeleton PNGs ──────────────────────────────────────────────────
    skel_paths = sorted(ensemble_dir.glob("*_ensemble_skeleton.png"))
    if not skel_paths:
        print(f"ERROR: no *_ensemble_skeleton.png files found in {ensemble_dir}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(skel_paths)} skeleton file(s) in {ensemble_dir}")

    # ── Index original images by stem ────────────────────────────────────────
    image_dir = Path(args.images)
    if not image_dir.is_dir():
        print(f"ERROR: --images directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    orig_by_stem = {
        p.stem: str(p)
        for p in image_dir.iterdir()
        if p.suffix.lower() in _IMG_EXTS
    }
    if not orig_by_stem:
        print(f"ERROR: no images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Set up ROI extractor ──────────────────────────────────────────────────
    extractor = ROIExtractor(
        roi_size_px=args.roi_size,
        stride_px=args.roi_stride,
        min_root_density=args.min_roi_density,
        min_skeleton_density=args.min_skeleton_density,
    )

    dilate_se = disk(args.dilate_skeleton) if args.dilate_skeleton > 0 else None

    images     = []
    masks      = {}
    all_rois   = []
    skipped    = []

    # ── Per-image: load skeleton → extract ROIs ───────────────────────────────
    for skel_path in skel_paths:
        stem = skel_path.stem.replace("_ensemble_skeleton", "")

        if stem not in orig_by_stem:
            print(f"  WARNING: no original image found for '{stem}' — skipping.")
            skipped.append(stem)
            continue

        orig_path = orig_by_stem[stem]
        print(f"\n  [{stem}]  loading original image...")

        try:
            rh = RhizotronImage(orig_path, args.scale)
        except Exception as exc:
            print(f"  WARNING: could not load {orig_path}: {exc} — skipping.")
            skipped.append(stem)
            continue

        # Load skeleton PNG; it is in rh.interior_gray coordinates
        skel_png = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
        if skel_png is None:
            print(f"  WARNING: could not read {skel_path} — skipping.")
            skipped.append(stem)
            continue

        skel_bool = skel_png > 0
        ih, iw = rh.interior_gray.shape

        if skel_bool.shape != (ih, iw):
            print(
                f"  WARNING: skeleton shape {skel_bool.shape} != "
                f"interior crop shape {(ih, iw)} for {stem} — skipping.\n"
                f"  Check that --scale matches the ensemble run and that the "
                f"same image file is being used."
            )
            skipped.append(stem)
            continue

        # Dilate skeleton so the density pre-filter can locate root windows
        mask = sk_dilation(skel_bool, dilate_se) if dilate_se is not None else skel_bool

        print(f"  [{stem}]  extracting ROIs "
              f"({skel_bool.sum()} skeleton px → dilated to {mask.sum()} px)...")
        rois = extractor.extract_rois(rh, mask, None, args.bins, n_workers=args.n_jobs)
        print(f"  [{stem}]  → {len(rois)} ROIs")

        images.append(rh)
        masks[rh.name] = mask
        all_rois.extend(rois)

    if skipped:
        print(f"\nSkipped {len(skipped)} image(s): {skipped}")

    if not images:
        print("ERROR: no images processed successfully.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal: {len(all_rois)} ROIs across {len(images)} image(s)")

    # ── Stage 3: cross-plant ROI matching ────────────────────────────────────
    print(f"\n[Stage 3]  Matching ROIs  (border_margin={args.border_margin}px)...")
    matches_df, sim_matrix_df = match_rois_across_plants(
        all_rois, "cosine", top_k=3, border_margin=args.border_margin,
    )
    print(f"           Match pairs generated: {len(matches_df)}")

    if not sim_matrix_df.empty:
        upper = sim_matrix_df.values[np.triu_indices(len(sim_matrix_df), k=1)]
        if upper.size > 0:
            print(
                f"           Inter-plant similarity — "
                f"mean {upper.mean():.3f}  "
                f"min {upper.min():.3f}  "
                f"max {upper.max():.3f}"
            )
        print(f"\n{sim_matrix_df.round(3).to_string()}\n")

    # ── Stage 4: write output files ───────────────────────────────────────────
    print("[Stage 4]  Writing outputs...")
    save_roi_coordinates(all_rois, str(output_dir / "roi_coordinates.csv"))
    save_similarity_matrix(sim_matrix_df, str(output_dir / "similarity_matrix.csv"))
    save_match_details(all_rois, matches_df, str(output_dir / "matched_rois_detail.csv"))

    panel_path = str(output_dir / "comparison_panel.png")
    try:
        save_visual_panel(images, masks, matches_df, panel_path, args.scale, args.bins)
        print(f"  Saved comparison panel  → {panel_path}")
    except Exception as exc:
        print(f"  WARNING: comparison panel failed ({exc})")

    print(f"\n  Done.  Outputs written to: {output_dir}/\n")


if __name__ == "__main__":
    main()
