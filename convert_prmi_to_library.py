#!/usr/bin/env python3
"""
convert_prmi_to_library.py — Extract training features from the PRMI dataset.

PRMI (Plant Root Minirhizotron Imagery) provides RGB minirhizotron images paired
with pixel-level binary masks (root=255, background=0).  This script samples
annotated points from each pair, extracts the same 30-element feature vector
used by the rhizotron_analyzer RF classifier, and saves .npz files that can be
passed to --external-features.

Dataset download
----------------
  https://datadryad.org/dataset/doi:10.5061/dryad.2v6wwpzp4
  (≈ 72 K image/mask pairs, multiple species in train/test/val splits)

Expected directory layout (flexible — script uses glob):
  prmi_root/
    images/   <image_stem>.jpg  (or PNG)
    masks/    <image_stem>.png  (binary, 0/255)

    OR any layout where mask can be found by replacing 'images' → 'masks'
    in the path, or where mask lives alongside image with _mask suffix.

Usage
-----
  python convert_prmi_to_library.py \\
      --prmi-dir /data/PRMI \\
      --output   external_features/prmi \\
      --samples-per-image 50 \\
      --scale 13.0

Output
------
  external_features/prmi/<stem>.npz   # keys: X (N×30 float32), y (N,) int
  external_features/prmi/meta.json    # provenance

Class mapping
-------------
  PRMI mask pixel=255 → class 0 (root)
  PRMI mask pixel=0   → class 2 (background)
  Class 1 (pore_edge) is not present in PRMI annotations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import label as nd_label
from skimage.morphology import skeletonize

# ── Feature extraction helpers copied from rhizotron_analyzer ──────────────

_CLASSIFIER_PATCH = 32
_N_GABOR_ORIENTATIONS = 4
_GABOR_LAMBDAS = [4.0, 8.0]


def _gabor_features(patch: np.ndarray) -> np.ndarray:
    from skimage.filters import gabor
    import warnings
    feats = []
    patch_f = patch.astype(np.float32) / 255.0
    for theta in np.linspace(0, np.pi, _N_GABOR_ORIENTATIONS, endpoint=False):
        for lam in _GABOR_LAMBDAS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                real, imag = gabor(patch_f, frequency=1.0 / lam, theta=theta, sigma_x=lam, sigma_y=lam)
            mag = np.abs(real + 1j * imag).astype(np.float32)
            feats.extend([float(mag.mean()), float((mag ** 2).mean())])
    return np.array(feats, dtype=np.float32)


def _skeleton_curvature(skel_patch: np.ndarray) -> float:
    import warnings
    ys, xs = np.where(skel_patch)
    if len(xs) < 3:
        return 0.0
    try:
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
    conn8 = np.ones((3, 3), dtype=int)
    labeled, _ = nd_label(skeleton, structure=conn8)
    h, w = skeleton.shape
    if not (0 <= cy < h and 0 <= cx < w):
        return False, 0
    comp_id = labeled[cy, cx]
    if comp_id == 0:
        return False, 0
    comp_mask = labeled == comp_id
    comp_len = int(comp_mask.sum())
    # tip pixels have exactly 1 neighbor in the skeleton
    from scipy.ndimage import convolve
    kern = np.ones((3, 3), dtype=np.uint8); kern[1, 1] = 0
    nbr_count = convolve(skeleton.astype(np.uint8), kern, mode="constant")
    tips = (skeleton & (nbr_count == 1))
    is_loop = not bool((tips & comp_mask).any())
    return is_loop, comp_len


def _extract_features_at(
    gray: np.ndarray,
    skeleton: np.ndarray,
    mask: np.ndarray,
    cy: int,
    cx: int,
    scale: float,
) -> np.ndarray:
    from skimage.filters import frangi
    import warnings

    h, w = gray.shape
    half = _CLASSIFIER_PATCH // 2

    y0 = max(0, cy - half); y1 = min(h, cy + half)
    x0 = max(0, cx - half); x1 = min(w, cx + half)
    raw = gray[y0:y1, x0:x1]
    pt = max(0, half - cy); pb = max(0, (cy + half) - h)
    pl = max(0, half - cx); pr = max(0, (cx + half) - w)
    gray_patch = np.pad(raw, ((pt, pb), (pl, pr)), mode="reflect")

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
        float(vess_patch[half, half]),
        float(gray_f.mean()),
        float(gray_f.std()),
        float(centre.mean()),
        float(centre.std()),
        float(vess_patch.mean()),
        float(vess_patch.std()),
    ]

    gab = _gabor_features(gray_patch)
    n_l = len(_GABOR_LAMBDAS)
    orient_energy = np.array([
        sum(gab[(i * n_l + j) * 2 + 1] for j in range(n_l))
        for i in range(_N_GABOR_ORIENTATIONS)
    ])
    orient_ratio = float(orient_energy.max() / (orient_energy.sum() + 1e-8))
    feats.extend(gab.tolist())
    feats.append(orient_ratio)

    on_skel = int(skeleton[cy, cx]) if 0 <= cy < h and 0 <= cx < w else 0
    feats.append(float(on_skel))

    sk_r = 8
    skel_patch = skeleton[
        max(0, cy - sk_r): min(h, cy + sk_r),
        max(0, cx - sk_r): min(w, cx + sk_r),
    ]
    feats.append(_skeleton_curvature(skel_patch))

    is_loop, comp_len = _comp_loop_and_length(skeleton, cy, cx)
    feats.append(float(is_loop))
    feats.append(float(np.log1p(comp_len)))

    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    feats.append(float(dt[cy, cx]))

    gx_c = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx_c ** 2 + gy_c ** 2)
    feats.append(float(grad[cy, cx]) / 255.0)

    return np.array(feats, dtype=np.float32)


# ── Dataset helpers ────────────────────────────────────────────────────────

def _find_pairs(prmi_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find (image, mask) path pairs in a PRMI directory tree.

    PRMI_official layout:
      {split}/images/{Species_WxH_DPI}/{stem}.jpg
      {split}/masks_pixel_gt/{Species_WxH_DPI}/GT_{stem}.png
    """
    pairs = []

    # Strategy 1: PRMI_official layout — images/ + masks_pixel_gt/ siblings
    for img_dir in sorted(prmi_dir.rglob("images")):
        if not img_dir.is_dir():
            continue
        mask_root = img_dir.parent / "masks_pixel_gt"
        if not mask_root.is_dir():
            continue
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            # Mask lives in the matching species subdirectory with GT_ prefix
            species_subdir = img_path.parent.relative_to(img_dir)
            mask_path = mask_root / species_subdir / f"GT_{img_path.stem}.png"
            if mask_path.exists():
                pairs.append((img_path, mask_path))
    if pairs:
        return pairs

    # Strategy 2: generic images/ + masks/ siblings (other PRMI layouts)
    for img_dir in sorted(prmi_dir.rglob("images")):
        if not img_dir.is_dir():
            continue
        for sibling in ("masks", "masks_pixel_gt", "groundtruth", "gt"):
            mask_root = img_dir.parent / sibling
            if not mask_root.is_dir():
                continue
            for img_path in sorted(img_dir.rglob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    continue
                species_subdir = img_path.parent.relative_to(img_dir)
                for mask_name in (
                    f"GT_{img_path.stem}.png",
                    img_path.stem + ".png",
                    img_path.name,
                ):
                    mask_path = mask_root / species_subdir / mask_name
                    if mask_path.exists():
                        pairs.append((img_path, mask_path))
                        break
        if pairs:
            return pairs

    # Strategy 3: _mask suffix alongside image
    for img_path in sorted(prmi_dir.rglob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if "_mask" in img_path.stem or img_path.stem.startswith("GT_"):
            continue
        mask_path = img_path.parent / (img_path.stem + "_mask.png")
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    return pairs


def _process_pair(
    img_path: Path,
    mask_path: Path,
    n_samples: int,
    scale: float,
    seed: int,
) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Process one image/mask pair.  Returns (stem, X, y) or None if unreadable.
    Uses an integer seed (not a Generator) so it is safely picklable for
    multiprocessing.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        return None

    if mask_raw.shape != gray.shape:
        mask_raw = cv2.resize(mask_raw, (gray.shape[1], gray.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    mask = mask_raw > 127
    skeleton = skeletonize(mask)

    root_yx = np.argwhere(mask)
    bg_yx = np.argwhere(~mask)
    if len(root_yx) == 0 or len(bg_yx) == 0:
        return None

    rng = np.random.default_rng(seed)
    n_root = min(n_samples, len(root_yx))
    n_bg = min(n_samples, len(bg_yx))
    root_idx = rng.choice(len(root_yx), size=n_root, replace=False)
    bg_idx = rng.choice(len(bg_yx), size=n_bg, replace=False)

    X_parts, y_parts = [], []
    for yi, xi in root_yx[root_idx]:
        X_parts.append(_extract_features_at(gray, skeleton, mask, int(yi), int(xi), scale))
        y_parts.append(0)
    for yi, xi in bg_yx[bg_idx]:
        X_parts.append(_extract_features_at(gray, skeleton, mask, int(yi), int(xi), scale))
        y_parts.append(2)

    return (
        img_path.stem,
        np.array(X_parts, dtype=np.float32),
        np.array(y_parts, dtype=int),
    )


def _worker(args: tuple):
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    img_path, mask_path, n_samples, scale, seed = args
    return _process_pair(Path(img_path), Path(mask_path), n_samples, scale, seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PRMI dataset to external feature .npz files for --external-features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prmi-dir", required=True, metavar="DIR",
                        help="Root directory of the PRMI dataset.")
    parser.add_argument("--output", default="external_features/prmi", metavar="DIR",
                        help="Output directory for .npz files.  (default: external_features/prmi)")
    parser.add_argument("--samples-per-image", type=int, default=50, metavar="N",
                        help="Max annotated points sampled per image per class.  (default: 50)")
    parser.add_argument("--scale", type=float, default=13.0, metavar="PX_PER_MM",
                        help="Estimated px/mm for the PRMI images.  CI-600 ≈ 13 px/mm.  (default: 13.0)")
    parser.add_argument("--max-images", type=int, default=None, metavar="N",
                        help="Process at most this many pairs (useful for quick test).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.  (default: 42)")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                        metavar="N",
                        help="Parallel worker processes.  (default: cpu_count - 1)")
    args = parser.parse_args()

    prmi_dir = Path(args.prmi_dir).expanduser()
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {prmi_dir} for image/mask pairs...")
    pairs = _find_pairs(prmi_dir)
    if not pairs:
        print("ERROR: no image/mask pairs found.  Check --prmi-dir layout.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(pairs)} pairs total.")

    # Pre-filter to images that have at least one root pixel.
    # PRMI includes many empty-mask images (no roots visible at that depth/date).
    # Results are cached in prmi_dir/root_pairs_cache.json so re-runs are instant.
    cache_path = prmi_dir / "root_pairs_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = set(json.load(f))
        root_pairs = [(img, msk) for img, msk in pairs if str(img) in cached]
        print(f"  Loaded pre-filter cache: {len(root_pairs)} root-containing pairs "
              f"({len(pairs) - len(root_pairs)} empty skipped).")
    else:
        print(f"  Pre-filtering {len(pairs)} masks for root pixels "
              f"(cached to {cache_path.name} for future runs)...")
        root_pairs = []
        for i, (img_path, mask_path) in enumerate(pairs):
            if i % 1000 == 0:
                print(f"    {i}/{len(pairs)}...", end="\r", flush=True)
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_raw is not None and mask_raw.max() > 127:
                root_pairs.append((img_path, mask_path))
        print(f"  {len(root_pairs)} have roots  "
              f"({len(pairs) - len(root_pairs)} empty-mask skipped).")
        with open(cache_path, "w") as f:
            json.dump([str(img) for img, _ in root_pairs], f)
        print(f"  Cache saved → {cache_path}")
    pairs = root_pairs

    if not pairs:
        print("ERROR: no pairs with root pixels found.", file=sys.stderr)
        sys.exit(1)

    if args.max_images:
        pairs = pairs[: args.max_images]
        print(f"Limiting to {len(pairs)} root-containing pairs (--max-images {args.max_images}).")

    total_samples = 0
    processed = 0
    n_jobs = min(args.n_jobs, len(pairs))
    print(f"Converting {len(pairs)} pairs using {n_jobs} workers...")

    # Build arg tuples — each pair gets its own deterministic seed
    work = [
        (str(img), str(msk), args.samples_per_image, args.scale, args.seed + i)
        for i, (img, msk) in enumerate(pairs)
    ]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_worker, w): w for w in work}
        for done_i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            img_name = Path(futures[fut][0]).name
            if result is None:
                print(f"  [{done_i}/{len(pairs)}] {img_name} — skipped (unreadable)")
                continue
            stem, X, y = result
            np.savez_compressed(out_dir / f"{stem}.npz", X=X, y=y)
            total_samples += len(y)
            processed += 1
            print(f"  [{done_i}/{len(pairs)}] {img_name} → {len(y)} samples")

    meta = {
        "source": "PRMI",
        "prmi_dir": str(prmi_dir),
        "pairs_found": len(pairs),
        "pairs_processed": processed,
        "total_samples": total_samples,
        "samples_per_image": args.samples_per_image,
        "scale_px_per_mm": args.scale,
        "classes": {0: "root", 2: "background"},
        "note": "PRMI has no pore_edge class (class 1). Images are minirhizotron tube camera frames.",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {processed} files, {total_samples} total samples → {out_dir}")
    print(f"Use with:  python rhizotron_analyzer.py --images ... --train --external-features {out_dir}")


if __name__ == "__main__":
    main()
