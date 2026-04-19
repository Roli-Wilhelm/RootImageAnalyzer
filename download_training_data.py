#!/usr/bin/env python3
"""
download_training_data.py — Download PRMI and convert it to feature arrays
for --external-features training.

PRMI (Plant Root Minirhizotron Imagery)
  72 K minirhizotron images with pixel-level binary masks, six species.
  Hosted on Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.2v6wwpzp4

Usage
-----
  # Preview file list without downloading
  python download_training_data.py --list-only

  # Download (9.3 GB, resumable) and convert
  python download_training_data.py

  # Limit conversion to 500 images for a quick test
  python download_training_data.py --max-images 500

After running, train with:
  python rhizotron_analyzer.py --images testimages/ --train \\
      --external-features external_features/prmi --source-weight 0.5
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional


# ── Download helper ────────────────────────────────────────────────────────

def _download(url: str, dest: Path, desc: str = "", chunk: int = 1 << 20) -> bool:
    """
    Download *url* to *dest* with a progress bar.  Supports resuming partial
    downloads via the HTTP Range header.  Returns True on success.
    """
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            code = resp.status
            total_from_header = int(resp.headers.get("Content-Length", 0))
            total = existing + total_from_header if code == 206 else total_from_header
            received = existing
            mode = "ab" if code == 206 else "wb"
            with open(dest, mode) as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    received += len(block)
                    if total:
                        pct = received / total * 100
                        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                        print(
                            f"\r  {desc}: [{bar}] {pct:5.1f}%  {received>>20}/{total>>20} MB",
                            end="", flush=True,
                        )
            print()
        return True
    except urllib.error.URLError as exc:
        print(f"\n  ERROR downloading {url}: {exc}")
        return False


def _get_json(url: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        print(f"  ERROR fetching {url}: {exc}")
        return None


# ── PRMI via Dryad API ─────────────────────────────────────────────────────
# Two-step lookup:
#   1. GET /api/v2/datasets/{encoded_doi}  → find stash:version href
#   2. GET /api/v2/versions/{id}/files     → list downloadable files
# (/datasets/{doi}/files returns 404 — undocumented quirk of the Dryad API)

_DRYAD_DOI     = "doi:10.5061/dryad.2v6wwpzp4"
_DRYAD_API     = "https://datadryad.org/api/v2"
_DRYAD_ENCODED = _DRYAD_DOI.replace(":", "%3A").replace("/", "%2F")
_DRYAD_DATASET = f"{_DRYAD_API}/datasets/{_DRYAD_ENCODED}"


def _list_prmi_files() -> List[dict]:
    meta = _get_json(_DRYAD_DATASET)
    if meta is None:
        return []
    version_href = meta.get("_links", {}).get("stash:version", {}).get("href", "")
    if not version_href:
        return []
    version_url = f"https://datadryad.org{version_href}"

    page, files = 1, []
    while True:
        data = _get_json(f"{version_url}/files?page={page}&per_page=100")
        if data is None:
            break
        batch = data.get("_embedded", {}).get("stash:files", [])
        if not batch:
            break
        files.extend(batch)
        if len(files) >= data.get("total", 0):
            break
        page += 1
    return [f for f in files if not f.get("path", "").endswith("README")]


def _file_id(f: dict) -> Optional[str]:
    href = f.get("_links", {}).get("self", {}).get("href", "")
    parts = href.rstrip("/").split("/")
    return parts[-1] if parts else None


def download_prmi(
    data_dir: Path,
    out_dir: Path,
    max_images: Optional[int],
    scale: float,
    list_only: bool,
    samples_per_image: int,
    skip_download: bool = False,
    n_jobs: int = 1,
) -> bool:
    print("\n" + "=" * 60)
    print("  PRMI dataset (Dryad)")
    print("=" * 60)
    print("  Fetching file list from Dryad API...")
    files = _list_prmi_files()

    if not files:
        print(
            "\n  Could not retrieve file list from Dryad API.\n"
            "  Please download manually:\n"
            f"    https://datadryad.org/dataset/{_DRYAD_DOI}\n"
            "  Then run:\n"
            f"    python convert_prmi_to_library.py --prmi-dir <extracted_dir> --output {out_dir}"
        )
        return False

    total_gb = sum(f.get("size", 0) for f in files) / (1 << 30)
    print(f"  Found {len(files)} file(s)  (total ≈ {total_gb:.1f} GB):")
    for f in files:
        print(f"    {f.get('path', '?'):50s}  {f.get('size', 0) >> 20:8d} MB")

    if list_only:
        return True

    if skip_download:
        print("  --skip-download set: using files already present in data dir.")
    elif total_gb > 1.0:
        print(
            f"\n  NOTE: {total_gb:.1f} GB download.  This will take a while on a slow connection.\n"
            "  The download supports resuming — if interrupted, just re-run this script.\n"
            "  If you get a 401 error, download manually and re-run with --skip-download."
        )

    prmi_dir = data_dir / "prmi"
    prmi_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        fname  = Path(f.get("path", "file")).name
        dest   = prmi_dir / fname
        size_b = f.get("size", 0)

        if dest.exists():
            size_mb = dest.stat().st_size >> 20
            print(f"  Already present: {fname} ({size_mb} MB) — skipping download.")
        elif skip_download:
            print(f"  --skip-download set but {fname} not found in {prmi_dir}")
            print(f"  Place the zip there and re-run.")
            return False
        else:
            fid = _file_id(f)
            if not fid:
                print(f"  SKIP (no file id): {fname}")
                continue
            url = f"{_DRYAD_API}/files/{fid}/download"
            print(f"  Downloading {fname} ({size_b >> 20} MB)...")
            print(
                "  NOTE: Dryad requires a free account for downloads.\n"
                "  If you get a 401 error, download manually from:\n"
                f"    https://datadryad.org/dataset/{_DRYAD_DOI}\n"
                f"  Save the zip to {prmi_dir}/ and re-run with --skip-download."
            )
            if not _download(url, dest, desc=fname):
                print(
                    f"\n  Download failed.  Manual alternative:\n"
                    f"    1. Log in at https://datadryad.org and download PRMI_official.zip\n"
                    f"    2. Move it to {prmi_dir}/\n"
                    f"    3. Re-run:  python download_training_data.py --skip-download"
                )
                return False

    # Extract
    extracted_dirs = []
    for zpath in sorted(prmi_dir.glob("*.zip")):
        extract_to = prmi_dir / zpath.stem
        if extract_to.exists():
            print(f"  Already extracted: {zpath.name}")
        else:
            print(f"  Extracting {zpath.name}  (may take several minutes)...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(prmi_dir)
        extracted_dirs.append(extract_to if extract_to.exists() else prmi_dir)

    target_dir = extracted_dirs[0] if extracted_dirs else prmi_dir

    # Convert
    cmd = [
        sys.executable, "convert_prmi_to_library.py",
        "--prmi-dir",          str(target_dir),
        "--output",            str(out_dir),
        "--samples-per-image", str(samples_per_image),
        "--scale",             str(scale),
    ]
    if max_images:
        cmd += ["--max-images", str(max_images)]
    if n_jobs > 1:
        cmd += ["--n-jobs", str(n_jobs)]

    print(f"\n  Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_training_data.py",
        description="Download PRMI and convert it to feature arrays for --external-features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # See what files are available (no download)
  python download_training_data.py --list-only

  # Full download + convert  (9.3 GB, resumable)
  python download_training_data.py

  # Limit conversion to 500 images for a quick test
  python download_training_data.py --max-images 500

After completion, train with:
  python rhizotron_analyzer.py --images testimages/ --train \\
      --external-features external_features/prmi --source-weight 0.5
        """,
    )
    parser.add_argument(
        "--data-dir", default="data/external", metavar="DIR", dest="data_dir",
        help="Directory to store the raw downloaded zip.  (default: data/external)",
    )
    parser.add_argument(
        "--output", default="external_features/prmi", metavar="DIR",
        help="Directory for converted .npz feature files.  (default: external_features/prmi)",
    )
    parser.add_argument(
        "--max-images", type=int, default=None, metavar="N", dest="max_images",
        help="Convert at most N image/mask pairs.  Full zip is always downloaded.",
    )
    parser.add_argument(
        "--samples-per-image", type=int, default=50, metavar="N", dest="samples_per_image",
        help="Points sampled per image per class during conversion.  (default: 50)",
    )
    parser.add_argument(
        "--scale", type=float, default=13.0, metavar="PX_PER_MM",
        help="px/mm for PRMI images.  CI-600 minirhizotron ≈ 13 px/mm.  (default: 13.0)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1),
        metavar="N", dest="n_jobs",
        help="Parallel workers for the conversion step.  (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--list-only", action="store_true", dest="list_only",
        help="Print available files and sizes without downloading.",
    )
    parser.add_argument(
        "--skip-download", action="store_true", dest="skip_download",
        help=(
            "Skip the download step and go straight to extraction + conversion.  "
            "Use this when you have already placed PRMI_official.zip in --data-dir/prmi/."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir  = Path(args.output).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = download_prmi(
        data_dir=data_dir,
        out_dir=out_dir,
        max_images=args.max_images,
        scale=args.scale,
        list_only=args.list_only,
        samples_per_image=args.samples_per_image,
        skip_download=args.skip_download,
        n_jobs=args.n_jobs,
    )

    if not args.list_only:
        print("\n" + "=" * 60)
        npz_files = list(out_dir.rglob("*.npz"))
        if npz_files:
            print(f"  Done.  {len(npz_files)} feature files → {out_dir}")
            print()
            print("  Train with external data:")
            print(f"    python rhizotron_analyzer.py --images testimages/ --train \\")
            print(f"        --external-features {out_dir} --source-weight 0.5")
        else:
            print(
                "  No feature files produced.  Check errors above, or download manually:\n"
                f"    https://datadryad.org/dataset/{_DRYAD_DOI}"
            )
        print("=" * 60)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
