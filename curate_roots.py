#!/usr/bin/env python3
"""
curate_roots.py — manual curation tool for skeleton traces.

Loads an automated skeleton PNG (e.g. from --ensemble) over the original
color image and lets the operator erase / draw / undo skeleton pixels by hand
before ROI matching.  The output is a binary PNG that rhizotron_analyzer.py's
--curated-skeletons flag can ingest in place of the automated skeleton.

Usage
-----
  python curate_roots.py \\
      --image realimages/DSC_0028.JPG \\
      --skeleton output/realrun_ensemble/ensemble/DSC_0028_ensemble_skeleton.png \\
      --operator alice \\
      --output-dir curated_skeletons/

Keybindings (printed with `h`)
------------------------------
  e          erase mode (default)
  d          draw mode (auto-thinned to 1-px-wide on stroke release)
  u          undo last stroke (stack ≥ 20)
  [ / ]      shrink / grow brush
  + / -      zoom in / out at cursor
  g          toggle ROI grid overlay
  c          toggle skeleton overlay
  h          print help to terminal
  s          save (overwrites curated PNG, writes timestamped backup)
  q          save (with prompt) and quit
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from skimage.morphology import skeletonize

# Minimum interval (seconds) between full overlay redraws during a drag.
# 0.18 s ≈ 5–6 fps — enough to see progress over VNC without flooding the
# link with 56 MB RGBA frames per mouse motion event.
_DRAG_REDRAW_INTERVAL_S = 0.18

# Import RhizotronImage so the displayed image is in the same coordinate
# space as the skeleton (interior_crop, landscape-rotated).
sys.path.insert(0, str(Path(__file__).parent))
from rhizotron_analyzer import (
    DEFAULT_ROI_SIZE_PX,
    DEFAULT_SCALE_PX_PER_MM,
    RhizotronImage,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="curate_roots.py",
        description=(
            "Manual curation of automated skeleton traces.  Output PNG is "
            "drop-in compatible with rhizotron_analyzer.py's "
            "--curated-skeletons flag."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image",    required=True, metavar="PATH",
                   help="Original rhizotron image (color JPG/PNG).")
    p.add_argument("--skeleton", required=True, metavar="PATH",
                   help="Automated skeleton PNG to start curation from "
                        "(e.g. *_ensemble_skeleton.png).")
    p.add_argument("--output-dir", default=".", metavar="DIR",
                   dest="output_dir",
                   help="Where to save <stem>_curated_skeleton.png and "
                        "<stem>_curation_log.json.  (default: current dir)")
    p.add_argument("--operator", default="unknown", metavar="NAME",
                   help="Operator name recorded in the curation log.  "
                        "(default: unknown)")
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE_PX_PER_MM,
                   metavar="PX_PER_MM",
                   help="Pixels per mm — must match the ensemble run that "
                        f"produced the skeleton.  (default: "
                        f"{DEFAULT_SCALE_PX_PER_MM})")
    p.add_argument("--config", default=None, metavar="PATH",
                   help="rhizobox_config.json — used only to set the ROI "
                        "grid overlay size.")
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE_PX,
                   metavar="PX", dest="roi_size",
                   help=f"ROI grid cell size for the `g` overlay (overridden "
                        f"by --config if present).  (default: "
                        f"{DEFAULT_ROI_SIZE_PX})")
    p.add_argument("--brush", type=int, default=6, metavar="PX",
                   help="Initial brush radius in image pixels.  (default: 6)")
    p.add_argument("--backend", default="TkAgg", metavar="BACKEND",
                   help="Matplotlib backend (must support GUI events).  "
                        "Common choices: TkAgg, Qt5Agg, GTK3Agg.  "
                        "(default: TkAgg)")
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Curation app
# ─────────────────────────────────────────────────────────────────────────────

HELP_TEXT = """
═══════════════════════════════════════════════════════════════════
  curate_roots.py keybindings
═══════════════════════════════════════════════════════════════════
  Modes
    e        erase mode — circular brush (default)
    b        box-erase  — drag a rectangle, erase everything inside
    d        draw mode  — circular brush, thinned to 1-px on release

  Editing
    LMB drag paint a stroke / select a box in the current mode
    u        undo last completed stroke (history depth: 20)
    [ / ]    shrink / grow brush radius (e and d modes)

  View
    + / =    zoom in (centered on cursor)
    -        zoom out (centered on cursor)
    0        reset zoom
    g        toggle ROI grid overlay
    c        toggle skeleton overlay

  File
    s        save curated PNG + timestamped backup, write JSON log
    q        save (with confirmation prompt) and quit
    h        print this help
═══════════════════════════════════════════════════════════════════
""".rstrip()

UNDO_DEPTH = 20


class CurationApp:

    def __init__(self, args, plt_module):
        self.args = args
        self.plt = plt_module

        # ── Load image and align to the same coords as the skeleton ──────────
        self.rh = RhizotronImage(args.image, args.scale)
        self.image_rgb = self.rh.interior_rgb
        H, W = self.image_rgb.shape[:2]
        self.shape = (H, W)

        # ── Load the starting skeleton ───────────────────────────────────────
        skel_in = cv2.imread(args.skeleton, cv2.IMREAD_GRAYSCALE)
        if skel_in is None:
            raise FileNotFoundError(f"Cannot read skeleton: {args.skeleton}")
        if skel_in.shape != (H, W):
            raise ValueError(
                f"Skeleton shape {skel_in.shape} does not match interior "
                f"crop shape {(H, W)}.  --scale must match the run that "
                f"produced the skeleton ({args.scale} px/mm here)."
            )
        self.skel = (skel_in > 0).astype(np.uint8)
        self.original_skeleton_path = args.skeleton

        # ── Output paths ─────────────────────────────────────────────────────
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = out_dir / "curation_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stem = self.rh.name
        self.curated_png_path  = out_dir / f"{stem}_curated_skeleton.png"
        self.curation_log_path = out_dir / f"{stem}_curation_log.json"
        self.backup_dir        = backup_dir

        # ── ROI grid size (for `g` overlay) ──────────────────────────────────
        self.roi_size = int(args.roi_size)
        if args.config:
            try:
                cfg = json.loads(Path(args.config).read_text())
                self.roi_size = int(cfg.get("roi_size_px", self.roi_size))
            except Exception as exc:
                print(f"WARNING: could not read --config {args.config}: {exc}")

        # ── Editing state ────────────────────────────────────────────────────
        self.mode = "erase"          # "erase" or "draw"
        self.brush_radius = int(args.brush)
        self.show_overlay = True
        self.show_grid    = False

        # Stroke-in-progress live preview (uint8, 0/1)
        self.dragging = False
        self.last_pos: Optional[Tuple[int, int]] = None
        self.stroke_temp = np.zeros(self.shape, dtype=np.uint8)
        self._last_drag_redraw = 0.0

        # Undo stack
        self.history: List[np.ndarray] = []

        # Stats for the log
        self.stats = {
            "pixels_erased": 0,
            "pixels_drawn":  0,
            "undo_operations": 0,
            "saves": 0,
        }
        self.session_start = time.time()

        # ── Build the figure ─────────────────────────────────────────────────
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        plt = self.plt
        # Hide the navigation toolbar — its Pan/Zoom buttons grab the same
        # mouse drag events the brush relies on, so dragging silently
        # pans/zooms instead of painting.
        plt.rcParams["toolbar"] = "none"

        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_axis_off()

        self.image_artist = self.ax.imshow(self.image_rgb, interpolation="nearest")
        self.overlay_artist = self.ax.imshow(
            self._build_overlay_rgba(), interpolation="nearest",
        )
        # Brush-cursor circle — cheap to redraw, follows the mouse so the
        # operator sees where the brush is even when the heavy overlay is
        # throttled (especially helpful over VNC).
        self.brush_cursor = Circle(
            (0, 0), self.brush_radius, fill=False,
            edgecolor="yellow", linewidth=1.5, linestyle="-",
            visible=False, zorder=110,
        )
        self.ax.add_patch(self.brush_cursor)

        # Mode-specific cursor icons — pencil triangle for draw, eraser
        # block for erase.  Both follow the brush cursor; only one is
        # visible at a time depending on self.mode.
        self.cursor_pencil = Polygon(
            np.zeros((4, 2)), closed=True,
            facecolor="lime", edgecolor="black", linewidth=0.8,
            visible=False, zorder=112,
        )
        self.cursor_eraser = Rectangle(
            (0, 0), 0, 0,
            facecolor="red", edgecolor="black", linewidth=0.8,
            visible=False, zorder=112,
        )
        self.ax.add_patch(self.cursor_pencil)
        self.ax.add_patch(self.cursor_eraser)

        # Marquee rectangle — visible only in box-erase mode while dragging.
        self.marquee = Rectangle(
            (0, 0), 0, 0,
            facecolor="red", alpha=0.2,
            edgecolor="red", linewidth=1.5, linestyle="--",
            visible=False, zorder=109,
        )
        self.ax.add_patch(self.marquee)
        self._box_anchor: Optional[Tuple[int, int]] = None  # (y, x)

        # Grid overlay artist (lazy — empty until toggled)
        self.grid_lines = []

        # On-canvas status line + key cheat sheet so the operator doesn't
        # need to read the terminal (helpful over VNC where the terminal
        # window may be hidden behind the figure).
        self.status_text = self.ax.text(
            0.005, 0.995, "", transform=self.ax.transAxes,
            color="yellow", fontsize=11, family="monospace",
            va="top", ha="left",
            bbox=dict(facecolor="black", alpha=0.65,
                      edgecolor="yellow", linewidth=0.5, pad=6),
            zorder=100,
        )
        self.help_text = self.ax.text(
            0.005, 0.005,
            "e=erase  b=box-erase  d=draw   u=undo   [ / ] brush   "
            "+ / - zoom   0=reset  g=grid  c=overlay  s=save  q=quit  h=help",
            transform=self.ax.transAxes,
            color="white", fontsize=10, family="monospace",
            va="bottom", ha="left",
            bbox=dict(facecolor="black", alpha=0.6,
                      edgecolor="none", pad=4),
            zorder=100,
        )

        # Save-confirmation banner — large green box centred at the top of
        # the canvas; appears briefly when the user presses `s`.
        self.save_banner = self.ax.text(
            0.5, 0.95, "",
            transform=self.ax.transAxes,
            color="white", fontsize=18, family="monospace", weight="bold",
            ha="center", va="top",
            bbox=dict(facecolor="#0a8a0a", alpha=0.92,
                      edgecolor="white", linewidth=1.5, pad=10),
            visible=False, zorder=120,
        )
        self._save_banner_timer = None

        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event",      self._on_key)
        self.fig.canvas.mpl_connect("close_event",          self._on_close)

        self._update_title()
        try:
            self.fig.canvas.manager.full_screen_toggle()
        except Exception:
            pass

        print(HELP_TEXT)
        print(f"\n  Image:    {self.args.image}")
        print(f"  Skeleton: {self.args.skeleton}")
        print(f"  Output:   {self.curated_png_path}")
        print(f"  Operator: {self.args.operator}\n")

    def _update_title(self) -> None:
        sk_count = int(self.skel.sum())
        title = (
            f"curate_roots — {self.rh.name}  |  mode={self.mode}  "
            f"brush={self.brush_radius}px  zoom={self._zoom_factor():.2f}x  "
            f"skeleton_px={sk_count}  undo_depth={len(self.history)}"
        )
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        # On-canvas status — visible even when the title bar isn't
        if hasattr(self, "status_text"):
            self.status_text.set_text(
                f"{self.rh.name}    mode={self.mode.upper():5s}  "
                f"brush={self.brush_radius}px  "
                f"zoom={self._zoom_factor():.2f}x  "
                f"skel_px={sk_count}  undo={len(self.history)}"
            )
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass

    # ── View helpers ─────────────────────────────────────────────────────────

    def _zoom_factor(self) -> float:
        """1.0 == fully zoomed-out (whole interior visible)."""
        H, W = self.shape
        x0, x1 = self.ax.get_xlim()
        y1, y0 = self.ax.get_ylim()       # imshow convention: y inverted
        view_w = max(1.0, abs(x1 - x0))
        view_h = max(1.0, abs(y1 - y0))
        zoom = max(W / view_w, H / view_h)
        return float(zoom)

    def _zoom_at(self, event, factor: float) -> None:
        if event.xdata is None or event.ydata is None:
            return
        x0, x1 = self.ax.get_xlim()
        y1, y0 = self.ax.get_ylim()
        cx, cy = float(event.xdata), float(event.ydata)
        new_w = (x1 - x0) / factor
        new_h = (y1 - y0) / factor
        # Clamp to image bounds
        H, W = self.shape
        new_w = min(new_w, W); new_h = min(new_h, H)
        nx0 = cx - new_w * (cx - x0) / max(1.0, (x1 - x0))
        nx1 = nx0 + new_w
        ny0 = cy - new_h * (cy - y0) / max(1.0, (y1 - y0))
        ny1 = ny0 + new_h
        # Clamp into image
        if nx0 < 0:        nx1 -= nx0;        nx0 = 0
        if nx1 > W:        nx0 -= (nx1 - W);  nx1 = W; nx0 = max(0, nx0)
        if ny0 < 0:        ny1 -= ny0;        ny0 = 0
        if ny1 > H:        ny0 -= (ny1 - H);  ny1 = H; ny0 = max(0, ny0)
        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ny1, ny0)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _reset_zoom(self) -> None:
        H, W = self.shape
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self._update_title()
        self.fig.canvas.draw_idle()

    # ── Overlay rendering ────────────────────────────────────────────────────

    def _build_overlay_rgba(self) -> np.ndarray:
        H, W = self.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        if not self.show_overlay:
            return rgba

        skel_bool = self.skel.astype(bool)
        # Cyan (R=0,G=255,B=255) at ~50% alpha = 128
        rgba[skel_bool] = [0, 255, 255, 128]

        if self.dragging and self.stroke_temp.any():
            tmp_bool = self.stroke_temp.astype(bool)
            if self.mode == "erase":
                # Erased pixels turn red briefly to confirm deletion
                paint = tmp_bool & skel_bool
                rgba[paint] = [255, 0, 0, 220]
            else:
                # Drawn pixels appear green while drawing
                rgba[tmp_bool] = [0, 255, 0, 200]
        return rgba

    def _refresh_overlay(self) -> None:
        self.overlay_artist.set_data(self._build_overlay_rgba())
        self.fig.canvas.draw_idle()

    def _draw_grid(self) -> None:
        # Remove any existing lines
        for ln in self.grid_lines:
            try: ln.remove()
            except Exception: pass
        self.grid_lines = []
        if not self.show_grid:
            self.fig.canvas.draw_idle()
            return
        H, W = self.shape
        s = self.roi_size
        for x in range(0, W + 1, s):
            self.grid_lines.append(
                self.ax.axvline(x, color="yellow", lw=0.5, alpha=0.4)
            )
        for y in range(0, H + 1, s):
            self.grid_lines.append(
                self.ax.axhline(y, color="yellow", lw=0.5, alpha=0.4)
            )
        self.fig.canvas.draw_idle()

    # ── Snapshot / undo ──────────────────────────────────────────────────────

    def _snapshot_for_undo(self) -> None:
        self.history.append(self.skel.copy())
        if len(self.history) > UNDO_DEPTH:
            self.history.pop(0)

    def _undo(self) -> None:
        if not self.history:
            print("  [undo] history empty")
            return
        self.skel = self.history.pop()
        self.stats["undo_operations"] += 1
        self.stroke_temp[:] = 0
        self.dragging = False
        self._refresh_overlay()
        self._update_title()

    # ── Stroke painting ──────────────────────────────────────────────────────

    def _paint_disk(self, y: int, x: int) -> None:
        cv2.circle(self.stroke_temp, (x, y), self.brush_radius, 1, thickness=-1)

    def _paint_line(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
        y1, x1 = p1
        y2, x2 = p2
        cv2.line(self.stroke_temp, (x1, y1), (x2, y2), 1,
                 thickness=max(1, self.brush_radius * 2))
        cv2.circle(self.stroke_temp, (x2, y2), self.brush_radius, 1, thickness=-1)

    def _commit_stroke(self) -> None:
        if not self.stroke_temp.any():
            self.stroke_temp[:] = 0
            return

        if self.mode == "erase":
            stroke_bool = self.stroke_temp.astype(bool)
            removed = int((self.skel.astype(bool) & stroke_bool).sum())
            self.skel[stroke_bool] = 0
            self.stats["pixels_erased"] += removed
            print(f"  [erase] -{removed} px  (brush r={self.brush_radius})")
        else:
            # Draw + thin to single-pixel width.  Thin ONLY the new stroke
            # then OR into the existing skeleton — re-thinning the whole
            # mask would shift legit existing 1-px lines because the global
            # medial axis can re-route through new junctions.
            stroke_bool = self.stroke_temp.astype(bool)
            stroke_thin = skeletonize(stroke_bool)
            before      = int(self.skel.astype(bool).sum())
            self.skel   = (self.skel.astype(bool) | stroke_thin).astype(np.uint8)
            added       = int(self.skel.sum()) - before
            self.stats["pixels_drawn"] += max(0, added)
            print(f"  [draw]  +{max(0, added)} px (stroke-only thinning)")

        self.stroke_temp[:] = 0

    # ── Save / quit ──────────────────────────────────────────────────────────

    def _save(self, verbose: bool = True) -> None:
        # Primary curated PNG (binary, white skel on black bg)
        cv2.imwrite(str(self.curated_png_path),
                    self.skel.astype(np.uint8) * 255)

        # Timestamped backup
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = self.backup_dir / (
            f"{self.curated_png_path.stem}_{ts}.png"
        )
        cv2.imwrite(str(backup), self.skel.astype(np.uint8) * 255)

        # Log JSON
        log = {
            "operator":            self.args.operator,
            "timestamp":           datetime.now().isoformat(timespec="seconds"),
            "image":               str(Path(self.args.image).resolve()),
            "image_name":          self.rh.name,
            "original_skeleton":   str(Path(self.args.skeleton).resolve()),
            "interior_shape":      [int(self.shape[0]), int(self.shape[1])],
            "scale_px_per_mm":     float(self.args.scale),
            "session_seconds":     int(time.time() - self.session_start),
            "skeleton_px_final":   int(self.skel.sum()),
            **self.stats,
        }
        self.curation_log_path.write_text(json.dumps(log, indent=2))
        self.stats["saves"] += 1

        if verbose:
            print(f"\n  ✔ saved curated skeleton → {self.curated_png_path}")
            print(f"  ✔ backup                 → {backup}")
            print(f"  ✔ log                    → {self.curation_log_path}\n")
        self._update_title()
        self._show_save_banner(backup.name)

    def _show_save_banner(self, backup_name: str) -> None:
        if not hasattr(self, "save_banner"):
            return  # called before UI built
        ts = datetime.now().strftime("%H:%M:%S")
        msg = (
            f"✔ SAVED  {ts}\n"
            f"{self.curated_png_path.name}   "
            f"({int(self.skel.sum())} skeleton px)\n"
            f"backup: {backup_name}"
        )
        self.save_banner.set_text(msg)
        self.save_banner.set_visible(True)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

        # Auto-hide after ~2.5 s.  Cancel any in-flight timer first so
        # multiple rapid saves don't clobber each other.
        if self._save_banner_timer is not None:
            try:    self._save_banner_timer.stop()
            except Exception: pass
            self._save_banner_timer = None
        try:
            t = self.fig.canvas.new_timer(interval=2500)
            t.single_shot = True
            t.add_callback(self._hide_save_banner)
            t.start()
            self._save_banner_timer = t
        except Exception:
            # Timer unavailable on this backend — banner stays until the
            # next redraw; not fatal.
            pass

    def _hide_save_banner(self) -> None:
        if not hasattr(self, "save_banner"):
            return
        self.save_banner.set_visible(False)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass
        if self._save_banner_timer is not None:
            try:    self._save_banner_timer.stop()
            except Exception: pass
            self._save_banner_timer = None

    def _on_close(self, _event) -> None:
        # Clean stop on window close — no implicit save (user must press s/q)
        pass

    def _quit(self) -> None:
        # Confirm save in the terminal (matplotlib has no modal dialog)
        unsaved = (self.stats["pixels_erased"] + self.stats["pixels_drawn"]
                   - 0 if self.stats["saves"] > 0 else
                   self.stats["pixels_erased"] + self.stats["pixels_drawn"])
        prompt = (
            f"\nSave before quitting?  "
            f"(erased={self.stats['pixels_erased']}  "
            f"drawn={self.stats['pixels_drawn']}  "
            f"saves so far={self.stats['saves']})  [Y/n]: "
        )
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            ans = "y"
        if ans in ("", "y", "yes"):
            self._save()
        else:
            print("  Skipped save.")
        self.plt.close(self.fig)

    # ── Event handlers ───────────────────────────────────────────────────────

    def _move_brush_cursor(self, x: float, y: float, visible: bool = True) -> None:
        # Color & dash style per mode
        if self.mode == "erase":
            color, ls = "red", "--"
        elif self.mode == "draw":
            color, ls = "lime", "-"
        else:                       # "box"
            color, ls = "red", ":"

        # The brush circle indicates the freehand brush coverage; in box
        # mode we hide it (size irrelevant — the marquee defines the area).
        show_circle = visible and self.mode in ("erase", "draw")
        self.brush_cursor.center = (float(x), float(y))
        self.brush_cursor.set_radius(self.brush_radius)
        self.brush_cursor.set_edgecolor(color)
        self.brush_cursor.set_linestyle(ls)
        self.brush_cursor.set_visible(show_circle)

        # Mode-specific icon (pencil triangle / eraser block).  Sized
        # roughly to brush_radius, anchored above-right of the cursor so
        # it doesn't obscure the click point.
        r        = max(8, int(self.brush_radius))
        icon_off = r + 4   # offset from cursor in image px
        ix, iy   = float(x) + icon_off, float(y) - icon_off

        if visible and self.mode == "draw":
            # Pencil tip — quadrilateral with a sharp tip pointing at cursor
            tip_x, tip_y = float(x) + r * 0.5, float(y) - r * 0.5
            shaft = float(r) * 1.4
            verts = np.array([
                [tip_x,                       tip_y                      ],
                [tip_x + shaft * 0.55,        tip_y - shaft              ],
                [tip_x + shaft,               tip_y - shaft * 0.55       ],
                [tip_x + shaft * 0.15,        tip_y - shaft * 0.15       ],
            ], dtype=float)
            self.cursor_pencil.set_xy(verts)
            self.cursor_pencil.set_visible(True)
            self.cursor_eraser.set_visible(False)
        elif visible and self.mode == "erase":
            # Eraser block — small wide rectangle to upper-right
            ew, eh = float(r) * 1.6, float(r) * 0.9
            self.cursor_eraser.set_xy((ix, iy - eh))
            self.cursor_eraser.set_width(ew)
            self.cursor_eraser.set_height(eh)
            self.cursor_eraser.set_visible(True)
            self.cursor_pencil.set_visible(False)
        else:
            self.cursor_pencil.set_visible(False)
            self.cursor_eraser.set_visible(False)

        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    # ── Box-erase helpers ────────────────────────────────────────────────────

    def _update_marquee(self, y0: int, x0: int, y1: int, x1: int) -> None:
        x_lo, x_hi = sorted([x0, x1])
        y_lo, y_hi = sorted([y0, y1])
        self.marquee.set_xy((x_lo, y_lo))
        self.marquee.set_width(max(0, x_hi - x_lo))
        self.marquee.set_height(max(0, y_hi - y_lo))
        self.marquee.set_visible(True)

    def _commit_box_erase(self, y0: int, x0: int, y1: int, x1: int) -> None:
        H, W = self.shape
        x_lo = max(0, min(W, min(x0, x1)))
        x_hi = max(0, min(W, max(x0, x1)))
        y_lo = max(0, min(H, min(y0, y1)))
        y_hi = max(0, min(H, max(y0, y1)))
        if x_hi - x_lo < 2 or y_hi - y_lo < 2:
            print("  [box-erase] selection too small — skipped")
            return
        roi = self.skel[y_lo:y_hi, x_lo:x_hi]
        removed = int(roi.astype(bool).sum())
        roi[:] = 0
        self.stats["pixels_erased"] += removed
        print(f"  [box-erase] -{removed} px  "
              f"({x_hi - x_lo}×{y_hi - y_lo} rect at "
              f"y=[{y_lo},{y_hi}] x=[{x_lo},{x_hi}])")

    # ── Event handlers ───────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        H, W = self.shape
        x = int(np.clip(event.xdata, 0, W - 1))
        y = int(np.clip(event.ydata, 0, H - 1))

        self._snapshot_for_undo()
        self.dragging = True
        self.last_pos = (y, x)

        if self.mode == "box":
            self._box_anchor = (y, x)
            self._update_marquee(y, x, y, x)
        else:
            self.stroke_temp[:] = 0
            self._paint_disk(y, x)

        self._move_brush_cursor(x, y, visible=True)
        self._refresh_overlay()
        self._last_drag_redraw = time.time()

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        H, W = self.shape
        x = int(np.clip(event.xdata, 0, W - 1))
        y = int(np.clip(event.ydata, 0, H - 1))

        # Always show the cursor (cheap — circle + icon)
        self._move_brush_cursor(x, y, visible=True)

        if not self.dragging:
            return

        if self.mode == "box":
            # Just update the marquee — don't touch the heavy overlay
            y0, x0 = self._box_anchor
            self._update_marquee(y0, x0, y, x)
            self.fig.canvas.draw_idle()
            return

        self._paint_line(self.last_pos, (y, x))
        self.last_pos = (y, x)

        # Throttle the heavy overlay redraw to ~5 fps so VNC can keep up
        now = time.time()
        if now - self._last_drag_redraw >= _DRAG_REDRAW_INTERVAL_S:
            self._refresh_overlay()
            self._last_drag_redraw = now

    def _on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False

        if self.mode == "box":
            if self._box_anchor is not None and event.inaxes == self.ax \
               and event.xdata is not None and event.ydata is not None:
                H, W = self.shape
                x = int(np.clip(event.xdata, 0, W - 1))
                y = int(np.clip(event.ydata, 0, H - 1))
                y0, x0 = self._box_anchor
                self._commit_box_erase(y0, x0, y, x)
            self._box_anchor = None
            self.marquee.set_visible(False)
        else:
            self._commit_stroke()

        self._refresh_overlay()
        self._update_title()

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == "e":
            self.mode = "erase";  self._update_title()
        elif k == "d":
            self.mode = "draw";   self._update_title()
        elif k == "b":
            self.mode = "box";    self._update_title()
        elif k == "u":
            self._undo()
        elif k == "[":
            self.brush_radius = max(1, self.brush_radius - 1)
            self._update_title()
        elif k == "]":
            self.brush_radius = min(200, self.brush_radius + 1)
            self._update_title()
        elif k in ("+", "=") :
            self._zoom_at(event, 1.5)
        elif k == "-":
            self._zoom_at(event, 1 / 1.5)
        elif k == "0":
            self._reset_zoom()
        elif k == "g":
            self.show_grid = not self.show_grid
            self._draw_grid()
        elif k == "c":
            self.show_overlay = not self.show_overlay
            self._refresh_overlay()
        elif k == "h":
            print(HELP_TEXT)
        elif k == "s":
            self._save()
        elif k == "q":
            self._quit()

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self) -> None:
        self.plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    import matplotlib
    try:
        matplotlib.use(args.backend)
    except Exception as exc:
        print(f"ERROR: matplotlib backend '{args.backend}' unavailable: {exc}",
              file=sys.stderr)
        sys.exit(1)
    import matplotlib.pyplot as plt

    app = CurationApp(args, plt)
    app.run()


if __name__ == "__main__":
    main()
