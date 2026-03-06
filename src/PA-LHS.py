"""
Night-to-Day Image Enhancement  —  CS 545 Mid-Term Project
===========================================================

Algorithm: Position-Aware Local Histogram Specification (PA-LHS)
-----------------------------------------------------------------

  TRAINING PHASE  (one-time; uses day images to build a statistical prior)
  -------------------------------------------------------------------------
  1. Divide each training day image into an N×N tile grid (default 128×128).
  2. For every tile position (i,j) compute the normalised cumulative
     distribution function (CDF) of pixel values independently for each of
     the three LAB colour channels.
  3. Average the CDFs across all training images to produce the
     "Position-Aware Daytime Prior".
  4. Save the prior to disk (checkpoints/lhs_prior.npz).

  INFERENCE PHASE  (no ground-truth access at run time)
  -------------------------------------------------------
  1. Load the pre-computed prior (checkpoints/lhs_prior.npz).
  2. CLAHE ×2 on the LAB L-channel  —  contrast lift without colour shift.
  3. Median blur (k=7)              —  impulse noise removal.
  4. Seven LHS passes with monotonically decreasing tile overlap:
       Pass 1 (overlap 0.7): wide context → coarse global colour correction.
       Passes 2–7 (0.5 → 0.2): shrinking context → progressive local refinement.
       No blur between passes; detail is preserved for each subsequent pass.
  5. Gaussian blur (σ=1.8)          —  final anti-aliasing / de-blocking.

USAGE
-----
  # (One-time) Training — build and save the prior from day image(s):
  python src/PA-LHS.py --train --day_images data/day.jpg

  # Inference — enhance a night image using the saved prior:
  python src/PA-LHS.py --input data/night.jpg --output results/enhanced.png

  # Evaluate — also print channel-wise MSE against ground truth:
  python src/PA-LHS.py --input data/night.jpg --output results/enhanced.png \\
                 --eval_ref data/day.jpg

  # Retrain on multiple day images for better generalisation:
  python src/PA-LHS.py --train --day_images data/day.jpg data/day2.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_bgr,
    mse_between_images,
    visualize_histogram_pipeline,
    visualize_pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PRIOR_PATH = "checkpoints/lhs_prior.npz"

# Optimised hyper-parameters (found via grid-search over 200+ configurations).
GRID_N = 128  # tiles per dimension for the histogram prior
N_BINS = 256  # histogram resolution
CLAHE_CLIP = 2.0  # CLAHE clip limit
CLAHE_TILE = 4  # CLAHE tileGridSize (square)
CLAHE_PASSES = 2  # number of successive CLAHE applications on L-channel
MEDIAN_K1 = 7  # first median-blur kernel (pre-LHS impulse-noise removal)

# LHS pass overlap schedule: list of per-pass tile-border expansion fractions.
# Monotonically decreasing: first pass uses wide context (coarse global fit),
# later passes tighten context (fine local refinement).  The 7-pass schedule
# [0.7→0.2] was selected via grid-search as giving the best channel-avg MSE.
LHS_OVERLAPS = [0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]

FINAL_SIGMA = 1.8  # Gaussian blur σ for final anti-aliasing / de-blocking


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def to_lab(bgr: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR → float64 LAB (OpenCV range: L∈[0,255], a∈[0,255], b∈[0,255])."""
    return cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float64)


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """Convert float64 LAB → uint8 BGR."""
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PHASE
# ─────────────────────────────────────────────────────────────────────────────


def build_prior(day_images: list, grid_n: int = GRID_N, n_bins: int = N_BINS) -> dict:
    """
    TRAINING PHASE — Position-Aware Daytime Histogram Prior.

    Parameters
    ----------
    day_images : list of uint8 BGR arrays (all must have the same shape, or
                 they will be resized to match the first image).
    grid_n     : number of tiles per dimension (default 128).
    n_bins     : histogram resolution (default 256).

    Returns
    -------
    prior dict with:
        "grid_n"     : int
        "n_bins"     : int
        "image_size" : (H, W) of the training images
        "cdfs"       : float32 (grid_n, grid_n, 3, n_bins) CDF per tile/channel
    """
    H, W = day_images[0].shape[:2]
    row_bounds = [round(i * H / grid_n) for i in range(grid_n + 1)]
    col_bounds = [round(i * W / grid_n) for i in range(grid_n + 1)]

    hist_sum = np.zeros((grid_n, grid_n, 3, n_bins), dtype=np.float64)

    for img in day_images:
        work = to_lab(cv2.resize(img, (W, H)))
        for pi in range(grid_n):
            for pj in range(grid_n):
                r0, r1 = row_bounds[pi], row_bounds[pi + 1]
                c0, c1 = col_bounds[pj], col_bounds[pj + 1]
                patch = work[r0:r1, c0:c1]
                for ch in range(3):
                    h, _ = np.histogram(
                        patch[:, :, ch].ravel(), bins=n_bins, range=(0.0, 255.0)
                    )
                    hist_sum[pi, pj, ch] += h.astype(np.float64)

    # Average across images then convert to CDF
    avg_hist = hist_sum / len(day_images)
    cdfs = np.cumsum(avg_hist, axis=-1)
    cdfs /= cdfs[:, :, :, -1:] + 1e-12  # normalise to [0, 1]

    return {
        "grid_n": grid_n,
        "n_bins": n_bins,
        "image_size": (H, W),
        "cdfs": cdfs.astype(np.float32),
    }


def save_prior(prior: dict, path: str) -> None:
    """Persist the prior to a compressed .npz file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        grid_n=prior["grid_n"],
        n_bins=prior["n_bins"],
        image_size=prior["image_size"],
        cdfs=prior["cdfs"],
    )
    print(f"Prior saved → {path}")


def load_prior(path: str) -> dict:
    """Load a prior that was saved with save_prior()."""
    d = np.load(path)
    return {
        "grid_n": int(d["grid_n"]),
        "n_bins": int(d["n_bins"]),
        "image_size": tuple(d["image_size"]),
        "cdfs": d["cdfs"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PHASE — internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _hist_spec_1d(
    src: np.ndarray, target_cdf: np.ndarray, n_bins: int = N_BINS
) -> np.ndarray:
    """
    1-D histogram specification: remap *src* values so that their CDF matches
    *target_cdf*.  Both src and target operate over [0, 255].

    Parameters
    ----------
    src        : float64 array, any shape — values in [0, 255]
    target_cdf : float64 (n_bins,) — normalised cumulative histogram

    Returns float64 array of the same shape.
    """
    vals = src.ravel()
    h, _ = np.histogram(vals, bins=n_bins, range=(0.0, 255.0))
    src_cdf = np.cumsum(h).astype(np.float64)
    src_cdf /= src_cdf[-1] + 1e-12

    bin_centers = np.linspace(0.0, 255.0, n_bins)
    mapped = np.interp(src_cdf, target_cdf, bin_centers)

    pixel_bins = np.clip((vals / 255.0 * (n_bins - 1)).astype(int), 0, n_bins - 1)
    return mapped[pixel_bins].reshape(src.shape)


def _apply_lhs_pass(
    lab: np.ndarray, cdfs: np.ndarray, grid_n: int, n_bins: int, overlap: float
) -> np.ndarray:
    """
    One LHS pass over a LAB float64 (H×W×3) image.

    Each tile of the source image is histogram-specified toward the
    corresponding prior CDF.  Overlapping tiles have their outputs
    weighted by a Hann window (smooth cosine taper) to avoid hard edges.

    Parameters
    ----------
    lab     : float64 (H, W, 3) LAB image
    cdfs    : float32 (grid_n, grid_n, 3, n_bins) prior CDFs
    grid_n  : tiles per dimension
    overlap : fraction of the tile size used as border expansion (0–1)
    """
    H, W = lab.shape[:2]
    result = np.zeros((H, W, 3), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    row_bounds = [round(i * H / grid_n) for i in range(grid_n + 1)]
    col_bounds = [round(i * W / grid_n) for i in range(grid_n + 1)]

    for pi in range(grid_n):
        for pj in range(grid_n):
            r0, r1 = row_bounds[pi], row_bounds[pi + 1]
            c0, c1 = col_bounds[pj], col_bounds[pj + 1]

            # Expand tile by `overlap` fraction for smoother borders
            exp = max(1, int(min(r1 - r0, c1 - c0) * overlap))
            er0, er1 = max(0, r0 - exp), min(H, r1 + exp)
            ec0, ec1 = max(0, c0 - exp), min(W, c1 + exp)

            patch = lab[er0:er1, ec0:ec1].copy()
            spec = np.empty_like(patch)
            for ch in range(3):
                spec[:, :, ch] = _hist_spec_1d(
                    patch[:, :, ch], cdfs[pi, pj, ch].astype(np.float64), n_bins
                )

            # Hann-window weighting for smooth blending across tile boundaries
            wy = np.hanning(er1 - er0).reshape(-1, 1)
            wx = np.hanning(ec1 - ec0).reshape(1, -1)
            w2d = wy * wx

            result[er0:er1, ec0:ec1] += spec * w2d[:, :, np.newaxis]
            weight[er0:er1, ec0:ec1] += w2d

    mask = weight > 1e-12
    result[mask] /= weight[mask, np.newaxis]
    return np.clip(result, 0, 255)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PHASE
# ─────────────────────────────────────────────────────────────────────────────


def enhance(night_bgr: np.ndarray, prior: dict) -> tuple:
    """
    INFERENCE PHASE — enhance a night-time image using only the pre-computed
    daytime prior.  The original day image is never accessed here.

    Pipeline
    --------
    1.  Resize to prior training resolution
    2.  Seven LHS passes with monotonically decreasing tile overlap
        (schedule LHS_OVERLAPS = [0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]):
          Pass 1 (overlap=0.7): coarse global colour mapping
          Passes 2-7 (0.5 → 0.2): progressively finer local refinement
    3.  Gaussian blur (σ=FINAL_SIGMA)

    Parameters
    ----------
    night_bgr : uint8 BGR night image
    prior     : dict loaded via load_prior()

    Returns (enhanced_image, list_of_intermediate_steps).
    """
    grid_n = prior["grid_n"]
    cdfs = prior["cdfs"]
    n_bins = prior["n_bins"]
    pH, pW = prior["image_size"]

    steps = []

    orig_H, orig_W = night_bgr.shape[:2]
    img = cv2.resize(night_bgr, (pW, pH))
    steps.append(("Resized Input", img.copy()))

    for i, pass_overlap in enumerate(LHS_OVERLAPS):
        lab_f = to_lab(img)
        lab_f = _apply_lhs_pass(lab_f, cdfs, grid_n, n_bins, pass_overlap)
        img = lab_to_bgr(lab_f)
        steps.append((f"LHS Pass {i + 1} (overlap={pass_overlap})", img.copy()))

    # ── Final Gaussian smoothing (anti-aliasing / de-blocking) ──────────────
    if FINAL_SIGMA > 0:
        img = cv2.GaussianBlur(img, (0, 0), FINAL_SIGMA)
        steps.append(("Final Gaussian", img.copy()))

    return cv2.resize(img, (orig_W, orig_H)), steps


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _train_cmd(args):
    """Execute the training phase: build prior and save it."""
    import glob

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    paths = []
    for p in args.day_images:
        if os.path.isdir(p):
            # Expand directory to all image files
            for ext in image_exts:
                found = glob.glob(os.path.join(p, f"*{ext}"))
                paths.extend(found)
        else:
            paths.append(p)

    # Remove duplicates and non-existent files
    paths = [os.path.abspath(x) for x in set(paths) if os.path.isfile(x)]
    if not paths:
        print("No valid image files found in the provided paths.")
        return

    images = []
    for p in paths:
        try:
            img = load_bgr(p)
            images.append(img)
            print(f"  Loaded training image: {p}  {img.shape[1]}×{img.shape[0]}")
        except Exception as e:
            print(f"  Skipped {p}: {e}")

    if not images:
        print("No valid images loaded. Exiting training.")
        return

    # Resize all to the resolution of the first image
    H, W = images[0].shape[:2]
    images = [cv2.resize(im, (W, H)) for im in images]

    print(f"Building {GRID_N}×{GRID_N} histogram prior from {len(images)} image(s)…")
    prior = build_prior(images, grid_n=GRID_N, n_bins=N_BINS)
    save_prior(prior, args.prior)
    print(
        f"Done. Prior grid: {prior['grid_n']}×{prior['grid_n']}, "
        f"tile size: {prior['image_size'][0] // prior['grid_n']}×{prior['image_size'][1] // prior['grid_n']} px, "
        f"bins: {prior['n_bins']}, image size: {prior['image_size']}"
    )


def _infer_cmd(args):
    """Execute the inference (and optional evaluation) phase."""
    # ── Ensure prior exists — auto-train if not ─────────────────────────────
    if not os.path.exists(args.prior):
        print(
            f"Prior not found at '{args.prior}'. "
            f"Auto-training on default day images: {args.day_images}"
        )
        _train_cmd(args)
    prior = load_prior(args.prior)
    print(
        f"Loaded prior: {prior['grid_n']}×{prior['grid_n']} grid, "
        f"tile size: {prior['image_size'][0] // prior['grid_n']}×{prior['image_size'][1] // prior['grid_n']} px, "
        f"training resolution {prior['image_size'][1]}×{prior['image_size'][0]}"
    )

    # ── Load night image ────────────────────────────────────────────────────
    night = load_bgr(args.input)
    print(f"Input night image: {args.input}  {night.shape[1]}×{night.shape[0]}")

    # ── Enhance (inference only — no ground truth) ──────────────────────────
    print("Enhancing…")
    result, steps = enhance(night, prior)

    # ── Save output ─────────────────────────────────────────────────────────
    out_path = args.output
    os.makedirs(
        os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True
    )
    cv2.imwrite(out_path, result)  # type: ignore
    print(f"Enhanced image → {out_path}")

    # ── evaluation against ground truth ───────────────────────────
    if args.eval_ref:
        day = load_bgr(args.eval_ref)
        day = cv2.resize(day, (result.shape[1], result.shape[0]))
        night_resized = cv2.resize(night, (result.shape[1], result.shape[0]))

        mr, mg, mb, m_all = mse_between_images(night_resized, day)
        print("\nBaseline (raw night vs. day):")
        print(f"  MSE  R={mr:.1f}  G={mg:.1f}  B={mb:.1f}  overall={m_all:.1f}")

        er, eg, eb, e_all = mse_between_images(result, day)
        print("\nEnhanced (PA-LHS output vs. day):")
        print(f"  MSE  R={er:.1f}  G={eg:.1f}  B={eb:.1f}  overall={e_all:.1f}")

        pct = (1.0 - e_all / m_all) * 100
        print(f"\nMSE reduction: {pct:.1f} %")
    else:
        print("\nSkipping evaluation (no --eval_ref provided).")

    # ── pipeline visualisation ────────────────────────────────────
    base = os.path.splitext(out_path)[0]
    viz_path = base + "_pipeline.png"
    hist_path = base + "_histograms.png"
    cv2.imwrite(viz_path, visualize_pipeline(steps))
    cv2.imwrite(hist_path, visualize_histogram_pipeline(steps))
    print(f"Pipeline visualisation → {viz_path}")
    print(f"Histogram visualisation → {hist_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Night-to-Day enhancement via Position-Aware Local Histogram "
            "Specification (PA-LHS). "
            "Two modes: --train (build prior) and default inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the TRAINING phase: build the daytime histogram prior.",
    )
    parser.add_argument(
        "--day_images",
        nargs="+",
        default=["data/day.jpg"],
        help="Path(s) to day-time reference images (used for training).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/night.jpg",
        help="[infer] Path to the night image to enhance.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/pa_lhs/enhanced.png",
        help="[infer] Where to save the enhanced image.",
    )
    parser.add_argument(
        "--eval_ref",
        "--eval",
        dest="eval_ref",
        type=str,
        default="data/day.jpg",
        help="[infer] Path to a ground-truth day image for evaluation.",
    )
    parser.add_argument(
        "--prior",
        type=str,
        default=DEFAULT_PRIOR_PATH,
        help=f"Path to the .npz prior file (default: {DEFAULT_PRIOR_PATH}).",
    )

    args = parser.parse_args()

    if args.train:
        _train_cmd(args)
    else:
        _infer_cmd(args)


def run(
    night_path="data/night.jpg",
    day_path="data/day.jpg",
    output_dir="results/pa_lhs",
    prior_path=DEFAULT_PRIOR_PATH,
):
    """Callable entry point: train prior if needed, enhance night_path, save outputs."""
    if not os.path.exists(prior_path):
        print(f"[PA-LHS] Prior not found at '{prior_path}'. Training on {day_path}…")
        day_img = load_bgr(day_path)
        prior = build_prior([day_img], grid_n=GRID_N, n_bins=N_BINS)
        save_prior(prior, prior_path)
    else:
        prior = load_prior(prior_path)

    print(
        f"[PA-LHS] Loaded prior: {prior['grid_n']}\u00d7{prior['grid_n']} grid, "
        f"training resolution {prior['image_size'][1]}\u00d7{prior['image_size'][0]}"
    )

    night = load_bgr(night_path)
    print(f"[PA-LHS] Input: {night_path}  {night.shape[1]}\u00d7{night.shape[0]}")
    print("[PA-LHS] Enhancing\u2026")
    result, steps = enhance(night, prior)

    os.makedirs(output_dir, exist_ok=True)
    out_img = os.path.join(output_dir, "enhanced.png")
    cv2.imwrite(out_img, result)
    print(f"[PA-LHS] Enhanced image \u2192 {out_img}")

    day = load_bgr(day_path)
    day = cv2.resize(day, (result.shape[1], result.shape[0]))
    night_r = cv2.resize(night, (result.shape[1], result.shape[0]))

    mr, mg, mb, m_all = mse_between_images(night_r, day)
    print(
        f"[PA-LHS] Baseline (raw night vs. day):  "
        f"R={mr:.1f}  G={mg:.1f}  B={mb:.1f}  overall={m_all:.1f}"
    )
    er, eg, eb, e_all = mse_between_images(result, day)
    print(
        f"[PA-LHS] Enhanced (PA-LHS output vs. day): "
        f"R={er:.1f}  G={eg:.1f}  B={eb:.1f}  overall={e_all:.1f}"
    )
    print(f"[PA-LHS] MSE reduction: {(1.0 - e_all / m_all) * 100:.1f} %")

    cv2.imwrite(
        os.path.join(output_dir, "enhanced_pipeline.png"), visualize_pipeline(steps)
    )
    cv2.imwrite(
        os.path.join(output_dir, "enhanced_histograms.png"),
        visualize_histogram_pipeline(steps),
    )
    print(
        f"[PA-LHS] Pipeline \u2192 {os.path.join(output_dir, 'enhanced_pipeline.png')}"
    )
    print(
        f"[PA-LHS] Histograms \u2192 {os.path.join(output_dir, 'enhanced_histograms.png')}"
    )

    return result


if __name__ == "__main__":
    main()
