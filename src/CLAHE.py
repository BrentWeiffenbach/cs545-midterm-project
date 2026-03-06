"""
Night-to-Day Image Enhancement  —  CS 545 Mid-Term Project
===========================================================

Algorithm: CLAHE + Gaussian Blur + Histogram Match
---------------------------------------------------
  1. CLAHE ×CLAHE_PASSES on the LAB L-channel  — contrast lift without colour shift.
  2. Gaussian blur (σ=FINAL_SIGMA)              — final anti-aliasing / de-blocking.
  3. Histogram match to the reference day image — global colour/tone alignment.

USAGE
-----
  # Enhance a night image:
  python src/CLAHE.py --input data/night.jpg --output results/clahe_enhanced.png

  # Also evaluate MSE against ground truth:
  python src/CLAHE.py --input data/night.jpg --output results/clahe_enhanced.png \\
                      --eval_ref data/day.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    histogram_match,
    load_bgr,
    mse_between_images,
    visualize_histogram_pipeline,
    visualize_pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLAHE_CLIP = 2.0  # CLAHE clip limit
CLAHE_TILE = 4  # CLAHE tileGridSize (square)
CLAHE_PASSES = 2  # number of successive CLAHE applications on L-channel
FINAL_SIGMA = 1.8  # Gaussian blur σ for final anti-aliasing / de-blocking
REF_DAY_PATH = "data/day.jpg"  # reference day image for histogram match


# ─────────────────────────────────────────────────────────────────────────────
# Enhancement
# ─────────────────────────────────────────────────────────────────────────────


def enhance(night_bgr: np.ndarray) -> tuple:
    """
    Enhance a night image with CLAHE on the LAB L-channel, a final Gaussian
    blur, and a histogram match to the reference day image (REF_DAY_PATH).

    Returns (enhanced_image, list_of_pipeline_steps).
    """
    ref_day = load_bgr(REF_DAY_PATH)
    steps = []
    img = night_bgr.copy()
    steps.append(("Input", img.copy()))

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    for _ in range(CLAHE_PASSES):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    steps.append(
        (f"CLAHE ×{CLAHE_PASSES} (clip={CLAHE_CLIP}, tile={CLAHE_TILE})", img.copy())
    )

    img = cv2.GaussianBlur(img, (0, 0), FINAL_SIGMA)
    steps.append((f"Gaussian Blur (σ={FINAL_SIGMA})", img.copy()))

    ref_resized = cv2.resize(ref_day, (img.shape[1], img.shape[0]))
    img = histogram_match(img, ref_resized)
    steps.append(("Histogram Match", img.copy()))

    return img, steps


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Night-to-Day enhancement via CLAHE + Gaussian blur."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/night.jpg",
        help="Path to the night image to enhance.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/clahe/clahe_enhanced.png",
        help="Where to save the enhanced image.",
    )
    parser.add_argument(
        "--eval_ref",
        "--eval",
        dest="eval_ref",
        type=str,
        default="data/day.jpg",
        help="Path to a ground-truth day image for MSE evaluation.",
    )
    args = parser.parse_args()

    night = load_bgr(args.input)
    print(f"Input: {args.input}  {night.shape[1]}×{night.shape[0]}")

    result, steps = enhance(night)

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )
    cv2.imwrite(args.output, result)
    print(f"Enhanced image → {args.output}")

    if args.eval_ref:
        day = load_bgr(args.eval_ref)
        day = cv2.resize(day, (result.shape[1], result.shape[0]))
        night_r = cv2.resize(night, (result.shape[1], result.shape[0]))

        mr, mg, mb, m_all = mse_between_images(night_r, day)
        print(
            f"\nBaseline (raw night vs. day):  R={mr:.1f}  G={mg:.1f}  B={mb:.1f}  overall={m_all:.1f}"
        )

        er, eg, eb, e_all = mse_between_images(result, day)
        print(
            f"Enhanced (CLAHE output vs. day): R={er:.1f}  G={eg:.1f}  B={eb:.1f}  overall={e_all:.1f}"
        )

        pct = (1.0 - e_all / m_all) * 100
        print(f"MSE reduction: {pct:.1f} %")

    base = os.path.splitext(args.output)[0]
    cv2.imwrite(base + "_pipeline.png", visualize_pipeline(steps))
    cv2.imwrite(base + "_histograms.png", visualize_histogram_pipeline(steps))
    print(f"Pipeline visualisation → {base}_pipeline.png")
    print(f"Histogram visualisation → {base}_histograms.png")


def run(
    night_path="data/night.jpg",
    day_path="data/day.jpg",
    output_dir="results/clahe",
):
    """Callable entry point: enhance night_path, save outputs to output_dir."""
    global REF_DAY_PATH
    REF_DAY_PATH = day_path

    night = load_bgr(night_path)
    print(f"[CLAHE] Input: {night_path}  {night.shape[1]}\u00d7{night.shape[0]}")

    result, steps = enhance(night)

    os.makedirs(output_dir, exist_ok=True)
    out_img = os.path.join(output_dir, "clahe_enhanced.png")
    cv2.imwrite(out_img, result)
    print(f"[CLAHE] Enhanced image \u2192 {out_img}")

    day = load_bgr(day_path)
    day = cv2.resize(day, (result.shape[1], result.shape[0]))
    night_r = cv2.resize(night, (result.shape[1], result.shape[0]))

    mr, mg, mb, m_all = mse_between_images(night_r, day)
    print(
        f"[CLAHE] Baseline (raw night vs. day):  "
        f"R={mr:.1f}  G={mg:.1f}  B={mb:.1f}  overall={m_all:.1f}"
    )
    er, eg, eb, e_all = mse_between_images(result, day)
    print(
        f"[CLAHE] Enhanced (CLAHE output vs. day): "
        f"R={er:.1f}  G={eg:.1f}  B={eb:.1f}  overall={e_all:.1f}"
    )
    print(f"[CLAHE] MSE reduction: {(1.0 - e_all / m_all) * 100:.1f} %")

    cv2.imwrite(
        os.path.join(output_dir, "clahe_pipeline.png"), visualize_pipeline(steps)
    )
    cv2.imwrite(
        os.path.join(output_dir, "clahe_histograms.png"),
        visualize_histogram_pipeline(steps),
    )
    print(
        f"[CLAHE] Pipeline visualisation \u2192 {os.path.join(output_dir, 'clahe_pipeline.png')}"
    )
    print(
        f"[CLAHE] Histogram visualisation \u2192 {os.path.join(output_dir, 'clahe_histograms.png')}"
    )

    return result


if __name__ == "__main__":
    main()
