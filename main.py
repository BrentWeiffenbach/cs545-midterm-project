"""
CS 545 Mid-Term Project - Master Runner
========================================

Runs every image-enhancement technique sequentially on the same hard-coded
night/day image pair and saves all outputs to organised sub-folders under
results/:

  results/
    clahe/                  ← CLAHE + Gaussian blur + histogram match
    retinex/                ← Single-Scale Retinex (SSR) & Multi-Scale (MSR)
    pa_lhs/                 ← Position-Aware Local Histogram Specification
    img2img/                ← img2img-turbo + classical post-processing
    foundational_models/    ← ChatGPT / foundational model outputs

Run:
  python main.py
"""

import os
import sys
import traceback

# Make src/ importable regardless of where the script is launched from.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

# ─── Hard-coded paths ────────────────────────────────────────────────────────
NIGHT_PATH = "data/night.jpg"
DAY_PATH = "data/day.jpg"
PRIOR_PATH = "checkpoints/lhs_prior.npz"

FOUNDATIONAL_INPUTS = [
    ("ChatGPT Day", "data/chatgpt.jpg"),
    ("Nano Banana", "data/nanoabanana.jpg"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _run_technique(name: str, fn) -> None:
    _section(name)
    try:
        fn()
        print(f"\n[{name}] ✓ Done.")
    except Exception:
        print(f"\n[{name}] ✗ Failed with error:")
        traceback.print_exc()


# ─── Technique runners ────────────────────────────────────────────────────────


def run_clahe():
    import CLAHE

    CLAHE.run(
        night_path=NIGHT_PATH,
        day_path=DAY_PATH,
        output_dir="results/clahe",
    )


def run_retinex():
    import retinex

    retinex.run(
        night_path=NIGHT_PATH,
        day_path=DAY_PATH,
        output_dir="results/retinex",
    )


def run_pa_lhs():
    import importlib

    palhs = importlib.import_module("PA-LHS")
    palhs.run(
        night_path=NIGHT_PATH,
        day_path=DAY_PATH,
        output_dir="results/pa_lhs",
        prior_path=PRIOR_PATH,
    )


def run_img2img():
    import importlib

    img2img = importlib.import_module("img2img-and-classical")
    img2img.run(
        night_path=NIGHT_PATH,
        day_path=DAY_PATH,
        output_dir="results/img2img",
    )


def run_foundational():
    import importlib

    fm = importlib.import_module("foundational-models")
    fm.run(
        ground_truth_path=DAY_PATH,
        inputs=FOUNDATIONAL_INPUTS,
        output_dir="results/foundational_models",
    )


# ─── Full figure builder ──────────────────────────────────────────────────────

# Each entry: (title label, path to pipeline PNG)
_PIPELINE_ROWS = [
    ("CLAHE + Gaussian Blur + Histogram Match", "results/clahe/clahe_pipeline.png"),
    ("Retinex - Multi-Scale (MSR)", "results/retinex/msr_pipeline.png"),
    ("Retinex - Single-Scale (SSR)", "results/retinex/ssr_pipeline.png"),
    (
        "PA-LHS (Position-Aware Local Histogram Specification)",
        "results/pa_lhs/enhanced_pipeline.png",
    ),
    (
        "img2img-turbo + Classical Post-processing",
        "results/img2img/pipeline_overview.png",
    ),
    (
        "Foundational Model - ChatGPT",
        "results/foundational_models/chatgpt_day_pipeline.png",
    ),
    (
        "Foundational Model - Nano Banana",
        "results/foundational_models/nano_banana_pipeline.png",
    ),
]

# Target width for every row in the combined figure (pixels).
_FIGURE_WIDTH = 2000
_TITLE_HEIGHT = 52  # pixels for the title banner above each pipeline row
_ROW_GAP = 16  # vertical gap between rows (white space)
_TITLE_BG = (30, 30, 30)  # dark background for title banner
_TITLE_FG = (255, 255, 255)
_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 1.1
_FONT_THICKNESS = 2


def build_full_figure(output_path="results/full_figure.png") -> None:
    import cv2
    import numpy as np

    rows = []
    for label, img_path in _PIPELINE_ROWS:
        if not os.path.isfile(img_path):
            print(f"  [full figure] Skipping '{label}' - file not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [full figure] Could not read: {img_path}")
            continue

        # Scale the pipeline image to the target width, preserving aspect ratio.
        h, w = img.shape[:2]
        scale = _FIGURE_WIDTH / w
        new_w = _FIGURE_WIDTH
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Build a title banner.
        banner = np.full((_TITLE_HEIGHT, new_w, 3), _TITLE_BG, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        tx = max(8, (new_w - tw) // 2)
        ty = (_TITLE_HEIGHT + th) // 2
        cv2.putText(
            banner,
            label,
            (tx, ty),
            _FONT,
            _FONT_SCALE,
            _TITLE_FG,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

        rows.append(banner)
        rows.append(img)

        # White gap between rows (except after the last one).
        gap = np.full((_ROW_GAP, new_w, 3), 255, dtype=np.uint8)
        rows.append(gap)

    if not rows:
        print("  [full figure] No pipeline images found - skipping.")
        return

    # Drop the trailing gap.
    rows = rows[:-1]

    figure = np.vstack(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, figure)
    print(
        f"\n[Full Figure] Saved → {output_path}  ({figure.shape[1]}×{figure.shape[0]} px)"
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CS 545 Mid-Term Project - running all techniques")
    print(f"Night image : {NIGHT_PATH}")
    print(f"Day image   : {DAY_PATH}")

    _run_technique("CLAHE", run_clahe)
    _run_technique("Retinex (SSR + MSR)", run_retinex)
    _run_technique("PA-LHS", run_pa_lhs)
    _run_technique("img2img + Classical", run_img2img)
    _run_technique("Foundational Models", run_foundational)

    _section("Building combined figure")
    try:
        build_full_figure("results/full_figure.png")
    except Exception:
        traceback.print_exc()

    _section("Summary")
    print("All techniques finished. Results saved to results/<technique>/")
    print()
