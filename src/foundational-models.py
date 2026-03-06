import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    histogram_match,
    mse_between_images,
    visualize_histogram_pipeline,
    visualize_pipeline,
)

# Hardcoded paths
GROUND_TRUTH_PATH = "data/day.jpg"
INPUT_IMAGES = [
    ("ChatGPT Day", "data/chatgpt.jpg"),
    ("Nano Banana", "data/nanoabanana.jpg"),
]
RESULTS_FOLDER = "results/foundational_models"


def process_input(label, input_path, ground_truth):
    """Load input image, resize to match ground truth, histogram-match, and report MSE."""
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {os.path.abspath(input_path)}")

    if img.shape != ground_truth.shape:
        target_size = (ground_truth.shape[1], ground_truth.shape[0])
        img = cv2.resize(img, target_size)

    # Histogram match to ground truth
    img_matched = histogram_match(img, ground_truth)

    # MSE after histogram matching
    a_r, a_g, a_b, a_overall = mse_between_images(img_matched, ground_truth)
    print(f"\n=== {label} — After Histogram Match ===")
    print(f"  Channel-wise MSE: R={a_r:.2f}, G={a_g:.2f}, B={a_b:.2f}")
    print(f"  Averaged MSE (overall): {a_overall:.2f}")

    pipeline_steps = [
        (f"{label} (Original)", img),
        ("Ground Truth (Day)", ground_truth),
        (f"{label} (Hist-Matched)", img_matched),
    ]
    return pipeline_steps


def run(
    ground_truth_path="data/day.jpg",
    inputs=None,
    output_dir="results/foundational_models",
):
    """Callable entry point: process foundational model outputs and save results."""
    if inputs is None:
        inputs = [
            ("ChatGPT Day", "data/chatgpt.jpg"),
            ("Nano Banana", "data/nanoabanana.jpg"),
        ]

    ground_truth = cv2.imread(ground_truth_path)
    if ground_truth is None:
        raise FileNotFoundError(
            f"Ground truth image not found at {os.path.abspath(ground_truth_path)}"
        )

    os.makedirs(output_dir, exist_ok=True)

    for label, input_path in inputs:
        pipeline_steps = process_input(label, input_path, ground_truth)
        safe_label = label.lower().replace(" ", "_")

        out_img = os.path.join(output_dir, f"{safe_label}_pipeline.png")
        out_hist = os.path.join(output_dir, f"{safe_label}_histograms.png")

        cv2.imwrite(out_img, visualize_pipeline(pipeline_steps))
        cv2.imwrite(out_hist, visualize_histogram_pipeline(pipeline_steps))

        print(f"[Foundational] Saved pipeline  \u2192 {out_img}")
        print(f"[Foundational] Saved histograms \u2192 {out_hist}")

    print(f"[Foundational] All results saved to: {output_dir}")


if __name__ == "__main__":
    ground_truth = cv2.imread(GROUND_TRUTH_PATH)
    if ground_truth is None:
        raise FileNotFoundError(
            f"Ground truth image not found at {os.path.abspath(GROUND_TRUTH_PATH)}"
        )

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for label, input_path in INPUT_IMAGES:
        pipeline_steps = process_input(label, input_path, ground_truth)

        safe_label = label.lower().replace(" ", "_")

        pipeline_canvas = visualize_pipeline(pipeline_steps)
        out_img = f"{RESULTS_FOLDER}/{safe_label}_pipeline.png"
        cv2.imwrite(out_img, pipeline_canvas)

        hist_canvas = visualize_histogram_pipeline(pipeline_steps)
        out_hist = f"{RESULTS_FOLDER}/{safe_label}_histograms.png"
        cv2.imwrite(out_hist, hist_canvas)

        print(f"\n  Saved pipeline image  → {out_img}")
        print(f"  Saved histogram image → {out_hist}")

    print(f"\nAll results saved to: {RESULTS_FOLDER}")
