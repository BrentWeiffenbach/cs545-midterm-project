import argparse
import os
import subprocess
import sys

import cv2

from utils import (
    histogram_match,
    mse_between_images,
    visualize_histogram_pipeline,
    visualize_pipeline,
)

# Hardcoded configuration
MODEL_NAME = "night_to_day"
INFERENCE_OUTPUT_DIR = "outputs/inference"
RESULTS_FOLDER = "results"
USE_FP16 = False

CANVAS_SIZE = 500
HISTOGRAM_SIZE = 600
STEP_MARGIN = 20


# ── img2img ──────────────────────────────────────────────────────────────────


def run_inference(input_image):
    inference_script = os.path.join(
        os.path.dirname(__file__),
        "src",
        "img2img-turbo",
        "src",
        "inference_unpaired.py",
    )
    cmd = [
        sys.executable,
        inference_script,
        "--model_name",
        MODEL_NAME,
        "--input_image",
        input_image,
        "--output_dir",
        INFERENCE_OUTPUT_DIR,
    ]
    if USE_FP16:
        cmd.append("--use_fp16")

    print(f"Running img2img inference: {MODEL_NAME}")
    subprocess.run(cmd, check=True)
    bname = os.path.basename(input_image)
    return os.path.join(INFERENCE_OUTPUT_DIR, bname)


# ── Image processing pipeline ────────────────────────────────────────────────


def run_pipeline(day_image_path, night_image_path, original_night_image_path):
    day_image = cv2.imread(day_image_path)
    if day_image is None:
        raise FileNotFoundError(f"Image not found at {os.path.abspath(day_image_path)}")
    night_image = cv2.imread(night_image_path)
    if night_image is None:
        raise FileNotFoundError(
            f"Image not found at {os.path.abspath(night_image_path)}"
        )
    original_night_image = cv2.imread(original_night_image_path)
    if original_night_image is None:
        raise FileNotFoundError(
            f"Image not found at {os.path.abspath(original_night_image_path)}"
        )

    shapes = [day_image.shape, night_image.shape, original_night_image.shape]
    if len(set(shapes)) > 1:
        print("Resizing images to the same dimensions...")
        target_size = (
            min(s[1] for s in shapes),
            min(s[0] for s in shapes),
        )
        day_image = cv2.resize(day_image, target_size)
        night_image = cv2.resize(night_image, target_size)
        original_night_image = cv2.resize(original_night_image, target_size)

    pipeline_steps = []
    pipeline_steps.append(("Original Night Image", original_night_image))
    pipeline_steps.append(("img2img Output", night_image))

    current_image = night_image.copy()

    # Step 1: Median blur
    current_image = cv2.medianBlur(current_image, 7)
    pipeline_steps.append(("Median Blur (k=7)", current_image))

    # Step 2: NL-Means denoise
    current_image = cv2.fastNlMeansDenoisingColored(
        current_image,
        None,
        h=20,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    pipeline_steps.append(("NL-Means Denoise", current_image))

    # Step 3: Histogram match
    current_image = histogram_match(current_image, day_image)
    pipeline_steps.append(("Histogram Match", current_image))

    pipeline_steps.append(("Final Processed Image", current_image))

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    cv2.imwrite(f"{RESULTS_FOLDER}/final_processed_image.png", current_image)

    pipeline_canvas = visualize_pipeline(pipeline_steps)
    cv2.imwrite(f"{RESULTS_FOLDER}/pipeline_overview.png", pipeline_canvas)

    pipeline_hist_canvas = visualize_histogram_pipeline(pipeline_steps)
    cv2.imwrite(
        f"{RESULTS_FOLDER}/pipeline_histogram_overview.png", pipeline_hist_canvas
    )

    mse_r_worst, mse_g_worst, mse_b_worst, mse_overall_worst = mse_between_images(
        day_image, original_night_image
    )
    print(
        f"Worst-case Channel-wise MSE: R={mse_r_worst:.2f}, G={mse_g_worst:.2f}, B={mse_b_worst:.2f}"
    )
    print(f"Worst-case averaged MSE (overall): {mse_overall_worst:.2f}")

    mse_r_final, mse_g_final, mse_b_final, mse_overall_final = mse_between_images(
        day_image, current_image
    )
    print(
        f"Final Channel-wise MSE: R={mse_r_final:.2f}, G={mse_g_final:.2f}, B={mse_b_final:.2f}"
    )
    print(f"Final averaged MSE (overall): {mse_overall_final:.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Night-to-Day full pipeline")
    parser.add_argument(
        "--input_night_image",
        type=str,
        default="data/night.jpg",
        help="Path to the input night image",
    )
    parser.add_argument(
        "--input_day_image",
        type=str,
        default="data/day.jpg",
        help="Path to the ground truth day image",
    )
    args = parser.parse_args()

    # Step 1: Run img2img-turbo inference
    img2img_output = run_inference(args.input_night_image)
    print(f"Inference output saved to: {img2img_output}")

    # Step 2: Run classical image processing pipeline on the inference output
    print("Running image processing pipeline...")
    run_pipeline(
        day_image_path=args.input_day_image,
        night_image_path=img2img_output,
        original_night_image_path=args.input_night_image,
    )
    print(f"Final results saved to: {RESULTS_FOLDER}")
