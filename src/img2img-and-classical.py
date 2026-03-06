import argparse
import os
import subprocess
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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


def visualize_pipeline(steps):
    n_steps = len(steps)
    title_height = 40
    canvas_height = n_steps * (CANVAS_SIZE + STEP_MARGIN + title_height) - STEP_MARGIN
    canvas_width = CANVAS_SIZE
    pipeline_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for idx, (title, img) in enumerate(steps):
        title_canvas = np.ones((title_height, CANVAS_SIZE, 3), dtype=np.uint8) * 255
        cv2.putText(
            title_canvas,
            f"Step {idx + 1}: {title}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        step_img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
        y_start = idx * (CANVAS_SIZE + STEP_MARGIN + title_height)
        pipeline_canvas[y_start : y_start + title_height, :, :] = title_canvas
        pipeline_canvas[
            y_start + title_height : y_start + title_height + CANVAS_SIZE, :, :
        ] = step_img
    return pipeline_canvas


def plot_histogram_image(image, title):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    colors = ("b", "g", "r")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_xlim(0, 256)
    ax.set_title(title)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape((height, width, 4))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img


def visualize_histogram_pipeline(steps):
    n_steps = len(steps)
    title_height = 40
    canvas_height = (
        n_steps * (HISTOGRAM_SIZE + STEP_MARGIN + title_height) - STEP_MARGIN
    )
    canvas_width = HISTOGRAM_SIZE
    pipeline_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for idx, (title, img) in enumerate(steps):
        title_canvas = np.ones((title_height, HISTOGRAM_SIZE, 3), dtype=np.uint8) * 255
        cv2.putText(
            title_canvas,
            f"Step {idx + 1}: {title} Histogram",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        hist_img = plot_histogram_image(img, title)
        hist_img = cv2.resize(hist_img, (HISTOGRAM_SIZE, HISTOGRAM_SIZE))
        y_start = idx * (HISTOGRAM_SIZE + STEP_MARGIN + title_height)
        pipeline_canvas[y_start : y_start + title_height, :, :] = title_canvas
        pipeline_canvas[
            y_start + title_height : y_start + title_height + HISTOGRAM_SIZE, :, :
        ] = hist_img
    return pipeline_canvas


def mse_between_images(img1, img2):
    mse_channels = np.mean(
        (img1.astype("float") - img2.astype("float")) ** 2, axis=(0, 1)
    )
    mse_r, mse_g, mse_b = mse_channels
    mse_overall = (mse_r + mse_g + mse_b) / 3
    return mse_r, mse_g, mse_b, mse_overall


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

    def match_histogram(source, reference):
        matched = np.zeros_like(source)
        for c in range(3):
            src = source[:, :, c].ravel()
            ref = reference[:, :, c].ravel()
            src_values, src_indices, src_counts = np.unique(
                src, return_inverse=True, return_counts=True
            )
            ref_values, ref_counts = np.unique(ref, return_counts=True)
            src_cdf = np.cumsum(src_counts).astype(np.float64)
            src_cdf /= src_cdf[-1]
            ref_cdf = np.cumsum(ref_counts).astype(np.float64)
            ref_cdf /= ref_cdf[-1]
            interp_values = np.interp(src_cdf, ref_cdf, ref_values)
            matched[:, :, c] = (
                interp_values[src_indices]
                .reshape(source.shape[0], source.shape[1])
                .astype(np.uint8)
            )
        return matched

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
    current_image = match_histogram(current_image, day_image)
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
