import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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


def visualize_comparison(input_img, gt_img, input_title, gt_title):
    CANVAS_SIZE = 500
    title_height = 40
    step_margin = 20
    images = [(input_title, input_img), (gt_title, gt_img)]
    n_steps = len(images)
    canvas_height = n_steps * (CANVAS_SIZE + step_margin + title_height) - step_margin
    canvas_width = CANVAS_SIZE
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for idx, (title, img) in enumerate(images):
        title_canvas = np.ones((title_height, CANVAS_SIZE, 3), dtype=np.uint8) * 255
        cv2.putText(
            title_canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        step_img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
        y_start = idx * (CANVAS_SIZE + step_margin + title_height)
        canvas[y_start : y_start + title_height, :, :] = title_canvas
        canvas[y_start + title_height : y_start + title_height + CANVAS_SIZE, :, :] = (
            step_img
        )
    return canvas


def main(input_image_path, ground_truth_path, results_folder):
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        raise FileNotFoundError(
            f"Image not found at {os.path.abspath(input_image_path)}"
        )
    gt_image = cv2.imread(ground_truth_path)
    if gt_image is None:
        raise FileNotFoundError(
            f"Image not found at {os.path.abspath(ground_truth_path)}"
        )

    if input_image.shape != gt_image.shape:
        print("Resizing images to the same dimensions...")
        target_size = (
            min(input_image.shape[1], gt_image.shape[1]),
            min(input_image.shape[0], gt_image.shape[0]),
        )
        input_image = cv2.resize(input_image, target_size)
        gt_image = cv2.resize(gt_image, target_size)

    os.makedirs(results_folder, exist_ok=True)
    comparison_canvas = visualize_comparison(
        input_image, gt_image, "Input Image", "Ground Truth"
    )
    cv2.imwrite(f"{results_folder}/comparison_overview.png", comparison_canvas)

    # MSE calculation
    mse_channels = np.mean(
        (gt_image.astype("float") - input_image.astype("float")) ** 2, axis=(0, 1)
    )
    mse_r, mse_g, mse_b = mse_channels
    mse_overall = (mse_r + mse_g + mse_b) / 3
    print(f"Channel-wise MSE: R={mse_r:.2f}, G={mse_g:.2f}, B={mse_b:.2f}")
    print(f"Averaged MSE (overall): {mse_overall:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Comparison MSE Check")
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to the input image to check",
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True, help="Path to the ground truth image"
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results",
        help="Path to save results images or statistics",
    )
    args = parser.parse_args()
    main(args.input_image, args.ground_truth, args.results_folder)
