import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

CANVAS_SIZE = 500
HISTOGRAM_SIZE = 600
STEP_MARGIN = 20


def load_bgr(path: str) -> np.ndarray:
    """Load a colour image as uint8 BGR. Raises FileNotFoundError if the file cannot be opened."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {os.path.abspath(path)}")
    return img


def histogram_match(source, reference):
    """Match the histogram of source to reference, channel-wise."""
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


def mse_between_images(img1, img2):
    """Return per-channel (R, G, B) and average MSE between two BGR images."""
    mse_channels = np.mean(
        (img1.astype("float") - img2.astype("float")) ** 2, axis=(0, 1)
    )
    mse_b, mse_g, mse_r = mse_channels
    mse_overall = float(mse_channels.mean())
    return mse_r, mse_g, mse_b, mse_overall


def visualize_pipeline(steps):
    n_steps = len(steps)
    title_height = 40
    canvas_width = n_steps * (CANVAS_SIZE + STEP_MARGIN) - STEP_MARGIN
    canvas_height = title_height + CANVAS_SIZE
    pipeline_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for idx, (title, img) in enumerate(steps):
        x_start = idx * (CANVAS_SIZE + STEP_MARGIN)
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
        pipeline_canvas[0:title_height, x_start : x_start + CANVAS_SIZE] = title_canvas
        pipeline_canvas[
            title_height : title_height + CANVAS_SIZE, x_start : x_start + CANVAS_SIZE
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
    canvas_width = n_steps * (HISTOGRAM_SIZE + STEP_MARGIN) - STEP_MARGIN
    canvas_height = title_height + HISTOGRAM_SIZE
    pipeline_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for idx, (title, img) in enumerate(steps):
        x_start = idx * (HISTOGRAM_SIZE + STEP_MARGIN)
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
        pipeline_canvas[0:title_height, x_start : x_start + HISTOGRAM_SIZE] = (
            title_canvas
        )
        pipeline_canvas[
            title_height : title_height + HISTOGRAM_SIZE,
            x_start : x_start + HISTOGRAM_SIZE,
        ] = hist_img
    return pipeline_canvas
