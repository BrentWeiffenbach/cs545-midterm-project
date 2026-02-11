import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

CANVAS_SIZE = 500
HISTOGRAM_SIZE = 600

STEP_MARGIN = 20

def visualize_pipeline(steps):
    n_steps = len(steps)
    title_height = 40
    canvas_height = n_steps * (CANVAS_SIZE + STEP_MARGIN + title_height) - STEP_MARGIN
    canvas_width = CANVAS_SIZE
    pipeline_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for idx, (title, img) in enumerate(steps):
        # Create title canvas
        title_canvas = np.ones((title_height, CANVAS_SIZE, 3), dtype=np.uint8) * 255
        cv2.putText(title_canvas, f"Step {idx+1}: {title}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        step_img = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE))
        y_start = idx * (CANVAS_SIZE + STEP_MARGIN + title_height)
        pipeline_canvas[y_start:y_start+title_height, :, :] = title_canvas
        pipeline_canvas[y_start+title_height:y_start+title_height+CANVAS_SIZE, :, :] = step_img
    return pipeline_canvas

def plot_histogram_image(image, title):
    # image: BGR uint8
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_xlim(0, 256)
    ax.set_title(title)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
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
    canvas_height = n_steps * (HISTOGRAM_SIZE + STEP_MARGIN + title_height) - STEP_MARGIN
    canvas_width = HISTOGRAM_SIZE
    pipeline_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for idx, (title, img) in enumerate(steps):
        # Create title canvas
        title_canvas = np.ones((title_height, HISTOGRAM_SIZE, 3), dtype=np.uint8) * 255
        cv2.putText(title_canvas, f"Step {idx+1}: {title} Histogram", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        hist_img = plot_histogram_image(img, title)
        hist_img = cv2.resize(hist_img, (HISTOGRAM_SIZE, HISTOGRAM_SIZE))
        y_start = idx * (HISTOGRAM_SIZE + STEP_MARGIN + title_height)
        pipeline_canvas[y_start:y_start+title_height, :, :] = title_canvas
        pipeline_canvas[y_start+title_height:y_start+title_height+HISTOGRAM_SIZE, :, :] = hist_img
    return pipeline_canvas

def main(day_image_path, night_image_path, results_folder):
    # Load the images
    day_image = cv2.imread(day_image_path)
    if day_image is None:
        raise FileNotFoundError(f"Image not found at {os.path.abspath(day_image_path)}")
    night_image = cv2.imread(night_image_path)
    if night_image is None:
        raise FileNotFoundError(f"Image not found at {os.path.abspath(night_image_path)}")
    
    # Resize images to the same size if needed
    # Professor said in class they should be same size, they are not right now so we will need to reupload data later
    if day_image.shape != night_image.shape:
        print("Resizing images to the same dimensions...")
        target_size = (min(day_image.shape[1], night_image.shape[1]), min(day_image.shape[0], night_image.shape[0]))
        day_image = cv2.resize(day_image, target_size)
        night_image = cv2.resize(night_image, target_size)
    

    # Store pipeline steps for visualization
    pipeline_steps = []
    pipeline_steps.append(("Original Night Image", night_image))

    current_image = night_image.copy()
    # Do some enhancements on night_image...
    # TODO
    
    pipeline_steps.append(("Final Processed Image", current_image))

    # Visualize pipeline (images)
    pipeline_canvas = visualize_pipeline(pipeline_steps)
    os.makedirs(results_folder, exist_ok=True)
    cv2.imwrite(f"{results_folder}/pipeline_overview.png", pipeline_canvas)

    # Visualize pipeline (histograms)
    pipeline_hist_canvas = visualize_histogram_pipeline(pipeline_steps)
    cv2.imwrite(f"{results_folder}/pipeline_histogram_overview.png", pipeline_hist_canvas)

    # Results
    # Worst-case Channel-wise MSE comparison between day and original night image
    mse_channels_worst = np.mean((day_image.astype("float") - night_image.astype("float")) ** 2, axis=(0, 1))
    mse_r_worst, mse_g_worst, mse_b_worst = mse_channels_worst
    mse_overall_worst = (mse_r_worst + mse_g_worst + mse_b_worst) / 3
    print(f"Worst-case Channel-wise MSE: R={mse_r_worst:.2f}, G={mse_g_worst:.2f}, B={mse_b_worst:.2f}")
    print(f"Worst-case averaged MSE (overall): {mse_overall_worst:.2f}")

    # Channel-wise MSE comparison between day and final processed night image
    mse_channels = np.mean((day_image.astype("float") - current_image.astype("float")) ** 2, axis=(0, 1))
    mse_r, mse_g, mse_b = mse_channels
    mse_overall = (mse_r + mse_g + mse_b) / 3
    print(f"Channel-wise MSE: R={mse_r:.2f}, G={mse_g:.2f}, B={mse_b:.2f}")
    print(f"Final averaged MSE (overall): {mse_overall:.2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Pipeline")
    parser.add_argument("--input_day_image", type=str, default="data/day.jpg", help="Path to the day input image")
    parser.add_argument("--input_night_image", type=str, default="data/night.jpg", help="Path to the night input image")
    parser.add_argument("--results_folder", type=str, default="results", help="Path to save results images, plots, or statistics")
    args = parser.parse_args()
    main(args.input_day_image, args.input_night_image, args.results_folder)