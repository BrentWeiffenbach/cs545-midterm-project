import argparse
import csv
import os

import cv2
import numpy as np

from retinex_algorithms import run_all_retinex_algorithms


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    err = mse(a, b)
    if err == 0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(err))


def brightness(img: np.ndarray) -> float:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def contrast_std(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def entropy_gray(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / np.maximum(np.sum(hist), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def make_grid(images: list[tuple[str, np.ndarray]], tile_size: tuple[int, int] = (420, 320), cols: int = 3) -> np.ndarray:
    title_h = 38
    margin = 12
    tw, th = tile_size
    rows = int(np.ceil(len(images) / cols))

    canvas_h = rows * (th + title_h + margin) + margin
    canvas_w = cols * (tw + margin) + margin
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245

    for idx, (name, img) in enumerate(images):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * (tw + margin)
        y0 = margin + r * (th + title_h + margin)

        tile = cv2.resize(img, (tw, th))
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + tw + 1, y0 + title_h + th + 1), (190, 190, 190), 1)
        cv2.putText(canvas, name, (x0 + 8, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        canvas[y0 + title_h : y0 + title_h + th, x0 : x0 + tw] = tile

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare Retinex enhancement variants")
    parser.add_argument("--input_image", type=str, default="data/night.jpg", help="Input image path")
    parser.add_argument("--reference_image", type=str, default="", help="Optional reference image path for MSE/PSNR")
    parser.add_argument("--output_dir", type=str, default="results/retinex_compare", help="Directory to save outputs")
    parser.add_argument("--ssr_sigma", type=float, default=80, help="Sigma for SSR")
    parser.add_argument("--scales", type=str, default="15,80,250", help="Comma-separated scales for MSR-based methods")
    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    if image is None:
        raise FileNotFoundError(f"Image not found at {os.path.abspath(args.input_image)}")

    scales = tuple(float(part.strip()) for part in args.scales.split(",") if part.strip())
    outputs = run_all_retinex_algorithms(image, ssr_sigma=args.ssr_sigma, sigma_scales=scales)

    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, "input.png"), image)
    for name, out in outputs.items():
        cv2.imwrite(os.path.join(args.output_dir, f"{name}.png"), out)

    reference = None
    if args.reference_image:
        reference = cv2.imread(args.reference_image)
        if reference is None:
            raise FileNotFoundError(f"Reference image not found at {os.path.abspath(args.reference_image)}")
        if reference.shape != image.shape:
            reference = cv2.resize(reference, (image.shape[1], image.shape[0]))

    metrics_rows: list[dict[str, float | str]] = []
    for name, out in outputs.items():
        row: dict[str, float | str] = {
            "method": name,
            "brightness_v_mean": brightness(out),
            "contrast_gray_std": contrast_std(out),
            "entropy_gray": entropy_gray(out),
        }
        if reference is not None:
            row["mse_to_reference"] = mse(reference, out)
            row["psnr_to_reference"] = psnr(reference, out)
        metrics_rows.append(row)

    metrics_csv_path = os.path.join(args.output_dir, "metrics.csv")
    fieldnames = list(metrics_rows[0].keys()) if metrics_rows else ["method"]
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    grid_images = [("Input", image)] + [(name.upper(), out) for name, out in outputs.items()]
    grid = make_grid(grid_images)
    grid_path = os.path.join(args.output_dir, "comparison_grid.png")
    cv2.imwrite(grid_path, grid)

    print("Saved Retinex comparison outputs:")
    print(f"- {grid_path}")
    print(f"- {metrics_csv_path}")
    print(f"- {os.path.join(args.output_dir, 'input.png')}")
    for name in outputs.keys():
        print(f"- {os.path.join(args.output_dir, f'{name}.png')}")


if __name__ == "__main__":
    main()