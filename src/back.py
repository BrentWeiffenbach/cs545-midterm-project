import argparse
import os

import cv2
import numpy as np


def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def enhance_di_retinex(
    day_image_bgr: np.ndarray,
    night_image_bgr: np.ndarray,
    target_mean: float = 0.50,
    target_std: float = 0.22,
    local_sigma: float = 11.0,
    highlight_threshold: float = 0.98,
) -> np.ndarray:
    day_image = _to_float01(day_image_bgr)
    night_image = _to_float01(night_image_bgr)

    day_intensity = cv2.cvtColor(day_image, cv2.COLOR_BGR2GRAY)
    night_intensity = cv2.cvtColor(night_image, cv2.COLOR_BGR2GRAY)

    day_local_mean = cv2.GaussianBlur(day_intensity, (0, 0), local_sigma)
    night_local_mean = cv2.GaussianBlur(night_intensity, (0, 0), local_sigma)

    day_local_var = cv2.GaussianBlur((day_intensity - day_local_mean) ** 2, (0, 0), local_sigma)
    night_local_var = cv2.GaussianBlur((night_intensity - night_local_mean) ** 2, (0, 0), local_sigma)

    day_local_std = np.sqrt(day_local_var + 1e-6)
    night_local_std = np.sqrt(night_local_var + 1e-6)

    target_mean_map = 0.5 * target_mean + 0.5 * day_local_mean
    target_std_map = 0.5 * target_std + 0.5 * day_local_std

    b_map = np.clip(target_mean_map - night_local_mean, -0.35, 0.55)
    a_map = np.clip(target_std_map / (night_local_std + 1e-3), 0.70, 2.00)

    b_map = np.expand_dims(b_map, axis=2)
    a_map = np.expand_dims(a_map, axis=2)

    enhanced = a_map * (night_image - (1.0 - b_map) * 0.5) + (1.0 + b_map) * 0.5
    enhanced = np.clip(enhanced, 0.0, 1.0)

    mask = (night_image < highlight_threshold).astype(np.float32)
    enhanced = mask * enhanced + (1.0 - mask) * night_image

    return np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="DI-Retinex-inspired low-light enhancement")
    parser.add_argument("--input_day_image", type=str, default="data/day.jpg", help="Input day image path")
    parser.add_argument("--input_night_image", type=str, default="data/night.jpg", help="Input night image path")
    parser.add_argument("--output_image", type=str, default="results/di_retinex.png", help="Output image path")
    parser.add_argument("--target_mean", type=float, default=0.50, help="Target local mean in [0, 1]")
    parser.add_argument("--target_std", type=float, default=0.22, help="Target local contrast std")
    parser.add_argument("--local_sigma", type=float, default=11.0, help="Gaussian sigma for local statistics")
    args = parser.parse_args()

    day_image = cv2.imread(args.input_day_image)
    if day_image is None:
        raise FileNotFoundError(f"Image not found at {os.path.abspath(args.input_day_image)}")

    night_image = cv2.imread(args.input_night_image)
    if night_image is None:
        raise FileNotFoundError(f"Image not found at {os.path.abspath(args.input_night_image)}")

    if day_image.shape != night_image.shape:
        target_size = (min(day_image.shape[1], night_image.shape[1]), min(day_image.shape[0], night_image.shape[0]))
        day_image = cv2.resize(day_image, target_size)
        night_image = cv2.resize(night_image, target_size)

    enhanced = enhance_di_retinex(
        day_image,
        night_image,
        target_mean=args.target_mean,
        target_std=args.target_std,
        local_sigma=args.local_sigma,
    )

    output_dir = os.path.dirname(args.output_image)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output_image, enhanced)
    print(f"Saved DI-Retinex result to: {args.output_image}")


if __name__ == "__main__":
    main()