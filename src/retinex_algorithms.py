import argparse
import os

import cv2
import numpy as np


def get_ksize(sigma: float) -> int:
    ksize = int(((sigma - 0.8) / 0.15) + 2.0)
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return ksize


def get_gaussian_blur(img: np.ndarray, sigma: float, ksize: int = 0) -> np.ndarray:
    if ksize == 0:
        ksize = get_ksize(sigma)
    sep_k = cv2.getGaussianKernel(ksize, sigma)
    kernel_2d = np.outer(sep_k, sep_k)
    return cv2.filter2D(img, -1, kernel_2d)


def ssr(img: np.ndarray, sigma: float) -> np.ndarray:
    img_f = img.astype(np.float64) + 1.0
    blur = get_gaussian_blur(img_f, sigma=sigma)
    return np.log10(img_f) - np.log10(blur + 1.0)


def msr(img: np.ndarray, sigma_scales: tuple[float, ...] = (15, 80, 250), apply_normalization: bool = True) -> np.ndarray:
    msr_img = np.zeros(img.shape, dtype=np.float64)
    for sigma in sigma_scales:
        msr_img += ssr(img, sigma)
    msr_img = msr_img / float(len(sigma_scales))

    if apply_normalization:
        if msr_img.ndim == 2:
            msr_img = cv2.normalize(msr_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        else:
            msr_img = cv2.normalize(msr_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return msr_img


def color_balance(img: np.ndarray, low_per: float, high_per: float) -> np.ndarray:
    total_pixels = img.shape[0] * img.shape[1]
    low_count = total_pixels * low_per / 100.0
    high_count = total_pixels * (100.0 - high_per) / 100.0

    channels = [img] if img.ndim == 2 else cv2.split(img)
    balanced_channels = []

    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [256], (0, 256)).ravel()
        cum_hist = np.cumsum(hist)
        li, hi = np.searchsorted(cum_hist, (low_count, high_count))
        li = int(np.clip(li, 0, 255))
        hi = int(np.clip(hi, 0, 255))

        if li >= hi:
            balanced_channels.append(ch)
            continue

        lut = np.array(
            [0 if i < li else (255 if i > hi else round((i - li) / (hi - li) * 255)) for i in range(256)],
            dtype=np.uint8,
        )
        balanced_channels.append(cv2.LUT(ch, lut))

    if len(balanced_channels) == 1:
        return balanced_channels[0]
    return cv2.merge(balanced_channels)


def msrcr(
    img: np.ndarray,
    sigma_scales: tuple[float, ...] = (15, 80, 250),
    alpha: float = 125,
    beta: float = 46,
    G: float = 192,
    b: float = -30,
    low_per: float = 1,
    high_per: float = 1,
) -> np.ndarray:
    img_f = img.astype(np.float64) + 1.0
    msr_img = msr(img_f, sigma_scales=sigma_scales, apply_normalization=False)
    crf = beta * (np.log10(alpha * img_f) - np.log10(np.sum(img_f, axis=2, keepdims=True)))
    msrcr_img = G * (msr_img * crf - b)
    msrcr_img = cv2.normalize(msrcr_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    msrcr_img = color_balance(msrcr_img, low_per=low_per, high_per=high_per)
    return msrcr_img


def msrcp(
    img: np.ndarray,
    sigma_scales: tuple[float, ...] = (15, 80, 250),
    low_per: float = 1,
    high_per: float = 1,
) -> np.ndarray:
    img_f = img.astype(np.float64)
    int_img = (np.sum(img_f, axis=2) / img_f.shape[2]) + 1.0

    msr_int = msr(int_img, sigma_scales=sigma_scales, apply_normalization=True)
    msr_cb = color_balance(msr_int, low_per=low_per, high_per=high_per).astype(np.float64) + 1.0

    B = 256.0 / (np.max(img_f, axis=2) + 1.0)
    A = np.minimum(B, msr_cb / int_img)
    msrcp_img = np.clip(np.expand_dims(A, axis=2) * img_f, 0.0, 255.0)
    return msrcp_img.astype(np.uint8)


def normalize_retinex_output(retinex_img: np.ndarray) -> np.ndarray:
    return cv2.normalize(retinex_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)


def run_all_retinex_algorithms(
    img_bgr: np.ndarray,
    ssr_sigma: float = 80,
    sigma_scales: tuple[float, ...] = (15, 80, 250),
) -> dict[str, np.ndarray]:
    ssr_img = normalize_retinex_output(ssr(img_bgr, sigma=ssr_sigma))
    msr_img = msr(img_bgr, sigma_scales=sigma_scales, apply_normalization=True)
    msrcr_img = msrcr(img_bgr, sigma_scales=sigma_scales)
    msrcp_img = msrcp(img_bgr, sigma_scales=sigma_scales)

    return {
        "ssr": ssr_img,
        "msr": msr_img,
        "msrcr": msrcr_img,
        "msrcp": msrcp_img,
    }


def parse_scales(scales_str: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in scales_str.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Retinex algorithms (SSR, MSR, MSRCR, MSRCP)")
    parser.add_argument("--input_image", type=str, default="data/night.jpg", help="Input image path")
    parser.add_argument("--output_dir", type=str, default="results/retinex", help="Directory to save outputs")
    parser.add_argument("--ssr_sigma", type=float, default=80, help="Sigma for SSR")
    parser.add_argument("--scales", type=str, default="15,80,250", help="Comma-separated scales for MSR-based methods")
    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    if image is None:
        raise FileNotFoundError(f"Image not found at {os.path.abspath(args.input_image)}")

    scales = parse_scales(args.scales)
    outputs = run_all_retinex_algorithms(image, ssr_sigma=args.ssr_sigma, sigma_scales=scales)

    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, "input.png"), image)
    for name, out in outputs.items():
        cv2.imwrite(os.path.join(args.output_dir, f"{name}.png"), out)

    print("Saved outputs:")
    print(f"- {os.path.join(args.output_dir, 'input.png')}")
    for name in outputs.keys():
        print(f"- {os.path.join(args.output_dir, f'{name}.png')}")


if __name__ == "__main__":
    main()