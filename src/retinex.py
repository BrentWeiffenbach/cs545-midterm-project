import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    mse_between_images,
    visualize_histogram_pipeline,
    visualize_pipeline,
)


def _build_cdf_lookup(
    source_channel: np.ndarray, reference_channel: np.ndarray
) -> np.ndarray:
    source_hist = np.bincount(source_channel.ravel(), minlength=256).astype(np.float64)
    reference_hist = np.bincount(reference_channel.ravel(), minlength=256).astype(
        np.float64
    )

    source_cdf = np.cumsum(source_hist)
    reference_cdf = np.cumsum(reference_hist)

    source_cdf /= source_cdf[-1]
    reference_cdf /= reference_cdf[-1]

    return (
        np.searchsorted(reference_cdf, source_cdf, side="left")
        .clip(0, 255)
        .astype(np.uint8)
    )


def match_histograms_cdf(
    source_bgr: np.ndarray, reference_bgr: np.ndarray
) -> np.ndarray:
    if source_bgr.dtype != np.uint8 or reference_bgr.dtype != np.uint8:
        raise ValueError("Both source and reference images must be uint8 BGR images.")
    if (
        source_bgr.ndim != 3
        or reference_bgr.ndim != 3
        or source_bgr.shape[2] != 3
        or reference_bgr.shape[2] != 3
    ):
        raise ValueError("Both source and reference images must have shape (H, W, 3).")

    matched = np.empty_like(source_bgr)
    for channel_index in range(3):
        lookup = _build_cdf_lookup(
            source_bgr[:, :, channel_index], reference_bgr[:, :, channel_index]
        )
        matched[:, :, channel_index] = lookup[source_bgr[:, :, channel_index]]

    return matched


def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex


def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(
            np.int32(img_retinex[:, :, i] * 100), return_counts=True
        )
        zero_count = 0
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(
            np.minimum(img_retinex[:, :, i], high_val), low_val
        )

        img_retinex[:, :, i] = (
            (img_retinex[:, :, i] - np.min(img_retinex[:, :, i]))
            / (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i]))
            * 255
        )
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(
            np.int32(img_retinex[:, :, i] * 100), return_counts=True
        )
        zero_count = 0
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(
            np.minimum(img_retinex[:, :, i], high_val), low_val
        )

        img_retinex[:, :, i] = (
            (img_retinex[:, :, i] - np.min(img_retinex[:, :, i]))
            / (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i]))
            * 255
        )
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def evaluate_retinex_algorithms(
    algorithms: dict[str, np.ndarray], day_image: np.ndarray
) -> None:
    # Channel-wise MSE comparison between day and final processed night image
    for algorithm, processed_night_image in algorithms.items():
        mse_channels = np.mean(
            (day_image.astype("float") - processed_night_image.astype("float")) ** 2,
            axis=(0, 1),
        )
        mse_r, mse_g, mse_b = mse_channels
        mse_overall = (mse_r + mse_g + mse_b) / 3

        print(
            f"Evaluation for {algorithm}:",
            f"R MSE={mse_r:.2f}, G MSE={mse_g:.2f}, B MSE={mse_b:.2f}, Overall MSE={mse_overall:.2f}",
        )


def post_process(img, reference=None):
    img = cv2.medianBlur(img, 3)
    img = cv2.fastNlMeansDenoisingColored(img, None, 20, 5, 7, 21)
    if reference is None:
        reference = cv2.imread("data/day.jpg")
        assert reference is not None, "Could not read data/day.jpg"
    img = match_histograms_cdf(img, reference)
    return img


def run(
    night_path="data/night.jpg",
    day_path="data/day.jpg",
    output_dir="results/retinex",
):
    variance_list = [15, 80, 30]
    variance = 300

    img = cv2.imread(night_path)
    assert img is not None, f"Could not read {night_path}"
    day_image = cv2.imread(day_path)
    assert day_image is not None, f"Could not read {day_path}"

    if day_image.shape[:2] != img.shape[:2]:
        day_image = cv2.resize(day_image, (img.shape[1], img.shape[0]))

    img_msr_raw = MSR(img.copy(), variance_list)
    img_ssr_raw = SSR(img.copy(), variance)

    img_msr = post_process(img_msr_raw.copy(), day_image)
    img_ssr = post_process(img_ssr_raw.copy(), day_image)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "msr_enhanced.png"), img_msr)
    cv2.imwrite(os.path.join(output_dir, "ssr_enhanced.png"), img_ssr)
    print(f"MSR enhanced image → {os.path.join(output_dir, 'msr_enhanced.png')}")
    print(f"SSR enhanced image → {os.path.join(output_dir, 'ssr_enhanced.png')}")

    msr_steps = [
        ("Input (Night)", img),
        ("After MSR", img_msr_raw),
        ("Post-processed (MSR)", img_msr),
    ]
    cv2.imwrite(
        os.path.join(output_dir, "msr_pipeline.png"), visualize_pipeline(msr_steps)
    )
    cv2.imwrite(
        os.path.join(output_dir, "msr_histograms.png"),
        visualize_histogram_pipeline(msr_steps),
    )
    print(f"MSR pipeline → {os.path.join(output_dir, 'msr_pipeline.png')}")
    print(f"MSR histograms → {os.path.join(output_dir, 'msr_histograms.png')}")

    ssr_steps = [
        ("Input (Night)", img),
        ("After SSR", img_ssr_raw),
        ("Post-processed (SSR)", img_ssr),
    ]
    cv2.imwrite(
        os.path.join(output_dir, "ssr_pipeline.png"), visualize_pipeline(ssr_steps)
    )
    cv2.imwrite(
        os.path.join(output_dir, "ssr_histograms.png"),
        visualize_histogram_pipeline(ssr_steps),
    )
    print(f"SSR pipeline → {os.path.join(output_dir, 'ssr_pipeline.png')}")
    print(f"SSR histograms → {os.path.join(output_dir, 'ssr_histograms.png')}")

    img_r = cv2.resize(img, (day_image.shape[1], day_image.shape[0]))
    mr, mg, mb, m_all = mse_between_images(img_r, day_image)
    print(
        f"\nBaseline (raw night vs. day): "
        f"R={mr:.1f}  G={mg:.1f}  B={mb:.1f}  overall={m_all:.1f}"
    )

    msr_r = cv2.resize(img_msr, (day_image.shape[1], day_image.shape[0]))
    er, eg, eb, e_all = mse_between_images(msr_r, day_image)
    print(
        f"MSR (enhanced vs. day):       "
        f"R={er:.1f}  G={eg:.1f}  B={eb:.1f}  overall={e_all:.1f}"
    )
    print(f"MSR MSE reduction: {(1.0 - e_all / m_all) * 100:.1f} %")

    ssr_r = cv2.resize(img_ssr, (day_image.shape[1], day_image.shape[0]))
    er2, eg2, eb2, e_all2 = mse_between_images(ssr_r, day_image)
    print(
        f"SSR (enhanced vs. day):       "
        f"R={er2:.1f}  G={eg2:.1f}  B={eb2:.1f}  overall={e_all2:.1f}"
    )
    print(f"SSR MSE reduction: {(1.0 - e_all2 / m_all) * 100:.1f} %")

    return img_msr, img_ssr


def main():
    run()


if __name__ == "__main__":
    main()
