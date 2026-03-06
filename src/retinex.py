import cv2
import numpy as np


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


def post_process(img):
    img = cv2.medianBlur(img, 3)
    img = cv2.fastNlMeansDenoisingColored(img, None, 20, 5, 7, 21)
    reference = cv2.imread("data/day.jpg")
    assert reference is not None, "Could not read data/day.jpg"
    img = match_histograms_cdf(img, reference)

    return img


def main():
    variance_list = [15, 80, 30]
    variance = 300

    img = cv2.imread("data/night.jpg")
    assert img is not None, "Could not read data/night.jpg"
    img_msr = MSR(img, variance_list)
    img_ssr = SSR(img, variance)

    img_msr = post_process(img_msr)
    img_ssr = post_process(img_ssr)

    cv2.imshow("Original", img)
    cv2.imshow("MSR", img_msr)
    cv2.imshow("SSR", img_ssr)
    cv2.imwrite("SSR.jpg", img_ssr)
    cv2.imwrite("MSR.jpg", img_msr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    day_image = cv2.imread("data/day.jpg")
    assert day_image is not None, "Could not read data/day.jpg"
    evaluate_retinex_algorithms({"MSR": img_msr, "SSR": img_ssr}, day_image=day_image)


if __name__ == "__main__":
    main()
