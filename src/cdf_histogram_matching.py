import numpy as np


def _build_cdf_lookup(source_channel: np.ndarray, reference_channel: np.ndarray) -> np.ndarray:
    source_hist = np.bincount(source_channel.ravel(), minlength=256).astype(np.float64)
    reference_hist = np.bincount(reference_channel.ravel(), minlength=256).astype(np.float64)

    source_cdf = np.cumsum(source_hist)
    reference_cdf = np.cumsum(reference_hist)

    source_cdf /= source_cdf[-1]
    reference_cdf /= reference_cdf[-1]

    return np.searchsorted(reference_cdf, source_cdf, side="left").clip(0, 255).astype(np.uint8)


def match_histograms_cdf(source_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    if source_bgr.dtype != np.uint8 or reference_bgr.dtype != np.uint8:
        raise ValueError("Both source and reference images must be uint8 BGR images.")
    if source_bgr.ndim != 3 or reference_bgr.ndim != 3 or source_bgr.shape[2] != 3 or reference_bgr.shape[2] != 3:
        raise ValueError("Both source and reference images must have shape (H, W, 3).")

    matched = np.empty_like(source_bgr)
    for channel_index in range(3):
        lookup = _build_cdf_lookup(source_bgr[:, :, channel_index], reference_bgr[:, :, channel_index])
        matched[:, :, channel_index] = lookup[source_bgr[:, :, channel_index]]

    return matched