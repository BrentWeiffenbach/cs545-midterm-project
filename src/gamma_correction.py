import numpy as np


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.15) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8.")
    if gamma <= 0:
        raise ValueError("Gamma must be > 0.")

    inv_gamma = 1.0 / gamma
    lookup = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.float32)
    lookup = np.clip(lookup, 0, 255).astype(np.uint8)
    return lookup[image]