import cv2
import numpy as np


def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def _decompose_illuminance_reflectance(intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    illuminance = cv2.bilateralFilter(intensity, d=9, sigmaColor=0.15, sigmaSpace=15)
    illuminance = np.clip(illuminance, 1e-4, 1.0)
    reflectance = np.clip(intensity / illuminance, 0.0, 2.5)
    return illuminance, reflectance


def enhance_nighttime_image(day_image_bgr: np.ndarray, night_image_bgr: np.ndarray) -> np.ndarray:
    day = _to_float01(day_image_bgr)
    night = _to_float01(night_image_bgr)

    day_hsv = cv2.cvtColor(day, cv2.COLOR_BGR2HSV)
    night_hsv = cv2.cvtColor(night, cv2.COLOR_BGR2HSV)

    day_intensity = day_hsv[:, :, 2]
    night_intensity = night_hsv[:, :, 2]

    night_illum, night_refl = _decompose_illuminance_reflectance(night_intensity)

    day_background = cv2.bilateralFilter(day_intensity, d=9, sigmaColor=0.15, sigmaSpace=15)
    night_background = cv2.bilateralFilter(night_intensity, d=9, sigmaColor=0.15, sigmaSpace=15)

    day_background = np.clip(day_background, 1e-4, 1.0)
    night_background = np.clip(night_background, 1e-4, 1.0)

    illumination_gain = np.clip(day_background / night_background, 0.8, 3.5)
    enhanced_illum = np.clip(night_illum * illumination_gain, 0.0, 1.0)
    enhanced_illum = np.power(enhanced_illum, 0.9)

    enhanced_intensity = np.clip(enhanced_illum * night_refl, 0.0, 1.0)

    enhanced_hsv = night_hsv.copy()
    enhanced_hsv[:, :, 2] = enhanced_intensity
    enhanced_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    return np.clip(enhanced_bgr * 255.0, 0, 255).astype(np.uint8)