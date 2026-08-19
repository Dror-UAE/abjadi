"""Line enhancement for stone OCR (inference subset)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancementConfig:
    """Adjustable preprocessing parameters; all sizes are in input pixels."""

    clahe_clip: float = 3.0
    clahe_tiles_x: int = 8
    clahe_tiles_y: int = 4
    gaussian_sigma: float = 1.2
    median_size: int = 3
    bilateral_diameter: int = 9
    bilateral_sigma_color: float = 45.0
    bilateral_sigma_space: float = 45.0
    unsharp_strength: float = 1.0
    blackhat_sizes: tuple[int, ...] = (7, 15, 25)
    tophat_sizes: tuple[int, ...] = (7, 15, 25)
    morph_open_size: int = 3
    morph_close_size: int = 3
    min_component_area: int = 18
    gradient_low: int = 45
    gradient_high: int = 130
    adaptive_block_size: int = 31
    adaptive_c: float = 5.0


def _illuminate(gray: np.ndarray) -> np.ndarray:
    _h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3.0, w / 28.0))
    norm = cv2.divide(gray, blur, scale=128)
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(norm)


def enhance_line_for_ocr(
    line_bgr: np.ndarray,
    config: EnhancementConfig | None = None,
) -> np.ndarray:
    """
    Preprocessing stage for a line crop before letter detection.

    Non-local means removes pit speckle, bilateral keeps stroke walls intact,
    and CLAHE restores local contrast lost to weathering.
    """
    config = config or EnhancementConfig()
    gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    illuminated = _illuminate(gray)
    denoised = cv2.fastNlMeansDenoising(illuminated, None, 7, 7, 21)
    denoised = cv2.bilateralFilter(
        denoised,
        config.bilateral_diameter,
        config.bilateral_sigma_color,
        config.bilateral_sigma_space,
    )
    return cv2.createCLAHE(
        clipLimit=config.clahe_clip,
        tileGridSize=(config.clahe_tiles_x, config.clahe_tiles_y),
    ).apply(denoised)
