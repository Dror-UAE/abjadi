"""
Image preprocessing for Musnad OCR inference.

Matches the production pipeline used during training and evaluation.
CPU-safe (OpenCV + NumPy + Pillow only in this module).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "config" / "preprocessing.json"

with CONFIG_PATH.open(encoding="utf-8") as f:
    _CFG = json.load(f)

IMG_SIZE = int(_CFG["image_size"])
MARGIN_RATIO = float(_CFG["margin_ratio"])
LETTERBOX_FILL = int(_CFG["letterbox_fill_gray"])
DESKEW_MAX_ABS = float(_CFG["deskew"]["max_abs_deg"])
DESKEW_MIN_APPLY = float(_CFG["deskew"]["min_apply_deg"])
STONE_CFG = _CFG["stone_preprocess"]


def _to_bgr_uint8(image: Image.Image) -> np.ndarray:
    if image.mode == "RGBA":
        rgb = Image.new("RGB", image.size, (255, 255, 255))
        rgb.paste(image, mask=image.split()[-1])
        arr = np.array(rgb)
    else:
        arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_ink_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3].astype(np.float32) / 255.0
        gray = rgb.mean(axis=2)
        ink = (alpha > 0.1) & (gray < 200)
        if ink.sum() < 10:
            ink = alpha > 0.2
        return ink
    if arr.ndim == 3:
        gray = arr.mean(axis=2)
    else:
        gray = arr.astype(np.float32)
    return gray < 200


def normalize_glyph(
    image: Image.Image,
    size: int = IMG_SIZE,
    margin_ratio: float = MARGIN_RATIO,
) -> Image.Image:
    """Crop to ink, center on white canvas, resize to size x size grayscale."""
    arr = np.array(image.convert("RGBA"))
    ink = _to_ink_mask(arr)
    if not ink.any():
        gray = np.array(image.convert("L"))
        ink = gray < 250

    ys, xs = np.where(ink)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ink_c = ink[y0:y1, x0:x1]

    h, w = ink_c.shape
    glyph = np.full((h, w), 255, dtype=np.uint8)
    glyph[ink_c] = 0

    side = max(h, w)
    pad = int(side * margin_ratio) + 1
    canvas_side = side + 2 * pad
    canvas = np.full((canvas_side, canvas_side), 255, dtype=np.uint8)
    oy = (canvas_side - h) // 2
    ox = (canvas_side - w) // 2
    canvas[oy : oy + h, ox : ox + w] = glyph

    return Image.fromarray(canvas, mode="L").resize(
        (size, size), Image.Resampling.LANCZOS
    )


def prepare_original_view(
    image: Image.Image,
    *,
    already_normalized: bool = False,
    size: int = IMG_SIZE,
) -> Image.Image:
    if already_normalized:
        return image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    gray = image.convert("L")
    w, h = gray.size
    if (w, h) == (size, size):
        return gray
    side = max(w, h)
    canvas = Image.new("L", (side, side), LETTERBOX_FILL)
    canvas.paste(gray, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def _fill_if_hollow(binary_white_ink: np.ndarray) -> np.ndarray:
    h, w = binary_white_ink.shape
    ff = binary_white_ink.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[seed[1], seed[0]] == 0:
            cv2.floodFill(ff, mask, seed, 128)
    holes = (ff == 0).astype(np.uint8) * 255
    filled = cv2.bitwise_or(binary_white_ink, holes)
    if filled.sum() > 3 * max(binary_white_ink.sum(), 1):
        return binary_white_ink
    if holes.sum() > 0.05 * binary_white_ink.size:
        return filled
    return binary_white_ink


def _keep_largest_component(binary_white_ink: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_white_ink, connectivity=8
    )
    if num <= 1:
        return binary_white_ink
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros_like(binary_white_ink)
    thresh = max(int(areas.max() * 0.10), 50)
    for i, area in enumerate(areas, start=1):
        if area >= thresh:
            keep[labels == i] = 255
    if not keep.any():
        largest = 1 + int(np.argmax(areas))
        keep[labels == largest] = 255
    return keep


def _normalize_brightness(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        return img
    out = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_stone_inscription(image: Image.Image) -> Tuple[np.ndarray, dict]:
    """Enhance stone carving photo into black-on-white glyph mask."""
    bgr = _to_bgr_uint8(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    min_side = int(STONE_CFG["upscale_min_side"])
    if max(h, w) < min_side:
        scale = min_side / max(h, w)
        gray = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    clahe = cv2.createCLAHE(
        clipLimit=float(STONE_CFG["clahe_clip"]),
        tileGridSize=(int(STONE_CFG["clahe_tile"]), int(STONE_CFG["clahe_tile"])),
    )
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(
        enhanced,
        d=int(STONE_CFG["bilateral_d"]),
        sigmaColor=float(STONE_CFG["bilateral_sigma_color"]),
        sigmaSpace=float(STONE_CFG["bilateral_sigma_space"]),
    )
    denoised = cv2.medianBlur(denoised, int(STONE_CFG["median_blur_ksize"]))

    blur = cv2.GaussianBlur(denoised, (0, 0), sigmaX=float(STONE_CFG["sharpen_sigma"]))
    sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    k = max(31, (min(sharpened.shape) // 8) | 1)
    background = cv2.medianBlur(sharpened, k)
    grooves = cv2.subtract(background, sharpened)
    grooves = cv2.normalize(grooves, None, 0, 255, cv2.NORM_MINMAX)

    _, groove_mask = cv2.threshold(
        grooves, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    edges = cv2.Canny(
        denoised,
        int(STONE_CFG["canny_low"]),
        int(STONE_CFG["canny_high"]),
    )
    edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    binary = cv2.bitwise_or(groove_mask, edges)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=2)
    binary = _keep_largest_component(binary)
    binary = _fill_if_hollow(binary)
    binary = cv2.dilate(
        binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    ink_on_white = cv2.bitwise_not(binary)
    ink_on_white = _normalize_brightness(ink_on_white)
    return ink_on_white, {}


def stone_to_model_image(image: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    ink_on_white, _ = preprocess_stone_inscription(image)
    pil = Image.fromarray(ink_on_white, mode="L")
    return normalize_glyph(pil.convert("RGBA"), size=size)


def prepare_stone_view(image: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    return stone_to_model_image(image, size=size)


def estimate_glyph_skew_deg(image: Image.Image, *, max_abs: float = DESKEW_MAX_ABS) -> float:
    gray = np.array(image.convert("L"))
    if max(gray.shape) < 8:
        return 0.0
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = []
    for m in (dark, light):
        frac = float((m > 0).mean())
        if 0.02 <= frac <= 0.55:
            candidates.append((abs(frac - 0.18), m))
    mask = min(candidates, key=lambda t: t[0])[1] if candidates else dark
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_cx, _cy), (_w, _h), angle = cv2.minAreaRect(pts)
    if angle < -45:
        angle += 90
    if abs(angle) > max_abs or abs(angle) < DESKEW_MIN_APPLY:
        return 0.0
    return float(-angle)


def deskew_image(
    image: Image.Image,
    *,
    angle: float | None = None,
    max_abs: float = DESKEW_MAX_ABS,
) -> tuple[Image.Image, float]:
    if angle is None:
        angle = estimate_glyph_skew_deg(image, max_abs=max_abs)
    angle = float(np.clip(angle, -max_abs, max_abs))
    if abs(angle) < DESKEW_MIN_APPLY:
        return image, 0.0
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    fill = int(np.median(arr.reshape(-1, 3), axis=0).mean()) if arr.size else LETTERBOX_FILL
    rotated = cv2.warpAffine(
        arr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(fill, fill, fill),
    )
    return Image.fromarray(rotated), angle


def image_to_tensor(gray: Image.Image, size: int = IMG_SIZE):
    """PIL grayscale -> 1x1xHxW float tensor in [0, 1]."""
    import torch

    gray = gray.convert("L")
    w, h = gray.size
    if (w, h) != (size, size):
        side = max(w, h)
        canvas = Image.new("L", (side, side), LETTERBOX_FILL)
        canvas.paste(gray, ((side - w) // 2, (side - h) // 2))
        gray = canvas.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
