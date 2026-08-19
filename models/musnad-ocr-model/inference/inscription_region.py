"""
Locate the carved Musnad inscription on scene photos and trimmed crops.

Runs *before* line banding when the frame still contains significant blank
stone margins. Tight full-frame inscriptions pass through unchanged.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PACKAGE_ROOT / "model"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
STONE_LINES_DIR = PACKAGE_ROOT / "data" / "stone_lines"

from dataclasses import dataclass

import cv2
import numpy as np

from .stone_glyph_segmentation import (
    LineBounds,
    _band_letter_score,
    _illuminate,
    _norm01,
    _row_letter_signal,
)


@dataclass(frozen=True)
class InscriptionRegion:
    x_left: int
    y_top: int
    x_right: int
    y_bottom: int
    applied: bool
    area_ratio: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "x_left": int(self.x_left),
            "y_top": int(self.y_top),
            "x_right": int(self.x_right),
            "y_bottom": int(self.y_bottom),
            "applied": bool(self.applied),
            "area_ratio": float(self.area_ratio),
            "reason": str(self.reason),
        }


def _full_image_stroke_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Groove + oriented stroke mask for the whole frame (not per-line)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(15, (min(h, w) // 6) | 1)
    background = cv2.medianBlur(gray, k)
    grooves = cv2.subtract(background, gray)
    grooves = cv2.normalize(grooves, None, 0, 255, cv2.NORM_MINMAX)

    f = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    orient = np.maximum(np.abs(gx), np.abs(gy))
    orient_u8 = cv2.normalize(orient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    fused = cv2.addWeighted(grooves, 0.55, orient_u8, 0.45, 0)
    _, mask = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    min_area = max(16, int(0.0006 * h * w))
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n_cc):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        if bw > 0 and bh > 0:
            aspect = max(bw, bh) / float(min(bw, bh))
            if aspect < 1.20 and area < 4 * min_area:
                continue
        clean[labels == i] = 255

    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, ker, iterations=2)
    return clean


def _density_bbox(
    mask: np.ndarray,
    *,
    row_frac: float = 0.08,
    col_frac: float = 0.06,
) -> tuple[int, int, int, int] | None:
    """Tight bbox around rows/columns with sustained stroke density."""
    h, w = mask.shape
    if h < 4 or w < 4:
        return None

    row_den = (mask > 0).mean(axis=1)
    col_den = (mask > 0).mean(axis=0)
    row_thr = max(row_frac, float(np.percentile(row_den, 88)) * 0.55)
    col_thr = max(col_frac, float(np.percentile(col_den, 88)) * 0.55)

    rows = row_den >= row_thr
    cols = col_den >= col_thr
    if not rows.any() or not cols.any():
        ys = np.where(row_den >= max(row_frac * 0.5, row_thr * 0.65))[0]
        xs = np.where(col_den >= max(col_frac * 0.5, col_thr * 0.65))[0]
        if ys.size == 0 or xs.size == 0:
            return None
        return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1

    y0 = int(np.argmax(rows))
    y1 = h - int(np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = w - int(np.argmax(cols[::-1]))
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return None
    return x0, y0, x1, y1


def _mask_bbox(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = image_bgr.shape[:2]
    mask = _full_image_stroke_mask(image_bgr)
    bbox = _density_bbox(mask)
    if bbox is None:
        ys, xs = np.where(mask > 0)
        min_pixels = max(48, int(0.0015 * h * w))
        if xs.size >= min_pixels:
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            if x1 - x0 >= 12 and y1 - y0 >= 12:
                return x0, y0, x1, y1
        return None
    return bbox


def _pad_bbox(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    w: int,
    h: int,
    *,
    pad_frac: float = 0.06,
) -> tuple[int, int, int, int]:
    bw = x1 - x0
    bh = y1 - y0
    px = max(4, int(round(pad_frac * bw)))
    py = max(4, int(round(pad_frac * bh)))
    return (
        max(0, x0 - px),
        max(0, y0 - py),
        min(w, x1 + px),
        min(h, y1 + py),
    )


def _column_letter_signal(ill: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Per-column carving activity inside a horizontal band."""
    strip = ill[y0:y1]
    sh = max(1, y1 - y0)
    f = strip.astype(np.float32)
    k = max(9, (sh // 3) | 1)
    local_mean = cv2.blur(f, (1, k))
    col_var = ((f - local_mean) ** 2).mean(axis=0)

    gy = cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.abs(gy)
    eth = float(np.percentile(edge, 85))
    edge_cov = (edge >= eth).mean(axis=0)

    sig = 0.55 * _norm01(col_var) + 0.45 * _norm01(edge_cov)
    sig = cv2.GaussianBlur(sig.reshape(1, -1), (0, 0), sigmaX=2.0).ravel()
    return _norm01(sig)


def _expand_peak_span(signal: np.ndarray, peak: int, floor_frac: float = 0.22) -> tuple[int, int]:
    """Walk outward from a row peak until the signal falls to ``floor_frac`` of peak."""
    peak_v = float(signal[peak])
    floor = max(0.12, floor_frac * peak_v)
    lo = peak
    while lo > 0 and float(signal[lo - 1]) >= floor:
        lo -= 1
    hi = peak
    h = len(signal)
    while hi < h - 1 and float(signal[hi + 1]) >= floor:
        hi += 1
    return lo, hi + 1


def _row_peaks_bbox(ill: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Multi-line vertical extent from separated row-signal peaks.

    Single-threshold active spans truncate early on the last line when blank
    stone below still has moderate row energy (test-28).
    """
    h, w = ill.shape
    row_s = _row_letter_signal(ill)
    global_max = float(row_s.max())
    if global_max <= 1e-6:
        return None

    peaks: list[int] = []
    for i in range(1, h - 1):
        if row_s[i] >= row_s[i - 1] and row_s[i] >= row_s[i + 1] and float(row_s[i]) >= 0.42 * global_max:
            if not peaks or i - peaks[-1] >= max(6, int(0.04 * h)):
                peaks.append(i)

    if not peaks:
        return None

    y0, y1 = h, 0
    for peak in peaks:
        lo, hi = _expand_peak_span(row_s, peak, floor_frac=0.20)
        y0 = min(y0, lo)
        y1 = max(y1, hi)

    if y1 - y0 < 12:
        return None

    col_s = _column_letter_signal(ill, y0, y1)
    peak_c = float(col_s.max())
    if peak_c <= 1e-6:
        return None
    thr = max(float(np.median(col_s)) + 0.12 * (peak_c - float(np.median(col_s))), 0.24 * peak_c)
    active = col_s >= thr
    if not active.any():
        xs = np.where(col_s >= max(0.18 * peak_c, float(np.median(col_s))))[0]
    else:
        xs = np.where(active)[0]
    if xs.size == 0:
        return None
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    if x1 - x0 < 12:
        return None
    return x0, y0, x1, y1


def _union_bboxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _extend_bbox_for_sparse_lower_line(
    ill: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Grow the bbox downward when a weak / short last text row sits below.

    Dense first lines dominate full-width row signal and stroke masks, so a
    sparse second line (test-22: 3 glyphs) is often cropped away by margin trim.
    Center-strip signal recovers that row the same way line banding does.
    """
    x0, y0, x1, y1 = bbox
    h, w = ill.shape
    if y1 >= h - 6:
        return bbox

    full_s = _row_letter_signal(ill)
    center_s = _row_letter_signal(ill, w // 4, 3 * w // 4)
    main_v = float(full_s[y0:y1].max()) if y1 > y0 else float(full_s.max())
    if main_v <= 1e-6:
        return bbox

    # Search starts after a short gutter below the current block.
    lo = min(h - 4, y1 + max(3, int(0.04 * h)))
    hi = h - 2
    if hi <= lo + 4:
        return bbox

    best: tuple[float, int] | None = None
    for sig in (center_s, full_s):
        for y in range(lo + 1, hi):
            v = float(sig[y])
            # Sparse rows are much weaker than a full first line.
            if v < max(0.14, 0.12 * main_v):
                continue
            if v > 0.55 * main_v:
                # Strong enough that the primary peak finder should have kept it;
                # if we are here it may be tooling noise — still allow when a
                # deep valley separates it from the main block.
                pass
            if not (sig[y] >= sig[y - 1] and sig[y] >= sig[y + 1]):
                continue
            valley = float(full_s[max(0, y1 - 1) : y + 1].min())
            # Need a blank-stone gutter between the dense block and this ridge.
            if valley > min(0.22, 0.55 * v):
                continue
            if valley > 0.40 * main_v:
                continue
            score = v + (0.15 if sig is center_s else 0.0)
            if best is None or score > best[0]:
                best = (score, y)

    if best is None:
        return bbox

    peak = best[1]
    _lo, hi_span = _expand_peak_span(center_s, peak, floor_frac=0.28)
    # Prefer not to stop early on sparse rows — also expand on full signal.
    _lo2, hi2 = _expand_peak_span(full_s, peak, floor_frac=0.22)
    new_y1 = max(y1, hi_span, hi2)

    # Widen x a little using columns inside the recovered lower band only.
    band_top = max(y1, peak - max(6, int(0.08 * h)))
    if new_y1 > band_top + 4:
        col_s = _column_letter_signal(ill, band_top, new_y1)
        thr = max(0.18 * float(col_s.max()), float(np.median(col_s)))
        xs = np.where(col_s >= thr)[0]
        if xs.size:
            x0 = min(x0, int(xs[0]))
            x1 = max(x1, int(xs[-1]) + 1)

    return x0, y0, x1, min(h, new_y1)


def _estimate_inscription_bbox(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Best-effort bbox covering all carved lines; None if nothing letter-like found."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ill = _illuminate(gray)

    candidates: list[tuple[int, int, int, int]] = []
    mask_bbox = _mask_bbox(image_bgr)
    if mask_bbox is not None:
        candidates.append(mask_bbox)

    peaks_bbox = _row_peaks_bbox(ill)
    if peaks_bbox is not None:
        candidates.append(peaks_bbox)

    if not candidates:
        return None

    chosen = _union_bboxes(candidates)
    chosen = _extend_bbox_for_sparse_lower_line(ill, chosen)
    x0, y0, x1, y1 = chosen
    min_h = max(16, int(0.05 * h))
    min_w = max(16, int(0.05 * w))
    if (y1 - y0) < min_h or (x1 - x0) < min_w:
        return None
    if _band_letter_score(ill, LineBounds(y0, y1, 0)) < 0.18:
        return None
    return x0, y0, x1, y1


def _should_apply_region(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[bool, float, str]:
    """
    Crop when blank stone margins are significant.

    Uses margin ratios, not absolute pixel area — test-28 is small but still
    has large irrelevant bottom stone.
    """
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)
    x0, y0, x1, y1 = bbox
    raw_ratio = ((x1 - x0) * (y1 - y0)) / max(img_area, 1.0)

    px0, py0, px1, py1 = _pad_bbox(x0, y0, x1, y1, w, h)
    # Decide on unpadded margins — padding can hide blank stone (test-28 bottom).
    margin_top = y0 / max(h, 1)
    margin_bot = (h - y1) / max(h, 1)
    margin_left = x0 / max(w, 1)
    margin_right = (w - x1) / max(w, 1)
    vertical_margin = margin_top + margin_bot
    horizontal_margin = margin_left + margin_right
    content_height = (y1 - y0) / max(h, 1)
    padded_ratio = ((px1 - px0) * (py1 - py0)) / max(img_area, 1.0)

    # Already a tight full-frame inscription — leave pipeline untouched.
    if (
        raw_ratio >= 0.92
        and vertical_margin < 0.06
        and horizontal_margin < 0.06
    ):
        return False, raw_ratio, "inscription_fills_frame"

    if max(margin_top, margin_bot, margin_left, margin_right) < 0.035:
        return False, raw_ratio, "margins_too_small_to_help"

    if raw_ratio < 0.025:
        return False, raw_ratio, "no_clear_inscription_cluster"

    # Trim when margins are meaningful (blank bottom stone, side padding, …).
    if vertical_margin >= 0.10 or horizontal_margin >= 0.10:
        if h >= 100 or vertical_margin >= 0.18 or horizontal_margin >= 0.18:
            return True, raw_ratio, "margin_trim"

    if margin_bot >= 0.08 or margin_top >= 0.08:
        if h >= 120 or margin_bot >= 0.20 or margin_top >= 0.20:
            return True, raw_ratio, "margin_trim"

    if vertical_margin >= 0.06 and content_height <= 0.88:
        if h >= 120 or vertical_margin >= 0.22:
            return True, raw_ratio, "margin_trim"

    if horizontal_margin >= 0.06 and raw_ratio <= 0.82:
        if h >= 120 or horizontal_margin >= 0.10:
            return True, raw_ratio, "margin_trim"

    return False, raw_ratio, "margins_too_small_to_help"


def isolate_inscription_if_sparse(image_bgr: np.ndarray) -> tuple[np.ndarray, InscriptionRegion]:
    """
    Return ``(work_image, region)``.

    When ``region.applied`` is False, ``work_image`` is the original array
    (same object) and offsets are zero — downstream code behaves as before.
    """
    h, w = image_bgr.shape[:2]
    full = InscriptionRegion(0, 0, w, h, False, 1.0, "full_frame")

    bbox = _estimate_inscription_bbox(image_bgr)
    if bbox is None:
        return image_bgr, InscriptionRegion(0, 0, w, h, False, 1.0, "no_bbox")

    apply, ratio, reason = _should_apply_region(image_bgr, bbox)
    if not apply:
        return image_bgr, InscriptionRegion(0, 0, w, h, False, ratio, reason)

    x0, y0, x1, y1 = _pad_bbox(*bbox, w, h)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return image_bgr, full
    return crop, InscriptionRegion(x0, y0, x1, y1, True, ratio, reason)
