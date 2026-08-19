"""
Detect Musnad text lines and experimental per-line glyph boundaries.

The stable first stage returns adaptive (not fixed) y_top and y_bottom for each
line. The optional second stage estimates x_left/x_right separately inside each
line. It does no classification and makes no CNN changes.

Method
------
1. Normalize lighting.
2. Per-row letter signal = local contrast + vertical-edge density.
3. Find line centers as separated peaks in that signal.
4. Cut line bands at the valleys between peaks.
5. Optionally estimate each line's own horizontal pitch and cut at low-activity
   columns near that pitch. Horizontal cuts are never shared between lines.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PACKAGE_ROOT / "model"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
STONE_LINES_DIR = PACKAGE_ROOT / "data" / "stone_lines"


import cv2
import numpy as np




# Signal descent (0..1 scale) required to call a bump its own text line, and the
# relaxed bar used when hunting a short or weathered line that was missed.
_PROM_LINE = 0.15
_PROM_WEAK = 0.06


@dataclass(frozen=True)
class LineBounds:
    y_top: int
    y_bottom: int  # exclusive
    line_index: int = 0

    @property
    def height(self) -> int:
        return max(0, self.y_bottom - self.y_top)

    def to_dict(self) -> dict:
        return {
            "line_index": int(self.line_index),
            "y_top": int(self.y_top),
            "y_bottom": int(self.y_bottom),
            "height": int(self.height),
        }


@dataclass(frozen=True)
class GlyphBounds:
    x_left: int
    x_right: int  # exclusive
    glyph_index: int

    @property
    def width(self) -> int:
        return max(0, self.x_right - self.x_left)

    def to_dict(self) -> dict:
        return {
            "glyph_index": int(self.glyph_index),
            "x_left": int(self.x_left),
            "x_right": int(self.x_right),
            "width": int(self.width),
        }


def load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def _illuminate(gray: np.ndarray) -> np.ndarray:
    _h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3.0, w / 28.0))
    norm = cv2.divide(gray, blur, scale=128)
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(norm)


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - float(x.min())) / (float(x.max() - x.min()) + 1e-6)


def _row_letter_signal(ill: np.ndarray, x0: int | None = None, x1: int | None = None) -> np.ndarray:
    h, w = ill.shape
    if x0 is None:
        x0 = 0
    if x1 is None:
        x1 = w
    strip = ill[:, x0:x1]
    sw = max(1, x1 - x0)
    f = strip.astype(np.float32)
    k = max(9, (sw // 10) | 1)
    local_mean = cv2.blur(f, (k, 1))
    row_var = ((f - local_mean) ** 2).mean(axis=1)

    gx = cv2.Sobel(strip, cv2.CV_32F, 1, 0, ksize=3)
    edge = np.abs(gx)
    eth = float(np.percentile(edge, 85))
    edge_cov = (edge >= eth).mean(axis=1)

    sig = 0.55 * _norm01(row_var) + 0.45 * _norm01(edge_cov)
    # Cap blur so dense tablets keep separate line peaks. Uncapped h/45 (~17 on
    # tall stones) smears neighbouring rows into one hump and merges them.
    sigma = max(1.5, min(6.0, h / 45.0))
    sig = cv2.GaussianBlur(sig.reshape(-1, 1), (0, 0), sigmaX=sigma).ravel()
    return _norm01(sig)


def _find_line_peaks(s: np.ndarray) -> list[int]:
    """Separated peaks in the row signal — one per text line."""
    h = len(s)
    # Prefer local maxima + NMS. Contiguous "active runs" merge when
    # carved horizontal separators keep the signal elevated between lines.
    peaks = _peaks_by_nms(s)
    if len(peaks) >= 1:
        return peaks
    return [int(np.argmax(s))]


def _prominence(s: np.ndarray, peak: int) -> float:
    """
    How far the signal must descend from a peak before it rises higher again.

    Ripples riding on one line's hump barely descend; separate text lines are
    divided by a deep valley of blank stone. This is what tells them apart.
    """
    h = len(s)
    v = float(s[peak])

    left = v
    y = peak - 1
    while y >= 0 and s[y] <= v:
        left = min(left, float(s[y]))
        y -= 1

    right = v
    y = peak + 1
    while y < h and s[y] <= v:
        right = min(right, float(s[y]))
        y += 1

    return v - max(left, right)


def _estimate_line_pitch(s: np.ndarray, seed_peaks: list[int]) -> int:
    """
    Typical spacing between text-line centres.

    Median gap of local maxima is a good start, but a double hump inside one
    line (common on vertically tooled stone) poisons that median low. Autocorr
    then recovers the true longer period when it is clearly larger.
    """
    h = len(s)
    if len(seed_peaks) >= 2:
        med = int(np.median([seed_peaks[i + 1] - seed_peaks[i] for i in range(len(seed_peaks) - 1)]))
    else:
        med = max(24, int(0.12 * h))

    centered = s - float(s.mean())
    ac = np.correlate(centered, centered, mode="full")[h - 1 :]
    ac = ac / (float(ac[0]) + 1e-6)
    # Search from ~seed spacing upward so short intra-line ripples do not win.
    lo = max(16, int(0.85 * med))
    hi = min(h // 2, max(lo + 1, int(2.8 * med), int(0.45 * h)))
    if hi <= lo + 1:
        return med
    ac_pitch = lo + int(np.argmax(ac[lo:hi]))
    ac_score = float(ac[ac_pitch])
    if ac_score >= 0.10 and ac_pitch >= int(1.4 * med):
        return ac_pitch
    if ac_score >= 0.20 and abs(ac_pitch - med) <= int(0.35 * med):
        return int(ac_pitch)
    return med


def _collapse_intra_line_peaks(s: np.ndarray, peaks: list[int], pitch: int) -> list[int]:
    """
    Drop a second hump inside the same text line.

    Tall lines (or lines with strong horizontal serifs) often produce two local
    maxima with only a shallow dip between them. Real line gutters are deep.
    """
    if len(peaks) < 2 or pitch <= 0:
        return peaks
    kept = [peaks[0]]
    for p in peaks[1:]:
        prev = kept[-1]
        gap = p - prev
        valley = float(s[prev : p + 1].min())
        ratio = valley / (min(float(s[prev]), float(s[p])) + 1e-6)
        # Closer than a full line pitch and no deep blank-stone gutter.
        if gap < int(0.85 * pitch) and ratio > 0.65 and valley > 0.45:
            if float(s[p]) > float(s[prev]):
                kept[-1] = p
            continue
        kept.append(p)
    return kept


def _peaks_by_nms(s: np.ndarray) -> list[int]:
    h = len(s)
    # Candidate local maxima that stand clear of their own surroundings
    cands: list[tuple[int, float]] = []
    for y in range(2, h - 2):
        if s[y] >= s[y - 1] and s[y] >= s[y + 1] and s[y] >= s[y - 2] and s[y] >= s[y + 2]:
            if float(s[y]) >= 0.32 and _prominence(s, y) >= _PROM_LINE:
                cands.append((y, float(s[y])))
    if not cands:
        return [int(np.argmax(s))]

    seed = sorted(y for y, _v in cands)
    strong = sorted(cands, key=lambda t: t[1], reverse=True)
    pitch = _estimate_line_pitch(s, seed)
    min_dist = max(12, int(0.55 * pitch))

    kept: list[int] = []
    for y, _v in strong:
        if any(abs(y - p) < min_dist for p in kept):
            continue
        kept.append(y)
    kept = sorted(kept)

    # Recover a missed peak in a large gap (short / weak line)
    recovered: list[int] = []
    for i, p in enumerate(kept):
        recovered.append(p)
        if i + 1 >= len(kept):
            break
        nxt = kept[i + 1]
        if nxt - p < int(1.55 * pitch):
            continue
        lo = p + max(8, int(0.35 * pitch))
        hi = nxt - max(8, int(0.35 * pitch))
        if hi <= lo:
            continue
        mid = lo + int(np.argmax(s[lo:hi]))
        if float(s[mid]) < 0.28 or _prominence(s, mid) < _PROM_WEAK:
            continue
        if all(abs(mid - q) >= min_dist for q in recovered):
            recovered.append(mid)
    # Also check after the last peak for a short final line. Needs a real pitch,
    # so only when two or more lines are already confirmed.
    last = recovered[-1]
    lo = last + max(8, int(0.70 * pitch))
    hi = min(h - 2, last + int(1.55 * pitch))
    if len(kept) >= 2 and hi > lo + 4:
        mid = lo + int(np.argmax(s[lo:hi]))
        # Short last lines can be weak, but must still be a bump of their own —
        # not just the trailing shoulder of the line above
        if float(s[mid]) >= 0.22 and _prominence(s, mid) >= _PROM_WEAK:
            if all(abs(mid - q) >= max(10, int(0.55 * pitch)) for q in recovered):
                recovered.append(mid)

    recovered = sorted(recovered)
    # Re-estimate pitch after NMS, then collapse double-humps inside one line.
    pitch = _estimate_line_pitch(s, recovered)
    return _collapse_intra_line_peaks(s, recovered, pitch)


def _maybe_add_short_last_line(peaks: list[int], center_s: np.ndarray) -> list[int]:
    """Recover a short centered last line that full-width signal misses."""
    # Without two known lines there is no trustworthy pitch, and guessing one
    # turns ripples inside a single line into extra lines.
    if len(peaks) < 2:
        return peaks
    h = len(center_s)
    last = peaks[-1]
    pitch = int(np.median([peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]))
    lo = last + max(10, int(0.65 * pitch))
    hi = min(h - 2, last + int(1.55 * pitch))
    if hi <= lo + 4:
        return peaks
    # Prefer a local maximum, not just the argmax of a declining shoulder
    best_y = None
    best_v = -1.0
    for y in range(lo + 2, hi - 2):
        v = float(center_s[y])
        if v < 0.48:
            continue
        if v >= float(center_s[y - 2]) and v >= float(center_s[y + 2]) and v > best_v:
            if _prominence(center_s, y) >= _PROM_WEAK:
                best_v = v
                best_y = y
    if best_y is None:
        return peaks
    if all(abs(best_y - p) >= max(10, int(0.55 * pitch)) for p in peaks):
        return sorted(peaks + [best_y])
    return peaks


def _maybe_add_weak_first_line(peaks: list[int], s: np.ndarray, center_s: np.ndarray) -> list[int]:
    """Recover a faint top row sitting above the first strong line peak."""
    if len(peaks) < 2:
        return peaks
    h = len(s)
    first = peaks[0]
    pitch = int(np.median([peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]))
    # Need room for a full prior line above the current first peak.
    if first < max(16, int(0.55 * pitch)):
        return peaks

    lo = 2
    hi = first - max(6, int(0.30 * pitch))
    if hi <= lo + 3:
        return peaks

    best: tuple[float, int] | None = None
    for sig in (s, center_s):
        for y in range(lo + 2, hi - 1):
            v = float(sig[y])
            if v < 0.18:
                continue
            if not (sig[y] >= sig[y - 2] and sig[y] >= sig[y + 2]):
                continue
            prom = _prominence(sig, y)
            if prom < 0.08:
                continue
            # Must also be a real bump in the full-width signal — center-only
            # edge noise (common on short crops) should not invent a top line.
            if float(s[y]) < 0.18 or _prominence(s, y) < 0.08:
                continue
            valley = float(sig[y : first + 1].min())
            if valley > min(0.20, 0.55 * v):
                continue
            # Keep spacing close to the tablet pitch.
            if abs((first - y) - pitch) > 0.55 * pitch:
                continue
            score = v + prom
            if best is None or score > best[0]:
                best = (score, y)

    if best is None:
        return peaks
    return sorted(peaks + [best[1]])


def _maybe_add_edge_last_line(peaks: list[int], s: np.ndarray, center_s: np.ndarray) -> list[int]:
    """
    Recover a bottom row clipped by the image edge.

    When the last glyphs sit on the frame border, the row signal rises again
    after the final valley but never forms a clean local maximum.
    """
    if len(peaks) < 2:
        return peaks
    h = len(s)
    last = peaks[-1]
    pitch = int(np.median([peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]))
    # Expected last-line centre near one pitch below the current last peak.
    target = last + pitch
    if target > h - 4:
        return peaks
    if h - last < max(14, int(0.55 * pitch)):
        return peaks

    valley_hi = min(h - 2, last + max(8, int(0.70 * pitch)))
    if valley_hi <= last + 2:
        return peaks
    valley_y = last + int(np.argmin(s[last : valley_hi + 1]))
    valley_v = float(s[valley_y])
    # Must actually descend after the known last line.
    if valley_v > 0.55 * float(s[last]):
        return peaks

    lo = valley_y + 2
    hi = h - 1
    if hi <= lo + 3:
        return peaks

    # Prefer a local bump; otherwise take the strongest rise toward the edge.
    best_y = None
    best_v = -1.0
    for sig in (center_s, s):
        for y in range(lo + 1, hi):
            v = float(sig[y])
            if v < 0.20:
                continue
            local = y < h - 3 and sig[y] >= sig[y - 2] and sig[y] >= sig[min(h - 1, y + 2)]
            near_edge = y >= h - max(8, int(0.35 * pitch))
            if not (local or near_edge):
                continue
            if abs(y - target) > 0.60 * pitch and not near_edge:
                continue
            if v > best_v:
                best_v = v
                best_y = y

    if best_y is None:
        return peaks
    # Require a real re-rise after the valley.
    if best_v < valley_v + 0.05 and best_v < 0.28:
        return peaks
    if any(abs(best_y - p) < max(10, int(0.50 * pitch)) for p in peaks):
        return peaks
    return sorted(peaks + [best_y])


def _maybe_add_weak_second_line(peaks: list[int], s: np.ndarray, center_s: np.ndarray) -> list[int]:
    """Recover a small second row when the dominant first row hides it."""
    if len(peaks) != 1:
        return peaks
    h = len(s)
    first = peaks[0]
    if first > int(0.55 * h):
        return peaks

    lo = first + max(18, int(0.38 * h))
    hi = h - 2
    if hi <= lo + 4:
        return peaks

    best: tuple[float, int] | None = None
    for sig in (s, center_s):
        for y in range(lo + 2, hi - 2):
            v = float(sig[y])
            if v < 0.20:
                continue
            if not (sig[y] >= sig[y - 2] and sig[y] >= sig[y + 2]):
                continue
            prom = _prominence(sig, y)
            if prom < 0.10:
                continue
            valley = float(sig[first : y + 1].min())
            if valley > min(0.18, 0.50 * v):
                continue
            score = v + prom
            if best is None or score > best[0]:
                best = (score, y)

    if best is None:
        return peaks
    second = best[1]
    if h - second < max(10, int(0.10 * h)):
        return peaks
    return sorted(peaks + [second])


def _bounds_around_peak(s: np.ndarray, peak: int, lo: int, hi: int) -> tuple[int, int]:
    """Top/bottom for one line, searching only inside [lo, hi)."""
    d = np.gradient(s)
    # Top: earliest strong rise between lo and peak
    upper = d[lo:peak] if peak > lo else np.array([0.0])
    max_rise = float(upper.max()) if upper.size else 0.0
    if max_rise > 1e-6:
        rise_thr = 0.85 * max_rise
        rise_ys = [lo + int(i) for i in range(len(upper)) if upper[i] >= rise_thr]
        y_top = min(rise_ys) if rise_ys else lo + int(np.argmax(upper))
    else:
        y_top = lo

    # Bottom: latest strong fall between peak and hi
    lower = d[peak:hi] if hi > peak else np.array([0.0])
    max_fall = float(lower.min()) if lower.size else 0.0
    if max_fall < -1e-6:
        fall_thr = 0.85 * max_fall
        fall_ys = [
            peak + int(i)
            for i in range(len(lower))
            if lower[i] <= fall_thr and s[peak + i] < 0.55
        ]
        if not fall_ys:
            fall_ys = [peak + int(np.argmin(lower))]
        y_bottom = max(fall_ys) + 1
    else:
        y_bottom = hi

    y_bottom = min(hi, y_bottom)
    y_top = max(lo, y_top)

    band_h = max(1, y_bottom - y_top)
    pad = max(2, int(round(0.045 * band_h)))
    y_top = max(lo, y_top - pad)
    y_bottom = min(hi, y_bottom + pad)
    # Extra margin so the crop border does not sit on the stroke tips
    y_top = max(lo, y_top - 3)
    y_bottom = min(hi, y_bottom + 3)
    return y_top, y_bottom


def _valley(sig: np.ndarray, a: int, b: int) -> int:
    a = max(0, a)
    b = min(len(sig), b)
    if b <= a + 1:
        return a
    return a + int(np.argmin(sig[a:b]))


def _bands_from_valleys(
    s: np.ndarray,
    center_s: np.ndarray,
    peaks: list[int],
    n_full: int,
    h: int,
) -> list[LineBounds]:
    """
    Cut each line at the lowest point of the signal between neighbouring peaks.

    The valley is the blank stone between two lines, so cutting there can never
    clip a stroke tip — unlike a slope threshold, which stops early wherever the
    carving fades out gradually. Peaks past index `n_full` were found in the
    centre strip, so their own valleys must be read from that signal too.
    """
    pitch = int(np.median(np.diff(peaks))) if len(peaks) > 1 else h

    def sig_for(i: int) -> np.ndarray:
        return center_s if i >= n_full else s

    edges = [_valley(sig_for(0), peaks[0] - pitch, peaks[0] + 1)]
    for i in range(len(peaks) - 1):
        sig = center_s if (i + 1) >= n_full else s
        edges.append(_valley(sig, peaks[i], peaks[i + 1]))
    last = len(peaks) - 1
    edges.append(_valley(sig_for(last), peaks[last], peaks[last] + pitch + 1))

    return [
        LineBounds(y_top=edges[i], y_bottom=min(h, edges[i + 1]), line_index=i)
        for i in range(len(peaks))
    ]


def _expand_line_band_edges(
    bounds: list[LineBounds],
    s: np.ndarray,
    h: int,
) -> list[LineBounds]:
    """
    Grow line crops so borders are not sitting on stroke tips.

    Valley cuts place the shared edge in the blank between rows with no pad.
    Dense tablets (e.g. test-28) then clip tall stems at the top/bottom of each
    band. Expand into remaining letter signal on the outer edges, and allow a
    small overlap across the shared valley.
    """
    if not bounds or h <= 0 or s.size == 0:
        return bounds

    out = [LineBounds(b.y_top, b.y_bottom, i) for i, b in enumerate(bounds)]
    med_h = float(np.median([max(1, b.height) for b in out]))
    edge_pad = max(6, int(round(0.14 * med_h)))
    share_pad = max(4, int(round(0.10 * med_h)))

    # Outer top: walk into letter signal, then pad.
    y0, y1 = out[0].y_top, out[0].y_bottom
    peak = float(s[y0:y1].max()) if y1 > y0 else 1.0
    thr = max(0.22, 0.40 * peak)
    top = y0
    while top > 0 and float(s[top - 1]) >= thr:
        top -= 1
    top = max(0, top - edge_pad)
    out[0] = LineBounds(top, out[0].y_bottom, 0)

    # Outer bottom.
    y0, y1 = out[-1].y_top, out[-1].y_bottom
    peak = float(s[y0:y1].max()) if y1 > y0 else 1.0
    thr = max(0.22, 0.40 * peak)
    bot = y1
    while bot < h and float(s[bot]) >= thr:
        bot += 1
    bot = min(h, bot + edge_pad)
    out[-1] = LineBounds(out[-1].y_top, bot, len(out) - 1)

    # Shared seams: slight overlap so the valley cut does not clip tips.
    if len(out) >= 2:
        min_h = max(8, int(0.45 * med_h))
        for i in range(len(out) - 1):
            a = out[i]
            b = out[i + 1]
            seam = (a.y_bottom + b.y_top) // 2
            a_bot = min(h, max(a.y_bottom, seam + share_pad))
            b_top = max(0, min(b.y_top, seam - share_pad))
            if a_bot - a.y_top >= min_h:
                out[i] = LineBounds(a.y_top, a_bot, i)
            if b.y_bottom - b_top >= min_h:
                out[i + 1] = LineBounds(b_top, b.y_bottom, i + 1)

    return [LineBounds(b.y_top, b.y_bottom, i) for i, b in enumerate(out)]


def _merge_tiny_internal_bands(bounds: list[LineBounds], h: int) -> list[LineBounds]:
    """Undo spurious slices caused by a second hump inside one text row."""
    if len(bounds) < 2:
        return bounds
    tiny_h = max(8, int(0.16 * h))
    # Leading tiny strip is almost never a real first text line.
    if len(bounds) >= 2 and bounds[0].height <= tiny_h and bounds[0].height < 0.55 * bounds[1].height:
        bounds = [
            LineBounds(bounds[0].y_top, bounds[1].y_bottom, 0),
            *bounds[2:],
        ]
        bounds = [LineBounds(b.y_top, b.y_bottom, i) for i, b in enumerate(bounds)]
    if len(bounds) < 3:
        return bounds
    merged: list[LineBounds] = []
    i = 0
    while i < len(bounds):
        cur = bounds[i]
        if 0 < i < len(bounds) - 1:
            prev = merged[-1] if merged else bounds[i - 1]
            nxt = bounds[i + 1]
            much_shorter = cur.height < 0.60 * min(prev.height, nxt.height)
            if cur.height <= tiny_h and much_shorter:
                # A thin internal band is usually the top of the following row
                # on compact inscriptions, not an independent line.
                merged.append(LineBounds(cur.y_top, nxt.y_bottom, cur.line_index))
                i += 2
                continue
        merged.append(cur)
        i += 1
    return [
        LineBounds(b.y_top, b.y_bottom, idx)
        for idx, b in enumerate(merged)
    ]


def _looks_like_single_line_crop(h: int, w: int) -> bool:
    """
    Panoramic / line-crop photos are almost always one text row.

    Multi-line tablets are taller relative to width (aspect closer to 1–3.5).
    A short wide strip (aspect ≳ 4) with stone grain creates fake horizontal
    peaks that must not become extra lines.
    """
    if h < 8 or w < 8:
        return True
    aspect = w / float(h)
    # Hard gate: panoramic inscription strips (e.g. test-24 ~7.6).
    # Keep milder aspects for multi-line tablets (e.g. test-8 ~3.4).
    if aspect >= 4.0:
        return True
    if aspect >= 5.5 and h <= 280:
        return True
    return False


def _band_letter_score(ill: np.ndarray, band: LineBounds) -> float:
    """
    How letter-like a horizontal band is vs bare stone grain.

    Real Musnad lines have concentrated vertical-edge energy and a clear
    horizontal activity ridge. Texture strips are flatter / noisier.
    """
    y0, y1 = band.y_top, band.y_bottom
    if y1 - y0 < 4:
        return 0.0
    strip = ill[y0:y1]
    f = strip.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # Vertical strokes dominate carved letters; grain is more isotropic.
    vert = float(np.mean(np.abs(gx)))
    horiz = float(np.mean(np.abs(gy)))
    anisotropy = vert / (horiz + 1e-6)
    edge_peak = float(np.percentile(mag, 90))
    edge_mean = float(mag.mean())
    # Column activity: letters create peaks across x; texture is flatter.
    col = mag.mean(axis=0)
    col_n = _norm01(col)
    col_peak = float(col_n.max()) if col_n.size else 0.0
    col_std = float(col_n.std()) if col_n.size else 0.0
    # Prefer bands with strong edges + vertical bias + structured columns.
    score = (
        0.35 * min(1.5, anisotropy)
        + 0.30 * min(1.0, edge_peak / (edge_mean + 1e-6) / 4.0)
        + 0.20 * col_peak
        + 0.15 * min(1.0, col_std * 3.0)
    )
    return float(score)


def _collapse_shallow_texture_peaks(s: np.ndarray, peaks: list[int]) -> list[int]:
    """
    Keep only peaks separated by a real blank-stone gutter.

    Intra-line texture (tooling, shadows, crossbars on tall glyphs) produces
    shallow dips between peaks that sit near a false short "pitch". Real
    multi-line tablets with packed rows can also have elevated valleys — those
    are kept only when many near-pitch peaks span a large fraction of the image
    (packed tablet), not when 2–3 humps sit inside one tall line (test-25).
    """
    if len(peaks) < 2:
        return peaks
    h = len(s)
    pitch = max(12, _estimate_line_pitch(s, peaks))
    strengths = [float(s[p]) for p in peaks]
    best = int(np.argmax(strengths))
    gaps = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    near_pitch = sum(1 for g in gaps if 0.70 * pitch <= g <= 1.45 * pitch)
    span = peaks[-1] - peaks[0]
    # Packed multi-line tablet (e.g. test-19): several rows, regular spacing,
    # peaks covering much of the inscription. Do NOT treat a short crop with
    # 2–3 intra-glyph humps as packed.
    packed_tablet = (
        len(peaks) >= 4
        and near_pitch >= max(2, len(gaps) - 1)
        and span >= int(0.45 * h)
    )
    # One peak clearly dominates → panoramic / single-line crop with ripples.
    if (not packed_tablet) and strengths[best] >= 1.35 * float(np.median(strengths)):
        shallow = 0
        for i, p in enumerate(peaks):
            if i == best:
                continue
            a, b = sorted((peaks[best], p))
            valley = float(s[a : b + 1].min())
            ratio = valley / (min(strengths[best], strengths[i]) + 1e-6)
            if ratio > 0.55:
                shallow += 1
        if shallow >= len(peaks) - 1:
            return [peaks[best]]

    kept = [peaks[0]]
    for p in peaks[1:]:
        prev = kept[-1]
        gap = p - prev
        valley = float(s[prev : p + 1].min())
        ratio = valley / (min(float(s[prev]), float(s[p])) + 1e-6)
        if packed_tablet:
            # Keep near-pitch rows even with elevated valleys; only merge clear
            # intra-line ripples (sub-pitch) or nearly flat near-pitch dips.
            sub_pitch = gap < int(0.78 * pitch)
            flat_near = gap <= int(1.20 * pitch) and ratio > 0.84
            collapse = (sub_pitch and ratio > 0.58) or flat_near
        else:
            # Default: shallow valley → same text line / glyph substructure.
            collapse = ratio > 0.58
        if collapse:
            if float(s[p]) > float(s[prev]):
                kept[-1] = p
            continue
        kept.append(p)
    return kept


def _filter_texture_bands(
    bounds: list[LineBounds],
    ill: np.ndarray,
    s: np.ndarray,
) -> list[LineBounds]:
    """Drop bands that look like bare stone; keep letter-like ridges."""
    if len(bounds) <= 1:
        return bounds
    scores = [_band_letter_score(ill, b) for b in bounds]
    best = float(max(scores))
    if best <= 1e-6:
        # Fall back to strongest row-signal band.
        peak_scores = [float(s[max(0, min(len(s) - 1, (b.y_top + b.y_bottom) // 2))]) for b in bounds]
        best_i = int(np.argmax(peak_scores))
        return [LineBounds(bounds[best_i].y_top, bounds[best_i].y_bottom, 0)]

    kept = [
        b for b, sc in zip(bounds, scores) if sc >= max(0.45 * best, best - 0.35)
    ]
    if not kept:
        best_i = int(np.argmax(scores))
        kept = [bounds[best_i]]
    # If survivors are few and one score dominates heavily, keep only that.
    if len(kept) >= 2:
        kept_scores = [_band_letter_score(ill, b) for b in kept]
        top = float(max(kept_scores))
        if top >= 1.45 * float(np.median(kept_scores)):
            kept = [kept[int(np.argmax(kept_scores))]]
    return [
        LineBounds(b.y_top, b.y_bottom, i) for i, b in enumerate(kept)
    ]


def _split_one_tall_band(
    band: LineBounds,
    s: np.ndarray,
    center_s: np.ndarray,
    ill: np.ndarray,
    ref_height: float,
) -> list[LineBounds]:
    """
    If a single band still holds two text rows, cut it at the blank-stone valley.

    Used after the main peak pass: under-split collapse can leave a ~2×-tall band
    (e.g. compact tablets). Only split when both halves look letter-like.
    """
    y0, y1 = band.y_top, band.y_bottom
    height = y1 - y0
    if height < max(28, int(1.60 * ref_height)):
        return [band]

    local = s[y0:y1]
    local_c = center_s[y0:y1]
    if local.size < 16:
        return [band]

    # Prefer peaks already visible in the global/center signals, remapped local.
    raw_peaks: list[int] = []
    for sig in (local, local_c):
        for y in range(3, len(sig) - 3):
            v = float(sig[y])
            if v < 0.18:
                continue
            if not (sig[y] >= sig[y - 2] and sig[y] >= sig[y + 2]):
                continue
            if _prominence(sig, y) < 0.08:
                continue
            raw_peaks.append(y)
    if len(raw_peaks) < 2:
        # Fallback: reuse the standard peak finder on the band crop signal.
        found = _find_line_peaks(local)
        found = _collapse_shallow_texture_peaks(local, found)
        raw_peaks = list(found)
    if len(raw_peaks) < 2:
        return [band]

    # Dedup nearby peaks; keep strongest in each cluster.
    raw_peaks = sorted(set(raw_peaks))
    clustered: list[int] = []
    min_gap = max(10, int(0.45 * ref_height))
    for p in raw_peaks:
        if not clustered:
            clustered.append(p)
            continue
        if p - clustered[-1] < min_gap:
            prev = clustered[-1]
            if float(local[p]) > float(local[prev]):
                clustered[-1] = p
            continue
        clustered.append(p)
    if len(clustered) < 2:
        return [band]

    # Choose the best adjacent peak pair with a deep gutter and letter-like halves.
    parent_score = _band_letter_score(ill, band)
    best_split: tuple[float, int, int, int] | None = None
    for i in range(len(clustered) - 1):
        p0, p1 = clustered[i], clustered[i + 1]
        if p1 - p0 < min_gap:
            continue
        valley_rel = int(np.argmin(local[p0 : p1 + 1])) + p0
        valley = float(local[valley_rel])
        peak_min = min(float(local[p0]), float(local[p1]))
        if peak_min <= 1e-6:
            continue
        ratio = valley / peak_min
        top = LineBounds(y0, y0 + valley_rel, 0)
        bot = LineBounds(y0 + valley_rel, y1, 1)
        if top.height < max(10, int(0.35 * ref_height)):
            continue
        if bot.height < max(10, int(0.35 * ref_height)):
            continue
        # Reject splits that create near-empty stone strips.
        sc0 = _band_letter_score(ill, top)
        sc1 = _band_letter_score(ill, bot)
        if sc0 < 0.42 * max(parent_score, 1e-6) or sc1 < 0.42 * max(parent_score, 1e-6):
            continue
        # Deep gutter is ideal. Compact tablets can have shallower valleys
        # between rows; allow those when both halves are strongly letter-like
        # and roughly one line tall each.
        balanced = (
            0.32 * height <= top.height <= 0.68 * height
            and 0.32 * height <= bot.height <= 0.68 * height
        )
        strong_halves = (
            sc0 >= 0.72 * max(parent_score, 1e-6)
            and sc1 >= 0.72 * max(parent_score, 1e-6)
        )
        if ratio > 0.52 and not (ratio <= 0.70 and balanced and strong_halves):
            continue
        # Prefer deeper gutters and more balanced letter-like halves.
        score = (1.0 - ratio) + 0.35 * min(sc0, sc1) + 0.15 * (sc0 + sc1)
        if best_split is None or score > best_split[0]:
            best_split = (score, valley_rel, p0, p1)

    if best_split is None:
        # Very tall band (~2+ lines) with no accepted pair: try recursive
        # strongest-two-peak cut when height is extreme.
        if height < 2.15 * ref_height or len(clustered) < 2:
            return [band]
        strengths = [(float(local[p]), p) for p in clustered]
        strengths.sort(reverse=True)
        p0, p1 = sorted((strengths[0][1], strengths[1][1]))
        valley_rel = int(np.argmin(local[p0 : p1 + 1])) + p0
        top = LineBounds(y0, y0 + valley_rel, 0)
        bot = LineBounds(y0 + valley_rel, y1, 1)
        if (
            top.height >= max(10, int(0.30 * ref_height))
            and bot.height >= max(10, int(0.30 * ref_height))
            and _band_letter_score(ill, top) >= 0.35 * max(parent_score, 1e-6)
            and _band_letter_score(ill, bot) >= 0.35 * max(parent_score, 1e-6)
        ):
            return [top, bot]
        return [band]

    _, valley_rel, _p0, _p1 = best_split
    return [
        LineBounds(y0, y0 + valley_rel, 0),
        LineBounds(y0 + valley_rel, y1, 1),
    ]


def _resplit_tall_bands(
    bounds: list[LineBounds],
    s: np.ndarray,
    center_s: np.ndarray,
    ill: np.ndarray,
) -> list[LineBounds]:
    """Split under-merged bands that still contain two (or more) text rows."""
    if not bounds:
        return bounds
    heights = [b.height for b in bounds]
    med = float(np.median(heights))
    # Reference height from typical bands; ignore already-tall outliers.
    typical = [hh for hh in heights if hh <= 1.40 * med] or heights
    ref = float(np.median(typical))
    # Single collapsed band: median height is the merge itself. Only replace the
    # ref with signal pitch when many peaks still argue for under-merge (packed
    # tablet collapse failure). A lone peak with intra-glyph humps (test-25)
    # must not be re-sliced by a short false pitch.
    if len(bounds) == 1:
        seed = _find_line_peaks(s)
        if len(seed) >= 4:
            pitch = float(
                _estimate_line_pitch(s, seed if seed else [int(np.argmax(s))])
            )
            ref = min(ref, max(18.0, 0.95 * pitch))
    out: list[LineBounds] = []
    for band in bounds:
        parts = _split_one_tall_band(band, s, center_s, ill, ref)
        # One more pass for ~3-line collapses left as a single tall band.
        if len(parts) > 1 or band.height >= 2.20 * ref:
            expanded: list[LineBounds] = []
            for part in parts:
                expanded.extend(
                    _split_one_tall_band(part, s, center_s, ill, ref)
                )
            parts = expanded
        out.extend(parts)
    return [LineBounds(b.y_top, b.y_bottom, i) for i, b in enumerate(out)]


def detect_line_bands(image_bgr: np.ndarray) -> list[LineBounds]:
    """Adaptive top/bottom for every text line (1 or many)."""
    if image_bgr.ndim != 3:
        raise ValueError("Expected a BGR color image")

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ill = _illuminate(gray)
    s = _row_letter_signal(ill)

    # Wide short crops (one inscription line) — do not invent multi-line peaks.
    if _looks_like_single_line_crop(h, w):
        peak = int(np.argmax(s))
        y_top, y_bottom = _bounds_around_peak(s, peak, 0, h)
        # Prefer covering the carved ridge; expand thin bands toward image mid.
        if y_bottom - y_top < max(24, int(0.35 * h)):
            half = max(y_bottom - y_top, int(0.40 * h)) // 2
            y_top = max(0, peak - half)
            y_bottom = min(h, peak + half)
        return [LineBounds(y_top=y_top, y_bottom=y_bottom, line_index=0)]

    peaks = _find_line_peaks(s)
    peaks = _collapse_shallow_texture_peaks(s, peaks)
    n_full = len(peaks)
    # Short last lines (few centered glyphs) are weak in full-width signal
    center_s = _row_letter_signal(ill, w // 3, 2 * w // 3)
    # Weak second-row recovery only applies when a single dominant peak remains.
    if len(peaks) == 1:
        peaks = _maybe_add_weak_second_line(peaks, s, center_s)
    if len(peaks) >= 2:
        peaks = _maybe_add_weak_first_line(peaks, s, center_s)
        peaks = _maybe_add_short_last_line(peaks, center_s)
        peaks = _maybe_add_edge_last_line(peaks, s, center_s)
        peaks = _collapse_shallow_texture_peaks(s, peaks)

    if len(peaks) >= 2:
        bounds = _bands_from_valleys(s, center_s, peaks, n_full, h)
    else:
        y_top, y_bottom = _bounds_around_peak(s, peaks[0], 0, h)
        bounds = [LineBounds(y_top=y_top, y_bottom=y_bottom, line_index=0)]

    for idx, band in enumerate(bounds):
        if band.height < max(6, int(0.05 * h)):
            half = max(6, int(0.12 * h))
            peak = peaks[min(idx, len(peaks) - 1)]
            bounds[idx] = LineBounds(max(0, peak - half), min(h, peak + half), idx)

    # Cap an over-tall final band (empty stone below last glyphs)
    if len(bounds) >= 2:
        med_h = float(np.median([b.height for b in bounds[:-1]]))
        last = bounds[-1]
        if last.height > 1.25 * med_h:
            y_bottom = min(h, last.y_top + int(round(1.15 * med_h)))
            bounds[-1] = LineBounds(last.y_top, y_bottom, last.line_index)

    bounds = _merge_tiny_internal_bands(bounds, h)
    # Under-split safeguard: re-cut bands that are still ~2 text rows tall.
    bounds = _resplit_tall_bands(bounds, s, center_s, ill)
    bounds = _filter_texture_bands(bounds, ill, s)
    bounds = _expand_line_band_edges(bounds, s, h)

    return bounds


def detect_line_top_bottom(image_bgr: np.ndarray) -> LineBounds:
    """Back-compat: return the tallest / dominant line only."""
    bands = detect_line_bands(image_bgr)
    return max(bands, key=lambda b: b.height)


def _column_letter_signal(line_bgr: np.ndarray) -> np.ndarray:
    """Per-column carving activity for one already-isolated text line."""
    gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    ill = _illuminate(gray)
    f = ill.astype(np.float32)
    h, _w = gray.shape

    # Contrast along each vertical slice plus edge energy. The magnitude must
    # include the vertical derivative, or a horizontal crossbar reads as empty
    # and a cut through the middle of a letter looks free.
    k = max(9, (h // 3) | 1)
    local_mean = cv2.blur(f, (1, k))
    col_var = ((f - local_mean) ** 2).mean(axis=0)
    gx = cv2.Sobel(ill, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ill, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.sqrt(gx * gx + gy * gy).mean(axis=0)

    sig = 0.55 * _norm01(col_var) + 0.45 * _norm01(edge_energy)
    sig = cv2.GaussianBlur(sig.reshape(1, -1), (0, 0), sigmaX=2.0).ravel()
    return _norm01(sig)


def _estimate_letter_pitch(signal: np.ndarray, line_height: int) -> int:
    """Estimate average glyph-to-glyph spacing from signal autocorrelation."""
    n = len(signal)
    centered = signal - float(signal.mean())
    ac = np.correlate(centered, centered, mode="full")[n - 1 :]
    ac /= float(ac[0]) + 1e-6

    # Musnad glyph width is related to line height, but spacing remains adaptive
    # inside this broad search interval.
    lo = max(8, int(0.18 * line_height))
    hi = min(n // 3, max(lo + 1, int(0.65 * line_height)))
    if hi <= lo:
        return max(8, line_height // 3)
    return lo + int(np.argmax(ac[lo:hi]))


def detect_glyph_bounds_in_line(line_bgr: np.ndarray) -> tuple[list[GlyphBounds], np.ndarray]:
    """
    Experimental left/right segmentation for one line. Not yet accurate.

    Every line is processed independently. No x-coordinate, pitch, or cut from
    another line is reused, because letters in separate lines need not align.

    Known limitation: a column profile cannot tell an inter-letter gap from a
    gap inside a letter, and neighbouring letters joined by a horizontal stroke
    have no gap at all, so some letters are still split. Resolving this needs
    letter-shape knowledge, not a projection.
    """
    if line_bgr.ndim != 3 or line_bgr.size == 0:
        raise ValueError("Expected a non-empty BGR line crop")

    h, w = line_bgr.shape[:2]
    signal = _column_letter_signal(line_bgr)
    pitch = _estimate_letter_pitch(signal, h)

    # Outer valleys delimit the inscription from its side margins.
    edge_span = min(w // 3, max(6, pitch))
    x_left = int(np.argmin(signal[:edge_span]))
    x_right = w - edge_span + int(np.argmin(signal[w - edge_span :]))
    if x_right <= x_left + pitch:
        return [GlyphBounds(x_left, x_right, 0)], signal

    glyph_count = max(1, int(round((x_right - x_left) / max(1, pitch))))
    target = (x_right - x_left) / glyph_count

    # Dynamic programming chooses low-signal cuts while keeping widths near the
    # independently estimated pitch. This avoids treating every carved stroke
    # as a separate glyph.
    layers: list[dict[int, tuple[float, int]]] = [{x_left: (0.0, -1)}]
    radius = max(4, int(round(0.45 * target)))
    min_width = max(5, int(round(0.45 * target)))
    max_width = max(min_width + 1, int(round(1.65 * target)))

    for cut_index in range(1, glyph_count):
        expected = x_left + cut_index * target
        start = max(x_left + min_width, int(round(expected)) - radius)
        stop = min(x_right - min_width, int(round(expected)) + radius)
        layer: dict[int, tuple[float, int]] = {}
        for x in range(start, stop + 1):
            best = (float("inf"), -1)
            for prev, (prev_cost, _parent) in layers[-1].items():
                width = x - prev
                if width < min_width or width > max_width:
                    continue
                width_cost = ((width - target) / max(1.0, 0.35 * target)) ** 2
                cost = prev_cost + 2.2 * float(signal[x]) + width_cost
                if cost < best[0]:
                    best = (cost, prev)
            if best[1] >= 0:
                layer[x] = best
        if not layer:
            break
        layers.append(layer)

    if len(layers) != glyph_count:
        cuts = [int(round(x_left + i * target)) for i in range(glyph_count + 1)]
    else:
        final_candidates: list[tuple[float, int]] = []
        for prev, (cost, _parent) in layers[-1].items():
            width = x_right - prev
            if min_width <= width <= max_width:
                width_cost = ((width - target) / max(1.0, 0.35 * target)) ** 2
                final_candidates.append((cost + width_cost, prev))
        if not final_candidates:
            cuts = [int(round(x_left + i * target)) for i in range(glyph_count + 1)]
        else:
            x = min(final_candidates)[1]
            reversed_cuts = [x_right, x]
            for layer_index in range(len(layers) - 1, 1, -1):
                x = layers[layer_index][x][1]
                reversed_cuts.append(x)
            reversed_cuts.append(x_left)
            cuts = list(reversed(reversed_cuts))

    glyphs = [
        GlyphBounds(cuts[i], cuts[i + 1], i)
        for i in range(len(cuts) - 1)
        if cuts[i + 1] > cuts[i]
    ]
    return glyphs, signal


def draw_glyph_bounds(line_bgr: np.ndarray, glyphs: list[GlyphBounds]) -> np.ndarray:
    out = line_bgr.copy()
    h = out.shape[0]
    for glyph in glyphs:
        color = (0, 255, 80)
        cv2.rectangle(
            out,
            (glyph.x_left, 0),
            (max(glyph.x_left, glyph.x_right - 1), h - 1),
            color,
            1,
        )
        cv2.putText(
            out,
            str(glyph.glyph_index),
            (glyph.x_left + 2, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_bounds(bgr: np.ndarray, bands: list[LineBounds] | LineBounds) -> np.ndarray:
    if isinstance(bands, LineBounds):
        bands = [bands]
    out = bgr.copy()
    h, w = out.shape[:2]
    colors = [(0, 255, 80), (0, 200, 255), (255, 180, 0), (255, 80, 180)]

    for band in bands:
        color = colors[band.line_index % len(colors)]
        y0 = int(np.clip(band.y_top, 0, h - 1))
        y1 = int(np.clip(band.y_bottom - 1, 0, h - 1))

        tint = out.copy()
        cv2.rectangle(tint, (0, y0), (w - 1, y1), color, -1)
        out = cv2.addWeighted(tint, 0.14, out, 0.86, 0)
        cv2.line(out, (0, y0), (w - 1, y0), color, 2)
        cv2.line(out, (0, y1), (w - 1, y1), color, 2)
        cv2.putText(
            out,
            f"L{band.line_index}: top={band.y_top} bottom={band.y_bottom}",
            (8, max(18, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def run_on_image(image_path: Path, out_dir: Path) -> dict:
    bgr = load_bgr(image_path)
    bands = detect_line_bands(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    signal = _row_letter_signal(_illuminate(gray))

    stem_dir = out_dir / image_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale line outputs from previous runs (count can change)
    for pattern in ("02_line*_crop.jpg", "02_line*_enhanced.jpg"):
        for old in stem_dir.glob(pattern):
            old.unlink(missing_ok=True)

    save_image(stem_dir / "00_input.jpg", bgr)
    save_image(stem_dir / "01_line_bounds.jpg", draw_bounds(bgr, bands))

    # Imported here: the gallery module imports this one for line detection.
    from .stone_enhancement_gallery import enhance_line_for_ocr

    for band in bands:
        crop = bgr[band.y_top : band.y_bottom, :]
        if not crop.size:
            continue
        save_image(stem_dir / f"02_line{band.line_index:02d}_crop.jpg", crop)
        enhanced = enhance_line_for_ocr(crop)
        save_image(
            stem_dir / f"02_line{band.line_index:02d}_enhanced.jpg",
            cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        )

    panel = np.zeros((bgr.shape[0], 100, 3), dtype=np.uint8)
    for y, v in enumerate(signal):
        cv2.line(panel, (0, y), (int(v * 92), y), (0, 200, 255), 1)
    for band in bands:
        cv2.line(panel, (0, band.y_top), (99, band.y_top), (0, 255, 80), 2)
        cv2.line(panel, (0, band.y_bottom - 1), (99, band.y_bottom - 1), (0, 255, 80), 2)
    save_image(stem_dir / "03_row_signal.jpg", np.hstack([draw_bounds(bgr, bands), panel]))

    result = {
        "image": str(image_path),
        "line_count": len(bands),
        "lines": [b.to_dict() for b in bands],
    }
    (stem_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Detect text-line top/bottom Y on stone")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUTS_DIR / "stone_glyph_segmentation",
    )
    args = p.parse_args()
    result = run_on_image(args.image, args.out_dir)
    print(f"{args.image.name}: lines={result['line_count']}")
    for line in result["lines"]:
        print(
            f"  L{line['line_index']}: top={line['y_top']} "
            f"bottom={line['y_bottom']} height={line['height']}"
        )


if __name__ == "__main__":
    main()
