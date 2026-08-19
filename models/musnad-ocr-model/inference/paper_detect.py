"""
Paper / clean-manuscript glyph detection for Musnad OCR v0.2.

Optimized for dark ink on light paper (controlled lighting). Word separators
are often colored (brown) bars — detected separately from black letter ink.
Stone carvings are intentionally out of scope here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .layout import GlyphBox, cluster_lines, order_line_rtl


def _to_rgb_u8(image: Image.Image) -> np.ndarray:
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        return np.array(bg)
    return np.array(image.convert("RGB"))


def _to_gray_u8(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(_to_rgb_u8(image), cv2.COLOR_RGB2GRAY)


def _ink_mask_paper(gray: np.ndarray) -> np.ndarray:
    """Binary ink=255 for dark letter strokes on paper."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    frac_a = float((binary > 0).mean())
    frac_o = float((otsu > 0).mean())
    if 0.01 <= frac_o <= 0.35 and abs(frac_o - 0.12) < abs(frac_a - 0.12):
        binary = otsu

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def _merge_overlaps(boxes: List[GlyphBox], iou_thresh: float = 0.15) -> List[GlyphBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
    kept: List[GlyphBox] = []

    def iou(a: GlyphBox, b: GlyphBox) -> float:
        x0 = max(a.x0, b.x0)
        y0 = max(a.y0, b.y0)
        x1 = min(a.x1, b.x1)
        y1 = min(a.y1, b.y1)
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        if inter <= 0:
            return 0.0
        return inter / max(a.area + b.area - inter, 1)

    def contained(inner: GlyphBox, outer: GlyphBox) -> bool:
        return (
            inner.x0 >= outer.x0
            and inner.y0 >= outer.y0
            and inner.x1 <= outer.x1
            and inner.y1 <= outer.y1
        )

    for box in boxes:
        drop = False
        for i, k in enumerate(kept):
            if contained(box, k) or iou(box, k) >= iou_thresh:
                kept[i] = GlyphBox(
                    min(k.x0, box.x0),
                    min(k.y0, box.y0),
                    max(k.x1, box.x1),
                    max(k.y1, box.y1),
                )
                drop = True
                break
            if contained(k, box):
                kept[i] = box
                drop = True
                break
        if not drop:
            kept.append(box)
    return kept


def detect_word_bars_paper(image: Image.Image) -> List[GlyphBox]:
    """
    Detect word-separator bars.

    On clean paper manuscripts separators are often colored (brown/copper).
    We do **not** use black thin-stroke morphology here — that confuses letter
    stems (alef-like) with ``|``. Black ``|`` bars are recovered via gaps.
    """
    rgb = _to_rgb_u8(image)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # Brown / copper separators: warmer than paper, not near-black ink.
    chromatic = (r > g + 8) & (r > b + 15) & (np.abs(r - g) > 10)
    mid = (gray > 70) & (gray < 210)
    color_mask = (chromatic & mid).astype(np.uint8) * 255

    color_mask = cv2.morphologyEx(
        color_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    )
    color_mask = cv2.morphologyEx(
        color_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars: List[GlyphBox] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < max(10, int(0.18 * h)):
            continue
        aspect = bw / max(bh, 1)
        if aspect > 0.55:
            continue
        if bw > max(24, int(0.06 * w)):
            continue
        bars.append(GlyphBox(x, y, x + bw, y + bh))
    return _merge_overlaps(bars, iou_thresh=0.2)


def _word_gap_threshold(ordered: Sequence[GlyphBox]) -> float:
    """
    Minimum horizontal gap (px) that counts as a word break.

    Uses **inter-letter gap** statistics, not glyph width — letter spacing on
    clean paper is ~4–8 px while word gaps are ~18–24 px.
    """
    if len(ordered) < 2:
        return 14.0
    gaps = [float(ordered[i].x0 - ordered[i + 1].x1) for i in range(len(ordered) - 1)]
    gaps.sort()
    med = gaps[len(gaps) // 2]
    mx = gaps[-1]
    # After upscale: letter gaps ~10–12 px, word gaps ~22–26 px on this font.
    if mx <= 18:
        return max(14.0, med * 3.2)
    return max(18.0, med * 2.05)


def inject_gap_separators(
    ordered: List[GlyphBox],
    *,
    gap_ratio: float = 1.15,
) -> List[Tuple[GlyphBox, bool]]:
    """Insert synthetic separators in large gaps (RTL-ordered)."""
    if len(ordered) < 2:
        return [(b, False) for b in ordered]
    gap_min = _word_gap_threshold(ordered)
    # Legacy callers passing gap_ratio > 1 treat it as a multiplier on med gap.
    if gap_ratio > 1.5:
        gaps = sorted(
            float(ordered[i].x0 - ordered[i + 1].x1) for i in range(len(ordered) - 1)
        )
        med = gaps[len(gaps) // 2]
        gap_min = max(gap_min, med * gap_ratio)

    out: List[Tuple[GlyphBox, bool]] = []
    for i, box in enumerate(ordered):
        out.append((box, False))
        if i + 1 >= len(ordered):
            break
        nxt = ordered[i + 1]
        gap = box.x0 - nxt.x1
        if gap >= gap_min:
            cx0 = nxt.x1 + max(1, gap // 2) - 1
            cx1 = cx0 + 2
            y0 = min(box.y0, nxt.y0)
            y1 = max(box.y1, nxt.y1)
            out.append((GlyphBox(cx0, y0, cx1, y1), True))
    return out


def _split_wide_boxes(
    boxes: List[GlyphBox],
    mask: np.ndarray,
    *,
    split_ratio: float = 1.55,
) -> List[GlyphBox]:
    """Split boxes that clearly contain two adjacent letters."""
    if len(boxes) < 2:
        return boxes
    widths = sorted(b.width for b in boxes)
    med_w = widths[len(widths) // 2]
    out: List[GlyphBox] = []
    h_mask, w_mask = mask.shape[:2]

    for box in boxes:
        if box.width <= med_w * split_ratio:
            out.append(box)
            continue
        x0 = max(0, box.x0)
        y0 = max(0, box.y0)
        x1 = min(w_mask, box.x1)
        y1 = min(h_mask, box.y1)
        strip = mask[y0:y1, x0:x1]
        if strip.size == 0:
            out.append(box)
            continue
        col = strip.sum(axis=0).astype(np.float32)
        if col.max() <= 0:
            out.append(box)
            continue
        # Search for ink valley in the interior (not at edges).
        margin = max(2, int(0.15 * (x1 - x0)))
        interior = col[margin : len(col) - margin]
        if interior.size < 4:
            out.append(box)
            continue
        split_at = int(interior.argmin()) + margin
        left_w = split_at
        right_w = (x1 - x0) - split_at
        if left_w < 6 or right_w < 6:
            out.append(box)
            continue
        out.append(GlyphBox(x0, y0, x0 + split_at, y1))
        out.append(GlyphBox(x0 + split_at, y0, x1, y1))
    return out


def _upscale_for_detect(image: Image.Image, *, min_width: int = 1400) -> Tuple[Image.Image, float]:
    """Upscale page images so JPEG glyphs separate cleanly."""
    w, h = image.size
    # Single-line strips: keep native resolution (upscale creates false line bands).
    if h < 200:
        return image, 1.0
    if w >= min_width:
        return image, 1.0
    scale = min_width / w
    nw, nh = int(w * scale), int(h * scale)
    up = image.resize((nw, nh), Image.Resampling.LANCZOS)
    return up, scale


def _scale_boxes(boxes: Sequence[GlyphBox], inv_scale: float) -> List[GlyphBox]:
    if inv_scale == 1.0:
        return list(boxes)
    out: List[GlyphBox] = []
    for b in boxes:
        out.append(
            GlyphBox(
                int(round(b.x0 * inv_scale)),
                int(round(b.y0 * inv_scale)),
                int(round(b.x1 * inv_scale)),
                int(round(b.y1 * inv_scale)),
            )
        )
    return out


def _line_bands_from_mask(mask: np.ndarray, *, min_rows: int = 6) -> List[Tuple[int, int]]:
    """Row bands that contain ink (one per text line)."""
    row = (mask > 0).sum(axis=1)
    bands: List[Tuple[int, int]] = []
    in_band = False
    y0 = 0
    for y, v in enumerate(row):
        if v > 0 and not in_band:
            y0 = y
            in_band = True
        elif v <= 0 and in_band:
            if y - y0 >= min_rows:
                bands.append((y0, y))
            in_band = False
    if in_band and mask.shape[0] - y0 >= min_rows:
        bands.append((y0, mask.shape[0]))
    return bands


def _segment_line_projection(
    mask: np.ndarray,
    y0: int,
    y1: int,
    *,
    img_w: int,
) -> List[GlyphBox]:
    """Segment one text line via vertical ink projection (clean paper / digital font)."""
    strip = mask[y0:y1, :]
    if strip.size == 0:
        return []
    col = strip.sum(axis=0).astype(np.float32)
    active = col > 0
    if not active.any():
        return []

    letters: List[GlyphBox] = []
    x = 0
    n = len(col)

    while x < n:
        if not active[x]:
            x += 1
            continue
        x_start = x
        while x < n and active[x]:
            x += 1
        x_end = x
        bw = x_end - x_start
        if bw < 3:
            continue
        seg = strip[:, x_start:x_end]
        rows = np.where(seg.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        sy0 = int(rows.min())
        sy1 = int(rows.max()) + 1
        letters.append(GlyphBox(x_start, y0 + sy0, x_end, y0 + sy1))

    if not letters:
        return letters

    widths = sorted(b.width for b in letters)
    med_w = widths[len(widths) // 2]
    split: List[GlyphBox] = []
    for box in letters:
        if box.width <= med_w * 1.38:
            split.append(box)
            continue
        seg = mask[box.y0 : box.y1, box.x0 : box.x1]
        if seg.size == 0:
            split.append(box)
            continue
        csum = seg.sum(axis=0).astype(np.float32)
        margin = max(2, int(0.10 * box.width))
        interior = csum[margin : len(csum) - margin]
        if interior.size < 4:
            split.append(box)
            continue
        cut = int(interior.argmin()) + margin
        if cut < 5 or (box.width - cut) < 5 or interior.min() > 0:
            split.append(box)
            continue
        split.append(GlyphBox(box.x0, box.y0, box.x0 + cut, box.y1))
        split.append(GlyphBox(box.x0 + cut, box.y0, box.x1, box.y1))
    return split


def _split_bar_segments(letters: List[GlyphBox]) -> Tuple[List[GlyphBox], List[GlyphBox]]:
    """Pull thin ``|`` ink runs out of the letter list (clean paper)."""
    if not letters:
        return [], []
    ordered = sorted(letters, key=lambda b: b.x0)
    keep: List[GlyphBox] = []
    bars: List[GlyphBox] = []
    for i, box in enumerate(ordered):
        aspect = box.width / max(box.height, 1)
        if box.width > 12 or aspect > 0.42:
            keep.append(box)
            continue
        left_gap = (box.x0 - ordered[i - 1].x1) if i > 0 else 999.0
        right_gap = (ordered[i + 1].x0 - box.x1) if i + 1 < len(ordered) else 999.0
        if left_gap >= 14 and right_gap >= 14:
            bars.append(box)
        else:
            keep.append(box)
    return keep, bars


def detect_glyphs_paper_projection(
    image: Image.Image,
) -> Tuple[List[GlyphBox], List[GlyphBox], np.ndarray, float]:
    """Projection-based glyph + bar detection for clean paper pages."""
    scaled, scale = _upscale_for_detect(image)
    inv = 1.0 / scale
    gray = _to_gray_u8(scaled)
    mask = _ink_mask_paper(gray)

    all_letters: List[GlyphBox] = []
    all_bars: List[GlyphBox] = []
    for y0, y1 in _line_bands_from_mask(mask):
        letters = _segment_line_projection(mask, y0, y1, img_w=mask.shape[1])
        letters, bars = _split_bar_segments(letters)
        all_letters.extend(letters)
        all_bars.extend(bars)

    all_letters = _scale_boxes(all_letters, inv)
    all_bars = _scale_boxes(all_bars, inv)
    orig_mask = _ink_mask_paper(_to_gray_u8(image)) if scale != 1.0 else mask
    return all_letters, all_bars, orig_mask, scale


def detect_glyphs_paper(
    image: Image.Image,
    *,
    min_area_frac: float = 0.00015,
    max_area_frac: float = 0.12,
    min_side: int = 8,
    pad: int = 2,
    use_projection: bool = True,
) -> Tuple[List[GlyphBox], np.ndarray]:
    """Detect black letter glyphs (excludes thin word bars)."""
    if use_projection:
        letters, _bars, mask, _scale = detect_glyphs_paper_projection(image)
        w, h = image.size
        if pad:
            letters = [b.pad(pad, w, h) for b in letters]
        return letters, mask

    gray = _to_gray_u8(image)
    h, w = gray.shape
    mask = _ink_mask_paper(gray)
    img_area = float(h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[GlyphBox] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area_frac * img_area or area > max_area_frac * img_area:
            continue
        if bw < min_side or bh < min_side:
            continue
        aspect = bw / max(bh, 1)
        if aspect > 8.0:
            continue
        # Tall thin bars → word separators (other detector).
        if aspect <= 0.28 and bh >= 1.6 * bw:
            continue
        if aspect < 0.08:
            continue
        boxes.append(GlyphBox(x, y, x + bw, y + bh))

    boxes = _merge_overlaps(boxes)
    boxes = _split_wide_boxes(boxes, mask)
    if pad:
        boxes = [b.pad(pad, w, h) for b in boxes]
    return boxes, mask


def detect_paper_layout(
    image: Image.Image,
) -> Tuple[List[List[Tuple[GlyphBox, bool]]], np.ndarray]:
    """
    Lines of (box, is_separator), each line in Musnad logical order (right → left).

    Unicode text must be stored in reading order. The UI's ``dir=rtl`` then
    positions that logical sequence correctly instead of reversing visual-order
    OCR output a second time.
    """
    letters, bars_proj, mask, _scale = detect_glyphs_paper_projection(image)
    w, h = image.size
    # Narrow single-line uploads: contour CV is more reliable than page projection.
    if h < 200:
        letters, mask = detect_glyphs_paper(image, use_projection=False)
        bars_proj = []
    else:
        letters = [b.pad(2, w, h) for b in letters]
        bars_proj = [b.pad(1, w, h) for b in bars_proj]
    if h < 200:
        letters = [b.pad(2, w, h) for b in letters]
    lines = cluster_lines(letters)
    layout: List[List[Tuple[GlyphBox, bool]]] = []
    color_bars = detect_word_bars_paper(image)

    for line in lines:
        ordered = order_line_rtl(line)
        line_y0 = min(b.y0 for b in ordered) - 4
        line_y1 = max(b.y1 for b in ordered) + 4
        bars: List[GlyphBox] = []
        for bar in color_bars:
            if any(bar.x0 >= L.x0 - 2 and bar.x1 <= L.x1 + 2 for L in ordered):
                continue
            left = [L for L in ordered if L.x1 <= bar.x0]
            right = [L for L in ordered if L.x0 >= bar.x1]
            if left and right and line_y0 <= bar.cy <= line_y1:
                bars.append(bar)

        if bars:
            combined = ordered + bars
            ordered_all = order_line_rtl(combined)

            def is_bar(box: GlyphBox) -> bool:
                return any(
                    abs(box.cx - bar.cx) <= 6 and abs(box.width - bar.width) <= 10
                    for bar in bars
                )

            tagged = [(b, is_bar(b)) for b in ordered_all]
        else:
            tagged = inject_gap_separators(ordered)
        layout.append(tagged)
    return layout, mask


def _inject_gap_separators_ltr(
    ordered: List[GlyphBox],
) -> List[Tuple[GlyphBox, bool]]:
    """Insert separators after large gaps (line ordered left → right)."""
    if len(ordered) < 2:
        return [(b, False) for b in ordered]
    gap_min = _word_gap_threshold_ltr(ordered)
    out: List[Tuple[GlyphBox, bool]] = []
    for i, box in enumerate(ordered):
        out.append((box, False))
        if i + 1 >= len(ordered):
            break
        nxt = ordered[i + 1]
        gap = nxt.x0 - box.x1
        if gap >= gap_min:
            cx0 = box.x1 + max(1, gap // 2) - 1
            cx1 = cx0 + 2
            y0 = min(box.y0, nxt.y0)
            y1 = max(box.y1, nxt.y1)
            out.append((GlyphBox(cx0, y0, cx1, y1), True))
    return out


def _word_gap_threshold_ltr(ordered: Sequence[GlyphBox]) -> float:
    if len(ordered) < 2:
        return 14.0
    gaps = [float(ordered[i + 1].x0 - ordered[i].x1) for i in range(len(ordered) - 1)]
    gaps.sort()
    med = gaps[len(gaps) // 2]
    mx = gaps[-1]
    if mx <= 18:
        return max(14.0, med * 3.2)
    return max(18.0, med * 2.05)


def crop_glyph(
    image: Image.Image,
    box: GlyphBox,
    *,
    pad: int = 6,
    neighbors: Optional[List[GlyphBox]] = None,
) -> Image.Image:
    """Crop one glyph with padding that cannot include neighboring glyph ink."""
    w, h = image.size
    b = box.pad(pad, w, h)

    if neighbors:
        others = [n for n in neighbors if n is not box and n.as_tuple() != box.as_tuple()]
        left = [n for n in others if n.cx < box.cx]
        right = [n for n in others if n.cx > box.cx]

        if left:
            nearest_left = max(left, key=lambda n: n.cx)
            # Split available whitespace halfway between adjacent ink boxes.
            left_boundary = (nearest_left.x1 + box.x0) // 2
            b = GlyphBox(max(b.x0, left_boundary), b.y0, b.x1, b.y1)
        if right:
            nearest_right = min(right, key=lambda n: n.cx)
            right_boundary = (box.x1 + nearest_right.x0 + 1) // 2
            b = GlyphBox(b.x0, b.y0, min(b.x1, right_boundary), b.y1)

    return image.crop((b.x0, b.y0, b.x1, b.y1))


def draw_boxes(
    image: Image.Image,
    boxes: List[GlyphBox],
    *,
    color: Tuple[int, int, int] = (180, 90, 40),
    labels: Optional[List[str]] = None,
) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    for i, box in enumerate(boxes):
        cv2.rectangle(rgb, (box.x0, box.y0), (box.x1, box.y1), color, 2)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(
                rgb,
                labels[i][:12],
                (box.x0, max(12, box.y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return Image.fromarray(rgb)


def _load_annotation_font(size: int = 14):
    from PIL import ImageFont

    candidates = [
        "seguiemj.ttf",
        "seguisym.ttf",
        "arialuni.ttf",
        "NotoSansOldSouthArabian-Regular.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_annotations(
    image: Image.Image,
    glyphs: List[dict],
    *,
    letter_color: Tuple[int, int, int] = (36, 120, 80),
    separator_color: Tuple[int, int, int] = (180, 90, 40),
    unknown_color: Tuple[int, int, int] = (170, 50, 50),
) -> Image.Image:
    """Draw glyph boxes + identify labels on a copy of the full image."""
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _load_annotation_font(14)
    font_small = _load_annotation_font(12)

    for g in glyphs:
        box = g.get("box")
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        is_sep = bool(g.get("is_separator"))
        ch = g.get("display") or g.get("character") or "?"
        name = g.get("name")
        conf = g.get("confidence")
        trusted = g.get("trusted")

        if is_sep:
            color = separator_color
            label = "|"
            if conf is not None:
                label = f"| {float(conf) * 100:.0f}%"
        elif ch in {None, "?", "UNKNOWN"} or trusted is False:
            color = unknown_color
            label = "?"
            if conf is not None:
                label = f"? {float(conf) * 100:.0f}%"
        else:
            color = letter_color
            if name and name not in {"WORD_SEPARATOR", "UNKNOWN"}:
                label = str(name)
            else:
                label = str(ch)
            if conf is not None:
                label = f"{label} {float(conf) * 100:.0f}%"

        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        text = label[:28]
        try:
            tb = draw.textbbox((0, 0), text, font=font_small)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(text) * 7, 12
        pad = 2
        lx = x0
        ly = y0 - th - 2 * pad - 2
        if ly < 0:
            ly = y0 + 2
        draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=color)
        draw.text((lx + pad, ly + pad), text, fill=(255, 255, 255), font=font_small)

        if not is_sep and ch and ch not in {"?", "UNKNOWN"} and not str(ch).startswith("NUM_"):
            try:
                draw.text((x0 + 3, y0 + 2), str(ch), fill=color, font=font)
            except Exception:
                pass

    return canvas
