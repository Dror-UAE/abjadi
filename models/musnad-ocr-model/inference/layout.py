"""
Musnad / Old South Arabian writing rules for layout OCR.

Paper-first v0.3.2 assumes clean manuscript rules:

  - Default reading direction: right-to-left (RTL)
  - Letters are separate (not cursive)
  - Words are split by a vertical bar, not whitespace
  - The bar glyph is visually the same as number ONE (U+10A7D)
  - Stone / boustrophedon are future domains
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# Unicode: Old South Arabian NUMBER ONE doubles as word separator in the standard.
WORD_SEPARATOR_CHAR = "𐩽"  # U+10A7D
WORD_SEPARATOR_DISPLAY = "|"
WORD_SEPARATOR_LABEL = "NUM_1"  # classifier label used in this project

DEFAULT_DIRECTION = "rtl"


@dataclass(frozen=True)
class GlyphBox:
    """Axis-aligned glyph box in image coordinates (x0,y0,x1,y1)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    @property
    def width(self) -> int:
        return max(1, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(1, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    def pad(self, px: int, img_w: int, img_h: int) -> "GlyphBox":
        return GlyphBox(
            max(0, self.x0 - px),
            max(0, self.y0 - px),
            min(img_w, self.x1 + px),
            min(img_h, self.y1 + px),
        )


def is_word_separator_label(label: Optional[str]) -> bool:
    """True if classifier output is the vertical-bar / number-one class."""
    if not label:
        return False
    s = str(label).strip()
    if s in {WORD_SEPARATOR_LABEL, WORD_SEPARATOR_CHAR, "1", "NUM_1", "|"}:
        return True
    # Never treat numeral class tags as readable letters in line text.
    if s.upper().startswith("NUM_"):
        # NUM_1 is the separator; other NUM_* are not separators by label alone.
        return s.upper() in {"NUM_1", "NUM1"}
    return False


def is_readable_letter(label: Optional[str]) -> bool:
    """False for separators, numeral tags, and unknowns."""
    if not label or label in {"?", "UNKNOWN"}:
        return False
    if is_word_separator_label(label):
        return False
    if str(label).upper().startswith("NUM_"):
        return False
    return True


def cluster_lines(
    boxes: Sequence[GlyphBox],
    *,
    y_tol_ratio: float = 0.55,
) -> List[List[GlyphBox]]:
    """
    Group glyph boxes into horizontal lines (top → bottom).

    ``y_tol_ratio`` is relative to median glyph height.
    """
    if not boxes:
        return []
    heights = sorted(b.height for b in boxes)
    med_h = heights[len(heights) // 2]
    tol = max(8.0, med_h * y_tol_ratio)

    ordered = sorted(boxes, key=lambda b: b.cy)
    lines: List[List[GlyphBox]] = []
    line_ys: List[float] = []
    for box in ordered:
        placed = False
        for i, y in enumerate(line_ys):
            if abs(box.cy - y) <= tol:
                lines[i].append(box)
                # Update running mean y
                n = len(lines[i])
                line_ys[i] = (y * (n - 1) + box.cy) / n
                placed = True
                break
        if not placed:
            lines.append([box])
            line_ys.append(box.cy)

    # Top-to-bottom by mean y
    order = sorted(range(len(lines)), key=lambda i: line_ys[i])
    return [lines[i] for i in order]


def order_line_rtl(boxes: Sequence[GlyphBox]) -> List[GlyphBox]:
    """Sort a line right → left (default Musnad reading order)."""
    return sorted(boxes, key=lambda b: -b.cx)


def order_line_ltr(boxes: Sequence[GlyphBox]) -> List[GlyphBox]:
    """Sort a line left → right (rare / boustrophedon even lines)."""
    return sorted(boxes, key=lambda b: b.cx)


def split_words_by_separator(
    glyphs: Sequence[dict],
    *,
    separator_key: str = "character",
) -> List[List[dict]]:
    """
    Split an already-ordered glyph list into words using vertical-bar labels.

    Separators are omitted from word contents but mark boundaries.
    """
    words: List[List[dict]] = []
    current: List[dict] = []
    for g in glyphs:
        if g.get("is_separator") or is_word_separator_label(g.get(separator_key)):
            if current:
                words.append(current)
                current = []
            continue
        current.append(g)
    if current:
        words.append(current)
    return words


def join_word_text(glyphs: Iterable[dict], *, key: str = "character") -> str:
    """Concatenate glyph characters (skip separators / numeral tags; keep ``?``)."""
    chars: List[str] = []
    for g in glyphs:
        if g.get("is_separator"):
            continue
        ch = g.get(key)
        if not ch:
            continue
        if is_word_separator_label(ch) or str(ch).upper().startswith("NUM_"):
            continue
        # Keep unknowns as ``?`` so dropped letters are visible in the reading.
        if ch in {"?", "UNKNOWN"}:
            chars.append("?")
            continue
        if not is_readable_letter(ch):
            continue
        chars.append(str(ch))
    return "".join(chars)


def format_line_text(
    glyphs: Sequence[dict],
    *,
    key: str = "character",
    word_sep: str = WORD_SEPARATOR_DISPLAY,
) -> str:
    """
    Build display text for one RTL-ordered line with word separators.

    Returns a string that still contains characters in visual RTL storage order
    (first char = rightmost glyph). UI should set ``dir=rtl``.
    """
    words = split_words_by_separator(glyphs, separator_key=key)
    parts = [join_word_text(w, key=key) for w in words]
    parts = [p for p in parts if p]
    return word_sep.join(parts)
