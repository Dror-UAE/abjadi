"""Post-slice empty-segment rejection.

This module deliberately runs *after* letter slicing. It does not move cuts,
does not merge letters, and does not assume any fixed number of letters. Its
only job is to remove candidate crops that are just bare stone / margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SegmentCandidate:
    index: int
    x_left: int
    x_right: int
    image: np.ndarray
    is_empty: bool = False
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return max(0, self.x_right - self.x_left)


def segments_from_cuts(crop_bgr: np.ndarray, cuts: list[int]) -> list[SegmentCandidate]:
    """Convert boundary cuts into candidate crops without changing cut positions."""
    width = crop_bgr.shape[1]
    edges = [0] + sorted(c for c in cuts if 0 < c < width) + [width]
    segments: list[SegmentCandidate] = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        if right <= left:
            continue
        segments.append(
            SegmentCandidate(
                index=index,
                x_left=left,
                x_right=right,
                image=crop_bgr[:, left:right],
            )
        )
    return segments


def _segment_scores(
    segment: SegmentCandidate,
    line_activity: np.ndarray | None,
    objectness: np.ndarray | None,
) -> dict[str, float]:
    """Content scores; all are independent of the number of letters."""
    gray = cv2.cvtColor(segment.image, cv2.COLOR_BGR2GRAY)
    f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    col_energy = mag.mean(axis=0)
    row_energy = mag.mean(axis=1)
    edge_peak = float(np.percentile(mag, 92))
    column_coherence = float(col_energy.max() / (col_energy.mean() + 1e-6))
    row_coherence = float(row_energy.max() / (row_energy.mean() + 1e-6))

    activity_mean = 1.0
    activity_peak = 1.0
    if line_activity is not None:
        sl = line_activity[segment.x_left : segment.x_right]
        if sl.size:
            activity_mean = float(sl.mean())
            activity_peak = float(sl.max())

    object_mean = 1.0
    object_peak = 1.0
    if objectness is not None:
        sl = objectness[segment.x_left : segment.x_right]
        if sl.size:
            object_mean = float(sl.mean())
            object_peak = float(sl.max())

    return {
        "edge_peak": edge_peak,
        "column_coherence": column_coherence,
        "row_coherence": row_coherence,
        "activity_mean": activity_mean,
        "activity_peak": activity_peak,
        "object_mean": object_mean,
        "object_peak": object_peak,
        "width": float(segment.width),
    }


def mark_empty_segments(
    segments: list[SegmentCandidate],
    *,
    line_activity: np.ndarray | None = None,
    objectness: np.ndarray | None = None,
    min_activity_mean: float = 0.12,
    min_activity_peak: float = 0.40,
    min_edge_peak: float = 0.90,
    min_object_mean: float = 0.40,
    min_object_peak: float = 0.55,
) -> list[SegmentCandidate]:
    """
    Mark candidates as empty using content scores only.

    A segment is rejected only when it looks like obvious bare stone. Faint real
    letters can have low mean activity, especially round/weathered glyphs, so a
    weak segment is kept if it still has a strong edge peak.
    """
    for position, segment in enumerate(segments):
        scores = _segment_scores(segment, line_activity, objectness)
        segment.scores = scores

        # Round glyphs (𐩰 / 𐩲) can have near-zero column activity while the
        # boundary objectness map still lights them strongly — do not drop those.
        almost_no_activity = (
            scores["activity_mean"] < 0.06
            and scores["activity_peak"] < 0.25
            and scores["object_mean"] < 0.55
            and scores["object_peak"] < 0.70
        )
        # Round/weathered glyphs (esp. small 𐩲) often have low column activity
        # but strong boundary objectness — do not treat those as bare stone.
        weak_activity = (
            scores["activity_mean"] < min_activity_mean
            and scores["activity_peak"] < min_activity_peak
            and scores["edge_peak"] < min_edge_peak
            and scores["object_mean"] < 0.55
            and scores["object_peak"] < 0.70
        )
        weak_object = (
            scores["object_mean"] < min_object_mean
            and scores["object_peak"] < min_object_peak
        )
        very_flat = (
            scores["edge_peak"] < 0.075
            and scores["column_coherence"] < 2.2
            and scores["row_coherence"] < 2.2
        )
        # First/last crops are often bare stone, but faint leading/trailing
        # circles (𐩲) also sit on the margin with weak edges — keep them when
        # the boundary objectness map still fires strongly.
        edge_margin_empty = (
            position in {0, len(segments) - 1}
            and scores["activity_mean"] < 0.28
            and scores["edge_peak"] < 1.15
            and scores["column_coherence"] < 1.70
            and scores["object_mean"] < 0.55
            and scores["object_peak"] < 0.70
        )
        h = int(segment.image.shape[0]) if segment.image.size else 0
        carved_thin_stem = (
            h > 0
            and 2 <= segment.width <= max(4, int(0.28 * h))
            and scores["activity_peak"] >= 0.40
            and scores["activity_mean"] >= 0.12
        )
        segment.is_empty = (not carved_thin_stem) and (
            almost_no_activity
            or weak_activity
            or edge_margin_empty
            or (very_flat and weak_object)
        )
    return segments


def kept_segments(segments: list[SegmentCandidate]) -> list[SegmentCandidate]:
    return [segment for segment in segments if not segment.is_empty]


def render_segment_sheet(
    segments: list[SegmentCandidate],
    *,
    per_row: int = 10,
    show_rejected: bool = False,
) -> np.ndarray:
    visible = segments if show_rejected else kept_segments(segments)
    if not visible:
        return np.zeros((80, 260, 3), dtype=np.uint8)

    cell_w = max(segment.image.shape[1] for segment in visible) + 8
    cell_h = max(segment.image.shape[0] for segment in visible) + 22
    rows = (len(visible) + per_row - 1) // per_row
    sheet = np.full((rows * cell_h, per_row * cell_w, 3), 30, np.uint8)
    for index, segment in enumerate(visible):
        r, c = divmod(index, per_row)
        y, x = r * cell_h, c * cell_w
        img = segment.image.copy()
        label_color = (0, 255, 80)
        if segment.is_empty:
            label_color = (60, 60, 255)
            overlay = img.copy()
            overlay[:] = (35, 35, 120)
            img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
        sheet[y + 20 : y + 20 + img.shape[0], x + 4 : x + 4 + img.shape[1]] = img
        label = segment.index if show_rejected else index + 1
        cv2.putText(
            sheet,
            str(label),
            (x + 4, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            label_color,
            1,
            cv2.LINE_AA,
        )
    return sheet
