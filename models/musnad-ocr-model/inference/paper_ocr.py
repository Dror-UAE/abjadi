"""
Musnad paper / manuscript line OCR (v0.3.2).

Pipeline (clean paper domain):
  1. Detect discrete glyphs on light paper (projection + gap word breaks)
  2. Cluster into lines (top → bottom)
  3. Order each line right → left (RTL logical storage order)
  4. Classify each crop with paper-font fine-tuned musnad_final
  5. Split words on vertical-bar / NUM_1 separators

No server / API — call recognize_paper() or MusnadOCR from your own backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from .layout import (
    WORD_SEPARATOR_DISPLAY,
    WORD_SEPARATOR_LABEL,
    format_line_text,
    is_word_separator_label,
    join_word_text,
    split_words_by_separator,
)
from .paper_detect import crop_glyph, detect_paper_layout, draw_annotations
from .predict import MusnadPredictor, load_external_image

# Unicode + Latin names for paper lookalike shape fixes
_CHARS = {
    "BETH": ("𐩨", "BETH"),
    "GIMEL": ("𐩴", "GIMEL"),
    "WAW": ("𐩥", "WAW"),
    "AYN": ("𐩲", "AYN"),
    "TETH": ("𐩷", "TETH"),
    "DHADHE": ("𐩳", "DHADHE"),
}


def _paper_ink_patch(crop: Image.Image) -> Optional[np.ndarray]:
    """Largest connected ink component, tightly cropped."""
    gray = np.array(crop.convert("L"))
    mask = (gray < 200).astype(np.uint8)
    if not mask.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    ys, xs = np.where(mask)
    patch = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if patch.shape[0] < 8 or patch.shape[1] < 6:
        return None
    return patch


def _paper_shape_metrics(patch: np.ndarray) -> dict:
    """Geometry scores used by paper lookalike fixes."""
    h, w = patch.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 3, 2 * w // 3
    center = patch[y0:y1, x0:x1]
    half = max(1, w // 12)
    mid_col = patch[:, max(0, w // 2 - half) : min(w, w // 2 + half + 1)]
    half_h = max(1, h // 12)
    mid_row = patch[max(0, h // 2 - half_h) : min(h, h // 2 + half_h + 1), :]
    band = max(2, h // 8)
    return {
        "h": h,
        "w": w,
        "hole": 1.0 - float(center.mean()) if center.size else 0.0,
        "vert": float(mid_col.mean()) if mid_col.size else 0.0,
        "horiz": float(mid_row.mean()) if mid_row.size else 0.0,
        "top": float(patch[:band, :].mean()),
        "bot": float(patch[-band:, :].mean()),
    }


def _paper_beth_gimel_shape(crop: Image.Image) -> Optional[str]:
    """Distinguish BETH (Π) from GIMEL (corner). Returns unicode or None."""
    patch = _paper_ink_patch(crop)
    if patch is None:
        return None
    h, w = patch.shape
    lower_columns = patch[h // 2 :, :].sum(axis=0) > 0
    lower_runs = 0
    in_run = False
    for active in lower_columns:
        if active and not in_run:
            lower_runs += 1
            in_run = True
        elif not active:
            in_run = False
    if lower_runs == 1:
        return _CHARS["GIMEL"][0]
    if lower_runs >= 2:
        return _CHARS["BETH"][0]

    top_h = max(3, h // 3)
    top = patch[:top_h, :]
    col_top: List[int] = []
    for c in range(w):
        rows = np.where(top[:, c])[0]
        col_top.append(int(rows.min()) if rows.size else top_h)
    left = col_top[: max(1, w // 3)]
    right = col_top[-max(1, w // 3) :]
    left_min = min(left)
    right_min = min(right)
    if left_min < right_min - 2:
        return _CHARS["GIMEL"][0]
    if abs(left_min - right_min) <= 2 and left_min <= 1:
        return _CHARS["BETH"][0]
    return None


def _paper_waw_ayn_shape(crop: Image.Image) -> Optional[str]:
    """Distinguish WAW (circle+stem) from AYN (empty ring)."""
    patch = _paper_ink_patch(crop)
    if patch is None:
        return None
    m = _paper_shape_metrics(patch)
    if m["vert"] >= 0.82:
        return _CHARS["WAW"][0]
    if m["hole"] >= 0.58 and m["vert"] <= 0.72:
        return _CHARS["AYN"][0]
    if m["vert"] >= 0.78 and m["hole"] <= 0.65:
        return _CHARS["WAW"][0]
    if m["hole"] >= 0.55 and m["vert"] + 0.05 < m["hole"]:
        return _CHARS["AYN"][0]
    return None


def _paper_box_family_shape(crop: Image.Image) -> Optional[str]:
    """Distinguish TETH / DHADHE / BETH on clean digital paper."""
    patch = _paper_ink_patch(crop)
    if patch is None:
        return None
    m = _paper_shape_metrics(patch)
    h, w = patch.shape

    lower_columns = patch[h // 2 :, :].sum(axis=0) > 0
    lower_runs = 0
    in_run = False
    for active in lower_columns:
        if active and not in_run:
            lower_runs += 1
            in_run = True
        elif not active:
            in_run = False

    # Empty interior + weak center stem → BETH (survives dense-page bottom bars).
    if m["hole"] >= 0.80 and m["vert"] <= 0.55:
        return _CHARS["BETH"][0]
    if m["bot"] <= 0.60 and m["top"] >= 0.70 and m["hole"] >= 0.70 and m["vert"] <= 0.55:
        return _CHARS["BETH"][0]
    if lower_runs == 2 and m["vert"] <= 0.55 and m["hole"] >= 0.70:
        return _CHARS["BETH"][0]

    if m["top"] >= 0.75 and m["bot"] >= 0.75:
        if m["vert"] >= 0.78 and m["hole"] <= 0.70 and m["vert"] >= m["horiz"] + 0.08:
            return _CHARS["TETH"][0]
        if m["horiz"] >= 0.72 and m["horiz"] >= m["vert"] + 0.12:
            return _CHARS["DHADHE"][0]
        if m["vert"] >= 0.88 and m["hole"] <= 0.65:
            return _CHARS["TETH"][0]
        if m["horiz"] >= 0.80 and m["vert"] <= 0.60:
            return _CHARS["DHADHE"][0]
        if m["hole"] >= 0.75 and m["vert"] <= 0.55:
            return _CHARS["BETH"][0]

    return None


def _override_pred(pred: dict, character: str, *, source: str) -> dict:
    conf = float(pred.get("confidence") or 0.0)
    name = next(n for n, (u, _) in _CHARS.items() if u == character)
    return {
        **pred,
        "character": character,
        "display": character,
        "name": name,
        "source": source,
        "trusted": conf >= 0.45,
    }


def _apply_paper_shape_fixes(crop: Image.Image, pred: dict) -> dict:
    """Apply geometry overrides for common digital-font lookalike pairs."""
    ch = pred.get("character")
    waw, ayn = _CHARS["WAW"][0], _CHARS["AYN"][0]
    teth, dhadhe, beth = _CHARS["TETH"][0], _CHARS["DHADHE"][0], _CHARS["BETH"][0]
    gimel = _CHARS["GIMEL"][0]

    # Always re-check TETH → BETH on dense pages.
    if ch == teth:
        hint = _paper_box_family_shape(crop)
        if hint == beth:
            return _override_pred(pred, beth, source="paper_box_family_shape")
        if hint == dhadhe:
            return _override_pred(pred, dhadhe, source="paper_box_family_shape")

    if ch in {teth, dhadhe, beth}:
        hint = _paper_box_family_shape(crop)
        if hint is not None and hint != ch:
            return _override_pred(pred, hint, source="paper_box_family_shape")

    if pred.get("character") == beth:
        hint = _paper_beth_gimel_shape(crop)
        if hint == gimel:
            return _override_pred(pred, gimel, source="paper_beth_gimel_shape")

    ch = pred.get("character")
    if ch in {waw, ayn}:
        hint = _paper_waw_ayn_shape(crop)
        if hint is not None and hint != ch:
            return _override_pred(pred, hint, source="paper_waw_ayn_shape")

    return pred


def _paper_pred_score(pred: dict) -> float:
    ch = pred.get("character")
    conf = float(pred.get("confidence") or 0.0)
    if not ch or ch in {"?", "UNKNOWN"}:
        return conf * 0.15
    return conf


def _classify_paper_glyph(
    predictor: MusnadPredictor,
    crop: Image.Image,
) -> dict:
    """
    Classify one paper glyph crop.

    Clean digital Musnad: CNN letters-only, no stone prototype gallery.
    Letterbox retry only when the first pass is unknown.
    """
    pred = predictor.predict(
        crop,
        compare_preprocess=False,
        use_prototypes=False,
        letters_only=True,
    )
    if pred.get("character") not in {None, "?", "UNKNOWN"}:
        return _apply_paper_shape_fixes(crop, pred)

    cw, ch = crop.size
    m = max(10, int(0.22 * max(cw, ch)))
    canvas = Image.new("RGB", (cw + 2 * m, ch + 2 * m), (255, 255, 255))
    canvas.paste(crop.convert("RGB"), (m, m))
    pred_lb = predictor.predict(
        canvas,
        compare_preprocess=False,
        use_prototypes=False,
        letters_only=True,
    )
    if _paper_pred_score(pred_lb) > _paper_pred_score(pred):
        pred = {**pred_lb, "source": f"{pred_lb.get('source', 'cnn')}+letterbox"}
    return _apply_paper_shape_fixes(crop, pred)


class MusnadOCR:
    """
    Full paper-line OCR engine.

    Loads the classifier once; reuse across images.
    """

    def __init__(self, *, force_cpu: bool = False) -> None:
        self.predictor = MusnadPredictor(force_cpu=force_cpu)
        self.device = self.predictor.device

    def recognize(
        self,
        image: Union[str, Path, Image.Image],
        *,
        out_dir: Optional[Path] = None,
        pad: int = 6,
        image_id: Optional[str] = None,
        save_overlay: bool = True,
        save_crops: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full paper OCR.

        When ``save_overlay`` is True (default), writes an annotated image with:
          - bounding box around each detected glyph
          - predicted name / separator marker
          - confidence percentage

        The overlay path is returned as ``result["overlay_path"]``.
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            pil = load_external_image(image_path)
            image_id = image_id or image_path.stem
            image_str = str(image_path)
        else:
            pil = image
            image_id = image_id or "image"
            image_str = image_id

        layout, _mask = detect_paper_layout(pil)

        line_results: List[dict] = []
        all_glyphs: List[dict] = []

        for li, line in enumerate(layout):
            glyphs: List[dict] = []
            letter_boxes = [b for b, sep in line if not sep]
            for gi, (box, is_sep) in enumerate(line):
                if is_sep:
                    item = {
                        "index": gi,
                        "box": box.as_tuple(),
                        "character": WORD_SEPARATOR_LABEL,
                        "display": WORD_SEPARATOR_DISPLAY,
                        "name": "WORD_SEPARATOR",
                        "confidence": 1.0,
                        "trusted": True,
                        "source": "word_bar",
                        "is_separator": True,
                    }
                    glyphs.append(item)
                    all_glyphs.append(item)
                    continue

                crop = crop_glyph(pil, box, pad=pad, neighbors=letter_boxes)
                pred = _classify_paper_glyph(self.predictor, crop)

                if out_dir is not None and save_crops:
                    crop_dir = Path(out_dir) / "crops"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop.save(crop_dir / f"l{li:02d}_g{gi:02d}.png")

                character = pred.get("character")
                if is_word_separator_label(character) or (
                    isinstance(character, str) and character.upper().startswith("NUM_")
                ):
                    aspect = box.width / max(box.height, 1)
                    if aspect <= 0.45:
                        item = {
                            "index": gi,
                            "box": box.as_tuple(),
                            "character": WORD_SEPARATOR_LABEL,
                            "display": WORD_SEPARATOR_DISPLAY,
                            "name": "WORD_SEPARATOR",
                            "confidence": float(pred.get("confidence") or 0.0),
                            "trusted": True,
                            "source": "classifier_as_bar",
                            "is_separator": True,
                        }
                        glyphs.append(item)
                        all_glyphs.append(item)
                        continue
                    remapped = None
                    for alt in pred.get("top_k") or []:
                        ch = alt.get("character")
                        if (
                            ch
                            and not is_word_separator_label(ch)
                            and not str(ch).upper().startswith("NUM_")
                            and ch != "?"
                        ):
                            remapped = alt
                            break
                    if remapped is not None:
                        character = remapped["character"]
                        pred = {
                            **pred,
                            "character": character,
                            "name": remapped.get("name") or pred.get("name"),
                            "confidence": float(remapped.get("confidence") or 0.0),
                            "source": "letters_only_remap",
                        }
                    else:
                        character = "?"
                        pred = {
                            **pred,
                            "character": "?",
                            "name": "UNKNOWN",
                            "confidence": float(pred.get("confidence") or 0.0),
                            "source": "letters_only_block_numeral",
                        }

                item = {
                    "index": gi,
                    "box": box.as_tuple(),
                    "character": character,
                    "display": character,
                    "name": pred.get("name"),
                    "confidence": float(pred.get("confidence") or 0.0),
                    "trusted": float(pred.get("confidence") or 0.0) >= 0.45
                    and character not in (None, "?"),
                    "source": pred.get("source"),
                    "is_separator": False,
                }
                glyphs.append(item)
                all_glyphs.append(item)

            words = split_words_by_separator(glyphs)
            word_texts = [join_word_text(w) for w in words]
            line_text = format_line_text(glyphs)
            line_results.append(
                {
                    "line": li,
                    "n": len([g for g in glyphs if not g.get("is_separator")]),
                    "text": line_text,
                    "text_display": line_text,
                    "words": word_texts,
                    "n_words": len([t for t in word_texts if t]),
                    "glyphs": glyphs,
                }
            )

        page_text = "\n".join(ln["text"] for ln in line_results if ln["text"])

        result: Dict[str, Any] = {
            "ok": True,
            "mode": "paper_line",
            "domain": "paper",
            "direction": "rtl",
            "text_direction": "rtl",
            "glyph_order": "rtl_logical",
            "word_separator": WORD_SEPARATOR_DISPLAY,
            "n_glyphs": sum(ln["n"] for ln in line_results),
            "n_lines": len(line_results),
            "text": page_text,
            "lines": line_results,
            "glyphs": all_glyphs,
            "device": str(self.device),
            "image": image_str,
        }

        if save_overlay:
            package_root = Path(__file__).resolve().parent.parent
            if out_dir is None:
                out_dir = package_root / "outputs" / image_id
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            overlay = draw_annotations(pil, all_glyphs)
            overlay_path = out_dir / "overlay.png"
            overlay.save(overlay_path)
            result["overlay_path"] = str(overlay_path)
            result["overlay"] = overlay

            serializable = {k: v for k, v in result.items() if k != "overlay"}
            (out_dir / "result.json").write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return result


def recognize_paper(
    image: Union[str, Path, Image.Image],
    *,
    force_cpu: bool = False,
    out_dir: Optional[Path] = None,
    pad: int = 6,
    save_overlay: bool = True,
) -> Dict[str, Any]:
    """One-shot full paper OCR helper (writes annotated overlay by default)."""
    engine = MusnadOCR(force_cpu=force_cpu)
    return engine.recognize(
        image, out_dir=out_dir, pad=pad, save_overlay=save_overlay
    )
