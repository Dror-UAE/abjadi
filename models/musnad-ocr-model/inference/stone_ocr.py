"""
Musnad stone / inscription OCR (v0.5.0).

Pipeline:
  1. Line banding (stone_glyph_segmentation)
  2. Letter localization (segment_v2: boundary cuts only)
  3. Classify each frozen crop with musnad_final (recognition only)
  4. Split words on vertical-bar / NUM_1 separators

Use MusnadStoneOCR — NOT MusnadOCR — for stone photos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image

from .device import resolve_device
from .layout import (
    WORD_SEPARATOR_DISPLAY,
    WORD_SEPARATOR_LABEL,
    format_line_text,
    is_word_separator_label,
    join_word_text,
    split_words_by_separator,
)
from .letter_detector import detect_image
from .segment_v2 import MODEL_PATH as SEG_V2_PATH
from .paper_detect import draw_annotations
from .predict import DEFAULT_CHECKPOINT, MusnadPredictor, load_external_image
from .stone_glyph_segmentation import load_bgr

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"


def _classify_stone_glyph(predictor: MusnadPredictor, crop_path: Path) -> dict:
    return predictor.predict(
        crop_path,
        compare_preprocess=True,
        use_prototypes=True,
        letters_only=False,
    )


def _maybe_separator(width: int, line_height: int, pred: dict) -> dict | None:
    character = pred.get("character")
    aspect = width / max(line_height, 1)
    is_num = is_word_separator_label(character) or (
        isinstance(character, str) and character.upper().startswith("NUM_")
    )
    if is_num and aspect <= 0.55:
        return {
            "character": WORD_SEPARATOR_LABEL,
            "display": WORD_SEPARATOR_DISPLAY,
            "name": "WORD_SEPARATOR",
            "confidence": float(pred.get("confidence") or 1.0),
            "trusted": True,
            "source": "classifier_as_bar",
            "is_separator": True,
        }
    return None


def _stone_crop_content(piece: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return {
        "edge_peak": float(np.percentile(mag, 92)),
        "edge_mean": float(mag.mean()),
        "edge_ratio": float((mag > 0.10).mean()),
    }


def _should_ignore_stone_candidate(det_conf: float, pred: dict, content: dict[str, float]) -> bool:
    id_conf = float(pred.get("confidence") or 0.0)
    character = pred.get("character")
    unknown = character in {None, "?", "UNKNOWN"}
    low_content = (
        content["edge_peak"] < 0.25
        and content["edge_mean"] < 0.13
        and content["edge_ratio"] < 0.45
    )
    if low_content and id_conf < 0.55:
        return True
    if det_conf < 0.70 and id_conf < 0.45 and low_content:
        return True
    if unknown and low_content and det_conf < 0.80 and id_conf < 0.55:
        return True
    return False


def _save_stone_overlay(
    image_bgr: np.ndarray,
    line_results: list[dict],
    out_dir: Path,
) -> tuple[str, Image.Image]:
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    all_glyphs = [g for ln in line_results for g in (ln.get("glyphs") or [])]
    overlay = draw_annotations(pil, all_glyphs)
    overlay_path = out_dir / "overlay.png"
    overlay.save(overlay_path)
    return str(overlay_path), overlay


def recognize_stone(
    image: Union[str, Path, Image.Image],
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    device=None,
    out_dir: Path | None = None,
    detector_model=None,
    predictor: MusnadPredictor | None = None,
    pad: int = 2,
    image_id: str | None = None,
    save_overlay: bool = True,
) -> dict[str, Any]:
    """Detect letters on stone, then identify each crop (RTL logical order)."""
    if device is None:
        device = resolve_device()
    if predictor is None:
        predictor = MusnadPredictor(force_cpu=(str(device) == "cpu"))
    if not SEG_V2_PATH.exists():
        raise FileNotFoundError(
            f"Missing segmentation v2: {SEG_V2_PATH}. "
            "Re-run sync_package to copy model/letter_boundary_v2.pth."
        )
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Missing classifier: {checkpoint}")

    if isinstance(image, (str, Path)):
        image_path = Path(image)
        image_id = image_id or image_path.stem
    else:
        image_path = None
        image_id = image_id or "image"

    if out_dir is None:
        out_dir = OUTPUTS_DIR / "stone_ocr" / image_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    if image_path is not None:
        detect = detect_image(
            image_path,
            device=device,
            model=None,
            out_dir=out_dir / "detect",
            mode="v2",
        )
        bgr = load_bgr(image_path)
    else:
        tmp = out_dir / "_input.png"
        rgb = np.array(image.convert("RGB"))
        Image.fromarray(rgb).save(tmp)
        detect = detect_image(
            tmp,
            device=device,
            model=None,
            out_dir=out_dir / "detect",
            mode="v2",
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    line_results: list[dict] = []
    for ln in detect.get("lines") or []:
        li = int(ln["line"])
        y0, y1 = int(ln["y_top"]), int(ln["y_bottom"])
        line_h = max(1, y1 - y0)
        crop_line = bgr[y0:y1]
        glyphs: list[dict] = []

        for gi, box in enumerate(ln.get("boxes") or []):
            a, b = int(box["x_left"]), int(box["x_right"])
            det_conf = float(box.get("confidence") or 0.0)
            a0 = max(0, a - pad)
            b0 = min(crop_line.shape[1], b + pad)
            piece = crop_line[:, a0:b0]
            if piece.size == 0 or b0 <= a0:
                continue

            crop_path = crop_dir / f"l{li:02d}_g{gi:02d}.png"
            rgb_piece = cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_piece).save(crop_path)

            pred = _classify_stone_glyph(predictor, crop_path)
            content = _stone_crop_content(piece)
            if _should_ignore_stone_candidate(det_conf, pred, content):
                continue
            sep = _maybe_separator(b - a, line_h, pred)
            if sep is not None:
                glyphs.append(
                    {
                        "index": gi,
                        "box": (a, y0, b, y1),
                        "detect_confidence": det_conf,
                        "content": content,
                        **sep,
                    }
                )
                continue

            character = pred.get("character")
            trust = pred.get("trust") or {}
            glyphs.append(
                {
                    "index": gi,
                    "box": (a, y0, b, y1),
                    "character": character,
                    "display": character if character not in (None, "?", "UNKNOWN") else "?",
                    "name": pred.get("name") or "UNKNOWN",
                    "confidence": float(pred.get("confidence") or 0.0),
                    "detect_confidence": det_conf,
                    "content": content,
                    "trusted": bool(trust.get("trusted", True))
                    and float(pred.get("confidence") or 0.0) >= 0.40
                    and character not in (None, "?", "UNKNOWN"),
                    "source": pred.get("source"),
                    "is_separator": False,
                }
            )

        glyphs.reverse()
        for i, g in enumerate(glyphs):
            g["index"] = i

        words = split_words_by_separator(glyphs)
        word_texts = [join_word_text(w) for w in words]
        line_text = format_line_text(glyphs)
        line_results.append(
            {
                "line": li,
                "n": len([g for g in glyphs if not g.get("is_separator")]),
                "n_detected": int(ln.get("n") or 0),
                "text": line_text,
                "text_display": line_text,
                "words": word_texts,
                "n_words": len([t for t in word_texts if t]),
                "glyphs": glyphs,
                "overlay_path": ln.get("overlay_path"),
                "segments_path": ln.get("segments_path"),
            }
        )

    page_text = "\n".join(ln["text"] for ln in line_results if ln.get("text"))
    all_glyphs = [g for ln in line_results for g in (ln.get("glyphs") or [])]
    result: dict[str, Any] = {
        "ok": True,
        "mode": "stone_line",
        "domain": "stone",
        "direction": "rtl",
        "word_separator": WORD_SEPARATOR_DISPLAY,
        "text": page_text,
        "n_glyphs": sum(ln["n"] for ln in line_results),
        "n_letters": sum(ln["n_detected"] for ln in line_results),
        "n_lines": len(line_results),
        "lines": line_results,
        "glyphs": all_glyphs,
        "out_dir": str(out_dir),
        "device": str(device),
    }

    if save_overlay:
        overlay_path, overlay = _save_stone_overlay(bgr, line_results, out_dir)
        result["overlay_path"] = overlay_path
        result["overlay"] = overlay
        (out_dir / "result.json").write_text(
            json.dumps({k: v for k, v in result.items() if k != "overlay"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        result["overlay_path"] = None

    return result


class MusnadStoneOCR:
    """Full stone inscription OCR — line banding + letter detect + classify."""

    def __init__(self, *, force_cpu: bool = False) -> None:
        self.device = resolve_device(force_cpu)
        self.predictor = MusnadPredictor(force_cpu=force_cpu)

    def recognize(
        self,
        image: Union[str, Path, Image.Image],
        *,
        out_dir: Optional[Path] = None,
        pad: int = 2,
        image_id: Optional[str] = None,
        save_overlay: bool = True,
    ) -> dict[str, Any]:
        return recognize_stone(
            image,
            device=self.device,
            out_dir=out_dir,
            predictor=self.predictor,
            pad=pad,
            image_id=image_id,
            save_overlay=save_overlay,
        )


def recognize_stone_image(
    image: Union[str, Path, Image.Image],
    *,
    force_cpu: bool = False,
    out_dir: Optional[Path] = None,
    pad: int = 2,
    image_id: Optional[str] = None,
    save_overlay: bool = True,
) -> dict[str, Any]:
    """One-shot full stone inscription OCR helper."""
    return MusnadStoneOCR(force_cpu=force_cpu).recognize(
        image,
        out_dir=out_dir,
        pad=pad,
        image_id=image_id,
        save_overlay=save_overlay,
    )
