#!/usr/bin/env python3
"""CLI entry for Musnad OCR — human-readable by default, JSON with --json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def json_ready(value: object) -> object:
    """Drop / stringify values that cannot go through json.dump."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "overlay":
                continue
            out[key] = json_ready(item)
        return out
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # PIL Image and other objects
    if hasattr(value, "save") and hasattr(value, "size"):
        return None
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Musnad OCR on an image")
    parser.add_argument("image", help="Path to glyph or paper image")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Full-page paper OCR (detect lines + glyphs). Use for multi-character images.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CUDA when available (default: CPU)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Alternatives in top_k for single-glyph mode (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON result instead of a short summary",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write paper-OCR overlay (default: outputs/<image-stem>)",
    )
    args = parser.parse_args()

    force_cpu = not args.gpu

    if args.paper:
        from inference.paper_ocr import recognize_paper

        out_dir = args.out_dir or (Path("outputs") / Path(args.image).stem)
        result = recognize_paper(
            args.image, force_cpu=force_cpu, out_dir=out_dir
        )
        if args.json:
            json.dump(json_ready(result), sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")
        else:
            text = result.get("text") or ""
            print(text)
            print(
                f"\n({result.get('n_lines', 0)} lines, "
                f"{result.get('n_glyphs', 0)} glyphs, device={result.get('device')})"
            )
            overlay = result.get("overlay_path")
            if overlay:
                print(f"overlay: {overlay}")
        return 0

    from inference.predict import predict_image

    result = predict_image(args.image, force_cpu=force_cpu, top_k=args.top_k)
    if args.json:
        json.dump(json_ready(result), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        ch = result.get("character") or "?"
        name = result.get("name") or ""
        conf = float(result.get("confidence") or 0.0)
        src = result.get("source") or ""
        print(f"{ch}  {name}  conf={conf:.3f}  source={src}")
        trust = result.get("trust") or {}
        if trust:
            print(
                f"trusted={trust.get('trusted')}  "
                f"reason={trust.get('reason')}"
            )
        print(
            "\n(tip: for multi-character / full-page images, re-run with --paper)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
