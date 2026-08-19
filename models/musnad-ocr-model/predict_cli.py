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
    parser.add_argument("image", help="Path to glyph, paper line, or stone photo")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper / manuscript line OCR (MusnadOCR).",
    )
    parser.add_argument(
        "--stone",
        action="store_true",
        help="Stone inscription OCR (MusnadStoneOCR).",
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
        help="Where to write OCR overlay + result.json",
    )
    args = parser.parse_args()

    if args.paper and args.stone:
        print("Use only one of --paper or --stone.", file=sys.stderr)
        return 2

    force_cpu = not args.gpu
    image_path = Path(args.image)
    out_dir = args.out_dir or (Path("outputs") / image_path.stem)

    if args.paper:
        from inference.paper_ocr import recognize_paper

        result = recognize_paper(
            image_path, force_cpu=force_cpu, out_dir=out_dir
        )
    elif args.stone:
        from inference.stone_ocr import recognize_stone_image

        result = recognize_stone_image(
            image_path, force_cpu=force_cpu, out_dir=out_dir
        )
    else:
        from inference.predict import predict_image

        result = predict_image(image_path, force_cpu=force_cpu, top_k=args.top_k)

    if args.json:
        json.dump(json_ready(result), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    if args.paper or args.stone:
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
        "\n(tip: for multi-character / full-page images, re-run with --paper or --stone)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
