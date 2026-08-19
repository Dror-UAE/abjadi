#!/usr/bin/env python3
"""
Persistent OCR worker — loads all model weights once, then processes
requests from stdin (one JSON line per request) and writes results to
stdout (one JSON line per response).

Protocol
--------
Request  (stdin,  one line):
    {"id": "<uuid>", "mode": "stone"|"paper", "image_path": "<abs path>", "out_dir": "<abs path>"}

Response (stdout, one line):
    {"id": "<uuid>", "ok": true,  ...OCR fields...}
    {"id": "<uuid>", "ok": false, "error": "<message>"}

Stderr is forwarded to the parent process and used only for diagnostics.
The worker never exits on a per-request error; it keeps running.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the package root is on the path when invoked directly
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# ---------------------------------------------------------------------------
# Lazy imports — only pulled in after sys.path is set
# ---------------------------------------------------------------------------
import torch  # noqa: E402  (must come after path fix)
from inference.paper_ocr import MusnadOCR  # noqa: E402
from inference.stone_ocr import MusnadStoneOCR  # noqa: E402


def _json_safe(value: Any) -> Any:
    """Recursively strip non-serialisable values (PIL Images, Paths, etc.)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if k != "overlay"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # PIL Image or anything else non-serialisable
    if hasattr(value, "save") and hasattr(value, "size"):
        return None
    return str(value)


def _emit(obj: dict) -> None:
    """Write one JSON line to stdout and flush immediately."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    print(f"[ocr_worker] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Boot — load both engines once
# ---------------------------------------------------------------------------
_log("loading models...")

try:
    _paper_engine = MusnadOCR(force_cpu=True)
    _log("paper OCR ready")
except Exception as exc:
    _log(f"WARNING: paper OCR failed to load: {exc}")
    _paper_engine = None

try:
    _stone_engine = MusnadStoneOCR(force_cpu=True)
    _log("stone OCR ready")
except Exception as exc:
    _log(f"WARNING: stone OCR failed to load: {exc}")
    _stone_engine = None

_log("worker ready — waiting for requests")

# Emit a ready signal so Node.js knows the worker is up before sending jobs.
_emit({"id": "__ready__", "ok": True, "status": "ready"})


# ---------------------------------------------------------------------------
# Request loop
# ---------------------------------------------------------------------------
for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue

    req: dict = {}
    try:
        req = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        _emit({"id": req.get("id", "?"), "ok": False, "error": f"invalid JSON: {exc}"})
        continue

    req_id = req.get("id", "?")
    mode = req.get("mode", "stone")
    image_path = req.get("image_path", "")
    out_dir = req.get("out_dir", "")

    if not image_path or not Path(image_path).exists():
        _emit({"id": req_id, "ok": False, "error": f"image not found: {image_path}"})
        continue

    _log(f"processing id={req_id} mode={mode}")

    try:
        if mode == "paper":
            if _paper_engine is None:
                _emit({"id": req_id, "ok": False, "error": "paper engine not loaded"})
                continue
            result = _paper_engine.recognize(
                image_path,
                out_dir=Path(out_dir) if out_dir else None,
                save_overlay=True,
            )
        else:
            if _stone_engine is None:
                _emit({"id": req_id, "ok": False, "error": "stone engine not loaded"})
                continue
            result = _stone_engine.recognize(
                image_path,
                out_dir=Path(out_dir) if out_dir else None,
                save_overlay=True,
            )

        _emit({"id": req_id, "ok": True, **_json_safe(result)})

    except Exception as exc:  # noqa: BLE001
        _log(f"error id={req_id}: {exc}\n{traceback.format_exc()}")
        _emit({"id": req_id, "ok": False, "error": str(exc)})

    finally:
        # Release any intermediate tensors / numpy arrays from this request.
        gc.collect()

_log("stdin closed — worker exiting")
