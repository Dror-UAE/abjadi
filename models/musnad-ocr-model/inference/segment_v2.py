"""
Segmentation v2 — one clean baseline, no repair chain, no musnad_final.

Pipeline:
  line crop → letter_boundary_v2 → peak cuts → boxes (geometry constraints only)

Geometry constraints are fixed constants, not per-image knobs:
  * peak if local max and p >= PEAK_THRESHOLD
  * collapse peaks closer than PEAK_NMS_PX (same column, not letter logic)
  * drop a span with width < MIN_BOX_PX
  * drop a span whose mean objectness < EMPTY_OBJECTNESS (bare margin)

Train / val / held-out test are split by source-image stem and frozen on disk.
Validation is for checkpoint choice only. The test split is scored once.

Usage:
  python -m src.segment_v2 --train
  python -m src.segment_v2 --eval
  python -m src.segment_v2 --image test_images/test-1.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PACKAGE_ROOT / "model"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
STONE_LINES_DIR = PACKAGE_ROOT / "data" / "stone_lines"


import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


from .device import dataloader_kwargs, print_device_info, resolve_device
from .letter_boundary_net import (
    INPUT_HEIGHT,
    TRAIN_WIDTH,
    LetterBoundaryNet,
    _column_loss_weight,
    _letter_objectness,
    _soft_targets,
    prepare_line,
)


MODEL_PATH = MODELS_DIR / "letter_boundary_v2.pth"
REPORT_PATH = OUTPUTS_DIR / "segment_v2" / "report.json"
SEED = 7
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

PEAK_THRESHOLD = 0.35
PEAK_NMS_PX = 3
MIN_BOX_PX = 2
EMPTY_OBJECTNESS = 0.12
EVAL_BOUNDARY_TOL_FRAC = 0.04
EVAL_BOUNDARY_TOL_MIN = 4
EVAL_IOU = 0.50


def _split_path() -> Path:
    from annotate_lines import REAL_LINES_DIR

    return REAL_LINES_DIR / "seg_v2_split.json"


def _stem(name: str | None) -> str:
    return Path(str(name or "")).stem.lower()


def _group_key(entry: dict) -> str:
    return _stem(entry.get("image_name")) or _stem(entry.get("crop"))


def load_or_create_split(entries: list[dict]) -> dict:
    """Image-level 70/15/15 split, frozen after the first write."""
    split_path = _split_path()
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        known = set(split.get("train", [])) | set(split.get("val", [])) | set(split.get("test", []))
        missing = sorted({_group_key(e) for e in entries} - known)
        if missing:
            # New annotations go to train only; val/test stay frozen.
            split["train"] = sorted(set(split.get("train", [])) | set(missing))
            split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
            print(
                f"Added {len(missing)} new image(s) to train split: {missing}",
                flush=True,
            )
        return split

    groups = sorted({_group_key(e) for e in entries if _group_key(e)})
    rng = random.Random(SEED)
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, int(round(TRAIN_FRAC * n)))
    n_val = max(1, int(round(VAL_FRAC * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1) if n >= 3 else 0
    n_test = n - n_train - n_val
    split = {
        "seed": SEED,
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
        "note": "Split by source-image stem. Do not retune using the test list.",
    }
    split_path = _split_path()
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Wrote frozen split {split_path}", flush=True)
    print(
        f"  images train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])}",
        flush=True,
    )
    return split


def split_entries(entries: list[dict], split: dict, part: str) -> list[dict]:
    wanted = {str(s).lower() for s in split[part]}
    return [e for e in entries if _group_key(e) in wanted]


def _prepare_item(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = entry["image"]
    scale = INPUT_HEIGHT / max(image.shape[0], 1)
    prepared = prepare_line(image)
    width = prepared.shape[1]
    boundaries = [b * scale for b in entry["boundaries"]]
    boundary_target = _soft_targets(width, boundaries)
    object_target = _letter_objectness(width, entry["letters"], scale)
    if not object_target.any():
        object_target = np.ones(width, dtype=np.float32)
    weight = _column_loss_weight(
        width,
        boundary_target,
        object_target,
        entry["letters"],
        scale,
        real=True,
    )
    return prepared, boundary_target, object_target, weight


class RealLineDataset(Dataset):
    def __init__(self, entries: list[dict], *, train: bool) -> None:
        self.items = [_prepare_item(e) for e in entries]
        self.train = train
        if not self.items:
            raise RuntimeError("No real lines in this split.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, boundary_target, object_target, column_weight = self.items[index]
        width = image.shape[1]
        if width >= TRAIN_WIDTH:
            start = random.randint(0, width - TRAIN_WIDTH) if self.train else 0
            sl = slice(start, start + TRAIN_WIDTH)
            image = image[:, sl]
            boundary_target = boundary_target[sl]
            object_target = object_target[sl]
            column_weight = column_weight[sl]
        else:
            pad = TRAIN_WIDTH - width
            image = np.pad(image, ((0, 0), (0, pad)), mode="edge")
            boundary_target = np.pad(boundary_target, (0, pad))
            object_target = np.pad(object_target, (0, pad))
            column_weight = np.pad(column_weight, (0, pad), constant_values=1.0)
        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            boundary_target = np.ascontiguousarray(boundary_target[::-1])
            object_target = np.ascontiguousarray(object_target[::-1])
            column_weight = np.ascontiguousarray(column_weight[::-1])
        target = np.stack(
            [boundary_target, object_target, column_weight]
        ).astype(np.float32)
        return torch.from_numpy(image.copy()).unsqueeze(0), torch.from_numpy(target)


def _loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    boundary_logits = logits[:, 0]
    object_logits = logits[:, 1]
    boundary_target = target[:, 0]
    object_target = target[:, 1]
    boundary_weight = target[:, 2]
    boundary_raw = nn.functional.binary_cross_entropy_with_logits(
        boundary_logits, boundary_target, reduction="none"
    )
    object_raw = nn.functional.binary_cross_entropy_with_logits(
        object_logits, object_target, reduction="none"
    )
    object_weight = 1.0 + 2.0 * object_target
    return (boundary_raw * boundary_weight).mean() + 0.35 * (object_raw * object_weight).mean()


def peaks_from_profile(profile: np.ndarray) -> list[int]:
    """Local maxima above a fixed threshold; 3px NMS only."""
    if profile.size < 3:
        return []
    smooth = cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=2.0
    ).ravel()
    interior = np.arange(1, smooth.shape[0] - 1)
    is_peak = (smooth[interior] >= smooth[interior - 1]) & (
        smooth[interior] > smooth[interior + 1]
    )
    xs = [int(x) for x in interior[is_peak] if float(smooth[x]) >= PEAK_THRESHOLD]
    xs.sort(key=lambda x: -float(smooth[x]))
    kept: list[int] = []
    for x in xs:
        if all(abs(x - p) >= PEAK_NMS_PX for p in kept):
            kept.append(x)
    return sorted(kept)


def boxes_from_profiles(
    boundary: np.ndarray,
    objectness: np.ndarray,
) -> list[tuple[int, int, float]]:
    w = int(boundary.shape[0])
    cuts = [c for c in peaks_from_profile(boundary) if 0 < c < w]
    edges = [0] + cuts + [w]
    boxes: list[tuple[int, int, float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right - left < MIN_BOX_PX:
            continue
        obj = float(objectness[left:right].mean()) if right > left else 0.0
        if obj < EMPTY_OBJECTNESS:
            continue
        conf = float(objectness[left:right].max()) if right > left else 0.0
        boxes.append((int(left), int(right), conf))
    return boxes


def load_v2_model(device: torch.device) -> LetterBoundaryNet:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Train with: python -m src.segment_v2 --train"
        )
    model = LetterBoundaryNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def segment_line(
    line_bgr: np.ndarray,
    model: LetterBoundaryNet,
    device: torch.device,
) -> list[tuple[int, int, float]]:
    from .letter_boundary_net import predict_profiles

    if line_bgr.size == 0 or min(line_bgr.shape[:2]) < 4:
        return []
    boundary, objectness = predict_profiles(line_bgr, model, device)
    return boxes_from_profiles(boundary, objectness)


def _iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / max(union, 1)


def score_line(
    pred_boxes: list[tuple[int, int]],
    gt_boxes: list[tuple[int, int]],
    gt_bounds: list[int],
    line_height: int,
    line_width: int,
) -> dict:
    tol = max(EVAL_BOUNDARY_TOL_MIN, int(EVAL_BOUNDARY_TOL_FRAC * max(line_height, 1)))
    pred_cuts = sorted(
        {a for a, _b in pred_boxes if a > 0}
        | {b for _a, b in pred_boxes if b < line_width}
    )
    hit = 0
    used: set[int] = set()
    for g in gt_bounds:
        best_i = None
        best_d = tol + 1
        for i, p in enumerate(pred_cuts):
            if i in used:
                continue
            d = abs(p - g)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is not None and best_d <= tol:
            hit += 1
            used.add(best_i)
    missed = len(gt_bounds) - hit
    false_b = len(pred_cuts) - len(used)
    prec_b = hit / max(len(pred_cuts), 1)
    rec_b = hit / max(len(gt_bounds), 1)

    pairs: list[tuple[float, int, int]] = []
    for gi, (g0, g1) in enumerate(gt_boxes):
        for pj, (p0, p1) in enumerate(pred_boxes):
            iou = _iou(g0, g1, p0, p1)
            if iou >= EVAL_IOU:
                pairs.append((iou, gi, pj))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    for _iou_v, gi, pj in pairs:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
    matched_gt = len(used_g)
    merge_miss = len(gt_boxes) - matched_gt
    oversplit = len(pred_boxes) - len(used_p)
    exact = (
        len(gt_boxes) == len(pred_boxes)
        and matched_gt == len(gt_boxes)
        and merge_miss == 0
        and oversplit == 0
    )
    return {
        "gt_letters": len(gt_boxes),
        "pred_boxes": len(pred_boxes),
        "boxes_matched_iou50": matched_gt,
        "merge_or_miss": merge_miss,
        "oversplit_or_extra": oversplit,
        "boundary_hit": hit,
        "boundary_missed": missed,
        "boundary_false": false_b,
        "boundary_precision": round(prec_b, 4),
        "boundary_recall": round(rec_b, 4),
        "exact": exact,
    }


def evaluate_entries(
    entries: list[dict],
    model: LetterBoundaryNet,
    device: torch.device,
    *,
    part: str,
) -> dict:
    rows = []
    for entry in entries:
        pred = segment_line(entry["image"], model, device)
        pred_boxes = [(a, b) for a, b, _c in pred]
        gt_boxes = [(int(L["x_left"]), int(L["x_right"])) for L in entry["letters"]]
        row = score_line(
            pred_boxes,
            gt_boxes,
            list(entry["boundaries"]),
            int(entry["height"]),
            int(entry["width"]),
        )
        row["crop"] = entry["crop"]
        row["image"] = entry.get("image_name")
        rows.append(row)

    n = max(len(rows), 1)
    def _sum(key: str) -> int:
        return int(sum(r[key] for r in rows))

    gt = _sum("gt_letters")
    pred_n = _sum("pred_boxes")
    matched = _sum("boxes_matched_iou50")
    merge = _sum("merge_or_miss")
    extra = _sum("oversplit_or_extra")
    b_hit = _sum("boundary_hit")
    b_miss = _sum("boundary_missed")
    b_false = _sum("boundary_false")
    summary = {
        "part": part,
        "n_lines": len(rows),
        "gt_letters": gt,
        "pred_boxes": pred_n,
        "box_match_rate": round(matched / max(gt, 1), 4),
        "merge_rate": round(merge / max(gt, 1), 4),
        "oversplit_rate": round(extra / max(pred_n, 1), 4),
        "boundary_precision": round(b_hit / max(b_hit + b_false, 1), 4),
        "boundary_recall": round(b_hit / max(b_hit + b_miss, 1), 4),
        "exact_line_accuracy": round(sum(1 for r in rows if r["exact"]) / n, 4),
        "lines": rows,
    }
    print(
        f"{part}: lines={summary['n_lines']}  box_match={summary['box_match_rate']:.3f}  "
        f"merge={summary['merge_rate']:.3f}  oversplit={summary['oversplit_rate']:.3f}  "
        f"bP={summary['boundary_precision']:.3f}  bR={summary['boundary_recall']:.3f}  "
        f"exact={summary['exact_line_accuracy']:.3f}",
        flush=True,
    )
    return summary


def train(args: argparse.Namespace) -> None:
    from annotate_lines import load_usable_real_lines

    device = print_device_info(resolve_device(args.cpu))
    entries = load_usable_real_lines()
    if len(entries) < 9:
        raise SystemExit("Need more usable real_lines crops before training v2.")
    split = load_or_create_split(entries)
    train_e = split_entries(entries, split, "train")
    val_e = split_entries(entries, split, "val")
    print(f"lines train={len(train_e)} val={len(val_e)} (test untouched)", flush=True)

    train_ds = RealLineDataset(train_e, train=True)
    loader_opts = dataloader_kwargs(device, num_workers=0)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_opts
    )

    model = LetterBoundaryNet().to(device)
    from .letter_boundary_net import MODEL_PATH as V1_PATH

    if V1_PATH.exists() and not args.from_scratch:
        ckpt = torch.load(V1_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Warm start from {V1_PATH}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = _loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
        scheduler.step()
        train_loss = total / max(len(train_ds), 1)
        model.eval()
        val_summary = evaluate_entries(val_e, model, device, part="val")
        score = val_summary["box_match_rate"]
        marker = ""
        if score > best:
            best = score
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "val_box_match_rate": score,
                    "epoch": epoch,
                },
                MODEL_PATH,
            )
            marker = "  saved"
        else:
            stale += 1
        print(
            f"epoch {epoch:02d}/{args.epochs}  train_loss {train_loss:.4f}  "
            f"val_box_match {score:.3f}{marker}",
            flush=True,
        )
        if args.patience and stale >= args.patience:
            print(f"early stop after {args.patience} epochs without val box-match gain", flush=True)
            break
    print(f"Best val box_match {best:.3f} -> {MODEL_PATH}", flush=True)


def run_eval(args: argparse.Namespace, *, include_test: bool) -> dict:
    from annotate_lines import load_usable_real_lines

    device = print_device_info(resolve_device(args.cpu))
    entries = load_usable_real_lines()
    split = load_or_create_split(entries)
    model = load_v2_model(device)
    report = {
        "model": str(MODEL_PATH),
        "split": str(_split_path()),
        "geometry": {
            "peak_threshold": PEAK_THRESHOLD,
            "peak_nms_px": PEAK_NMS_PX,
            "min_box_px": MIN_BOX_PX,
            "empty_objectness": EMPTY_OBJECTNESS,
        },
        "eval": {
            "boundary_tol_frac": EVAL_BOUNDARY_TOL_FRAC,
            "iou": EVAL_IOU,
        },
        "parts": {},
    }
    for part in ("train", "val"):
        report["parts"][part] = evaluate_entries(
            split_entries(entries, split, part), model, device, part=part
        )
        report["parts"][part].pop("lines", None)
    if include_test:
        test_full = evaluate_entries(
            split_entries(entries, split, "test"), model, device, part="test"
        )
        report["parts"]["test"] = test_full
        report["test_stems"] = split["test"]
    else:
        report["test_stems"] = split["test"]
        report["note"] = "Test not scored. Pass --eval-test once for the held-out number."
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}", flush=True)
    return report


def detect_image_v2(image_path: Path | str, *, device=None, out_dir: Path | None = None) -> dict:
    from .inscription_region import isolate_inscription_if_sparse
    from .letter_detector import render_detections, segment_tiles
    from .stone_glyph_segmentation import detect_line_bands, load_bgr

    image_path = Path(image_path)
    if device is None:
        device = resolve_device()
    model = load_v2_model(device)
    image = load_bgr(image_path)
    work, region = isolate_inscription_if_sparse(image)
    rx, ry = (region.x_left, region.y_top) if region.applied else (0, 0)
    bands = detect_line_bands(work)
    if out_dir is None:
        out_dir = OUTPUTS_DIR / "segment_v2" / image_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines_out = []
    for index, band in enumerate(bands, start=1):
        crop = work[band.y_top : band.y_bottom]
        boxes = segment_line(crop, model, device)
        overlay = render_detections(crop, boxes)
        tiles = segment_tiles(crop, boxes)
        out_overlay = out_dir / f"line{index:02d}_boxes.jpg"
        out_tiles = out_dir / f"line{index:02d}_segments.jpg"
        cv2.imwrite(str(out_overlay), overlay)
        cv2.imwrite(str(out_tiles), tiles)
        box_rows = [
            {"x_left": int(a) + rx, "x_right": int(b) + rx, "confidence": float(c)}
            for a, b, c in boxes
        ]
        lines_out.append(
            {
                "line": index - 1,
                "y_top": int(band.y_top) + ry,
                "y_bottom": int(band.y_bottom) + ry,
                "n": len(boxes),
                "boxes": box_rows,
                "overlay_path": str(out_overlay),
                "segments_path": str(out_tiles),
            }
        )
    return {
        "ok": True,
        "mode": "segment_v2",
        "image": str(image_path),
        "n_lines": len(lines_out),
        "n_letters": int(sum(ln["n"] for ln in lines_out)),
        "lines": lines_out,
        "out_dir": str(out_dir),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation v2 baseline")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Score the frozen held-out test split once. Do not use this to retune.",
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.train:
        train(args)
        run_eval(args, include_test=False)
    elif args.eval or args.eval_test:
        run_eval(args, include_test=bool(args.eval_test))
    elif args.image:
        result = detect_image_v2(args.image, device=resolve_device(args.cpu))
        for ln in result["lines"]:
            print(
                f"line {ln['line'] + 1}: {ln['n']} letters -> {ln['overlay_path']}",
                flush=True,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
