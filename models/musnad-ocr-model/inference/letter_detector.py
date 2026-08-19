"""
Shape-aware Musnad letter detector for stone lines.

This is a separate model from ``musnad_final.pth``. Default inference is
``--mode v2`` (src/segment_v2.py): boundary cuts only. Recognition is not used
for boxing. Legacy modes: learned / boundary_first / dp.

Training labels come from ``data/stone_lines`` (exact letter boxes by
construction) plus optional hand-annotated real lines.

Usage:
  python -m src.letter_detector --build-cache
  python -m src.letter_detector --train --epochs 12
  python -m src.letter_detector --image test_images/real-stone-letters-one-line-2.jpg
"""

from __future__ import annotations

import argparse
import json
import pickle
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


from .stone_enhancement import enhance_line_for_ocr

MODEL_PATH = MODELS_DIR / "letter_detector.pth"
CACHE_PATH = STONE_LINES_DIR / "letter_detector_windows.pkl"
INPUT_SIZE = 64
CACHE_VERSION = 7
# Keep ranking pairs bounded so the cache fits in laptop RAM.
MAX_RANK_PAIRS = 80_000


# ---------------------------------------------------------------------------
# Front-end
# ---------------------------------------------------------------------------


def prepare_window(crop: np.ndarray) -> np.ndarray:
    """Letterbox a (already enhanced) gray/BGR crop to INPUT_SIZE×INPUT_SIZE."""
    if crop.size == 0:
        return np.full((INPUT_SIZE, INPUT_SIZE), 0.5, dtype=np.float32)
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = crop.shape
    side = max(h, w, 1)
    canvas = np.full((side, side), int(np.median(crop)), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = crop
    resized = cv2.resize(canvas, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def enhance_line(line_bgr: np.ndarray) -> np.ndarray:
    """Shared enhancement for a whole line crop before windowing."""
    if line_bgr.ndim == 2:
        line_bgr = cv2.cvtColor(line_bgr, cv2.COLOR_GRAY2BGR)
    return enhance_line_for_ocr(line_bgr)


# ---------------------------------------------------------------------------
# Training sample mining
# ---------------------------------------------------------------------------


def _iou_1d(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, a0) + max(b1, b0) - inter
    return inter / max(union, 1)


def _sample_windows_from_line(
    image: np.ndarray,
    letters: list[dict],
    rng: random.Random,
    *,
    positives_per_letter: int = 2,
    negatives_per_letter: int = 4,
) -> tuple[list[tuple[np.ndarray, float]], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Mine complete-letter positives, hard negatives, and full>half ranking pairs.

    Positives: exact box, mild jitter that still covers ~one glyph.
    Negatives: half / near-complete chops, straddle, empty stone, thin strip.
    Pairs: (full_crop, incomplete_crop) so training can enforce full ≫ half.
    """
    h, w = image.shape[:2]
    if not letters:
        return [], []

    samples: list[tuple[np.ndarray, float]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    boxes = [(int(L["x_left"]), int(L["x_right"])) for L in letters]
    boxes = [(max(0, a), min(w, b)) for a, b in boxes if b - a >= 3]

    for left, right in boxes:
        width = right - left
        full_crop = image[:, left:right]
        # Exact + mild jitter positives
        for _ in range(positives_per_letter):
            pad = int(rng.uniform(-0.08, 0.12) * width)
            a = max(0, left - pad)
            b = min(w, right + pad)
            if b - a < 3:
                continue
            # Reject if we swallowed a neighbour
            if any(
                _iou_1d(a, b, o0, o1) > 0.35 and (o0, o1) != (left, right)
                for o0, o1 in boxes
            ):
                continue
            samples.append((image[:, a:b], 1.0))

        # Half-letter negatives (the critical failure mode)
        mid = (left + right) // 2
        rank_halves: list[np.ndarray] = []
        for a, b in ((left, mid), (mid, right), (left, left + max(3, width // 3))):
            if b - a >= 3:
                crop = image[:, a:b]
                samples.append((crop, 0.0))
                rank_halves.append(crop)

        # Near-complete chops (~70%): these currently beat full glyphs at
        # inference because they still look "letter-like".
        span = max(3, int(0.70 * width))
        if span < width - 1:
            for a, b in ((left, left + span), (right - span, right)):
                if b - a >= 3:
                    crop = image[:, a:b]
                    samples.append((crop, 0.0))
                    if len(rank_halves) < 4:
                        rank_halves.append(crop)
            if width - span >= 4:
                start = left + (width - span) // 2
                crop = image[:, start : start + span]
                samples.append((crop, 0.0))
                if len(rank_halves) < 5:
                    rank_halves.append(crop)

        # Ranking pairs: full must outrank each incomplete view (capped).
        for half in rank_halves[:4]:
            pairs.append((full_crop, half))

        # Interior chops: valleys inside ○ / X / multi-stem letters must score
        # as incomplete even when the slice still looks "carved".
        for k in range(3):
            a = left + (k * width) // 3
            b = left + ((k + 1) * width) // 3
            if b - a >= 3:
                samples.append((image[:, a:b], 0.0))
        for _ in range(2):
            span = max(3, int(rng.uniform(0.28, 0.55) * width))
            start = rng.randint(left, max(left, right - span))
            samples.append((image[:, start : start + span], 0.0))

        # Straddle: this letter + neighbour (critical when letters are packed tight)
        neighbours = sorted(
            boxes,
            key=lambda box: min(abs(box[0] - right), abs(left - box[1])),
        )
        straddled = 0
        for o0, o1 in neighbours:
            if (o0, o1) == (left, right):
                continue
            gap = min(abs(o0 - right), abs(left - o1))
            if gap > 0.35 * width:
                continue
            a, b = min(left, o0), max(right, o1)
            if b - a >= 3 and b - a < 3.5 * width:
                samples.append((image[:, a:b], 0.0))
                straddled += 1
                if straddled >= 2:
                    break
        # Thin stem (``|`` separator) glued to a neighbour — always a negative.
        if width / max(h, 1) <= 0.28:
            for o0, o1 in neighbours[:3]:
                if (o0, o1) == (left, right):
                    continue
                if 0 <= o0 - right <= max(6, int(0.25 * h)):
                    samples.append((image[:, left : min(w, o1)], 0.0))
                if 0 <= left - o1 <= max(6, int(0.25 * h)):
                    samples.append((image[:, max(0, o0) : right], 0.0))
        # Also a tight crop that only barely includes the neighbour edge
        for o0, o1 in neighbours[:3]:
            if (o0, o1) == (left, right):
                continue
            if 0 <= o0 - right <= max(4, int(0.2 * width)):
                a, b = left, min(w, o0 + max(3, (o1 - o0) // 3))
                if b - a >= 3:
                    samples.append((image[:, a:b], 0.0))
                break
            if 0 <= left - o1 <= max(4, int(0.2 * width)):
                a, b = max(0, o1 - max(3, (o1 - o0) // 3)), right
                if b - a >= 3:
                    samples.append((image[:, a:b], 0.0))
                break

    # Empty / margin / textured-stone negatives (pits that look like strokes)
    occupied = np.zeros(w, dtype=bool)
    for a, b in boxes:
        occupied[a:b] = True
    free = np.where(~occupied)[0]
    for _ in range(max(2, negatives_per_letter)):
        if len(free) < 8:
            break
        start = int(free[rng.randrange(len(free))])
        span = rng.randint(max(6, h // 4), max(8, int(0.9 * h)))
        end = min(w, start + span)
        if end - start < 4:
            continue
        if occupied[start:end].mean() > 0.15:
            continue
        samples.append((image[:, start:end], 0.0))

    return samples, pairs


def _to_cache_image(prepared: np.ndarray) -> np.ndarray:
    """Store prepared windows as uint8 to keep the pickle RAM-friendly."""
    return np.clip(prepared * 255.0, 0, 255).astype(np.uint8)


def _from_cache_image(stored: np.ndarray) -> np.ndarray:
    if stored.dtype == np.uint8:
        return stored.astype(np.float32) / 255.0
    return np.asarray(stored, dtype=np.float32)


def _reservoir_append(
    bag: list,
    item,
    *,
    cap: int,
    seen: int,
    rng: random.Random,
) -> int:
    """Reservoir-sample ``item`` into ``bag`` (cap size). Returns new seen count."""
    seen += 1
    if len(bag) < cap:
        bag.append(item)
    else:
        j = rng.randrange(seen)
        if j < cap:
            bag[j] = item
    return seen


def build_window_cache(
    root: Path = STONE_LINES_DIR,
    *,
    max_lines: int | None = None,
    seed: int = 7,
) -> Path:
    """Extract and cache prepared window tensors for training."""
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    if max_lines is not None:
        labels = labels[:max_lines]
    rng = random.Random(seed)

    print(f"Mining windows from {len(labels)} lines...", flush=True)
    items: list[tuple[np.ndarray, float, int]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pairs_seen = 0
    step = max(1, len(labels) // 20)
    for index, entry in enumerate(labels, start=1):
        path = root / "images" / entry["image"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        enhanced = enhance_line(image)
        letters = entry.get("letters") or []
        if not letters and entry.get("boundaries"):
            edges = [0] + list(entry["boundaries"]) + [entry["width"]]
            letters = [
                {"x_left": edges[i], "x_right": edges[i + 1]}
                for i in range(len(edges) - 1)
                if edges[i + 1] - edges[i] >= 3
            ]
        crops, line_pairs = _sample_windows_from_line(enhanced, letters, rng)
        for crop, label in crops:
            items.append((_to_cache_image(prepare_window(crop)), float(label), 0))
        for full, half in line_pairs:
            pairs_seen = _reservoir_append(
                pairs,
                (_to_cache_image(prepare_window(full)), _to_cache_image(prepare_window(half))),
                cap=MAX_RANK_PAIRS,
                seen=pairs_seen,
                rng=rng,
            )
        if index % step == 0 or index == len(labels):
            print(
                f"  {index}/{len(labels)}  windows={len(items)}  pairs={len(pairs)}",
                flush=True,
            )

    from annotate_lines import load_usable_real_lines

    real_lines = load_usable_real_lines()
    for _repeat in range(4):
        for entry in real_lines:
            enhanced = enhance_line(entry["image"])
            crops, line_pairs = _sample_windows_from_line(
                enhanced,
                entry["letters"],
                rng,
                positives_per_letter=3,
                negatives_per_letter=6,
            )
            for crop, label in crops:
                items.append((_to_cache_image(prepare_window(crop)), float(label), 1))
            for full, half in line_pairs:
                pairs_seen = _reservoir_append(
                    pairs,
                    (
                        _to_cache_image(prepare_window(full)),
                        _to_cache_image(prepare_window(half)),
                    ),
                    cap=MAX_RANK_PAIRS,
                    seen=pairs_seen,
                    rng=rng,
                )
    print(
        f"After real lines: {len(items)} windows  pairs={len(pairs)}  "
        f"real_flag={sum(1 for _w, _y, h in items if h)}",
        flush=True,
    )

    positives = sum(1 for row in items if row[1] >= 0.5)
    print(
        f"Cache: {len(items)} windows  pos={positives}  neg={len(items) - positives}  "
        f"rank_pairs={len(pairs)} (seen={pairs_seen})",
        flush=True,
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as handle:
        pickle.dump(
            {"version": CACHE_VERSION, "items": items, "pairs": pairs},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Wrote {CACHE_PATH}", flush=True)
    return CACHE_PATH


def _balance_window_items(
    items: list,
    *,
    neg_ratio: float,
    seed: int,
) -> list:
    """
    Keep every positive and every real-stone (hard) negative; downsample only
    easy synthetic negatives.
    """
    if neg_ratio <= 0:
        return items

    def _y(row) -> float:
        return float(row[1])

    def _hard(row) -> bool:
        return len(row) > 2 and int(row[2]) == 1

    pos = [it for it in items if _y(it) >= 0.5]
    hard_neg = [it for it in items if _y(it) < 0.5 and _hard(it)]
    easy_neg = [it for it in items if _y(it) < 0.5 and not _hard(it)]
    cap = max(len(pos), int(round(neg_ratio * len(pos))))
    cap = max(0, cap - len(hard_neg))
    rng = random.Random(seed)
    if len(easy_neg) > cap:
        easy_neg = rng.sample(easy_neg, cap)
    out = pos + hard_neg + easy_neg
    rng.shuffle(out)
    print(
        f"Balanced windows: kept {len(pos)} pos + {len(hard_neg)} hard-neg + "
        f"{len(easy_neg)} easy-neg (neg_ratio={neg_ratio:g})",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Dataset / model
# ---------------------------------------------------------------------------


def _augment_gray(image: np.ndarray, train: bool) -> np.ndarray:
    if train and random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
    if train and random.random() < 0.3:
        image = np.clip(image + random.uniform(-0.08, 0.08), 0, 1).astype(np.float32)
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)
    return image


class WindowDataset(Dataset):
    def __init__(self, items: list[tuple[np.ndarray, float]], train: bool) -> None:
        self.items = items
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.items[index][0], self.items[index][1]
        image = _augment_gray(_from_cache_image(image), self.train)
        return torch.from_numpy(image).unsqueeze(0), torch.tensor(label, dtype=torch.float32)


class RankPairDataset(Dataset):
    """(full glyph, incomplete chop) pairs for margin ranking."""

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]], train: bool) -> None:
        self.pairs = pairs
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        full, half = self.pairs[index]
        full = _from_cache_image(full)
        half = _from_cache_image(half)
        # Same horizontal flip on both so the pair stays aligned.
        if self.train and random.random() < 0.5:
            full = np.ascontiguousarray(full[:, ::-1])
            half = np.ascontiguousarray(half[:, ::-1])
        if self.train and random.random() < 0.3:
            delta = random.uniform(-0.08, 0.08)
            full = np.clip(full + delta, 0, 1).astype(np.float32)
            half = np.clip(half + delta, 0, 1).astype(np.float32)
        if not full.flags["C_CONTIGUOUS"]:
            full = np.ascontiguousarray(full)
        if not half.flags["C_CONTIGUOUS"]:
            half = np.ascontiguousarray(half)
        return (
            torch.from_numpy(full).unsqueeze(0),
            torch.from_numpy(half).unsqueeze(0),
        )


class LetterCompletenessNet(nn.Module):
    """Small CNN: is this crop one complete Musnad glyph?"""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = print_device_info(resolve_device(args.cpu))
    need_cache = not CACHE_PATH.exists() or args.rebuild_cache
    if CACHE_PATH.exists() and not args.rebuild_cache:
        with CACHE_PATH.open("rb") as handle:
            preview = pickle.load(handle)
        if int(preview.get("version", -1)) != CACHE_VERSION:
            print(
                f"Cache version {preview.get('version')} != {CACHE_VERSION}; rebuilding...",
                flush=True,
            )
            need_cache = True
    if need_cache:
        build_window_cache(max_lines=args.max_lines)
    with CACHE_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    items = payload["items"]
    pairs = list(payload.get("pairs") or [])
    items = _balance_window_items(items, neg_ratio=args.neg_ratio, seed=args.seed)
    random.Random(args.seed).shuffle(items)
    split = max(1, int(0.1 * len(items)))
    val_items, train_items = items[:split], items[split:]
    print(
        f"train={len(train_items)}  val={len(val_items)}  rank_pairs={len(pairs)}",
        flush=True,
    )

    workers = args.workers
    if workers is None:
        # Windows + large in-memory datasets: worker processes pickle the whole
        # cache and can hang for minutes with no logs. Keep workers on the main
        # process unless the user asks otherwise.
        workers = 0
    loader_opts = dataloader_kwargs(device, num_workers=workers)
    train_loader = DataLoader(
        WindowDataset(train_items, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_opts,
    )
    val_loader = DataLoader(
        WindowDataset(val_items, train=False),
        batch_size=args.batch_size,
        **loader_opts,
    )
    pair_loader = None
    if pairs:
        pair_bs = max(16, min(args.batch_size, 256))
        pair_loader = DataLoader(
            RankPairDataset(pairs, train=True),
            batch_size=pair_bs,
            shuffle=True,
            **loader_opts,
        )

    model = LetterCompletenessNet().to(device)
    # Mild LR scale with batch size so larger batches stay stable.
    base_batch = 64
    lr = args.lr * max(1.0, args.batch_size / base_batch) ** 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rank_margin = 1.0
    rank_weight = 0.75
    print(
        f"batch={args.batch_size}  lr={lr:.4g}  amp={use_amp}  workers={workers}  "
        f"patience={args.patience}  rank_w={rank_weight}  rank_m={rank_margin}",
        flush=True,
    )

    def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Positives are rarer in some batches; mild pos weight keeps recall up.
        return nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=torch.tensor([1.4], device=logits.device)
        )

    def rank_loss_fn(full_logits: torch.Tensor, half_logits: torch.Tensor) -> torch.Tensor:
        # Enforce score(full) >= score(half) + margin on raw logits.
        target = torch.ones_like(full_logits)
        return nn.functional.margin_ranking_loss(
            full_logits, half_logits, target, margin=rank_margin
        )

    best = float("inf")
    stale = 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_batches = len(train_loader)
        report = max(1, min(50, n_batches // 20))
        print(
            f"epoch {epoch:02d}/{args.epochs} starting ({n_batches} batches)...",
            flush=True,
        )
        train_iter = iter(train_loader)
        pair_iter = iter(pair_loader) if pair_loader is not None else None
        print("  waiting for first batch...", flush=True)
        for step in range(1, n_batches + 1):
            images, labels = next(train_iter)
            if step == 1:
                print(f"  epoch {epoch:02d}  first batch OK, training...", flush=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(images), labels)
                if pair_iter is not None:
                    try:
                        fulls, halves = next(pair_iter)
                    except StopIteration:
                        pair_iter = iter(pair_loader)
                        fulls, halves = next(pair_iter)
                    fulls = fulls.to(device, non_blocking=True)
                    halves = halves.to(device, non_blocking=True)
                    # One forward on the stacked pair batch.
                    stacked = torch.cat([fulls, halves], dim=0)
                    logits = model(stacked)
                    n = fulls.size(0)
                    loss = loss + rank_weight * rank_loss_fn(logits[:n], logits[n:])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * images.size(0)
            if step % report == 0 or step == n_batches:
                print(
                    f"  epoch {epoch:02d}  batch {step}/{n_batches}  "
                    f"loss {total / max(1, step * args.batch_size):.4f}",
                    flush=True,
                )
        scheduler.step()
        train_loss = total / max(1, len(train_items))

        model.eval()
        total = correct = count = 0.0
        rank_ok = rank_n = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    batch_loss = loss_fn(logits, labels)
                total += batch_loss.item() * images.size(0)
                preds = (torch.sigmoid(logits.float()) >= 0.5).float()
                correct += (preds == labels).sum().item()
                count += images.size(0)
            if pair_loader is not None:
                # Spot-check ranking on a fixed subset of pairs (no aug).
                check = DataLoader(
                    RankPairDataset(pairs[: min(2048, len(pairs))], train=False),
                    batch_size=min(256, args.batch_size),
                    **loader_opts,
                )
                for fulls, halves in check:
                    fulls = fulls.to(device, non_blocking=True)
                    halves = halves.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        stacked = torch.cat([fulls, halves], dim=0)
                        logits = model(stacked).float()
                    n = fulls.size(0)
                    rank_ok += (logits[:n] > logits[n:] + 0.25).sum().item()
                    rank_n += n
        val_loss = total / max(1, len(val_items))
        acc = correct / max(1, count)
        rank_acc = rank_ok / max(1, rank_n) if rank_n else 0.0
        marker = ""
        if val_loss < best:
            best = val_loss
            stale = 0
            torch.save({"state_dict": model.state_dict()}, MODEL_PATH)
            marker = "  saved"
        else:
            stale += 1
        rank_msg = f"  rank {rank_acc*100:.1f}%" if rank_n else ""
        print(
            f"epoch {epoch:02d}/{args.epochs}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  acc {acc*100:.1f}%{rank_msg}{marker}",
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print(
                f"Early stop at epoch {epoch} (no val improvement for {args.patience})",
                flush=True,
            )
            break
    print(f"Best val loss {best:.4f} -> {MODEL_PATH}", flush=True)


def load_detector(device: torch.device) -> LetterCompletenessNet:
    model = LetterCompletenessNet().to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Train with:\n"
            "  python -m src.letter_detector --train --epochs 12"
        )
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference: boundary candidates + shape-aware keep/merge/drop
# ---------------------------------------------------------------------------


def score_crop(
    enhanced_gray: np.ndarray,
    left: int,
    right: int,
    model: LetterCompletenessNet,
    device: torch.device,
) -> float:
    """Completeness score for one window on an already-enhanced line."""
    return _score_crops_batch(enhanced_gray, [(left, right)], model, device)[0]


def _score_crops_batch(
    enhanced_gray: np.ndarray,
    spans: list[tuple[int, int]],
    model: LetterCompletenessNet,
    device: torch.device,
) -> list[float]:
    """Batched completeness scores. Same values as repeated ``score_crop``."""
    if not spans:
        return []
    tensors: list[torch.Tensor] = []
    valid: list[int] = []
    out = [0.0] * len(spans)
    for i, (left, right) in enumerate(spans):
        if right - left < 3:
            continue
        tensors.append(torch.from_numpy(prepare_window(enhanced_gray[:, left:right])).unsqueeze(0))
        valid.append(i)
    if not tensors:
        return out
    with torch.no_grad():
        for start in range(0, len(tensors), 128):
            batch = torch.stack(tensors[start : start + 128]).to(device)
            vals = torch.sigmoid(model(batch)).cpu().tolist()
            for idx, val in zip(valid[start : start + 128], vals):
                out[idx] = float(val)
    return out


def _activity_has_two_lobes(
    activity: np.ndarray,
    left: int,
    right: int,
    line_height: int,
) -> bool:
    """True when a box has two separated ink peaks (glued neighbours)."""
    sl = activity[left:right]
    if sl.size < 8:
        return False
    peak = float(sl.max()) + 1e-6
    min_sep = max(4, int(0.10 * line_height))
    found: list[int] = []
    for i in range(1, sl.size - 1):
        if not (
            float(sl[i]) >= float(sl[i - 1])
            and float(sl[i]) >= float(sl[i + 1])
            and float(sl[i]) >= 0.50 * peak
        ):
            continue
        if not found or i - found[-1] >= min_sep:
            found.append(i)
        elif float(sl[i]) > float(sl[found[-1]]):
            found[-1] = i
    return len(found) >= 2


def _seam_is_between_two_bodies(
    activity: np.ndarray,
    left: int,
    seam: int,
    right: int,
) -> bool:
    """
    True when the seam is a valley between two ink bodies.

    A 𐩥 middle bar is a *peak* inside one letter. A word ``|`` after 𐩥 sits
    past a valley — that pair must stay two letters (test-1 L2 annotation).
    """
    n = int(activity.size)
    if n < 4 or right - left < 6:
        return False
    s0 = min(max(int(seam), left + 1), right - 2)
    s0 = min(max(s0, 1), n - 2)
    act_s = float(activity[s0])
    left_sl = activity[left:s0]
    right_sl = activity[s0:right]
    if left_sl.size == 0 or right_sl.size == 0:
        return False
    left_peak = float(left_sl.max())
    right_peak = float(right_sl.max())
    if min(left_peak, right_peak) < 0.16:
        return False
    # Quiet compared with both bodies → letter gap, not an interior stem.
    if act_s <= 0.55 * min(left_peak, right_peak):
        return True
    if (
        act_s <= float(activity[max(s0 - 2, 0)])
        and act_s <= float(activity[min(s0 + 2, n - 1)])
        and act_s <= 0.70 * min(left_peak, right_peak)
    ):
        return True
    return False


def score_windows(
    line_bgr: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    *,
    step: int = 4,
) -> dict[tuple[int, int], float]:
    """Score every plausible glyph-width window on the line."""
    enhanced = enhance_line(line_bgr)
    h, w = enhanced.shape[:2]
    w_min = max(8, int(0.18 * h))
    w_max = min(w, int(1.35 * h))
    starts = list(range(0, max(1, w - w_min + 1), step))
    pairs: list[tuple[int, int]] = []
    tensors: list[torch.Tensor] = []
    for a in starts:
        for width in range(w_min, w_max + 1, step):
            b = a + width
            if b > w:
                break
            tensors.append(torch.from_numpy(prepare_window(enhanced[:, a:b])).unsqueeze(0))
            pairs.append((a, b))

    scores: dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for i in range(0, len(tensors), 256):
            batch = torch.stack(tensors[i : i + 256]).to(device)
            conf = torch.sigmoid(model(batch)).cpu().tolist()
            for pair, c in zip(pairs[i : i + 256], conf):
                scores[pair] = float(c)
    return scores


def _candidate_cuts(
    line_bgr: np.ndarray,
    device: torch.device,
    *,
    source_height: int | None = None,
) -> tuple[list[int], np.ndarray | None, np.ndarray | None]:
    """
    Prefer the trained boundary net's cuts as candidate edges.

    Returns ``(cuts, boundary_profile_or_None, objectness_or_None)``.
    """
    try:
        from .letter_boundary_net import (
            boundaries_from_profile,
            load_boundary_model,
            predict_profiles,
        )

        boundary_model = load_boundary_model(device)
        profile, objectness = predict_profiles(line_bgr, boundary_model, device)
        cuts = boundaries_from_profile(
            profile, line_bgr.shape[0], source_height=source_height
        )
        from .letter_boundary_net import suppress_empty_segments

        cuts = suppress_empty_segments(line_bgr, cuts, objectness=objectness)
        return cuts, profile, objectness
    except SystemExit:
        h, w = line_bgr.shape[:2]
        grid = max(6, h // 4)
        return list(range(grid, w, grid)), None, None


def _drop_mid_glyph_boundary_cuts(
    cuts: list[int],
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    stroke_mask: np.ndarray,
    line_height: int,
    line_width: int,
    boundary_profile: np.ndarray | None = None,
) -> list[int]:
    """
    Remove close-pair peaks that slice one complete letter into weak halves.

    Peak NMS keeps separator-thin pairs (``|``). A false mid-glyph peak
    (test-27 L1 @429) looks the same in the curve but bisects a connected
    stroke while the merged crop scores as one letter.

    On dense short lines, two packed glyphs often also score as one "complete"
    window (test-3). Never drop annotation-strength peaks, and only drop when
    *both* halves look incomplete — not when one side is already a solid letter.
    """
    if len(cuts) < 2 or line_height <= 0:
        return cuts
    min_part = max(4, int(0.10 * line_height))
    thin_lim = max(12, int(0.28 * line_height))
    edges = [0] + sorted(c for c in cuts if 0 < c < line_width) + [line_width]
    drop: set[int] = set()
    for i in range(1, len(edges) - 1):
        left, cut, right = edges[i - 1], edges[i], edges[i + 1]
        if cut in drop:
            continue
        if right - left < 2 * min_part + 2:
            continue
        # Only reconsider thin mid-slices (separator-pair geometry).
        if min(cut - left, right - cut) > thin_lim:
            continue
        # Strong boundary-net peaks are real seams on dense tablets (test-3).
        if _boundary_cut_strength(boundary_profile, cut) >= 0.82:
            continue
        if not _cut_bisects_connected_stroke(stroke_mask, cut):
            continue
        left_s = score_crop(enhanced, left, cut, model, device)
        right_s = score_crop(enhanced, cut, right, model, device)
        merge_s = score_crop(enhanced, left, right, model, device)
        # Both halves must look incomplete; a strong half means a real letter
        # boundary with a weak neighbour scrap, not a mid-glyph false peak.
        if max(left_s, right_s) >= 0.48:
            continue
        # Merged glyph clearly better than either half → false mid cut.
        if merge_s >= max(left_s, right_s) + 0.18 and merge_s >= 0.55:
            drop.add(cut)
            continue
        if merge_s >= 0.70 and max(left_s, right_s) < 0.42:
            drop.add(cut)
    return [c for c in cuts if c not in drop]


def _drop_ring_center_boundary_cuts(
    cuts: list[int],
    line_bgr: np.ndarray,
    activity: np.ndarray,
    line_height: int,
    line_width: int,
    stroke_mask: np.ndarray | None = None,
    *,
    dense: bool = False,
) -> list[int]:
    """
    Remove a boundary peak that sits inside 𐩲 (hole) or 𐩥 (middle bar).

    Completeness often scores both arcs as complete (~1.0), so
    ``_drop_mid_glyph_boundary_cuts`` keeps the cut. On dense tablets (test-3)
    the hole also looks like a packed-letter gap.
    """
    if len(cuts) < 1 or line_height <= 0 or line_width <= 0:
        return cuts
    edges = [0] + sorted(c for c in cuts if 0 < c < line_width) + [line_width]
    drop: set[int] = set()
    for i in range(1, len(edges) - 1):
        left, cut, right = edges[i - 1], edges[i], edges[i + 1]
        if _looks_like_round_letter_mid_cut(
            line_bgr,
            activity,
            stroke_mask,
            left,
            cut,
            right,
            line_height,
            loose=True,
            dense=dense,
        ):
            drop.add(cut)
    return [c for c in cuts if c not in drop]


def _drop_spurious_dense_cuts(
    cuts: list[int],
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    line_width: int,
    *,
    ref_h: int,
    boundary_profile: np.ndarray | None = None,
) -> list[int]:
    """
    On short noisy rows (test-38), drop interior boundary ripples inside a
    span that reads better as one letter than as two weak halves.
    """
    if ref_h >= 72 or len(cuts) < 2 or line_width <= 0:
        return cuts
    pitch_est = line_width / max(len(cuts) + 1, 1)
    min_wide = max(24, int(1.10 * pitch_est))
    edges = [0] + sorted(c for c in cuts if 0 < c < line_width) + [line_width]
    drop: set[int] = set()
    for i in range(1, len(edges) - 1):
        left, cut, right = edges[i - 1], edges[i], edges[i + 1]
        if cut in drop or right - left < min_wide:
            continue
        if _boundary_cut_strength(boundary_profile, cut) >= 0.72:
            continue
        merge_s = score_crop(enhanced, left, right, model, device)
        left_s = score_crop(enhanced, left, cut, model, device)
        right_s = score_crop(enhanced, cut, right, model, device)
        if merge_s >= max(left_s, right_s) + 0.10 and merge_s >= 0.42:
            drop.add(cut)
    return [c for c in cuts if c not in drop]


def _drop_thin_scrap_dense_cuts(
    cuts: list[int],
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    line_width: int,
    stroke_mask: np.ndarray,
    *,
    ref_h: int,
    boundary_profile: np.ndarray | None = None,
) -> list[int]:
    """
    Remove strong boundary ripples that only peel a thin empty scrap off a
    solid neighbour (test-38 L3 @116 inside one wide glyph).
    """
    if ref_h >= 72 or len(cuts) < 1 or line_width <= 0:
        return cuts
    edges = [0] + sorted(c for c in cuts if 0 < c < line_width) + [line_width]
    drop: set[int] = set()
    for i in range(1, len(edges) - 1):
        left, cut, right = edges[i - 1], edges[i], edges[i + 1]
        parent = right - left
        if parent < 12:
            continue
        if _boundary_cut_strength(boundary_profile, cut) >= 0.92:
            continue
        if _clear_stroke_gutter(stroke_mask, cut):
            continue
        w_l = cut - left
        w_r = right - cut
        thin = min(w_l, w_r)
        if thin / parent > 0.34:
            continue
        left_s = score_crop(enhanced, left, cut, model, device)
        right_s = score_crop(enhanced, cut, right, model, device)
        if max(left_s, right_s) < 0.62:
            continue
        if min(left_s, right_s) >= 0.34:
            continue
        drop.add(cut)
    return [c for c in cuts if c not in drop]


def _filter_inactive_boundary_cuts(
    cuts: list[int],
    activity: np.ndarray,
    line_width: int,
    line_height: int,
    *,
    min_side_peak: float = 0.18,
    boundary_profile: np.ndarray | None = None,
) -> list[int]:
    """
    Drop boundary peaks that sit on bare stone (test-38 L5 empty margin).

    Real letter boundaries have carved stroke evidence on both sides; texture
    ripples in blank areas do not.
    """
    if not cuts or activity.size == 0 or line_height <= 0:
        return cuts
    band = max(6, int(0.22 * line_height))
    kept: list[int] = []
    for c in cuts:
        if not (0 < c < line_width):
            continue
        if boundary_profile is not None and _boundary_cut_strength(boundary_profile, c) >= 0.48:
            kept.append(int(c))
            continue
        lo = max(0, c - band)
        hi = min(line_width, c + band)
        left_sl = activity[lo:c]
        right_sl = activity[c:hi]
        left_peak = float(left_sl.max()) if left_sl.size else 0.0
        right_peak = float(right_sl.max()) if right_sl.size else 0.0
        if min(left_peak, right_peak) >= min_side_peak:
            kept.append(int(c))
            continue
        # Keep edge cuts when one side is the line margin and the other is ink.
        if c <= band + 2 and right_peak >= min_side_peak:
            kept.append(int(c))
        elif line_width - c <= band + 2 and left_peak >= min_side_peak:
            kept.append(int(c))
    return sorted(kept)


def _snap_boundary_cuts_by_completeness(
    cuts: list[int],
    peak_candidates: list[int],
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    line_height: int,
    line_width: int,
    *,
    min_gain: float = 0.18,
) -> list[int]:
    """
    Move each boundary cut to the nearby peak that best splits its neighbors.

    Peak NMS often keeps a strong mid-glyph ripple (test-27 L1 @111) over the
    weaker true seam (@91). Completeness of the left/right crops recovers the
    annotated location without inventing new cuts.

    Only snap when the alternate is clearly better (``min_gain``) so already-
    good peaks on clean lines are not dragged onto nearby ripples.
    """
    if not cuts or line_height <= 0 or line_width <= 0:
        return cuts
    search = max(10, int(0.30 * line_height))
    min_part = max(4, int(0.10 * line_height))
    working = [0] + sorted(c for c in cuts if 0 < c < line_width) + [line_width]
    for i in range(1, len(working) - 1):
        left, cut, right = working[i - 1], working[i], working[i + 1]
        if right - left < 2 * min_part + 2:
            continue
        lo = max(left + min_part, cut - search)
        hi = min(right - min_part, cut + search)
        local = [cut]
        for p in peak_candidates:
            if lo <= int(p) <= hi:
                local.append(int(p))
        local = sorted(set(local))
        if len(local) == 1:
            continue
        cur_s = score_crop(enhanced, left, cut, model, device) + score_crop(
            enhanced, cut, right, model, device
        )
        best_c = cut
        best_s = cur_s
        for c in local:
            if c == cut:
                continue
            if not (left + min_part <= c <= right - min_part):
                continue
            s = score_crop(enhanced, left, c, model, device) + score_crop(
                enhanced, c, right, model, device
            )
            if s > best_s + 1e-6:
                best_s = s
                best_c = c
        if best_c != cut and best_s >= cur_s + min_gain:
            working[i] = int(best_c)
    out: list[int] = []
    for c in working[1:-1]:
        if 0 < c < line_width and (not out or c > out[-1] + 2):
            out.append(c)
    return out


def _add_validated_interior_boundary_cuts(
    cuts: list[int],
    profile: np.ndarray,
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    line_height: int,
    line_width: int,
    *,
    min_span_frac: float = 0.26,
    min_profile: float = 0.42,
    min_gain: float = 0.08,
) -> list[int]:
    """
    Insert missed letter seams inside spans that are wide enough for two glyphs.

    Dense short lines (test-3) often suppress a true cut between two stronger
    peaks, so NMS never offers it. Only add an interior column when the
    boundary curve is elevated there *and* left+right completeness beats the
    merged crop — avoids re-opening open-center letters (test-27 @429).
    """
    if profile is None or profile.size == 0 or line_height <= 0:
        return cuts
    min_span = max(14, int(min_span_frac * line_height))
    min_part = max(3, int(0.08 * line_height))
    smooth = cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=2.0
    ).ravel()
    edges = [0] + sorted(c for c in cuts if 0 < c < line_width) + [line_width]
    extra: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right - left < min_span:
            continue
        lo = left + min_part
        hi = right - min_part
        if hi <= lo:
            continue
        region = smooth[lo : hi + 1]
        if region.size == 0:
            continue
        merge_s = score_crop(enhanced, left, right, model, device)
        # Rank local maxima + the global argmax inside the span.
        cands: list[int] = [lo + int(np.argmax(region))]
        for x in range(lo, hi + 1):
            if x <= 0 or x >= line_width - 1:
                continue
            if float(smooth[x]) < min_profile:
                continue
            if float(smooth[x]) >= float(smooth[x - 1]) and float(smooth[x]) >= float(
                smooth[x + 1]
            ):
                cands.append(x)
        # Dense packing often leaves only a shoulder, not a peak — probe the
        # highest column in each third of the span, plus a fine scan when the
        # span is short (test-3 packed pairs).
        span = hi - lo
        for frac in (0.33, 0.50, 0.67):
            x = lo + int(frac * span)
            if lo <= x <= hi and float(smooth[x]) >= min_profile:
                cands.append(x)
        if right - left <= max(28, int(0.42 * line_height)):
            step = 2 if (hi - lo) > 16 else 1
            for x in range(lo, hi + 1, step):
                if float(smooth[x]) >= min_profile:
                    cands.append(x)
        ranked = sorted(set(cands), key=lambda x: -float(smooth[x]))
        best_cut: int | None = None
        best_gain = min_gain - 1e-6
        min_piece = max(5, int(0.10 * line_height))
        for cut in ranked[:10]:
            if float(smooth[cut]) < min_profile:
                continue
            if (cut - left) < min_piece or (right - cut) < min_piece:
                continue
            left_s = score_crop(enhanced, left, cut, model, device)
            right_s = score_crop(enhanced, cut, right, model, device)
            gain = left_s + right_s - merge_s
            if gain < min_gain:
                continue
            # Both sides must look like letter pieces, not a mid-glyph scrap.
            if min(left_s, right_s) < 0.28:
                continue
            if gain > best_gain:
                best_gain = gain
                best_cut = int(cut)
        if best_cut is not None:
            extra.append(best_cut)
    if not extra:
        return cuts
    return sorted(set(cuts) | set(extra))


def _boundary_cut_strength(
    profile: np.ndarray | None,
    cut_x: int,
    *,
    radius: int = 3,
) -> float:
    """Peak boundary probability near ``cut_x`` (0 if no profile)."""
    if profile is None or profile.size == 0:
        return 0.0
    w = int(profile.shape[0])
    if w < 2:
        return 0.0
    x0 = max(0, int(cut_x) - radius)
    x1 = min(w, int(cut_x) + radius + 1)
    if x1 <= x0:
        return 0.0
    return float(np.max(profile[x0:x1]))


def _span_crosses_strong_boundary(
    edges: list[int],
    i: int,
    j: int,
    profile: np.ndarray | None,
    *,
    strong: float = 0.48,
    stroke_mask: np.ndarray | None = None,
    activity: np.ndarray | None = None,
    line_height: int = 0,
    allow_round_hole: bool = False,
) -> bool:
    """
    True when a multi-atom span would glue across a confident boundary peak.

    Annotation / boundary-net cuts are already good on real lines; the global
    decoder must not erase them just because a merged crop scores well.
    Interior stems of a connected glyph (𐩥 middle bar) are not letter cuts.
    """
    if profile is None or j <= i + 1:
        return False
    span_w = edges[j] - edges[i]
    one_letter = line_height > 0 and span_w <= max(int(0.72 * line_height), 48)
    for k in range(i + 1, j):
        cut = edges[k]
        if _boundary_cut_strength(profile, cut) < strong:
            continue
        if (
            one_letter
            and stroke_mask is not None
            and _cut_bisects_connected_stroke(stroke_mask, cut)
            and not _clear_stroke_gutter(stroke_mask, cut)
        ):
            if activity is not None and _seam_is_between_two_bodies(
                activity, edges[i], cut, edges[j]
            ):
                if allow_round_hole and _stroke_column_is_round_mid(
                    stroke_mask, edges[i], cut, edges[j]
                ):
                    if _is_round_plus_separator(
                        activity,
                        stroke_mask,
                        edges[i],
                        cut,
                        edges[j],
                        line_height,
                    ):
                        return True
                    continue
                return True
            continue
        return True
    return False


def _add_deep_activity_cuts(
    cuts: list[int],
    activity: np.ndarray,
    line_height: int,
    line_width: int,
) -> list[int]:
    """Rescue missed cuts inside wide spans with a very clear blank valley."""
    if line_height <= 0 or line_width <= 0 or activity.size == 0:
        return cuts
    edges = [0] + sorted(c for c in cuts if 0 < c < line_width) + [line_width]
    extra: list[int] = []
    min_span = max(38, int(0.48 * line_height))
    for left, right in zip(edges[:-1], edges[1:]):
        width = right - left
        if width < min_span:
            continue
        pad = max(4, int(0.18 * width))
        lo, hi = left + pad, right - pad
        if hi <= lo:
            continue
        span = activity[left:right]
        inner = activity[lo:hi]
        if span.size == 0 or inner.size == 0:
            continue
        cut = lo + int(inner.argmin())
        valley = float(activity[cut])
        left_peak = float(activity[left:cut].max()) if cut > left else 0.0
        right_peak = float(activity[cut:right].max()) if right > cut else 0.0
        mean = float(span.mean())
        # Very conservative: both sides must contain strong carving and the
        # valley must be close to bare stone. DP can still merge if needed.
        if (
            valley <= 0.08
            and valley <= 0.35 * mean
            and min(left_peak, right_peak) >= 0.24
            and all(abs(cut - c) > 3 for c in cuts)
        ):
            extra.append(cut)
    if not extra:
        return cuts
    return sorted(set(cuts + extra))


def _ink_profile_for_cuts(
    activity: np.ndarray,
    left: int,
    right: int,
    stroke_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    1-D profile for finding stem peaks / gutters inside a crop.

    On dense small stone lines, letter-activity can wash out while the carved
    stroke mask still shows clear parallel stems and valleys — prefer stroke
    density when activity is weak.
    """
    if right <= left:
        return np.zeros(0, dtype=np.float64)
    act = np.asarray(activity[left:right], dtype=np.float64)
    if stroke_mask is None or stroke_mask.size == 0:
        return act
    col = stroke_mask[:, left:right].sum(axis=0).astype(np.float64)
    if col.size == 0 or float(col.max()) <= 0:
        return act
    col_n = col / float(col.max())
    act_max = float(act.max()) if act.size else 0.0
    if act_max < 0.40:
        return col_n
    act_n = act / max(act_max, 1e-6)
    return 0.40 * act_n + 0.60 * col_n


def _activity_valley_cuts(
    activity: np.ndarray,
    left: int,
    right: int,
    *,
    line_height: int,
    pitch: float,
    stroke_mask: np.ndarray | None = None,
) -> list[int]:
    """
    Candidate cuts at deep valleys between ink peaks inside a crop.

    Smart multi-glyph signal: one Musnad letter is ~1 pitch wide; a pack of
    parallel stems with a gutter between peaks is usually two glyphs (or
    letter + ``|``), not one.
    """
    if activity.size == 0 or right - left < 8 or line_height <= 0:
        return []
    sl = _ink_profile_for_cuts(activity, left, right, stroke_mask=stroke_mask)
    if sl.size < 8:
        return []
    k = max(3, (int(0.05 * line_height) | 1))
    kernel = np.ones(k, dtype=np.float64) / float(k)
    sm = np.convolve(sl, kernel, mode="same")
    peak_floor = max(0.18, min(0.42, 0.50 * float(sm.max())))
    min_sep = max(3, int(0.08 * line_height))
    peaks: list[int] = []
    for i in range(2, len(sm) - 2):
        if sm[i] < peak_floor:
            continue
        if sm[i] < sm[i - 1] or sm[i] < sm[i + 1]:
            continue
        if peaks and i - peaks[-1] < min_sep:
            if sm[i] > sm[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    if len(peaks) < 2:
        return []

    min_part = max(4, int(0.10 * line_height))
    # Prefer valleys that leave each side roughly letter-sized when possible.
    cuts: list[tuple[float, int]] = []
    for a, b in zip(peaks, peaks[1:]):
        if b <= a + 1:
            continue
        valley_rel = a + int(np.argmin(sm[a : b + 1]))
        valley = float(sm[valley_rel])
        peak = min(float(sm[a]), float(sm[b]))
        if peak < peak_floor or valley > 0.70 * peak:
            continue
        if valley_rel < min_part or (len(sm) - valley_rel) < min_part:
            continue
        left_w = float(valley_rel)
        right_w = float(len(sm) - valley_rel)
        # Stronger when both sides look like a glyph (or thin ``|`` + glyph).
        side_ok = (
            (0.28 * pitch <= left_w <= 1.30 * pitch and 0.28 * pitch <= right_w <= 1.30 * pitch)
            or (min(left_w, right_w) <= 0.35 * pitch and max(left_w, right_w) >= 0.35 * pitch)
            or (left_w >= 0.22 * pitch and right_w >= 0.22 * pitch)
            or (left_w >= min_part and right_w >= min_part and peak >= 0.35)
        )
        if not side_ok:
            continue
        depth = (peak - valley) / max(peak, 1e-6)
        cuts.append((depth, left + valley_rel))
    cuts.sort(reverse=True)
    return [c for _d, c in cuts]


def _split_low_conf_wide_boxes(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    line_height: int,
    stroke_mask: np.ndarray | None = None,
    activity: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """
    Split oversized / multi-peak crops into separate glyphs.

    Smart geometry (not a per-image hardcode):
      1) wide + low completeness
      2) parallel stem peaks with a stroke/activity gutter between them —
         even when completeness is high (`|||` packs often score ≥0.8)
    Connected multi-stem letters (crossbars) stay intact via stroke continuity.
    """
    if not boxes or line_height <= 0:
        return boxes
    out: list[tuple[int, int, float]] = []
    min_part = max(4, int(0.10 * line_height))
    min_width = max(28, int(0.40 * line_height))
    pitch = _estimate_letter_pitch(boxes, line_height)

    def _parallel_stem_pack(left: int, right: int, cut: int) -> bool:
        """True when both sides of the cut look like vertical bars, not glyph halves."""
        w1 = cut - left
        w2 = right - cut
        if w1 < 2 or w2 < 2 or line_height <= 0:
            return False
        r1 = _vertical_stem_ratio(line_bgr[:, left:cut])
        r2 = _vertical_stem_ratio(line_bgr[:, cut:right])
        both_stems = r1 >= 1.35 and r2 >= 1.35
        # Dense small lines (tablet rows) have relatively wider ``|`` bars.
        thin_lim = 0.32 if line_height < 80 else 0.26
        both_thin = w1 / line_height <= thin_lim and w2 / line_height <= thin_lim
        thin_plus = (
            min(w1, w2) / line_height <= thin_lim
            and max(w1, w2) / line_height <= 0.55
            and max(w1, w2) >= 1.30 * min(w1, w2)
        )
        clear = stroke_mask is None or _clear_stroke_gutter(stroke_mask, cut)
        return bool(clear and both_stems and (both_thin or thin_plus))

    def _try_cut(
        left: int,
        right: int,
        conf: float,
        cut: int,
        *,
        need_gain: float,
    ) -> tuple[int, float, float] | None:
        if not (left + min_part <= cut <= right - min_part):
            return None
        if stroke_mask is not None and _cut_bisects_connected_stroke(stroke_mask, cut):
            if (right - left) <= 1.40 * pitch:
                return None
        left_score = score_crop(enhanced, left, cut, model, device)
        right_score = score_crop(enhanced, cut, right, model, device)
        if min(left_score, right_score) < 0.22:
            return None
        gain = (left_score + right_score) / 2.0 - conf
        if gain < need_gain and not (
            need_gain <= 0.05
            and min(left_score, right_score) >= 0.35
            and max(left_score, right_score) >= 0.50
        ):
            return None
        return cut, left_score, right_score

    def _force_valley_cut(
        left: int, right: int, cut: int
    ) -> tuple[int, float, float] | None:
        if not (left + min_part <= cut <= right - min_part):
            return None
        if not _parallel_stem_pack(left, right, cut):
            return None
        if stroke_mask is not None and _cut_bisects_connected_stroke(stroke_mask, cut):
            if (right - left) <= 1.40 * pitch:
                return None
        left_score = max(0.35, score_crop(enhanced, left, cut, model, device))
        right_score = max(0.35, score_crop(enhanced, cut, right, model, device))
        return cut, left_score, right_score

    for left, right, conf in boxes:
        width = right - left
        valley_cuts: list[int] = []
        if activity is not None:
            valley_cuts = _activity_valley_cuts(
                activity,
                left,
                right,
                line_height=line_height,
                pitch=pitch,
                stroke_mask=stroke_mask,
            )
        oversized = width >= max(1.15 * pitch, 0.48 * line_height)
        low_conf_wide = width >= min_width and conf < 0.62
        multi_peak = len(valley_cuts) >= 1 and width >= max(12, int(0.26 * line_height))
        if not oversized and not low_conf_wide and not multi_peak:
            out.append((left, right, conf))
            continue
        if (
            width <= 1.20 * pitch
            and stroke_mask is not None
            and not multi_peak
            and _box_is_single_stroke_body(stroke_mask, left, right)
        ):
            out.append((left, right, conf))
            continue
        crop = line_bgr[:, left:right]
        if _looks_like_small_ring(crop) or _radial_hollow_ring(crop, strict=True):
            out.append((left, right, conf))
            continue

        chosen: tuple[int, float, float] | None = None
        if valley_cuts:
            for cut in valley_cuts:
                # Aggressive only for parallel vertical-stem packs; complex
                # letters need completeness to actually prefer the split.
                if _parallel_stem_pack(left, right, cut):
                    need = -0.25
                elif oversized or low_conf_wide:
                    need = 0.05 if conf < 0.70 else 0.12
                else:
                    need = 0.10
                hit = _try_cut(left, right, conf, cut, need_gain=need)
                if hit is not None:
                    chosen = hit
                    break
            if chosen is None and multi_peak:
                for cut in valley_cuts:
                    hit = _force_valley_cut(left, right, cut)
                    if hit is not None:
                        chosen = hit
                        break

        if chosen is None and (low_conf_wide or oversized):
            best: tuple[float, int, float, float] | None = None
            parallel_stem_split: tuple[float, int, float, float] | None = None
            need_gain = 0.12 if oversized and conf >= 0.70 else 0.22
            lo = left + min_part
            hi = right - min_part
            for cut in range(lo, hi + 1, 2):
                if stroke_mask is not None and _cut_bisects_connected_stroke(
                    stroke_mask, cut
                ):
                    if width <= 1.40 * pitch:
                        continue
                left_score = score_crop(enhanced, left, cut, model, device)
                right_score = score_crop(enhanced, cut, right, model, device)
                if min(left_score, right_score) < 0.35:
                    continue
                gain = (left_score + right_score) / 2.0 - conf
                if gain < need_gain:
                    continue
                score = gain + 0.05 * min(left_score, right_score)
                if (
                    parallel_stem_split is None
                    and min(
                        _vertical_stem_ratio(line_bgr[:, left:cut]),
                        _vertical_stem_ratio(line_bgr[:, cut:right]),
                    )
                    >= 1.45
                ):
                    parallel_stem_split = (score, cut, left_score, right_score)
                if best is None or score > best[0]:
                    best = (score, cut, left_score, right_score)
            pick = parallel_stem_split or best
            if pick is not None:
                _s, cut, ls, rs = pick
                chosen = (cut, ls, rs)

        if chosen is None:
            out.append((left, right, conf))
            continue
        cut, left_score, right_score = chosen
        stack = [(left, cut, left_score), (cut, right, right_score)]
        while stack:
            a, b, c = stack.pop()
            child_w = b - a
            child_cuts: list[int] = []
            if activity is not None and child_w >= max(12, int(0.26 * line_height)):
                child_cuts = _activity_valley_cuts(
                    activity,
                    a,
                    b,
                    line_height=line_height,
                    pitch=pitch,
                    stroke_mask=stroke_mask,
                )
            still_pack = len(child_cuts) >= 1 or child_w >= max(
                1.20 * pitch, 0.50 * line_height
            )
            if not still_pack:
                out.append((a, b, c))
                continue
            child_cut = None
            for cand in child_cuts:
                if _parallel_stem_pack(a, b, cand):
                    need = -0.25
                else:
                    need = 0.10
                hit = _try_cut(a, b, c, cand, need_gain=need)
                if hit is not None:
                    child_cut = hit
                    break
                hit = _force_valley_cut(a, b, cand)
                if hit is not None:
                    child_cut = hit
                    break
            if child_cut is None:
                out.append((a, b, c))
                continue
            mid, ls, rs = child_cut
            stack.append((a, mid, ls))
            stack.append((mid, b, rs))
    out.sort(key=lambda t: t[0])
    return out

def _vertical_stem_ratio(crop_bgr: np.ndarray) -> float:
    """How much a crop behaves like a mostly vertical stem."""
    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return float(gx.mean() / (gy.mean() + 1e-6))


def _score_span(
    enhanced: np.ndarray,
    line_bgr: np.ndarray,
    left: int,
    right: int,
    model: LetterCompletenessNet,
    device: torch.device,
    cnn: torch.nn.Module | None,
    cnn_weight: float,
) -> float:
    tensor = torch.from_numpy(prepare_window(enhanced[:, left:right])).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        shape = float(torch.sigmoid(model(tensor)).item())
    if cnn is not None and cnn_weight > 0:
        cnn_score = _cnn_completeness(line_bgr, left, right, cnn, device)
        shape = (1.0 - cnn_weight) * shape + cnn_weight * cnn_score
    return shape


def _cnn_predict_crop(
    crop_bgr: np.ndarray,
    left: int,
    right: int,
    cnn: torch.nn.Module,
    device: torch.device,
    index_to_label: list[str] | None = None,
) -> tuple[str | None, float]:
    """Frozen ``musnad_final.pth`` top label + confidence for one span."""
    from PIL import Image

    from model import _to_model_tensor
    from .predict import prepare_stone_view

    piece = crop_bgr[:, left:right]
    if piece.size == 0 or right - left < 2:
        return None, 0.0
    pil = Image.fromarray(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
    tensor = _to_model_tensor(prepare_stone_view(pil)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(cnn(tensor), dim=1)[0]
        conf, idx = probs.max(dim=0)
        conf_f = float(conf.item())
        i = int(idx.item())
    if index_to_label and 0 <= i < len(index_to_label):
        return str(index_to_label[i]), conf_f
    return str(i), conf_f


def _cnn_completeness(
    crop_bgr: np.ndarray,
    left: int,
    right: int,
    cnn: torch.nn.Module,
    device: torch.device,
) -> float:
    """Frozen ``musnad_final.pth`` max-softmax as a completeness aid."""
    _label, conf = _cnn_predict_crop(crop_bgr, left, right, cnn, device)
    return conf


_RECOGNITION_BUNDLE_CACHE: dict[
    str, tuple[torch.nn.Module, list[str], dict | None]
] = {}


def _load_recognition_bundle(
    device: torch.device,
) -> tuple[torch.nn.Module, list[str], dict | None] | None:
    """Load the recognizer + labels + prototype bank once per device."""
    key = str(device)
    if key in _RECOGNITION_BUNDLE_CACHE:
        return _RECOGNITION_BUNDLE_CACHE[key]
    try:
        from model import load_model
        from .predict import DEFAULT_CHECKPOINT, PROTOTYPES_PATH
        from prototypes import load_prototypes

        model, ckpt = load_model(MUSNAD_FINAL_PATH, device)
        model.eval()
        labels = [str(x) for x in ckpt.get("index_to_char", [])]
        bank = load_prototypes(PROTOTYPES_PATH)
        bundle = (model, labels, bank)
        _RECOGNITION_BUNDLE_CACHE[key] = bundle
        return bundle
    except Exception:
        return None


def _load_cnn_scorer(device: torch.device) -> torch.nn.Module | None:
    bundle = _load_recognition_bundle(device)
    return bundle[0] if bundle is not None else None


def _batch_recognition_span_scores(
    line_bgr: np.ndarray,
    spans: list[tuple[int, int]],
    bundle: tuple[torch.nn.Module, list[str], dict | None] | None,
    device: torch.device,
) -> list[dict]:
    """
    Batch recognition evidence for segmentation candidate spans.

    This deliberately uses CNN margin, entropy and prototype agreement rather
    than raw max-softmax alone: the recognizer is closed-set and can otherwise
    be confident on fragments or multi-letter crops.
    """
    if not spans:
        return []
    if bundle is None:
        return [
            {
                "trust": 0.0,
                "label": None,
                "cnn_conf": 0.0,
                "cnn_margin": 0.0,
                "proto_sim": 0.0,
                "proto_margin": 0.0,
                "agreement": False,
            }
            for _ in spans
        ]

    from PIL import Image

    from model import _to_model_tensor
    from .predict import prepare_stone_view

    cnn, labels, bank = bundle
    line_id = id(line_bgr)
    use_cache = line_id == _REC_SPAN_CACHE_LINE_ID
    cached_out: list[dict | None] = [None] * len(spans)
    missing_pos: list[int] = []
    work_spans: list[tuple[int, int]]
    if use_cache:
        for i, sp in enumerate(spans):
            hit = _REC_SPAN_CACHE.get(sp)
            if hit is not None:
                cached_out[i] = hit
            else:
                missing_pos.append(i)
        work_spans = [spans[i] for i in missing_pos]
        if not work_spans:
            return [rec for rec in cached_out if rec is not None]
    else:
        work_spans = list(spans)

    tensors: list[torch.Tensor] = []
    for left, right in work_spans:
        piece = line_bgr[:, left:right]
        if piece.size == 0:
            tensors.append(torch.full((1, 128, 128), 0.7, dtype=torch.float32))
            continue
        pil = Image.fromarray(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
        tensors.append(_to_model_tensor(prepare_stone_view(pil)))

    proto_tensor: torch.Tensor | None = None
    proto_labels: list[str] = []
    proto_counts: torch.Tensor | None = None
    if bank is not None:
        raw_proto = bank.get("prototypes")
        if isinstance(raw_proto, torch.Tensor) and raw_proto.ndim == 2:
            proto_tensor = nn.functional.normalize(
                raw_proto.to(device=device, dtype=torch.float32), p=2, dim=1
            )
            proto_labels = [str(x) for x in bank.get("index_to_char", [])]
            counts = bank.get("counts")
            if isinstance(counts, torch.Tensor):
                proto_counts = counts.to(device)

    results: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(tensors), 128):
            batch = torch.stack(tensors[start : start + 128]).to(
                device, non_blocking=device.type == "cuda"
            )
            if hasattr(cnn, "extract_features") and hasattr(cnn, "classifier"):
                features = cnn.extract_features(batch)
                logits = cnn.classifier(features)
            else:
                logits = cnn(batch)
                features = None

            probs = torch.softmax(logits.float(), dim=1)
            top = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
            p1 = top.values[:, 0]
            p2 = top.values[:, 1] if probs.shape[1] > 1 else torch.zeros_like(p1)
            pred_idx = top.indices[:, 0]
            entropy = -torch.sum(
                probs * torch.log(torch.clamp(probs, min=1e-8)), dim=1
            ) / max(float(np.log(max(2, probs.shape[1]))), 1e-6)

            proto_sims: torch.Tensor | None = None
            proto_top: torch.return_types.topk | None = None
            if proto_tensor is not None and features is not None:
                norm_features = nn.functional.normalize(features.float(), p=2, dim=1)
                proto_sims = torch.mm(norm_features, proto_tensor.t())
                if proto_counts is not None and len(proto_counts) == proto_sims.shape[1]:
                    proto_sims[:, proto_counts <= 0] = -1.0
                proto_top = torch.topk(
                    proto_sims, k=min(2, proto_sims.shape[1]), dim=1
                )

            for row in range(batch.shape[0]):
                ci = int(pred_idx[row].item())
                label = labels[ci] if 0 <= ci < len(labels) else str(ci)
                cnn_conf = float(p1[row].item())
                cnn_margin = float((p1[row] - p2[row]).item())
                cnn_evidence = (
                    0.50 * cnn_conf
                    + 0.30 * float(np.clip(cnn_margin / 0.35, 0.0, 1.0))
                    + 0.20 * float(np.clip(1.0 - entropy[row].item(), 0.0, 1.0))
                )

                proto_label: str | None = None
                proto_sim = proto_margin = proto_evidence = 0.0
                if proto_top is not None:
                    pi = int(proto_top.indices[row, 0].item())
                    proto_label = (
                        proto_labels[pi] if 0 <= pi < len(proto_labels) else str(pi)
                    )
                    proto_sim = float(proto_top.values[row, 0].item())
                    second = (
                        float(proto_top.values[row, 1].item())
                        if proto_top.values.shape[1] > 1
                        else -1.0
                    )
                    proto_margin = proto_sim - second
                    proto_evidence = (
                        0.70 * float(np.clip((proto_sim - 0.35) / 0.45, 0.0, 1.0))
                        + 0.30
                        * float(np.clip(proto_margin / 0.15, 0.0, 1.0))
                    )

                agreement = proto_label is not None and proto_label == label
                if proto_top is None:
                    trust = 0.85 * cnn_evidence
                else:
                    trust = (
                        0.55 * cnn_evidence
                        + 0.45 * proto_evidence
                        + (0.08 if agreement else -0.10)
                    )
                results.append(
                    {
                        "trust": float(np.clip(trust, 0.0, 1.0)),
                        "label": label,
                        "cnn_conf": cnn_conf,
                        "cnn_margin": cnn_margin,
                        "proto_sim": proto_sim,
                        "proto_margin": proto_margin,
                        "agreement": agreement,
                    }
                )
    if use_cache:
        for pos, rec in zip(missing_pos, results):
            cached_out[pos] = rec
            _REC_SPAN_CACHE[spans[pos]] = rec
        return [rec for rec in cached_out if rec is not None]
    return results


_REC_SPAN_CACHE: dict[tuple[int, int], dict] = {}
_REC_SPAN_CACHE_LINE_ID = 0


def _reset_recognition_span_cache(line_bgr: np.ndarray) -> None:
    global _REC_SPAN_CACHE, _REC_SPAN_CACHE_LINE_ID
    _REC_SPAN_CACHE = {}
    _REC_SPAN_CACHE_LINE_ID = id(line_bgr)


def _is_numeral_or_separator_label(label: str | None) -> bool:
    if not label:
        return False
    s = str(label).strip().upper()
    return s.startswith("NUM_") or s in {"1", "|", "WORD_SEPARATOR"}


def _identify_crop_for_detect_merge(
    line_bgr: np.ndarray,
    x0: int,
    x1: int,
    device: torch.device,
) -> tuple[str | None, float]:
    """
    Identify helper used ONLY inside detection merge.

    This does not change stone OCR identify filters / readout rules.
    """
    from tempfile import TemporaryDirectory

    from PIL import Image

    from .layout import is_word_separator_label
    from .predict import DEFAULT_CHECKPOINT
    from .predict import predict_image

    if not Path(DEFAULT_CHECKPOINT).exists():
        return None, 0.0
    piece = line_bgr[:, x0:x1]
    if piece.size == 0 or x1 - x0 < 2:
        return None, 0.0
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "crop.png"
        Image.fromarray(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB)).save(path)
        pred = predict_image(
            path,
            checkpoint=DEFAULT_CHECKPOINT,
            device=device,
            # Keep detect-merge isolated from stone preprocess experiments.
            compare_preprocess=False,
            save_debug=False,
            use_prototypes=True,
            letters_only=False,
        )
    label = pred.get("character")
    conf = float(pred.get("confidence") or 0.0)
    name = str(pred.get("name") or "").upper()
    if name.startswith("NUM") or is_word_separator_label(label):
        return "NUM_1", conf
    if name:
        return name, conf
    return str(label) if label is not None else None, conf


def _merge_completeness_oversplit(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    line_height: int,
    activity: np.ndarray | None = None,
    stroke_mask: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """Rejoin over-splits with the detect completeness net (no identify)."""
    if len(boxes) < 2 or line_height <= 0:
        return boxes

    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        left, right, conf = boxes[i]
        if i + 1 < len(boxes):
            n_left, n_right, n_conf = boxes[i + 1]
            gap = n_left - right
            w1 = right - left
            w2 = n_right - n_left
            wm = n_right - left
            r1 = w1 / line_height
            r2 = w2 / line_height
            rm = wm / line_height
            thin_stem = min(r1, r2) <= 0.28
            thin_edge = min(r1, r2) <= 0.18
            # Never rejoin a separator-thin stem with a full letter body.
            sep_letter = False
            if gap <= 3 and (
                (_is_thin_bar(w1, line_height) and r2 >= 0.30)
                or (_is_thin_bar(w2, line_height) and r1 >= 0.30)
            ):
                thin_l, thin_r = (left, right) if _is_thin_bar(w1, line_height) else (n_left, n_right)
                sep_letter = _carved_thin_stem(activity, thin_l, thin_r, line_height)
            # Parallel stem pack just split by valley cuts — do not glue back.
            parallel_pack = (
                gap <= 3
                and thin_stem
                and max(r1, r2) <= 0.45
                and stroke_mask is not None
                and _clear_stroke_gutter(stroke_mask, right)
            )
            geom_ok = (not sep_letter) and (not parallel_pack) and gap <= 3 and (
                (
                    0.38 <= rm <= 0.90
                    and min(r1, r2) <= 0.42
                    and max(r1, r2) <= 0.62
                )
                or (
                    # Wide body + thin tip strip (𐩩 X cut on the edge).
                    thin_edge
                    and 0.45 <= rm <= 1.05
                    and max(r1, r2) <= 0.92
                )
            )
            if geom_ok:
                merge_conf = _score_span(
                    enhanced,
                    line_bgr,
                    left,
                    n_right,
                    model,
                    device,
                    cnn=None,
                    cnn_weight=0.0,
                )
                continuous = True
                if activity is not None and wm > 0:
                    mid = right
                    pad = max(1, int(0.03 * line_height))
                    mid_m = float(activity[max(0, mid - pad) : mid + pad + 1].mean())
                    left_m = float(activity[left:right].mean()) if right > left else 0.0
                    right_m = float(activity[n_left:n_right].mean()) if n_right > n_left else 0.0
                    if mid_m < 0.08 and min(left_m, right_m) >= 0.20:
                        continuous = False
                if stroke_mask is not None and _clear_stroke_gutter(stroke_mask, right):
                    continuous = False
                if continuous and (
                    merge_conf >= max(conf, n_conf) + 0.05
                    or (thin_stem and merge_conf >= 0.55 and merge_conf + 0.10 >= max(conf, n_conf))
                    or (
                        thin_stem
                        and min(conf, n_conf) < 0.55
                        and max(conf, n_conf) >= 0.70
                        and merge_conf >= 0.45
                    )
                    or (
                        thin_edge
                        and merge_conf >= 0.50
                        and merge_conf + 0.05 >= max(conf, n_conf)
                    )
                    or (
                        # Weak detect tip stuck on a strong neighbor (common on
                        # line edges / half-cut X tips).
                        thin_edge
                        and min(conf, n_conf) < 0.25
                        and max(conf, n_conf) >= 0.70
                        and merge_conf >= 0.40
                    )
                ):
                    out.append((left, n_right, max(conf, n_conf, merge_conf)))
                    i += 2
                    continue
        out.append((left, right, conf))
        i += 1
    return out


def _absorb_weak_edge_slivers(
    boxes: list[tuple[int, int, float]],
    line_height: int,
    activity: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """
    Merge a thin, low-confidence edge tip into its neighbor.

    Typical case: left/right tip of 𐩩 cut off as its own weak box (det≪0.3).
    Does NOT absorb carved word-separator bars (``|``), which often score low
    on the completeness net but have strong vertical ink.
    """
    if len(boxes) < 2 or line_height <= 0:
        return boxes
    out = list(boxes)

    def _weak_thin(box: tuple[int, int, float]) -> bool:
        a, b, c = box
        if not ((b - a) / line_height <= 0.22 and c < 0.30):
            return False
        # Carved ``|`` — keep as its own box.
        if _carved_thin_stem(activity, a, b, line_height):
            return False
        return True

    # Leading sliver → next
    if _weak_thin(out[0]):
        _a, b, c0 = out[0]
        a1, b1, c1 = out[1]
        if a1 - b <= 3:
            out = [(out[0][0], b1, max(c0, c1))] + out[2:]
    # Trailing sliver → previous
    if len(out) >= 2 and _weak_thin(out[-1]):
        a0, b0, c0 = out[-2]
        a1, b1, c1 = out[-1]
        if a1 - b0 <= 3:
            out = out[:-2] + [(a0, b1, max(c0, c1))]
    return out


def _estimate_letter_pitch(
    boxes: list[tuple[int, int, float]],
    line_height: int,
) -> float:
    """Robust full-letter width estimate for this line."""
    if line_height <= 0:
        return 40.0
    fallback = max(18.0, 0.55 * float(line_height))
    if not boxes:
        return fallback
    lo = 0.32 * line_height
    hi = 1.05 * line_height
    widths = [float(b - a) for a, b, _c in boxes if lo <= (b - a) <= hi]
    if len(widths) >= 2:
        med = float(np.median(widths))
        # Drop multi-glyph outliers so merges don't raise the pitch and
        # then hide themselves from the oversized-split gate.
        core = [w for w in widths if w <= 1.35 * med]
        if len(core) >= 2:
            return float(np.median(core))
        return med
    all_w = sorted(float(b - a) for a, b, _c in boxes if b > a)
    if not all_w:
        return fallback
    return float(np.median(all_w))


def _is_word_separator_box(
    width: float,
    line_height: int,
    pitch: float,
    left_w: float | None,
    right_w: float | None,
) -> bool:
    """
    Thin stem between two full letters → keep as ``|`` (do not merge).

    Uses pitch and aspect so a real separator (e.g. width 26 on test-8) is
    protected even when it is slightly wider than a naive tip threshold.
    """
    if line_height <= 0 or pitch <= 0:
        return False
    if left_w is None or right_w is None:
        return False
    full = 0.42 * pitch
    if left_w < full or right_w < full:
        return False
    # User rule: ≤0.32×pitch OR aspect ≤0.22, plus a neighbor-relative guard
    # for slightly wider bars that are still clearly thinner than both letters.
    if width <= 0.32 * pitch:
        return True
    if width / line_height <= 0.24:
        return True
    if width <= 0.55 * min(left_w, right_w) and width / line_height <= 0.30:
        return True
    return False


def _merge_safe_letter_fragments(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    device: torch.device,
    line_height: int,
    activity: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """
    Intelligent letter rejoin that refuses to swallow word separators.

    - full | thin | full  → never merge (separator)
    - edge tip / half-letter with identify support → merge
    """
    if len(boxes) < 2 or line_height <= 0:
        return boxes

    pitch = _estimate_letter_pitch(boxes, line_height)
    tip_lim = max(0.24 * line_height, 0.38 * pitch)
    max_merge = 1.45 * pitch
    max_edge_merge = 1.55 * pitch

    def _act_ok(a: int, mid: int, b: int) -> bool:
        if activity is None or b <= a:
            return True
        pad = max(1, int(0.03 * line_height))
        mid_m = float(activity[max(0, mid - pad) : mid + pad + 1].mean())
        left_m = float(activity[a:mid].mean()) if mid > a else 0.0
        right_m = float(activity[mid:b].mean()) if b > mid else 0.0
        if mid_m < 0.07 and min(left_m, right_m) >= 0.18:
            return False
        return True

    def _widths_around(i: int) -> tuple[float | None, float, float, float | None]:
        a, b, _c = boxes_now[i]
        a2, b2, _c2 = boxes_now[i + 1]
        left_w = (
            float(boxes_now[i - 1][1] - boxes_now[i - 1][0]) if i > 0 else None
        )
        right_w = (
            float(boxes_now[i + 2][1] - boxes_now[i + 2][0])
            if i + 2 < len(boxes_now)
            else None
        )
        return left_w, float(b - a), float(b2 - a2), right_w

    boxes_now = list(boxes)
    changed = True
    while changed and len(boxes_now) >= 2:
        changed = False
        cands: list[tuple[int, int, float]] = []  # priority, index, merged_conf
        for i in range(len(boxes_now) - 1):
            a, b, c = boxes_now[i]
            a2, b2, c2 = boxes_now[i + 1]
            if a2 - b > 3:
                continue
            left_w, w1, w2, right_w = _widths_around(i)
            wm = float(b2 - a)
            thin = min(w1, w2)
            at_edge = i == 0 or i + 1 == len(boxes_now) - 1
            merge_cap = max_edge_merge if at_edge else max_merge
            if wm > merge_cap:
                continue
            if thin > tip_lim and min(w1, w2) > 0.48 * pitch:
                continue
            # Protect ``|`` between two full letters.
            if thin == w1 and _is_word_separator_box(w1, line_height, pitch, left_w, w2):
                continue
            if thin == w2 and _is_word_separator_box(w2, line_height, pitch, w1, right_w):
                continue
            if not _act_ok(a, b, b2):
                continue

            tip_case = thin <= tip_lim and max(w1, w2) >= 0.45 * pitch
            half_case = (
                thin <= 0.55 * pitch
                and max(w1, w2) <= 0.70 * pitch
                and 0.55 * pitch <= wm <= max_merge
            )
            if not (tip_case or half_case or (at_edge and thin <= tip_lim)):
                continue

            m_label, m_conf = _identify_crop_for_detect_merge(line_bgr, a, b2, device)
            # Edge tips: geometry is enough (identify often underrates wide+tip).
            if at_edge and thin <= tip_lim and max(w1, w2) >= 0.50 * pitch and wm <= max_edge_merge:
                cands.append((0, i, max(c, c2, float(m_conf or 0.0))))
                continue
            if m_label is None or _is_numeral_or_separator_label(m_label):
                continue
            if m_conf < 0.60:
                continue

            l_label, l_conf = _identify_crop_for_detect_merge(line_bgr, a, b, device)
            r_label, r_conf = _identify_crop_for_detect_merge(line_bgr, a2, b2, device)
            # Do not glue two strong different full letters.
            if (
                min(l_conf, r_conf) >= 0.85
                and l_label != r_label
                and min(w1, w2) > 0.35 * pitch
                and not _is_numeral_or_separator_label(l_label)
                and not _is_numeral_or_separator_label(r_label)
            ):
                continue
            # Prefer merge when union is a real letter and at least one side
            # was a separator/numeral fragment or weak half.
            one_frag = (
                _is_numeral_or_separator_label(l_label)
                or _is_numeral_or_separator_label(r_label)
                or min(l_conf, r_conf) < 0.70
                or thin <= tip_lim
            )
            if not one_frag and m_conf < max(l_conf, r_conf) + 0.08:
                continue
            # Internal tip glued to a full body (not full|thin|full — already skipped).
            if tip_case and m_conf >= 0.60 and wm <= max_merge:
                cands.append((1, i, max(c, c2, m_conf)))
                continue
            if half_case and m_conf >= 0.68 and wm <= max_merge:
                cands.append((2, i, max(c, c2, m_conf)))
                continue

        if not cands:
            break
        # Merge highest-priority (tips first), thinnest first within priority.
        def _key(t: tuple[int, int, float]) -> tuple:
            pri, idx, _mc = t
            a, b, _ = boxes_now[idx]
            a2, b2, _ = boxes_now[idx + 1]
            return (pri, min(b - a, b2 - a2), idx)

        _pri, i, mc = min(cands, key=_key)
        a, b, c = boxes_now[i]
        _a2, b2, c2 = boxes_now[i + 1]
        boxes_now = boxes_now[:i] + [(a, b2, mc)] + boxes_now[i + 2 :]
        changed = True

    return boxes_now


def _merge_identify_aided_oversplit(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    device: torch.device,
    line_height: int,
) -> list[tuple[int, int, float]]:
    """
    Detect-only second pass: use musnad_final as a merge advisor for hard
    half-cuts (𐩣 / 𐩢 / 𐩨 / 𐩩) where completeness alone keeps the split.

    Isolated from stone OCR identify filtering — this only moves box edges.
    """
    if len(boxes) < 2 or line_height <= 0:
        return boxes

    # X-shaped TAW is often cut so one half looks like SHIN.
    _x_like = {"TAW", "SHIN", "THAW"}

    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        left, right, conf = boxes[i]
        if i + 1 < len(boxes):
            n_left, n_right, n_conf = boxes[i + 1]
            gap = n_left - right
            r1 = (right - left) / line_height
            r2 = (n_right - n_left) / line_height
            rm = (n_right - left) / line_height
            thin_sliver = min(r1, r2) <= 0.22
            # Allow a wider main half when the other piece is only a thin edge
            # sliver (classic TAW/X oversplit: body + tip strip).
            geom_ok = gap <= 3 and (
                (
                    0.38 <= rm <= 0.90
                    and min(r1, r2) <= 0.42
                    and max(r1, r2) <= 0.62
                )
                or (
                    thin_sliver
                    and 0.45 <= rm <= 1.05
                    and max(r1, r2) <= 0.92
                )
            )
            if geom_ok:
                m_label, m_conf = _identify_crop_for_detect_merge(
                    line_bgr, left, n_right, device
                )
                if (
                    m_label is not None
                    and not _is_numeral_or_separator_label(m_label)
                    and m_conf >= 0.70
                ):
                    l_label, l_conf = _identify_crop_for_detect_merge(
                        line_bgr, left, right, device
                    )
                    r_label, r_conf = _identify_crop_for_detect_merge(
                        line_bgr, n_left, n_right, device
                    )
                    stem_part = _is_numeral_or_separator_label(l_label) or _is_numeral_or_separator_label(
                        r_label
                    )
                    if (
                        stem_part
                        and m_label not in {l_label, r_label}
                        and m_conf >= 0.85
                    ):
                        out.append((left, n_right, max(conf, n_conf, m_conf)))
                        i += 2
                        continue
                    if (
                        m_conf >= 0.85
                        and m_conf >= max(l_conf, r_conf) + 0.12
                        and m_label not in {l_label, r_label}
                    ):
                        out.append((left, n_right, max(conf, n_conf, m_conf)))
                        i += 2
                        continue
                    # Thin edge cut off an X/TAW (or similar): one half may already
                    # say TAW while the incomplete body says SHIN — still merge.
                    x_confusion = (
                        thin_sliver
                        and m_label in _x_like
                        and (
                            {l_label, r_label} & _x_like
                            or min(l_conf, r_conf) < 0.78
                        )
                    )
                    weak_fragment = thin_sliver and (
                        min(l_conf, r_conf) < 0.78
                        or min(r1, r2) <= 0.16
                        or l_label != r_label
                    )
                    if (
                        thin_sliver
                        and m_conf >= 0.72
                        and m_conf + 0.05 >= max(l_conf, r_conf)
                        and (x_confusion or weak_fragment)
                        # Don't glue two strong, different full letters.
                        and not (
                            min(l_conf, r_conf) >= 0.88
                            and l_label != r_label
                            and min(r1, r2) > 0.18
                        )
                    ):
                        out.append((left, n_right, max(conf, n_conf, m_conf)))
                        i += 2
                        continue
        out.append((left, right, conf))
        i += 1
    return out


def _span_ink(
    activity: np.ndarray, left: int, right: int
) -> tuple[float, float, bool]:
    """Return (mean, peak, has_carving) for a horizontal span."""
    sl = activity[left:right]
    if sl.size == 0:
        return 0.0, 0.0, False
    mean = float(sl.mean())
    peak = float(sl.max())
    has_carving = mean >= 0.10 or peak >= 0.35
    return mean, peak, has_carving


def _keep_marked_empty_segment(segment, conf: float) -> bool:
    """Only rescue marked-empty crops when they have very strong evidence."""
    scores = getattr(segment, "scores", {}) or {}
    img = getattr(segment, "image", None)
    h = int(img.shape[0]) if img is not None and getattr(img, "size", 0) else 0
    width = int(getattr(segment, "width", 0) or 0)
    # Completeness on ``|`` is often well below 0.85; still keep a carved stem.
    if (
        h > 0
        and 2 <= width <= max(4, int(0.28 * h))
        and float(scores.get("activity_peak", 0.0)) >= 0.40
        and float(scores.get("activity_mean", 0.0)) >= 0.12
    ):
        return True
    if conf < 0.85:
        return False
    # Hollow / barred circles often have weak column activity; objectness or
    # edge energy is enough to prove the crop is a real glyph.
    if float(scores.get("object_mean", 0.0)) >= 0.70 or float(scores.get("object_peak", 0.0)) >= 0.85:
        return True
    return (
        float(scores.get("edge_peak", 0.0)) >= 0.20
        and float(scores.get("activity_peak", 0.0)) >= 0.35
    )


def _is_thin_bar(width: int, line_height: int) -> bool:
    """Word-separator-sized vertical stem (narrow vs line height)."""
    if line_height <= 0 or width < 2:
        return False
    # ~0.24 catches slightly wide carved bars (test-1 ``|`` ≈0.22·h) while
    # leaving typical half-glyphs on taller lines free to rejoin.
    return width / line_height <= 0.24


def _carved_thin_stem(
    activity: np.ndarray | None,
    left: int,
    right: int,
    line_height: int,
) -> bool:
    """True when a thin crop has real vertical carving (likely ``|``, not tip junk)."""
    width = right - left
    if not _is_thin_bar(width, line_height):
        return False
    if activity is None or activity.size == 0 or right <= left:
        return False
    sl = activity[max(0, left) : min(len(activity), right)]
    if sl.size < 2:
        return False
    return float(sl.max()) >= 0.40 and float(sl.mean()) >= 0.14


def _edge_bar_cut(
    activity: np.ndarray,
    left: int,
    right: int,
    line_height: int,
    *,
    side: str,
    stroke_mask: np.ndarray | None = None,
) -> int | None:
    """
    If a wide crop has a true word-separator bar stuck on one edge, return the
    cut column. Criteria are strict so multi-stem letters are not bisected.
    """
    width = right - left
    if width < max(12, int(0.45 * line_height)):
        return None
    sl = activity[left:right]
    if sl.size < 8:
        return None
    # Separators are thinner than a letter, but can reach ~0.22–0.24·h.
    max_bar = max(3, int(0.24 * line_height))
    min_bar = max(2, int(0.03 * line_height))
    n = len(sl)

    if side == "left":
        bar_end = min(n - 4, max_bar)
        if bar_end < min_bar:
            return None
        peak = int(np.argmax(sl[:bar_end]))
        if float(sl[peak]) < 0.40:
            return None
        search_hi = min(n - 2, max(bar_end + 1, peak + max_bar + 2))
        if search_hi <= peak + 1:
            return None
        valley_rel = peak + 1 + int(np.argmin(sl[peak + 1 : search_hi]))
        valley = float(sl[valley_rel])
        body = float(sl[valley_rel:].max()) if valley_rel < n else 0.0
        # Deep gutter between bar and letter; body must be a real glyph.
        if valley > 0.35 * float(sl[peak]):
            return None
        if body < 0.42:
            return None
        bar_w = valley_rel
        body_w = n - valley_rel
        if not (min_bar <= bar_w <= max_bar):
            return None
        if body_w / max(line_height, 1) < 0.35:
            return None
        cut_x = left + valley_rel
        if stroke_mask is not None and _is_full_height_span(
            stroke_mask, left, cut_x, line_height, threshold=0.68
        ):
            bar_w = cut_x - left
            if bar_w / max(line_height, 1) > 0.20:
                return None
        return cut_x

    bar_start = max(0, n - max_bar)
    if n - bar_start < min_bar:
        return None
    peak = bar_start + int(np.argmax(sl[bar_start:]))
    if float(sl[peak]) < 0.40:
        return None
    search_lo = max(1, min(bar_start - 1, peak - max_bar - 2))
    if peak <= search_lo:
        return None
    valley_rel = search_lo + int(np.argmin(sl[search_lo:peak]))
    valley = float(sl[valley_rel])
    body = float(sl[: valley_rel + 1].max()) if valley_rel > 0 else 0.0
    if valley > 0.35 * float(sl[peak]):
        return None
    if body < 0.42:
        return None
    bar_w = n - valley_rel
    body_w = valley_rel
    if not (min_bar <= bar_w <= max_bar):
        return None
    if body_w / max(line_height, 1) < 0.35:
        return None
    cut_x = left + valley_rel
    if stroke_mask is not None and _is_full_height_span(
        stroke_mask, cut_x, right, line_height, threshold=0.68
    ):
        bar_w = right - cut_x
        if bar_w / max(line_height, 1) > 0.20:
            return None
    return cut_x


def _detach_edge_bars(
    boxes: list[tuple[int, int, float]],
    activity: np.ndarray,
    line_height: int,
    stroke_mask: np.ndarray | None = None,
    boundary_profile: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """Split glued word-separator bars off the sides of wider letter crops."""
    if line_height <= 0 or not boxes:
        return boxes
    out: list[tuple[int, int, float]] = []
    for left, right, conf in boxes:
        cut_l = _edge_bar_cut(
            activity, left, right, line_height, side="left", stroke_mask=stroke_mask
        )
        cut_r = _edge_bar_cut(
            activity, left, right, line_height, side="right", stroke_mask=stroke_mask
        )
        # Never apply both sides — that almost always means a multi-stem letter.
        if cut_l is not None and cut_r is not None:
            out.append((left, right, conf))
            continue
        cut = cut_l if cut_l is not None else cut_r
        if cut is None or not (left + 2 < cut < right - 2):
            out.append((left, right, conf))
            continue
        # Do not undo a recognition-guided merge by slicing through a connected
        # letter body. True separators sit in a gutter, not across a stroke.
        if stroke_mask is not None and _cut_bisects_connected_stroke(
            stroke_mask, cut
        ):
            out.append((left, right, conf))
            continue
        out.append((left, cut, conf))
        out.append((cut, right, conf))
    return _rejoin_split_thin_bars(
        out, line_height, boundary_profile=boundary_profile
    )


def _rejoin_split_thin_bars(
    boxes: list[tuple[int, int, float]],
    line_height: int,
    boundary_profile: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """Rejoin a ``|`` (or ``||``) that was over-cut into adjacent thin scraps."""
    if len(boxes) < 2 or line_height <= 0:
        return boxes
    current = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[tuple[int, int, float]] = []
        i = 0
        while i < len(current):
            left, right, conf = current[i]
            merged = False
            # Greedy chain: join consecutive thin bars up to word-separator width.
            j = i + 1
            run_right = right
            run_conf = conf
            while j < len(current):
                n_left, n_right, n_conf = current[j]
                if n_left - run_right > 2:
                    break
                # Do not glue distinct glyphs that the boundary net separated.
                if _boundary_cut_strength(boundary_profile, max(run_right, n_left)) >= 0.48:
                    break
                w_run = run_right - left
                w_next = n_right - n_left
                w_total = n_right - left
                if (
                    w_run / line_height <= 0.30
                    and w_next / line_height <= 0.30
                    and w_total / line_height <= 0.42
                ):
                    run_right = n_right
                    run_conf = 0.5 * (run_conf + n_conf)
                    j += 1
                    merged = True
                    continue
                break
            if merged:
                out.append((left, run_right, run_conf))
                i = j
                changed = True
                continue
            out.append((left, right, conf))
            i += 1
        current = out
    return current


def _thin_wide_merge_blocked(
    edges: list[int],
    i: int,
    j: int,
    line_height: int,
    activity: np.ndarray | None = None,
    stroke_mask: np.ndarray | None = None,
) -> bool:
    """
    Block merging a separator-thin stem with a much wider neighbour.

    Multi-stem letters have similar piece widths; ``|`` + letter does not.
    Requires the thin piece to look carved so letter-half rejoins still work.

    If ``stroke_mask`` is provided, only block when the thin atom sits in a
    clear gutter (true ``|``). Cuts that bisect a connected letter body are
    allowed to merge back.
    """
    if j - i < 2 or line_height <= 0:
        return False
    widths = [edges[k + 1] - edges[k] for k in range(i, j)]
    thin_atoms = [
        atom
        for atom in range(i, j)
        if _is_thin_bar(edges[atom + 1] - edges[atom], line_height)
    ]
    wide = [w for w in widths if w / line_height >= 0.32]
    if not thin_atoms or not wide:
        return False

    for atom in thin_atoms:
        a0, a1 = edges[atom], edges[atom + 1]
        if activity is not None and not _carved_thin_stem(
            activity, a0, a1, line_height
        ):
            continue
        # Internal faces of this thin atom inside the proposed merge.
        internal_cuts = [c for c in (a0, a1) if edges[i] < c < edges[j]]
        if not internal_cuts:
            continue
        if stroke_mask is None:
            return True
        for cut in internal_cuts:
            if _cut_bisects_connected_stroke(stroke_mask, cut):
                # Stem is part of a connected letter — allow merge.
                continue
            if _clear_stroke_gutter(stroke_mask, cut):
                return True
            # Activity valley without a clear stroke gutter: still treat as
            # separator-like when the thin atom is clearly carved.
            return True
    return False


def _pitch_from_edges(edges: list[int], line_height: int) -> float:
    """Estimate typical letter width from current boundary atoms."""
    fallback = max(18.0, 0.55 * float(line_height))
    if len(edges) < 2 or line_height <= 0:
        return fallback
    widths = [float(edges[i + 1] - edges[i]) for i in range(len(edges) - 1)]
    letterish = [w for w in widths if 0.35 * line_height <= w <= 1.15 * line_height]
    if len(letterish) >= 2:
        return float(np.median(letterish))
    strong = sorted(w for w in widths if w >= 0.25 * line_height)
    if len(strong) >= 2:
        return float(np.median(strong))
    return fallback


def _atom_is_internal_separator(
    edges: list[int],
    atom_i: int,
    pitch: float,
    line_height: int,
) -> bool:
    """
    True when atom is a word bar between two full letters.

    Only internal ``full | full`` — edge tips are allowed to coalesce.
    """
    n = len(edges) - 1
    if atom_i < 0 or atom_i >= n or pitch <= 0 or line_height <= 0:
        return False
    w = float(edges[atom_i + 1] - edges[atom_i])
    if atom_i == 0 or atom_i == n - 1:
        return False
    left_w = float(edges[atom_i] - edges[atom_i - 1])
    right_w = float(edges[atom_i + 2] - edges[atom_i + 1])
    full = 0.42 * pitch
    if left_w < full or right_w < full:
        return False
    if w <= 0.32 * pitch:
        return True
    if w / line_height <= 0.22:
        return True
    if w <= 0.55 * min(left_w, right_w) and w / line_height <= 0.30:
        return True
    return False


def _deep_activity_valley(activity: np.ndarray, cut: int, line_height: int) -> bool:
    """True if cut sits in a clear blank gutter (real letter boundary)."""
    if activity.size == 0 or cut <= 0 or cut >= len(activity):
        return False
    pad = max(2, int(0.03 * line_height))
    mid = float(activity[max(0, cut - pad) : cut + pad + 1].mean())
    left = float(activity[max(0, cut - 3 * pad) : cut].mean()) if cut > 0 else 0.0
    right = (
        float(activity[cut : min(len(activity), cut + 3 * pad)].mean())
        if cut < len(activity)
        else 0.0
    )
    return mid < 0.08 and min(left, right) >= 0.16


def _coalesce_edges_by_letter_pitch(
    edges: list[int],
    activity: np.ndarray,
    line_height: int,
) -> list[int]:
    """
    Remove intra-letter cuts before completeness DP.

    Completeness scores halves ≫ full glyphs on stone, so DP will never merge.
    Pitch + activity must decide letter atoms first; completeness only ranks them.
    """
    if len(edges) < 3 or line_height <= 0:
        return edges
    pitch = _pitch_from_edges(edges, line_height)
    max_join = 1.22 * pitch

    out = list(edges)
    changed = True
    while changed and len(out) >= 3:
        changed = False
        n_atoms = len(out) - 1
        for i in range(1, n_atoms):
            # Cut between atom (i-1) and atom i sits at out[i].
            if _atom_is_internal_separator(out, i - 1, pitch, line_height):
                continue
            if _atom_is_internal_separator(out, i, pitch, line_height):
                continue
            wl = float(out[i] - out[i - 1])
            wr = float(out[i + 1] - out[i])
            if wl + wr > max_join:
                continue
            # Only join when at least one side is fragment-thin.
            if min(wl, wr) > 0.52 * pitch:
                continue
            if _deep_activity_valley(activity, out[i], line_height):
                continue
            del out[i]
            changed = True
            break
    return out


def _carved_letter_mask(line_bgr: np.ndarray) -> np.ndarray:
    """
    Binary mask of the carved *letter effect* (connected grooves), not pits.

    Real Musnad strokes are darker channels with coherent structure. Random
    stone holes/cracks are smaller, more isotropic, and poorly connected.
    """
    if line_bgr.ndim == 2:
        gray = line_bgr
    else:
        gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Local background − image = dark groove response (carving).
    k = max(15, (min(h, w) // 4) | 1)
    background = cv2.medianBlur(gray, k)
    grooves = cv2.subtract(background, gray)
    grooves = cv2.normalize(grooves, None, 0, 255, cv2.NORM_MINMAX)
    # Oriented stroke energy: letters have long H/V structure; pits do not.
    enhanced = enhance_line(line_bgr)
    if enhanced.ndim == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    # Prefer elongated responses over isotropic blobs.
    orient = np.maximum(np.abs(gx), np.abs(gy))
    orient_u8 = cv2.normalize(orient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fused = cv2.addWeighted(grooves, 0.55, orient_u8, 0.45, 0)
    _, mask = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Drop tiny pit components; keep stroke-scale blobs.
    min_area = max(12, int(0.0025 * h * w))
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n_cc):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        # Reject roughly round pit-like blobs (letter strokes are elongated).
        if bw > 0 and bh > 0:
            aspect = max(bw, bh) / float(min(bw, bh))
            if aspect < 1.25 and area < 3 * min_area:
                continue
        clean[labels == i] = 255
    # Light close to reconnect broken stroke walls inside one letter.
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, ker, iterations=1)
    return clean


def _span_vertical_fill(
    stroke_mask: np.ndarray,
    left: int,
    right: int,
    line_height: int,
) -> float:
    """Fraction of line rows that contain carved stroke in ``[left, right)``."""
    if stroke_mask.size == 0 or right <= left or line_height <= 0:
        return 0.0
    w = stroke_mask.shape[1]
    roi = stroke_mask[:, max(0, left) : min(w, right)]
    if roi.size == 0:
        return 0.0
    rows = (roi > 0).any(axis=1)
    return float(rows.sum()) / float(line_height)


def _is_full_height_span(
    stroke_mask: np.ndarray,
    left: int,
    right: int,
    line_height: int,
    *,
    threshold: float = 0.72,
) -> bool:
    """True when ink in the span covers most of the line crop height."""
    return (
        _span_vertical_fill(stroke_mask, left, right, line_height) >= threshold
    )


def _clear_stroke_gutter(
    stroke_mask: np.ndarray,
    cut_x: int,
    *,
    band: int = 1,
    probe: int = 8,
) -> bool:
    """
    True when the cut column is nearly empty while both sides have ink.

    Stone grain can falsely bridge two parallel stems into one CC; a column
    gutter is the reliable signal they are separate glyphs.
    """
    if stroke_mask.size == 0:
        return False
    h, w = stroke_mask.shape[:2]
    if cut_x <= 1 or cut_x >= w - 1:
        return False
    x0 = max(0, cut_x - band)
    x1 = min(w, cut_x + band + 1)
    gutter = float(stroke_mask[:, x0:x1].sum())
    left = float(stroke_mask[:, max(0, cut_x - probe) : cut_x].sum())
    right = float(stroke_mask[:, cut_x : min(w, cut_x + probe)].sum())
    if left < 12 or right < 12:
        return False
    # Gutter much emptier than either side.
    return gutter <= 0.12 * min(left, right)


def _cut_bisects_connected_stroke(
    stroke_mask: np.ndarray,
    cut_x: int,
    *,
    band: int = 3,
) -> bool:
    """
    True if the same connected stroke body has mass on BOTH sides of cut_x.

    That means the cut is slicing through one letter's self-connected strokes,
    not sitting in the gap between two letters. A clear column gutter overrides
    weak grain bridges between parallel stems.
    """
    h, w = stroke_mask.shape[:2]
    if w < 4 or cut_x <= 1 or cut_x >= w - 1:
        return False
    if _clear_stroke_gutter(stroke_mask, cut_x, band=max(1, band // 2)):
        return False
    n_cc, labels = cv2.connectedComponents(stroke_mask, connectivity=8)
    if n_cc <= 1:
        return False
    x0 = max(0, cut_x - band)
    x1 = min(w, cut_x + band + 1)
    # Components touching the cut column band.
    touch = set(int(v) for v in np.unique(labels[:, x0:x1]) if v > 0)
    for lab in touch:
        left_mass = int(np.count_nonzero(labels[:, :cut_x] == lab))
        right_mass = int(np.count_nonzero(labels[:, cut_x:] == lab))
        # Significant stroke on both sides → do not cut here.
        if left_mass >= 8 and right_mass >= 8:
            return True
    return False


def _coalesce_edges_by_stroke_continuity(
    edges: list[int],
    stroke_mask: np.ndarray,
    line_height: int,
) -> list[int]:
    """
    Remove cuts that slice through a self-connected carved letter.

    Keep cuts that fall between separate stroke bodies (true letter gaps / ``|``).
    """
    if len(edges) < 3 or stroke_mask.size == 0 or line_height <= 0:
        return edges
    pitch = _pitch_from_edges(edges, line_height)
    out = list(edges)
    changed = True
    while changed and len(out) >= 3:
        changed = False
        n_atoms = len(out) - 1
        for i in range(1, n_atoms):
            if _atom_is_internal_separator(out, i - 1, pitch, line_height):
                continue
            if _atom_is_internal_separator(out, i, pitch, line_height):
                continue
            cut = out[i]
            wl = float(out[i] - out[i - 1])
            wr = float(out[i + 1] - out[i])
            # Only remove when joining still looks like one letter-ish span.
            if wl + wr > 1.35 * pitch:
                continue
            if not _cut_bisects_connected_stroke(stroke_mask, cut):
                continue
            del out[i]
            changed = True
            break
    return out


def _box_is_single_stroke_body(
    stroke_mask: np.ndarray,
    left: int,
    right: int,
) -> bool:
    """True when the crop is dominated by one connected carved body."""
    if stroke_mask.size == 0 or right <= left:
        return False
    roi = stroke_mask[:, left:right]
    if not np.any(roi):
        return False
    n_cc, _labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if n_cc <= 1:
        return False
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_cc)]
    if not areas:
        return False
    areas.sort(reverse=True)
    # One dominant body (not two side-by-side letters).
    if len(areas) == 1:
        return True
    return areas[0] >= 2.2 * areas[1]


def _letter_effect_score(
    line_bgr: np.ndarray,
    stroke_mask: np.ndarray,
    left: int,
    right: int,
) -> float:
    """
    How much a crop looks like a carved Musnad letter vs stone damage.

    Uses groove strength, stroke dominance, and elongation. Random pits/cracks
    score low; deliberate carved bodies score higher.
    """
    if right - left < 2 or stroke_mask.size == 0:
        return 0.0
    h = line_bgr.shape[0]
    roi_mask = stroke_mask[:, left:right]
    ink = float((roi_mask > 0).mean()) if roi_mask.size else 0.0
    if ink < 0.01:
        return 0.0

    gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY) if line_bgr.ndim == 3 else line_bgr
    patch = gray[:, left:right].astype(np.float32)
    # Local groove depth: darker than neighborhood.
    k = max(9, (min(h, right - left) // 3) | 1)
    bg = cv2.medianBlur(gray, k)[:, left:right].astype(np.float32)
    depth = np.maximum(0.0, bg - patch)
    depth_peak = float(np.percentile(depth, 90)) / 255.0
    depth_mean = float(depth.mean()) / 255.0

    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
        roi_mask, connectivity=8
    )
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_cc)]
    if not areas:
        return 0.0
    areas.sort(reverse=True)
    total = float(sum(areas)) + 1e-6
    dominance = areas[0] / total
    # Elongation of the dominant blob.
    dom_i = 1 + int(np.argmax([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_cc)]))
    bw = max(1, int(stats[dom_i, cv2.CC_STAT_WIDTH]))
    bh = max(1, int(stats[dom_i, cv2.CC_STAT_HEIGHT]))
    elongation = max(bw, bh) / float(min(bw, bh))
    elong_term = float(np.clip((elongation - 1.0) / 3.0, 0.0, 1.0))

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    aniso = float(np.abs(gx).mean() / (np.abs(gy).mean() + 1e-6))
    aniso_term = float(np.clip((aniso - 0.8) / 2.0, 0.0, 1.0))

    score = (
        0.30 * float(np.clip(depth_peak * 3.0, 0.0, 1.0))
        + 0.20 * float(np.clip(depth_mean * 8.0, 0.0, 1.0))
        + 0.25 * dominance
        + 0.15 * elong_term
        + 0.10 * aniso_term
    )
    # Very sparse ink with many tiny blobs → stone damage.
    if ink < 0.08 and len(areas) >= 3 and dominance < 0.55:
        score *= 0.55
    return float(np.clip(score, 0.0, 1.0))


def _filter_by_letter_effect(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    stroke_mask: np.ndarray,
    line_height: int,
    activity: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """
    Absorb or drop crops that look like stone noise, not carved letters.

    Keeps: strong completeness, clear separators between full letters, and
    boxes with a strong letter-effect score. Weak noise tips get glued into a
    stronger neighbor when stroke continuity / pitch allows.
    """
    if len(boxes) < 1 or line_height <= 0:
        return boxes
    pitch = _estimate_letter_pitch(boxes, line_height)
    effects = [
        _letter_effect_score(line_bgr, stroke_mask, a, b) for a, b, _c in boxes
    ]

    def _is_sep_at(i: int) -> bool:
        a, b, _c = boxes[i]
        w = float(b - a)
        left_w = float(boxes[i - 1][1] - boxes[i - 1][0]) if i > 0 else None
        right_w = (
            float(boxes[i + 1][1] - boxes[i + 1][0]) if i + 1 < len(boxes) else None
        )
        if _is_word_separator_box(w, line_height, pitch, left_w, right_w):
            return True
        # Edge ``|`` has only one neighbor — still protect carved thin stems.
        return _carved_thin_stem(activity, a, b, line_height)

    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        a, b, conf = boxes[i]
        eff = effects[i]
        w = b - a

        # Adjacent weak pair sliced through one connected stroke → rejoin.
        # Never across a clear stroke gutter (parallel ``|`` packs).
        if i + 1 < len(boxes):
            a2, b2, c2 = boxes[i + 1]
            if (
                a2 - b <= 3
                and (b2 - a) <= 1.35 * pitch
                and _cut_bisects_connected_stroke(stroke_mask, b)
                and not _clear_stroke_gutter(stroke_mask, b)
                and not _is_sep_at(i)
                and not _is_sep_at(i + 1)
            ):
                out.append((a, b2, max(conf, c2)))
                i += 2
                continue

        # Trailing / leading thin tip → absorb into neighbor (geometry).
        tip = w <= max(0.26 * line_height, 0.40 * pitch)
        if tip and (not _is_sep_at(i)) and i == len(boxes) - 1 and out:
            prev_a, prev_b, prev_c = out[-1]
            if (a - prev_b) <= 3 and (b - prev_a) <= 1.55 * pitch:
                out[-1] = (prev_a, b, max(prev_c, conf))
                i += 1
                continue
        if tip and (not _is_sep_at(i)) and i == 0 and len(boxes) >= 2:
            a2, b2, c2 = boxes[1]
            if (a2 - b) <= 3 and (b2 - a) <= 1.55 * pitch and not _is_sep_at(1):
                out.append((a, b2, max(conf, c2)))
                i += 2
                continue

        keep_strong = conf >= 0.72 or (conf >= 0.55 and eff >= 0.42)
        keep_sep = _is_sep_at(i)
        keep_bar = w / line_height <= 0.28 and conf >= 0.55 and eff >= 0.28
        keep_letterish = eff >= 0.48 and w >= 0.40 * pitch

        if keep_strong or keep_sep or keep_bar or keep_letterish:
            out.append((a, b, conf))
            i += 1
            continue

        # Weak noise / fragment: prefer absorb into stronger neighbor.
        absorbed = False
        if i + 1 < len(boxes):
            a2, b2, c2 = boxes[i + 1]
            if a2 - b <= 3 and not _clear_stroke_gutter(stroke_mask, b):
                wm = b2 - a
                eff2 = effects[i + 1]
                if (
                    wm <= 1.35 * pitch
                    and min(conf, c2) < 0.70
                    and (
                        _cut_bisects_connected_stroke(stroke_mask, b)
                        or (eff + eff2) >= 0.50
                        or max(eff, eff2) >= 0.40
                    )
                    and not _is_sep_at(i + 1)
                ):
                    out.append((a, b2, max(conf, c2)))
                    i += 2
                    absorbed = True
        if absorbed:
            continue

        if out and (a - out[-1][1]) <= 3:
            prev_a, prev_b, prev_c = out[-1]
            wm = b - prev_a
            if (
                wm <= 1.40 * pitch
                and not keep_sep
                and not _clear_stroke_gutter(stroke_mask, prev_b)
            ):
                out[-1] = (prev_a, b, max(prev_c, conf))
                absorbed = True
        if absorbed:
            i += 1
            continue

        # Edge junk with almost no letter effect → drop.
        if eff < 0.25 and conf < 0.50 and (i == 0 or i == len(boxes) - 1):
            i += 1
            continue
        if eff >= 0.32 or conf >= 0.50:
            out.append((a, b, conf))
        i += 1

    return out


# Round / curved Musnad letters that commonly glue to ``|`` or a neighbour
# while the merged box stays near one pitch wide (so oversized triggers miss).
_ROUND_GLUE_LETTERS = frozenset({"𐩥", "𐩲", "𐩧"})

# MEM (𐩣): curved body + vertical stem. The body half often scores as RESH
# (𐩧) and the stem as NUM_1 / ``|``, so peel/DP keep amputating it.
_MEM_LABELS = frozenset({"𐩣", "MEM"})
_MEM_BODY_LOOKALIKES = frozenset({"𐩧", "RESH", "𐩣", "MEM"})

# HETH (𐩢): two vertical stems with crossbars. One stem is often decoded as
# NUM_1 / ``|`` while the remaining half confuses as TAW / KAPH / etc.
_HETH_LABELS = frozenset({"𐩢", "HETH", "HA"})

# BETH (𐩨): two parallel uprights with crossbars (ladder shape). DP often
# cuts between the stems; one half stays 𐩨, the other becomes NUM_2.
_BETH_LABELS = frozenset({"𐩨", "BETH", "BA"})


def _is_round_glue_label(label: str | None) -> bool:
    return bool(label) and str(label) in _ROUND_GLUE_LETTERS


def _is_mem_label(label: str | None) -> bool:
    if not label:
        return False
    s = str(label).strip()
    return s in _MEM_LABELS or s.upper() == "MEM"


def _is_heth_label(label: str | None) -> bool:
    if not label:
        return False
    s = str(label).strip()
    return s in _HETH_LABELS or s.upper() in {"HETH", "HA"}


def _is_beth_label(label: str | None) -> bool:
    if not label:
        return False
    s = str(label).strip()
    return s in _BETH_LABELS or s.upper() in {"BETH", "BA"}


def _is_stem_body_letter_label(label: str | None) -> bool:
    return _is_mem_label(label) or _is_heth_label(label)


def _recognition_resplit_wide_boxes(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    activity: np.ndarray,
    stroke_mask: np.ndarray,
    recognition_bundle: tuple[torch.nn.Module, list[str], dict | None] | None,
    *,
    max_passes: int = 3,
    source_height: int | None = None,
    allow_panoramic_resplit: bool = False,
    boundary_profile: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """
    Recognition-guided repair for merges the lattice never offered a cut for.

    Compares joint evidence on a wide/low-trust box vs a split at activity /
    gutter candidates. Geometry proposes cuts; recognition decides. Connected
    two-stem letters (e.g. beth) stay intact unless a thin separator peels off.

    Also targets near-pitch merges of round letters (𐩥 / 𐩲 / 𐩧) that often
    swallow a neighbouring ``|`` without looking "wide".
    """
    if not boxes or line_bgr.size == 0:
        return boxes
    h = int(line_bgr.shape[0])
    if h <= 0 or activity.size == 0:
        return boxes

    enhanced = enhance_line(line_bgr)
    pitch = _estimate_letter_pitch(boxes, h)
    min_part = max(4, int(0.10 * h))
    # Tall single-line panoramas (e.g. test-24) have naturally wide glyphs.
    # Multi-line tablets still need merge repair even when a row is tall/wide.
    ref_h = int(source_height) if source_height is not None else h
    short_dense = ref_h < 72
    if (
        not allow_panoramic_resplit
        and (line_bgr.shape[1] / max(h, 1)) >= 4.0
        and ref_h >= 100
    ):
        return boxes
    current = list(boxes)

    def _joint(comp: float, rec: dict) -> float:
        trust = float(rec.get("trust") or 0.0)
        label = rec.get("label")
        sep = _is_numeral_or_separator_label(label)
        return float(
            np.clip(0.36 * float(comp) + 0.50 * trust + (0.08 if sep else 0.0), 0.0, 1.0)
        )

    for _ in range(max_passes):
        out: list[tuple[int, int, float]] = []
        changed = False
        for left, right, conf in current:
            width = right - left
            oversized = width >= max(1.35 * pitch, 0.55 * h)
            letter_pair_wide = width >= max(1.45 * pitch, 0.58 * h)
            # Triple+ merges (e.g. test-27 L1 left cluster) need recursive valley cuts.
            very_wide = width >= max(2.15 * pitch, 0.95 * h)
            # 𐩥/𐩲/𐩧 merges often sit near 1 pitch with mid/high confidence.
            near_pitch = (
                max(0.82 * pitch, 0.28 * h)
                <= width
                <= max(1.62 * pitch, 0.72 * h)
            )
            # Wide/low-trust only — never treat a ~1-pitch HETH/MEM/WAW as "wide"
            # just because the line band is tall (0.22·h made every glyph resplit).
            # Near-pitch + low conf still counts when the box is clearly >1 pitch
            # (test-16 ``|``+letter glue at ~1.3× with conf≈0.59).
            low_conf_wide = (
                width >= max(1.20 * pitch, 0.45 * h)
                and conf < 0.62
            )
            incomplete = conf < 0.50 and width >= max(1.10 * pitch, 0.38 * h)
            if not oversized and not low_conf_wide and not near_pitch and not incomplete:
                out.append((left, right, conf))
                continue
            round_only = near_pitch and not oversized and not low_conf_wide and not incomplete
            if short_dense and round_only:
                out.append((left, right, conf))
                continue

            # Candidate cuts: stroke gutters + activity valleys + edge peel
            # probes (``|`` often sits between two ink peaks, not on a valley).
            cands: list[int] = []
            for x in range(left + min_part, right - min_part + 1):
                if _clear_stroke_gutter(stroke_mask, x):
                    cands.append(x)
            sl = activity[left:right]
            peak = float(sl.max()) if sl.size else 0.0
            if peak > 1e-6:
                for i in range(min_part, width - min_part):
                    x = left + i
                    if not (
                        float(activity[x]) <= float(activity[x - 1])
                        and float(activity[x]) <= float(activity[x + 1])
                    ):
                        continue
                    if float(activity[x]) > 0.62 * peak:
                        continue
                    cands.append(x)
            max_bar = max(min_part + 1, int(0.38 * h))
            for bar_w in range(min_part, max_bar + 1):
                if left + bar_w <= right - min_part:
                    cands.append(left + bar_w)
                if right - bar_w >= left + min_part:
                    cands.append(right - bar_w)
            # Boundary-net peaks inside the box (including weak ripples) are
            # strong split hypotheses for low-completeness merges.
            if boundary_profile is not None and boundary_profile.size == line_bgr.shape[1]:
                from .letter_boundary_net import boundary_peak_maps

                _s, peaks, _p = boundary_peak_maps(
                    boundary_profile, h, min_prominence_frac=0.05, ripple_height=ref_h
                )
                del _s, _p
                peak_min = 0.40 if (short_dense and conf < 0.50) else (
                    0.62 if short_dense else 0.52
                )
                for x in peaks:
                    if left + min_part <= x <= right - min_part:
                        if _boundary_cut_strength(boundary_profile, int(x)) >= peak_min:
                            cands.append(int(x))
            kept: list[int] = []
            for x in sorted(set(cands)):
                if not kept or x - kept[-1] >= 2:
                    kept.append(x)
            if not kept and (incomplete or (conf < 0.50 and (oversized or low_conf_wide or very_wide))):
                lo = left + min_part
                hi = right - min_part
                if hi > lo and activity.size:
                    sl = activity[lo : hi + 1]
                    kept.append(lo + int(np.argmin(sl)))
            if not kept:
                out.append((left, right, conf))
                continue

            spans = [(left, right)]
            for cut in kept:
                spans.append((left, cut))
                spans.append((cut, right))
            # Unique preserve order
            uniq: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for sp in spans:
                if sp in seen or sp[1] - sp[0] < 2:
                    continue
                seen.add(sp)
                uniq.append(sp)

            tensors = [
                torch.from_numpy(prepare_window(enhanced[:, a:b])).unsqueeze(0)
                for a, b in uniq
            ]
            comps: list[float] = []
            with torch.no_grad():
                for start in range(0, len(tensors), 64):
                    batch = torch.stack(tensors[start : start + 64]).to(device)
                    comps.extend(
                        float(x) for x in torch.sigmoid(model(batch)).cpu().tolist()
                    )
            recs = _batch_recognition_span_scores(
                line_bgr, uniq, recognition_bundle, device
            )
            score_map = {
                sp: (_joint(comp, rec), rec, comp)
                for sp, comp, rec in zip(uniq, comps, recs)
            }
            merge_joint, merge_rec, merge_comp = score_map[(left, right)]
            merge_is_stem_body = _is_stem_body_letter_label(merge_rec.get("label")) and (
                float(merge_rec.get("trust") or 0.0) >= 0.70 or merge_comp >= 0.55
            )
            # Tall one-pitch stem-body glyphs (HETH/MEM) must not be opened.
            # Only override that when the box is clearly a multi-letter blob.
            if (
                near_pitch
                and merge_is_stem_body
                and not very_wide
                and width < max(1.55 * pitch, 0.62 * h)
            ):
                out.append((left, right, conf))
                continue

            best: tuple[float, int, float, float] | None = None
            single_body = _box_is_single_stroke_body(stroke_mask, left, right)
            for cut in kept:
                j_l, rec_l, _c_l = score_map[(left, cut)]
                j_r, rec_r, _c_r = score_map[(cut, right)]
                gain = j_l + j_r - 0.55 - merge_joint
                w_l = cut - left
                w_r = right - cut
                lab_l = rec_l.get("label")
                lab_r = rec_r.get("label")
                sep_l = _is_numeral_or_separator_label(lab_l)
                sep_r = _is_numeral_or_separator_label(lab_r)
                peel_l = sep_l and w_l / h <= 0.42
                peel_r = sep_r and w_r / h <= 0.42
                peel = peel_l or peel_r
                round_l = _is_round_glue_label(lab_l)
                round_r = _is_round_glue_label(lab_r)
                round_hit = round_l or round_r
                gutter = _clear_stroke_gutter(stroke_mask, cut)
                bisect = _cut_bisects_connected_stroke(stroke_mask, cut)
                cut_str = _boundary_cut_strength(boundary_profile, cut)

                allow = False
                carved_l = _carved_thin_stem(activity, left, cut, h)
                carved_r = _carved_thin_stem(activity, cut, right, h)
                peak_box = float(np.max(activity[left:right]) + 1e-6)

                # Near-pitch scan is only for round-letter glue repairs, but
                # still allow dual-separator (``||``) probes on those boxes.
                # Empty-completeness merges (test-27 L1 460-523) must still
                # peel a glued thin bar even when the box looks ~1 pitch wide.
                if (
                    round_only
                    and merge_comp >= 0.25
                    and not round_hit
                    and not (sep_l and sep_r)
                ):
                    continue

                # Intact MEM/HETH must not be opened into body + false ``|``.
                if merge_is_stem_body and peel:
                    continue

                # Bar + letter on wide/low-conf boxes (not near-pitch-only).
                if (
                    not allow
                    and peel
                    and not (sep_l and sep_r)
                    and not merge_is_stem_body
                    and not (round_only and merge_comp >= 0.25)
                    and gain >= 0.12
                    and max(j_l, j_r) >= 0.55
                    and min(j_l, j_r) >= 0.42
                    and (
                        (
                            peel_l
                            and 0.06 <= w_l / max(h, 1) <= 0.28
                            and w_r / max(h, 1) >= 0.24
                            and not sep_r
                            and (gutter or float(activity[cut]) <= 0.42 * peak_box)
                            and (carved_l or float(np.max(activity[left:cut])) >= 0.28)
                        )
                        or (
                            peel_r
                            and 0.06 <= w_r / max(h, 1) <= 0.28
                            and w_l / max(h, 1) >= 0.24
                            and not sep_l
                            and (gutter or float(activity[cut]) <= 0.42 * peak_box)
                            and (carved_r or float(np.max(activity[cut:right])) >= 0.28)
                        )
                    )
                    and not (bisect and single_body and merge_comp >= 0.55)
                ):
                    allow = True

                # Touching thin scrap + letter: merge completeness near empty but
                # recognition still trusts the glued crop (test-27 L1 @477).
                if (
                    not allow
                    and peel
                    and not (sep_l and sep_r)
                    and not merge_is_stem_body
                    and merge_comp < 0.15
                    and gain >= 0.25
                    and max(j_l, j_r) >= 0.70
                    and min(j_l, j_r) >= 0.45
                    and min(w_l, w_r) / max(h, 1) <= 0.28
                    and max(w_l, w_r) / max(h, 1) >= 0.22
                ):
                    allow = True

                # Multi-stem valley for wide incomplete merges.
                if (
                    not allow
                    and not round_only
                    and letter_pair_wide
                    and merge_comp < 0.25
                    and gain >= 0.15
                    and min(j_l, j_r) >= 0.50
                    and not sep_l
                    and not sep_r
                    and float(activity[cut]) <= 0.50 * peak_box
                    and (
                        gutter
                        or (
                            float(np.max(activity[left:cut])) >= 0.35
                            and float(np.max(activity[cut:right])) >= 0.35
                        )
                    )
                    and not (merge_is_stem_body and peel)
                    and not (bisect and single_body and merge_comp >= 0.40)
                ):
                    allow = True

                if gutter and gain >= 0.02 and (
                    peel or (carved_l or carved_r) or min(j_l, j_r) >= 0.45
                ):
                    # Clear gutter between two bars is a valid ``||`` split; other
                    # dual-NUM_1 cases need the dedicated rule below.
                    if (not (sep_l and sep_r)) or min(j_l, j_r) >= 0.55:
                        allow = True
                elif (
                    peel
                    and not (sep_l and sep_r)
                    and gain >= 0.04
                    and max(j_l, j_r) >= 0.50
                    and gutter
                ):
                    # Clean gutter separator peel.
                    allow = True
                elif (
                    peel
                    and not (sep_l and sep_r)
                    and gain >= 0.20
                    and max(j_l, j_r) >= 0.60
                    and min(w_l, w_r) / max(h, 1) <= 0.24
                    and (
                        (peel_l and carved_l and j_r >= 0.55)
                        or (peel_r and carved_r and j_l >= 0.55)
                    )
                ):
                    # Grain-bridged ``|`` + letter. Never peel a stem off a
                    # connected multi-stem / solid glyph (beth, etc.).
                    if not (bisect and single_body):
                        allow = True
                elif (
                    sep_l
                    and sep_r
                    and gain >= 0.15
                    and min(j_l, j_r) >= 0.50
                    and min(w_l, w_r) / max(h, 1) <= 0.32
                    # Halves of a real letter often score NUM_1+NUM_1 while the
                    # merge is the true glyph — only split weak/empty || merges
                    # with a clear activity dip between the bars.
                    and _is_numeral_or_separator_label(merge_rec.get("label"))
                    and not _is_heth_label(merge_rec.get("label"))
                    and (
                        (
                            merge_comp < 0.80
                            and float(merge_rec.get("trust") or 0.0) < 0.40
                            and max(w_l, w_r) / max(h, 1) <= 0.34
                        )
                        # Empty-completeness || still gets high prototype trust.
                        or (
                            merge_comp < 0.15
                            and max(w_l, w_r) / max(h, 1) <= 0.40
                        )
                    )
                    and float(activity[cut])
                    <= 0.55 * float(np.max(activity[left:right]) + 1e-6)
                ):
                    # Glued word-separator pair (``||``).
                    allow = True
                elif (
                    round_hit
                    and gain >= 0.08
                ):
                    # 𐩥 / 𐩲 are circular (not ultra-thin). 𐩧 is a curved stem.
                    def _round_ok(lab: str | None, w: int, j: float) -> bool:
                        if lab in {"𐩥", "𐩲"}:
                            # Small rings often get middling joint after a tight
                            # peel; 0.48 still rejects random low-trust scraps.
                            return j >= 0.48 and w / max(h, 1) >= 0.18
                        if lab == "𐩧":
                            return j >= 0.55 and 0.20 <= w / max(h, 1) <= 0.55
                        return False

                    def _sep_ok(w: int, carved: bool, is_sep: bool) -> bool:
                        # Real ``|`` band; reject tip slivers and uncarved grain.
                        return (
                            is_sep
                            and 0.08 <= w / max(h, 1) <= 0.30
                            and (gutter or carved)
                        )

                    sep_other_l = (
                        round_r
                        and _round_ok(lab_r, w_r, j_r)
                        and _sep_ok(w_l, carved_l, sep_l or peel_l)
                    )
                    sep_other_r = (
                        round_l
                        and _round_ok(lab_l, w_l, j_l)
                        and _sep_ok(w_r, carved_r, sep_r or peel_r)
                    )
                    # 𐩲/𐩥 + ``|`` often share a weak grain bridge (single_body)
                    # while merge completeness stays low — still peel.
                    ring_sep = sep_other_l or sep_other_r
                    letter_other = (
                        (
                            round_l
                            and _round_ok(lab_l, w_l, j_l)
                            and (not sep_r)
                            and j_r >= 0.55
                            and gain >= 0.20
                            and not single_body
                        )
                        or (
                            round_r
                            and _round_ok(lab_r, w_r, j_r)
                            and (not sep_l)
                            and j_l >= 0.55
                            and gain >= 0.20
                            and not single_body
                        )
                    )
                    if ring_sep:
                        merge_lab = merge_rec.get("label")
                        merge_trust = float(merge_rec.get("trust") or 0.0)
                        # MEM/HETH body + stem≈NUM_1: never peel that stem off.
                        if _is_stem_body_letter_label(merge_lab) and (
                            merge_trust >= 0.70 or merge_comp >= 0.55
                        ):
                            allow = False
                        # Do not amputate a stem tip from an already-complete
                        # round letter (common false NUM_1 on 𐩥 edges).
                        elif (
                            _is_round_glue_label(merge_lab)
                            and merge_comp >= 0.85
                            and single_body
                            and bisect
                        ):
                            allow = False
                        elif (
                            bisect
                            and single_body
                            and gain < 0.10
                            and merge_comp >= 0.75
                        ):
                            allow = False
                        else:
                            allow = True
                    elif letter_other:
                        allow = True

                # Letter+letter on a wide box. Kept outside the round_hit elif so a
                # high-trust 𐩥 half cannot suppress splitting a 2–3 glyph merge
                # that has near-zero completeness (test-6 leftmost cluster).
                if (
                    not allow
                    and not round_only
                    and letter_pair_wide
                    and gain >= 0.12
                    and j_l >= 0.55
                    and j_r >= 0.55
                    and not sep_l
                    and not sep_r
                    and (not short_dense or merge_comp < 0.45)
                ):
                    allow = True
                # Completeness already says this is not one letter. Split when
                # either both sides improve, or one side is a real letter and
                # the leftover is no worse (later passes finish a 3-glyph blob).
                if (
                    not allow
                    and not round_only
                    and (oversized or letter_pair_wide or low_conf_wide or very_wide or incomplete)
                    and merge_comp < 0.48
                    and conf < 0.55
                    and max(float(_c_l), float(_c_r)) >= merge_comp + 0.04
                    and min(float(_c_l), float(_c_r)) >= merge_comp - 0.12
                    and not (merge_is_stem_body and merge_comp >= 0.42)
                ):
                    allow = True
                # 3-glyph blobs: every 2-way cut is still incomplete, so neither
                # half beats the merge. Cut at a valley; later passes finish.
                # Do not use this on a ~1-letter box (open/round glyphs on test-2).
                blob = very_wide or letter_pair_wide or merge_comp < 0.32
                if (
                    not allow
                    and not round_only
                    and incomplete
                    and blob
                    and merge_comp < 0.50
                    and float(np.max(activity[left:cut])) >= 0.25
                    and float(np.max(activity[cut:right])) >= 0.25
                    and min(w_l, w_r) >= min_part
                    and not (merge_is_stem_body and merge_comp >= 0.50 and not very_wide)
                    and not (bisect and single_body and merge_comp >= 0.48)
                ):
                    allow = True
                # Tall single-line crops: two letters may sit under ~1.6×pitch so
                # letter_pair_wide never fires, but conf stays mid (𐩲+𐩧 glue).
                if (
                    not allow
                    and not round_only
                    and low_conf_wide
                    and gain >= 0.12
                    and j_l >= 0.50
                    and j_r >= 0.50
                    and not sep_l
                    and not sep_r
                    and not single_body
                    and not (merge_is_stem_body and peel)
                ):
                    allow = True
                # 𐩲+𐩥 / 𐩲+𐩧 ring+crescent glue (~1.5–1.75 pitch). Merge keeps
                # high 𐩲 trust while one half scores weakly (test-25).
                if (
                    not allow
                    and _is_round_glue_label(merge_rec.get("label"))
                    and conf < 0.65
                    and width <= max(1.75 * pitch, 0.88 * h)
                    and gain >= 0.04
                    and min(j_l, j_r) >= 0.40
                    and max(j_l, j_r) >= 0.52
                    and (_is_round_glue_label(lab_l) or _is_round_glue_label(lab_r))
                    and not sep_l
                    and not sep_r
                    and float(activity[cut]) <= 0.55 * peak_box
                    and min(w_l, w_r) / max(h, 1) >= 0.15
                    and not single_body
                ):
                    allow = True
                # Empty-completeness letter pair. Require a strong gain and a
                # depressed cut so grain-bridged true letters stay intact.
                if (
                    not allow
                    and not round_only
                    and letter_pair_wide
                    and merge_comp < 0.12
                    and gain >= 0.22
                    and min(j_l, j_r) >= 0.50
                    and not sep_l
                    and not sep_r
                    and float(activity[cut]) <= 0.70 * peak_box
                    and not (merge_is_stem_body and peel)
                ):
                    allow = True
                # Prototype trust alone can score a multi-letter blob as 𐩥; if
                # completeness is empty, prefer any solid two-way split.
                if (
                    not allow
                    and not round_only
                    and oversized
                    and merge_comp < 0.20
                    and gain >= 0.10
                    and min(j_l, j_r) >= 0.50
                    and not sep_l
                    and not sep_r
                    and not (merge_is_stem_body and peel)
                ):
                    allow = True
                # Very-wide multi-letter clusters: one side may still be a residual
                # merge (low joint). Carve at a deep valley; later passes finish.
                if (
                    not allow
                    and not round_only
                    and very_wide
                    and merge_comp < 0.25
                    and gain >= 0.08
                    and max(j_l, j_r) >= 0.62
                    and float(activity[cut]) <= 0.45 * peak_box
                    and not (sep_l and sep_r)
                    and not (merge_is_stem_body and peel)
                ):
                    allow = True
                # Grain-bridged ``|`` + letter: merge completeness empty, letter
                # half may score poorly on stone noise — still peel the stem.
                # Activity peaks on clear bars can sit under the usual carved
                # threshold (0.40); accept a weaker peak when the side is sep.
                if (
                    not allow
                    and not round_only
                    and merge_comp < 0.20
                    and float(activity[cut]) <= 0.50 * peak_box
                    and not (sep_l and sep_r)
                    and not (merge_is_stem_body and peel)
                    and not bisect
                    and peel
                    and (
                        (
                            peel_l
                            and 0.08 <= w_l / max(h, 1) <= 0.28
                            and j_l >= 0.40
                            and w_r / max(h, 1) >= 0.18
                            and (
                                carved_l
                                or float(np.max(activity[left:cut])) >= 0.22
                            )
                        )
                        or (
                            peel_r
                            and 0.08 <= w_r / max(h, 1) <= 0.28
                            and j_r >= 0.40
                            and w_l / max(h, 1) >= 0.18
                            and (
                                carved_r
                                or float(np.max(activity[cut:right])) >= 0.22
                            )
                        )
                    )
                ):
                    allow = True
                if (
                    not allow
                    and not round_only
                    and not bisect
                    and not single_body
                    and gain >= (0.18 if short_dense else 0.08)
                    and min(j_l, j_r) >= 0.48
                    and not (sep_l and sep_r)
                    and (
                        gutter
                        or cut_str >= (0.60 if short_dense else 0.48)
                    )
                ):
                    allow = True

                if allow:
                    depth = -float(activity[cut]) / peak_box
                    prev_depth = (
                        -float(activity[best[1]]) / peak_box if best is not None else None
                    )
                    if best is None or (gain, depth) > (best[0], prev_depth):
                        best = (gain, cut, j_l, j_r)

            if (
                best is None
                and incomplete
                and merge_comp < 0.50
                and (very_wide or letter_pair_wide or merge_comp < 0.32)
                and kept
                and not (merge_is_stem_body and merge_comp >= 0.50 and not very_wide)
            ):
                cut = min(
                    kept,
                    key=lambda x: (
                        float(activity[x]),
                        -min(x - left, right - x),
                    ),
                )
                j_l, _rec_l, _c_l = score_map[(left, cut)]
                j_r, _rec_r, _c_r = score_map[(cut, right)]
                if (
                    min(cut - left, right - cut) >= min_part
                    and float(np.max(activity[left:cut])) >= 0.25
                    and float(np.max(activity[cut:right])) >= 0.25
                ):
                    best = (j_l + j_r - 0.55 - merge_joint, cut, j_l, j_r)

            if best is None:
                out.append((left, right, conf))
                continue
            _gain, cut, j_l, j_r = best
            out.append((left, cut, float(j_l)))
            out.append((cut, right, float(j_r)))
            changed = True

        current = out
        if not changed:
            break
    current.sort(key=lambda t: t[0])
    return current


def _rejoin_stem_body_false_splits(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    device: torch.device,
    line_height: int,
    recognition_bundle: tuple[torch.nn.Module, list[str], dict | None] | None = None,
    boundary_profile: np.ndarray | None = None,
    model: LetterCompletenessNet | None = None,
) -> list[tuple[int, int, float]]:
    """
    Rejoin stem+body letters when one stem was decoded as a word separator.

    Covers:
    - MEM (𐩣): body≈RESH / weak MEM + stem≈NUM_1
    - HETH (𐩢): one upright≈NUM_1 + other half confuses as TAW/KAPH/…
    - BETH (𐩨): bisected between its two uprights → 𐩨 + NUM_2/NUM_1
    - BETH mid-bisect: letter-width / thin+body halves; scan for 𐩨 (test-27 L2)

    Uses calibrated recognition when available (HETH halves often still get a
    high identify score on the incomplete body alone). Only merges when the
    combined crop is a clearly better stem-body letter than the body half —
    preserves a real ``|`` after an already-complete MEM/HETH.
    """
    if len(boxes) < 2 or line_height <= 0:
        return boxes

    stroke_mask = _carved_letter_mask(line_bgr)

    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        left, right, conf = boxes[i]
        if i + 1 < len(boxes):
            n_left, n_right, n_conf = boxes[i + 1]
            gap = n_left - right
            w1 = right - left
            w2 = n_right - n_left
            wm = n_right - left
            r1 = w1 / line_height
            r2 = w2 / line_height
            rm = wm / line_height
            thin = min(r1, r2) <= 0.28

            # Keep confident boundary-net cuts, with one exception: open-center
            # letters (𐩨 etc.) put a strong peak in the hollow between uprights.
            # If both halves look like NUM scraps and the union is a strong
            # stem-body letter, rejoin despite the peak (test-16 L1).
            seam = max(right, n_left)
            cut_str = _boundary_cut_strength(boundary_profile, seam)
            classic_stem_body = min(r1, r2) <= 0.26 and max(r1, r2) >= 0.32
            if cut_str >= 0.70 or (cut_str >= 0.48 and not classic_stem_body):
                rescued = False
                if (
                    gap <= 2
                    and 0.35 <= rm <= 1.15
                    and max(r1, r2) <= 0.55
                    and recognition_bundle is not None
                ):
                    triad = _batch_recognition_span_scores(
                        line_bgr,
                        [(left, right), (n_left, n_right), (left, n_right)],
                        recognition_bundle,
                        device,
                    )
                    l_lab, r_lab, m_lab = (
                        triad[0].get("label"),
                        triad[1].get("label"),
                        triad[2].get("label"),
                    )
                    l_t = float(triad[0].get("trust") or 0.0)
                    r_t = float(triad[1].get("trust") or 0.0)
                    m_t = float(triad[2].get("trust") or 0.0)
                    halves_scrap = (
                        _is_numeral_or_separator_label(l_lab)
                        and _is_numeral_or_separator_label(r_lab)
                    ) or (l_t < 0.55 and r_t < 0.55)
                    stem_body = (
                        _is_beth_label(m_lab)
                        or _is_heth_label(m_lab)
                        or _is_stem_body_letter_label(m_lab)
                    )
                    # Open-center 𐩨: peak sits in a column gutter. Do not use
                    # this for touching neighbours that only look like 𐩨 when
                    # glued (test-16 271-317).
                    open_center = _clear_stroke_gutter(stroke_mask, seam)
                    # Very thin NUM scraps → letter, but never across a peak as
                    # strong as an annotation-quality cut (test-26 false merges).
                    thin_scraps = (
                        halves_scrap
                        and m_t >= 0.75
                        and max(r1, r2) <= 0.22
                        and 0.28 <= rm <= 0.55
                        and cut_str < 0.85
                    )
                    # 𐩨/HETH/MEM body + thin NUM upright scrap (test-2 L3/L4).
                    # Strong mid-glyph peak must not keep the upright amputated.
                    # Incomplete 𐩨 often reads as 𐩡 (test-4 L2); accept that
                    # lookalike when the union is clearly beth/heth/mem.
                    body_l = (
                        (
                            _is_beth_label(l_lab)
                            or _is_heth_label(l_lab)
                            or _is_stem_body_letter_label(l_lab)
                            or (
                                stem_body
                                and l_lab in ("𐩡", "𐩧")
                                and l_t >= 0.55
                            )
                        )
                        and l_t >= (0.50 if _is_beth_label(l_lab) else 0.55)
                        and r1 >= 0.28
                    )
                    body_r = (
                        (
                            _is_beth_label(r_lab)
                            or _is_heth_label(r_lab)
                            or _is_stem_body_letter_label(r_lab)
                            or (
                                stem_body
                                and r_lab in ("𐩡", "𐩧")
                                and r_t >= 0.55
                            )
                        )
                        and r_t >= (0.50 if _is_beth_label(r_lab) else 0.55)
                        and r2 >= 0.28
                    )
                    thin_sep_l = (
                        r1 <= 0.14
                        and (
                            _is_numeral_or_separator_label(l_lab) or l_t < 0.55
                        )
                    )
                    thin_sep_r = (
                        r2 <= 0.14
                        and (
                            _is_numeral_or_separator_label(r_lab) or r_t < 0.55
                        )
                    )
                    body_plus_stem_scrap = (
                        stem_body
                        and cut_str < 0.95
                        and 0.35 <= rm <= 1.05
                        and (
                            m_t >= 0.85
                            or (
                                _is_beth_label(m_lab)
                                and m_t >= 0.50
                                and min(r1, r2) <= 0.14
                            )
                        )
                        and (
                            (body_l and thin_sep_r)
                            or (body_r and thin_sep_l)
                            or (
                                # Halves mislabeled, union is clearly 𐩨 (test-2 L3).
                                thin
                                and min(r1, r2) <= 0.14
                                and _is_beth_label(m_lab)
                                and m_t >= 0.90
                                and max(r1, r2) <= 0.50
                            )
                        )
                    )
                    # Do not glue a real ``|`` onto an already-complete 𐩨
                    # (test-16 L1 NUM+𐩨 dropped completeness 0.84→0.60).
                    if body_plus_stem_scrap and model is not None:
                        enhanced_line = enhance_line(line_bgr)
                        m_c = float(
                            score_crop(
                                enhanced_line, left, n_right, model, device
                            )
                        )
                        if body_l:
                            b_c = float(
                                score_crop(
                                    enhanced_line, left, right, model, device
                                )
                            )
                            thin_c = float(
                                score_crop(
                                    enhanced_line, n_left, n_right, model, device
                                )
                            )
                        else:
                            b_c = float(
                                score_crop(
                                    enhanced_line, n_left, n_right, model, device
                                )
                            )
                            thin_c = float(
                                score_crop(
                                    enhanced_line, left, right, model, device
                                )
                            )
                        # Real word ``|`` keeps ink (test-16); amputated tips do not.
                        if thin_c >= 0.28 or m_c + 0.08 < b_c:
                            body_plus_stem_scrap = False
                    # Only open-center stem-body may override a strong cut, and
                    # never a near-certain boundary peak (annotation-quality).
                    strong_letter = (
                        m_t >= 0.75
                        and not _is_numeral_or_separator_label(m_lab)
                        and (
                            (
                                stem_body
                                and halves_scrap
                                and m_t >= 0.85
                                and open_center
                                and cut_str < 0.90
                            )
                            or thin_scraps
                            or body_plus_stem_scrap
                        )
                    )
                    if (halves_scrap and strong_letter) or body_plus_stem_scrap:
                        out.append(
                            (left, n_right, max(conf, n_conf, m_t))
                        )
                        i += 2
                        rescued = True
                if not rescued:
                    out.append((left, right, conf))
                    i += 1
                continue

            # BETH bisected between uprights: 𐩨 + NUM_2 while full glyph is
            # still ~1 pitch — scan merge widths (combined box may say NUM_3).
            # Require a truly thin sep half. Letter-width neighbours misread as
            # NUM_1 (test-16) must not be absorbed.
            # Open-center 𐩨 *has* a column gutter between uprights — do not treat
            # that as a real ``|`` when the scrap is an ultra-thin upright tip
            # (test-2 L4).
            seam_gutter = _clear_stroke_gutter(stroke_mask, max(right, n_left))
            seam_cut = _boundary_cut_strength(boundary_profile, max(right, n_left))
            # Weak mid-glyph peaks: upright tip can be ~0.17·h (test-2 L3).
            # Keep the stricter 0.14 gate when the cut is annotation-strong so
            # real ``|`` (often ≥0.15·h) is not absorbed.
            tip_lim = 0.20 if seam_cut < 0.40 else 0.14
            ultra_thin_tip = min(r1, r2) <= tip_lim
            if (
                gap <= 2
                and 0.40 <= rm <= 1.30
                and thin
                and ultra_thin_tip
                and recognition_bundle is not None
                and (not seam_gutter or ultra_thin_tip)
            ):
                pair_recs = _batch_recognition_span_scores(
                    line_bgr,
                    [(left, right), (n_left, n_right)],
                    recognition_bundle,
                    device,
                )
                l_lab = pair_recs[0].get("label")
                r_lab = pair_recs[1].get("label")
                l_t = float(pair_recs[0].get("trust") or 0.0)
                r_t = float(pair_recs[1].get("trust") or 0.0)
                left_thin_tip = r1 <= tip_lim
                right_thin_tip = r2 <= tip_lim
                beth_split = (
                    _is_beth_label(l_lab) and right_thin_tip
                ) or (
                    _is_beth_label(r_lab) and left_thin_tip
                ) or (
                    # Thin tip + neighbour; span scan confirms 𐩨 (test-2 L3).
                    # Incomplete 𐩨 often reads as 𐩡 with body ~0.50·h (test-4 L2).
                    (left_thin_tip or right_thin_tip)
                    and max(r1, r2) <= 0.55
                    and (
                        _is_beth_label(l_lab)
                        or _is_beth_label(r_lab)
                        or l_lab in ("𐩡", "𐩧")
                        or r_lab in ("𐩡", "𐩧")
                        or _is_numeral_or_separator_label(l_lab)
                        or _is_numeral_or_separator_label(r_lab)
                    )
                )
                if beth_split:
                    # Exact tip+body can be a weak 𐩨 when the far upright was
                    # absorbed by the following letter (test-2 L3 → need ~+6px).
                    scan_limit = n_right
                    third: tuple[int, int, float] | None = None
                    if i + 2 < len(boxes):
                        nn_left, nn_right, nn_conf = boxes[i + 2]
                        if nn_left - n_right <= 2:
                            steal = max(8, int(0.22 * line_height))
                            min_rem = max(6, int(0.12 * line_height))
                            scan_limit = min(nn_right - min_rem, n_right + steal)
                            if scan_limit > n_right:
                                third = (nn_left, nn_right, nn_conf)
                    step = max(1, max(scan_limit - n_left, 1) // 8)
                    ends = list(range(n_left, scan_limit + 1, step))
                    if n_right not in ends:
                        ends.append(n_right)
                    if scan_limit not in ends:
                        ends.append(scan_limit)
                    min_w = max(8, int(0.28 * line_height))
                    spans = [
                        (left, end)
                        for end in ends
                        if end - left >= min_w and end > right
                    ]
                    if spans:
                        span_recs = _batch_recognition_span_scores(
                            line_bgr, spans, recognition_bundle, device
                        )
                        # Trust alone picks incomplete 𐩨 crops (test-2 L3
                        # end=481 @1.0 before the true end ~493). Rank by
                        # completeness+trust joint when the net is available.
                        enhanced_line = (
                            enhance_line(line_bgr) if model is not None else None
                        )
                        best_end: int | None = None
                        best_conf = 0.0
                        best_rank = -1.0
                        for (a0, end), rec in zip(spans, span_recs):
                            if not _is_beth_label(rec.get("label")):
                                continue
                            t = float(rec.get("trust") or 0.0)
                            if enhanced_line is not None:
                                comp = float(
                                    score_crop(
                                        enhanced_line, a0, end, model, device
                                    )
                                )
                                rank = _span_joint_score(comp, rec)
                            else:
                                rank = t
                            if rank > best_rank + 1e-6 or (
                                abs(rank - best_rank) <= 1e-6 and t > best_conf
                            ):
                                best_rank = rank
                                best_conf = t
                                best_end = end
                        # Stealing from the next letter requires a clear 𐩨.
                        need = (
                            0.85
                            if (
                                third is not None
                                and best_end is not None
                                and best_end > n_right
                            )
                            else 0.55
                        )
                        if (
                            best_end is not None
                            and best_conf >= need
                            and best_end > right
                        ):
                            out.append(
                                (
                                    left,
                                    best_end,
                                    max(conf, n_conf, best_conf),
                                )
                            )
                            if third is not None and best_end > n_right:
                                if best_end < third[1]:
                                    out.append(
                                        (best_end, third[1], third[2])
                                    )
                                i += 3
                            else:
                                if best_end < n_right:
                                    out.append(
                                        (best_end, n_right, n_conf)
                                    )
                                i += 2
                            continue

            # BETH mid-bisect (test-27 L2): after thin-bar merge, uprights are
            # ~letter-width NUM_* / lookalike halves. Exact union often scores
            # NUM_3; scan start/end for 𐩨. Open-center 𐩨 has a column gutter,
            # so do not require no-gutter. Only letter-width pairs (not the
            # earlier thin+body triple stage) to avoid remnant inflation.
            if (
                gap <= 2
                and recognition_bundle is not None
                # Half-pitch uprights can land ~0.22–0.28·h after upscale
                # (test-2 L3 𐩨 NUM_1+𐩡).
                and 0.20 <= r1 <= 0.55
                and 0.20 <= r2 <= 0.55
                and 0.45 <= rm <= 1.25
            ):
                pair_recs = _batch_recognition_span_scores(
                    line_bgr,
                    [(left, right), (n_left, n_right)],
                    recognition_bundle,
                    device,
                )
                l_lab = pair_recs[0].get("label")
                r_lab = pair_recs[1].get("label")
                l_t = float(pair_recs[0].get("trust") or 0.0)
                r_t = float(pair_recs[1].get("trust") or 0.0)
                sep_l = _is_numeral_or_separator_label(l_lab)
                sep_r = _is_numeral_or_separator_label(r_lab)
                if sep_l or sep_r or _is_beth_label(l_lab) or _is_beth_label(r_lab):
                    strong_l = (
                        (not sep_l)
                        and l_t >= 0.85
                        and not _is_beth_label(l_lab)
                    )
                    strong_r = (
                        (not sep_r)
                        and r_t >= 0.85
                        and not _is_beth_label(r_lab)
                    )
                    # Real neighbours, or strong letter + NUM bar.
                    if not (strong_l and strong_r):
                        absorb_soft = (strong_l and sep_r) or (
                            strong_r and sep_l
                        )
                        beth_min = 0.82 if absorb_soft else 0.65
                        min_w = max(8, int(0.40 * line_height))
                        max_w = int(0.85 * line_height)
                        start_step = max(1, (right - left) // 10)
                        end_step = max(1, (n_right - n_left) // 10)
                        starts = list(range(left, right, start_step))
                        if left not in starts:
                            starts.insert(0, left)
                        ends = list(range(n_left, n_right + 1, end_step))
                        if n_right not in ends:
                            ends.append(n_right)
                        spans = [
                            (s, e)
                            for s in starts
                            for e in ends
                            if min_w <= (e - s) <= max_w
                            and e > right
                            and s < n_left
                        ]
                        if spans:
                            span_recs = _batch_recognition_span_scores(
                                line_bgr,
                                spans,
                                recognition_bundle,
                                device,
                            )
                            enhanced_line = (
                                enhance_line(line_bgr) if model is not None else None
                            )
                            best_s: int | None = None
                            best_e: int | None = None
                            best_conf = 0.0
                            for (s, e), rec in zip(spans, span_recs):
                                t = float(rec.get("trust") or 0.0)
                                if not _is_beth_label(rec.get("label")):
                                    continue
                                if t < beth_min:
                                    continue
                                # Trust-only picks |+letter as 𐩨 (test-16 L1
                                # NUM@c0.77 + 𐩡@c0.77 → 𐩨@c0.55).
                                if enhanced_line is not None:
                                    comp = float(
                                        score_crop(
                                            enhanced_line, s, e, model, device
                                        )
                                    )
                                    if comp < 0.70:
                                        continue
                                if t > best_conf:
                                    best_conf = t
                                    best_s, best_e = s, e
                            if (
                                best_s is not None
                                and best_e is not None
                                and best_conf >= beth_min
                                and (best_e - best_s) >= int(0.55 * (n_right - left))
                            ):
                                # Absorb tiny side remnants into the Beth box
                                # so mid-bisect does not inflate letter counts.
                                slim = max(4, int(0.14 * line_height))
                                if best_s - left <= slim:
                                    best_s = left
                                if n_right - best_e <= slim:
                                    best_e = n_right
                                if best_s > left:
                                    out.append((left, best_s, conf))
                                out.append(
                                    (
                                        best_s,
                                        best_e,
                                        max(conf, n_conf, best_conf),
                                    )
                                )
                                if best_e < n_right:
                                    out.append((best_e, n_right, n_conf))
                                i += 2
                                continue

            if gap <= 2 and thin and 0.40 <= rm <= 1.15 and max(r1, r2) <= 0.95:
                spans = [(left, right), (n_left, n_right), (left, n_right)]
                rec_l = rec_r = rec_m = None
                if recognition_bundle is not None:
                    recs = _batch_recognition_span_scores(
                        line_bgr, spans, recognition_bundle, device
                    )
                    rec_l, rec_r, rec_m = recs

                # Prefer recognition labels; fall back to identify names.
                if rec_m is not None:
                    m_label = rec_m.get("label")
                    m_conf = float(rec_m.get("trust") or 0.0)
                    l_label = rec_l.get("label") if rec_l else None
                    l_conf = float(rec_l.get("trust") or 0.0) if rec_l else 0.0
                    r_label = rec_r.get("label") if rec_r else None
                    r_conf = float(rec_r.get("trust") or 0.0) if rec_r else 0.0
                else:
                    m_label, m_conf = _identify_crop_for_detect_merge(
                        line_bgr, left, n_right, device
                    )
                    l_label, l_conf = _identify_crop_for_detect_merge(
                        line_bgr, left, right, device
                    )
                    r_label, r_conf = _identify_crop_for_detect_merge(
                        line_bgr, n_left, n_right, device
                    )

                stem_l = _is_numeral_or_separator_label(l_label)
                stem_r = _is_numeral_or_separator_label(r_label)
                # HETH bisected down the middle → NUM_1 + NUM_1; rejoin.
                if (
                    stem_l
                    and stem_r
                    and gap <= 2
                    and 0.35 <= rm <= 1.20
                    and rec_m is not None
                    and _is_heth_label(m_label)
                    and m_conf >= 0.70
                ):
                    out.append((left, n_right, max(conf, n_conf, float(m_conf))))
                    i += 2
                    continue
                if stem_l ^ stem_r and _is_stem_body_letter_label(m_label) and m_conf >= 0.85:
                    body_label = r_label if stem_l else l_label
                    body_conf = r_conf if stem_l else l_conf
                    # Already-complete stem-body letter: keep a real neighbouring ``|``.
                    if (
                        _is_stem_body_letter_label(body_label)
                        and body_conf >= 0.75
                        and m_conf < body_conf + 0.08
                    ):
                        out.append((left, right, conf))
                        i += 1
                        continue

                    body_lookalike = False
                    if _is_mem_label(m_label):
                        body_lookalike = body_label is not None and (
                            str(body_label) in _MEM_BODY_LOOKALIKES
                            or str(body_label).upper() in _MEM_BODY_LOOKALIKES
                        )
                    if _is_heth_label(m_label):
                        # Incomplete HETH half often confuses as TAW / SHIN / KAPH.
                        body_lookalike = not _is_heth_label(body_label)

                    # RESH-like MEM halves often get trust≈1.0 — still rejoin when
                    # the merge is strongly MEM and the body is not already MEM.
                    mem_resh_half = (
                        _is_mem_label(m_label)
                        and body_lookalike
                        and m_conf >= 0.88
                        and not _is_mem_label(body_label)
                    )
                    heth_half = (
                        _is_heth_label(m_label)
                        and body_lookalike
                        and m_conf >= 0.88
                        and body_conf < 0.90
                    )
                    if m_conf >= body_conf + 0.12 or mem_resh_half or heth_half:
                        out.append(
                            (left, n_right, max(conf, n_conf, float(m_conf)))
                        )
                        i += 2
                        continue
        out.append((left, right, conf))
        i += 1
    return out


def _span_joint_score(comp: float, rec: dict) -> float:
    """Combine completeness + musnad_final trust for local cut repair."""
    trust = float(rec.get("trust") or 0.0)
    label = rec.get("label")
    sep = _is_numeral_or_separator_label(label)
    return float(
        np.clip(0.40 * float(comp) + 0.55 * trust + (0.05 if sep else 0.0), 0.0, 1.0)
    )


def _repair_adjacent_separators_by_law(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    device: torch.device,
    line_height: int,
    recognition_bundle: tuple[torch.nn.Module, list[str], dict | None] | None = None,
    model: LetterCompletenessNet | None = None,
) -> list[tuple[int, int, float]]:
    """
    Musnad layout law: never two ``|`` / NUM bars with no letter between them.

    A run of adjacent separator-looking boxes is usually a split multi-stem
    letter (𐩨 / 𐩢 / …) optionally flanked by real word separators. Search the
    run for a complete letter span; keep at most one thin leftover bar on each
    side. If no letter is found, collapse the whole run into one box.
    """
    if (
        len(boxes) < 2
        or line_height <= 0
        or recognition_bundle is None
        or model is None
    ):
        return boxes

    enhanced = enhance_line(line_bgr)
    spans = [(a, b) for a, b, _c in boxes]
    recs = _batch_recognition_span_scores(
        line_bgr, spans, recognition_bundle, device
    )
    is_sep = [_is_numeral_or_separator_label(r.get("label")) for r in recs]

    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        if not is_sep[i]:
            out.append(boxes[i])
            i += 1
            continue
        j = i
        while j + 1 < len(boxes) and is_sep[j + 1]:
            gap = boxes[j + 1][0] - boxes[j][1]
            if gap > 3:
                break
            j += 1
        if j == i:
            out.append(boxes[i])
            i += 1
            continue

        run_left = boxes[i][0]
        run_right = boxes[j][1]
        atom_edges = sorted(
            {
                boxes[k][0]
                for k in range(i, j + 1)
            }
            | {
                boxes[k][1]
                for k in range(i, j + 1)
            }
            | {run_left, run_right}
        )
        min_letter = max(10, int(0.28 * line_height))
        max_letter = int(0.90 * line_height)
        min_bar = max(4, int(0.08 * line_height))
        max_bar = int(0.34 * line_height)
        step = max(2, int(0.04 * line_height))

        starts = list(range(run_left, run_right - min_letter + 1, step))
        ends = list(range(run_left + min_letter, run_right + 1, step))
        for e in atom_edges:
            if run_left <= e <= run_right - min_letter and e not in starts:
                starts.append(e)
            if run_left + min_letter <= e <= run_right and e not in ends:
                ends.append(e)
        starts = sorted(set(starts))
        ends = sorted(set(ends))

        cand_spans = [
            (s, e)
            for s in starts
            for e in ends
            if min_letter <= (e - s) <= max_letter and e > s
        ]
        # Cap lattice size on long runs.
        if len(cand_spans) > 220:
            stride = max(1, len(cand_spans) // 220)
            cand_spans = cand_spans[::stride]
            if (run_left, run_right) not in cand_spans:
                cand_spans.append((run_left, run_right))

        best: tuple[float, int, int, float] | None = None
        if cand_spans:
            span_recs = _batch_recognition_span_scores(
                line_bgr, cand_spans, recognition_bundle, device
            )
            span_comps = _score_crops_batch(
                enhanced, cand_spans, model, device
            )
            for (s, e), rec, comp in zip(cand_spans, span_recs, span_comps):
                lab = rec.get("label")
                if _is_numeral_or_separator_label(lab):
                    continue
                trust = float(rec.get("trust") or 0.0)
                if comp < 0.45 and trust < 0.75:
                    continue
                if comp < 0.35:
                    continue
                # Prefer a letter that leaves thin bar leftovers on the sides
                # (pattern ``| letter |``), which is the usual false || split.
                left_w = s - run_left
                right_w = run_right - e
                side_ok = True
                if left_w > 0 and not (min_bar <= left_w <= max_bar):
                    if left_w > max_bar:
                        side_ok = False
                if right_w > 0 and not (min_bar <= right_w <= max_bar):
                    if right_w > max_bar:
                        side_ok = False
                score = (
                    0.55 * comp
                    + 0.35 * trust
                    + (0.10 if side_ok else 0.0)
                    + (
                        0.08
                        if (
                            _is_beth_label(lab)
                            or _is_heth_label(lab)
                            or _is_stem_body_letter_label(lab)
                        )
                        else 0.0
                    )
                )
                if best is None or score > best[0]:
                    best = (score, s, e, max(trust, comp))

        if best is not None:
            _sc, s, e, conf = best
            if s - run_left >= min_bar:
                out.append((run_left, s, conf))
            elif s > run_left:
                # Tiny scrap — absorb into the letter.
                s = run_left
            out.append((s, e, conf))
            if run_right - e >= min_bar:
                out.append((e, run_right, conf))
            elif e < run_right:
                # Extend letter instead of leaving a second thin scrap that
                # would recreate || with the following separator.
                out[-1] = (s, run_right, conf)
            i = j + 1
            continue

        # No letter recovered: collapse the illegal || run into one box.
        run_conf = max(c for _a, _b, c in boxes[i : j + 1])
        out.append((run_left, run_right, run_conf))
        i = j + 1
    return out


def _rejoin_by_recognition(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    device: torch.device,
    line_height: int,
    recognition_bundle: tuple[torch.nn.Module, list[str], dict | None] | None = None,
    boundary_profile: np.ndarray | None = None,
    model: LetterCompletenessNet | None = None,
) -> list[tuple[int, int, float]]:
    """
    Local cut repair with musnad_final + completeness (any letter).

    For each suspicious adjacent pair, search the combined span for the best
    partition: keep as-is, move the cut, or merge. Half-cuts rejoin when the
    union is one complete letter; multi-letter glues lose to a 2-way split that
    scores two strong letters.
    """
    if (
        len(boxes) < 2
        or line_height <= 0
        or recognition_bundle is None
        or model is None
    ):
        return boxes

    enhanced = enhance_line(line_bgr)
    pitch = _estimate_letter_pitch(boxes, line_height)
    min_part = max(6, int(0.12 * line_height))
    step = max(2, int(0.04 * line_height))
    seg_cost = 0.55
    from .letter_boundary_net import letter_activity_profile

    activity = letter_activity_profile(line_bgr)
    stroke_mask = _carved_letter_mask(line_bgr)
    n_act = int(activity.size)

    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        left, right, conf = boxes[i]
        if i + 1 >= len(boxes):
            out.append((left, right, conf))
            i += 1
            continue

        n_left, n_right, n_conf = boxes[i + 1]
        gap = n_left - right
        wm = n_right - left
        if gap > max(6, int(0.12 * line_height)):
            out.append((left, right, conf))
            i += 1
            continue

        # One or two letter widths only.
        if wm < 0.55 * pitch or wm > 2.70 * pitch:
            out.append((left, right, conf))
            i += 1
            continue

        w1 = right - left
        w2 = n_right - n_left
        thin = min(w1, w2) <= 0.28 * line_height
        # Two letter-width halves of one glyph (test-8 L1 X/𐩩): both confs
        # are high so the usual suspicious gates miss them.
        half_pair = (
            0.22 * line_height <= min(w1, w2) <= 0.42 * line_height
            and 0.28 * line_height <= max(w1, w2) <= 0.48 * line_height
            and wm <= max(1.35 * pitch, 0.72 * line_height)
        )
        seam = max(right, n_left)
        s0 = min(max(int(seam), 1), max(n_act - 2, 1))
        peak_u = (
            float(np.max(activity[left:n_right]) + 1e-6)
            if n_right > left and n_act
            else 1.0
        )
        act_s = float(activity[s0]) if n_act else 0.0
        # 𐩥 / phi mid-bar: the cut sits on a stem *peak*, not a letter gap.
        # Both halves often score complete, so conf/cut-strength gates skip.
        seam_peak = (
            act_s >= 0.40 * peak_u
            and act_s >= float(activity[max(s0 - 2, 0)])
            and act_s >= float(activity[min(s0 + 2, n_act - 1)])
        )
        stem_pair = (
            gap <= 2
            and min(w1, w2) >= 0.16 * line_height
            and max(w1, w2) <= 0.55 * line_height
            and wm <= max(1.55 * pitch, 0.80 * line_height)
            and (
                seam_peak
                or _cut_bisects_connected_stroke(stroke_mask, seam)
            )
        )
        suspicious = (
            conf < 0.62
            or n_conf < 0.62
            or thin
            or half_pair
            or stem_pair
            or _boundary_cut_strength(boundary_profile, seam) < 0.35
        )
        if not suspicious:
            out.append((left, right, conf))
            i += 1
            continue

        cuts = list(range(left + min_part, n_right - min_part + 1, step))
        if right not in cuts and left + min_part <= right <= n_right - min_part:
            cuts.append(right)
        if n_left not in cuts and left + min_part <= n_left <= n_right - min_part:
            cuts.append(n_left)
        cuts = sorted(set(cuts))

        spans: list[tuple[int, int]] = [(left, n_right)]
        for cut in cuts:
            spans.append((left, cut))
            spans.append((cut, n_right))
        uniq: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for sp in spans:
            if sp not in seen and sp[1] - sp[0] >= 3:
                seen.add(sp)
                uniq.append(sp)

        recs = _batch_recognition_span_scores(
            line_bgr, uniq, recognition_bundle, device
        )
        comps = _score_crops_batch(enhanced, uniq, model, device)
        score_map = {
            sp: (_span_joint_score(comp, rec), float(comp), rec)
            for sp, comp, rec in zip(uniq, comps, recs)
        }

        merge_j, merge_comp, merge_rec = score_map[(left, n_right)]
        cur_l = score_map.get((left, right))
        cur_r = score_map.get((n_left, n_right))
        if cur_l is None or cur_r is None:
            cur_spans = [(left, right), (n_left, n_right)]
            cur_recs = _batch_recognition_span_scores(
                line_bgr, cur_spans, recognition_bundle, device
            )
            cur_comps = _score_crops_batch(enhanced, cur_spans, model, device)
            cur_l = (
                _span_joint_score(cur_comps[0], cur_recs[0]),
                cur_comps[0],
                cur_recs[0],
            )
            cur_r = (
                _span_joint_score(cur_comps[1], cur_recs[1]),
                cur_comps[1],
                cur_recs[1],
            )
        current_score = cur_l[0] + cur_r[0] - seg_cost

        best_score = current_score
        best: tuple[str, int | None] = ("keep", None)

        one_letter = wm <= max(1.25 * pitch, 0.58 * line_height)
        one_or_wide = wm <= max(1.85 * pitch, 0.88 * line_height)
        merge_lab = merge_rec.get("label")
        thin_sep_body = (
            thin
            and min(w1, w2) <= 0.14 * line_height
            and max(w1, w2) >= 0.22 * line_height
        )
        cut_str = _boundary_cut_strength(
            boundary_profile, max(right, n_left)
        )
        # Completeness-dominant false split (test-2 L3 𐩨): open-center
        # letters often bisect into confident lookalike halves (NUM_1+𐩡)
        # whose joint trust beats the true union when prototypes disagree
        # (𐩨 @ low trust). Completeness of the union is the reliable signal;
        # real ``|``+letter merges stay incomplete (~0.0–0.1).
        # Empty NUM scrap next to an incomplete letter: allow a stronger cut
        # (test-4 L1 𐩡 left upright peeled as NUM).
        empty_scrap = min(float(cur_l[1]), float(cur_r[1])) <= 0.15
        completeness_merge = (
            one_letter
            and wm <= max(1.15 * pitch, 0.62 * line_height)
            and merge_comp >= 0.58
            and merge_comp >= float(cur_l[1]) + 0.18
            and merge_comp >= float(cur_r[1]) + 0.12
            and not _is_numeral_or_separator_label(merge_lab)
            and (
                cut_str < (0.85 if empty_scrap else 0.55)
                or (
                    thin
                    and min(w1, w2) <= 0.22 * line_height
                    and merge_comp >= 0.62
                )
            )
            and min(w1, w2) <= 0.34 * line_height
            and max(w1, w2) <= 0.48 * line_height
        )
        # Open-center / two-stem letter split into two bodies (test-2).
        # Wider than one_letter so empty-gap grow cannot reclaim the other stem.
        completeness_merge = completeness_merge or (
            one_or_wide
            and merge_comp >= 0.68
            and merge_comp >= max(float(cur_l[1]), float(cur_r[1])) + 0.16
            and not _is_numeral_or_separator_label(merge_lab)
            and min(w1, w2) >= 0.14 * line_height
            and max(w1, w2) <= 0.62 * line_height
            and gap <= max(3, int(0.08 * line_height))
        )
        # Empty grain tip (c≈0 NUM) next to an already-complete letter body
        # (test-4 L2 𐩨/𐩲 tips; test-8 L1 𐩹 upright ~0.18·h). Body completeness
        # alone does not rise, so the delta gate above misses these — absorb
        # the tip when the union stays a non-separator letter.
        empty_tip_absorb = (
            one_letter
            and thin
            and min(w1, w2) <= 0.22 * line_height
            and merge_comp >= 0.55
            and not _is_numeral_or_separator_label(merge_lab)
            and merge_comp + 0.05 >= max(float(cur_l[1]), float(cur_r[1]))
            and (
                (
                    float(cur_l[1]) < 0.20
                    and _is_numeral_or_separator_label(cur_l[2].get("label"))
                    and float(cur_r[1]) >= 0.50
                )
                or (
                    float(cur_r[1]) < 0.20
                    and _is_numeral_or_separator_label(cur_r[2].get("label"))
                    and float(cur_l[1]) >= 0.50
                )
            )
        )
        # Peeled upright: the scrap often scores as a complete ``|`` (high
        # completeness), so empty_tip_absorb never fires. A true ``|``+letter
        # union stays incomplete; a letter missing its own stem does not.
        stem_reattach = (
            thin
            and min(w1, w2) <= 0.22 * line_height
            and max(w1, w2) >= 0.28 * line_height
            and wm <= max(1.35 * pitch, 0.72 * line_height)
            and merge_comp >= 0.52
            and not _is_numeral_or_separator_label(merge_lab)
            and merge_comp + 0.04 >= max(float(cur_l[1]), float(cur_r[1]))
        )
        # Round 𐩥/𐩲 mid-cut into weak NUM + lookalike (test-8 L2).
        round_half_absorb = (
            one_letter
            and _is_round_glue_label(merge_lab)
            and merge_comp >= 0.55
            and float(merge_rec.get("trust") or 0.0) >= 0.65
            and min(w1, w2) <= 0.30 * line_height
            and (
                (
                    _is_numeral_or_separator_label(cur_l[2].get("label"))
                    and float(cur_l[1]) < 0.35
                    and float(cur_r[1]) >= 0.45
                )
                or (
                    _is_numeral_or_separator_label(cur_r[2].get("label"))
                    and float(cur_r[1]) < 0.35
                    and float(cur_l[1]) >= 0.45
                )
            )
        )
        # 𐩥 split on its middle bar: the ring scores as 𐩲/𐩥 and the bar is a
        # thin low-trust scrap (often also labelled 𐩥). A real word ``|`` has
        # high NUM trust and similar width to other separators — do not absorb.
        lab_l = cur_l[2].get("label")
        lab_r = cur_r[2].get("label")
        t_l = float(cur_l[2].get("trust") or 0.0)
        t_r = float(cur_r[2].get("trust") or 0.0)
        thin_left = w1 <= min(0.18 * line_height, 0.45 * max(w1, w2))
        thin_right = w2 <= min(0.18 * line_height, 0.45 * max(w1, w2))
        round_scrap_absorb = (
            wm <= max(1.50 * pitch, 0.78 * line_height)
            and max(w1, w2) >= 0.28 * line_height
            and (
                _is_round_glue_label(merge_lab)
                or _is_round_glue_label(lab_l)
                or _is_round_glue_label(lab_r)
            )
            and not _is_numeral_or_separator_label(merge_lab)
            and (
                (
                    thin_right
                    and _is_round_glue_label(lab_l)
                    and t_l >= 0.80
                    and t_r < 0.55
                )
                or (
                    thin_left
                    and _is_round_glue_label(lab_r)
                    and t_r >= 0.80
                    and t_l < 0.55
                )
            )
        )
        def _peak_frac(a: int, b: int) -> float:
            sl = activity[a:b] if n_act else None
            if sl is None or sl.size < 3:
                return 0.5
            return float(int(np.argmax(sl))) / float(max(len(sl) - 1, 1))

        # 𐩥 cut on the middle bar: that bar scores as a *complete* ``|``
        # (high NUM trust), so the low-trust scrap rule above never fires.
        # A real word ``|`` after a *complete* 𐩥 sits in a low-activity gap.
        # The amputated ring is incomplete and its ink peak is on the seam.
        seam_on_ink = act_s >= 0.28
        pf_l = _peak_frac(left, right)
        pf_r = _peak_frac(n_left, n_right)
        round_bar_reattach = (
            gap <= 2
            and wm <= max(1.60 * pitch, 0.82 * line_height)
            and min(w1, w2) <= 0.28 * line_height
            and max(w1, w2) >= 0.22 * line_height
            and seam_on_ink
            and not _seam_is_between_two_bodies(activity, left, seam, n_right)
            and (
                (
                    _is_numeral_or_separator_label(lab_l)
                    and float(cur_l[1]) < 0.40
                    and _is_round_glue_label(lab_r)
                    and float(cur_r[1]) < 0.45
                    and pf_r <= 0.38
                )
                or (
                    _is_numeral_or_separator_label(lab_r)
                    and float(cur_r[1]) < 0.40
                    and _is_round_glue_label(lab_l)
                    and float(cur_l[1]) < 0.45
                    and pf_l >= 0.62
                )
            )
        )
        # Two halves of 𐩥: the cut sits on the shared stem *peak* (connected
        # ink). A word ``|`` after a finished 𐩥 sits in a valley — keep it.
        # The bar half often scores as a complete NUM; that is the split, not
        # a reason to refuse the merge.
        between_bodies = _seam_is_between_two_bodies(
            activity, left, seam, n_right
        )
        stem_peak_now = (
            act_s >= 0.22
            and act_s >= float(activity[max(s0 - 2, 0)])
            and act_s >= float(activity[min(s0 + 2, n_act - 1)])
        )
        phi_halves = (
            gap <= 2
            and wm <= max(1.45 * pitch, 0.70 * line_height)
            and min(w1, w2) >= 0.16 * line_height
            and max(w1, w2) <= 0.50 * line_height
            and max(w1, w2) <= 1.55 * min(w1, w2)
            and stem_peak_now
            and not between_bodies
            and pf_l >= 0.42
            and pf_r <= 0.58
            and _cut_bisects_connected_stroke(stroke_mask, seam)
            and not _clear_stroke_gutter(stroke_mask, seam)
            and (
                _is_round_glue_label(lab_l)
                or _is_round_glue_label(lab_r)
                or _is_round_glue_label(merge_lab)
            )
        )
        merge_bonus = 0.0
        if (
            thin_sep_body
            and merge_comp >= 0.55
            and not _is_numeral_or_separator_label(merge_lab)
        ):
            # Thin upright scrap + body: prefer the complete glyph strongly.
            merge_bonus = 0.22 if merge_comp >= 0.70 else 0.12
            if (
                _is_beth_label(merge_lab)
                or _is_heth_label(merge_lab)
                or _is_stem_body_letter_label(merge_lab)
            ):
                merge_bonus = max(merge_bonus, 0.25)
        if completeness_merge or empty_tip_absorb or round_half_absorb or stem_reattach or round_scrap_absorb or round_bar_reattach or phi_halves:
            best_score = merge_j + 1.0
            best = ("merge", None)
        elif (
            one_letter
            and merge_comp >= 0.40
            and merge_j >= 0.72
            and not _is_numeral_or_separator_label(merge_lab)
            and merge_j - seg_cost * 0.35 + merge_bonus >= best_score + 0.02
            # Do not glue two already-complete letters into a false 𐩨/𐩢
            # (test-2 L1 𐩧@c0.92 + weak 𐩬 → 𐩨). A thin stem scrap is not
            # a second letter — that is the missing upright of the body.
            and not (
                not thin
                and (
                    _is_beth_label(merge_lab)
                    or _is_heth_label(merge_lab)
                )
                and (
                    (
                        float(cur_l[1]) >= 0.85
                        and not _is_numeral_or_separator_label(
                            cur_l[2].get("label")
                        )
                        and not _is_beth_label(cur_l[2].get("label"))
                        and not _is_heth_label(cur_l[2].get("label"))
                    )
                    or (
                        float(cur_r[1]) >= 0.85
                        and not _is_numeral_or_separator_label(
                            cur_r[2].get("label")
                        )
                        and not _is_beth_label(cur_r[2].get("label"))
                        and not _is_heth_label(cur_r[2].get("label"))
                    )
                )
            )
        ):
            best_score = merge_j - seg_cost * 0.35 + merge_bonus
            best = ("merge", None)

        # Letter mid-bisect: both halves look like letters, but the union is a
        # strong complete glyph (test-8 L1 X/𐩩 → 𐩧+𐩰). Force merge only when
        # the union is clearly more complete than either half — otherwise dense
        # lines (test-9) over-merge two real neighbours.
        letter_midbisect = (
            merge_comp >= 0.80
            and float(merge_rec.get("trust") or 0.0) >= 0.70
            and not _is_numeral_or_separator_label(merge_lab)
            and not _is_numeral_or_separator_label(cur_l[2].get("label"))
            and not _is_numeral_or_separator_label(cur_r[2].get("label"))
            and min(w1, w2) >= 0.20 * line_height
            and wm <= max(1.45 * pitch, 0.75 * line_height)
            and merge_comp
            >= max(float(cur_l[1]), float(cur_r[1])) + 0.05
        )
        if letter_midbisect or round_scrap_absorb or round_bar_reattach or phi_halves:
            best_score = merge_j + 1.0
            best = ("merge", None)

        if not (
            completeness_merge
            or empty_tip_absorb
            or round_half_absorb
            or letter_midbisect
            or round_scrap_absorb
            or round_bar_reattach
            or phi_halves
        ):
            for cut in cuts:
                j_l, _c_l, rec_l = score_map[(left, cut)]
                j_r, _c_r, rec_r = score_map[(cut, n_right)]
                if (
                    _is_numeral_or_separator_label(rec_l.get("label"))
                    and _is_numeral_or_separator_label(rec_r.get("label"))
                    and max(cut - left, n_right - cut) > 0.34 * line_height
                ):
                    continue
                score = j_l + j_r - seg_cost
                if score > best_score + 0.04:
                    best_score = score
                    best = ("split", cut)

        kind, cut = best
        if kind == "merge":
            out.append(
                (
                    left,
                    n_right,
                    max(conf, n_conf, float(merge_rec.get("trust") or 0.0)),
                )
            )
            i += 2
            continue
        if kind == "split" and cut is not None:
            _j_l, _c_l, rec_l = score_map[(left, cut)]
            _j_r, _c_r, rec_r = score_map[(cut, n_right)]
            out.append((left, cut, float(rec_l.get("trust") or _j_l)))
            out.append((cut, n_right, float(rec_r.get("trust") or _j_r)))
            i += 2
            continue

        out.append((left, right, conf))
        i += 1
    return out


def _peel_glued_separators_by_recognition(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    device: torch.device,
    line_height: int,
    recognition_bundle: tuple[torch.nn.Module, list[str], dict | None] | None = None,
    model: LetterCompletenessNet | None = None,
) -> list[tuple[int, int, float]]:
    """
    Peel a glued ``|`` / NUM_1 off a letter inside one box (e.g. ``|``+𐩥).

    Pair-based cut repair cannot see this: the bar and letter are already one
    high-confidence span. Compare thin edge peels vs the merge with musnad_final
    + completeness; keep the peel when it wins.
    """
    if (
        not boxes
        or line_height <= 0
        or recognition_bundle is None
        or model is None
    ):
        return boxes

    enhanced = enhance_line(line_bgr)
    pitch = _estimate_letter_pitch(boxes, line_height)
    min_bar = max(6, int(0.08 * line_height))
    max_bar = max(min_bar + 1, int(0.30 * line_height))
    step = max(2, int(0.03 * line_height))
    seg_cost = 0.55
    out: list[tuple[int, int, float]] = []

    for left, right, conf in boxes:
        width = right - left
        # Slightly wider than one pitch — classic bar+letter glue.
        if width < 0.90 * pitch or width > 1.70 * pitch:
            out.append((left, right, conf))
            continue
        if width < min_bar + max(8, int(0.22 * line_height)):
            out.append((left, right, conf))
            continue

        cuts: list[int] = []
        for bar in range(min_bar, max_bar + 1, step):
            if left + bar <= right - min_bar:
                cuts.append(left + bar)
            if right - bar >= left + min_bar:
                cuts.append(right - bar)
        cuts = sorted(set(cuts))
        if not cuts:
            out.append((left, right, conf))
            continue

        spans: list[tuple[int, int]] = [(left, right)]
        for cut in cuts:
            spans.append((left, cut))
            spans.append((cut, right))
        uniq: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for sp in spans:
            if sp not in seen and sp[1] - sp[0] >= 3:
                seen.add(sp)
                uniq.append(sp)

        recs = _batch_recognition_span_scores(
            line_bgr, uniq, recognition_bundle, device
        )
        comps = _score_crops_batch(enhanced, uniq, model, device)
        score_map = {
            sp: (_span_joint_score(comp, rec), float(comp), rec)
            for sp, comp, rec in zip(uniq, comps, recs)
        }
        merge_j, merge_comp, merge_rec = score_map[(left, right)]
        merge_lab = merge_rec.get("label")
        intact_stem_body = (
            merge_comp >= 0.55
            and float(merge_rec.get("trust") or 0.0) >= 0.75
            and (
                _is_beth_label(merge_lab)
                or _is_heth_label(merge_lab)
                or _is_stem_body_letter_label(merge_lab)
            )
        )

        best: tuple[float, int, float, float, bool] | None = None

        for cut in cuts:
            j_l, c_l, rec_l = score_map[(left, cut)]
            j_r, c_r, rec_r = score_map[(cut, right)]
            w_l = cut - left
            w_r = right - cut
            sep_l = _is_numeral_or_separator_label(rec_l.get("label"))
            sep_r = _is_numeral_or_separator_label(rec_r.get("label"))
            peel_l = sep_l and min_bar <= w_l <= max_bar and not sep_r
            peel_r = sep_r and min_bar <= w_r <= max_bar and not sep_l
            # Empty right/left margin misread as 𐩡 (test-16 L1 𐩧+16px → 𐩣).
            junk_l = c_l < 0.15 and min_bar <= w_l <= max_bar
            junk_r = c_r < 0.15 and min_bar <= w_r <= max_bar
            strong_l = (not sep_l) and c_l >= 0.85 and j_l >= 0.70
            strong_r = (not sep_r) and c_r >= 0.85 and j_r >= 0.70
            junk_peel = (junk_l and strong_r) or (junk_r and strong_l)
            kept_lab = rec_r.get("label") if junk_l else rec_l.get("label")
            # Trim empty margin off a misread 𐩣/… when the kept glyph is a
            # different complete letter (test-16 𐩧+gap → 𐩣). Do not trim
            # intact 𐩨/𐩢 themselves (that eats the next letter).
            junk_ok = junk_peel and (
                not intact_stem_body
                or (
                    kept_lab != merge_lab
                    and not _is_beth_label(kept_lab)
                    and not _is_heth_label(kept_lab)
                    and not _is_stem_body_letter_label(kept_lab)
                )
            )
            if intact_stem_body and not junk_ok:
                continue
            if not (peel_l or peel_r or junk_ok):
                continue
            if junk_ok:
                letter_j = j_r if junk_l else j_l
                score = letter_j + 0.20
                if best is None or score > best[0]:
                    best = (score, cut, j_l, j_r, True)
                continue
            # Empty grain / pit scraps get NUM@trust≈1 with ~0 completeness
            # (test-4). Real ``|`` bars keep some ink mass. On large spaced
            # lines, letter upright tips also score ~0.12–0.18 as NUM
            # (test-8 L1 𐩹) — require a clearer bar.
            sep_comp = c_l if peel_l else c_r
            if sep_comp < 0.18:
                continue
            # Letter side must look like a real glyph, not another scrap.
            letter_j = j_r if peel_l else j_l
            letter_comp = c_r if peel_l else c_l
            letter_rec = rec_r if peel_l else rec_l
            if letter_j < 0.70:
                continue
            if _is_numeral_or_separator_label(letter_rec.get("label")):
                continue
            # Same letter on merge and body + weak NUM tip = amputated stroke,
            # not a glued separator (test-8 L1 𐩹). Real ``|``+letter peels keep
            # a stronger sep half (ink mass).
            letter_lab = letter_rec.get("label")
            if (
                merge_lab
                and letter_lab
                and merge_lab == letter_lab
                and not _is_numeral_or_separator_label(merge_lab)
                and sep_comp < 0.28
            ):
                continue
            # Complete 𐩥/𐩲: peel often invents NUM tip + 𐩲 lookalike body
            # (test-9 L1). Only allow a thin real ``|`` when the letter half
            # stays a round glyph.
            if (
                _is_round_glue_label(merge_lab)
                and merge_comp >= 0.50
                and float(merge_rec.get("trust") or 0.0) >= 0.70
            ):
                sep_w = w_l if peel_l else w_r
                if not (
                    sep_comp >= 0.22
                    and sep_w <= 0.24 * line_height
                    and _is_round_glue_label(letter_lab)
                ):
                    continue
            # Do not amputate a stroke from an already-complete letter just
            # because the tip scores as NUM on empty stone texture.
            # Exception: a real ``|`` glued onto 𐩥/𐩲 (test-3).
            if (
                merge_comp >= 0.75
                and float(merge_rec.get("trust") or 0.0) >= 0.70
                and not _is_numeral_or_separator_label(merge_lab)
                and letter_comp + 0.08 < merge_comp
            ):
                sep_w = w_l if peel_l else w_r
                round_bar = (
                    _is_round_glue_label(letter_lab)
                    and _is_thin_bar(sep_w, line_height)
                    and sep_comp >= 0.22
                )
                if not round_bar:
                    continue
            # If both merge and letter-half are the same stem-body class, this
            # peel is amputating the letter's own upright — skip.
            if (
                (
                    _is_beth_label(merge_lab)
                    and _is_beth_label(letter_lab)
                )
                or (
                    _is_heth_label(merge_lab)
                    and _is_heth_label(letter_lab)
                )
                or (
                    _is_stem_body_letter_label(merge_lab)
                    and _is_stem_body_letter_label(letter_lab)
                )
            ):
                continue
            score = j_l + j_r - seg_cost
            if score < merge_j + 0.04:
                continue
            if best is None or score > best[0]:
                best = (score, cut, j_l, j_r, False)

        if best is None:
            out.append((left, right, conf))
            continue
        _score, cut, j_l, j_r, drop_junk = best
        if drop_junk:
            j_left, c_left, _rec_left = score_map[(left, cut)]
            j_right, c_right, _rec_right = score_map[(cut, right)]
            if c_left >= c_right:
                out.append((left, cut, float(j_l)))
                if c_right >= 0.12:
                    out.append((cut, right, float(j_r)))
            else:
                if c_left >= 0.12:
                    out.append((left, cut, float(j_l)))
                out.append((cut, right, float(j_r)))
            continue
        out.append((left, cut, float(j_l)))
        out.append((cut, right, float(j_r)))

    out.sort(key=lambda t: t[0])
    return out


def _split_touching_close_glyphs_by_recognition(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    device: torch.device,
    line_height: int,
    recognition_bundle: tuple[torch.nn.Module, list[str], dict | None] | None = None,
    model: LetterCompletenessNet | None = None,
    activity: np.ndarray | None = None,
    stroke_mask: np.ndarray | None = None,
    boundary_profile: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """
    Final-only split for tightly packed neighbours glued into one box (test-6).

    Leaves all earlier rejoin / ``||`` / beth-heth repair unchanged. Splits only
    when a mid cut sits on a gutter/valley/peak AND both sides beat the merge
    by a clear recognition+completeness margin. Intact high-completeness
    stem-body letters (𐩨 / 𐩢 / 𐩣 …) are never opened.
    """
    if (
        not boxes
        or line_height <= 0
        or recognition_bundle is None
        or model is None
    ):
        return boxes

    from .letter_boundary_net import letter_activity_profile

    if activity is None:
        activity = letter_activity_profile(line_bgr)
    if stroke_mask is None:
        stroke_mask = _carved_letter_mask(line_bgr)

    enhanced = enhance_line(line_bgr)
    pitch = _estimate_letter_pitch(boxes, line_height)
    min_part = max(5, int(0.08 * line_height))
    max_bar = max(min_part + 1, int(0.30 * line_height))
    seg_cost = 0.55
    out: list[tuple[int, int, float]] = []

    for left, right, conf in boxes:
        width = right - left
        # Compact |+letter glues are often only ~0.75–1.1× pitch on upscaled
        # short lines (test-6 L1 leading ``|``+𐩲).
        if width < max(0.72 * pitch, 0.18 * line_height):
            out.append((left, right, conf))
            continue
        if width > max(2.05 * pitch, 0.95 * line_height):
            out.append((left, right, conf))
            continue

        two_lobes = _activity_has_two_lobes(
            activity, left, right, line_height
        )
        # Intact compact single body: skip the expensive cut scan.
        # Keep this NARROW — glued 𐩲/𐩥+neighbour often has high conf, no
        # two-lobe valley (grain bridge), and width ~1.3–1.6× pitch
        # (test-6 L3). Old 1.20×pitch / 0.55·h gate skipped those glues.
        if (
            conf >= 0.75
            and not two_lobes
            and width <= max(0.95 * pitch, 0.36 * line_height)
        ):
            out.append((left, right, conf))
            continue

        merge_comp = score_crop(enhanced, left, right, model, device)
        merge_rec = _batch_recognition_span_scores(
            line_bgr, [(left, right)], recognition_bundle, device
        )[0]
        merge_lab = merge_rec.get("label")
        merge_t = float(merge_rec.get("trust") or 0.0)
        merge_j = _span_joint_score(merge_comp, merge_rec)

        # Open-center 𐩨/HETH/MEM have two ink peaks. Never treat that as glue
        # (test-2 L3 amputated the left upright into a 5px NUM scrap).
        intact_stem_body = (
            merge_comp >= 0.45
            and (
                _is_beth_label(merge_lab)
                or _is_heth_label(merge_lab)
                or _is_stem_body_letter_label(merge_lab)
            )
            and not _is_round_glue_label(merge_lab)
        )
        if intact_stem_body:
            out.append((left, right, conf))
            continue

        # Only dense-scan when the box may hide a glued round ``o``.
        maybe_round_glue = (
            two_lobes
            or merge_comp < 0.25
            or _is_round_glue_label(merge_lab)
        )

        # Fast path: a complete single glyph does not need a cut scan — except
        # near-pitch merges that may hide a glued round letter.
        if (
            merge_comp >= 0.55
            and conf >= 0.75
            and not maybe_round_glue
            and (
                (
                    merge_t >= 0.70
                    and not _is_numeral_or_separator_label(merge_lab)
                )
                or (
                    merge_comp >= 0.70
                    and (
                        _is_beth_label(merge_lab)
                        or _is_heth_label(merge_lab)
                        or _is_stem_body_letter_label(merge_lab)
                    )
                )
            )
        ):
            out.append((left, right, conf))
            continue

        # Do not reopen letters the earlier pipeline intentionally kept whole.
        if (
            merge_comp >= 0.50
            and merge_t >= 0.70
            and (
                _is_beth_label(merge_lab)
                or _is_heth_label(merge_lab)
                or _is_stem_body_letter_label(merge_lab)
            )
            and not _is_round_glue_label(merge_lab)
        ):
            out.append((left, right, conf))
            continue
        # Thick single ``|`` (high completeness NUM) is not a glued pair.
        if (
            _is_numeral_or_separator_label(merge_lab)
            and merge_comp >= 0.55
            and merge_t >= 0.75
            and not maybe_round_glue
        ):
            out.append((left, right, conf))
            continue

        peak = float(np.max(activity[left:right]) + 1e-6)
        cands: list[int] = []
        step = max(2, int(0.04 * line_height))
        for x in range(left + min_part, right - min_part + 1, step):
            gutter = _clear_stroke_gutter(stroke_mask, x)
            valley = (
                float(activity[x]) <= float(activity[max(left, x - 1)])
                and float(activity[x]) <= float(activity[min(right - 1, x + 1)])
                and float(activity[x]) <= 0.58 * peak
            )
            peak_cut = _boundary_cut_strength(boundary_profile, x) >= 0.45
            if gutter or valley or peak_cut:
                cands.append(x)
        # Classic thin-bar peels at both edges.
        for bar in range(min_part, max_bar + 1, max(2, int(0.04 * line_height))):
            if left + bar <= right - min_part:
                cands.append(left + bar)
            if right - bar >= left + min_part:
                cands.append(right - bar)
        # Round ``o`` letters are often a compact blob beside a stem — probe
        # denser cuts across the middle without requiring a gutter.
        if maybe_round_glue:
            mid_lo = left + max(min_part, int(0.18 * width))
            mid_hi = right - max(min_part, int(0.18 * width))
            mid_step = max(3, int(0.05 * line_height))
            for x in range(mid_lo, mid_hi + 1, mid_step):
                cands.append(x)
            # Compact ring peels (~0.15–0.45·h) on either edge.
            for bar in range(
                max(min_part, int(0.14 * line_height)),
                max(max_bar, int(0.48 * line_height)) + 1,
                max(2, int(0.04 * line_height)),
            ):
                if left + bar <= right - min_part:
                    cands.append(left + bar)
                if right - bar >= left + min_part:
                    cands.append(right - bar)
        kept: list[int] = []
        for x in sorted(set(cands)):
            if not kept or x - kept[-1] >= 2:
                kept.append(x)
        if not kept:
            out.append((left, right, conf))
            continue

        spans: list[tuple[int, int]] = [(left, right)]
        for cut in kept:
            spans.append((left, cut))
            spans.append((cut, right))
        uniq: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for sp in spans:
            if sp not in seen and sp[1] - sp[0] >= 3:
                seen.add(sp)
                uniq.append(sp)

        recs = _batch_recognition_span_scores(
            line_bgr, uniq, recognition_bundle, device
        )
        comps = _score_crops_batch(enhanced, uniq, model, device)
        score_map = {
            sp: (_span_joint_score(comp, rec), float(comp), rec)
            for sp, comp, rec in zip(uniq, comps, recs)
        }

        best: tuple[float, int, float, float] | None = None
        empty_merge = merge_comp < 0.20
        for cut in kept:
            j_l, c_l, rec_l = score_map[(left, cut)]
            j_r, c_r, rec_r = score_map[(cut, right)]
            w_l = cut - left
            w_r = right - cut
            lab_l = rec_l.get("label")
            lab_r = rec_r.get("label")
            sep_l = _is_numeral_or_separator_label(lab_l)
            sep_r = _is_numeral_or_separator_label(lab_r)
            gutter = _clear_stroke_gutter(stroke_mask, cut)
            valley = float(activity[cut]) <= 0.58 * peak
            peak_cut = _boundary_cut_strength(boundary_profile, cut) >= 0.45
            min_peel = max(min_part, int(0.12 * line_height))
            thin_peel = (
                sep_l
                and min_peel <= w_l <= max_bar
                and not sep_r
            ) or (
                sep_r
                and min_peel <= w_r <= max_bar
                and not sep_l
            )
            round_l = (
                _is_round_glue_label(lab_l)
                and 0.12 * line_height <= w_l <= 0.55 * line_height
                and j_l >= 0.45
            )
            round_r = (
                _is_round_glue_label(lab_r)
                and 0.12 * line_height <= w_r <= 0.55 * line_height
                and j_r >= 0.45
            )
            round_hit = round_l or round_r
            if not (
                gutter
                or valley
                or peak_cut
                or thin_peel
                or empty_merge
                or round_hit
            ):
                continue

            # Ultra-thin NUM/| scraps are amputated uprights, not glued letters
            # (test-2 L3 𐩨 left stem → 5px NUM).
            if min(w_l, w_r) < min_peel and (sep_l or sep_r) and not round_hit:
                continue

            if min(j_l, j_r) < (0.40 if (empty_merge or round_hit) else 0.48):
                continue

            # Never cut a stem-body letter into body + NUM scrap when merge
            # already looks like that same letter with decent completeness.
            if (
                merge_comp >= 0.40
                and (
                    _is_beth_label(merge_lab)
                    or _is_heth_label(merge_lab)
                    or _is_stem_body_letter_label(merge_lab)
                )
                and (sep_l ^ sep_r)
                and not round_hit
            ):
                continue
            # Empty NUM tip peels (c≈0) are amputated strokes, not glued ``|``
            # (test-4 L1/L2; test-8 L1).
            if (sep_l ^ sep_r) and min(c_l, c_r) < 0.18 and not round_hit:
                continue

            letter_l = (not sep_l) and j_l >= 0.48
            letter_r = (not sep_r) and j_r >= 0.48
            peel = (sep_l and letter_r and w_l <= max_bar) or (
                sep_r and letter_l and w_r <= max_bar
            )
            pair = letter_l and letter_r
            # Round ``o`` + neighbour (letter or ``|``), even when grain bridges
            # the gap so there is no clear gutter (test-6 lines with 𐩥/𐩲).
            round_peel = (round_l and (letter_r or sep_r) and not round_r) or (
                round_r and (letter_l or sep_l) and not round_l
            )
            if not (peel or pair or round_peel):
                continue

            # Do not slice a ~1-letter glyph through its own connected strokes
            # (𐩥 middle bar). A valley between two bodies is a real neighbour.
            bisect = _cut_bisects_connected_stroke(stroke_mask, cut)
            interior_stem = (
                bisect
                and not gutter
                and not _seam_is_between_two_bodies(activity, left, cut, right)
            )
            if interior_stem and width <= max(1.45 * pitch, 0.70 * line_height):
                continue
            if (
                bisect
                and not gutter
                and width <= max(1.22 * pitch, 0.42 * line_height)
            ):
                continue
            # Complete trusted 𐩥/𐩲: never mid-bisect into lookalike halves
            # (test-8 L2 𐩥 → NUM+𐩬). Only allow a thin real ``|`` edge peel.
            if (
                _is_round_glue_label(merge_lab)
                and merge_comp >= 0.55
                and merge_t >= 0.65
            ):
                sep_w = w_l if sep_l else w_r
                real_bar_peel = (
                    peel
                    and sep_w <= 0.20 * line_height
                    and min(c_l, c_r) >= 0.28
                    and (
                        _is_round_glue_label(lab_l if not sep_l else lab_r)
                    )
                )
                if not real_bar_peel:
                    continue

            split_score = j_l + j_r - seg_cost
            need = (
                0.04
                if (empty_merge or round_peel)
                else (0.06 if (peel or gutter) else 0.14)
            )
            if split_score < merge_j + need:
                continue
            # High-completeness merges may still be round+neighbour glue.
            if (
                merge_comp >= 0.70
                and not peel
                and not gutter
                and not round_peel
            ):
                continue
            if best is None or split_score > best[0]:
                best = (split_score, cut, j_l, j_r)

        if best is None:
            out.append((left, right, conf))
            continue
        _sc, cut, j_l, j_r = best
        out.append((left, cut, max(conf * 0.9, float(j_l))))
        out.append((cut, right, max(conf * 0.9, float(j_r))))

    out.sort(key=lambda t: t[0])
    return out


def _detect_boundary_first(
    line_bgr: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    *,
    cnn_weight: float = 0.10,
    merge_margin: float = 0.04,
    reward_floor: float = 0.42,
    source_height: int | None = None,
    allow_panoramic_resplit: bool = False,
) -> list[tuple[int, int, float]]:
    """
    Recognition-guided segmentation over a lattice of candidate boundaries.

    Boundary/geometry proposes possible cuts. A global decoder then chooses the
    partition using completeness + calibrated recognition/prototype evidence +
    soft geometry. Confident boundary-net peaks are kept: multi-letter merges
    and thin-bar rejoins may not erase them (annotation-quality cuts).
    """
    from .empty_segment_filter import SegmentCandidate, mark_empty_segments
    from .letter_boundary_net import letter_activity_profile

    _reset_recognition_span_cache(line_bgr)
    h, w = line_bgr.shape[:2]
    ref_h = int(source_height) if source_height is not None else h
    enhanced = enhance_line(line_bgr)
    activity = letter_activity_profile(line_bgr)
    stroke_mask = _carved_letter_mask(line_bgr)
    raw_cuts, boundary_profile, objectness = _candidate_cuts(
        line_bgr, device, source_height=ref_h
    )
    # Preserve candidate cuts for the decoder. Pre-coalescing them would let
    # geometry irreversibly decide segmentation before recognition sees spans.
    if boundary_profile is not None:
        from .letter_boundary_net import boundary_peak_maps

        _smooth, peak_candidates, _prom = boundary_peak_maps(
            boundary_profile, h, ripple_height=ref_h
        )
        _smooth2, weak_peaks, _ = boundary_peak_maps(
            boundary_profile, h, min_prominence_frac=0.05, ripple_height=ref_h
        )
        del _smooth, _smooth2, _prom
        peak_pool = sorted(set(peak_candidates) | set(weak_peaks))
        raw_cuts = _drop_mid_glyph_boundary_cuts(
            raw_cuts,
            enhanced,
            model,
            device,
            stroke_mask,
            h,
            w,
            boundary_profile=boundary_profile,
        )
        raw_cuts = _drop_ring_center_boundary_cuts(
            raw_cuts,
            line_bgr,
            activity,
            h,
            w,
            stroke_mask=stroke_mask,
            dense=ref_h < 72,
        )
        if ref_h < 72:
            raw_cuts = _drop_spurious_dense_cuts(
                raw_cuts,
                enhanced,
                model,
                device,
                w,
                ref_h=ref_h,
                boundary_profile=boundary_profile,
            )
            raw_cuts = _drop_thin_scrap_dense_cuts(
                raw_cuts,
                enhanced,
                model,
                device,
                w,
                stroke_mask,
                ref_h=ref_h,
                boundary_profile=boundary_profile,
            )
        raw_cuts = _snap_boundary_cuts_by_completeness(
            raw_cuts,
            peak_pool,
            enhanced,
            model,
            device,
            h,
            w,
        )
        raw_cuts = _add_validated_interior_boundary_cuts(
            raw_cuts,
            boundary_profile,
            enhanced,
            model,
            device,
            h,
            w,
        ) if ref_h >= 72 else raw_cuts
    cuts = _add_deep_activity_cuts(
        [c for c in raw_cuts if 0 < c < w], activity, h, w
    ) if ref_h >= 72 else [c for c in raw_cuts if 0 < c < w]
    if boundary_profile is not None:
        from .letter_boundary_net import suppress_empty_segments

        cuts = _filter_inactive_boundary_cuts(
            cuts, activity, w, h, boundary_profile=boundary_profile
        )
        cuts = suppress_empty_segments(line_bgr, cuts, objectness=objectness)
    edges = [0] + sorted(set(cuts)) + [w]
    n = len(edges) - 1
    if n == 0:
        return []

    pitch = _pitch_from_edges(edges, h)
    recognition_bundle = _load_recognition_bundle(device)

    # Candidate spans: allow several boundary atoms to form one glyph, bounded
    # by line-relative width so the lattice remains compact.
    span_score: dict[tuple[int, int], float] = {}
    span_meta: dict[tuple[int, int], dict] = {}
    tensors: list[torch.Tensor] = []
    recognition_spans: list[tuple[int, int]] = []
    keys: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 5) + 1):
            a, b = edges[i], edges[j]
            if b - a < 3:
                continue
            if j > i + 1 and b - a > max(1.65 * pitch, 1.20 * h):
                break
            # Do not erase confident boundary-net cuts (matches annotation UI).
            # Interior 𐩥 stems are allowed through so DP can keep one letter.
            if _span_crosses_strong_boundary(
                edges,
                i,
                j,
                boundary_profile,
                stroke_mask=stroke_mask,
                activity=activity,
                line_height=h,
                allow_round_hole=ref_h < 72,
            ):
                continue
            tensors.append(torch.from_numpy(prepare_window(enhanced[:, a:b])).unsqueeze(0))
            recognition_spans.append((a, b))
            keys.append((i, j))

    completeness: list[float] = []
    with torch.no_grad():
        for start in range(0, len(tensors), 128):
            batch = torch.stack(tensors[start : start + 128]).to(device)
            completeness.extend(
                float(x) for x in torch.sigmoid(model(batch)).cpu().tolist()
            )

    recognition = _batch_recognition_span_scores(
        line_bgr, recognition_spans, recognition_bundle, device
    )
    for key, comp, rec in zip(keys, completeness, recognition):
        i, j = key
        a, b = edges[i], edges[j]
        span_w = b - a
        ratio = span_w / max(pitch, 1.0)
        # Broad width prior: recognition remains free to choose naturally wide
        # glyphs; NUM_1 / separators receive a separate thin-span prior.
        letter_geom = float(np.exp(-0.5 * ((ratio - 0.82) / 0.48) ** 2))
        separator = _is_numeral_or_separator_label(rec.get("label"))
        carved_stem = _carved_thin_stem(activity, a, b, h)
        thin_geom = (
            float(np.clip(1.0 - abs(span_w / max(h, 1) - 0.16) / 0.18, 0.0, 1.0))
            if separator and carved_stem
            else 0.0
        )
        geometry = max(letter_geom, thin_geom)
        trust = float(rec.get("trust") or 0.0)
        # Closed-set NUM_1 is common on letter stems. Only trust it when the
        # span also looks like a carved thin separator; otherwise discount.
        if separator and span_w / max(h, 1) <= 0.34 and not carved_stem:
            trust *= 0.25
        # Prototype trust on a near-empty multi-letter blob (common false 𐩥)
        # must not beat a partition into weaker real pieces.
        if (not separator) and float(comp) < 0.12 and ratio >= 1.6:
            trust *= 0.25
        joint = (
            0.36 * float(comp)
            + 0.50 * trust
            + 0.14 * geometry
        )
        # Prefer keeping MEM/HETH intact; a stem is a frequent false NUM_1.
        if (
            _is_stem_body_letter_label(rec.get("label"))
            and trust >= 0.80
            and float(comp) >= 0.55
        ):
            joint = min(1.0, joint + 0.10)
        # Weak edge strips that only "look like" NUM_1 via prototypes.
        if (
            separator
            and span_w / max(h, 1) <= 0.30
            and not carved_stem
            and comp < 0.15
        ):
            joint = min(joint, 0.12)
        mean_ink, peak_ink, has_carving = _span_ink(activity, a, b)
        # Hard block: carved thin stem glued to a much wider neighbour is almost
        # always letter+| (or |+letter), not a multi-stem glyph.
        if _thin_wide_merge_blocked(edges, i, j, h, activity, stroke_mask):
            joint = min(joint, 0.18)
        span_score[key] = float(np.clip(joint, 0.0, 1.0))
        span_meta[key] = {
            **rec,
            "completeness": float(comp),
            "geometry": geometry,
            "mean_ink": mean_ink,
        }

    # Viterbi-style global partition. Additive evidence lets two trusted letters
    # beat one confident merged crop; per-segment cost prevents fragment spam.
    best: dict[int, tuple[float, int | None, float]] = {0: (0.0, None, 0.0)}
    for j in range(1, n + 1):
        cand: list[tuple[float, int, float]] = []
        for i in range(max(0, j - 5), j):
            if i not in best or (i, j) not in span_score:
                continue
            conf = span_score[(i, j)]
            a, b = edges[i], edges[j]
            mean, _peak, has_carving = _span_ink(activity, a, b)
            meta = span_meta[(i, j)]
            if not has_carving and conf < 0.58:
                continue

            # A closed-set recognizer can still be confident on fragments.
            # This per-glyph cost requires two proposed letters to provide
            # substantially more total evidence than one complete span.
            segment_cost = 0.62 if ref_h < 72 else 0.55
            reward = conf - segment_cost
            span_w = b - a
            is_separator = _is_numeral_or_separator_label(meta.get("label"))
            if span_w / max(h, 1) < 0.10 and not is_separator:
                reward -= 0.22
            if span_w > 1.45 * pitch and not is_separator:
                reward -= 0.10 * min(2.0, span_w / max(pitch, 1.0) - 1.45)
            if _thin_wide_merge_blocked(edges, i, j, h, activity, stroke_mask):
                continue

            # A connected-stroke cut is a soft penalty, never an irreversible
            # deletion. Strong recognition can override a grain bridge.
            # Exception: closed-set NUM_1 often fires on a letter's own stem;
            # peeling that fragment through a connected body is almost always
            # wrong, so keep a stronger soft penalty there.
            if j < n:
                cut = edges[j]
                if _cut_bisects_connected_stroke(stroke_mask, cut):
                    reward -= 0.20
                    if is_separator and span_w / max(h, 1) <= 0.30:
                        reward -= 0.22
                elif _clear_stroke_gutter(stroke_mask, cut):
                    reward += 0.05
                    if is_separator and span_w / max(h, 1) <= 0.28:
                        reward += 0.04
            # Prefer peeling a true gutter `|` that recognition already trusts.
            if (
                is_separator
                and span_w / max(h, 1) <= 0.28
                and _carved_thin_stem(activity, a, b, h)
            ):
                if j < n and _clear_stroke_gutter(stroke_mask, edges[j]):
                    reward += 0.06
                if i > 0 and _clear_stroke_gutter(stroke_mask, edges[i]):
                    reward += 0.04
            if has_carving and conf < 0.35:
                reward += 0.08 * min(1.0, mean)
            cand.append((best[i][0] + reward, i, conf))

        if not cand and j - 1 in best:
            conf = span_score.get((j - 1, j), 0.0)
            cand.append((best[j - 1][0] + conf - 0.68, j - 1, conf))
        if cand:
            score, prev, conf = max(cand, key=lambda t: t[0])
            best[j] = (score, prev, conf)

    boxes: list[tuple[int, int, float]] = []
    if n in best:
        cur = n
        while cur is not None and cur > 0:
            _score, prev, conf = best[cur]
            if prev is None:
                break
            boxes.append((edges[prev], edges[cur], conf))
            cur = prev
        boxes.reverse()

    # Geometry refinement only. Do not run the old split/re-merge chain: it
    # would override the globally decoded recognition decision.
    boxes = _detach_edge_bars(
        boxes,
        activity,
        h,
        stroke_mask=stroke_mask,
        boundary_profile=boundary_profile,
    )
    # Repair merges where the cut lattice never offered a boundary: recognition
    # compares the wide box against gutter/valley splits.
    boxes = _recognition_resplit_wide_boxes(
        boxes,
        line_bgr,
        model,
        device,
        activity,
        stroke_mask,
        recognition_bundle,
        source_height=source_height,
        allow_panoramic_resplit=allow_panoramic_resplit,
        boundary_profile=boundary_profile,
    )
    # Repair MEM/HETH stems decoded as ``|`` after DP / resplit.
    boxes = _rejoin_stem_body_false_splits(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        boundary_profile=boundary_profile,
        model=model,
    )
    # General musnad_final compare: rejoin any letter falsely split by a peak.
    boxes = _rejoin_by_recognition(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        boundary_profile=boundary_profile,
        model=model,
    )
    # Layout law: never ``||`` with no letter between (split 𐩨 etc.).
    boxes = _repair_adjacent_separators_by_law(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        model=model,
    )
    boxes = _rejoin_split_thin_bars(boxes, h, boundary_profile=boundary_profile)
    # Thin-bar merge can recreate a letter-width mid-bisect pair (test-27 L2
    # 𐩨) that was three fragments before the first rejoin pass.
    boxes = _rejoin_stem_body_false_splits(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        boundary_profile=boundary_profile,
        model=model,
    )
    boxes = _rejoin_by_recognition(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        boundary_profile=boundary_profile,
        model=model,
    )
    # Peel ``|`` glued inside a letter box (test-1 L3 ``|``+𐩥).
    before_peel = [(a, b) for a, b, _c in boxes]
    boxes = _peel_glued_separators_by_recognition(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        model=model,
    )
    # Peel may amputate a 𐩨/HETH upright tip — glue ultra-thin scraps back.
    # Skip when peel did not change boxes (same result, less work).
    if [(a, b) for a, b, _c in boxes] != before_peel:
        boxes = _rejoin_stem_body_false_splits(
            boxes,
            line_bgr,
            device,
            h,
            recognition_bundle=recognition_bundle,
            boundary_profile=boundary_profile,
            model=model,
        )
        boxes = _rejoin_by_recognition(
            boxes,
            line_bgr,
            device,
            h,
            recognition_bundle=recognition_bundle,
            boundary_profile=boundary_profile,
            model=model,
        )
    # Final pass: peel/stem repair can recreate illegal ``||`` runs.
    boxes = _repair_adjacent_separators_by_law(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        model=model,
    )
    # Final-only: split tightly glued neighbours (test-6). Does not change
    # earlier rejoin / beth / ``||`` repair — only opens boxes when both sides
    # clearly beat the merge on a gutter/valley cut.
    boxes = _split_touching_close_glyphs_by_recognition(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        model=model,
        activity=activity,
        stroke_mask=stroke_mask,
        boundary_profile=boundary_profile,
    )
    # Glue amputated upright tips / incomplete letters the close-glyph split
    # may still peel (test-4 L1 𐩡, L2 𐩨 tip).
    boxes = _rejoin_stem_body_false_splits(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        boundary_profile=boundary_profile,
        model=model,
    )
    boxes = _rejoin_by_recognition(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        boundary_profile=boundary_profile,
        model=model,
    )

    boxes = _grow_boxes_into_gaps(
        boxes, activity, w, h, stroke_mask=stroke_mask
    )
    boxes = _claim_connected_gap_strokes(
        boxes, stroke_mask, activity, w, h
    )
    # Grow can leave a 𐩥 middle-bar scrap still next to the ring; rejoin once
    # more so that scrap is absorbed after the empty-gap pad.
    boxes = _rejoin_by_recognition(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        boundary_profile=boundary_profile,
        model=model,
    )
    # Last: rejoin 𐩲 / 𐩥 halves that survived as two high-confidence boxes.
    boxes = _merge_dense_ring_halves(
        boxes,
        line_bgr,
        activity,
        h,
        stroke_mask=stroke_mask,
        dense=ref_h < 72,
    )
    boxes = _merge_round_letter_halves(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        model=model,
        activity=activity,
        stroke_mask=stroke_mask,
        dense=ref_h < 72,
    )
    boxes = _peel_separator_from_round_boxes(
        boxes, activity, stroke_mask, h, line_bgr=line_bgr
    )
    # Round-letter rejoin can swallow a following ``|``; peel it back off.
    boxes = _peel_glued_separators_by_recognition(
        boxes,
        line_bgr,
        device,
        h,
        recognition_bundle=recognition_bundle,
        model=model,
    )
    # Isolated ``|`` in a stone gutter is never a DP box (empty-gap collapse).
    boxes = _insert_gap_word_separators(
        boxes, activity, w, h, stroke_mask=stroke_mask
    )

    segments = [
        SegmentCandidate(index=k + 1, x_left=a, x_right=b, image=line_bgr[:, a:b])
        for k, (a, b, _c) in enumerate(boxes)
    ]
    mark_empty_segments(
        segments,
        line_activity=activity,
        objectness=objectness,
    )
    kept: list[tuple[int, int, float]] = []
    for seg, (_a, _b, conf) in zip(segments, boxes):
        if (not seg.is_empty) or _keep_marked_empty_segment(seg, conf):
            kept.append((seg.x_left, seg.x_right, conf))
            continue
        # Dense/small lines: keep faint hollow circles the empty filter drops.
        if _is_dense_small_packing(edges, h) or h < 72:
            if _looks_like_small_ring(seg.image):
                kept.append((seg.x_left, seg.x_right, max(conf, 0.45)))
    return kept


def _insert_gap_word_separators(
    boxes: list[tuple[int, int, float]],
    activity: np.ndarray,
    line_width: int,
    line_height: int,
    stroke_mask: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """
    Turn a carved mid-gap ``|`` into its own box.

    ``suppress_empty_segments`` often collapses that stem into one gutter cut,
    so grow-into-gap never sees a crop — only leftover ink between letters.
    """
    if len(boxes) < 2 or line_height <= 0 or activity.size == 0:
        return boxes
    min_bar = max(2, int(0.03 * line_height))
    max_bar = max(min_bar + 1, int(0.24 * line_height))
    min_gap = min_bar + 4
    n = int(activity.size)
    extra: list[tuple[int, int, float]] = []
    for i in range(len(boxes) - 1):
        gap_l = int(boxes[i][1])
        gap_r = int(boxes[i + 1][0])
        if gap_r - gap_l < min_gap:
            continue
        lo = max(0, gap_l)
        hi = min(n, gap_r, line_width)
        if hi - lo < min_gap:
            continue
        sl = activity[lo:hi]
        peak_rel = int(np.argmax(sl))
        peak = float(sl[peak_rel])
        if peak < 0.40:
            continue
        ink = max(0.16, 0.35 * peak)
        a_rel = peak_rel
        while a_rel > 0 and float(sl[a_rel - 1]) >= ink:
            a_rel -= 1
        b_rel = peak_rel + 1
        while b_rel < sl.size and float(sl[b_rel]) >= ink:
            b_rel += 1
        width = b_rel - a_rel
        if not (min_bar <= width <= max_bar):
            continue
        if a_rel < 2 or sl.size - b_rel < 2:
            continue
        if float(sl[:a_rel].mean()) >= 0.22 * peak:
            continue
        if float(sl[b_rel:].mean()) >= 0.22 * peak:
            continue
        a = lo + a_rel
        b = lo + b_rel
        if not _carved_thin_stem(activity, a, b, line_height):
            continue
        # Neighbour letter stem sitting in the gutter — not a word separator.
        left_w = int(boxes[i][1] - boxes[i][0])
        right_w = int(boxes[i + 1][1] - boxes[i + 1][0])
        if left_w < 0.32 * line_height or right_w < 0.32 * line_height:
            continue
        if stroke_mask is not None and (
            _cut_bisects_connected_stroke(stroke_mask, a)
            or _cut_bisects_connected_stroke(stroke_mask, b)
        ):
            continue
        extra.append((a, b, float(np.clip(peak, 0.55, 1.0))))
    if not extra:
        return boxes
    merged = list(boxes) + extra
    merged.sort(key=lambda t: t[0])
    return merged


def _claim_connected_gap_strokes(
    boxes: list[tuple[int, int, float]],
    stroke_mask: np.ndarray | None,
    activity: np.ndarray | None,
    line_width: int,
    line_height: int,
) -> list[tuple[int, int, float]]:
    """
    Pull a letter's leftover prong out of the unassigned gap.

    Grow stops at the first activity valley, so a connected trident/𐩥 arm
    past that valley is left unboxed (or later stolen as a fake ``|``).
    """
    if (
        not boxes
        or stroke_mask is None
        or stroke_mask.size == 0
        or line_height <= 0
    ):
        return boxes
    h, w = stroke_mask.shape[:2]
    if h < 4 or w < 4:
        return boxes
    _n_cc, labels = cv2.connectedComponents(stroke_mask, connectivity=8)
    del _n_cc
    # One extra prong, not a neighbouring letter (grain often bridges those).
    max_extra = max(5, int(0.22 * line_height))
    empty_skip = max(2, int(0.06 * line_height))
    out = [(int(a), int(b), float(c)) for a, b, c in boxes]

    def _col_labs(x: int) -> set[int]:
        if x < 0 or x >= w:
            return set()
        return {int(v) for v in np.unique(labels[:, x]) if v > 0}

    def _two_bodies(left: int, seam: int, right: int) -> bool:
        if activity is None or activity.size == 0:
            return False
        return _seam_is_between_two_bodies(activity, left, seam, right)

    def _extend(start: int, stop: int, step: int, own: set[int], body_l: int, body_r: int) -> int:
        x = start
        travelled = 0
        while (step > 0 and x < stop) or (step < 0 and x > stop):
            if travelled >= max_extra:
                break
            if _clear_stroke_gutter(stroke_mask, x):
                break
            if _two_bodies(body_l, x, body_r):
                break
            labs = _col_labs(x)
            if labs & own:
                x += step
                travelled += 1
                continue
            if not labs:
                found = None
                for peek in range(1, empty_skip + 1):
                    nx = x + step * peek
                    if (step > 0 and nx >= stop) or (step < 0 and nx <= stop):
                        break
                    if _clear_stroke_gutter(stroke_mask, nx):
                        break
                    if _two_bodies(body_l, nx, body_r):
                        break
                    pl = _col_labs(nx)
                    if pl & own:
                        found = nx
                        break
                    if pl:
                        break
                if found is None:
                    break
                travelled += abs(found - x)
                x = found + step
                continue
            break
        return x if step > 0 else x + 1

    grown: list[tuple[int, int, float]] = []
    for i, (left, right, conf) in enumerate(out):
        prev_end = grown[i - 1][1] if i else 0
        next_start = out[i + 1][0] if i + 1 < len(out) else min(line_width, w)
        next_right = out[i + 1][1] if i + 1 < len(out) else min(line_width, w)
        prev_left = grown[i - 1][0] if i else 0
        own: set[int] = set()
        x0 = max(0, min(left, w))
        x1 = max(x0 + 1, min(right, w))
        if x1 > x0:
            own = {int(v) for v in np.unique(labels[:, x0:x1]) if v > 0}
        a, b = left, right
        if own:
            b = min(next_start, _extend(right, next_start, 1, own, left, next_right))
            a = max(prev_end, _extend(left - 1, prev_end, -1, own, prev_left, right))
        grown.append((a, b, conf))
    return grown


def _grow_boxes_into_gaps(
    boxes: list[tuple[int, int, float]],
    activity: np.ndarray,
    line_width: int,
    line_height: int,
    stroke_mask: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """
    Recover a clipped stroke without eating the next letter.

    Empty-only padding cannot fix a cut that already sits on ink. This:
    1. Snaps a high-activity shared seam onto the nearest activity valley.
    2. Claims leftover ink in the *unassigned gap* that still touches the box,
       stopping at a valley or the neighbour box (never inside it).
    3. Adds a small empty-stone margin, stopping at the gap midpoint.
    """
    if not boxes or line_height <= 0:
        return boxes
    n = int(activity.size)
    if n <= 0:
        return boxes
    ink = 0.16
    min_w = max(4, int(0.10 * line_height))
    slack = max(4, int(0.14 * line_height))
    pad = max(2, int(0.07 * line_height))
    max_ink = max(pad, int(0.22 * line_height))

    def _act(x: int) -> float:
        if x < 0 or x >= n:
            return 0.0
        return float(activity[x])

    def _valley(lo: int, hi: int) -> int | None:
        lo = max(0, min(lo, n - 1))
        hi = max(lo, min(hi, n - 1))
        sl = activity[lo : hi + 1]
        peak = float(sl.max() + 1e-6)
        best_i = None
        best_v = peak
        for i, v in enumerate(sl.tolist()):
            fv = float(v)
            if fv > 0.55 * peak:
                continue
            if fv < best_v:
                best_v = fv
                best_i = i
        if best_i is None:
            return None
        return lo + best_i

    out = [(int(a), int(b), float(c)) for a, b, c in boxes]

    # 1) Move a cut off the stroke onto the gutter between two ink peaks.
    for i in range(len(out) - 1):
        a, b, c1 = out[i]
        c, d, c2 = out[i + 1]
        seam = b if c >= b else (b + c) // 2
        seam_i = min(max(seam, 0), n - 1)
        # Packed neighbours have no empty gap (c≈b). Do not hunt a nearby
        # interior valley (𐩩 ridge / 𐩲 hole) if this seam already separates
        # two ink bodies (test-1 L2 𐩥 then 𐩩).
        if _seam_is_between_two_bodies(activity, a, seam_i, d):
            continue
        if stroke_mask is not None and _clear_stroke_gutter(stroke_mask, seam_i):
            continue
        if _act(seam_i) < ink and c - b >= 2:
            continue
        lo = max(a + min_w, min(b, c) - slack)
        hi = min(d - min_w, max(b, c) + slack)
        if hi <= lo:
            continue
        new_seam = _valley(lo, hi)
        if new_seam is None:
            continue
        if new_seam - a < min_w or d - new_seam < min_w:
            continue
        if (
            stroke_mask is not None
            and _cut_bisects_connected_stroke(stroke_mask, new_seam)
            and not _cut_bisects_connected_stroke(stroke_mask, seam_i)
        ):
            continue
        out[i] = (a, new_seam, c1)
        out[i + 1] = (new_seam, d, c2)

    # 2–3) Claim gap ink attached to this box, then empty margin.
    grown: list[tuple[int, int, float]] = []
    for i, (left, right, conf) in enumerate(out):
        prev_end = grown[i - 1][1] if i else 0
        next_start = out[i + 1][0] if i + 1 < len(out) else line_width
        left_mid = (prev_end + left) // 2
        right_mid = (right + next_start) // 2
        a = left
        while a > prev_end and left - a < max_ink:
            col = a - 1
            if col < 0:
                break
            val = _act(col)
            prev = _act(col - 1) if col else val
            nxt = _act(col + 1)
            # Stop at a valley: this ink belongs to the previous glyph.
            if val >= ink and val <= prev and val <= nxt and left - a >= 2:
                break
            # Empty column then rising ink: next glyph, do not take it.
            if val < ink and prev < ink and nxt >= ink and left - a >= 1:
                break
            a = col
        b = right
        while b < next_start and b - right < max_ink:
            col = b
            if col >= n:
                break
            val = _act(col)
            prev = _act(col - 1) if col else val
            nxt = _act(col + 1) if col + 1 < n else val
            if val >= ink and val <= prev and val <= nxt and b - right >= 2:
                break
            if val < ink and nxt < ink and prev >= ink and b - right >= 1:
                # trailing empty after this glyph; keep going into pad below
                pass
            if val < ink and prev < ink and nxt >= ink and b - right >= 1:
                break
            b = col + 1
        while a > max(prev_end, left_mid) and left - a < pad:
            col = a - 1
            if col < 0:
                break
            if _act(col) >= ink:
                break
            a = col
        while b < min(next_start, right_mid) and b - right < pad:
            col = b
            if col >= n:
                break
            if _act(col) >= ink:
                break
            b = col + 1
        grown.append((a, min(b, line_width), conf))
    return _absorb_clip_slivers(grown, activity, line_height)


def _absorb_clip_slivers(
    boxes: list[tuple[int, int, float]],
    activity: np.ndarray,
    line_height: int,
) -> list[tuple[int, int, float]]:
    """Glue a 1–2px stroke sliver that is a cut through a letter, not a ``|``."""
    if len(boxes) < 2 or line_height <= 0:
        return boxes
    sliver_w = max(3, int(0.10 * line_height))
    min_body = max(6, int(0.20 * line_height))
    n = int(activity.size)
    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        a, b, c = boxes[i]
        if i + 1 < len(boxes):
            d, e, f = boxes[i + 1]
            gap = d - b
            w1 = b - a
            w2 = e - d
            if gap <= 2 and min(w1, w2) <= sliver_w and max(w1, w2) >= min_body:
                seam = min(max((b + d) // 2, 0), max(n - 1, 0))
                peak = float(np.max(activity[a:e]) + 1e-6) if e > a and n else 1.0
                # Real separators sit in a valley; a clipped stroke does not.
                if n and float(activity[seam]) > 0.50 * peak:
                    out.append((a, e, max(c, f)))
                    i += 2
                    continue
        out.append((a, b, c))
        i += 1
    return out


def _learned_lattice_cuts(
    line_bgr: np.ndarray,
    device: torch.device,
    *,
    source_height: int | None = None,
) -> tuple[list[int], np.ndarray | None, np.ndarray | None]:
    """
    Candidate cut columns from the trained boundary net only.

    Extra weak peaks stay in the lattice so completeness can merge fragments;
    they are not treated as final cuts.
    """
    from .letter_boundary_net import (
        MODEL_PATH as BOUNDARY_PATH,
        boundaries_from_profile,
        boundary_peak_maps,
        load_boundary_model,
        predict_profiles,
        suppress_empty_segments,
    )

    h, w = line_bgr.shape[:2]
    if not BOUNDARY_PATH.exists():
        grid = max(6, h // 4)
        return list(range(grid, w, grid)), None, None

    bmodel = load_boundary_model(device)
    profile, objectness = predict_profiles(line_bgr, bmodel, device)
    cuts = boundaries_from_profile(
        profile,
        h,
        threshold=0.28,
        pair_prominence=0.08,
        source_height=source_height,
    )
    _smooth, weak, _prom = boundary_peak_maps(
        profile,
        h,
        threshold=0.20,
        min_prominence_frac=0.06,
        ripple_height=source_height,
    )
    del _smooth, _prom
    cuts = sorted(set(int(c) for c in cuts) | set(int(x) for x in weak))
    cuts = [c for c in cuts if 0 < c < w]
    cuts = suppress_empty_segments(line_bgr, cuts, objectness=objectness)
    return cuts, objectness, profile


def _detect_letters_learned(
    line_bgr: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    *,
    source_height: int | None = None,
) -> list[tuple[int, int, float]]:
    """
    Letter-aware cuts: boundary net + completeness CNN + DP.

    Completeness may score several letters as one 64×64 window; a strong
    boundary-net peak is not allowed to be skipped. Empty stone is dropped
    with objectness. musnad_final is not used.
    """
    from .empty_segment_filter import SegmentCandidate, mark_empty_segments
    from .letter_boundary_net import letter_activity_profile

    h, w = line_bgr.shape[:2]
    if w < 4 or h < 4:
        return []

    enhanced = enhance_line(line_bgr)
    activity = letter_activity_profile(line_bgr)
    cuts, objectness, boundary_profile = _learned_lattice_cuts(
        line_bgr, device, source_height=source_height
    )
    cuts = _add_deep_activity_cuts(
        [c for c in cuts if 0 < c < w], activity, h, w
    )
    edges = [0] + sorted(set(cuts)) + [w]
    n = len(edges) - 1
    if n == 0:
        return []

    max_atoms = 3
    max_span = max(8, int(1.25 * h))
    keys: list[tuple[int, int]] = []
    tensors: list[torch.Tensor] = []
    for i in range(n):
        for j in range(i + 1, min(n, i + max_atoms) + 1):
            a, b = edges[i], edges[j]
            if b - a < 3:
                continue
            if b - a > max_span:
                break
            if _span_crosses_strong_boundary(
                edges, i, j, boundary_profile, strong=0.40, line_height=h
            ):
                continue
            tensors.append(torch.from_numpy(prepare_window(enhanced[:, a:b])).unsqueeze(0))
            keys.append((i, j))

    completeness: dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for start in range(0, len(tensors), 128):
            batch = torch.stack(tensors[start : start + 128]).to(device)
            vals = torch.sigmoid(model(batch)).cpu().tolist()
            for key, val in zip(keys[start : start + 128], vals):
                completeness[key] = float(val)

    segment_cost = 0.42
    best: dict[int, tuple[float, int | None, float]] = {0: (0.0, None, 0.0)}
    for j in range(1, n + 1):
        cand: list[tuple[float, int, float]] = []
        for i in range(max(0, j - max_atoms), j):
            if i not in best or (i, j) not in completeness:
                continue
            a, b = edges[i], edges[j]
            conf = completeness[(i, j)]
            if objectness is not None:
                obj = float(objectness[a:b].mean()) if b > a else 0.0
                if obj < 0.18 and conf < 0.55:
                    continue
            mean_act = float(activity[a:b].mean()) if b > a else 0.0
            if mean_act < 0.03 and conf < 0.50:
                continue
            reward = conf - segment_cost
            if b - a > 0.90 * h:
                reward -= 0.18 * min(2.0, (b - a) / max(h, 1) - 0.90)
            cand.append((best[i][0] + reward, i, conf))
        if not cand and j - 1 in best:
            conf = completeness.get((j - 1, j), 0.0)
            cand.append((best[j - 1][0] + conf - 0.70, j - 1, conf))
        if cand:
            score, prev, conf = max(cand, key=lambda t: t[0])
            best[j] = (score, prev, conf)

    boxes: list[tuple[int, int, float]] = []
    if n in best:
        cur = n
        while cur is not None and cur > 0:
            _score, prev, conf = best[cur]
            if prev is None:
                break
            boxes.append((edges[prev], edges[cur], conf))
            cur = prev
        boxes.reverse()

    boxes = _learned_merge_if_more_complete(
        boxes, enhanced, model, device, line_height=h, profile=boundary_profile
    )
    boxes = _learned_split_if_parts_win(
        boxes,
        enhanced,
        model,
        device,
        cuts,
        h,
        activity=activity,
        profile=boundary_profile,
    )

    segments = [
        SegmentCandidate(index=k + 1, x_left=a, x_right=b, image=line_bgr[:, a:b])
        for k, (a, b, _c) in enumerate(boxes)
    ]
    mark_empty_segments(
        segments,
        line_activity=activity,
        objectness=objectness,
    )
    kept: list[tuple[int, int, float]] = []
    for seg, (_a, _b, conf) in zip(segments, boxes):
        if (not seg.is_empty) or conf >= 0.62:
            kept.append((seg.x_left, seg.x_right, conf))
    return kept


def _learned_merge_if_more_complete(
    boxes: list[tuple[int, int, float]],
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    *,
    line_height: int,
    profile: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """Glue two neighbours only when they still look like one letter-width."""
    if len(boxes) < 2:
        return boxes
    out: list[tuple[int, int, float]] = [boxes[0]]
    max_w = max(8, int(0.85 * max(line_height, 1)))
    for left, right, conf in boxes[1:]:
        a0, b0, c0 = out[-1]
        if left != b0:
            out.append((left, right, conf))
            continue
        if right - a0 > max_w:
            out.append((left, right, conf))
            continue
        if _boundary_cut_strength(profile, b0) >= 0.40:
            out.append((left, right, conf))
            continue
        merge = score_crop(enhanced, a0, right, model, device)
        if merge >= max(c0, conf) + 0.10 and merge >= 0.45:
            out[-1] = (a0, right, merge)
        else:
            out.append((left, right, conf))
    return out


def _learned_split_if_parts_win(
    boxes: list[tuple[int, int, float]],
    enhanced: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    candidate_cuts: list[int],
    line_height: int,
    *,
    activity: np.ndarray | None = None,
    profile: np.ndarray | None = None,
    _depth: int = 0,
) -> list[tuple[int, int, float]]:
    """
    Split a wide box at a boundary peak or ink valley.

    Completeness on a letterboxed multi-letter crop is often ~1, so a strong
    interior cut wins even when the merged score is high.
    """
    if not boxes:
        return boxes
    cut_set = sorted(set(candidate_cuts))
    out: list[tuple[int, int, float]] = []
    min_part = max(4, int(0.08 * max(line_height, 1)))
    pitch = float(max(line_height, 1))
    for left, right, conf in boxes:
        width = right - left
        if width < max(12, int(0.50 * line_height)):
            out.append((left, right, conf))
            continue
        interior = [c for c in cut_set if left + min_part <= c <= right - min_part]
        if activity is not None:
            interior = sorted(
                set(interior)
                | set(
                    _activity_valley_cuts(
                        activity, left, right, line_height=line_height, pitch=pitch
                    )
                )
            )
        merge = score_crop(enhanced, left, right, model, device)
        wide = width >= int(0.70 * line_height)
        pack = width >= int(0.50 * line_height)
        incomplete = merge < 0.35 or conf < 0.35
        best: tuple[float, int, float, float] | None = None
        for cut in interior:
            left_s = score_crop(enhanced, left, cut, model, device)
            right_s = score_crop(enhanced, cut, right, model, device)
            strong = _boundary_cut_strength(profile, cut) >= 0.40
            if incomplete and pack:
                if left_s < 0.08 and right_s < 0.08:
                    continue
                gain = left_s + right_s + (0.25 if strong else 0.0)
            else:
                if left_s < 0.22 or right_s < 0.22:
                    continue
                if wide and strong:
                    gain = left_s + right_s
                else:
                    gain = (left_s + right_s) - merge - 0.32
                    if gain <= 0.02 and not (wide and min(left_s, right_s) >= 0.40):
                        continue
            if best is None or gain > best[0]:
                best = (gain, cut, left_s, right_s)
        if best is None and incomplete and pack and activity is not None:
            pad = max(min_part, int(0.12 * width))
            inner = activity[left + pad : right - pad]
            if inner.size:
                cut = left + pad + int(np.argmin(inner))
                best = (
                    0.0,
                    cut,
                    score_crop(enhanced, left, cut, model, device),
                    score_crop(enhanced, cut, right, model, device),
                )
        if best is None:
            out.append((left, right, conf))
            continue
        _gain, cut, left_s, right_s = best
        out.append((left, cut, left_s))
        out.append((cut, right, right_s))
    if _depth < 5 and any(
        c < 0.35 and (b - a) >= int(0.50 * line_height) for a, b, c in out
    ):
        return _learned_split_if_parts_win(
            out,
            enhanced,
            model,
            device,
            candidate_cuts,
            line_height,
            activity=activity,
            profile=profile,
            _depth=_depth + 1,
        )
    return out


def detect_letters(
    line_bgr: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    *,
    step: int = 4,
    reward_floor: float = 0.55,
    merge_bonus: float = 0.06,
    cnn_weight: float = 0.45,
    mode: str = "v2",
    allow_panoramic_resplit: bool = False,
) -> list[tuple[int, int, float]]:
    """
    Detect complete letters without a fixed letter count.

    ``v2`` (default): Segmentation v2 — boundary net only, no musnad_final,
    no completeness, no repair chain.

    ``learned`` / ``boundary_first`` / ``dp``: legacy detectors.

    Short line crops (small letters) are temporarily upscaled so faint rings
    like 𐩲 survive; large/sparse lines are unchanged.
    """
    if mode == "v2":
        from .segment_v2 import load_v2_model, segment_line

        bmodel = load_v2_model(device)
        return segment_line(line_bgr, bmodel, device)

    scale = _small_letter_upscale(line_bgr.shape[0])
    source_height = int(line_bgr.shape[0])
    work = line_bgr
    if scale > 1.01:
        work = cv2.resize(
            line_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    if mode == "learned":
        boxes = _detect_letters_learned(
            work,
            model,
            device,
            source_height=source_height,
        )
    elif mode == "boundary_first":
        boxes = _detect_boundary_first(
            work,
            model,
            device,
            cnn_weight=min(cnn_weight, 0.15),
            source_height=source_height,
            allow_panoramic_resplit=allow_panoramic_resplit,
        )
    else:
        boxes = _detect_letters_dp(
            work,
            model,
            device,
            step=step,
            reward_floor=reward_floor,
            merge_bonus=merge_bonus,
            cnn_weight=cnn_weight,
        )

    if scale > 1.01:
        boxes = [
            (int(round(a / scale)), int(round(b / scale)), conf)
            for a, b, conf in boxes
            if int(round(b / scale)) - int(round(a / scale)) >= 2
        ]
    return boxes


def _small_letter_upscale(line_height: int) -> float:
    """Scale factor so short lines reach a comfortable working height."""
    if line_height <= 0:
        return 1.0
    # Only short crops — tall museum lines already work well.
    if line_height >= 72:
        return 1.0
    # Dense tablet rows (test-18 style, ~45-60px tall) need more horizontal
    # sampling for the boundary profile. Very tiny rows already had stable
    # behavior with the older 96px target, so leave those less aggressive.
    if line_height < 40:
        return min(2.5, 96 / float(line_height))
    target = 128 if line_height < 64 else 104
    return min(3.0, target / float(line_height))


def _is_dense_small_packing(edges: list[int], line_height: int) -> bool:
    """Many narrow atoms relative to line height (small letters, crowded line)."""
    if line_height <= 0 or len(edges) < 5:
        return False
    widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    if not widths:
        return False
    med = float(np.median(widths))
    n = len(widths)
    # Tall sparse lines: only very narrow atoms count as "dense".
    # Short/upscaled lines: a full glyph is often ~0.4–0.6·h wide, so allow more.
    if line_height < 100:
        return n >= 6 and med <= 0.55 * line_height
    return n >= 6 and med <= 0.32 * line_height


def _stroke_column_is_round_mid(
    stroke_mask: np.ndarray | None,
    left: int,
    cut: int,
    right: int,
) -> bool:
    """
    True when the cut sits inside 𐩲 (hole with top/bottom caps) or 𐩥 (bar).

    Both sides must be letter-wide, not a thin word ``|``.
    """
    if stroke_mask is None or stroke_mask.size == 0:
        return False
    h, w = stroke_mask.shape[:2]
    if right - left < 4 or cut <= left or cut >= right:
        return False
    if cut <= 1 or cut >= w - 1:
        return False
    if _clear_stroke_gutter(stroke_mask, cut):
        return False
    left_w = cut - left
    right_w = right - cut
    if min(left_w, right_w) < max(3, int(0.14 * h)):
        return False
    if max(left_w, right_w) > 0.55 * h:
        return False
    if (right - left) > (0.70 * h if h >= 72 else 0.88 * h):
        return False
    if max(left_w, right_w) > 1.55 * min(left_w, right_w):
        return False
    x0 = max(0, left)
    x1 = min(w, right)
    crop = stroke_mask[:, x0:x1]
    if crop.size == 0:
        return False
    hh, ww = crop.shape[:2]
    cx = min(max(cut - x0, 1), ww - 2)
    col = crop[:, max(0, cx - 1) : cx + 2]
    if col.size == 0:
        return False
    col = col.max(axis=1).astype(np.float32)
    top = float(col[: max(2, hh // 4)].mean())
    bot = float(col[min(hh - 2, 3 * hh // 4) :].mean())
    cen = float(col[hh // 3 : 2 * hh // 3].mean())
    left_body = crop[:, : max(1, cx - 1)]
    right_body = crop[:, min(ww, cx + 1) :]
    if left_body.size == 0 or right_body.size == 0:
        return False
    if float(left_body.mean()) < 0.02 or float(right_body.mean()) < 0.02:
        return False
    ring_hole = top >= 0.12 and bot >= 0.12 and cen <= 0.45 * max(top, bot, 1e-6)
    phi_bar = (
        min(top, cen, bot) >= 0.10
        and _cut_bisects_connected_stroke(stroke_mask, cut)
    )
    return ring_hole or phi_bar


def _is_round_plus_separator(
    activity: np.ndarray | None,
    stroke_mask: np.ndarray | None,
    left: int,
    cut: int,
    right: int,
    line_height: int,
    line_bgr: np.ndarray | None = None,
) -> bool:
    """
    True when the pair is a finished 𐩥/𐩲 plus a word ``|``, not 𐩥 halves.

    Requires a full-height thin stem, a valley/gutter (not a connected mid-bar),
    and a round body on the wide side.
    """
    if line_height <= 0 or cut <= left or right <= cut:
        return False
    w1 = cut - left
    w2 = right - cut
    thin, wide = min(w1, w2), max(w1, w2)
    if not _is_thin_bar(thin, line_height):
        return False
    if wide < max(thin + 3, int(0.22 * line_height)):
        return False
    if wide < 1.28 * thin:
        return False
    t0, t1 = (left, cut) if w1 <= w2 else (cut, right)
    b0, b1 = (cut, right) if w1 <= w2 else (left, cut)
    if not _carved_thin_stem(activity, t0, t1, line_height):
        return False
    if stroke_mask is not None and not _is_full_height_span(
        stroke_mask, t0, t1, line_height, threshold=0.42
    ):
        return False
    gutter = stroke_mask is not None and _clear_stroke_gutter(stroke_mask, cut)
    between = activity is not None and _seam_is_between_two_bodies(
        activity, left, cut, right
    )
    if not (gutter or between):
        return False
    if stroke_mask is not None and _cut_bisects_connected_stroke(stroke_mask, cut):
        return False
    if line_bgr is None:
        return True
    body = line_bgr[:, max(0, b0) : max(b0 + 1, b1)]
    return bool(
        _radial_hollow_ring(body)
        or _ink_hollow_ring(body)
        or _looks_like_small_ring(body)
    )


def _looks_like_round_letter_mid_cut(
    line_bgr: np.ndarray,
    activity: np.ndarray,
    stroke_mask: np.ndarray | None,
    left: int,
    cut: int,
    right: int,
    line_height: int,
    *,
    loose: bool = True,
    dense: bool = False,
) -> bool:
    """True when left|right of ``cut`` are halves of one 𐩲 / 𐩥."""
    if line_height <= 0 or right - left < 4:
        return False
    width = right - left
    if width < 0.20 * line_height:
        return False
    max_w = 0.88 * line_height if dense else 0.70 * line_height
    if width > max_w:
        return False
    if _is_round_plus_separator(
        activity, stroke_mask, left, cut, right, line_height, line_bgr=line_bgr
    ):
        return False
    if _ring_half_pair(
        line_bgr, activity, left, cut, right, line_height, loose=loose
    ):
        return True
    if not dense:
        return False
    return _stroke_column_is_round_mid(stroke_mask, left, cut, right)


def _ring_half_pair(
    line_bgr: np.ndarray,
    activity: np.ndarray,
    left: int,
    mid: int,
    right: int,
    line_height: int,
    *,
    loose: bool = False,
) -> bool:
    """
    True when two adjacent crops look like left/right halves of a hollow circle (𐩲).

    Strict on purpose so multi-stem letters and ``|`` pairs are not glued.
    ``loose`` (short/dense lines): allow near-square rings and use a radial
    annulus check — ink hole tests fail on faint weathered stone.
    """
    if line_height <= 0 or mid <= left + 1 or right <= mid + 1:
        return False
    r1 = (mid - left) / line_height
    r2 = (right - mid) / line_height
    r_m = (right - left) / line_height
    half_max = 0.55 if loose else 0.42
    merge_max = 1.10 if loose else 0.72
    half_min = 0.10 if loose else 0.12
    if not (half_min <= r1 <= half_max and half_min <= r2 <= half_max):
        return False
    if abs(r1 - r2) > (0.22 if loose else 0.16):
        return False
    if not (0.28 <= r_m <= merge_max):
        return False

    crop = line_bgr[:, left:right]
    if loose:
        # Short lines: radial ring on the joined crop is the main signal.
        if not _radial_hollow_ring(crop):
            return False
        # Mid column must show arc top/bottom with a bright open center.
        # Rejects stem+neighbor false rings that pass a coarse radial score.
        if not _mid_column_ring_gap(crop):
            return False
        # Soft mid-valley: skip when the whole span is faint (common on stone).
        pad = max(1, int(0.03 * line_height))
        mid_band = activity[max(0, mid - pad) : mid + pad + 1]
        left_m = float(activity[left:mid].mean()) if mid > left else 0.0
        right_m = float(activity[mid:right].mean()) if right > mid else 0.0
        mid_m = float(mid_band.mean()) if mid_band.size else 0.0
        peak = max(left_m, right_m, mid_m)
        if peak >= 0.12 and mid_m > 0.85 * min(left_m, right_m) + 1e-6:
            return False
        return True

    # Mid cut should sit in a column valley (cut through the open center).
    pad = max(1, int(0.03 * line_height))
    mid_band = activity[max(0, mid - pad) : mid + pad + 1]
    left_m = float(activity[left:mid].mean()) if mid > left else 0.0
    right_m = float(activity[mid:right].mean()) if right > mid else 0.0
    mid_m = float(mid_band.mean()) if mid_band.size else 0.0
    if mid_m > 0.55 * min(left_m, right_m) + 1e-6:
        return False
    if min(left_m, right_m) < 0.12:
        return False

    return _radial_hollow_ring(crop, strict=True) or _ink_hollow_ring(crop)


def _mid_column_ring_gap(crop_bgr: np.ndarray) -> bool:
    """True when the center column is bright in the middle (hole) vs top/bottom arcs."""
    if crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hh, ww = gray.shape
    if hh < 8 or ww < 6:
        return False
    mid = gray[:, max(0, ww // 2 - 1) : min(ww, ww // 2 + 2)]
    if mid.size == 0:
        return False
    top = float(mid[: max(2, hh // 4)].mean())
    bot = float(mid[min(hh - 2, 3 * hh // 4) :].mean())
    cen = float(mid[hh // 3 : 2 * hh // 3].mean())
    # Open center must be brighter than *both* arc caps (rejects two tall stems).
    return cen > max(top, bot) + 4.5


def _radial_hollow_ring(crop_bgr: np.ndarray, *, strict: bool = False) -> bool:
    """Carved circle: darker mid-radius band around a quieter brighter center."""
    if crop_bgr.size == 0:
        return False
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return False
    aspect = w / float(h)
    # Full 𐩲 is roughly square; reject tall thin merges (stems glued together).
    lo_a, hi_a = (0.62, 1.15) if not strict else (0.55, 1.25)
    if not (lo_a <= aspect <= hi_a):
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(min(cy, cx, h - 1 - cy, w - 1 - cx))
    if rmax < 4:
        return False
    # Tight mid-radius band — a wide annulus mixes ring + outer stone.
    inner = radius <= 0.22 * rmax
    ring = (radius > 0.25 * rmax) & (radius <= 0.50 * rmax)
    outer = (radius > 0.55 * rmax) & (radius <= 0.95 * rmax)
    if int(inner.sum()) < 8 or int(ring.sum()) < 16 or int(outer.sum()) < 8:
        return False
    inner_g = float(gray[inner].mean())
    ring_g = float(gray[ring].mean())
    outer_g = float(gray[outer].mean())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    inner_e = float(mag[inner].mean())
    ring_e = float(mag[ring].mean())
    # Groove darker than the hole and than surrounding stone.
    dark_ring = ring_g < inner_g - 8.0 and ring_g < outer_g - 4.0
    edge_ring = ring_e > inner_e * 1.25 + 1e-6
    if not dark_ring:
        return False
    if strict:
        return edge_ring

    # Angular coverage: ring signal in most octants (not a one-sided blob).
    ang = np.arctan2(yy - cy, xx - cx)
    sectors = 0
    for k in range(8):
        a0 = -np.pi + k * (np.pi / 4.0)
        a1 = a0 + np.pi / 4.0
        mask = ring & (ang >= a0) & (ang < a1)
        if int(mask.sum()) < 3:
            continue
        if float(gray[mask].mean()) < inner_g - 3.0 or float(mag[mask].mean()) > inner_e * 1.05:
            sectors += 1
    return sectors >= 6


def _ink_hollow_ring(crop_bgr: np.ndarray) -> bool:
    """Fallback: blur-minus-gray ink with empty center (works on cleaner crops)."""
    if crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hh, ww = gray.shape
    if hh < 8 or ww < 8:
        return False
    blur = cv2.blur(gray, (max(3, ww // 5) | 1, max(3, hh // 5) | 1))
    ink = np.clip(blur - gray, 0, None)
    if float(ink.max()) < 1e-3:
        return False
    ink /= float(ink.max())
    cy0, cy1 = hh // 3, 2 * hh // 3
    cx0, cx1 = ww // 3, 2 * ww // 3
    hole = float(ink[cy0:cy1, cx0:cx1].mean())
    ring = float(
        (
            ink[:cy0, :].mean()
            + ink[cy1:, :].mean()
            + ink[:, :cx0].mean()
            + ink[:, cx1:].mean()
        )
        / 4.0
    )
    if ring < 0.12 or hole > 0.55 * ring:
        return False
    mid_col = ink[:, max(0, ww // 2 - 1) : min(ww, ww // 2 + 2)]
    if mid_col.size == 0:
        return False
    top = float(mid_col[: max(2, hh // 4)].mean())
    bot = float(mid_col[min(hh - 2, 3 * hh // 4) :].mean())
    cen = float(mid_col[cy0:cy1].mean())
    return top >= 0.08 and bot >= 0.08 and cen <= 0.75 * max(top, bot)


def _merge_dense_ring_halves(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    activity: np.ndarray,
    line_height: int,
    stroke_mask: np.ndarray | None = None,
    *,
    dense: bool = False,
) -> list[tuple[int, int, float]]:
    """Join adjacent boxes that are left/right halves of 𐩲 / 𐩥."""
    if len(boxes) < 2 or line_height <= 0:
        return boxes

    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        left, right, conf = boxes[i]
        if i + 1 < len(boxes):
            n_left, n_right, n_conf = boxes[i + 1]
            if n_left - right <= 2 and _looks_like_round_letter_mid_cut(
                line_bgr,
                activity,
                stroke_mask,
                left,
                right,
                n_right,
                line_height,
                loose=True,
                dense=dense,
            ):
                out.append((left, n_right, max(conf, n_conf, 0.55)))
                i += 2
                continue
        out.append((left, right, conf))
        i += 1
    return out


def _merge_round_letter_halves(
    boxes: list[tuple[int, int, float]],
    line_bgr: np.ndarray,
    device: torch.device,
    line_height: int,
    recognition_bundle: tuple[torch.nn.Module, list[str], dict | None] | None = None,
    model: LetterCompletenessNet | None = None,
    activity: np.ndarray | None = None,
    stroke_mask: np.ndarray | None = None,
    *,
    dense: bool = False,
) -> list[tuple[int, int, float]]:
    """
    Rejoin 𐩥 / 𐩲 when recognition prefers the union over two fragments.

    Dense tablets (test-3) split these at the hole or middle bar; both halves
    can still look complete, so geometry-only merge is not enough.
    """
    if (
        len(boxes) < 2
        or line_height <= 0
        or recognition_bundle is None
        or model is None
    ):
        return boxes

    enhanced = enhance_line(line_bgr)
    out: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        left, right, conf = boxes[i]
        if i + 1 >= len(boxes):
            out.append((left, right, conf))
            i += 1
            continue
        n_left, n_right, n_conf = boxes[i + 1]
        gap = n_left - right
        w1 = right - left
        w2 = n_right - n_left
        wm = n_right - left
        if (
            gap > 2
            or wm < 0.20 * line_height
            or wm > (0.88 * line_height if dense else 0.72 * line_height)
            or min(w1, w2) < max(3, int(0.08 * line_height))
        ):
            out.append((left, right, conf))
            i += 1
            continue
        seam = right if n_left >= right else (right + n_left) // 2
        act = (
            activity
            if activity is not None
            else np.zeros(max(n_right, 1), dtype=np.float32)
        )
        round_mid = _looks_like_round_letter_mid_cut(
            line_bgr,
            act,
            stroke_mask,
            left,
            seam,
            n_right,
            line_height,
            loose=True,
            dense=dense,
        )
        # Real word ``|`` after a finished round letter — never absorb it.
        if _is_round_plus_separator(
            activity,
            stroke_mask,
            left,
            seam,
            n_right,
            line_height,
            line_bgr=line_bgr,
        ):
            out.append((left, right, conf))
            i += 1
            continue
        spans = [(left, right), (n_left, n_right), (left, n_right)]
        recs = _batch_recognition_span_scores(
            line_bgr, spans, recognition_bundle, device
        )
        comps = _score_crops_batch(enhanced, spans, model, device)
        merge_lab = recs[2].get("label")
        merge_t = float(recs[2].get("trust") or 0.0)
        merge_c = float(comps[2])
        if not _is_round_glue_label(merge_lab) and not round_mid:
            out.append((left, right, conf))
            i += 1
            continue
        if (
            round_mid
            and merge_lab
            and not _is_round_glue_label(merge_lab)
            and merge_t >= 0.70
            and not _is_numeral_or_separator_label(merge_lab)
        ):
            out.append((left, right, conf))
            i += 1
            continue
        if (
            not round_mid
            and merge_t < 0.48
            and merge_c < 0.45
        ):
            out.append((left, right, conf))
            i += 1
            continue

        def _strong_other(rec: dict, comp: float) -> bool:
            lab = rec.get("label")
            return (
                float(comp) >= 0.70
                and float(rec.get("trust") or 0.0) >= 0.75
                and bool(lab)
                and not _is_round_glue_label(lab)
                and not _is_numeral_or_separator_label(lab)
            )

        if _strong_other(recs[0], comps[0]) and _strong_other(recs[1], comps[1]):
            out.append((left, right, conf))
            i += 1
            continue
        thin_sep = min(w1, w2) <= 0.24 * line_height and (
            _is_numeral_or_separator_label(recs[0].get("label"))
            or _is_numeral_or_separator_label(recs[1].get("label"))
        )
        if thin_sep and (
            (
                stroke_mask is not None
                and _clear_stroke_gutter(stroke_mask, seam)
            )
            or (
                activity is not None
                and _seam_is_between_two_bodies(
                    activity, left, seam, n_right
                )
            )
        ):
            out.append((left, right, conf))
            i += 1
            continue
        if (
            round_mid
            or merge_c >= max(float(comps[0]), float(comps[1])) - 0.02
            or merge_t
            >= max(
                float(recs[0].get("trust") or 0.0),
                float(recs[1].get("trust") or 0.0),
            )
        ):
            out.append((left, n_right, max(conf, n_conf, merge_c, 0.55)))
            i += 2
            continue
        out.append((left, right, conf))
        i += 1
    return out


def _peel_separator_from_round_boxes(
    boxes: list[tuple[int, int, float]],
    activity: np.ndarray,
    stroke_mask: np.ndarray | None,
    line_height: int,
    line_bgr: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    """Split a word ``|`` glued to the edge of a finished 𐩥 / 𐩲 box."""
    if len(boxes) < 1 or line_height <= 0:
        return boxes
    min_bar = max(3, int(0.08 * line_height))
    max_bar = max(min_bar + 1, int(0.24 * line_height))
    out: list[tuple[int, int, float]] = []
    for left, right, conf in boxes:
        width = right - left
        if width < min_bar + max(8, int(0.28 * line_height)):
            out.append((left, right, conf))
            continue
        crop = None if line_bgr is None else line_bgr[:, left:right]
        if crop is not None and not (
            _radial_hollow_ring(crop)
            or _ink_hollow_ring(crop)
            or _looks_like_small_ring(crop)
        ):
            # Whole box is not round+bar; skip (avoids splitting 𐩨/𐩢).
            body_ok = False
            for bar in (min_bar, max_bar):
                if left + bar < right and (
                    _radial_hollow_ring(line_bgr[:, left + bar : right])
                    or _looks_like_small_ring(line_bgr[:, left + bar : right])
                ):
                    body_ok = True
                    break
                if right - bar > left and (
                    _radial_hollow_ring(line_bgr[:, left : right - bar])
                    or _looks_like_small_ring(line_bgr[:, left : right - bar])
                ):
                    body_ok = True
                    break
            if not body_ok:
                out.append((left, right, conf))
                continue
        cuts: list[int] = []
        for bar in range(min_bar, max_bar + 1):
            if left + bar < right - min_bar:
                cuts.append(left + bar)
            if right - bar > left + min_bar:
                cuts.append(right - bar)
        best: int | None = None
        best_thin = 10**9
        for cut in set(cuts):
            if not _is_round_plus_separator(
                activity,
                stroke_mask,
                left,
                cut,
                right,
                line_height,
                line_bgr=line_bgr,
            ):
                continue
            thin = min(cut - left, right - cut)
            if thin < best_thin:
                best_thin = thin
                best = int(cut)
        if best is None:
            out.append((left, right, conf))
            continue
        out.append((left, best, conf))
        out.append((best, right, conf))
    out.sort(key=lambda t: t[0])
    return out


def _looks_like_small_ring(crop_bgr: np.ndarray) -> bool:
    """Faint hollow circle that empty-filter might otherwise drop."""
    if crop_bgr.size == 0:
        return False
    h, w = crop_bgr.shape[:2]
    if h < 6 or w < 4:
        return False
    if w > 1.25 * h or w < 0.12 * h:
        return False
    if _radial_hollow_ring(crop_bgr) or _ink_hollow_ring(crop_bgr):
        return True
    # Small circles often sit in the vertical middle of a tall line band.
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink = np.clip(cv2.blur(gray, (5, 5)) - gray, 0, None)
    row = ink.mean(axis=1)
    peak = float(row.max())
    if peak < 1e-3:
        return False
    ys = np.where(row >= 0.22 * peak)[0]
    if ys.size < 4:
        return False
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    ch = y1 - y0
    if ch < 6:
        return False
    pad = max(0, (w - ch) // 2)
    sub = crop_bgr[max(0, y0 - pad) : min(h, y1 + pad)]
    if sub.shape[0] < 8:
        return False
    return _radial_hollow_ring(sub) or _ink_hollow_ring(sub)


def _detect_letters_dp(
    line_bgr: np.ndarray,
    model: LetterCompletenessNet,
    device: torch.device,
    *,
    step: int = 4,
    reward_floor: float = 0.55,
    merge_bonus: float = 0.06,
    cnn_weight: float = 0.45,
) -> list[tuple[int, int, float]]:
    """Older DP keep/merge/skip path over boundary candidate cuts."""
    del step
    from .letter_boundary_net import letter_activity_profile

    h, w = line_bgr.shape[:2]
    enhanced = enhance_line(line_bgr)
    activity = letter_activity_profile(line_bgr)
    raw_cuts, boundary_profile, _objectness = _candidate_cuts(line_bgr, device)
    cuts = [c for c in raw_cuts if 0 < c < w]
    edges = [0] + cuts + [w]
    n = len(edges) - 1
    if n == 0:
        return []

    cnn = _load_cnn_scorer(device)

    span_score: dict[tuple[int, int], float] = {}
    tensors: list[torch.Tensor] = []
    keys: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 3) + 1):
            a, b = edges[i], edges[j]
            if b - a < 3:
                continue
            tensors.append(torch.from_numpy(prepare_window(enhanced[:, a:b])).unsqueeze(0))
            keys.append((i, j))
    with torch.no_grad():
        for start in range(0, len(tensors), 128):
            batch = torch.stack(tensors[start : start + 128]).to(device)
            confs = torch.sigmoid(model(batch)).cpu().tolist()
            for key, conf in zip(keys[start : start + 128], confs):
                i, j = key
                a, b = edges[i], edges[j]
                shape = float(conf)
                if cnn is not None and cnn_weight > 0:
                    cnn_score = _cnn_completeness(line_bgr, a, b, cnn, device)
                    shape = (1.0 - cnn_weight) * shape + cnn_weight * cnn_score
                span_score[key] = shape

    best: dict[int, tuple[float, int | None, float]] = {0: (0.0, None, 0.0)}
    for j in range(1, n + 1):
        cand: list[tuple[float, int, float]] = []
        for i in range(max(0, j - 3), j):
            if i not in best or (i, j) not in span_score:
                continue
            conf = span_score[(i, j)]
            a, b = edges[i], edges[j]
            mean, peak, has_carving = _span_ink(activity, a, b)
            strong_ink = mean >= 0.20 or peak >= 0.55
            oversplit_conf = _oversplit_stem_pair(
                edges, i, j, h, activity, span_score, boundary_profile
            )
            if oversplit_conf is None:
                oversplit_conf = _oversplit_stem_triple(
                    edges, i, j, h, activity, span_score
                )
            if oversplit_conf is not None:
                conf = oversplit_conf
            elif (j - i) == 1 and strong_ink and conf < 0.40:
                conf = max(conf, 0.32 + 0.35 * min(1.0, mean))
            if conf < reward_floor and not has_carving and oversplit_conf is None:
                continue
            if conf < 0.18 and has_carving and not strong_ink and oversplit_conf is None:
                continue
            width = b - a
            ratio = width / max(h, 1)
            min_ratio = 0.08 if (j - i) == 1 else 0.12
            if ratio < min_ratio:
                continue
            if 0.22 <= ratio <= 0.95:
                shape_prior = 0.12
            elif ratio < 0.30:
                shape_prior = 0.02
            else:
                shape_prior = -0.06
            merge_term = merge_bonus if (j - i) > 1 else 0.0
            if (j - i) > 1:
                parts = [span_score.get((k, k + 1), 0.0) for k in range(i, j)]
                if parts and conf > max(parts) + 0.03:
                    merge_term += 0.08
            if oversplit_conf is not None:
                parts = [span_score.get((k, k + 1), 0.0) for k in range(i, j)]
                singles_est = sum(max(0.0, p - reward_floor + 0.10) for p in parts)
                reward = singles_est + 0.12
                cand.append((best[i][0] + reward, i, conf))
                continue
            weak_penalty = 0.0 if conf >= reward_floor else (reward_floor - conf) * 0.35
            reward = conf - reward_floor + shape_prior + merge_term - weak_penalty
            cand.append((best[i][0] + reward, i, conf))
        if j - 1 in best:
            a, b = edges[j - 1], edges[j]
            mean, peak, has_carving = _span_ink(activity, a, b)
            strong_ink = mean >= 0.20 or peak >= 0.55
            looks_empty = mean < 0.08 and peak < 0.28
            if looks_empty:
                cand.append((best[j - 1][0] - 0.02, j - 1, 0.0))
            elif not cand and strong_ink:
                forced = max(0.30, 0.28 + 0.4 * min(1.0, mean))
                ratio = (b - a) / max(h, 1)
                if ratio >= 0.08:
                    reward = forced - reward_floor + 0.02
                    cand.append((best[j - 1][0] + reward, j - 1, forced))
                else:
                    cand.append((best[j - 1][0] - 0.25, j - 1, 0.0))
            elif not cand:
                cand.append((best[j - 1][0] - 0.25, j - 1, 0.0))
            elif not has_carving:
                cand.append((best[j - 1][0] - 0.02, j - 1, 0.0))
        if cand:
            score, prev, conf = max(cand)
            best[j] = (score, prev, conf)

    if n not in best:
        return []

    boxes: list[tuple[int, int, float]] = []
    node = n
    while node > 0 and best[node][1] is not None:
        prev = best[node][1]
        conf = best[node][2]
        if conf > 0:
            boxes.append((edges[prev], edges[node], conf))
        node = prev
    boxes.reverse()
    boxes = _merge_packed_thin_boxes(boxes, h)

    from .empty_segment_filter import SegmentCandidate, mark_empty_segments

    segments = [
        SegmentCandidate(index=i + 1, x_left=a, x_right=b, image=line_bgr[:, a:b])
        for i, (a, b, _c) in enumerate(boxes)
    ]
    mark_empty_segments(
        segments,
        line_activity=activity,
        objectness=None,
    )
    return [
        (seg.x_left, seg.x_right, conf)
        for seg, (_a, _b, conf) in zip(segments, boxes)
        if (not seg.is_empty) or _keep_marked_empty_segment(seg, conf)
    ]


def _oversplit_stem_pair(
    edges: list[int],
    i: int,
    j: int,
    h: int,
    activity: np.ndarray,
    span_score: dict[tuple[int, int], float],
    boundary_profile: np.ndarray | None = None,
) -> float | None:
    """
    Detect multi-stem letters cut between close vertical stems.

    Completeness often scores each stem as a full letter and the whole glyph
    as junk (or merely not better than two singles). Return a synthetic merge
    confidence when geometry says this is one glyph.
    """
    if j - i != 2 or h <= 0:
        return None
    a, mid, b = edges[i], edges[i + 1], edges[j]
    r1 = (mid - a) / h
    r2 = (b - mid) / h
    r_m = (b - a) / h
    p1 = span_score.get((i, i + 1), 0.0)
    p2 = span_score.get((i + 1, j), 0.0)
    merge = span_score.get((i, j), 0.0)

    # Dense-tablet path: very thin similar stems packed together.
    dense_thin = (
        0.14 <= r1 <= 0.40
        and 0.14 <= r2 <= 0.40
        and abs(r1 - r2) <= 0.18
        and 0.32 <= r_m <= 0.78
        and min(p1, p2) >= 0.48
    )
    # Classic SAT-sized path (kept from earlier).
    classic = (
        0.28 <= r1 <= 0.40
        and 0.28 <= r2 <= 0.40
        and abs(r1 - r2) <= 0.08
        and 0.58 <= r_m <= 0.75
        and min(p1, p2) >= 0.55
        and merge < 0.35
    )
    if not (dense_thin or classic):
        return None

    mean1 = float(activity[a:mid].mean()) if mid > a else 0.0
    mean2 = float(activity[mid:b].mean()) if b > mid else 0.0
    if mean1 < 0.15 or mean2 < 0.15:
        return None
    pad = max(2, int(0.04 * h))
    internal = float(activity[max(0, mid - pad) : mid + pad].mean())
    if internal < 0.08:
        return None

    # Classic path still requires a weaker mid boundary peak.
    if classic and not dense_thin and boundary_profile is not None and 0 <= mid < len(boundary_profile):
        outer_l = edges[i] if i > 0 else None
        outer_r = edges[j] if j < len(edges) - 1 else None
        mid_p = float(boundary_profile[mid])
        if outer_l is not None and outer_r is not None and outer_r < len(boundary_profile):
            left_p = float(boundary_profile[outer_l])
            right_p = float(boundary_profile[outer_r])
            if not (mid_p < left_p - 0.05 and mid_p < right_p - 0.05):
                return None

    # If merge already outscores both halves, DP can keep it without a force.
    # Force when halves look "complete" alone (the failure mode on dense stone).
    if merge >= max(p1, p2) + 0.05 and min(r1, r2) > 0.26:
        return None
    return 0.5 * (p1 + p2)


def _oversplit_stem_triple(
    edges: list[int],
    i: int,
    j: int,
    h: int,
    activity: np.ndarray,
    span_score: dict[tuple[int, int], float],
) -> float | None:
    """Merge three consecutive thin stems (NUM_3 / multi-bar glyphs)."""
    if j - i != 3 or h <= 0:
        return None
    widths = [edges[k + 1] - edges[k] for k in range(i, j)]
    ratios = [w / h for w in widths]
    if not all(0.14 <= r <= 0.34 for r in ratios):
        return None
    r_m = (edges[j] - edges[i]) / h
    if not (0.48 <= r_m <= 0.95):
        return None
    parts = [span_score.get((k, k + 1), 0.0) for k in range(i, j)]
    if min(parts) < 0.50:
        return None
    merge = span_score.get((i, j), 0.0)
    # Only force when the triple is not already preferred as one glyph.
    if merge >= max(parts) + 0.05:
        return None
    a, b = edges[i], edges[j]
    if float(activity[a:b].mean()) < 0.18:
        return None
    return float(sum(parts) / len(parts))


def _merge_packed_thin_boxes(
    boxes: list[tuple[int, int, float]],
    h: int,
) -> list[tuple[int, int, float]]:
    """
    Glue adjacent thin high-confidence boxes on dense tablets.

    After DP, multi-stem glyphs often remain as 2–3 touching skinny boxes.
    Merge them when each piece is stem-narrow and the combination is still
    one letter wide.
    """
    if h <= 0 or len(boxes) < 2:
        return boxes
    merged: list[tuple[int, int, float]] = []
    i = 0
    while i < len(boxes):
        left, right, conf = boxes[i]
        while i + 1 < len(boxes):
            n_left, n_right, n_conf = boxes[i + 1]
            if n_left - right > 2:
                break
            r1 = (right - left) / h
            r2 = (n_right - n_left) / h
            r_m = (n_right - left) / h
            if not (
                r1 <= 0.40
                and r2 <= 0.40
                and r_m <= 0.85
                and min(conf, n_conf) >= 0.45
            ):
                break
            # Prefer merging when at least one piece is clearly stem-thin.
            if max(r1, r2) > 0.36 and min(r1, r2) > 0.28:
                break
            left, right, conf = left, n_right, 0.5 * (conf + n_conf)
            i += 1
        merged.append((left, right, conf))
        i += 1
    return merged



def render_detections(line_bgr: np.ndarray, boxes: list[tuple[int, int, float]]) -> np.ndarray:
    vis = line_bgr.copy()
    for left, right, conf in boxes:
        color = (0, 255, 80) if conf >= 0.7 else (0, 200, 255)
        cv2.rectangle(vis, (left, 1), (right - 1, vis.shape[0] - 2), color, 2)
        cv2.putText(
            vis,
            f"{conf:.2f}",
            (left + 2, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def segment_tiles(line_bgr: np.ndarray, boxes: list[tuple[int, int, float]]) -> np.ndarray:
    from .empty_segment_filter import SegmentCandidate, render_segment_sheet

    segments = [
        SegmentCandidate(index=i + 1, x_left=a, x_right=b, image=line_bgr[:, a:b])
        for i, (a, b, _c) in enumerate(boxes)
        if b > a
    ]
    return render_segment_sheet(segments, show_rejected=False)


def detect_image(
    image_path: Path | str,
    *,
    device: torch.device | None = None,
    out_dir: Path | None = None,
    model: LetterCompletenessNet | None = None,
    step: int = 4,
    reward_floor: float = 0.55,
    mode: str = "v2",
    save_crops: bool = True,
    crop_pad: int = 2,
) -> dict:
    """
    Run line banding + letter localization on a full inscription image.

    Same pipeline as ``python -m src.letter_detector --image …``.
    Writes per-line ``*_boxes.jpg`` / ``*_segments.jpg`` under ``out_dir``,
    and optional per-letter crops under ``out_dir/crops/``.
    """
    from .inscription_region import isolate_inscription_if_sparse
    from .stone_glyph_segmentation import detect_line_bands, load_bgr

    image_path = Path(image_path)
    if device is None:
        device = resolve_device()
    if mode != "v2" and model is None:
        model = load_detector(device)

    image = load_bgr(image_path)
    work, region = isolate_inscription_if_sparse(image)
    rx, ry = (region.x_left, region.y_top) if region.applied else (0, 0)
    bands = detect_line_bands(work)
    if out_dir is None:
        out_dir = OUTPUTS_DIR / "letter_detector" / image_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "crops"
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    lines_out: list[dict] = []
    allow_panoramic_resplit = len(bands) >= 2
    for index, band in enumerate(bands, start=1):
        crop = work[band.y_top : band.y_bottom]
        boxes = detect_letters(
            crop,
            model,
            device,
            step=step,
            reward_floor=reward_floor,
            mode=mode,
            allow_panoramic_resplit=allow_panoramic_resplit,
        )
        overlay = render_detections(crop, boxes)
        tiles = segment_tiles(crop, boxes)
        out_overlay = out_dir / f"line{index:02d}_boxes.jpg"
        out_tiles = out_dir / f"line{index:02d}_segments.jpg"
        cv2.imwrite(str(out_overlay), overlay)
        cv2.imwrite(str(out_tiles), tiles)

        crop_paths: list[str] = []
        box_rows: list[dict] = []
        for gi, (a, b, conf) in enumerate(boxes):
            row = {
                "x_left": int(a) + rx,
                "x_right": int(b) + rx,
                "confidence": float(conf),
            }
            if save_crops and b > a:
                a0 = max(0, int(a) - crop_pad)
                b0 = min(crop.shape[1], int(b) + crop_pad)
                piece = crop[:, a0:b0]
                if piece.size > 0:
                    crop_path = crop_dir / f"l{index:02d}_g{gi:02d}.png"
                    cv2.imwrite(str(crop_path), piece)
                    row["crop_path"] = str(crop_path)
                    crop_paths.append(str(crop_path))
            box_rows.append(row)

        lines_out.append(
            {
                "line": index - 1,
                "y_top": int(band.y_top) + ry,
                "y_bottom": int(band.y_bottom) + ry,
                "n": len(boxes),
                "boxes": box_rows,
                "crop_paths": crop_paths,
                "overlay_path": str(out_overlay),
                "segments_path": str(out_tiles),
            }
        )

    return {
        "ok": True,
        "mode": "stone_detect",
        "image": str(image_path),
        "n_lines": len(lines_out),
        "n_letters": int(sum(ln["n"] for ln in lines_out)),
        "lines": lines_out,
        "inscription_region": region.to_dict(),
        "out_dir": str(out_dir),
        "crops_dir": str(crop_dir) if save_crops else None,
        "device": str(device),
    }


def _iou_span(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, a0) + max(b1, b0) - inter
    return inter / max(union, 1)


def eval_stems(
    stems: list[str],
    *,
    device: torch.device | None = None,
    mode: str = "learned",
) -> dict:
    """Compare learned cuts to hand labels."""
    from annotate_lines import load_usable_real_lines

    if device is None:
        device = resolve_device()
    model = load_detector(device)

    wanted = {s.lower() for s in stems}
    lines = [
        e
        for e in load_usable_real_lines()
        if Path(str(e.get("image_name") or e["crop"])).stem.lower() in wanted
        or str(e["crop"]).rsplit("_line", 1)[0].lower() in wanted
    ]
    report: dict = {"mode": mode, "stems": stems, "lines": []}
    for entry in lines:
        crop = entry["image"]
        h = crop.shape[0]
        tol = max(4, int(0.04 * h))
        gt_bounds = list(entry["boundaries"])
        gt_boxes = [(int(L["x_left"]), int(L["x_right"])) for L in entry["letters"]]
        pred = detect_letters(crop, model, device, mode=mode)
        pred_boxes = [(a, b) for a, b, _c in pred]
        pred_cuts = sorted({b for a, b in pred_boxes[:-1]}) if pred_boxes else []
        # also left edges except 0
        pred_cuts = sorted(
            {a for a, _b in pred_boxes if a > 0} | {b for _a, b in pred_boxes if b < crop.shape[1]}
        )

        hit_b = 0
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
                hit_b += 1
                used.add(best_i)
        missed_b = len(gt_bounds) - hit_b
        false_b = len(pred_cuts) - len(used)

        matched_gt = 0
        used_p: set[int] = set()
        for g0, g1 in gt_boxes:
            best_j = None
            best_iou = 0.5
            for j, (p0, p1) in enumerate(pred_boxes):
                if j in used_p:
                    continue
                iou = _iou_span(g0, g1, p0, p1)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j is not None:
                matched_gt += 1
                used_p.add(best_j)
        merges = len(gt_boxes) - matched_gt
        extras = len(pred_boxes) - len(used_p)

        scores: list[dict] = []
        enhanced = enhance_line(crop)
        for a, b, conf in pred:
            scores.append(
                {
                    "x_left": a,
                    "x_right": b,
                    "completeness": round(score_crop(enhanced, a, b, model, device), 3),
                    "det_conf": round(float(conf), 3),
                }
            )

        row = {
            "crop": entry["crop"],
            "image": entry.get("image_name"),
            "gt_letters": len(gt_boxes),
            "pred_boxes": len(pred_boxes),
            "boundaries_hit": hit_b,
            "boundaries_missed": missed_b,
            "false_internal_cuts": false_b,
            "boxes_matched_iou50": matched_gt,
            "unmatched_gt_merges_or_miss": merges,
            "unmatched_pred_half_or_extra": extras,
            "scores": scores,
        }
        report["lines"].append(row)
        print(
            f"{entry['crop']}: gt={len(gt_boxes)} pred={len(pred_boxes)}  "
            f"bounds hit {hit_b}/{len(gt_bounds)} miss {missed_b} false {false_b}  "
            f"box match {matched_gt} merge/miss {merges} half/extra {extras}",
            flush=True,
        )

    out = OUTPUTS_DIR / "letter_detector" / "eval_learned.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return report


def run_on_image(args: argparse.Namespace) -> None:
    result = detect_image(
        args.image,
        device=resolve_device(args.cpu),
        step=args.step,
        reward_floor=args.reward_floor,
        mode=args.mode,
        save_crops=not args.no_crops,
    )
    if not result["n_lines"]:
        print("No text lines detected.")
        return
    for ln in result["lines"]:
        print(
            f"line {ln['line'] + 1}: {ln['n']} letters -> "
            f"{ln['overlay_path']}  |  {ln['segments_path']}",
            flush=True,
        )
    if result.get("crops_dir"):
        print(
            f"crops: {result['n_letters']} files -> {result['crops_dir']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Shape-aware Musnad letter detector")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (default 0; safer on Windows)")
    parser.add_argument("--patience", type=int, default=2, help="Early stop after N epochs without val gain (0=off)")
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=2.0,
        help="Max negatives per positive when loading the cache (0=keep all)",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training")
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--reward-floor", type=float, default=0.55)
    parser.add_argument(
        "--mode",
        choices=("v2", "learned", "boundary_first", "dp"),
        default="v2",
        help="v2=segmentation baseline (default); others are legacy detectors",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Do not write per-letter crops under outputs/.../crops/",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--eval-stems",
        type=str,
        default=None,
        help="Comma-separated image stems to score against real_lines labels",
    )
    args = parser.parse_args()

    if args.build_cache or args.rebuild_cache and not args.train:
        build_window_cache(max_lines=args.max_lines)
    elif args.train:
        train(args)
    elif args.eval_stems:
        eval_stems([s.strip() for s in args.eval_stems.split(",") if s.strip()], mode=args.mode)
    elif args.image:
        run_on_image(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
