"""
Learned letter-boundary detector for Musnad stone lines.

A projection profile cannot tell an inter-letter gap from a gap inside a
letter, so segmentation needs letter-shape knowledge. This small network
learns that knowledge from synthetic stone lines whose boundaries are exact
by construction (src/generate_stone_lines.py). It predicts, for every pixel
column of a line crop, the probability that a letter boundary passes there.

It is a separate model; ``musnad_final.pth`` is untouched.

Usage:
  python -m src.letter_boundary_net --train
  python -m src.letter_boundary_net --image test_images/real-stone-letters-one-line.jpg
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections.abc import Callable
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

MODEL_PATH = MODELS_DIR / "letter_boundary.pth"
INPUT_HEIGHT = 48
TRAIN_WIDTH = 384
TARGET_SIGMA = 2.5
CACHE_VERSION = 4
BLANK_NEGATIVE_FRACTION = 0.20


def prepare_line(line_bgr: np.ndarray) -> np.ndarray:
    """
    Shared train/inference front-end: fix height, enhance, scale to [0,1].

    Downscaling before enhancement is what makes this cheap: denoising cost
    follows pixel count, and the model only ever sees INPUT_HEIGHT rows.
    """
    if line_bgr.ndim == 2:
        line_bgr = cv2.cvtColor(line_bgr, cv2.COLOR_GRAY2BGR)
    h, w = line_bgr.shape[:2]
    new_w = max(16, int(round(w * INPUT_HEIGHT / h)))
    small = cv2.resize(line_bgr, (new_w, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    return enhance_line_for_ocr(small).astype(np.float32) / 255.0


def _soft_targets(width: int, boundaries: list[float]) -> np.ndarray:
    """Gaussian bump around each true boundary column."""
    target = np.zeros(width, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    for boundary in boundaries:
        target = np.maximum(
            target,
            np.exp(-0.5 * ((xs - boundary) / TARGET_SIGMA) ** 2).astype(np.float32),
        )
    return target


def _letter_objectness(width: int, letters: list[dict], scale: float) -> np.ndarray:
    """1 where a complete glyph/separator bbox exists, 0 on empty stone."""
    target = np.zeros(width, dtype=np.float32)
    for letter in letters:
        left = int(round(float(letter["x_left"]) * scale))
        right = int(round(float(letter["x_right"]) * scale))
        left = max(0, min(width, left))
        right = max(0, min(width, right))
        if right > left:
            target[left:right] = 1.0
    if target.any():
        target = cv2.GaussianBlur(target.reshape(1, -1), (0, 0), sigmaX=1.5).ravel()
        target = np.clip(target, 0.0, 1.0).astype(np.float32)
    return target


def _full_line_objectness(width: int) -> np.ndarray:
    """Fallback for hand annotations where only cuts, not glyph boxes, exist."""
    return np.ones(width, dtype=np.float32)


def _column_loss_weight(
    width: int,
    boundary_target: np.ndarray,
    object_target: np.ndarray,
    letters: list[dict],
    scale: float,
    *,
    real: bool = False,
) -> np.ndarray:
    """
    Per-column loss weight. Real stone gets extra mass on:

    * true cuts between packed glyphs (hard positive boundaries)
    * columns inside a glyph body (hard negatives: do not cut ○/X/H valleys)
    """
    weight = (
        1.0
        + (14.0 if real else 10.0) * boundary_target
        + (12.0 if real else 5.0) * object_target * (1.0 - boundary_target)
    ).astype(np.float32)
    if not letters:
        return weight
    boxes = []
    for letter in letters:
        left = int(round(float(letter["x_left"]) * scale))
        right = int(round(float(letter["x_right"]) * scale))
        left = max(0, min(width, left))
        right = max(0, min(width, right))
        if right > left:
            boxes.append((left, right))
    boxes.sort()
    xs = np.arange(width, dtype=np.float32)
    for i in range(len(boxes) - 1):
        gap = boxes[i + 1][0] - boxes[i][1]
        cut = 0.5 * (boxes[i][1] + boxes[i + 1][0])
        bump = np.exp(-0.5 * ((xs - cut) / TARGET_SIGMA) ** 2).astype(np.float32)
        if gap <= 2:
            weight += 10.0 * bump
        elif gap <= max(3, int(0.08 * max(8, width / 12))):
            weight += 6.0 * bump
    return np.clip(weight, 1.0, 40.0).astype(np.float32)


def _blank_stone_feature(width: int) -> np.ndarray:
    """
    Model-ready empty-stone strip with no letters and no boundaries.

    Real photos contain long margins and inter-line stone texture. If the model
    only sees crops containing glyphs, it can turn stone grain into boundary
    peaks, so blank negatives are part of the training distribution.
    """
    base = random.uniform(0.42, 0.74)
    low = np.random.randn(8, max(8, width // 24)).astype(np.float32)
    low = cv2.resize(low, (width, INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
    low = cv2.GaussianBlur(low, (0, 0), sigmaX=random.uniform(3.0, 9.0))
    fine = np.random.randn(INPUT_HEIGHT, width).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), sigmaX=random.uniform(0.4, 1.4))
    feature = base + low * random.uniform(0.03, 0.10) + fine * random.uniform(0.01, 0.05)
    if random.random() < 0.65:
        x = np.linspace(-0.5, 0.5, width, dtype=np.float32)
        feature += x[None, :] * random.uniform(-0.18, 0.18)
    # Vertical combing so blank stone looks like tooled tablets, not flat noise.
    if random.random() < 0.6:
        comb = np.random.randn(1, width).astype(np.float32)
        comb = cv2.resize(comb, (width, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        comb = cv2.GaussianBlur(comb, (0, 0), sigmaX=random.uniform(0.4, 1.5))
        feature += comb * random.uniform(0.02, 0.08)
    return np.clip(feature, 0.0, 1.0).astype(np.float32)


def _blank_negative_items(count: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(count):
        width = random.randint(TRAIN_WIDTH // 2, TRAIN_WIDTH * 2)
        image = _blank_stone_feature(width)
        zero = np.zeros(width, dtype=np.float32)
        items.append((image, zero, zero.copy(), np.ones(width, dtype=np.float32)))
    return items


def build_cache(root: Path) -> Path:
    """Enhance every line once and store the model-ready arrays next to them."""
    cache_path = root / f"prepared_h{INPUT_HEIGHT}.pkl"
    labels_path = root / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    labels_mtime_ns = labels_path.stat().st_mtime_ns
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if (
            isinstance(cached, dict)
            and cached.get("version") == CACHE_VERSION
            and cached.get("labels_mtime_ns") == labels_mtime_ns
            and len(cached.get("items", [])) == len(labels)
        ):
            print(
                f"Using cached features: {cache_path} "
                f"({len(cached['items'])} lines)",
                flush=True,
            )
            return cache_path
        print("Dataset changed; rebuilding prepared-feature cache.", flush=True)

    print(f"Preparing {len(labels)} lines (one-time, cached afterwards)...", flush=True)
    items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    step = max(1, len(labels) // 40)
    for index, entry in enumerate(labels, start=1):
        image = cv2.imread(str(root / "images" / entry["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        scale = INPUT_HEIGHT / image.shape[0]
        prepared = prepare_line(image)
        boundaries = [b * scale for b in entry["boundaries"]]
        boundary_target = _soft_targets(prepared.shape[1], boundaries)
        object_target = _letter_objectness(
            prepared.shape[1],
            entry.get("letters", []),
            scale,
        )
        if not object_target.any():
            object_target = _full_line_objectness(prepared.shape[1])
        weight = _column_loss_weight(
            prepared.shape[1],
            boundary_target,
            object_target,
            entry.get("letters") or [],
            scale,
            real=False,
        )
        items.append((prepared, boundary_target, object_target, weight))
        if index % step == 0 or index == len(labels):
            pct = 100.0 * index / len(labels)
            print(f"  prepared {index}/{len(labels)}  ({pct:.0f}%)", flush=True)

    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "version": CACHE_VERSION,
                "labels_mtime_ns": labels_mtime_ns,
                "items": items,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Cached {len(items)} prepared lines -> {cache_path}", flush=True)
    return cache_path


def load_real_annotations() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Prepared features for hand-marked real lines, if any exist.

    These are the only examples whose boundaries were not created by pasting
    crops apart, so they are the only ones that can teach the model that a gap
    inside a letter is not a boundary.
    """
    from annotate_lines import load_usable_real_lines

    items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for entry in load_usable_real_lines():
        image = entry["image"]
        scale = INPUT_HEIGHT / image.shape[0]
        prepared = prepare_line(image)
        boundaries = [b * scale for b in entry["boundaries"]]
        boundary_target = _soft_targets(prepared.shape[1], boundaries)
        object_target = _letter_objectness(
            prepared.shape[1], entry["letters"], scale
        )
        if not object_target.any():
            object_target = _full_line_objectness(prepared.shape[1])
        weight = _column_loss_weight(
            prepared.shape[1],
            boundary_target,
            object_target,
            entry["letters"],
            scale,
            real=True,
        )
        items.append((prepared, boundary_target, object_target, weight))
    return items


class LineBoundaryDataset(Dataset):
    """Cached prepared lines; random width crops and flips per item."""

    def __init__(
        self,
        root: Path,
        train: bool,
        val_fraction: float = 0.1,
        real_items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
        real_share: float = 0.0,
    ) -> None:
        cache_path = build_cache(root)
        with cache_path.open("rb") as handle:
            items = pickle.load(handle)["items"]
        split = max(1, int(len(items) * val_fraction))
        self.items = items[split:] if train else items[:split]
        self.train = train
        if not self.items:
            raise RuntimeError(f"No usable samples under {root}")

        if real_items and real_share > 0:
            # Hand-marked lines are few; repeat them so they carry real weight
            # against the far larger synthetic set.
            target = int(len(self.items) * real_share / max(1e-6, 1 - real_share))
            repeats = max(1, round(target / len(real_items)))
            self.items = self.items + real_items * repeats
            self.real_count = len(real_items) * repeats
        else:
            self.real_count = 0

        if train:
            blank_count = max(1, int(len(self.items) * BLANK_NEGATIVE_FRACTION))
            self.items.extend(_blank_negative_items(blank_count))
            self.blank_count = blank_count
        else:
            self.blank_count = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        packed = self.items[index]
        image, boundary_target, object_target = packed[0], packed[1], packed[2]
        column_weight = packed[3] if len(packed) > 3 else None
        width = image.shape[1]
        if width >= TRAIN_WIDTH:
            start = random.randint(0, width - TRAIN_WIDTH) if self.train else 0
            image = image[:, start : start + TRAIN_WIDTH]
            boundary_target = boundary_target[start : start + TRAIN_WIDTH]
            object_target = object_target[start : start + TRAIN_WIDTH]
            if column_weight is not None:
                column_weight = column_weight[start : start + TRAIN_WIDTH]
        else:
            pad = TRAIN_WIDTH - width
            image = np.pad(image, ((0, 0), (0, pad)), mode="edge")
            boundary_target = np.pad(boundary_target, (0, pad))
            object_target = np.pad(object_target, (0, pad))
            if column_weight is not None:
                column_weight = np.pad(column_weight, (0, pad), constant_values=1.0)
        if column_weight is None:
            column_weight = (
                1.0
                + 10.0 * boundary_target
                + 5.0 * object_target * (1.0 - boundary_target)
            ).astype(np.float32)
        if self.train and random.random() < 0.5:
            image = image[:, ::-1].copy()
            boundary_target = boundary_target[::-1].copy()
            object_target = object_target[::-1].copy()
            column_weight = column_weight[::-1].copy()
        target = np.stack([boundary_target, object_target, column_weight]).astype(
            np.float32
        )
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(target)


class _DilatedResidual(nn.Module):
    """Width-dilated 1-D block; residual so deep stacks stay trainable."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class LetterBoundaryNet(nn.Module):
    """
    Column-resolution boundary scorer: pools height, keeps full width.

    Deciding whether a column splits one letter or separates two requires
    seeing the neighbouring letters, so the dilated stack is sized to span
    several letter widths rather than a single glyph.
    """

    def __init__(self, channels: int = 96) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1)),
            )

        self.tower = nn.Sequential(
            block(1, 32), block(32, 48), block(48, 64), block(64, channels)
        )
        self.context = nn.Sequential(
            *[_DilatedResidual(channels, d) for d in (1, 2, 4, 8, 16)]
        )
        self.head = nn.Sequential(
            nn.Conv1d(channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tower(x)  # (B, C, 3, W)
        columns = features.mean(dim=2)  # (B, C, W)
        return self.head(self.context(columns))  # (B, 2, W) logits


def train(args: argparse.Namespace) -> None:
    device = print_device_info(resolve_device(args.cpu))
    root = Path(args.data_dir)
    if not (root / "labels.json").exists():
        print(f"No dataset at {root}. Run: python -m src.generate_stone_lines")
        sys.exit(1)

    real_items = load_real_annotations()
    real_val: list = []
    if real_items:
        # Hold out real lines for validation; synthetic scores near 1.0 and
        # would hide exactly the failures these annotations exist to catch.
        rng = random.Random(7)
        rng.shuffle(real_items)
        holdout = max(1, len(real_items) // 5)
        real_val, real_items = real_items[:holdout], real_items[holdout:]
        print(
            f"Real annotated lines: {len(real_items)} train / {len(real_val)} val",
            flush=True,
        )
    else:
        print(
            "No real annotations found. Train on synthetic only "
            "(run: python -m src.annotate_lines --all)",
            flush=True,
        )

    train_ds = LineBoundaryDataset(
        root, train=True, real_items=real_items, real_share=args.real_share
    )
    val_ds = LineBoundaryDataset(root, train=False)
    print(
        f"train={len(train_ds)} (real copies {train_ds.real_count})  "
        f"blank negatives {train_ds.blank_count})  val={len(val_ds)}",
        flush=True,
    )

    loader_kwargs = dataloader_kwargs(device)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_kwargs)

    model = LetterBoundaryNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Boundary columns are rare; upweight them so the net cannot win by
        # predicting "no boundary" everywhere. The second channel teaches
        # whether a column belongs to any glyph at all, so empty stone can be
        # rejected at inference instead of becoming a crop.
        boundary_logits = logits[:, 0]
        object_logits = logits[:, 1]
        boundary_target = target[:, 0]
        object_target = target[:, 1]
        if target.shape[1] >= 3:
            boundary_weight = target[:, 2]
        else:
            boundary_weight = (
                1.0
                + 10.0 * boundary_target
                + 5.0 * object_target * (1.0 - boundary_target)
            )
        boundary_raw = nn.functional.binary_cross_entropy_with_logits(
            boundary_logits, boundary_target, reduction="none"
        )
        boundary_loss = (boundary_raw * boundary_weight).mean()

        object_weight = 1.0 + 2.0 * object_target
        object_raw = nn.functional.binary_cross_entropy_with_logits(
            object_logits, object_target, reduction="none"
        )
        object_loss = (object_raw * object_weight).mean()
        return boundary_loss + 0.35 * object_loss

    best_val = float("inf")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_batches = len(train_loader)
        report_every = max(1, n_batches // 5)
        for step, (images, targets) in enumerate(train_loader, start=1):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            if step % report_every == 0 or step == n_batches:
                print(
                    f"  epoch {epoch:02d}  batch {step}/{n_batches}  "
                    f"loss {total / (step * train_loader.batch_size):.4f}",
                    flush=True,
                )
        scheduler.step()
        train_loss = total / len(train_ds)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                total += loss_fn(model(images), targets).item() * images.size(0)
        val_loss = total / len(val_ds)

        real_loss = None
        if real_val:
            total = 0.0
            with torch.no_grad():
                for packed in real_val:
                    image, boundary_target, object_target = packed[0], packed[1], packed[2]
                    column_weight = packed[3] if len(packed) > 3 else (
                        1.0
                        + 14.0 * boundary_target
                        + 12.0 * object_target * (1.0 - boundary_target)
                    )
                    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
                    stacked = np.stack(
                        [boundary_target, object_target, column_weight]
                    ).astype(np.float32)
                    y = torch.from_numpy(stacked).unsqueeze(0).to(device)
                    total += loss_fn(model(x), y).item()
            real_loss = total / len(real_val)

        # Select on real lines when they exist: synthetic validation saturates.
        score = real_loss if real_loss is not None else val_loss
        marker = ""
        if score < best_val:
            best_val = score
            torch.save({"state_dict": model.state_dict()}, MODEL_PATH)
            marker = "  saved"
        extra = f"  real {real_loss:.4f}" if real_loss is not None else ""
        print(
            f"epoch {epoch:02d}/{args.epochs}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}{extra}{marker}",
            flush=True,
        )
    label = "real" if real_val else "synthetic"
    print(f"Best {label} val loss {best_val:.4f} -> {MODEL_PATH}", flush=True)


def load_boundary_model(device: torch.device) -> LetterBoundaryNet:
    model = LetterBoundaryNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as error:
        raise SystemExit(
            f"{MODEL_PATH} was trained with a different architecture.\n"
            "Retrain it with: python -m src.letter_boundary_net --train --epochs 30\n"
            f"({error})"
        ) from error
    model.eval()
    return model


def predict_profiles(
    line_bgr: np.ndarray,
    model: LetterBoundaryNet,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary and glyph-presence probabilities at the ORIGINAL line width."""
    prepared = prepare_line(line_bgr)
    tensor = torch.from_numpy(prepared).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy()
    original_w = line_bgr.shape[1]
    xs = np.linspace(0, probs.shape[1] - 1, original_w)
    boundary = np.interp(xs, np.arange(probs.shape[1]), probs[0])
    objectness = np.interp(xs, np.arange(probs.shape[1]), probs[1])
    return boundary, objectness


def predict_boundary_profile(
    line_bgr: np.ndarray,
    model: LetterBoundaryNet,
    device: torch.device,
) -> np.ndarray:
    """Per-column boundary probability at the ORIGINAL line width."""
    boundary, _objectness = predict_profiles(line_bgr, model, device)
    return boundary


def letter_activity_profile(line_bgr: np.ndarray) -> np.ndarray:
    """
    Per-column evidence that the line contains carved strokes, not bare stone.

    This is deliberately independent from the boundary net. It is used only to
    reject empty segments after peak picking, so stone texture or wide margins
    do not become glyph crops.
    """
    prepared = prepare_line(line_bgr)
    gx = cv2.Sobel(prepared, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prepared, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    dark = np.maximum(0.0, float(np.percentile(prepared, 65)) - prepared)
    score = 0.65 * gradient + 0.35 * dark
    column = score.mean(axis=0)
    column = cv2.GaussianBlur(column.reshape(1, -1), (0, 0), sigmaX=2.0).ravel()
    lo, hi = float(np.percentile(column, 10)), float(np.percentile(column, 98))
    norm = np.clip((column - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    xs = np.linspace(0, norm.shape[0] - 1, line_bgr.shape[1])
    return np.interp(xs, np.arange(norm.shape[0]), norm).astype(np.float32)


def _peak_prominence(signal: np.ndarray, peak: int) -> float:
    """How far the curve must fall from `peak` before it climbs higher again."""
    height = float(signal[peak])
    left = height
    for x in range(peak - 1, -1, -1):
        if signal[x] > height:
            break
        left = min(left, float(signal[x]))
    right = height
    for x in range(peak + 1, signal.shape[0]):
        if signal[x] > height:
            break
        right = min(right, float(signal[x]))
    return height - max(left, right)


def boundary_peak_maps(
    profile: np.ndarray,
    line_height: int,
    threshold: float = 0.35,
    *,
    min_prominence_frac: float = 0.15,
    ripple_height: int | None = None,
) -> tuple[np.ndarray, list[int], dict[int, float]]:
    """
    Smooth the boundary curve and list local peaks above threshold.

    Returns ``(smooth_profile, peak_xs, prominence_by_x)``.
    """
    smooth = cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=2.0
    ).ravel()
    if smooth.size < 3:
        return smooth, [], {}
    level = max(threshold, 0.5 * float(smooth.max()))
    # Short / dense rows (test-38): require a clearer peak above stone ripples.
    ref_h = int(ripple_height) if ripple_height is not None else int(line_height)
    if ref_h > 0 and ref_h < 72:
        level = max(level, 0.62 * float(smooth.max()))

    interior = np.arange(1, smooth.shape[0] - 1)
    is_peak = (smooth[interior] >= smooth[interior - 1]) & (
        smooth[interior] > smooth[interior + 1]
    )
    candidates = [int(x) for x in interior[is_peak] if smooth[x] >= level]
    prominence = {x: _peak_prominence(smooth, x) for x in candidates}
    candidates = [
        x for x in candidates if prominence[x] >= min_prominence_frac * level
    ]
    return smooth, candidates, prominence


def boundaries_from_profile(
    profile: np.ndarray,
    line_height: int,
    threshold: float = 0.35,
    pair_prominence: float = 0.10,
    close_pair_ok: Callable[[int, int], bool] | None = None,
    *,
    source_height: int | None = None,
) -> list[int]:
    """
    Peak-pick the probability curve into concrete cut columns.

    Two cuts closer than a letter width normally mean one letter is being
    split, so they are suppressed. The exception is a word separator, a
    genuinely narrow glyph needing a cut on each side. Prominence decides that
    exception rather than raw confidence: a separator edge and a spurious
    ripple can sit at the same height, but only the separator edge stands
    clear of its surroundings.

    ``close_pair_ok(weaker, stronger)`` may veto the separator exception (e.g.
    when the weaker peak bisects a connected stroke — test-27 L1 @429).
    """
    ref_h = int(source_height) if source_height is not None else int(line_height)
    smooth, candidates, prominence = boundary_peak_maps(
        profile,
        line_height,
        threshold=threshold,
        min_prominence_frac=0.26 if ref_h < 72 else 0.15,
        ripple_height=ref_h,
    )
    if not candidates:
        return []

    letter_dist = max(8, int(0.30 * line_height))
    # Allow slightly closer cut pairs so both sides of a ``|`` survive NMS.
    separator_dist = max(3, int(0.08 * line_height))
    pair_prom = max(pair_prominence, 0.22 if ref_h < 72 else pair_prominence)
    picked: list[int] = []
    for x in sorted(candidates, key=lambda c: -float(smooth[c])):
        conflict = False
        for p in picked:
            distance = abs(x - p)
            if distance >= letter_dist:
                continue
            both_clear = (
                prominence[x] >= pair_prom
                and prominence[p] >= pair_prom
            )
            sep_pair = both_clear and distance >= separator_dist
            if sep_pair and close_pair_ok is not None:
                weaker = x if float(smooth[x]) <= float(smooth[p]) else p
                stronger = p if weaker == x else x
                sep_pair = bool(close_pair_ok(weaker, stronger))
            if not sep_pair:
                conflict = True
                break
        if not conflict:
            picked.append(x)
    return sorted(picked)


def suppress_empty_segments(
    line_bgr: np.ndarray,
    cuts: list[int],
    *,
    objectness: np.ndarray | None = None,
    min_peak_activity: float = 0.18,
    min_mean_activity: float = 0.035,
    min_objectness: float = 0.25,
) -> list[int]:
    """
    Remove cuts that would create a crop containing no letter strokes.

    Leading/trailing empty margins are discarded. If two cuts isolate only
    blank stone between two active glyphs, collapse that blank piece into one
    boundary at its centre.
    """
    if not cuts:
        return cuts

    activity = letter_activity_profile(line_bgr)
    if objectness is None:
        objectness = np.ones_like(activity)
    else:
        objectness = np.asarray(objectness, dtype=np.float32)
    edges = [0] + sorted(c for c in cuts if 0 < c < line_bgr.shape[1]) + [line_bgr.shape[1]]
    segments: list[tuple[int, int, bool]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        sl = activity[left:right]
        obj = objectness[left:right]
        peak = float(sl.max(initial=0.0)) if sl.size else 0.0
        mean = float(sl.mean()) if sl.size else 0.0
        obj_mean = float(obj.mean()) if obj.size else 0.0
        h = int(line_bgr.shape[0])
        width = right - left
        # Word-separator ``|`` is a thin stem. Objectness is trained on
        # letter-shaped blobs and often stays below min_objectness here —
        # do not collapse that crop into the inter-letter gutter.
        carved_thin = (
            h > 0
            and 2 <= width <= max(4, int(0.28 * h))
            and peak >= 0.40
            and mean >= 0.12
        )
        active = (
            peak >= min_peak_activity
            and mean >= min_mean_activity
            and obj_mean >= min_objectness
        ) or carved_thin
        segments.append((left, right, active))

    active_indices = [i for i, (_left, _right, active) in enumerate(segments) if active]
    if not active_indices:
        return []

    # Drop empty margins outside the first/last active segment.
    first, last = active_indices[0], active_indices[-1]
    segments = segments[first : last + 1]

    filtered: list[int] = []
    i = 0
    while i < len(segments) - 1:
        left_a, right_a, active_a = segments[i]
        left_b, right_b, active_b = segments[i + 1]
        if active_a and active_b:
            filtered.append(right_a)
            i += 1
            continue
        if active_a and not active_b:
            # One or more blank pieces between active glyphs: a single cut in
            # the middle of the blank corridor is enough.
            blank_left = left_b
            j = i + 1
            while j < len(segments) and not segments[j][2]:
                j += 1
            if j < len(segments):
                blank_right = segments[j - 1][1]
                filtered.append((blank_left + blank_right) // 2)
            i = max(j, i + 1)
            continue
        i += 1

    return sorted(set(filtered))


def segment_tiles(crop: np.ndarray, cuts: list[int], per_row: int = 10) -> np.ndarray:
    """Lay the cut-out segments on a numbered grid so each one can be judged."""
    edges = [0] + list(cuts) + [crop.shape[1]]
    pieces = [crop[:, edges[i] : edges[i + 1]] for i in range(len(edges) - 1)]
    pieces = [p for p in pieces if p.shape[1] > 0]
    if not pieces:
        return np.zeros((10, 10, 3), np.uint8)

    cell_w = max(p.shape[1] for p in pieces) + 8
    cell_h = crop.shape[0] + 22
    rows = (len(pieces) + per_row - 1) // per_row
    sheet = np.full((rows * cell_h, per_row * cell_w, 3), 30, np.uint8)
    for index, piece in enumerate(pieces):
        r, c = divmod(index, per_row)
        y, x = r * cell_h, c * cell_w
        sheet[y + 20 : y + 20 + piece.shape[0], x + 4 : x + 4 + piece.shape[1]] = piece
        cv2.putText(
            sheet, str(index + 1), (x + 4, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1, cv2.LINE_AA,
        )
    return sheet


def run_on_image(args: argparse.Namespace) -> None:
    from .empty_segment_filter import (
        kept_segments,
        mark_empty_segments,
        render_segment_sheet,
        segments_from_cuts,
    )
    from .stone_glyph_segmentation import detect_line_bands, load_bgr

    device = resolve_device(args.cpu)
    model = load_boundary_model(device)

    image_path = Path(args.image)
    image = load_bgr(image_path)
    lines = detect_line_bands(image)
    if not lines:
        print("No text lines detected.")
        return

    out_dir = Path("outputs/letter_boundary") / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, line in enumerate(lines, start=1):
        crop = image[line.y_top : line.y_bottom]
        profile, objectness = predict_profiles(crop, model, device)
        cuts = boundaries_from_profile(
            profile,
            crop.shape[0],
            threshold=args.threshold,
            pair_prominence=args.pair_prominence,
        )
        candidates = segments_from_cuts(crop, cuts)
        candidates = mark_empty_segments(
            candidates,
            line_activity=letter_activity_profile(crop),
            objectness=objectness,
            min_activity_mean=args.min_segment_mean_activity,
            min_activity_peak=args.min_segment_activity,
            min_edge_peak=args.min_segment_edge_peak,
            min_object_mean=args.min_objectness,
            min_object_peak=args.min_objectness_peak,
        )

        vis = crop.copy()
        h = vis.shape[0]
        for x in cuts:
            cv2.line(vis, (x, 0), (x, h - 1), (0, 255, 80), 2)
        curve = np.zeros((60, vis.shape[1], 3), dtype=np.uint8)
        pts = np.column_stack(
            [np.arange(vis.shape[1]), 58 - (profile * 56).astype(int)]
        ).astype(np.int32)
        cv2.polylines(curve, [pts], False, (0, 200, 255), 1)
        obj_pts = np.column_stack(
            [np.arange(vis.shape[1]), 58 - (objectness * 56).astype(int)]
        ).astype(np.int32)
        cv2.polylines(curve, [obj_pts], False, (255, 120, 0), 1)
        stacked = np.vstack([vis, curve])
        out = out_dir / f"line{index:02d}_boundaries.jpg"
        cv2.imwrite(str(out), stacked)
        candidate_tiles = out_dir / f"line{index:02d}_candidate_segments.jpg"
        cv2.imwrite(
            str(candidate_tiles),
            render_segment_sheet(candidates, show_rejected=True),
        )
        tiles = out_dir / f"line{index:02d}_segments.jpg"
        cv2.imwrite(str(tiles), render_segment_sheet(candidates))
        kept = kept_segments(candidates)
        print(
            f"line {index}: {len(kept)}/{len(candidates)} kept -> "
            f"{out}  |  {tiles}  |  debug {candidate_tiles}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Letter boundary network")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(STONE_LINES_DIR))
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--real-share",
        type=float,
        default=0.40,
        help="Share of training samples drawn from hand-marked real lines",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum boundary probability for a column to be considered",
    )
    parser.add_argument(
        "--pair-prominence",
        type=float,
        default=0.10,
        help="Prominence both peaks need before a narrow (separator) segment is cut",
    )
    parser.add_argument(
        "--min-segment-activity",
        type=float,
        default=0.45,
        help="Drop any predicted segment whose strongest stroke evidence is below this",
    )
    parser.add_argument(
        "--min-segment-mean-activity",
        type=float,
        default=0.15,
        help="Drop any predicted segment whose average stroke evidence is below this",
    )
    parser.add_argument(
        "--min-segment-edge-peak",
        type=float,
        default=0.98,
        help="Keep weak segments if their strongest local edge is above this",
    )
    parser.add_argument(
        "--min-objectness",
        type=float,
        default=0.25,
        help="Drop any predicted segment whose mean learned letter-presence is below this",
    )
    parser.add_argument(
        "--min-objectness-peak",
        type=float,
        default=0.55,
        help="Drop any predicted segment whose strongest letter-presence is below this",
    )
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.image:
        run_on_image(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
