"""
Musnad OCR production inference (no server).

Load the packaged checkpoint + prototype/shape banks and run the same
prediction pipeline as the training project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from .preprocessing import (
    IMG_SIZE,
    deskew_image,
    image_to_tensor,
    prepare_original_view,
    prepare_stone_view,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PACKAGE_ROOT / "model"
CONFIG_DIR = PACKAGE_ROOT / "config"
DEFAULT_CHECKPOINT = MODEL_DIR / "musnad_final.pth"
PROTOTYPES_PATH = MODEL_DIR / "class_prototypes.pt"
SHAPE_BANK_PATH = MODEL_DIR / "shape_bank.pt"
LABELS_PATH = CONFIG_DIR / "labels.json"
PREPROCESSING_PATH = CONFIG_DIR / "preprocessing.json"

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
FEATURE_DIM = 48

with LABELS_PATH.open(encoding="utf-8") as f:
    _LABELS_CFG = json.load(f)
with PREPROCESSING_PATH.open(encoding="utf-8") as f:
    _PRE_CFG = json.load(f)

INDEX_TO_CHAR: List[str] = list(_LABELS_CFG["index_to_char"])
NUM_CLASSES = int(_LABELS_CFG["num_classes"])
TTA_ANGLES_DEG = tuple(_PRE_CFG["rotation_tta"]["angles_deg"])
TTA_TRIGGER = float(_PRE_CFG["rotation_tta"]["trigger_if_confidence_below"])
TRUST_CFG = _PRE_CFG["trust_fusion"]

_NAME_BY_CHAR = {
    e["character"]: e
    for e in _LABELS_CFG.get("entries", [])
    if e.get("type") == "letter"
}
_NUMERAL_BY_LABEL = {
    e["character"]: e
    for e in _LABELS_CFG.get("entries", [])
    if e.get("type") == "numeral"
}

SHAPE_GROUPS: List[Set[str]] = [
    {"NUM_1", "𐩠", "𐩡", "𐩬", "𐩱"},
    {"NUM_2", "NUM_3", "NUM_4", "NUM_5", "NUM_6"},
    {"NUM_10", "𐩥", "𐩲", "𐩶"},
    {"𐩣", "𐩨", "𐩩", "𐩯", "𐩴"},
    {"NUM_50", "𐩤", "𐩫", "𐩬", "𐩱", "𐩴"},
    {"𐩥", "𐩷", "𐩸", "𐩹"},
    {"𐩠", "𐩨", "𐩴", "𐩵", "𐩹", "𐩺"},
    {"𐩧", "𐩺", "𐩻", "𐩼"},
    {"𐩦", "𐩩", "𐩪", "𐩯"},
    {"𐩥", "𐩧", "𐩰"},
    {"𐩢", "𐩭", "𐩮"},
    {"NUM_100", "NUM_1000"},
]

HARD_LOOKALIKE_PAIRS: List[Set[str]] = [
    {"𐩩", "𐩯"},
    {"𐩬", "𐩱"},
    {"𐩥", "𐩲"},
    {"𐩨", "𐩴"},
    {"𐩠", "𐩡"},
    {"𐩫", "𐩬"},
    {"𐩳", "𐩷"},
    {"𐩨", "𐩷"},
    {"𐩨", "𐩳"},
]


def hard_lookalike_partners(label: Optional[str]) -> Set[str]:
    if not label:
        return set()
    partners: Set[str] = set()
    for pair in HARD_LOOKALIKE_PAIRS:
        if label in pair:
            partners.update(p for p in pair if p != label)
    return partners


def resolve_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GlyphFocusAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        mid = max(channels // 8, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        channel_w = self.channel(x).view(b, c, 1, 1)
        x = x * channel_w
        h_map = F.adaptive_avg_pool2d(x, (h, 1)).expand(-1, -1, h, w)
        v_map = F.adaptive_avg_pool2d(x, (1, w)).expand(-1, -1, h, w)
        cues = torch.cat(
            [h_map.mean(1, keepdim=True), v_map.mean(1, keepdim=True)], dim=1
        )
        return x * self.spatial(cues)


class GeMPool(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.pow(1.0 / p)


class MusnadCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.glyph_attention = GlyphFocusAttention(256)
        self.pool = GeMPool(p=3.0)
        self.backbone = nn.Sequential(self.stem, self.glyph_attention, self.pool)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stem(x)
        feats = self.glyph_attention(feats)
        return self.pool(feats).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def load_model(path: Path, device: torch.device) -> Tuple[MusnadCNN, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MusnadCNN(num_classes=ckpt.get("num_classes", NUM_CLASSES))
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model, ckpt


def load_prototypes(path: Path = PROTOTYPES_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def load_shape_bank(path: Path = SHAPE_BANK_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_numeral_label(label: str) -> bool:
    return label in _NUMERAL_BY_LABEL or str(label).startswith("NUM_")


def _enrich_label(char: str) -> Tuple[Optional[str], Optional[str]]:
    if char in _NUMERAL_BY_LABEL:
        return f"NUMBER {_NUMERAL_BY_LABEL[char]['value']}", None
    meta = _NAME_BY_CHAR.get(char)
    if meta:
        return meta.get("name"), meta.get("codepoint")
    return None, None


def confusable_map(labels: Sequence[str]) -> Dict[str, Set[str]]:
    label_set = set(labels)
    out: Dict[str, Set[str]] = {lab: set() for lab in labels}
    for group in SHAPE_GROUPS:
        present = [g for g in group if g in label_set]
        for a in present:
            out[a].update(b for b in present if b != a)
    return out


def are_confusable(a: str, b: str, mapping: Dict[str, Set[str]]) -> bool:
    return b in mapping.get(a, set()) or a in mapping.get(b, set())


@dataclass(frozen=True)
class InferenceBranch:
    key: str
    label: str
    prepare: Callable[[Image.Image], Image.Image]


INFERENCE_BRANCHES: List[InferenceBranch] = [
    InferenceBranch("original", "original", prepare_original_view),
    InferenceBranch("preprocessed", "preprocessed", prepare_stone_view),
]


def load_external_image(image_path: Path | str) -> Image.Image:
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported format '{suffix}'. Use one of: "
            + ", ".join(sorted(SUPPORTED_SUFFIXES))
        )
    try:
        image = Image.open(image_path)
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not read image file: {image_path}") from exc
    return image


@torch.no_grad()
def infer_tensor(
    model: nn.Module,
    tensor: torch.Tensor,
    index_to_char: Sequence[str],
    device: torch.device,
    top_k: int = 3,
    *,
    letters_only: bool = False,
) -> dict:
    tensor = tensor.to(device, non_blocking=device.type == "cuda")
    logits = model(tensor)
    if letters_only:
        masked = logits.clone()
        for i, lab in enumerate(index_to_char):
            if _is_numeral_label(str(lab)):
                masked[0, i] = -1e9
        logits = masked
    probs = torch.softmax(logits, dim=1)[0]
    conf, pred_idx = torch.max(probs, dim=0)
    pred_idx = int(pred_idx.item())
    char = index_to_char[pred_idx]
    name, codepoint = _enrich_label(char)
    topk = torch.topk(probs, k=min(top_k, probs.numel()))
    alternatives = []
    for p, i in zip(topk.values.tolist(), topk.indices.tolist()):
        ch = index_to_char[int(i)]
        alt_name, _ = _enrich_label(ch)
        alternatives.append(
            {"character": ch, "name": alt_name, "confidence": float(p)}
        )
    return {
        "character": char,
        "name": name,
        "codepoint": codepoint,
        "confidence": float(conf.item()),
        "index": pred_idx,
        "top_k": alternatives,
    }


def select_best_attempt(attempts: Sequence[dict]) -> dict:
    if not attempts:
        raise ValueError("No inference attempts to select from")
    by_key = {a.get("branch"): a for a in attempts}
    original = by_key.get("original")
    if original is not None:
        for other in attempts:
            if other.get("branch") == "original":
                continue
            o_num = _is_numeral_label(original["character"])
            t_num = _is_numeral_label(other["character"])
            if o_num == t_num:
                continue
            if not o_num and float(original.get("confidence") or 0.0) >= 0.08:
                return original
            if o_num and not t_num and float(other.get("confidence") or 0.0) >= 0.20:
                if float(other["confidence"]) >= float(original["confidence"]) - 0.05:
                    return other
            if other["confidence"] < original["confidence"] + 0.55:
                return original
    return max(attempts, key=lambda a: a["confidence"])


@torch.no_grad()
def match_prototypes(
    features: torch.Tensor,
    bank: dict,
    *,
    top_k: int = 3,
    letters_only: bool = False,
) -> dict:
    prototypes = bank["prototypes"]
    labels = bank["index_to_char"]
    if features.ndim == 1:
        features = features.unsqueeze(0)
    features = F.normalize(features.cpu(), p=2, dim=-1, eps=1e-6)
    sims = torch.mm(features, prototypes.t())[0]
    counts = bank.get("counts")
    if counts is not None:
        sims = sims.clone()
        sims[counts <= 0] = -1.0
    if letters_only:
        sims = sims.clone()
        for i, lab in enumerate(labels):
            if _is_numeral_label(str(lab)):
                sims[i] = -1.0
    conf, pred_idx = torch.max(sims, dim=0)
    pred_idx = int(pred_idx.item())
    topk = torch.topk(sims, k=min(top_k, sims.numel()))
    alternatives = [
        {"character": labels[int(i)], "similarity": float(p)}
        for p, i in zip(topk.values.tolist(), topk.indices.tolist())
    ]
    second = alternatives[1]["similarity"] if len(alternatives) > 1 else -1.0
    return {
        "character": labels[pred_idx],
        "index": pred_idx,
        "similarity": float(conf.item()),
        "margin": float(conf.item() - second),
        "top_k": alternatives,
    }


def _to_ink_mask(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = []
    for m in (dark, light):
        frac = float((m > 0).mean())
        if 0.02 <= frac <= 0.55:
            candidates.append((abs(frac - 0.18), m))
    mask = min(candidates, key=lambda t: t[0])[1] if candidates else dark
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (mask > 0).astype(np.uint8)


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    skel = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)


def _projection_bins(mask: np.ndarray, bins: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    row = mask.sum(axis=1).astype(np.float32)
    col = mask.sum(axis=0).astype(np.float32)
    row = np.array([row[i * IMG_SIZE // bins : (i + 1) * IMG_SIZE // bins].mean() for i in range(bins)])
    col = np.array([col[i * IMG_SIZE // bins : (i + 1) * IMG_SIZE // bins].mean() for i in range(bins)])
    row = row / (row.max() + 1e-6)
    col = col / (col.max() + 1e-6)
    return row, col


def extract_shape_signature(image: Image.Image | np.ndarray) -> np.ndarray:
    gray = np.array(image.convert("L")) if isinstance(image, Image.Image) else image
    mask = _to_ink_mask(gray)
    skel = _skeletonize(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    aspect = bw / bh
    fill = float(len(xs)) / float(bw * bh)
    cx, cy = float(xs.mean()), float(ys.mean())
    contours, hierarchy = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    holes = 0
    if hierarchy is not None:
        for h in hierarchy[0]:
            if h[3] >= 0:
                holes += 1
    n_cc = int(cv2.connectedComponents(mask)[0]) - 1
    euler = float(n_cc - holes)
    nbr = _neighbor_count(skel)
    endpoints = int(((skel > 0) & (nbr == 1)).sum())
    junctions = int(((skel > 0) & (nbr >= 3)).sum())
    skel_len = int(skel.sum()) + 1
    skel_pts = np.column_stack(np.where(skel > 0))
    horiz = vert = diag = 0.0
    if len(skel_pts) > 4:
        for y, x in skel_pts[:: max(1, len(skel_pts) // 80)]:
            patch = skel[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
            if patch.shape != (3, 3):
                continue
            if patch[1, 0] or patch[1, 2]:
                horiz += 1
            if patch[0, 1] or patch[2, 1]:
                vert += 1
            if patch[0, 0] or patch[0, 2] or patch[2, 0] or patch[2, 2]:
                diag += 1
        total = horiz + vert + diag + 1e-6
        horiz, vert, diag = horiz / total, vert / total, diag / total
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = 0.5 * max(bw, bh)
    ring = ((rr > 0.25 * r_max) & (rr < 0.55 * r_max)).astype(np.float32)
    core = (rr <= 0.20 * r_max).astype(np.float32)
    ring_fill = float((mask * ring).sum() / (ring.sum() + 1e-6))
    core_fill = float((mask * core).sum() / (core.sum() + 1e-6))
    col = mask[:, int(np.clip(cx, 0, IMG_SIZE - 1))]
    row = mask[int(np.clip(cy, 0, IMG_SIZE - 1)), :]
    vbar = float(col.mean())
    hbar = float(row.mean())
    q = []
    for y_slice, x_slice in (
        (slice(0, IMG_SIZE // 2), slice(0, IMG_SIZE // 2)),
        (slice(0, IMG_SIZE // 2), slice(IMG_SIZE // 2, IMG_SIZE)),
        (slice(IMG_SIZE // 2, IMG_SIZE), slice(0, IMG_SIZE // 2)),
        (slice(IMG_SIZE // 2, IMG_SIZE), slice(IMG_SIZE // 2, IMG_SIZE)),
    ):
        q.append(float(mask[y_slice, x_slice].mean()))
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    prow, pcol = _projection_bins(mask, bins=8)
    feat = np.concatenate(
        [
            np.array(
                [
                    aspect,
                    fill,
                    holes / 3.0,
                    endpoints / 10.0,
                    junctions / 10.0,
                    skel_len / 200.0,
                    horiz,
                    vert,
                    diag,
                    ring_fill,
                    core_fill,
                    vbar,
                    hbar,
                    abs(vbar - hbar),
                    ring_fill * vbar,
                    euler / 5.0,
                    n_cc / 5.0,
                    float(cx) / IMG_SIZE,
                    float(cy) / IMG_SIZE,
                    bw / IMG_SIZE,
                    bh / IMG_SIZE,
                    float(mask[:, : IMG_SIZE // 3].mean()),
                    float(mask[:, -IMG_SIZE // 3 :].mean()),
                    float(mask[: IMG_SIZE // 3, :].mean()),
                    float(mask[-IMG_SIZE // 3 :, :].mean()),
                    *q,
                ],
                dtype=np.float32,
            ),
            hu.astype(np.float32),
            prow.astype(np.float32),
            pcol.astype(np.float32),
        ]
    )
    if feat.size < FEATURE_DIM:
        feat = np.pad(feat, (0, FEATURE_DIM - feat.size))
    return feat[:FEATURE_DIM].astype(np.float32)


def _whiten(vec: torch.Tensor, bank: dict) -> torch.Tensor:
    mean = bank.get("feat_mean")
    std = bank.get("feat_std")
    if mean is None or std is None:
        return vec / (torch.linalg.vector_norm(vec) + 1e-6)
    return (vec - mean) / std


@torch.no_grad()
def match_shape(
    signature: np.ndarray,
    bank: dict,
    *,
    top_k: int = 3,
    letters_only: bool = False,
) -> dict:
    sig = torch.from_numpy(np.asarray(signature, dtype=np.float32))
    mats = bank["signatures"]
    labels = bank["index_to_char"]
    counts = bank.get("counts")
    n_classes = mats.shape[0]
    dim = int(bank.get("feature_dim", mats.shape[1]))
    if sig.numel() != dim:
        sig = F.pad(sig, (0, dim - sig.numel())) if sig.numel() < dim else sig[:dim]
    q = _whiten(sig, bank)
    exemplars = bank.get("exemplars")
    ex_labs = bank.get("exemplar_labels")
    if exemplars is not None and ex_labs is not None and exemplars.numel() > 0:
        refs = _whiten(exemplars, bank)
        ex_dists = torch.linalg.vector_norm(refs - q.unsqueeze(0), dim=1)
        dists = torch.full((n_classes,), 1e6, dtype=torch.float32)
        dists.scatter_reduce_(0, ex_labs, ex_dists, reduce="amin", include_self=True)
    else:
        refs = _whiten(mats, bank) if bank.get("feat_mean") is not None else mats
        dists = torch.linalg.vector_norm(refs - q.unsqueeze(0), dim=1)
    if counts is not None:
        dists = dists.clone()
        dists[counts <= 0] = 1e6
    if letters_only:
        dists = dists.clone()
        for i, lab in enumerate(labels):
            if str(lab).startswith("NUM_"):
                dists[i] = 1e6
    sims = 1.0 / (1.0 + dists)
    conf, pred_idx = torch.max(sims, dim=0)
    pred_idx = int(pred_idx.item())
    topk = torch.topk(sims, k=min(top_k, sims.numel()))
    alternatives = [
        {"character": labels[int(i)], "similarity": float(p)}
        for p, i in zip(topk.values.tolist(), topk.indices.tolist())
    ]
    second = alternatives[1]["similarity"] if len(alternatives) > 1 else 0.0
    return {
        "character": labels[pred_idx],
        "index": pred_idx,
        "similarity": float(conf.item()),
        "margin": float(conf.item() - second),
        "top_k": alternatives,
    }


def _same_glyph_type(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return True
    return _is_numeral_label(a) == _is_numeral_label(b)


def fuse_trust(
    cnn_pred: dict,
    proto_pred: Optional[dict],
    *,
    min_cnn_conf: float = TRUST_CFG["min_cnn_conf"],
    min_proto_sim: float = TRUST_CFG["min_proto_sim"],
    min_margin: float = TRUST_CFG["min_margin"],
    shape_pred: Optional[dict] = None,
    prefer_letters: bool = True,
    letters_only: bool = False,
) -> dict:
    """Combine CNN, prototype gallery, and general stroke-shape matching."""
    cnn_char = cnn_pred["character"]
    cnn_conf = float(cnn_pred["confidence"])
    alts = cnn_pred.get("top_k") or []
    second_cnn = float(alts[1]["confidence"]) if len(alts) > 1 else 0.0
    cnn_margin = cnn_conf - second_cnn

    shape_char = shape_pred["character"] if shape_pred else None
    shape_sim = float(shape_pred["similarity"]) if shape_pred else 0.0
    shape_margin = float(shape_pred["margin"]) if shape_pred else 0.0

    if letters_only:
        if _is_numeral_label(cnn_char):
            for alt in alts:
                ch = alt.get("character")
                if ch and not _is_numeral_label(str(ch)):
                    cnn_char = ch
                    cnn_conf = float(alt.get("confidence") or 0.0)
                    break
        if _is_numeral_label(shape_char):
            shape_char = None
            shape_sim = 0.0
            shape_margin = 0.0
        if proto_pred is not None and _is_numeral_label(proto_pred.get("character")):
            proto_pred = None

    if proto_pred is None and shape_pred is None:
        trusted = cnn_conf >= 0.55 and cnn_margin >= min_margin
        return {
            "character": cnn_char if trusted else None,
            "trusted": trusted,
            "trust": cnn_conf * (0.5 + 0.5 * min(max(cnn_margin, 0.0) / 0.2, 1.0)),
            "reason": "cnn_only",
            "cnn": cnn_pred,
            "prototype": None,
            "shape": None,
        }

    proto_char = proto_pred["character"] if proto_pred else None
    proto_sim = float(proto_pred["similarity"]) if proto_pred else 0.0
    proto_margin = float(proto_pred["margin"]) if proto_pred else 0.0

    type_scale = {"cnn": 1.0, "proto": 1.0, "shape": 1.0}
    if letters_only:
        if _is_numeral_label(proto_char):
            type_scale["proto"] = 0.0
        if _is_numeral_label(shape_char):
            type_scale["shape"] = 0.0
    elif prefer_letters and not _is_numeral_label(cnn_char):
        if _is_numeral_label(proto_char):
            type_scale["proto"] = 0.35
        if _is_numeral_label(shape_char):
            type_scale["shape"] = 0.35

    votes: Dict[str, float] = {}
    shape_weight = 0.45 * max(shape_sim, 0.0) * type_scale["shape"] if shape_char else 0.0
    if shape_margin < 0.05:
        shape_weight *= 0.25
    for lab, w in (
        (cnn_char, 0.45 * cnn_conf * type_scale["cnn"]),
        (proto_char, 0.40 * max(proto_sim, 0.0) * type_scale["proto"] if proto_char else 0.0),
        (shape_char, shape_weight),
    ):
        if not lab:
            continue
        if letters_only and _is_numeral_label(str(lab)):
            continue
        votes[lab] = votes.get(lab, 0.0) + w

    if not votes:
        return {
            "character": None,
            "trusted": False,
            "trust": 0.0,
            "reason": "no_letter_votes",
            "cnn": cnn_pred,
            "prototype": proto_pred,
            "shape": shape_pred,
        }

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    chosen = ranked[0][0]
    trust = float(ranked[0][1])

    if not _same_glyph_type(cnn_char, chosen):
        both_agree = (
            proto_char == chosen
            and shape_char == chosen
            and proto_sim >= 0.78
            and shape_sim >= 0.28
            and shape_margin >= 0.08
        )
        if cnn_conf >= 0.22 and not both_agree:
            chosen = cnn_char
            trust = 0.45 * cnn_conf + 0.15
    if (
        proto_char
        and _same_glyph_type(cnn_char, proto_char)
        and proto_sim >= 0.68
        and cnn_conf < 0.40
        and (proto_margin >= 0.02 or cnn_conf < 0.30)
    ):
        chosen = proto_char
        trust = 0.65 * proto_sim + 0.15
    if (
        shape_char
        and _same_glyph_type(cnn_char, shape_char)
        and cnn_conf < 0.40
        and shape_margin >= 0.08
        and shape_sim >= 0.22
        and (proto_char is None or proto_sim < 0.82 or proto_char == shape_char)
    ):
        if not (
            proto_char
            and proto_char != shape_char
            and proto_sim >= 0.80
            and proto_margin >= 0.08
            and _same_glyph_type(cnn_char, proto_char)
        ):
            chosen = shape_char
            trust = 0.55 * shape_sim + 0.25 * max(proto_sim, 0.0) + 0.15

    agree_proto = proto_char == chosen
    agree_shape = shape_char == chosen
    agree_cnn = cnn_char == chosen

    if agree_proto and agree_cnn and proto_sim >= min_proto_sim and (
        cnn_conf >= min_cnn_conf or proto_sim >= 0.70
    ):
        trusted = True
        reason = "cnn_proto_agree"
    elif (
        agree_proto
        and agree_shape
        and _same_glyph_type(cnn_char, chosen)
        and proto_sim >= 0.55
        and (
            shape_sim >= 0.40
            or (shape_sim >= 0.30 and shape_margin >= 0.10)
        )
    ):
        trusted = True
        reason = "proto_shape_agree"
        trust = 0.50 * proto_sim + 0.35 * shape_sim + 0.10
    elif (
        agree_shape
        and _same_glyph_type(cnn_char, chosen)
        and shape_margin >= 0.08
        and shape_sim >= 0.22
        and cnn_conf < 0.40
    ):
        trusted = True
        reason = "shape_override"
        trust = 0.55 * max(shape_sim, 0.25) + 0.25 * shape_margin + 0.15
    elif (
        agree_proto
        and _same_glyph_type(cnn_char, chosen)
        and proto_sim >= 0.65
        and (proto_margin >= 0.015 or cnn_conf < 0.40 or agree_shape)
    ):
        trusted = True
        reason = "prototype_override"
        trust = 0.65 * proto_sim + 0.15 * max(proto_margin, 0.0) + 0.10
        if agree_shape:
            trust = min(0.95, trust + 0.08)
    elif agree_shape and shape_sim >= 0.50 and (agree_cnn or shape_margin >= 0.12):
        if _same_glyph_type(cnn_char, chosen) or agree_cnn:
            trusted = True
            reason = "shape_structure"
            trust = 0.50 * shape_sim + 0.25 * max(proto_sim, cnn_conf) + 0.15
        else:
            trusted = False
            reason = "type_mismatch_reject"
            chosen = None
            trust *= 0.4
    else:
        if letters_only and cnn_conf >= 0.50 and cnn_margin >= 0.12:
            chosen = cnn_char
            trusted = True
            reason = "cnn_letters_only"
            trust = 0.55 * cnn_conf + 0.25
        elif not _same_glyph_type(cnn_char, chosen) and agree_cnn and cnn_conf >= 0.20:
            chosen = cnn_char
            trusted = True
            reason = "prefer_cnn_letter_number"
            trust = 0.55 * max(cnn_conf, 0.25) + 0.25
        elif agree_cnn and cnn_conf >= 0.45 and cnn_margin >= 0.08:
            trusted = True
            reason = "cnn_confident"
            trust = 0.55 * cnn_conf + 0.2
            chosen = cnn_char
        elif letters_only and cnn_conf >= 0.40 and cnn_margin >= 0.08:
            chosen = cnn_char
            trusted = True
            reason = "cnn_letters_only_soft"
            trust = 0.50 * cnn_conf + 0.20
        else:
            trusted = False
            reason = "disagree_reject"
            trust *= 0.5
            chosen = None

    return {
        "character": chosen if trusted else None,
        "trusted": trusted,
        "trust": float(min(max(trust, 0.0), 1.0)),
        "reason": reason,
        "cnn": cnn_pred,
        "prototype": proto_pred,
        "shape": shape_pred,
        "agreed": agree_cnn and (agree_proto or agree_shape),
    }


@torch.no_grad()
def disambiguate_lookalikes(
    model: nn.Module,
    tensor: torch.Tensor,
    cnn_pred: dict,
    index_to_char: Sequence[str],
    device: torch.device,
    bank: dict,
    *,
    shape_pred: Optional[dict] = None,
    margin_threshold: float = TRUST_CFG["lookalike_margin_threshold"],
) -> dict:
    top = cnn_pred.get("top_k") or []
    if len(top) < 1:
        return {"applied": False, "prediction": cnn_pred}
    a = top[0]["character"]
    b = top[1]["character"] if len(top) > 1 else None
    mapping = confusable_map(list(index_to_char))
    hard_partners = hard_lookalike_partners(a)
    margin = (
        float(top[0]["confidence"] - top[1]["confidence"]) if len(top) > 1 else 1.0
    )
    soft = float(top[0]["confidence"]) < 0.40
    hard_case = bool(hard_partners)
    if not soft and not hard_case and (
        b is None
        or not are_confusable(a, b, mapping)
        or margin >= margin_threshold
    ):
        return {
            "applied": False,
            "prediction": cnn_pred,
            "lookalikes": [a] + ([b] if b else []),
            "margin": margin,
        }
    if soft and not hard_case and not (
        (b is not None and are_confusable(a, b, mapping))
        or any(
            are_confusable(a, t["character"], mapping)
            or (b is not None and are_confusable(b, t["character"], mapping))
            for t in top[2:4]
        )
    ):
        return {
            "applied": False,
            "prediction": cnn_pred,
            "lookalikes": [a] + ([b] if b else []),
            "margin": margin,
        }

    prototypes = bank["prototypes"]
    labels = bank["index_to_char"]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    candidates = [a]
    if b is not None:
        candidates.append(b)
    if soft:
        candidates.extend(t["character"] for t in top[:4])
    candidates = sorted(
        set(candidates)
        | mapping.get(a, set())
        | (mapping.get(b, set()) if b else set())
        | hard_partners
    )
    if shape_pred and shape_pred.get("character"):
        candidates = sorted(set(candidates) | {shape_pred["character"]})

    shape_bonus = {}
    if shape_pred:
        for alt in shape_pred.get("top_k") or []:
            shape_bonus[alt["character"]] = float(alt["similarity"])
        shape_bonus[shape_pred["character"]] = float(shape_pred["similarity"])

    feats = F.normalize(model.extract_features(tensor.to(device)), p=2, dim=-1).cpu()
    scores = []
    for lab in candidates:
        if lab not in label_to_idx:
            continue
        idx = label_to_idx[lab]
        counts = bank.get("counts")
        if counts is not None and float(counts[idx]) <= 0:
            continue
        proto_sim = float(torch.dot(feats[0], prototypes[idx]).item())
        s_sim = shape_bonus.get(lab, 0.0)
        combined = 0.55 * proto_sim + 0.45 * s_sim if shape_bonus else proto_sim
        if lab in hard_partners and proto_sim >= 0.62:
            combined += 0.04
        scores.append((lab, combined, proto_sim, s_sim))
    if not scores:
        return {"applied": False, "prediction": cnn_pred}
    scores.sort(key=lambda x: x[1], reverse=True)
    best_lab, best_combined, _best_proto, _best_shape = scores[0]

    if hard_case and best_lab != a:
        cnn_conf = float(cnn_pred.get("confidence") or 0.0)
        a_score = next((sc for lab, sc, _, _ in scores if lab == a), 0.0)
        if cnn_conf >= 0.75 and (best_combined - a_score) < 0.03:
            return {
                "applied": False,
                "prediction": cnn_pred,
                "lookalikes": sorted({a} | hard_partners),
                "margin": margin,
                "reason": "hard_pair_keep_cnn",
            }

    name, codepoint = _enrich_label(best_lab)
    updated = {
        **cnn_pred,
        "character": best_lab,
        "name": name,
        "codepoint": codepoint,
        "confidence": float(cnn_pred["confidence"]),
        "source": "lookalike_prototype",
        "index": label_to_idx.get(best_lab, cnn_pred.get("index")),
        "lookalike_score": float(best_combined),
    }
    return {"applied": True, "prediction": updated, "lookalikes": [a] + ([b] if b else []), "margin": margin}


class MusnadPredictor:
    """Load once, predict many times. CPU-safe by default."""

    def __init__(
        self,
        *,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        prototypes: Path = PROTOTYPES_PATH,
        shape_bank: Path = SHAPE_BANK_PATH,
        force_cpu: bool = False,
    ) -> None:
        self.device = resolve_device(force_cpu=force_cpu)
        self.model, self.ckpt = load_model(checkpoint, self.device)
        self.index_to_char = list(self.ckpt.get("index_to_char", INDEX_TO_CHAR))
        self.prototypes = load_prototypes(prototypes)
        self.shape_bank = load_shape_bank(shape_bank)

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | Image.Image,
        *,
        top_k: int = 3,
        already_normalized: bool = False,
        compare_preprocess: bool = True,
        use_prototypes: bool = TRUST_CFG["use_prototypes"],
        letters_only: bool = False,
    ) -> dict:
        if isinstance(image, (str, Path)):
            pil_raw = load_external_image(Path(image))
        else:
            pil_raw = image
        pil, skew_deg = deskew_image(pil_raw)
        branches = list(INFERENCE_BRANCHES)
        if not compare_preprocess:
            branches = [b for b in branches if b.key == "original"]

        attempts: List[dict] = []
        best_view_tensor = None
        for branch in branches:
            if branch.key == "original":
                view = prepare_original_view(pil, already_normalized=already_normalized)
            else:
                view = branch.prepare(pil)
            tensor = image_to_tensor(view)
            pred = infer_tensor(
                self.model,
                tensor,
                self.index_to_char,
                self.device,
                top_k,
                letters_only=letters_only,
            )
            pred["source"] = branch.label
            pred["branch"] = branch.key
            attempts.append(pred)
            if best_view_tensor is None or pred["confidence"] >= max(
                a["confidence"] for a in attempts
            ):
                best_view_tensor = tensor

        best = select_best_attempt(attempts)
        best_tensor = best_view_tensor
        best_pil = pil
        if (
            float(best.get("confidence", 0.0)) < TTA_TRIGGER
            and not letters_only
            and _PRE_CFG["rotation_tta"]["enabled"]
        ):
            for ang in TTA_ANGLES_DEG:
                rotated, applied = deskew_image(pil, angle=ang, max_abs=25.0)
                if abs(applied) < 1.0:
                    continue
                view = prepare_original_view(rotated, already_normalized=already_normalized)
                tensor = image_to_tensor(view)
                pred = infer_tensor(
                    self.model,
                    tensor,
                    self.index_to_char,
                    self.device,
                    top_k,
                    letters_only=letters_only,
                )
                pred["source"] = f"original@{applied:.0f}deg"
                pred["branch"] = "original"
                if pred["confidence"] > best["confidence"] + float(
                    _PRE_CFG["rotation_tta"]["min_improvement"]
                ):
                    best = pred
                    best_tensor = tensor
                    best_pil = rotated

        proto_pred = None
        shape_pred = None
        if use_prototypes and self.prototypes is not None:
            feats = self.model.extract_features(best_tensor.to(self.device))
            proto_pred = match_prototypes(
                feats[0], self.prototypes, top_k=top_k, letters_only=letters_only
            )
            name, codepoint = _enrich_label(proto_pred["character"])
            proto_pred["name"] = name
            proto_pred["codepoint"] = codepoint

        if use_prototypes and TRUST_CFG["use_shape_bank"] and self.shape_bank is not None:
            sig = extract_shape_signature(best_pil)
            shape_pred = match_shape(
                sig, self.shape_bank, top_k=top_k, letters_only=letters_only
            )
            name, codepoint = _enrich_label(shape_pred["character"])
            shape_pred["name"] = name
            shape_pred["codepoint"] = codepoint

        lookalike = {"applied": False, "prediction": best}
        if use_prototypes and self.prototypes is not None:
            lookalike = disambiguate_lookalikes(
                self.model,
                best_tensor,
                best,
                self.index_to_char,
                self.device,
                self.prototypes,
                shape_pred=shape_pred,
            )
            if lookalike.get("applied") and not (
                letters_only
                and _is_numeral_label(str(lookalike["prediction"].get("character")))
            ):
                best = lookalike["prediction"]

        trust = None
        if use_prototypes and (proto_pred is not None or shape_pred is not None):
            trust = fuse_trust(
                best,
                proto_pred,
                shape_pred=shape_pred,
                prefer_letters=True,
                letters_only=letters_only,
            )
            if trust["trusted"] and trust["character"] is not None:
                if trust["character"] != best["character"]:
                    src_tag = trust["reason"]
                    if "shape" in src_tag:
                        src_tag = "shape"
                    elif "proto" in src_tag:
                        src_tag = "prototype"
                    else:
                        src_tag = trust["reason"]
                    name, codepoint = _enrich_label(trust["character"])
                    best = {
                        **best,
                        "character": trust["character"],
                        "name": name,
                        "codepoint": codepoint,
                        "confidence": float(trust["trust"]),
                        "source": src_tag,
                    }
            elif trust is not None and not trust["trusted"]:
                if letters_only and float(best.get("confidence") or 0.0) >= 0.35:
                    best = {
                        **best,
                        "source": f"low-trust-keep ({trust['reason']})",
                        "confidence": float(trust["trust"]),
                    }
                else:
                    best = {
                        **best,
                        "character": "?",
                        "name": "UNKNOWN",
                        "codepoint": None,
                        "confidence": float(trust["trust"]),
                        "source": f"low-trust ({trust['reason']})",
                    }

        return {
            "character": best["character"],
            "name": best.get("name"),
            "codepoint": best.get("codepoint"),
            "confidence": best["confidence"],
            "source": best.get("source"),
            "top_k": best.get("top_k"),
            "attempts": attempts,
            "prototype": proto_pred,
            "shape": shape_pred,
            "trust": trust,
            "lookalike": lookalike,
            "skew_deg": float(skew_deg),
            "device": str(self.device),
        }


def predict_image(
    image_path: str | Path,
    *,
    force_cpu: bool = False,
    checkpoint=None,
    device=None,
    save_debug=None,
    **kwargs,
) -> dict:
    """One-shot prediction helper (legacy kwargs from stone detect merge)."""
    if device is not None:
        force_cpu = str(device) == "cpu"
    kwargs.pop("checkpoint", None)
    kwargs.pop("save_debug", None)
    predictor = MusnadPredictor(force_cpu=force_cpu)
    return predictor.predict(image_path, **kwargs)
