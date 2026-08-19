# Musnad OCR — Production Model Package

Self-contained **production weights + inference code** for Musnad (Ancient South Arabian) script.

| What you get | Where |
|--------------|--------|
| Trained classifier (`musnad_final.pth`) | `model/` |
| Prototype + shape banks (stone classify) | `model/` |
| Stone detect weights (letter boxes) | `model/` |
| Paper line OCR (detect → classify → RTL text + overlay) | `inference/paper_ocr.py` |
| Stone inscription OCR (multi-line + overlay) | `inference/stone_ocr.py` (`MusnadStoneOCR`) |
| Single-glyph classify (paper or stone crop) | `inference/predict.py` |
| Web UI (paper + stone) | **`musnad_ocr/` project** |

**No server or API is included.** Call the Python API from your own backend.

## Quick start

```bash
cd musnad-ocr-model
pip install -r requirements.txt
```

**Paper line:**

```bash
python -c "from inference.paper_ocr import MusnadOCR; r=MusnadOCR(force_cpu=True).recognize('path/to/line.png', out_dir='outputs/paper'); print(r['text'], r.get('overlay_path'))"
```

**Stone inscription (full photo — use this, not paper OCR):**

```bash
python -c "from inference.stone_ocr import MusnadStoneOCR; r=MusnadStoneOCR(force_cpu=True).recognize('path/to/stone.png', out_dir='outputs/stone'); print(r['n_lines'], r['text'], r.get('overlay_path'))"
```

**Single stone letter crop:**

```bash
python -c "from inference.predict import MusnadPredictor; p=MusnadPredictor(force_cpu=True); r=p.predict('path/to/crop.png', compare_preprocess=True, use_prototypes=True); print(r['name'], r['confidence'])"
```

> **Important:** Do **not** use `MusnadOCR` on stone photos — paper projection typically finds only one line on weathered stone. Use `MusnadStoneOCR` instead.

Verify weights after copy or sync: open `VERSION.json` (version + SHA-256 per file).

## Package layout

```
musnad-ocr-model/
├── model/
│   ├── musnad_final.pth
│   ├── class_prototypes.pt
│   ├── shape_bank.pt
│   └── letter_boundary_v2.pth   # stone letter cuts (Segmentation v2)
├── config/
│   ├── labels.json
│   ├── preprocessing.json
│   ├── paper_ocr.json
│   ├── stone_ocr.json
│   └── lookalikes.json
├── inference/
│   ├── predict.py
│   ├── preprocessing.py
│   ├── layout.py
│   ├── paper_detect.py          # includes draw_annotations()
│   ├── paper_ocr.py
│   ├── stone_ocr.py             # MusnadStoneOCR
│   ├── letter_detector.py
│   ├── letter_boundary_net.py
│   ├── segment_v2.py
│   ├── inscription_region.py
│   ├── stone_glyph_segmentation.py
│   ├── stone_enhancement.py
│   ├── empty_segment_filter.py
│   └── device.py
├── VERSION.json
├── requirements.txt
└── README.md
```

## Trained models (`model/`)

| File | Purpose |
|------|---------|
| `musnad_final.pth` | Main CNN — 39 classes. **Required for all modes.** |
| `class_prototypes.pt` | Prototype gallery (stone classify re-ranking) |
| `shape_bank.pt` | Stroke-shape similarity (with prototypes) |
| `letter_boundary_v2.pth` | Stone letter cuts (boundary net, Segmentation v2) |

- **Paper OCR:** `musnad_final.pth` only  
- **Stone inscription OCR:** `musnad_final.pth` + `letter_boundary_v2.pth` + prototypes + shape bank  
- **Stone single-glyph crop:** `musnad_final.pth` + prototypes + shape bank  

## Requirements

- Python 3.10+
- CPU or NVIDIA GPU (CUDA optional)

```bash
pip install -r requirements.txt
```

## 1) Paper / manuscript line OCR

```python
from inference.paper_ocr import MusnadOCR

ocr = MusnadOCR(force_cpu=True)
result = ocr.recognize("path/to/paper_line.png", out_dir="out/paper")

print(result["text"])
print(result["overlay_path"])   # overlay.png — full image, boxes + labels
print(result["n_lines"], result["n_glyphs"])
```

Annotated overlay (`save_overlay=True` by default):

- Green box + Latin name + confidence (e.g. `NUN 98%`)
- Orange = word separator `|`
- Red = unknown / low trust
- Musnad character drawn inside each box when the font is available

Also available: `result["overlay"]` (PIL), `result["glyphs"]` (all boxes with coordinates).

One-shot: `from inference.paper_ocr import recognize_paper`

### Paper pipeline

1. Detect ink glyphs (projection + upscale)
2. Word bars or gap-based separators
3. Cluster lines top → bottom, order RTL
4. Classify with paper fine-tuned CNN (`letters_only=True`, no stone prototypes)
5. BETH↔GIMEL shape fix
6. Write `overlay.png`

## 2) Single-glyph classification

For an **already cropped** letter or numeral:

```python
from inference.predict import MusnadPredictor

predictor = MusnadPredictor(force_cpu=True)

# Paper crop
predictor.predict("crop.png", use_prototypes=False, letters_only=True, compare_preprocess=False)

# Stone carving crop
predictor.predict("crop.png", compare_preprocess=True, use_prototypes=True, letters_only=False)
```

## 3) Stone inscription OCR

Full photo → line banding → letter detect → classify → RTL text + **full-image overlay**.

```python
from inference.stone_ocr import MusnadStoneOCR

ocr = MusnadStoneOCR(force_cpu=True)
result = ocr.recognize("path/to/stone_inscription.jpg", out_dir="out/stone")

print(result["n_lines"], result["n_glyphs"])
print(result["text"])
print(result["overlay_path"])   # full scanned image with all boxes + identify labels
overlay_pil = result["overlay"]  # PIL image for your UI
```

Options:

- `save_overlay=True` (default) — writes `out/stone/overlay.png` and `result.json`
- `save_overlay=False` — skip annotated image

One-shot: `from inference.stone_ocr import recognize_stone_image`

### Stone pipeline

1. Line banding (`stone_glyph_segmentation`)
2. Letter cuts per line (`segment_v2` + `letter_boundary_v2.pth`)
3. Classify each frozen crop (stone preprocess + prototypes)
4. RTL word split
5. Draw annotations on **full original image** (`draw_annotations`)

Browser UI: `python -m src.webapp` in `musnad_ocr/` (Stone mode).

## 4) Sync from training project

After changes in `musnad_ocr/`:

```bash
cd musnad_ocr
python -m src.sync_package
```

Copies weights, configs, and inference modules into this folder. Check `VERSION.json` after sync.

## 5) Testing

From `musnad-ocr-model/`:

```bash
# Paper
python -c "
from inference.paper_ocr import MusnadOCR
r = MusnadOCR(force_cpu=True).recognize('../musnad_ocr/test_images/test-1.webp', out_dir='outputs/paper_test')
print('lines', r['n_lines'], 'glyphs', r['n_glyphs'], 'overlay', r.get('overlay_path'))
"

# Stone (multi-line)
python -c "
from inference.stone_ocr import MusnadStoneOCR
r = MusnadStoneOCR(force_cpu=True).recognize('../musnad_ocr/test_images/test-8.png', out_dir='outputs/stone_test')
print('lines', r['n_lines'], 'glyphs', r['n_glyphs'], 'overlay', r.get('overlay_path'))
"

# Version + weight hashes
python -c "import json; m=json.load(open('VERSION.json')); print(m['version'], list(m['files']))"
```

Expected on `test-8.png`: **2 lines**, annotated `overlay.png` at full image size.

## Integration (other apps / backends)

1. Copy this entire `musnad-ocr-model/` folder into your project.
2. `pip install -r requirements.txt`
3. **Paper:** `MusnadOCR().recognize(...)`
4. **Stone photo:** `MusnadStoneOCR().recognize(...)` — not `MusnadOCR`
5. Use `result["overlay_path"]` or `result["overlay"]` for the annotated preview.
6. `force_cpu=True` when no NVIDIA GPU.

Required together:

- `model/musnad_final.pth`
- `config/labels.json`, `config/preprocessing.json`
- all of `inference/`

Stone inscription also needs `letter_boundary_v2.pth` and prototype/shape files.

## Scope

| Mode | Package |
|------|---------|
| Paper line OCR + overlay | Yes (`MusnadOCR`) |
| Stone inscription OCR + overlay | Yes (`MusnadStoneOCR`) |
| Single-glyph classify | Yes (`MusnadPredictor`) |
| Web UI | No — use `musnad_ocr/` |
| HTTP API | No — build your own |

## Version

**v0.4.2** (2026-08-10)

### Changelog

**v0.4.2**
- Stone OCR writes **full-image `overlay.png`** with detect + identify labels (same style as paper)
- `result["overlay"]`, `result["glyphs"]`, `save_overlay` flag on `MusnadStoneOCR`
- `draw_annotations` synced into `paper_detect.py` (shared by paper + stone)

**v0.4.1**
- Full stone pipeline packaged: `MusnadStoneOCR` (line banding, letter detect, stone classify)
- Sync detector modules + weights; **do not use `MusnadOCR` on stone**

**v0.4.0**
- Stone preprocess hardening (zig-zag letters); bundle stone detect weights

**v0.3.x**
- Paper fine-tune, lookalike fixes, BETH/TETH geometry overrides, `VERSION.json` hashes

**v0.2**
- Initial packaged paper line OCR
