# Musnad OCR — Production Model Package

Self-contained **full OCR pipeline** for Musnad (Ancient South Arabian) script,
plus single-glyph classification. Includes the exact production weights and the
inference code needed to reproduce the same predictions as the training project.

**No server or API is included.** Integrate this package into your own backend.

## Package layout

```
musnad-ocr-model/
├── model/
│   ├── musnad_final.pth        # Production CNN checkpoint (39 classes, paper fine-tuned)
│   ├── class_prototypes.pt     # Prototype gallery (stone / mixed domain; optional for paper)
│   └── shape_bank.pt           # Stroke-shape similarity bank
├── config/
│   ├── labels.json             # Label map + metadata (letters + numerals)
│   ├── preprocessing.json      # Single-glyph preprocess / TTA / trust params
│   ├── paper_ocr.json          # Full paper-line OCR pipeline config
│   └── lookalikes.json         # Confusable letter pairs
├── inference/
│   ├── predict.py              # Single-glyph classifier
│   ├── preprocessing.py        # Image preprocessing (original + stone views)
│   ├── layout.py               # RTL ordering + word-separator rules
│   ├── paper_detect.py         # Paper glyph + bar detection (projection-based)
│   └── paper_ocr.py            # Full pipeline: detect → classify → text
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- CPU or NVIDIA GPU (CUDA optional — runs on CPU when GPU is unavailable)

```bash
pip install -r requirements.txt
```

## 1) Full paper / line OCR (recommended)

Use this for a manuscript line or page (multiple glyphs):

```python
from inference.paper_ocr import MusnadOCR

ocr = MusnadOCR(force_cpu=True)  # False = use CUDA when available
result = ocr.recognize("path/to/paper_line.png", out_dir="out/run1")

print(result["text"])          # full page text (RTL logical order)
print(result["overlay_path"])  # annotated image with boxes + labels + confidence
print(result["n_lines"])
print(result["n_glyphs"])
for line in result["lines"]:
    print(line["text"], line["words"])
```

The annotated visualization (`overlay.png`) is written by default (`save_overlay=True`):

- Bounding box around every detected glyph  
- Predicted Latin name (e.g. `NUN 98%`) or `|` for word separators  
- Confidence percentage on each label  
- Color coding: green = letter, orange = word bar, red = unknown  

Use `result["overlay_path"]` in the mobile app, or the in-memory `result["overlay"]` PIL image.
`result["glyphs"]` also lists every box with `character`, `name`, `confidence`, and `box`.

One-shot helper:

```python
from inference.paper_ocr import recognize_paper

result = recognize_paper("path/to/paper_line.png", force_cpu=True, out_dir="out/run1")
```

### What the full pipeline does

1. **Detect** dark ink glyphs on light paper (projection segmentation + upscale for JPEG pages)  
2. **Detect word bars** (colored `|`) or insert separators from **letter-gap** statistics  
3. **Cluster lines** top → bottom  
4. **Order each line** right → left (RTL **logical** storage order — UI `dir=rtl` displays correctly)  
5. **Crop** each glyph with neighbor-clamped padding (no adjacent-letter contamination)  
6. **Classify** each crop with paper-font fine-tuned `musnad_final.pth` (`letters_only=True`, no stone prototypes)  
7. **Apply** BETH↔GIMEL shape fix for digital font lookalikes  
8. **Split words** on vertical-bar / `NUM_1` separators  
9. **Write annotated overlay** (`overlay.png`) with boxes, names, and confidence  

Render `result["text"]` in the UI with `dir="rtl"`. Join words with spaces in display (not `|`).

### Result shape (summary)

```json
{
  "ok": true,
  "mode": "paper_line",
  "direction": "rtl",
  "text_direction": "rtl",
  "glyph_order": "rtl_logical",
  "text": "…",
  "n_lines": 1,
  "n_glyphs": 12,
  "lines": [
    {
      "line": 0,
      "text": "…",
      "words": ["…", "…"],
      "glyphs": [{"character": "…", "name": "…", "box": [x0,y0,x1,y1], "...": "..."}]
    }
  ]
}
```

## 2) Single-glyph classification

Use this only when you already have a cropped letter/number image:

```python
from inference.predict import MusnadPredictor

predictor = MusnadPredictor(force_cpu=True)
result = predictor.predict("path/to/glyph.png")

print(result["character"])
print(result["name"])
print(result["confidence"])
```

For paper crops, pass `use_prototypes=False` and `letters_only=True` to match line OCR:

```python
result = predictor.predict(
    "path/to/glyph.png",
    use_prototypes=False,
    letters_only=True,
    compare_preprocess=False,
)
```

## Model details

| Item | Value |
|------|-------|
| Input (classifier) | `1 × 128 × 128` grayscale crop |
| Classes | 39 (29 letters + 10 numeral forms) |
| Architecture | MusnadCNN (Conv + glyph attention + GeM) |
| Checkpoint | `model/musnad_final.pth` |
| Full OCR domain | Clean paper / digital Musnad font (Segoe Historic) |
| Paper fine-tune | Synthetic paper-font dataset, LR 1e-4, 10 epochs |

**Do not retrain or modify the checkpoint** unless you intentionally want a new model version.

## Integration notes (Node / other backends)

1. Copy this entire `musnad-ocr-model/` folder into your monorepo.  
2. `pip install -r requirements.txt`  
3. Call `MusnadOCR.recognize(...)` from your own Python entry script.  
4. Use `force_cpu=True` on machines without NVIDIA GPU.  

Keep these files together — removing any of them changes predictions:

- `model/musnad_final.pth`
- `config/labels.json`
- `config/preprocessing.json`
- `config/paper_ocr.json`
- all of `inference/`

Optional (stone / mixed domain single-glyph only):

- `model/class_prototypes.pt`
- `model/shape_bank.pt`

## Scope

| Mode | Included |
|------|----------|
| Single glyph classify | Yes |
| Paper line / page OCR | Yes (detect → RTL → words) |
| Stone inscription OCR | Not in this detector (classifier still supports stone crops) |
| Server / API | No — build your own |

## Version

**v0.3.3** (2026-07-28) — stronger BETH vs TETH on dense pages.

Re-sync after training / pipeline changes (from `musnad_ocr/`):

```bash
python -m src.sync_package
```

### Changelog

**v0.3.3**
- Fix `𐩨` misread as `𐩷`: empty-interior / weak-stem BETH wins even when a false bottom bar appears on crowded scans
- TETH only when center vertical stem is strong and interior is filled

**v0.3.2**
- Paper geometry overrides for `𐩥`↔`𐩲` (WAW/AYN stem-in-circle)
- Paper geometry overrides for `𐩷`↔`𐩳`↔`𐩨` (TETH/DHADHE/BETH bar layout)
- Extra hard lookalike pairs in `lookalikes.json`

**v0.3.1**
- Full package sync: weights, labels, lookalikes, layout, paper_detect, paper_ocr config
- Added `VERSION.json` with SHA-256 hashes of model files
- Lookalikes config includes shape groups + hard pairs from `confusable.py`

**v0.3**
- Fine-tuned `musnad_final.pth` on Segoe Historic / digital paper font
- Projection detection, gap word breaks, neighbor-clamped crops, RTL logical order
- BETH↔GIMEL shape fix; prototypes disabled on paper

**v0.2**
- Initial packaged paper line OCR pipeline

Exported from production checkpoint `musnad_final.pth` with the production paper OCR pipeline.
