"""Illustration overlay/bubble translator + typesetter (v0.6).

Does not overwrite sources. Prefer --vision (Gemini 3.7 Flash).
"""
from __future__ import annotations

import ast
import asyncio
import json
from difflib import SequenceMatcher
import os
import re
import shutil
import sys
import time
import types
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
except Exception:
    pass
SRC = Path(os.environ.get("LETTER_SRC", str(HERE / "input")))
LOG_DIR = Path(os.environ.get("LETTER_LOG_DIR", str(HERE / "logs")))
NAMES_PATH = Path(os.environ.get("LETTER_NAMES", str(HERE / "names.json")))
MIT_ROOT = Path(os.environ.get("MIT_ROOT", r"D:\manga-image-translator"))
MIT_PKG = MIT_ROOT / "manga_translator"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434/api/chat")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.8:27b-uncensored")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SFX_RE = re.compile(
    r"^[嗯啊唔哦呵哈呜呃嗷哟哼哒砰轰啾欸诶唔]+[?？!！.。…~～—\-♪♡♥\s]*$"
)
BURST_RE = re.compile(r"^[!！?？¡.。…~～♪♡♥\s]+$")
GOLD_PATH = Path(os.environ.get("LETTER_GOLD", str(HERE / "gold_set.json")))
ONLY = None
RETRANSLATE = False
LANG = "en"
DST = Path(os.environ.get("LETTER_DST", str(HERE / "output")))
GOLD_JOBS = None
CLEAN = None  # optional folder of unlettered plates, same filenames as lettered src
VISION = False  # cloud VL locate+OCR instead of local detector

FONT_PATHS = {
    "en": (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\arial.ttf"),
    "ja": (r"C:\Windows\Fonts\YuGothB.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msgothic.ttc"),
}

_NAME_TABLE: dict | None = None


def load_name_table() -> dict:
    global _NAME_TABLE
    if _NAME_TABLE is None:
        _NAME_TABLE = json.loads(NAMES_PATH.read_text(encoding="utf-8"))
    return _NAME_TABLE


def name_prompt_lines(lang: str) -> str:
    names = load_name_table().get("names") or []
    bits = []
    for e in names:
        zh = " / ".join(e.get("zh") or [])
        tgt = e.get(lang) or e.get("en")
        if zh and tgt:
            bits.append(f"{zh} → {tgt}")
    return "; ".join(bits[:40])


def stub_mit() -> None:
    def pkg(name: str, path: Path) -> None:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        m.__package__ = name
        sys.modules[name] = m

    sys.path.insert(0, str(MIT_ROOT))
    pkg("manga_translator", MIT_PKG)
    pkg("manga_translator.detection", MIT_PKG / "detection")
    pkg("manga_translator.inpainting", MIT_PKG / "inpainting")
    pkg("manga_translator.ocr", MIT_PKG / "ocr")


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_bgr(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(path))


def _morph_close(m: np.ndarray, kernel) -> np.ndarray:
    if m is None or m.size == 0:
        return m
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


def expand_mask(mask: np.ndarray, px: int = 8) -> np.ndarray:
    m = (mask > 127).astype(np.uint8) * 255
    m = _morph_close(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=-1)
    return filled


def _stroke_neighborhood(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    white = ((gray >= 195) & (chroma < 55)).astype(np.uint8) * 255
    dark = (gray < 48).astype(np.uint8) * 255
    return cv2.dilate(cv2.bitwise_or(white, dark), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def _sample_fill_color(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    pix = rgb[mask > 0]
    if len(pix) < 20:
        return None
    pix = np.ascontiguousarray(pix.astype(np.float32))
    if len(pix) < 40:
        return pix.mean(axis=0)
    _, _, centers = cv2.kmeans(
        pix,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers[0] if _chroma(centers[0]) >= _chroma(centers[1]) else centers[1]


def _sample_glyph_color(roi: np.ndarray) -> np.ndarray | None:
    """Fill from ink pixels only — not the white sticker cloud behind 呜呜."""
    if roi.size == 0:
        return None
    chroma = roi.max(axis=2) - roi.min(axis=2)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    glyph = ((chroma >= 45) | (gray < 50)).astype(np.uint8) * 255
    if np.count_nonzero(glyph) < 20:
        return _sample_fill_color(roi, np.ones(roi.shape[:2], np.uint8) * 255)
    return _sample_fill_color(roi, glyph)


def _chroma_ink(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = ((H <= 45) & (S >= 70) & (V >= 70)).astype(np.uint8) * 255
    # Neon overlay pink (阿穹) is high-S; magenta SFX (不行了) sits at H>=150.
    # Skin is H~130 / S~50-90 — do not treat it as ink.
    pink = (
        ((((H >= 125) | (H <= 12)) & (S >= 140) & (V >= 80))
         | ((H >= 150) & (S >= 40) & (V >= 120)))
    ).astype(np.uint8) * 255
    cyan = ((H >= 75) & (H <= 110) & (S >= 45) & (V >= 160)).astype(np.uint8) * 255
    dark = ((gray < 58) & (S < 140) & (chroma < 100)).astype(np.uint8) * 255
    ink = cv2.bitwise_or(yellow, pink)
    ink = cv2.bitwise_or(ink, cyan)
    return cv2.bitwise_or(ink, dark)


def overlay_ink(rgb: np.ndarray) -> np.ndarray:
    """Unused by the new seed-band grow; kept for debug scripts."""
    return np.zeros(rgb.shape[:2], np.uint8)


def _comp_box(stats, i) -> tuple[int, int, int, int]:
    return (
        int(stats[i, cv2.CC_STAT_LEFT]),
        int(stats[i, cv2.CC_STAT_TOP]),
        int(stats[i, cv2.CC_STAT_WIDTH]),
        int(stats[i, cv2.CC_STAT_HEIGHT]),
    )


def _h_overlap_frac(a, b) -> float:
    ax, _, aw, _ = a
    bx, _, bw, _ = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    return inter / max(min(aw, bw), 1)


def _v_gap(a, b) -> int:
    ay, ah = a[1], a[3]
    by, bh = b[1], b[3]
    if ay + ah < by:
        return by - (ay + ah)
    if by + bh < ay:
        return ay - (by + bh)
    return 0


def _h_gap(a, b) -> int:
    ax, aw = a[0], a[2]
    bx, bw = b[0], b[2]
    if ax + aw < bx:
        return bx - (ax + aw)
    if bx + bw < ax:
        return ax - (bx + bw)
    return 0


def _v_overlap_frac(a, b) -> float:
    ay, ah = a[1], a[3]
    by, bh = b[1], b[3]
    inter = max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / max(min(ah, bh), 1)


def _captionish(box, area, img_h) -> bool:
    _, _, ww, hh = box
    if area < 70 or ww < 16 or hh < 14 or hh > int(0.28 * img_h):
        return False
    ar = ww / max(hh, 1)
    if ar >= 1.15:
        return True
    if hh >= int(ww * 1.4) and hh <= int(0.62 * img_h) and ww >= 12:
        return True
    # Compact overlay 拟声 (嗯? 哎哟!) is closer to square.
    return ar >= 0.8 and area <= 8000 and hh <= int(0.16 * img_h)


def complete_caption_mask(rgb: np.ndarray, det_mask: np.ndarray) -> np.ndarray:
    """Grow each detector hit in a local band so stacked same-color lines are fully erased."""
    h, w = det_mask.shape
    seed = (det_mask > 127).astype(np.uint8) * 255
    if np.count_nonzero(seed) < 20:
        return cv2.bitwise_or(seed, independent_outlined_overlay(rgb))
    stroke = _stroke_neighborhood(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out = seed.copy()
    seed_cc = cv2.dilate(seed, np.ones((7, 7), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(seed_cc, 8)
    for i in range(1, num):
        m = ((labels == i).astype(np.uint8) * 255)
        fill = _sample_fill_color(rgb, m)
        if fill is None:
            continue
        x, y, ww, hh = _comp_box(stats, i)
        x0, x1 = max(0, x - 24), min(w, x + ww + 40)
        y0, y1 = max(0, y - int(0.2 * hh)), min(h, y + hh + int(1.6 * hh) + 16)
        roi = rgb[y0:y1, x0:x1].astype(np.float32)
        dist = np.linalg.norm(roi - fill.reshape(1, 1, 3), axis=2)
        st = stroke[y0:y1, x0:x1] > 0
        if _lum(fill) < 60:
            sim = ((gray[y0:y1, x0:x1] < 55) & st).astype(np.uint8) * 255
        else:
            sim = ((dist < 58) & st).astype(np.uint8) * 255
        sim = _morph_close(sim, cv2.getStructuringElement(cv2.MORPH_RECT, (23, 7)))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sim, 8)
        keep = np.zeros_like(sim)
        seed_local = m[y0:y1, x0:x1] > 0
        include: set[int] = set()
        for j in range(1, n2):
            if np.any((lab2 == j) & seed_local):
                include.add(j)
        changed = True
        while changed:
            changed = False
            for j in range(1, n2):
                if j in include:
                    continue
                box = _comp_box(st2, j)
                area = int(st2[j, cv2.CC_STAT_AREA])
                if not _captionish(box, area, h) and area < 200:
                    continue
                if box[3] > int(0.22 * h) or box[2] / max(box[3], 1) < 1.35:
                    continue
                for k in list(include):
                    b2 = _comp_box(st2, k)
                    stacked = _h_overlap_frac(box, b2) >= 0.28 and _v_gap(box, b2) <= int(1.85 * max(box[3], b2[3])) + 18
                    same_line = _v_overlap_frac(box, b2) >= 0.4 and _h_gap(box, b2) <= 90
                    if stacked or same_line:
                        include.add(j)
                        changed = True
                        break
        for j in include:
            keep[lab2 == j] = 255
        out[y0:y1, x0:x1] = cv2.bitwise_or(out[y0:y1, x0:x1], keep)
    out = cv2.bitwise_or(out, seed)
    out = _morph_close(out, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
    return out


def _keep_compact_overlay_ccs(ink: np.ndarray, h: int, w: int) -> np.ndarray:
    """Keep glyph-sized CCs. Filter before OR so sky/clothes do not swallow 嗯?."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    kept = np.zeros(ink.shape, np.uint8)
    for i in range(1, num):
        box = _comp_box(stats, i)
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww, hh = box[2], box[3]
        if area < 60 or ww < 12 or hh < 12:
            continue
        if hh > int(0.22 * h) or ww > int(0.22 * w) or area > 0.008 * h * w:
            continue
        ar = ww / max(hh, 1)
        if ar < 0.28 and hh / max(ww, 1) < 1.15:
            continue
        kept[labels == i] = 255
    return kept


def independent_outlined_overlay(rgb: np.ndarray) -> np.ndarray:
    """Outlined overlay the detector missed: black 嗯? / 停下 and colored 哎哟 / 不行了.

    Each ink source is CC-filtered on its own so chromatic sky does not merge
    a compact SFX into a page-sized blob that the size gate then drops.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    white = ((gray >= 195) & (chroma < 55)).astype(np.uint8) * 255
    near_white = cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    dark_w = ((gray < 52) & (hsv[:, :, 1] < 110) & (near_white > 0)).astype(np.uint8) * 255
    color_w = ((chroma >= 50) & (hsv[:, :, 2] >= 90) & (near_white > 0)).astype(np.uint8) * 255
    dark_any = ((gray < 46) & (hsv[:, :, 1] < 130) & (chroma < 95)).astype(np.uint8) * 255
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = ((H <= 45) & (S >= 70) & (V >= 70)).astype(np.uint8) * 255
    pink = (
        ((((H >= 125) | (H <= 12)) & (S >= 140) & (V >= 80))
         | ((H >= 150) & (S >= 40) & (V >= 120)))
    ).astype(np.uint8) * 255
    cyan = ((H >= 75) & (H <= 110) & (S >= 45) & (V >= 160)).astype(np.uint8) * 255
    color_any = cv2.bitwise_or(cv2.bitwise_or(yellow, pink), cyan)
    layers = [
        _morph_close(dark_w, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))),
        _morph_close(color_w, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))),
        _morph_close(dark_any, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))),
        _morph_close(color_any, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))),
    ]
    kept = np.zeros((h, w), np.uint8)
    for layer in layers:
        kept = cv2.bitwise_or(kept, _keep_compact_overlay_ccs(layer, h, w))
    return _keep_compact_overlay_ccs(kept, h, w)


def absorb_trailing_punct(rgb: np.ndarray, mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Eat leftover ?! after a caption box (often missed because they are not chromatic fill)."""
    h, w = mask.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    punct = (((gray < 50) | ((gray > 200) & (chroma < 60)))).astype(np.uint8) * 255
    out = mask.copy()
    for box in boxes:
        x, y, bw, bh = [int(v) for v in box]
        x0 = max(0, x + bw - 8)
        x1 = min(w, x + bw + int(max(bh * 0.45, 28)))
        y0, y1 = max(0, y - 8), min(h, y + bh + 8)
        if y1 <= y0 or x1 <= x0:
            continue
        extra = punct[y0:y1, x0:x1]
        extra = _morph_close(extra, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        if extra is None or extra.size == 0:
            continue
        out[y0:y1, x0:x1] = cv2.bitwise_or(out[y0:y1, x0:x1], extra)
    return out


def mask_row_bands(mask: np.ndarray, min_h: int = 12) -> list[tuple[int, int, int, int]]:
    """Split a caption mask into horizontal bands at empty-row gaps."""
    h, w = mask.shape
    rows = (mask > 127).any(axis=1)
    bands = []
    i = 0
    while i < h:
        if not rows[i]:
            i += 1
            continue
        j = i + 1
        while j < h and rows[j]:
            j += 1
        if j - i >= min_h:
            cols = np.where(mask[i:j].any(axis=0))[0]
            if cols.size:
                bands.append((int(cols.min()), i, int(cols.max() - cols.min() + 1), j - i))
        i = j
    return bands


def unmatched_mask_boxes(mask: np.ndarray, existing: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """OCR windows for mask bands the detector did not already cover (stacked leftover lines, 嗯?)."""
    extras = []
    for box in mask_row_bands(mask):
        remnant = list(box)
        for eb in existing:
            ix = max(0, min(box[0] + box[2], eb[0] + eb[2]) - max(box[0], eb[0]))
            iy = max(0, min(box[1] + box[3], eb[1] + eb[3]) - max(box[1], eb[1]))
            inter = ix * iy
            ba = max(box[2] * box[3], 1)
            ea = max(eb[2] * eb[3], 1)
            if inter / ba < 0.35 and inter / ea < 0.35:
                continue
            # Same band already OCR'd: drop. Below the OCR box: keep the leftover strip.
            below = (box[1] + box[3]) - (eb[1] + eb[3])
            if below >= 16 and _h_overlap_frac(box, eb) >= 0.25:
                ny = eb[1] + eb[3] + 4
                remnant[1] = ny
                remnant[3] = max(0, box[1] + box[3] - ny)
            else:
                remnant[3] = 0
                break
        if remnant[3] >= 14 and remnant[2] >= 16:
            extras.append((int(remnant[0]), int(remnant[1]), int(remnant[2]), int(remnant[3])))
    return extras


def extra_ink_boxes(mask: np.ndarray, existing: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Unmatched caption-like ink → extra OCR windows (missed whole lines)."""
    h, w = mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8) * 255, 8)
    extras = []
    for i in range(1, num):
        box = _comp_box(stats, i)
        area = stats[i, cv2.CC_STAT_AREA]
        if not _captionish(box, area, h) and not (area >= 120 and box[2] >= 40 and box[3] <= int(0.22 * h)):
            continue
        hit = False
        for eb in existing:
            if _h_overlap_frac(box, eb) > 0.35 and _v_gap(box, eb) <= 8:
                hit = True
                break
            ix = max(0, min(box[0] + box[2], eb[0] + eb[2]) - max(box[0], eb[0]))
            iy = max(0, min(box[1] + box[3], eb[1] + eb[3]) - max(box[1], eb[1]))
            inter = ix * iy
            if inter / max(area, 1) > 0.4 or inter / max(eb[2] * eb[3], 1) > 0.4:
                hit = True
                break
        if not hit:
            extras.append(box)
    return extras


def grow_chroma_box(rgb: np.ndarray, box, img_w: int, img_h: int):
    """Widen a split overlay glyph (阿→阿穹?!, 不→不行了) along yellow/pink/dark ink."""
    x, y, bw, bh = [int(v) for v in box[:4]]
    ink = _chroma_ink(rgb)
    roi = ink[max(0, y) : min(img_h, y + bh), max(0, x) : min(img_w, x + bw)]
    tall = bh >= int(max(bw, 1) * 1.35)
    if roi.any():
        ys, xs = np.where(roi > 0)
        ih = int(ys.max() - ys.min()) + 1
        iw = int(xs.max() - xs.min()) + 1
        if ih >= int(iw * 1.2):
            tall = True
    if bw >= 110 and bh >= 80 and not tall:
        return x, y, bw, bh
    if tall:
        x0, x1 = max(0, x - 6), min(img_w, x + bw + 6)
        extra = min(int(0.12 * img_h), max(28, bh))
        y0 = max(0, y - extra)
        y1 = min(img_h, y + bh + extra)
        rows = np.where(ink[y0:y1, x0:x1].any(axis=1))[0]
        if rows.size == 0:
            return x, y, bw, bh
        ny0 = y0 + int(rows.min())
        ny1 = y0 + int(rows.max()) + 1
        return x, ny0, bw, max(bh, ny1 - ny0)
    y0, y1 = max(0, y - 8), min(img_h, y + bh + 8)
    band = ink[y0:y1]
    cap = min(int(0.2 * img_w), max(80, int(bw * 2.2)))

    def _has(a: int, b: int) -> bool:
        a, b = max(0, a), min(img_w, b)
        return a < b and bool(band[:, a:b].any())

    x0, x1 = x, x + bw
    while x0 > 0 and (x - x0) < cap and _has(x0 - 16, x0):
        x0 -= 8
    while x1 < img_w and (x1 - x - bw) < cap and _has(x1, x1 + 16):
        x1 += 8
    sl0, sl1 = max(0, x0), min(img_w, x1)
    cols = np.where(band[:, sl0:sl1].any(axis=0))[0]
    if cols.size == 0:
        return x, y, bw, bh
    gx0 = sl0 + int(cols.min())
    gx1 = sl0 + int(cols.max()) + 1
    return gx0, y, max(bw, gx1 - gx0), bh


def split_ink_columns(rgb: np.ndarray, box) -> list[tuple[int, int, int, int]]:
    """Split a wide vertical bubble into RTL columns so OCR does not mash 评估/数据/阈值."""
    x, y, bw, bh = [int(v) for v in box[:4]]
    if bh < int(max(bw, 1) * 1.3) or bw < 48:
        return [box]
    gray = cv2.cvtColor(rgb[y : y + bh, x : x + bw], cv2.COLOR_RGB2GRAY)
    ink = (gray < 90).astype(np.uint8)
    ink = _morph_close(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))
    col = ink.any(axis=0)
    runs: list[tuple[int, int]] = []
    i = 0
    n = col.size
    while i < n:
        if not col[i]:
            i += 1
            continue
        j = i + 1
        while j < n and col[j]:
            j += 1
        if j - i >= 6:
            runs.append((i, j))
        i = j
    if len(runs) < 2 or len(runs) > 6:
        return [box]
    return [(x + a, y, b - a, bh) for a, b in runs]


def _merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int]],
    gap: int = 28,
    max_w: int | None = None,
    max_h: int | None = None,
    img_h: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Join split SFX glyphs (不/行/了, 嗯/?) into one OCR window."""
    boxes = [tuple(int(v) for v in b[:4]) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            cur = a
            for j, b in enumerate(boxes):
                if j <= i or used[j]:
                    continue
                if cur[2] * cur[3] > 6000 or b[2] * b[3] > 6000:
                    continue
                top = img_h is not None and (cur[1] < int(0.08 * img_h) or b[1] < int(0.08 * img_h))
                landscape = cur[2] > cur[3] * 1.1 and b[2] > b[3] * 1.1
                if top and landscape:
                    near = _v_overlap_frac(cur, b) >= 0.35 and _h_gap(cur, b) <= gap
                else:
                    near = (
                        _h_overlap_frac(cur, b) >= 0.2 and _v_gap(cur, b) <= gap
                    ) or (
                        _v_overlap_frac(cur, b) >= 0.2 and _h_gap(cur, b) <= gap
                    )
                if not near:
                    continue
                x = min(cur[0], b[0])
                y = min(cur[1], b[1])
                nw = max(cur[0] + cur[2], b[0] + b[2]) - x
                nh = max(cur[1] + cur[3], b[1] + b[3]) - y
                if max_w is not None and nw > max_w:
                    continue
                if max_h is not None and nh > max_h:
                    continue
                cur = (x, y, nw, nh)
                used[j] = True
                changed = True
            out.append(cur)
        boxes = out
    return boxes


def same_band_orphan_boxes(rgb: np.ndarray, items: list[dict]) -> list[tuple[int, int, int, int]]:
    """Yellow/pink overlay fragments in a line's y-band that OCR missed (face-split captions)."""
    if not items:
        return []
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellow = ((H <= 45) & (S >= 70) & (V >= 70)).astype(np.uint8) * 255
    pink = (((H >= 125) | (H <= 12)) & (S >= 60) & (V >= 80)).astype(np.uint8) * 255
    ink = cv2.bitwise_or(yellow, pink)
    extras = []
    existing = [it["box"] for it in items]
    for it in items:
        if is_vertical_box(it["box"]):
            continue
        x, y, bw, bh = [int(v) for v in it["box"]]
        y0, y1 = max(0, y - 8), min(h, y + bh + 8)
        band = ink[y0:y1]
        num, _labels, stats, _ = cv2.connectedComponentsWithStats((band > 0).astype(np.uint8), 8)
        for i in range(1, num):
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP]) + y0
            bbw = int(stats[i, cv2.CC_STAT_WIDTH])
            bbh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 80 or bbw < 18 or bbh < 12:
                continue
            if not (0.4 * bh <= bbh <= 2.4 * bh):
                continue
            box = (bx, by, bbw, bbh)
            hit = False
            for eb in existing + extras:
                ix = max(0, min(box[0] + box[2], eb[0] + eb[2]) - max(box[0], eb[0]))
                iy = max(0, min(box[1] + box[3], eb[1] + eb[3]) - max(box[1], eb[1]))
                inter = ix * iy
                if inter / max(bbw * bbh, 1) > 0.35:
                    hit = True
                    break
            if not hit:
                extras.append(box)
    return extras


def quad_from_box(box) -> "object":
    from manga_translator.utils import Quadrilateral

    x, y, w, h = [int(v) for v in box]
    pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
    return Quadrilateral(pts, "", 1.0)


def expand_box_to_mask(mask: np.ndarray, box, y_slack_ratio=0.2) -> tuple[int, int, int, int]:
    """Grow a typeset box only a little. Never flood across a comic page."""
    x, y, w, h = [int(v) for v in box]
    H, W = mask.shape
    max_w = int(w * 1.25) + 16
    max_h = int(h * 1.35) + 12
    y0 = max(0, int(y - h * 0.15))
    y1 = min(H, int(y + h * (1 + y_slack_ratio)))
    x0 = max(0, x - 8)
    x1 = min(W, x + w + 8)
    band = mask[y0:y1, x0:x1]
    if band.size == 0 or np.count_nonzero(band) == 0:
        return x, y, w, h
    cols = np.where(band.any(axis=0))[0]
    rows = np.where(band.any(axis=1))[0]
    if cols.size == 0 or rows.size == 0:
        return x, y, w, h
    nx = x0 + int(cols.min())
    ny = y0 + int(rows.min())
    nw = int(cols.max() - cols.min() + 1)
    nh = int(rows.max() - rows.min() + 1)
    if nw > max_w:
        cx = x + w / 2
        nx = int(max(0, min(nx, cx - max_w / 2)))
        nw = max_w
    if nh > max_h:
        ny = y
        nh = max_h
    return nx, ny, max(w, nw), max(h, nh)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS.get(LANG, FONT_PATHS["en"]):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw, text, f, sw):
    bbox = draw.textbbox((0, 0), text, font=f, stroke_width=sw)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_words(draw, text, f, sw, max_w, max_lines=8):
    if LANG == "ja" or (text and " " not in text and CJK_RE.search(text)):
        lines, cur = [], ""
        for ch in text:
            trial = cur + ch
            tw, _ = _text_size(draw, trial, f, sw)
            if cur and tw > max_w:
                lines.append(cur)
                cur = ch
            else:
                cur = trial
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            return None
        return lines or [text]
    words = text.split()
    if not words:
        return [text]
    lines, cur = [], words[0]
    for word in words[1:]:
        trial = cur + " " + word
        tw, _ = _text_size(draw, trial, f, sw)
        if tw <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    if len(lines) > max_lines:
        return None
    return lines


def glyph_target(box, fill_bubble: bool = False) -> int:
    """Match original Chinese line height instead of squeezing English into the box."""
    _x, _y, w, h = box
    if fill_bubble:
        # Walk down from a large size until wrapped English fits the oval.
        return int(np.clip(min(w * 0.55, h * 0.38), 30, 120))
    if h <= 110:
        return int(np.clip(h * 0.88, 22, 64))
    n = max(2, int(round(h / 70)))
    return int(np.clip(h / n * 0.9, 22, 56))


def draw_block(
    draw: ImageDraw.ImageDraw,
    box,
    text: str,
    fill,
    outline,
    img_h: int,
    allow_grow: bool = True,
    img_w: int | None = None,
    fill_bubble: bool = False,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    pad = max(6, int(min(w, h) * 0.08)) if fill_bubble else 2
    max_w = max(24, w - pad * 2)
    tall = h >= int(max(w, 1) * 1.4) and not fill_bubble
    start = glyph_target(box, fill_bubble=fill_bubble)
    if tall:
        start = min(start, 32)
    min_size = 16 if fill_bubble else (12 if tall else 18)
    max_lines = max(8, h // 16) if (tall or fill_bubble) else 6
    if fill_bubble:
        allow_grow = False
    chosen = None
    used_h = h
    inner_h = max(24, h - pad * 2)
    for s in range(start, min_size - 1, -1):
        f = font(s)
        sw = max(2 if tall else 3, s // 8)
        lines = wrap_words(draw, text, f, sw, max_w, max_lines)
        if not lines:
            continue
        widths, heights = [], []
        ok = True
        for line in lines:
            tw, th = _text_size(draw, line, f, sw)
            widths.append(tw)
            heights.append(th)
            if tw > max_w + 12:
                ok = False
        need = sum(heights) + 4 * (len(lines) - 1)
        # Only grow when nothing is stacked below; overlay captions stay near the
        # original band (do not flood the illustration with huge English).
        if fill_bubble:
            room = min(inner_h, img_h - y - pad)
        elif allow_grow:
            cap = img_h - y - 4 if tall else int(max(h, min(h * 2.2 + 28, 160)))
            room = min(img_h - y - 4, cap)
        else:
            room = min(h, img_h - y - 4)
        if ok and need <= room + 8:
            chosen = (f, sw, lines, widths, heights)
            used_h = h if fill_bubble else (int(need + 6) if allow_grow else min(h, int(need + 6)))
            break
    if chosen is None:
        f, sw = font(min_size), max(2, min_size // 8)
        lines = wrap_words(draw, text, f, sw, max_w, max(max_lines, 12)) or [text]
        widths, heights = [], []
        for line in lines:
            tw, th = _text_size(draw, line, f, sw)
            widths.append(tw)
            heights.append(max(th, 14))
        chosen = (f, sw, lines, widths, heights)
        if fill_bubble or tall or not allow_grow:
            used_h = h
        else:
            used_h = min(int(max(h, min(h * 2.2 + 28, 160))), img_h - y - 4)
    f, sw, lines, widths, heights = chosen
    gap = 4.0
    total_h = sum(heights) + gap * max(len(lines) - 1, 0)
    if fill_bubble and len(lines) > 1:
        spare = max(0.0, inner_h - total_h)
        extra = min(spare / (len(lines) - 1), max(4.0, heights[0] * 0.55))
        if extra > 1:
            gap = 4.0 + extra
            total_h = sum(heights) + gap * (len(lines) - 1)
    box_h = min(used_h, img_h - y)
    top = y + (pad if fill_bubble else 0)
    room_h = inner_h if fill_bubble else box_h
    cy = top + max(0, (room_h - total_h) / 2)
    for line, lw, lh in zip(lines, widths, heights):
        tx = x + max(0, (w - lw) / 2)
        draw.text((tx, cy), line, font=f, fill=fill, stroke_width=sw, stroke_fill=outline)
        cy += lh + gap
    return x, y, w, box_h


def draw_vertical(
    draw: ImageDraw.ImageDraw, box, text: str, fill, outline, img_h: int, img_w: int
) -> tuple[int, int, int, int]:
    """Japanese stays vertical in the bubble; English wraps horizontally inside the same box."""
    x, y, w, h = [int(v) for v in box]
    if LANG != "ja":
        return draw_block(draw, box, text, fill, outline, img_h, False, img_w)
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return x, y, w, h
    pad = 3
    inner_h, inner_w = max(16, h - pad * 2), max(16, w - pad * 2)
    # One or more columns, right-to-left.
    size = int(np.clip(min(inner_w * 0.85, inner_h / max(len(chars), 1) * 0.92), 12, 42))
    chosen = None
    for s in range(size, 11, -1):
        f = font(s)
        sw = max(2, s // 9)
        _, ch_h = _text_size(draw, "国", f, sw)
        col_cap = max(1, int(inner_h / max(ch_h + 1, 1)))
        ncols = int(np.ceil(len(chars) / col_cap))
        col_w = inner_w / max(ncols, 1)
        if col_w < s * 0.7 and ncols > 1:
            continue
        chosen = (f, sw, ch_h, col_cap, ncols, col_w)
        break
    if chosen is None:
        f, sw = font(14), 2
        _, ch_h = _text_size(draw, "国", f, sw)
        col_cap = max(1, int(inner_h / max(ch_h + 1, 1)))
        ncols = int(np.ceil(len(chars) / col_cap))
        col_w = inner_w / max(ncols, 1)
        chosen = (f, sw, ch_h, col_cap, ncols, col_w)
    f, sw, ch_h, col_cap, ncols, col_w = chosen
    for i, ch in enumerate(chars):
        col = i // col_cap
        row = i % col_cap
        cx = x + pad + inner_w - (col + 0.5) * col_w
        cy = y + pad + row * (ch_h + 1)
        tw, th = _text_size(draw, ch, f, sw)
        draw.text((cx - tw / 2, cy), ch, font=f, fill=fill, stroke_width=sw, stroke_fill=outline)
    return x, y, w, h


def _lum(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _chroma(c) -> int:
    return int(max(c)) - int(min(c))


def sample_fill_outline(rgb: np.ndarray, mask: np.ndarray, box) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Ink + stroke from original overlay pixels, so speakers keep distinct colors."""
    x, y, w, h = [int(v) for v in box[:4]]
    x2 = min(rgb.shape[1], x + max(w, 1))
    y2 = min(rgb.shape[0], y + max(h, 1))
    x, y = max(0, x), max(0, y)
    roi = rgb[y:y2, x:x2]
    m = mask[y:y2, x:x2]
    pix = roi.reshape(-1, 3)
    if m.size and np.count_nonzero(m > 127) >= 40:
        pix = roi[m > 127]
    pix = np.ascontiguousarray(pix.astype(np.float32))
    k = 2 if len(pix) >= 40 else 1
    if k == 1:
        mean = pix.mean(axis=0)
        fill = tuple(int(np.clip(v, 0, 255)) for v in mean)
        outline = (255, 255, 255) if _lum(fill) < 140 else (20, 20, 20)
        return fill, outline
    _, _labels, centers = cv2.kmeans(
        pix,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5),
        6,
        cv2.KMEANS_PP_CENTERS,
    )
    c0 = centers[0]
    c1 = centers[1]
    ch0, ch1 = _chroma(c0), _chroma(c1)
    if abs(ch0 - ch1) >= 22:
        fill_c, out_c = (c0, c1) if ch0 > ch1 else (c1, c0)
    else:
        fill_c, out_c = (c0, c1) if _lum(c0) < _lum(c1) else (c1, c0)
    fill = tuple(int(np.clip(v, 0, 255)) for v in fill_c)
    outline = tuple(int(np.clip(v, 0, 255)) for v in out_c)
    if abs(_lum(fill) - _lum(outline)) < 40:
        outline = (255, 255, 255) if _lum(fill) < 140 else (25, 25, 25)
    return fill, outline


def is_vertical_box(box) -> bool:
    _x, _y, w, h = box
    return h >= 36 and h >= int(w * 1.4)


def _merged_kind(parts: list[dict]) -> str:
    kinds = [str(p.get("kind") or "") for p in parts]
    if any(k == "bubble" for k in kinds):
        return "bubble"
    nonempty = [k for k in kinds if k]
    if nonempty and all(k == "sfx" for k in nonempty):
        return "sfx"
    if nonempty and all(k == "overlay" for k in nonempty):
        return "overlay"
    return nonempty[0] if nonempty else ""


def cluster_vertical_groups(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Nearby tall boxes in one bubble become one block (columns right-to-left)."""
    verts, hors = [], []
    for it in items:
        (verts if is_vertical_box(it["box"]) else hors).append(it)
    groups: list[dict] = []
    used = [False] * len(verts)
    for i, a in enumerate(verts):
        if used[i]:
            continue
        g = [a]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, b in enumerate(verts):
                if used[j]:
                    continue
                if any(
                    _v_overlap_frac(b["box"], x["box"]) >= 0.3 and _h_gap(b["box"], x["box"]) <= 80
                    for x in g
                ):
                    g.append(b)
                    used[j] = True
                    changed = True
        g.sort(key=lambda it: -it["box"][0])
        xs = min(p["box"][0] for p in g)
        ys = min(p["box"][1] for p in g)
        x2 = max(p["box"][0] + p["box"][2] for p in g)
        y2 = max(p["box"][1] + p["box"][3] for p in g)
        groups.append(
            {
                "box": (xs, ys, x2 - xs, y2 - ys),
                "text": "".join(p["text"] for p in g),
                "vertical": True,
                "kind": _merged_kind(g),
                "fg": g[0].get("fg"),
            }
        )
    return groups, hors


def join_comma_fragments(rows: list[dict]) -> list[dict]:
    """Same-row fragments split by a face ('你的身体，' + '也必须…').

    Mask uses the union so the leftover half-line is erased; typeset stays in the
    wider fragment so English does not sit on the face/body in the gap.
    """
    hors = [r for r in rows if not r.get("vertical")]
    verts = [r for r in rows if r.get("vertical")]
    hors = sorted(hors, key=lambda r: (r["box"][1], r["box"][0]))
    out: list[dict] = []
    skip = set()
    for i, a in enumerate(hors):
        if i in skip:
            continue
        t = a["text"].rstrip()
        if t.endswith(("，", ",", "、")):
            ay = a["box"][1] + a["box"][3] / 2
            xa, ya, wa, ha = a["box"]
            best = None
            best_gap = None
            for j, b in enumerate(hors):
                if j <= i or j in skip:
                    continue
                if not same_speaker_fill(a, b):
                    continue
                by = b["box"][1] + b["box"][3] / 2
                if abs(by - ay) > 36:
                    continue
                if b["box"][0] + 8 < xa:
                    continue
                gap = _h_gap(a["box"], b["box"])
                if gap > 140:
                    continue
                if best is None or gap < best_gap:
                    best, best_gap = j, gap
            if best is not None:
                b = hors[best]
                xb, yb, wb, hb = b["box"]
                xs, ys = min(xa, xb), min(ya, yb)
                union = (xs, ys, max(xa + wa, xb + wb) - xs, max(ya + ha, yb + hb) - ys)
                wider = a if wa >= wb else b
                merged = dict(wider)
                merged["text"] = t + b["text"]
                merged["box"] = union
                merged["typeset_box"] = tuple(wider["box"])
                merged["fg"] = a.get("fg") or b.get("fg")
                out.append(merged)
                skip.add(best)
                continue
        out.append(a)
    return verts + out


def cluster_rows(items: list[dict], y_tol: int = 40) -> list[dict]:
    """Merge only nearby boxes; do not join different panels/speakers."""
    items = sorted(items, key=lambda it: (it["box"][1] + it["box"][3] / 2, it["box"][0]))
    rows: list[dict] = []
    for it in items:
        cy = it["box"][1] + it["box"][3] / 2
        cx = it["box"][0] + it["box"][2] / 2
        placed = False
        for row in rows:
            if abs(cy - row["cy"]) > y_tol:
                continue
            if not all(same_speaker_fill(it, p) for p in row["items"]):
                continue
            gap = min(_h_gap(it["box"], p["box"]) for p in row["items"])
            # Same-color fragments only. Different speakers (black vs purple) stay split
            # even when they share a y-band.
            if gap <= 48:
                row["items"].append(it)
                row["cy"] = float(np.mean([x["box"][1] + x["box"][3] / 2 for x in row["items"]]))
                placed = True
                break
        if not placed:
            rows.append({"items": [it], "cy": cy, "cx": cx})
    out = []
    for row in rows:
        parts = sorted(row["items"], key=lambda it: it["box"][0])
        xs = min(p["box"][0] for p in parts)
        ys = min(p["box"][1] for p in parts)
        x2 = max(p["box"][0] + p["box"][2] for p in parts)
        y2 = max(p["box"][1] + p["box"][3] for p in parts)
        text = "".join(p["text"] for p in parts)
        out.append(
            {
                "box": (xs, ys, x2 - xs, y2 - ys),
                "text": text,
                "vertical": False,
                "fg": parts[0].get("fg"),
                "kind": _merged_kind(parts),
            }
        )
    return out


def cluster_all(items: list[dict]) -> list[dict]:
    vgroups, hors = cluster_vertical_groups(items)
    return join_comma_fragments(vgroups + cluster_rows(hors))


def mask_from_items(det_mask: np.ndarray, items: list[dict], rgb: np.ndarray | None = None, pad: int = 8) -> np.ndarray:
    """Inpaint only around accepted text boxes. Never flood the page."""
    H, W = det_mask.shape
    out = np.zeros((H, W), np.uint8)
    for it in items:
        x, y, bw, bh = [int(v) for v in it["box"][:4]]
        overlay = rgb is not None and is_overlay_caption(rgb, it["box"])
        compact = max(bw, bh) <= 180 and bw < 2.4 * max(bh, 1) and not it.get("vertical")
        short = len(re.sub(r"\s+", "", it.get("text") or "")) <= 4
        compact = compact or short
        local_pad = pad
        yp = pad
        # Overlay on bodies: glyph-sized pad only. A 40px y-band over a chest
        # caption is what melted hands / neck / jaw.
        if overlay or compact:
            local_pad = 14 if overlay else min(pad, 4)
            yp = 10 if overlay else local_pad
        elif bw >= 2 * max(bh, 1) and not it.get("vertical"):
            yp = max(pad, 16)
        x0, y0 = max(0, x - local_pad), max(0, y - yp)
        x1, y1 = min(W, x + bw + local_pad), min(H, y + bh + yp)
        patch = det_mask[y0:y1, x0:x1]
        glyph_only = overlay or compact
        if (not glyph_only) and np.count_nonzero(patch) >= 16:
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], (patch > 127).astype(np.uint8) * 255)
        else:
            if rgb is None or x1 <= x0 or y1 <= y0:
                continue
            roi = rgb[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            fill = _sample_glyph_color(roi)
            if fill is not None:
                dist = np.linalg.norm(roi.astype(np.float32) - fill.reshape(1, 1, 3), axis=2)
                glyphs = (dist < (52 if overlay else 55)).astype(np.uint8) * 255
                if overlay and glyphs.size:
                    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                    chroma = roi.max(axis=2) - roi.min(axis=2)
                    white = ((gray >= 190) & (chroma < 60)).astype(np.uint8) * 255
                    near_w = cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                    outlined = ((gray < 58) & (near_w > 0)).astype(np.uint8) * 255
                    stroke = (((gray < 52) | ((gray >= 190) & (chroma < 60)))).astype(np.uint8) * 255
                    near = cv2.dilate(cv2.bitwise_or(glyphs, outlined), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                    glyphs = cv2.bitwise_or(cv2.bitwise_or(glyphs, outlined), cv2.bitwise_and(stroke, near))
                out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], glyphs)
    return expand_mask(out, 4)


def line_band_ink(rgb: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Overlay-caption ink in each line's y-band, clipped to the line's x-neighborhood."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    white = ((gray >= 190) & (chroma < 60)).astype(np.uint8) * 255
    near_w = cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    outlined = ((gray < 62) & (near_w > 0)).astype(np.uint8) * 255
    ink = cv2.bitwise_or(_chroma_ink(rgb), outlined)
    out = np.zeros((h, w), np.uint8)
    for r in rows:
        if r.get("vertical"):
            continue
        x, y, bw, bh = [int(v) for v in r["box"]]
        y0, y1 = max(0, y - 8), min(h, y + bh + 8)
        if y1 <= y0:
            continue
        # Full-width y-band so 什... / 你在做 left of the OCR box still erase.
        band = ink[y0:y1]
        if band.size == 0:
            continue
        num, labels, stats, _ = cv2.connectedComponentsWithStats((band > 0).astype(np.uint8), 8)
        keep = np.zeros_like(band)
        ox0, ox1 = x, x + bw
        min_h, max_h = max(12, int(0.35 * bh)), int(max(bh * 2.4, 0.12 * h))
        for i in range(1, num):
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            bbw = int(stats[i, cv2.CC_STAT_WIDTH])
            bbh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 35 or bbh < min_h or bbh > max_h or bbw > int(0.7 * w):
                continue
            overlaps = not (bx + bbw < ox0 - 8 or bx > ox1 + 8)
            gap = min(abs(bx - ox1), abs(ox0 - (bx + bbw)))
            same_line = gap <= max(220, int(1.1 * bw)) or overlaps
            if same_line:
                keep[labels == i] = 255
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        out[y0:y1] = cv2.bitwise_or(out[y0:y1], keep)
    return out


def grow_box_along_line(rgb: np.ndarray, box, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Extend a line left/right along the same fill so 失态 / 另外一边 are not left behind."""
    x, y, w, h = [int(v) for v in box]
    y0 = max(0, y - int(0.15 * h))
    y1 = min(img_h, y + h + int(0.15 * h))
    x0 = max(0, x - int(1.2 * w) - 24)
    x1 = min(img_w, x + w + int(1.2 * w) + 24)
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return x, y, w, h
    seed = np.zeros(roi.shape[:2], np.uint8)
    sx0, sx1 = x - x0, x - x0 + w
    sx0, sx1 = max(0, sx0), min(roi.shape[1], sx1)
    band = roi[:, sx0:sx1]
    chroma = band.max(axis=2) - band.min(axis=2)
    seed[:, sx0:sx1] = (chroma >= 40).astype(np.uint8) * 255
    fill = _sample_fill_color(roi, seed)
    if fill is None:
        return x, y, w, h
    dist = np.linalg.norm(roi.astype(np.float32) - fill.reshape(1, 1, 3), axis=2)
    sim = (dist < 52).astype(np.uint8) * 255
    sim = _morph_close(sim, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    cols = np.where(sim.any(axis=0))[0]
    if cols.size == 0:
        return x, y, w, h
    # Contiguous run overlapping the original x-range — do not span a panel gap
    # to a neighbor caption of similar darkness.
    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    prev = start
    for c in cols[1:]:
        c = int(c)
        if c > prev + 8:
            runs.append((start, prev + 1))
            start = c
        prev = c
    runs.append((start, prev + 1))
    ox0, ox1 = x - x0, x - x0 + w
    hit = None
    for a, b in runs:
        if b > ox0 - 4 and a < ox1 + 4:
            hit = (a, b)
            break
    if hit is None:
        return x, y, w, h
    nx0 = x0 + hit[0]
    nx1 = x0 + hit[1]
    nx0 = max(nx0, x - int(1.1 * w) - 24)
    nx1 = min(nx1, x + w + int(1.1 * w) + 24)
    nw = max(w, nx1 - nx0)
    return nx0, y, nw, h


def canonicalize_ocr(text: str) -> str:
    """Drop OCR noise and apply misspelling aliases. Never expand 穹→阿穹."""
    t = (text or "").strip()
    if not t:
        return t
    t = re.sub(
        r"(?<=[\u4e00-\u9fff])[A-Za-z0-9|ＩｌI]{1,3}(?=[\u4e00-\u9fff])",
        "",
        t,
    )
    t = re.sub(r"^「+", "", t)
    t = t.replace("は事", "了事")
    t = t.replace("耶么", "那么").replace("耶麼", "那么")
    t = t.replace("达哒", "哒哒").replace("達哒", "哒哒")
    t = re.sub(r"[○●□■※�]+", "", t)
    t = re.sub(r"，{2,}", "，", t)
    names = load_name_table().get("names") or []
    for e in names:
        for bad in e.get("ocr_aliases") or []:
            if bad and bad in t:
                good = min((a for a in (e.get("zh") or []) if a), key=len, default=None)
                if good:
                    t = t.replace(bad, good)
    return t


def parse_translate_list(raw: str, n: int) -> list[str] | None:
    raw = (raw or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    blobs = []
    m = re.search(r"\[.*\]", raw, flags=re.S)
    if m:
        blobs.append(m.group(0))
    blobs.append(raw)
    for blob in blobs:
        for loader in (json.loads, ast.literal_eval):
            try:
                out = loader(blob)
            except Exception:
                continue
            if isinstance(out, list) and out:
                got = [str(x) for x in out]
                if len(got) < n:
                    got.extend([""] * (n - len(got)))
                return got[:n]
    qs = re.findall(r'"((?:\\.|[^"\\])*)"', raw)
    if len(qs) >= n:
        return [bytes(q, "utf-8").decode("unicode_escape") if "\\" in q else q for q in qs[:n]]
    return None


def ollama_translate(lines: list[str]) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    names = name_prompt_lines(LANG)
    if LANG == "ja":
        system = (
            "中国語のイラスト字幕を自然な日本語に翻訳する。逐語訳しない。"
            "固有名は公式訳のみ使う："
            f"{names}。"
            "阿穹 / 穹 は必ず「穹」。知更鸟 は「ロビン」。"
            "嗯? / 嗯？ → ん？  给你点颜色瞧瞧 → 痛い目を見せてやる。"
            "性的表現は検閲しない。注釈・番号・Markdown禁止。"
            "入力と同じ長さ・同じ順の JSON 文字列配列だけを返す。"
        )
    else:
        system = (
            "Translate Chinese illustration overlay captions into idiomatic natural English. "
            "Never calque Chinese idioms. Use official character names only: "
            f"{names}. "
            "阿穹 / 穹 is always Caelus, never Aqiong or Qiong. 知更鸟 is Robin. "
            "空 as the Genshin traveler is Aether. 嗯? / 嗯？ → Huh? "
            "给你点颜色瞧瞧 / 给你点颜色看看 → I'll teach you a lesson. "
            "Keep explicit sexual meaning. Do not censor. No notes, numbering, or markdown. "
            "Keep each English line concise and close to the Chinese visual length; "
            "prefer short wording so text does not cover faces. Translate SFX too (嗯？→Huh?, 是......→Yes...). "
            "Reply with a JSON array of strings, same length and order as the input array."
        )
    if VISION:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vision_locate import cloud_complete

        raw, via = cloud_complete(
            system=system,
            user=json.dumps(lines, ensure_ascii=False),
            max_tokens=1200,
        )
        print(f"translate via={via} n={len(lines)}", flush=True)
    else:
        body = {
            "model": MODEL,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_predict": 800},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(lines, ensure_ascii=False)},
            ],
        }
        req = urllib.request.Request(
            OLLAMA,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=240) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        raw = payload.get("message", {}).get("content", "")
    got = parse_translate_list(raw, len(lines))
    if got:
        return got, time.perf_counter() - t0
    bits = [b.strip(" -•\t") for b in (raw or "").splitlines() if b.strip()]
    if len(bits) == len(lines):
        return bits, time.perf_counter() - t0
    if len(lines) == 1 and (raw or "").strip():
        return [raw.strip()], time.perf_counter() - t0
    raise RuntimeError("translate parse fail: " + (raw or "")[:400])


def polish_translation(zh: str, text: str) -> str:
    """Force official names and catch leftover calques."""
    z = re.sub(r"\s+", "", zh)
    z_core = re.sub(r"[?？!！.。…~～—\-♪♡♥\s]+", "", z)
    table = load_name_table()
    for e in table.get("sfx") or []:
        ez = re.sub(r"\s+", "", e.get("zh") or "")
        ez_core = re.sub(r"[?？!！.。…~～—\-♪♡♥\s]+", "", ez)
        if z == ez or (z_core and z_core == ez_core):
            return e.get(LANG) or e.get("en") or text
    if SFX_RE.match(z) and LANG == "en":
        return "Huh?" if "?" in zh or "？" in zh else "Mm"
    if SFX_RE.match(z) and LANG == "ja":
        return "ん？" if "?" in zh or "？" in zh else "ん"
    # Longest aliases first so 阿穹 wins over 穹, 知更鸟小姐 over 知更鸟.
    names = sorted(table.get("names") or [], key=lambda e: max((len(a) for a in e.get("zh") or [""]), default=0), reverse=True)
    for e in names:
        aliases = e.get("zh") or []
        if not any(a and a in z for a in aliases):
            continue
        # Skip one-character 星/空 unless the line is that name (not 星期日 / 天空).
        if aliases in (["星"], ["空"]):
            if not re.search(r"^(阿)?" + re.escape(aliases[0]) + r"[?？!！.。…~～]*$", z):
                continue
        canonical = e.get(LANG) or e.get("en")
        if not canonical:
            continue
        for wrong in e.get("en_wrong") or []:
            text = re.sub(re.escape(wrong), canonical, text, flags=re.I)
        if LANG == "en":
            text = re.sub(r"A-?\s*Qiong", "Caelus", text, flags=re.I)
            text = re.sub(r"\bAqiong\b", "Caelus", text, flags=re.I)
            if "穹" in z:
                text = re.sub(r"\bQiong\b", "Caelus", text, flags=re.I)
            text = re.sub(r"Nightin\w*", "Robin", text, flags=re.I)
            text = re.sub(r"\bZhiying\b", "Robin", text, flags=re.I)
        if canonical.lower() not in text.lower() and any(len(a) >= 2 and a in z for a in aliases):
            # Replace the longest alias leftover if the model kept pinyin-ish forms.
            pass
        break
    if LANG == "en":
        el = text.lower()
        if ("颜色看" in z or "颜色瞧" in z) and ("color" in el or "colour" in el):
            if "看来" in z or "looks like" in el:
                return "Looks like I'm going to have to teach you a lesson."
            return "I'll teach you a lesson."
    return text


_SFX_OCR_ALIAS = {
    "說吸": "吮吸",
    "说吸": "吮吸",
    "吮呎": "吮吸",
    "嗚咽": "呜咽",
}


def _sfx_core(text: str) -> str:
    t = re.sub(r"\s+", "", text or "")
    t = _SFX_OCR_ALIAS.get(t, t)
    return re.sub(r"[?？!！.。…~～—\-♪♡♥\s]+", "", t)


def is_sfx(text: str) -> bool:
    """True for 拟声 / moans. User asked to leave these in Chinese."""
    t = re.sub(r"\s+", "", text or "")
    if not t:
        return False
    if SFX_RE.match(t):
        return True
    if BURST_RE.match(t):
        return True
    if "哒" in t and len(re.sub(r"[?？!！.。…~～—\-♪♡♥\s达]+", "", t)) <= 4:
        return True
    core = _sfx_core(t)
    if core in ("不行了", "吮吸", "嗚咽", "呜咽", "停下", "啊啊", "颤抖", "發抖", "发抖", "仁仁"):
        return True
    for e in load_name_table().get("sfx") or []:
        ez = re.sub(r"\s+", "", e.get("zh") or "")
        if t == ez or (core and core == _sfx_core(ez)):
            return True
    return False


def white_frac(rgb: np.ndarray, box) -> float:
    x, y, bw, bh = [int(v) for v in box[:4]]
    h, w = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + max(bw, 1)), min(h, y + max(bh, 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    chroma = roi.max(axis=2) - roi.min(axis=2)
    return float(np.mean((gray >= 200) & (chroma < 50)))


def is_white_sticker(rgb: np.ndarray, box, img_h: int, img_w: int) -> bool:
    """Handwritten SFX on a white cloud — do not inpaint or translate."""
    x, y, bw, bh = [int(v) for v in box[:4]]
    if bw * bh > int(0.06 * img_h * img_w):
        return False
    if bh >= int(max(bw, 1) * 1.4) and bh >= 80:
        return False
    if max(bw, bh) > int(0.28 * max(img_h, img_w)):
        return False
    return white_frac(rgb, box) >= 0.38


def should_skip_caption(it: dict, rgb: np.ndarray, img_h: int, img_w: int) -> bool:
    """Skip short 拟声 / burst marks / white stickers. Keep dialogue boxes and long overlay."""
    text = it.get("text") or ""
    if is_sfx(text) or BURST_RE.match(re.sub(r"\s+", "", text)):
        return True
    core = _sfx_core(text)
    box = it["box"]
    short = len(core) <= 4
    compact = max(box[2], box[3]) <= 220 and box[2] * box[3] <= int(0.045 * img_h * img_w)
    if short and compact and is_white_sticker(rgb, box, img_h, img_w):
        return True
    return False


def box_ink_score(rgb: np.ndarray, box) -> int:
    x, y, bw, bh = [int(v) for v in box[:4]]
    h, w = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + max(bw, 1)), min(h, y + max(bh, 1))
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(_chroma_ink(rgb)[y0:y1, x0:x1]))


def absorb_same_band_glyphs(rgb: np.ndarray, box, img_w: int, img_h: int):
    """Pull leftover same-fill glyphs in this y-band (什 left of 么) into the box."""
    x, y, bw, bh = [int(v) for v in box[:4]]
    y0, y1 = max(0, y - 6), min(img_h, y + bh + 6)
    x0, x1 = max(0, x - int(max(bw, 48)) - 16), min(img_w, x + bw + int(max(bw, 48)) + 16)
    roi = rgb[y0:y1, x0:x1]
    if roi.size == 0:
        return x, y, bw, bh
    ink = _chroma_ink(rgb)[y0:y1, x0:x1]
    seed = np.zeros(ink.shape, np.uint8)
    sx0, sx1 = x - x0, x - x0 + bw
    sx0, sx1 = max(0, sx0), min(ink.shape[1], sx1)
    seed[:, sx0:sx1] = ink[:, sx0:sx1]
    fill = _sample_fill_color(roi, seed)
    if fill is None:
        return x, y, bw, bh
    dist = np.linalg.norm(roi.astype(np.float32) - fill.reshape(1, 1, 3), axis=2)
    sim = ((ink > 0) & (dist < 58)).astype(np.uint8) * 255
    sim = _morph_close(sim, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sim, 8)
    keep = np.zeros_like(sim)
    ox0, ox1 = x - x0, x - x0 + bw
    for i in range(1, num):
        bx, by, bbw, bbh, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < 40 or area > int(0.02 * img_h * img_w) or bbh > int(0.18 * img_h):
            continue
        if not _captionish((bx, by, bbw, bbh), area, img_h) and area > 900:
            continue
        overlaps = not (bx + bbw < ox0 - 4 or bx > ox1 + 4)
        gap = min(abs(bx - ox1), abs(ox0 - (bx + bbw)))
        near = gap <= max(180, int(0.9 * bw)) and _v_overlap_frac(
            (bx, by, bbw, bbh), (ox0, 0, bw, sim.shape[0])
        ) >= 0.2
        if overlaps or near:
            keep[labels == i] = 255
    cols = np.where(keep.any(axis=0))[0]
    rows = np.where(keep.any(axis=1))[0]
    if cols.size == 0 or rows.size == 0:
        return x, y, bw, bh
    nx0 = x0 + int(cols.min())
    nx1 = x0 + int(cols.max()) + 1
    ny0 = y0 + int(rows.min())
    ny1 = y0 + int(rows.max()) + 1
    return nx0, ny0, max(bw, nx1 - nx0), max(bh, ny1 - ny0)


def clamp_typeset_box(orig, grown) -> tuple[int, int, int, int]:
    """Keep English on the source caption. Do not jump to a distant mask blob."""
    ox, oy, ow, oh = [int(v) for v in orig[:4]]
    nx, ny, nw, nh = [int(v) for v in grown[:4]]
    ocx, ocy = ox + ow / 2, oy + oh / 2
    ncx, ncy = nx + nw / 2, ny + nh / 2
    if abs(ncx - ocx) > 40 or abs(ncy - ocy) > 36:
        return ox, oy, ow, oh
    return ox, oy, min(nw, int(ow * 1.15) + 8), min(nh, int(oh * 1.2) + 8)


def expand_box_to_bubble(rgb: np.ndarray, box, img_h: int, img_w: int):
    """Grow a text quad to the white speech-bubble interior. Reject page-sized white.

    Vision boxes sit on ink, so the box center is often not white. Seed flood-fill
    from nearby white pixels instead of the glyph centroid.
    """
    x, y, bw, bh = [int(v) for v in box[:4]]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    white = ((gray >= 198) & (chroma < 55)).astype(np.uint8) * 255
    white = _morph_close(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # Bridge glyph gaps so a vertical column does not split the oval into pockets.
    white = _morph_close(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33)))
    page_lim = int(0.18 * img_h * img_w)
    text_area = max(bw * bh, 1)
    pad = int(max(bw, bh) * 0.9 + 36)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img_w, x + bw + pad), min(img_h, y + bh + pad)
    seeds: list[tuple[int, int]] = []
    inner = white[y : y + bh, x : x + bw]
    if inner.size:
        ys, xs = np.where(inner > 0)
        if xs.size:
            seeds.append((int(xs.mean()) + x, int(ys.mean()) + y))
    roi = white[y0:y1, x0:x1]
    if roi.size:
        ys, xs = np.where(roi > 0)
        if xs.size:
            cx, cy = x + bw / 2.0, y + bh / 2.0
            d = (xs.astype(np.float32) + x0 - cx) ** 2 + (ys.astype(np.float32) + y0 - cy) ** 2
            order = np.argsort(d)
            for i in order[:8]:
                seeds.append((int(xs[i] + x0), int(ys[i] + y0)))
    best = None
    best_area = 0
    seen = set()
    for sx, sy in seeds:
        sx = int(np.clip(sx, 0, img_w - 1))
        sy = int(np.clip(sy, 0, img_h - 1))
        if white[sy, sx] == 0 or (sx, sy) in seen:
            continue
        seen.add((sx, sy))
        ff = white.copy()
        ffm = np.zeros((img_h + 2, img_w + 2), np.uint8)
        cv2.floodFill(ff, ffm, (sx, sy), 128)
        ys, xs = np.where(ff == 128)
        if not xs.size or xs.size > page_lim or xs.size < int(text_area * 0.8):
            continue
        nx, ny = int(xs.min()), int(ys.min())
        nw = int(xs.max() - xs.min() + 1)
        nh = int(ys.max() - ys.min() + 1)
        ix = max(0, min(x + bw, nx + nw) - max(x, nx))
        iy = max(0, min(y + bh, ny + nh) - max(y, ny))
        if ix * iy < 0.45 * text_area:
            continue
        solidity = float(xs.size) / max(nw * nh, 1)
        if solidity < 0.52:
            continue
        if nw > int(0.42 * img_w) and nh > int(0.42 * img_h):
            continue
        if nw * nh > 10 * text_area:
            continue
        if nw > bw * 3.4 and nh > bh * 2.6:
            continue
        if xs.size > best_area:
            best_area = int(xs.size)
            m = max(8, int(min(nw, nh) * 0.07))
            best = (
                max(0, nx + m),
                max(0, ny + m),
                max(bw, nw - 2 * m),
                max(bh, nh - 2 * m),
            )
    if best is None:
        return x, y, bw, bh
    nx, ny, nw, nh = best
    nx = max(0, min(nx, img_w - 8))
    ny = max(0, min(ny, img_h - 8))
    nw = min(nw, img_w - nx)
    nh = min(nh, img_h - ny)
    return nx, ny, nw, nh


def is_speech_bubble(rgb: np.ndarray, box, img_h: int, img_w: int) -> bool:
    if is_white_sticker(rgb, box, img_h, img_w):
        return False
    wf = white_frac(rgb, box)
    tall = is_vertical_box(box)
    if wf < (0.16 if tall else 0.32):
        return False
    inner = expand_box_to_bubble(rgb, box, img_h, img_w)
    area = inner[2] * inner[3]
    if area > int(0.18 * img_h * img_w):
        return False
    if area < int(1.2 * max(box[2] * box[3], 1)) and white_frac(rgb, box) < 0.45:
        return False
    return True


def leftover_after_lama(orig: np.ndarray, inpainted: np.ndarray, box) -> bool:
    """True when we tried to erase glyphs but a visible fraction remains."""
    x, y, bw, bh = [int(v) for v in box[:4]]
    h, w = orig.shape[:2]
    pad = 12
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    roi_o = orig[y0:y1, x0:x1]
    roi_i = inpainted[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi_o, cv2.COLOR_RGB2GRAY)
    chroma = roi_o.max(axis=2) - roi_o.min(axis=2)
    white = ((gray >= 190) & (chroma < 60)).astype(np.uint8) * 255
    near_w = cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    outlined = ((gray < 62) & (near_w > 0)).astype(np.uint8)
    ink = ((_chroma_ink(orig)[y0:y1, x0:x1] > 0) | (outlined > 0))
    n_ink = int(np.count_nonzero(ink))
    if n_ink < 40:
        return False
    diff = np.linalg.norm(roi_i.astype(np.float32) - roi_o.astype(np.float32), axis=2)
    erased = ink & (diff >= 24)
    still = ink & (diff < 24)
    n_still = int(np.count_nonzero(still))
    n_erased = int(np.count_nonzero(erased))
    if n_erased < 40:
        return False
    return n_still >= 50 and n_still / n_ink >= 0.12


def leftover_caption_ink(rgb: np.ndarray, mask: np.ndarray, box) -> bool:
    """True when original glyphs in this caption were not fully covered by the mask."""
    x, y, bw, bh = [int(v) for v in box[:4]]
    h, w = rgb.shape[:2]
    pad = 10
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    chroma = rgb[y0:y1, x0:x1].max(axis=2) - rgb[y0:y1, x0:x1].min(axis=2)
    white = ((gray >= 190) & (chroma < 60)).astype(np.uint8) * 255
    near_w = cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    outlined = ((gray < 62) & (near_w > 0)).astype(np.uint8) * 255
    ink = cv2.bitwise_or(_chroma_ink(rgb)[y0:y1, x0:x1], outlined)
    miss = ((ink > 0) & (mask[y0:y1, x0:x1] == 0)).astype(np.uint8) * 255
    n_ink = int(np.count_nonzero(ink))
    n_miss = int(np.count_nonzero(miss))
    if n_ink < 50 or n_miss < 80:
        return False
    if n_miss / max(n_ink, 1) < 0.22:
        return False
    num, _lab, stats, _ = cv2.connectedComponentsWithStats((miss > 0).astype(np.uint8), 8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bbw = int(stats[i, cv2.CC_STAT_WIDTH])
        bbh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area >= 80 and bbw >= 12 and bbh >= 14 and bbh <= int(0.2 * h):
            return True
    return False


def restore_box(dst: np.ndarray, src: np.ndarray, box, pad: int = 8) -> None:
    x, y, bw, bh = [int(v) for v in box[:4]]
    h, w = src.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0:y1, x0:x1]


def mask_from_vision_items(rgb: np.ndarray, items: list[dict]) -> np.ndarray:
    """Erase using VL boxes: bubble interior + overlay/SFX glyphs. No page flood."""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    white = ((gray >= 198) & (chroma < 55)).astype(np.uint8) * 255
    for it in items:
        box = it.get("box")
        if not box:
            continue
        kind = str(it.get("kind") or "")
        bubble = kind == "bubble" or is_speech_bubble(rgb, box, h, w)
        if bubble:
            nx, ny, nw, nh = expand_box_to_bubble(rgb, box, h, w)
            patch = white[ny : ny + nh, nx : nx + nw]
            ink = _chroma_ink(rgb)[ny : ny + nh, nx : nx + nw]
            dark = (gray[ny : ny + nh, nx : nx + nw] < 90).astype(np.uint8) * 255
            combined = np.maximum(patch, np.maximum(ink, dark))
            out[ny : ny + nh, nx : nx + nw] = np.maximum(
                out[ny : ny + nh, nx : nx + nw], combined
            )
        else:
            one = mask_from_items(out, [it], rgb, pad=6)
            out = np.maximum(out, one)
    return expand_mask(out, 4)


def strip_skin_from_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    skin = (((H <= 25) | (H >= 160)) & (S >= 18) & (S <= 125) & (V >= 70) & (V <= 250))
    ink = _chroma_ink(rgb) > 0
    keep = (mask > 0) & ((~skin) | ink)
    return keep.astype(np.uint8) * 255


def resolve_clean_plate(job: dict, lettered: np.ndarray) -> tuple[np.ndarray | None, str]:
    path = job.get("clean")
    if not path:
        return None, "inpaint"
    p = Path(path)
    if not p.exists():
        return None, "inpaint_missing_clean"
    c = cv2.cvtColor(load_bgr(p), cv2.COLOR_BGR2RGB)
    if c.shape[:2] != lettered.shape[:2]:
        return None, "inpaint_clean_size_mismatch"
    return c, "clean"


def _item_fg(it: dict):
    fg = it.get("fg")
    if fg is None:
        return None
    return tuple(int(v) for v in fg[:3])


def _rgb_hue(c) -> float:
    pix = np.uint8([[list(c)]])
    return float(cv2.cvtColor(pix, cv2.COLOR_RGB2HSV)[0, 0, 0])


def same_speaker_fill(a, b, tol: float = 62.0) -> bool:
    """Black overlay vs purple overlay are different speakers even on the same y-band."""
    fa = _item_fg(a) if isinstance(a, dict) else a
    fb = _item_fg(b) if isinstance(b, dict) else b
    if fa is None or fb is None:
        return False
    fa = np.array(fa, dtype=np.float32)
    fb = np.array(fb, dtype=np.float32)
    dark_a = bool(_lum(fa) < 80 and _chroma(fa) < 70)
    dark_b = bool(_lum(fb) < 80 and _chroma(fb) < 70)
    if dark_a != dark_b:
        return False
    if float(np.linalg.norm(fa - fb)) > tol:
        return False
    if (not dark_a) and _chroma(fa) >= 50 and _chroma(fb) >= 50:
        dh = abs(_rgb_hue(fa) - _rgb_hue(fb))
        dh = min(dh, 180.0 - dh)
        if dh > 25:
            return False
    return True


def is_overlay_caption(rgb: np.ndarray, box) -> bool:
    """True when the box sits on illustration (skin/clothes/water), not a white bubble."""
    x, y, bw, bh = [int(v) for v in box[:4]]
    h, w = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + max(bw, 1)), min(h, y + max(bh, 1))
    if x1 <= x0 or y1 <= y0:
        return True
    roi = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    chroma = roi.max(axis=2) - roi.min(axis=2)
    white_frac = float(np.mean((gray >= 200) & (chroma < 50)))
    return white_frac < 0.28


def load_translation_cache(path: Path, lang: str) -> dict:
    cache = {}
    paths = [path]
    if lang == "en":
        legacy = LOG_DIR / "qingge_batch.jsonl"
        if legacy != path:
            paths.append(legacy)
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            if lang == "ja":
                got = row.get("ja") or (row.get("en") if row.get("lang") == "ja" else None)
            else:
                if row.get("lang") == "ja":
                    continue
                got = row.get("en")
            if got:
                cache[row["file"]] = {"zh": row.get("zh") or [], "tr": got}
    return cache


def warmup_ollama() -> float:
    t0 = time.perf_counter()
    ollama_translate(["测试"])
    return time.perf_counter() - t0


def log_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def main() -> None:
    t_all = time.perf_counter()
    DST.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"qingge_{LANG}.jsonl"
    summary_path = LOG_DIR / (
        f"qingge_{LANG}_sample_summary.json" if ONLY else f"qingge_{LANG}_summary.json"
    )
    trans_cache = load_translation_cache(log_path, LANG)
    print(f"lang={LANG} dst={DST} translation_cache {len(trans_cache)}", flush=True)
    det = ocr = lama = None
    inpaint_cfg = ocr_cfg = None
    load_s = 0.0
    stub_mit()
    from manga_translator.config import InpainterConfig, OcrConfig
    from manga_translator.detection.ctd import ComicTextDetector
    from manga_translator.inpainting.inpainting_lama_mpe import LamaLargeInpainter
    from manga_translator.ocr.model_48px import Model48pxOCR

    device = "cuda"
    t_load = time.perf_counter()
    if not VISION:
        det = ComicTextDetector()
        det._downloaded = True
        await det.load(device)
        ocr = Model48pxOCR()
        ocr._downloaded = True
        await ocr.load(device)
    lama = LamaLargeInpainter()
    lama._downloaded = True
    await lama.load(device)
    load_s = time.perf_counter() - t_load
    print(f"load_models {load_s:.2f}s vision={VISION}", flush=True)
    if VISION:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vision_locate import MODEL as _VM

        from vision_locate import google_api_key, zenmux_api_key

        print(f"vision_model {_VM} google_key={bool(google_api_key())} zenmux_key={bool(zenmux_api_key())}", flush=True)
    inpaint_cfg = InpainterConfig()
    inpaint_cfg.inpainting_precision = "bf16"
    ocr_cfg = OcrConfig() if not VISION else None
    warm_s = 0.0
    if not VISION:
        try:
            warm_s = warmup_ollama()
            print(f"ollama_warmup {warm_s:.2f}s", flush=True)
        except Exception as e:
            print("ollama_warmup_fail", e, flush=True)
            warm_s = -1
    jobs = []
    if GOLD_JOBS:
        jobs = GOLD_JOBS
        if ONLY:
            want = set(ONLY)
            jobs = [j for j in jobs if Path(j["src"]).name in want or j["key"] in want]
        if CLEAN:
            for j in jobs:
                if "clean" not in j:
                    cp = CLEAN / Path(j["src"]).name
                    if cp.exists():
                        j["clean"] = cp
    else:
        files = sorted(
            [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in SRC.glob(ext)],
            key=lambda p: p.name,
        )
        if ONLY:
            want = set(ONLY)
            files = [f for f in files if f.name in want]
        jobs = [{"src": f, "dst": DST / f.name, "key": f.name} for f in files]
        if CLEAN:
            for j in jobs:
                cp = CLEAN / Path(j["src"]).name
                if cp.exists():
                    j["clean"] = cp
    stats = {
        "start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(jobs),
        "passthrough": 0,
        "translated": 0,
        "skipped_existing": 0,
        "failed": 0,
        "load_s": round(load_s, 2),
        "ollama_warmup_s": round(warm_s, 2),
    }
    for i, job in enumerate(jobs, 1):
        src = job["src"]
        dst = job["dst"]
        cache_key = job["key"]
        t_img = time.perf_counter()
        row = {"file": cache_key, "i": i}
        try:
            bgr = load_bgr(src)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            t0 = time.perf_counter()
            if VISION:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from vision_locate import locate as vision_locate

                vitems = vision_locate(src, w, h)
                mask = np.zeros((h, w), np.uint8)
                grown = mask
                rec = []
                extras = []
                extra_zh = [it["text"] for it in vitems]
                items = []
                for it in vitems:
                    t = canonicalize_ocr(it["text"])
                    if not t:
                        continue
                    x, y, bw, bh = it["box"]
                    roi = rgb[y : y + bh, x : x + bw]
                    fill = _sample_glyph_color(roi) if roi.size else None
                    fg = tuple(int(v) for v in fill) if fill is not None else (30, 30, 30)
                    items.append({"text": t, "box": it["box"], "fg": fg, "bg": (255, 255, 255), "kind": it.get("kind")})
                row["detect_s"] = round(time.perf_counter() - t0, 3)
                row["ocr_s"] = 0.0
                row["vision"] = True
                row["extra_zh"] = extra_zh
                textlines = None
                tls2 = None
                rec = rec
            else:
                textlines, raw_mask, mask = await det.detect(
                    rgb,
                    detect_size=1024,
                    text_threshold=0.4,
                    box_threshold=0.55,
                    unclip_ratio=2.5,
                    invert=False,
                    gamma_correct=False,
                    rotate=False,
                    auto_rotate=False,
                    verbose=False,
                )
                row["detect_s"] = round(time.perf_counter() - t0, 3)
                if mask is None:
                    mask = raw_mask
                if mask is None:
                    mask = np.zeros((h, w), np.uint8)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                grown = complete_caption_mask(rgb, mask)
                t0 = time.perf_counter()
                rec = []
                textlines = textlines
            if not VISION and textlines:
                rec = await ocr.recognize(rgb, textlines, ocr_cfg, False)

            def tl_item(tl) -> dict | None:
                text = canonicalize_ocr((tl.text or "").strip())
                if not text or not CJK_RE.search(text):
                    return None
                xs, ys = tl.pts[:, 0], tl.pts[:, 1]
                x1, y1 = int(max(0, xs.min())), int(max(0, ys.min()))
                x2, y2 = int(min(w, xs.max())), int(min(h, ys.max()))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    return None
                return {
                    "text": text,
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "fg": (int(tl.fg_r), int(tl.fg_g), int(tl.fg_b)),
                    "bg": (int(tl.bg_r), int(tl.bg_g), int(tl.bg_b)),
                }

            if not VISION:
                items = [it for it in (tl_item(tl) for tl in rec) if it]

            def accept_extra(it: dict) -> bool:
                n_cjk = len(CJK_RE.findall(it["text"]))
                z = re.sub(r"[?？!！.。…~～—\-♪♡♥\s]+", "", it["text"])
                if should_skip_caption(it, rgb, h, w):
                    return False
                if n_cjk < 1:
                    return False
                if n_cjk < 2 and z not in ("穹",):
                    return False
                if it["box"][2] > int(0.48 * w):
                    return False
                for old in items:
                    if it["text"] in old["text"] or old["text"] in it["text"]:
                        return False
                    ix = max(0, min(it["box"][0] + it["box"][2], old["box"][0] + old["box"][2]) - max(it["box"][0], old["box"][0]))
                    iy = max(0, min(it["box"][1] + it["box"][3], old["box"][1] + old["box"][3]) - max(it["box"][1], old["box"][1]))
                    inter = ix * iy
                    if inter / max(it["box"][2] * it["box"][3], 1) > 0.45:
                        return False
                return True

            if not VISION:
                extra_zh = []
            if not VISION:
                tls2, raw2, mask2 = await det.detect(
                rgb,
                detect_size=1024,
                text_threshold=0.28,
                box_threshold=0.38,
                unclip_ratio=2.0,
                invert=False,
                gamma_correct=False,
                rotate=False,
                auto_rotate=False,
                verbose=False,
            )
            if (not VISION) and tls2:
                rec2 = await ocr.recognize(rgb, tls2, ocr_cfg, False)
                for tl in rec2:
                    it = tl_item(tl)
                    if it and accept_extra(it):
                        items.append(it)
                        extra_zh.append(it["text"])
            existing_boxes = [it["box"] for it in items]
            ov_ink = grown if VISION else independent_outlined_overlay(rgb)
            # Do not filter overlay SFX against detector boxes — those boxes can
            # sit on a nearby speaker and swallow 阿穹 / 不行了.
            overlay_ex = extra_ink_boxes(ov_ink, [])
            overlay_ex = _merge_nearby_boxes(
                overlay_ex,
                gap=24,
                max_w=int(0.24 * w),
                max_h=int(0.22 * h),
                img_h=h,
            )
            overlay_ex = [
                b for b in overlay_ex if b[2] * b[3] <= int(0.03 * h * w)
            ]
            overlay_ex = [grow_chroma_box(rgb, b, w, h) for b in overlay_ex]
            pad_ex = 10
            overlay_ex = [
                (
                    max(0, b[0] - pad_ex),
                    max(0, b[1] - pad_ex),
                    min(w - max(0, b[0] - pad_ex), b[2] + 2 * pad_ex),
                    min(h - max(0, b[1] - pad_ex), b[3] + 2 * pad_ex),
                )
                for b in overlay_ex
            ]
            grown_ex = extra_ink_boxes(grown, existing_boxes)
            grown_ex += same_band_orphan_boxes(rgb, items)
            grown_ex = [b for b in grown_ex if b[2] <= int(0.48 * w)]
            overlay_ex = [
                b
                for b in overlay_ex
                if b[2] <= int(0.48 * w) and max(b[2], b[3]) <= int(0.28 * max(h, w))
            ]
            overlay_ex.sort(key=lambda b: b[2] * b[3])
            overlay_ex = overlay_ex[:24]
            overlay_ex = [b for b in overlay_ex if not is_white_sticker(rgb, b, h, w)]
            extras = overlay_ex + grown_ex[:16]
            if VISION:
                extras = []
            if extras:
                ocr_rgb = rgb.copy()
                ink = _chroma_ink(rgb)
                for x, y, bw, bh in extras:
                    x, y, bw, bh = int(x), int(y), int(bw), int(bh)
                    x1, y1 = min(w, x + bw), min(h, y + bh)
                    x, y = max(0, x), max(0, y)
                    if x1 <= x or y1 <= y:
                        continue
                    patch = ink[y:y1, x:x1]
                    sl = ocr_rgb[y:y1, x:x1]
                    sl[patch == 0] = (255, 255, 255)
                    sl[patch > 0] = (20, 20, 20)
                extra_rec = await ocr.recognize(ocr_rgb, [quad_from_box(b) for b in extras], ocr_cfg, False)
                for tl in extra_rec:
                    it = tl_item(tl)
                    if it and accept_extra(it):
                        items.append(it)
                        extra_zh.append(it["text"])
            if not VISION:
                items = [it for it in items if not should_skip_caption(it, rgb, h, w)]
                row["ocr_s"] = round(time.perf_counter() - t0, 3)
                row["extra_zh"] = extra_zh
            # Re-OCR boxes that grew along the same overlay line (漏掉的后半句 / 什 leftover).
            reocr = []
            if not VISION:
                for it in items:
                    if is_vertical_box(it["box"]):
                        continue
                    nb = grow_box_along_line(rgb, it["box"], w, h)
                    if is_overlay_caption(rgb, it["box"]):
                        nb = absorb_same_band_glyphs(rgb, nb, w, h)
                    if nb[2] > it["box"][2] + 8 or nb[3] > it["box"][3] + 8:
                        reocr.append((it, nb))
                        it["box"] = nb
            if (not VISION) and reocr:
                recg = await ocr.recognize(rgb, [quad_from_box(nb) for _, nb in reocr], ocr_cfg, False)
                for (it, _nb), tl in zip(reocr, recg):
                    nt = canonicalize_ocr((tl.text or "").strip())
                    if nt and CJK_RE.search(nt) and len(nt) >= len(it["text"]):
                        old = it["text"]
                        if (
                            nt.startswith(old)
                            or old in nt
                            or SequenceMatcher(None, old, nt).ratio() >= 0.5
                        ):
                            it["text"] = nt
            # Stacked overlays that OCR-copied a neighbor: shrink the upper box and re-read.
            restack = []
            for ia, a in enumerate(items):
                for b in items[ia + 1 :]:
                    ta, tb = a["text"], b["text"]
                    if (
                        SequenceMatcher(None, ta, tb).ratio() < 0.7
                        and ta not in tb
                        and tb not in ta
                    ):
                        continue
                    upper, lower = (a, b) if a["box"][1] <= b["box"][1] else (b, a)
                    if _v_gap(upper["box"], lower["box"]) > 30:
                        continue
                    if _h_overlap_frac(upper["box"], lower["box"]) < 0.25:
                        continue
                    ux, uy, uw, uh = upper["box"]
                    cut = lower["box"][1] - uy - 4
                    if cut >= 18 and cut < uh:
                        upper["box"] = (ux, uy, uw, cut)
                        restack.append(upper)
            if (not VISION) and restack:
                recs = await ocr.recognize(
                    rgb, [quad_from_box(it["box"]) for it in restack], ocr_cfg, False
                )
                for it, tl in zip(restack, recs):
                    nt = canonicalize_ocr((tl.text or "").strip())
                    if nt and CJK_RE.search(nt):
                        it["text"] = nt
            if not items:
                shutil.copy2(src, dst)
                stats["passthrough"] += 1
                row["status"] = "passthrough_no_cjk" if rec or extras else "passthrough"
                row["ocr"] = [(tl.text or "") for tl in rec]
                row["total_s"] = round(time.perf_counter() - t_img, 3)
                print(f"{i}/{len(jobs)} {row['status']} {cache_key} ocr={[tl.text for tl in rec]}", flush=True)
                log_row(log_path, row)
                continue
            # Wide vertical bubbles: OCR each column so 评估/数据/阈值 stay separate.
            col_jobs: list[tuple[int, list]] = []
            for idx, it in enumerate(items):
                if not is_vertical_box(it["box"]):
                    continue
                cols = split_ink_columns(rgb, it["box"])
                if len(cols) >= 2:
                    col_jobs.append((idx, cols))
            if (not VISION) and col_jobs:
                flat = [c for _, cols in col_jobs for c in cols]
                crec = await ocr.recognize(rgb, [quad_from_box(c) for c in flat], ocr_cfg, False)
                k = 0
                cmap: dict[int, list[dict]] = {}
                for idx, cols in col_jobs:
                    got = []
                    for _c in cols:
                        itc = tl_item(crec[k]) if k < len(crec) else None
                        k += 1
                        if itc:
                            got.append(itc)
                    if got:
                        cmap[idx] = got
                if cmap:
                    new_items: list[dict] = []
                    for idx, it in enumerate(items):
                        if idx in cmap:
                            new_items.extend(cmap[idx])
                        else:
                            new_items.append(it)
                    items = new_items
            if not VISION:
                items = [it for it in items if not should_skip_caption(it, rgb, h, w)]
            # Same OCR text in two boxes: keep the one sitting on more glyph ink.
            ranked = sorted(items, key=lambda it: -box_ink_score(rgb, it["box"]))
            kept: list[dict] = []
            for it in ranked:
                dup = False
                for old in kept:
                    ta, tb = it["text"], old["text"]
                    similar = (
                        ta == tb
                        or (len(ta) >= 2 and (ta in tb or tb in ta))
                        or SequenceMatcher(None, ta, tb).ratio() >= 0.78
                    )
                    if not similar:
                        continue
                    xa, ya, wa, ha = old["box"]
                    xb, yb, wb, hb = it["box"]
                    xs, ys = min(xa, xb), min(ya, yb)
                    old["box"] = (xs, ys, max(xa + wa, xb + wb) - xs, max(ya + ha, yb + hb) - ys)
                    if len(it["text"]) > len(old["text"]):
                        old["text"] = it["text"]
                        if it.get("fg"):
                            old["fg"] = it["fg"]
                    dup = True
                    break
                if not dup:
                    kept.append(it)
            items = kept
            rows = cluster_all(items)
            rows.sort(key=lambda r: (r["box"][1], r["box"][0]))
            src_texts = [r["text"] for r in rows]
            cached = trans_cache.get(cache_key)
            t0 = time.perf_counter()
            if (not RETRANSLATE) and cached and len(cached["tr"]) == len(src_texts):
                tr_texts = cached["tr"]
                row["cached"] = True
            else:
                tr_texts = ollama_translate(src_texts)[0]
            tr_texts = [polish_translation(z, e) for z, e in zip(src_texts, tr_texts)]
            row["translate_s"] = round(time.perf_counter() - t0, 3)
            row["zh"] = src_texts
            row["lang"] = LANG
            row[LANG] = tr_texts
            if LANG == "en":
                row["en"] = tr_texts
            if VISION:
                color_mask = mask_from_vision_items(rgb, rows + items)
                mask = color_mask
            else:
                color_mask = mask_from_items(
                    grown,
                    rows + [{"box": it["box"]} for it in items],
                    rgb,
                    pad=8,
                )
                color_mask = cv2.bitwise_or(color_mask, line_band_ink(rgb, rows))
                color_mask = absorb_trailing_punct(rgb, color_mask, [r["box"] for r in rows])
                band = line_band_ink(rgb, rows)
                clip = cv2.bitwise_or(
                    mask_from_items(grown, rows + [{"box": it["box"]} for it in items], rgb, pad=20),
                    band,
                )
                color_mask = cv2.bitwise_and(color_mask, clip)
                color_mask = strip_skin_from_mask(rgb, color_mask)
                mask = expand_mask(color_mask, 3)
                mask = strip_skin_from_mask(rgb, mask)
            canvas, mode = resolve_clean_plate(job, rgb)
            row["mode"] = mode
            t0 = time.perf_counter()
            if canvas is not None:
                inpainted = canvas
                row["lama_s"] = 0.0
            else:
                inpainted = await lama.inpaint(rgb, mask, inpaint_cfg, 1536, False)
                row["lama_s"] = round(time.perf_counter() - t0, 3)
            skipped_incomplete = []
            if not VISION:
                for block in rows:
                    ts = block.get("typeset_box") or block["box"]
                    if leftover_caption_ink(rgb, mask, ts):
                        # Mode A: put original pixels back. Mode B: never paste lettered
                        # Chinese onto the clean plate — just skip typesetting.
                        if canvas is None:
                            restore_box(inpainted, rgb, ts, pad=14)
                        skipped_incomplete.append(block["text"])
            row["skip_incomplete"] = skipped_incomplete
            skip_zh = set(skipped_incomplete)
            im = Image.fromarray(inpainted)
            draw = ImageDraw.Draw(im)
            placed = []
            seen = []
            for block, tr in zip(rows, tr_texts):
                zh = block["text"]
                if zh in skip_zh:
                    continue
                hit = None
                for s in seen:
                    tr_dup = (
                        tr == s["tr"]
                        or (len(tr) > 8 and (tr in s["tr"] or s["tr"] in tr))
                        or SequenceMatcher(None, tr.lower(), s["tr"].lower()).ratio() >= 0.72
                    )
                    zh_dup = (
                        zh == s["zh"]
                        or zh in s["zh"]
                        or s["zh"] in zh
                        or SequenceMatcher(None, zh, s["zh"]).ratio() >= 0.55
                    )
                    if tr_dup and zh_dup:
                        hit = s
                        break
                if hit:
                    if len(tr) > len(hit["tr"]):
                        hit["tr"] = tr
                        hit["block"] = block
                    continue
                seen.append({"zh": zh, "tr": tr, "block": block})
            for s in seen:
                block, tr = s["block"], s["tr"]
                ts_box = block.get("typeset_box") or block["box"]
                kind = str(block.get("kind") or "")
                bubble = kind == "bubble" or (
                    kind not in ("overlay", "sfx") and is_speech_bubble(rgb, ts_box, h, w)
                )
                if bubble:
                    box = expand_box_to_bubble(rgb, ts_box, h, w)
                    # Mode B may have deleted the oval (99). Keep fill only if
                    # the destination patch is still a pale cavity.
                    if canvas is not None:
                        x0, y0, bw0, bh0 = [int(v) for v in box]
                        patch = inpainted[y0 : y0 + bh0, x0 : x0 + bw0]
                        if patch.size:
                            g = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                            ch = patch.max(axis=2) - patch.min(axis=2)
                            pale = float(np.mean((g >= 165) & (ch < 80)))
                        else:
                            pale = 0.0
                        if pale < 0.18:
                            box = tuple(int(v) for v in ts_box[:4])
                            bubble = False
                else:
                    box = tuple(int(v) for v in ts_box[:4])
                fill, outline = sample_fill_outline(rgb, color_mask, ts_box)
                placed.append(
                    {
                        "box": box,
                        "en": tr,
                        "fill": fill,
                        "outline": outline,
                        "vertical": bool(block.get("vertical")),
                        "bubble": bubble,
                        "overlay": (not bubble) and (kind == "overlay" or is_overlay_caption(rgb, ts_box)),
                    }
                )
            # Merge only heavy overlaps (same line split); keep stacked captions separate.
            merged_place = []
            for p in placed:
                hit = None
                px, py, pw, ph = p["box"]
                pa = max(pw * ph, 1)
                for q in merged_place:
                    qx, qy, qw, qh = q["box"]
                    ix = max(0, min(px + pw, qx + qw) - max(px, qx))
                    iy = max(0, min(py + ph, qy + qh) - max(py, qy))
                    inter = ix * iy
                    if inter / pa > 0.55 or inter / max(qw * qh, 1) > 0.55:
                        if p.get("bubble") != q.get("bubble"):
                            continue
                        if bool(p.get("overlay")) != bool(q.get("overlay")):
                            continue
                        if same_speaker_fill(
                            {"fg": p.get("fill")}, {"fg": q.get("fill")}
                        ):
                            hit = q
                            break
                if hit is None:
                    merged_place.append(dict(p))
                else:
                    hx, hy, hw, hh = hit["box"]
                    nx, ny = min(hx, px), min(hy, py)
                    hit["box"] = (
                        nx,
                        ny,
                        max(hx + hw, px + pw) - nx,
                        max(hy + hh, py + ph) - ny,
                    )
                    hit["en"] = (hit["en"] + " " + p["en"]).strip()
            merged_place.sort(key=lambda p: (p["box"][1], p["box"][0]))
            for pi, p in enumerate(merged_place):
                p["allow_grow"] = False
                x, y, bw, bh = p["box"]
                if p.get("overlay") and pi + 1 < len(merged_place):
                    nxt = merged_place[pi + 1]
                    if nxt.get("overlay") and abs(nxt["box"][0] - x) < max(bw, nxt["box"][2]) * 0.8:
                        limit = max(22, nxt["box"][1] - y - 8)
                        p["box"] = (x, y, bw, min(bh, limit))
            colors = []
            for p in merged_place:
                if p.get("vertical") and not p.get("bubble"):
                    box_used = draw_vertical(
                        draw, p["box"], p["en"], p["fill"], p["outline"], h, w
                    )
                else:
                    box_used = draw_block(
                        draw,
                        p["box"],
                        p["en"],
                        p["fill"],
                        p["outline"],
                        h,
                        False,
                        w,
                        fill_bubble=bool(p.get("bubble")),
                    )
                colors.append({"fill": p["fill"], "outline": p["outline"], "box": list(box_used)})
            row["colors"] = colors
            out = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            save_bgr(dst, out)
            stats["translated"] += 1
            row["status"] = "ok"
            row["n_blocks"] = len(rows)
            row["total_s"] = round(time.perf_counter() - t_img, 3)
            print(
                f"{i}/{len(jobs)} ok {cache_key} mode={row.get('mode')} blocks={len(rows)} "
                f"skip={row.get('skip_incomplete')} "
                f"det={row['detect_s']} ocr={row['ocr_s']} tr={row['translate_s']} "
                f"lama={row['lama_s']} total={row['total_s']}",
                flush=True,
            )
        except Exception as e:
            import traceback
            stats["failed"] += 1
            row["status"] = "fail"
            row["error"] = traceback.format_exc()[:1500]
            row["total_s"] = round(time.perf_counter() - t_img, 3)
            print(f"{i}/{len(jobs)} FAIL {cache_key} {row['error']}", flush=True)
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
        log_row(log_path, row)
    stats["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["elapsed_s"] = round(time.perf_counter() - t_all, 2)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--retranslate", action="store_true")
    ap.add_argument("--lang", choices=("en", "ja"), default="en")
    ap.add_argument("--src", default=None, help="Folder of lettered (Chinese) images")
    ap.add_argument("--dst", default=None, help="Output folder (created if missing)")
    ap.add_argument("--mit-root", default=None, help="Path to manga-image-translator clone")
    ap.add_argument("--names", default=None, help="JSON name table (default: names.json next to this script)")
    ap.add_argument(
        "--clean",
        default=None,
        help="Optional unlettered plates, same filenames as --src. Skip LaMa and typeset on the plate.",
    )
    ap.add_argument(
        "--vision",
        action="store_true",
        help="Locate+translate with Gemini 3.7 Flash (Google AI Studio first, ZenMux fallback).",
    )
    ap.add_argument("--regression", action="store_true", help="Run experiments/nsfw_local/regression_set.txt")
    ap.add_argument("--gold", action="store_true", help="Run locked gold_set.json across three series")
    ns = ap.parse_args()
    ONLY = ns.only
    GOLD_JOBS = None
    if ns.gold:
        spec = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        jobs = []
        for series in spec["series"]:
            src_root = Path(series["src"])
            dst_root = Path(series["dst"][ns.lang])
            dst_root.mkdir(parents=True, exist_ok=True)
            for it in series["items"]:
                p = src_root / it["file"]
                if not p.exists():
                    print("missing_gold", series["id"], it["file"], flush=True)
                    continue
                job = {"src": p, "dst": dst_root / it["file"], "key": series["id"] + "/" + it["file"]}
                clean_root = series.get("clean")
                if clean_root:
                    cp = Path(clean_root) / it["file"]
                    if cp.exists():
                        job["clean"] = cp
                jobs.append(job)
        GOLD_JOBS = jobs
    if ns.regression:
        rs = Path(os.environ.get("LETTER_REGRESSION", str(HERE / "regression_set.txt")))
        names = []
        for line in rs.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
        ONLY = names
    RETRANSLATE = ns.retranslate
    LANG = ns.lang
    if ns.src:
        SRC = Path(ns.src)
    if ns.dst:
        DST = Path(ns.dst)
    if ns.mit_root:
        MIT_ROOT = Path(ns.mit_root)
        MIT_PKG = MIT_ROOT / "manga_translator"
    if ns.names:
        NAMES_PATH = Path(ns.names)
    CLEAN = Path(ns.clean) if ns.clean else None
    VISION = bool(ns.vision)
    if not ns.gold and (not ns.src or not ns.dst):
        ap.error("provide --src and --dst (or --gold with gold_set.json)")
    asyncio.run(main())
