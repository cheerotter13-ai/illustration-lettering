"""Shared image helpers for the Mode B batch pipeline."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def skill_root() -> Path:
    return Path(__file__).resolve().parent.parent


def engine_dir() -> Path:
    import os

    env = (os.environ.get("LETTER_ENGINE") or "").strip()
    if env:
        return Path(env)
    bundled = skill_root() / "engine"
    if (bundled / "letter_b.py").exists():
        return bundled
    here = skill_root()
    for cand in (here.parent.parent.parent, here.parent.parent, here.parent):
        if (cand / "letter_b.py").exists():
            return cand
    return bundled


def iter_images(root: Path, *, recursive: bool = False) -> list[Path]:
    if not root or not root.exists():
        return []
    it = root.rglob("*") if recursive else root.iterdir()
    out = [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXT]
    out.sort(key=lambda p: p.name)
    return out


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def ahash(im: Image.Image, size: int = 16) -> np.ndarray:
    g = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    return (arr > arr.mean()).astype(np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def mad(a: Image.Image, b: Image.Image) -> float:
    if a.size != b.size:
        return 255.0
    x = np.asarray(a, dtype=np.int16)
    y = np.asarray(b, dtype=np.int16)
    return float(np.abs(x - y).mean())


def norm_name(name: str) -> str:
    stem = Path(name).stem
    stem = stem.replace(" ", "")
    stem = re.sub(r"_0001$", "", stem)
    return stem.lower()


def thin_v_gutter(im: Image.Image) -> tuple[int, int, float]:
    arr = np.asarray(im.convert("L"))
    h, w = arr.shape
    dark = (arr < 22).mean(axis=0)
    best_x, best_s = w // 2, -1.0
    for x in range(w // 3, 2 * w // 3):
        s = float(dark[max(0, x - 1) : x + 2].mean())
        if s > best_s:
            best_s, best_x = s, x
    x0 = x1 = best_x
    while x0 > 0 and dark[x0 - 1] > 0.55:
        x0 -= 1
    while x1 < w - 1 and dark[x1] > 0.55:
        x1 += 1
    return x0, max(x1, x0 + 4), best_s


def thin_h_gutter(im: Image.Image) -> tuple[int, int, float]:
    arr = np.asarray(im.convert("L"))
    h, w = arr.shape
    dark = (arr < 22).mean(axis=1)
    best_y, best_s = h // 2, -1.0
    for y in range(h // 4, 3 * h // 4):
        s = float(dark[max(0, y - 1) : y + 2].mean())
        if s > best_s:
            best_s, best_y = s, y
    y0 = y1 = best_y
    while y0 > 0 and dark[y0 - 1] > 0.55:
        y0 -= 1
    while y1 < h - 1 and dark[y1] > 0.55:
        y1 += 1
    return y0, max(y1, y0 + 4), best_s


def caption_ink(rgb: np.ndarray) -> np.ndarray:
    """Dark / magenta / green overlay ink (Odette-style captions)."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    gray = (r + g + b) / 3
    dark = (gray < 58) & (chroma < 100)
    magenta = (r > 90) & (b > 90) & (g + 40 < r) & (g + 40 < b)
    green = (g > 90) & (g > r + 25) & (g > b + 15)
    return (dark | magenta | green)


def has_overlay_captions(lettered: Image.Image, clean: Image.Image | None) -> bool:
    arr = np.asarray(lettered)
    h, w = arr.shape[:2]
    ink = caption_ink(arr)
    if clean is not None and clean.size == lettered.size:
        carr = np.asarray(clean)
        diff = np.abs(arr.astype(np.int16) - carr.astype(np.int16)).mean(axis=2) > 18
        ink = ink & diff
    ys, xs = np.where(ink)
    if len(ys) < 80:
        return False
    # Horizontal bands: many ink pixels in a short vertical span, wide horizontally.
    row_frac = ink.mean(axis=1)
    band = row_frac > 0.04
    if not band.any():
        return False
    # Prefer top 20% / bottom 30% (typical overlay captions).
    top = band[: max(1, h // 5)].mean() if h > 20 else 0
    bot = band[int(h * 0.7) :].mean() if h > 20 else 0
    mid = band[h // 5 : int(h * 0.7)].mean() if h > 20 else 0
    if top > 0.08 or bot > 0.08:
        return True
    xs_span = int(xs.max() - xs.min()) if len(xs) else 0
    ys_span = int(ys.max() - ys.min()) if len(ys) else 0
    if xs_span > 0.28 * w and ys_span < 0.22 * h:
        return True
    return bool(mid > 0.18)


def choose_axis(lettered: Image.Image, same: Image.Image | None) -> str:
    w, h = lettered.size
    if same is not None:
        if same.size[0] * 1.6 < w:
            return "lr"
        if same.size[1] * 1.2 < h:
            return "tb"
    if w > h * 1.15:
        return "lr"
    return "tb"


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
