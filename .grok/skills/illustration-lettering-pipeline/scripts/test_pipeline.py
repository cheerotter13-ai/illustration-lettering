#!/usr/bin/env python3
"""Synthetic classify + gutter + stitch checks. No API."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import has_overlay_captions, thin_v_gutter  # noqa: E402


def _bar_image(path: Path, *, caption: bool, size=(320, 240), extra_w=0) -> None:
    w, h = size[0] + extra_w, size[1]
    im = Image.new("RGB", (w, h), (180, 160, 140))
    d = ImageDraw.Draw(im)
    d.rectangle((20, 30, 80, 90), fill=(90, 70, 60))
    if caption:
        d.rectangle((16, h - 36, w - 16, h - 12), fill=(25, 25, 25))
    im.save(path)


def test_overlay_detect() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a = td / "a.png"
        b = td / "b.png"
        _bar_image(a, caption=True)
        _bar_image(b, caption=False)
        la = Image.open(a).convert("RGB")
        lb = Image.open(b).convert("RGB")
        assert has_overlay_captions(la, lb), "captioned vs clean should be overlay"
        assert not has_overlay_captions(lb, lb), "identical should not be overlay"


def test_prepare_and_gutter() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, clean, work = td / "src", td / "clean", td / "work"
        src.mkdir()
        clean.mkdir()
        _bar_image(src / "1 (1).png", caption=True)
        _bar_image(clean / "1 (1).png", caption=False)
        _bar_image(src / "sfx.png", caption=False)
        sfx_c = Image.open(src / "sfx.png").convert("RGB")
        d = ImageDraw.Draw(sfx_c)
        d.ellipse((90, 70, 150, 140), outline=(20, 20, 20), width=8)
        d.text((100, 90), "boom", fill=(20, 20, 20))
        sfx_c.save(src / "sfx.png")
        _bar_image(clean / "sfx.png", caption=False)
        _bar_image(src / "same.png", caption=False)
        _bar_image(clean / "same.png", caption=False)
        # double: wide lettered, two clean halves
        dbl = Image.new("RGB", (640, 240), (180, 160, 140))
        dd = ImageDraw.Draw(dbl)
        dd.rectangle((316, 0, 324, 240), fill=(0, 0, 0))
        dd.rectangle((20, 204, 280, 228), fill=(25, 25, 25))
        dbl.save(src / "wide.png")
        _bar_image(clean / "wide_L.png", caption=False, size=(320, 240))
        _bar_image(clean / "wide_R.png", caption=False, size=(320, 240))
        cmd = [
            sys.executable,
            str(HERE / "pipeline.py"),
            "prepare",
            "--src",
            str(src),
            "--clean",
            str(clean),
            "--work",
            str(work),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        man = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
        kinds = {f["file"]: f["kind"] for f in man["files"]}
        assert kinds["1 (1).png"] == "overlay", kinds
        assert kinds["same.png"] == "identical", kinds
        assert kinds["sfx.png"] in ("sfx", "overlay"), kinds
        assert kinds["wide.png"] == "double", kinds
        ov = list((work / "lettered_overlay").iterdir())
        assert any(p.name == "1 (1).png" for p in ov)
        im = Image.open(src / "wide.png").convert("RGB")
        x0, x1, s = thin_v_gutter(im)
        assert s > 0.5, s
        assert 300 < x0 < 330, x0


def main() -> int:
    test_overlay_detect()
    test_prepare_and_gutter()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
