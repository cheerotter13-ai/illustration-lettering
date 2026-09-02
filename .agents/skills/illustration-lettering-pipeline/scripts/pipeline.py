#!/usr/bin/env python3
"""Mode B batch pipeline: classify, recover, run letter_b, split/stitch doubles."""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from common import (
    ahash,
    choose_axis,
    copy_file,
    dump_json,
    engine_dir,
    hamming,
    has_overlay_captions,
    iter_images,
    load_json,
    load_rgb,
    mad,
    norm_name,
    skill_root,
    thin_h_gutter,
    thin_v_gutter,
)

HAM_OK = 12
MAD_IDENTICAL = 2.5
GUTTER_MIN = 0.35


def _work(ns) -> Path:
    p = Path(ns.work)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_setup(ns) -> int:
    from setup_config import main as setup_main

    argv = ["setup_config.py", "--base-url", ns.base_url, "--api-key", ns.api_key, "--model", ns.model]
    if ns.test:
        argv.append("--test")
    if ns.also_dotenv:
        argv.append("--also-dotenv")
    sys.argv = argv
    return setup_main()


def cmd_add_name(ns) -> int:
    path = engine_dir() / "names.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data.setdefault("names", [])
    zh = [z.strip() for z in ns.zh if z.strip()]
    for rec in names:
        if rec.get("en") == ns.en or any(z in rec.get("zh", []) for z in zh):
            rec["zh"] = list(dict.fromkeys(zh + list(rec.get("zh") or [])))
            rec["en"] = ns.en
            rec["ja"] = ns.ja
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("updated", path, rec["zh"][0], ns.en, ns.ja)
            return 0
    names.append({"zh": zh, "en": ns.en, "ja": ns.ja})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("added", path, zh[0], ns.en, ns.ja)
    return 0


def _index_cleans(clean: Path, omake: Path | None) -> list[tuple[str, Path, tuple[int, int]]]:
    items = []
    seen = set()
    roots = [clean]
    if omake and omake.exists():
        roots.append(omake)
    for root in roots:
        rec = iter_images(root, recursive=True)
        for p in rec:
            if p.name in seen:
                continue
            seen.add(p.name)
            with Image.open(p) as im:
                items.append((p.name, p, im.size))
    return items


def _match_clean(src_name: str, src_size, cleans) -> tuple[Path | None, str]:
    exact = [c for c in cleans if c[0] == src_name]
    if exact:
        return exact[0][1], "exact"
    n = norm_name(src_name)
    fuzzy = [c for c in cleans if norm_name(c[0]) == n]
    if len(fuzzy) == 1:
        return fuzzy[0][1], "fuzzy_name"
    same = [c for c in fuzzy if c[2] == src_size]
    if len(same) == 1:
        return same[0][1], "fuzzy_name_size"
    return None, ""


def _visual_match(lettered: Image.Image, cleans, used: set[str]) -> tuple[Path | None, int, str]:
    target = ahash(lettered)
    want = lettered.size
    best = None
    best_d = 10**9
    best_name = ""
    for name, path, size in cleans:
        if name in used or size != want:
            continue
        with Image.open(path) as im:
            d = hamming(target, ahash(im.convert("RGB")))
        if d < best_d:
            best_d, best, best_name = d, path, name
    if best is not None and best_d <= HAM_OK:
        return best, best_d, best_name
    return None, best_d if best_d < 10**9 else -1, best_name


def cmd_prepare(ns) -> int:
    src = Path(ns.src)
    clean = Path(ns.clean)
    work = _work(ns)
    omake = Path(ns.omake) if ns.omake else (src / "omake")
    if not src.is_dir() or not clean.is_dir():
        print("need --src and --clean directories", file=sys.stderr)
        return 2
    lettered = iter_images(src)
    cleans = _index_cleans(clean, omake if omake.exists() else None)
    used: set[str] = set()
    files = []
    overlay_dir = work / "lettered_overlay"
    clean_ov = work / "clean_overlay"
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    if clean_ov.exists():
        shutil.rmtree(clean_ov)
    overlay_dir.mkdir(parents=True)
    clean_ov.mkdir(parents=True)

    for sp in lettered:
        rec = {"file": sp.name, "src": str(sp), "kind": "", "how": "", "clean": None}
        im = load_rgb(sp)
        cp, how = _match_clean(sp.name, im.size, cleans)
        if cp is None:
            vp, ham, vname = _visual_match(im, cleans, used)
            if vp is not None:
                cp, how = vp, f"visual_same_size:{vname}:ham={ham}"
                rec["ham"] = ham
        cim = load_rgb(cp) if cp else None
        if cp:
            rec["clean"] = str(cp)
            rec["how"] = how
            used.add(Path(cp).name)
            if cim.size != im.size:
                rec["kind"] = "double"
                rec["lettered_size"] = list(im.size)
                rec["clean_size"] = list(cim.size)
            else:
                d = mad(im, cim)
                rec["mad"] = round(d, 3)
                if d < MAD_IDENTICAL:
                    rec["kind"] = "identical"
                elif has_overlay_captions(im, cim):
                    rec["kind"] = "overlay"
                    copy_file(sp, overlay_dir / sp.name)
                    copy_file(Path(cp), clean_ov / sp.name)
                else:
                    rec["kind"] = "sfx"
        else:
            gx0, gx1, gs = thin_v_gutter(im)
            gy0, gy1, gsy = thin_h_gutter(im)
            rec["gutter_v"] = [gx0, gx1, round(gs, 3)]
            rec["gutter_h"] = [gy0, gy1, round(gsy, 3)]
            w, h = im.size
            spread = w > h * 1.15 or h > w * 1.25
            if max(gs, gsy) >= GUTTER_MIN and spread:
                rec["kind"] = "double"
                rec["how"] = "unpaired_gutter"
            else:
                rec["kind"] = "unpaired"
                rec["how"] = rec["how"] or "passthrough_unpaired"
        files.append(rec)
        print(f"{rec['kind']:10} {sp.name} {rec.get('how','')}", flush=True)

    counts = dict(Counter(f["kind"] for f in files))
    manifest = {
        "src": str(src),
        "clean": str(clean),
        "omake": str(omake) if omake.exists() else None,
        "n": len(files),
        "counts": counts,
        "files": files,
    }
    dump_json(work / "manifest.json", manifest)
    print("MANIFEST", json.dumps({"n": len(files), "counts": counts}, ensure_ascii=False), flush=True)
    return 0


def _copy_passthrough(work: Path, dests: list[Path]) -> None:
    man = load_json(work / "manifest.json")
    src = Path(man["src"])
    for rec in man["files"]:
        if rec["kind"] not in ("sfx", "identical", "unpaired"):
            continue
        sp = src / rec["file"]
        if rec["kind"] == "unpaired" and rec.get("how") != "passthrough_unpaired":
            continue
        for dst in dests:
            if not dst:
                continue
            copy_file(sp, dst / rec["file"])


def _letter_env(work: Path, cloud: str) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["LETTER_SKIP_DONE"] = env.get("LETTER_SKIP_DONE") or "1"
    env["LETTER_LOG_DIR"] = str(work / "logs")
    env["LETTER_LOCATE_CACHE"] = str(work / "logs" / "locate_cache")
    env["LETTER_NAMES"] = str(engine_dir() / "names.json")
    if cloud == "gemini":
        env["LETTER_FORCE_CLOUD"] = "gemini"
        for k in ("LETTER_OPENAI_BASE", "LETTER_OPENAI_KEY", "LETTER_OPENAI_MODEL"):
            env.pop(k, None)
    elif cloud == "endpoint":
        env.pop("LETTER_FORCE_CLOUD", None)
    return env


def _run_letter_b(src: Path, clean: Path, dst: Path, lang: str, work: Path, cloud: str, only: list[str] | None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(engine_dir() / "letter_b.py"),
        "--src",
        str(src),
        "--clean",
        str(clean),
        "--dst",
        str(dst),
        "--lang",
        lang,
    ]
    if only:
        cmd.append("--only")
        cmd.extend(only)
    log = work / "run.log"
    env = _letter_env(work, cloud)
    print("RUN", " ".join(cmd), "cloud=" + cloud, flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n# " + " ".join(cmd) + "\n")
        p = subprocess.run(cmd, cwd=str(engine_dir()), env=env, stdout=f, stderr=subprocess.STDOUT)
    print("END lang=" + lang + " exit=" + str(p.returncode), flush=True)
    return p.returncode


def _dests(ns) -> dict[str, Path]:
    out = {}
    if getattr(ns, "dst", None):
        lang = getattr(ns, "lang", None) or "en"
        out[lang] = Path(ns.dst)
    if getattr(ns, "dst_en", None):
        out["en"] = Path(ns.dst_en)
    if getattr(ns, "dst_ja", None):
        out["ja"] = Path(ns.dst_ja)
    langs = getattr(ns, "langs", None)
    if langs:
        keep = {x.strip() for x in langs.split(",") if x.strip()}
        out = {k: v for k, v in out.items() if k in keep}
    return out


def cmd_run(ns) -> int:
    work = Path(ns.work)
    dests = _dests(ns)
    if not dests:
        print("need --dst or --dst-en/--dst-ja", file=sys.stderr)
        return 2
    _copy_passthrough(work, list(dests.values()))
    src = work / "lettered_overlay"
    clean = work / "clean_overlay"
    n = len(iter_images(src))
    print("overlay_staged", n, flush=True)
    if n == 0:
        return 0
    only = list(ns.only) if ns.only else None
    cloud = ns.cloud
    for lang, dst in dests.items():
        rc = _run_letter_b(src, clean, dst, lang, work, cloud, only)
        if rc != 0:
            return rc
    return 0


def _best_half_clean(crop: Image.Image, cleans, exclude: set[str]) -> tuple[Path | None, int, str]:
    ch = ahash(crop)
    cw, chh = crop.size
    best = None
    best_d = 10**9
    best_name = ""
    for name, path, size in cleans:
        if name in exclude:
            continue
        aw, ah = size
        if min(aw, cw) <= 0:
            continue
        d = 0
        with Image.open(path) as im:
            d = hamming(ch, ahash(im.convert("RGB")))
        if size == crop.size:
            d -= 8
        if abs(aw / ah - cw / chh) > 0.35:
            d += 12
        if abs(aw - cw) > 80 and abs(ah - chh) > 80:
            d += 6
        if d < best_d:
            best_d, best, best_name = d, path, name
    if best is None:
        return None, -1, ""
    return best, best_d, best_name


def cmd_split_doubles(ns) -> int:
    work = Path(ns.work)
    man = load_json(work / "manifest.json")
    src = Path(man["src"])
    clean = Path(man["clean"])
    omake = Path(man["omake"]) if man.get("omake") else None
    cleans = _index_cleans(clean, omake)
    doubles = [f for f in man["files"] if f["kind"] == "double"]
    root = work / "doubles"
    lettered_dir = root / "lettered"
    panels_l = root / "panels" / "lettered"
    panels_c = root / "panels" / "clean"
    erase_in = root / "erase_in"
    for d in (lettered_dir, panels_l, panels_c, erase_in):
        d.mkdir(parents=True, exist_ok=True)
    meta = {}
    erase = []
    for rec in doubles:
        name = rec["file"]
        im = load_rgb(src / name)
        w, h = im.size
        same = None
        if rec.get("clean"):
            try:
                same = load_rgb(Path(rec["clean"]))
            except Exception:
                same = None
        axis = choose_axis(im, same)
        if axis == "lr":
            g0, g1, gs = thin_v_gutter(im)
            a_box = (0, 0, g0, h)
            b_box = (g1, 0, w, h)
        else:
            g0, g1, gs = thin_h_gutter(im)
            a_box = (0, 0, w, g0)
            b_box = (0, g1, w, h)
        copy_file(src / name, lettered_dir / name)
        panels = {}
        exclude: set[str] = set()
        for key, box in (("A", a_box), ("B", b_box)):
            crop = im.crop(box)
            fn = Path(name).stem + "__" + key + ".png"
            crop.save(panels_l / fn)
            overlay = has_overlay_captions(crop, None)
            cp, ham, cname = _best_half_clean(crop, cleans, exclude)
            good = cp is not None and ham <= HAM_OK + 8 and abs(Image.open(cp).size[0] - crop.size[0]) < 120
            entry = {
                "file": fn,
                "box": list(box),
                "size": list(crop.size),
                "overlay": overlay,
                "clean": str(cp) if cp else None,
                "clean_name": cname,
                "ham": ham,
                "mode_b": False,
                "needs_erase": False,
            }
            if overlay and good:
                cim = load_rgb(cp)
                if cim.size != crop.size:
                    cim = cim.resize(crop.size, Image.Resampling.LANCZOS)
                cim.save(panels_c / fn)
                exclude.add(cname)
                entry["mode_b"] = True
            elif overlay and not good:
                crop.save(erase_in / fn)
                entry["needs_erase"] = True
                entry["mode_b"] = True
                erase.append({"file": fn, "parent": name, "size": list(crop.size), "box": list(box)})
            else:
                entry["mode_b"] = False
            panels[key] = entry
        meta[name] = {
            "mode": axis,
            "size": [w, h],
            "gutter": [axis, g0, g1],
            "gutter_score": round(gs, 3),
            "panels": panels,
        }
        print("double", name, axis, "gutter", g0, g1, "erase", [k for k, v in panels.items() if v["needs_erase"]], flush=True)
    dump_json(root / "meta.json", meta)
    dump_json(root / "erase_manifest.json", erase)
    print("DOUBLES", len(meta), "erase", len(erase), flush=True)
    if erase:
        print("STOP: erase these with image_edit, save to", root / "erase_out", flush=True)
    return 0


def cmd_fit_erase(ns) -> int:
    work = Path(ns.work)
    erase = load_json(work / "doubles" / "erase_manifest.json")
    src_dir = work / "doubles" / "erase_out"
    dst = work / "doubles" / "panels" / "clean"
    dst.mkdir(parents=True, exist_ok=True)
    missing = []
    for rec in erase:
        p = src_dir / rec["file"]
        if not p.exists():
            missing.append(rec["file"])
            continue
        want = tuple(rec["size"])
        im = load_rgb(p)
        if im.size != want:
            im = im.resize(want, Image.Resampling.LANCZOS)
        im.save(dst / rec["file"])
        print("fit", rec["file"], want, flush=True)
    if missing:
        print("missing erase_out", missing, file=sys.stderr)
        return 2
    return 0


def cmd_run_doubles(ns) -> int:
    work = Path(ns.work)
    meta = load_json(work / "doubles" / "meta.json")
    src = work / "doubles" / "panels" / "lettered"
    clean = work / "doubles" / "panels" / "clean"
    pair_l = work / "doubles" / "modeb_src"
    pair_c = work / "doubles" / "modeb_clean"
    if pair_l.exists():
        shutil.rmtree(pair_l)
    if pair_c.exists():
        shutil.rmtree(pair_c)
    pair_l.mkdir(parents=True)
    pair_c.mkdir(parents=True)
    n = 0
    for rec in meta.values():
        for p in rec["panels"].values():
            if not p.get("mode_b"):
                continue
            lp = src / p["file"]
            cp = clean / p["file"]
            if not lp.exists() or not cp.exists():
                print("skip half (need clean)", p["file"], flush=True)
                continue
            copy_file(lp, pair_l / p["file"])
            copy_file(cp, pair_c / p["file"])
            n += 1
    if n == 0:
        print("no double halves ready for Mode B", flush=True)
        return 0
    dests = _dests(ns)
    cloud = ns.cloud
    for lang in dests:
        out = work / "doubles" / f"out_{lang}"
        rc = _run_letter_b(pair_l, pair_c, out, lang, work / "doubles", cloud, None)
        if rc != 0:
            return rc
    return 0


def cmd_stitch(ns) -> int:
    work = Path(ns.work)
    meta = load_json(work / "doubles" / "meta.json")
    man = load_json(work / "manifest.json")
    src = Path(man["src"])
    dests = _dests(ns)
    for lang, dst in dests.items():
        panel_dst = work / "doubles" / f"out_{lang}"
        dst.mkdir(parents=True, exist_ok=True)
        for name, rec in meta.items():
            canvas = load_rgb(src / name)
            w, h = rec["size"]
            gutter = rec["gutter"]
            if len(gutter) == 3:
                axis, g0, g1 = gutter[0], int(gutter[1]), int(gutter[2])
            else:
                axis, g0, g1 = rec["mode"], int(gutter[0]), int(gutter[1])
            mode = rec["mode"]
            for key, p in rec["panels"].items():
                if not p.get("mode_b"):
                    continue
                outp = panel_dst / p["file"]
                if not outp.exists():
                    print("missing panel", lang, p["file"], flush=True)
                    continue
                out = load_rgb(outp)
                if mode == "lr":
                    box = (0, 0, g0, h) if key == "A" else (g1, 0, w, h)
                else:
                    box = (0, 0, w, g0) if key == "A" else (0, g1, w, h)
                tw, th = box[2] - box[0], box[3] - box[1]
                if out.size != (tw, th):
                    out = out.resize((tw, th), Image.Resampling.LANCZOS)
                canvas.paste(out, (box[0], box[1]))
            canvas.save(dst / name)
            print("stitched", lang, name, canvas.size, mode, flush=True)
    return 0


def cmd_verify(ns) -> int:
    work = Path(ns.work)
    man = load_json(work / "manifest.json")
    src = Path(man["src"])
    src_files = [p.name for p in iter_images(src)]
    dests = _dests(ns)
    overlay = {f["file"] for f in man["files"] if f["kind"] == "overlay"}
    problems = []
    report = {"n_src": len(src_files), "dests": {}, "overlay_still_lettered": {}, "missing": {}}
    for lang, dst in dests.items():
        have = {p.name for p in iter_images(dst)} if dst.exists() else set()
        miss = [n for n in src_files if n not in have]
        still = []
        for name in sorted(overlay):
            a = src / name
            b = dst / name
            if b.exists() and filecmp.cmp(a, b, shallow=False):
                still.append(name)
        summary = {}
        sp = work / "logs" / f"lettering_{lang}_summary.json"
        if sp.exists():
            summary = load_json(sp)
        report["dests"][lang] = {"n": len(have), "failed": summary.get("failed"), "translated": summary.get("translated")}
        report["missing"][lang] = miss
        report["overlay_still_lettered"][lang] = still
        if miss:
            problems.append(f"{lang} missing {len(miss)}")
        if still:
            problems.append(f"{lang} overlay still lettered {len(still)}")
        if summary.get("failed"):
            problems.append(f"{lang} failed={summary.get('failed')}")
        print(lang, "n", len(have), "/", len(src_files), "still", len(still), "fail", summary.get("failed"), flush=True)
    dump_json(work / "verify.json", report)
    if problems:
        print("VERIFY_FAIL", "; ".join(problems), flush=True)
        return 2
    print("VERIFY_OK", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pipeline.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup")
    s.add_argument("--base-url", required=True)
    s.add_argument("--api-key", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--test", action="store_true")
    s.add_argument("--also-dotenv", action="store_true")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("add-name")
    s.add_argument("--zh", nargs="+", required=True)
    s.add_argument("--en", required=True)
    s.add_argument("--ja", required=True)
    s.set_defaults(func=cmd_add_name)

    s = sub.add_parser("prepare")
    s.add_argument("--src", required=True)
    s.add_argument("--clean", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--omake", default=None)
    s.set_defaults(func=cmd_prepare)

    def add_run_flags(p):
        p.add_argument("--work", required=True)
        p.add_argument("--dst", default=None)
        p.add_argument("--dst-en", default=None)
        p.add_argument("--dst-ja", default=None)
        p.add_argument("--lang", choices=("en", "ja"), default=None)
        p.add_argument("--langs", default=None)
        p.add_argument("--cloud", choices=("auto", "endpoint", "gemini"), default="auto")
        p.add_argument("--only", nargs="*", default=None)

    s = sub.add_parser("run")
    add_run_flags(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("split-doubles")
    s.add_argument("--work", required=True)
    s.set_defaults(func=cmd_split_doubles)

    s = sub.add_parser("fit-erase")
    s.add_argument("--work", required=True)
    s.set_defaults(func=cmd_fit_erase)

    s = sub.add_parser("run-doubles")
    add_run_flags(s)
    s.set_defaults(func=cmd_run_doubles)

    s = sub.add_parser("stitch")
    add_run_flags(s)
    s.set_defaults(func=cmd_stitch)

    s = sub.add_parser("verify")
    add_run_flags(s)
    s.add_argument("--src", default=None)
    s.set_defaults(func=cmd_verify)
    return ap


def main() -> int:
    ns = build_parser().parse_args()
    return int(ns.func(ns) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
