#!/usr/bin/env python3
"""Local web UI for illustration-lettering. Bind 127.0.0.1 only."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from user_config import apply_to_env, load, save, test_connection

WEB = HERE / "web"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IS_MAC = sys.platform == "darwin"
HOST = "127.0.0.1"
PORT = int(os.environ.get("LETTER_WEB_PORT", "8765"))

_lock = threading.Lock()
_job = {
    "running": False,
    "log": [],
    "error": None,
    "dst": "",
    "done": False,
}
_stop_file: Path | None = None
_proc: subprocess.Popen | None = None
_GPU_PY_CANDIDATES = (
    os.environ.get("LETTER_PYTHON") or "",
    r"E:\SD\ComfyUI-aki-v3\python\python.exe",
    sys.executable,
)


def _letter_python(mode: str) -> str:
    override = (os.environ.get("LETTER_PYTHON") or "").strip()
    if override:
        return override
    if mode != "a":
        return sys.executable
    for cand in _GPU_PY_CANDIDATES:
        p = (cand or "").strip()
        if p and Path(p).is_file() and _cuda_ok(p):
            return p
    return sys.executable


def _cuda_ok(py: str) -> bool:
    try:
        r = subprocess.run(
            [py, "-c", "import torch; print('1' if torch.cuda.is_available() else '0')"],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except Exception:
        return False
    return (r.stdout or "").strip().splitlines()[-1:] == ["1"]


def _mit_ok(mit_root: str) -> bool:
    root = Path(mit_root or "").expanduser()
    if not root.is_dir():
        return False
    return (root / "manga_translator").is_dir() or (root / "manga_translator" / "inpainting").exists()


def probe_mode_a(mit_root: str = "") -> dict:
    if IS_MAC:
        return {
            "mode_a_ok": False,
            "cuda": False,
            "mit": False,
            "letter_python": sys.executable,
            "reason": "Mac 没有 NVIDIA CUDA，请用 Mode B（无字底图）。",
        }
    cfg = load()
    mit = (mit_root or cfg.get("mit_root") or os.environ.get("MIT_ROOT") or "").strip()
    py = _letter_python("a")
    cuda = _cuda_ok(py)
    mit_ok = _mit_ok(mit)
    if cuda and mit_ok:
        reason = "已检测到 CUDA 与 manga-image-translator，可以用 Mode A 擦字。"
    elif not cuda and not mit_ok:
        reason = "本机没检测到 NVIDIA CUDA，也没找到 manga-image-translator。请用 Mode B：准备同名无字底图即可。Mode A 需要自己安装 MIT + CUDA 版 PyTorch。"
    elif not cuda:
        reason = "找到了 MIT 目录，但当前 Python 没有 CUDA。请用带 GPU 的解释器（设置 LETTER_PYTHON），或改用 Mode B。"
    else:
        reason = "有 CUDA，但还没填写有效的 manga-image-translator 根目录。可先用 Mode B；要用 Mode A 请填 MIT 路径。"
    return {
        "mode_a_ok": bool(cuda and mit_ok),
        "cuda": cuda,
        "mit": mit_ok,
        "letter_python": py,
        "reason": reason,
    }


class SettingsIn(BaseModel):
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    lang: str = "en"
    mode: str = "b"
    src: str = ""
    dst: str = ""
    clean: str = ""
    mit_root: str = ""


class TestIn(BaseModel):
    base_url: str
    api_key: str = ""
    model: str


class RunIn(SettingsIn):
    pass


class PathIn(BaseModel):
    path: str = Field(min_length=1)


app = FastAPI(title="Illustration Lettering", version="0.10.0")
_assets = WEB / "assets"
if _assets.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


def _public_settings(cfg: dict) -> dict:
    key = cfg.get("api_key") or ""
    probe = probe_mode_a(cfg.get("mit_root") or "")
    return {
        **cfg,
        "api_key": key,
        "api_key_set": bool(key),
        "platform": sys.platform,
        **probe,
    }


def _count_images(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    names = [
        p.name
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    names.sort()
    return names


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def pair_report(src: Path, clean: Path) -> dict:
    """Mode B pairing: same filename, same pixel size. Do not start if names/sizes disagree."""
    src_names = _count_images(src)
    clean_names = _count_images(clean)
    ss, cs = set(src_names), set(clean_names)
    missing_clean = sorted(ss - cs)
    extra_clean = sorted(cs - ss)
    matched = sorted(ss & cs)
    size_mismatch = []
    for name in matched:
        a = _image_size(src / name)
        b = _image_size(clean / name)
        if a is None or b is None or a != b:
            size_mismatch.append(
                {
                    "file": name,
                    "src": list(a) if a else None,
                    "clean": list(b) if b else None,
                }
            )
    blocking = bool(missing_clean or size_mismatch or not matched)
    return {
        "ok": not blocking,
        "src_count": len(src_names),
        "clean_count": len(clean_names),
        "matched": len(matched) - len(size_mismatch),
        "missing_clean": missing_clean,
        "extra_clean": extra_clean,
        "size_mismatch": size_mismatch,
        "blocking": blocking,
    }


class PairIn(BaseModel):
    src: str
    clean: str


def _append(line: str) -> None:
    with _lock:
        _job["log"].append(line.rstrip("\n"))
        if len(_job["log"]) > 4000:
            _job["log"] = _job["log"][-3000:]


@app.get("/", response_class=HTMLResponse)
def index():
    html = WEB / "index.html"
    if not html.exists():
        raise HTTPException(500, "web/index.html missing")
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/meta")
def meta(mit_root: str = ""):
    probe = probe_mode_a(mit_root)
    return {
        "version": (HERE / "VERSION").read_text(encoding="utf-8").strip()
        if (HERE / "VERSION").exists()
        else "0.10.0",
        "platform": sys.platform,
        "port": PORT,
        **probe,
    }


@app.get("/api/settings")
def get_settings():
    return _public_settings(load())


@app.post("/api/settings")
def post_settings(body: SettingsIn):
    data = body.model_dump()
    if IS_MAC:
        data["mode"] = "b"
    path = save(data)
    apply_to_env(data)
    return {"ok": True, "path": str(path)}


@app.post("/api/test")
def post_test(body: TestIn):
    return test_connection(body.base_url, body.api_key, body.model)


@app.post("/api/pairs")
def inspect_pairs(body: PairIn):
    src = Path(body.src).expanduser()
    clean = Path(body.clean).expanduser()
    if not src.is_dir() or not clean.is_dir():
        raise HTTPException(400, "源目录或底图目录不存在")
    return pair_report(src, clean)


@app.post("/api/folder")
def inspect_folder(body: PathIn):
    p = Path(body.path).expanduser()
    if not p.exists():
        raise HTTPException(400, "路径不存在")
    if not p.is_dir():
        raise HTTPException(400, "不是文件夹")
    names = _count_images(p)
    return {"ok": True, "path": str(p.resolve()), "count": len(names), "sample": names[:12]}


@app.get("/api/job")
def job_status():
    with _lock:
        dst = _job["dst"]
        logs = list(_job["log"][-200:])
        running = _job["running"]
        done = _job["done"]
        err = _job["error"]
    previews = []
    if dst:
        folder = Path(dst)
        if folder.is_dir():
            for name in _count_images(folder)[-24:]:
                previews.append(name)
    return {
        "running": running,
        "done": done,
        "error": err,
        "log": logs,
        "dst": dst,
        "previews": previews,
    }


@app.get("/api/preview/{name}")
def preview(name: str):
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, "bad name")
    with _lock:
        dst = _job["dst"]
    if not dst:
        raise HTTPException(404, "no job")
    path = Path(dst) / name
    if not path.is_file():
        raise HTTPException(404, "missing")
    return FileResponse(path)


@app.post("/api/stop")
def stop_job():
    global _stop_file, _proc
    if _stop_file:
        _stop_file.write_text("1", encoding="utf-8")
    if _proc and _proc.poll() is None:
        try:
            _proc.terminate()
        except Exception:
            pass
    return {"ok": True}


@app.post("/api/run")
def start_job(body: RunIn):
    with _lock:
        if _job["running"]:
            raise HTTPException(409, "已有任务在跑")
    data = body.model_dump()
    if not data["base_url"].strip() or not data["model"].strip():
        raise HTTPException(400, "请填写 Base URL 和模型名")
    if not data["src"].strip() or not data["dst"].strip():
        raise HTTPException(400, "请填写带字图目录和输出目录")
    src = Path(data["src"]).expanduser()
    dst = Path(data["dst"]).expanduser()
    if not src.is_dir():
        raise HTTPException(400, "带字图目录不存在")
    if not _count_images(src):
        raise HTTPException(400, "带字图目录里没有图片")
    mode = data["mode"] if not IS_MAC else "b"
    if mode == "b":
        clean = Path(data["clean"]).expanduser() if data["clean"] else None
        if not clean or not clean.is_dir():
            raise HTTPException(400, "Mode B 需要无字底图目录")
        report = pair_report(src, clean)
        if report["blocking"]:
            raise HTTPException(
                400,
                {
                    "message": "带字图和无字底图对不上。请按下面名单改文件名或重新导出同尺寸底图后再跑。多出来的底图不会使用。",
                    **report,
                },
            )
    else:
        probe = probe_mode_a(data.get("mit_root") or "")
        if not probe["mode_a_ok"]:
            raise HTTPException(400, probe["reason"])
    dst.mkdir(parents=True, exist_ok=True)
    save(data)
    apply_to_env(data)
    global _stop_file
    _stop_file = Path(tempfile.gettempdir()) / "illustration-lettering.stop"
    if _stop_file.exists():
        _stop_file.unlink()
    os.environ["LETTER_STOP_FILE"] = str(_stop_file)
    with _lock:
        _job.update(
            running=True,
            done=False,
            error=None,
            log=["=== 开始 ==="],
            dst=str(dst.resolve()),
        )
    threading.Thread(target=_run_letter, args=(data, mode), daemon=True).start()
    return {"ok": True}


def _run_letter(data: dict, mode: str) -> None:
    global _proc
    py = _letter_python(mode)
    cmd = [
        py,
        str(HERE / "letter.py"),
        "--src",
        str(Path(data["src"]).expanduser()),
        "--dst",
        str(Path(data["dst"]).expanduser()),
        "--lang",
        data.get("lang") or "en",
        "--translate",
        "openai",
    ]
    if mode == "b":
        cmd.extend(["--mode-b", "--clean", str(Path(data["clean"]).expanduser())])
    if data.get("mit_root"):
        cmd.extend(["--mit-root", str(Path(data["mit_root"]).expanduser())])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    _append(f"python={py}")
    try:
        _proc = subprocess.Popen(
            cmd,
            cwd=str(HERE),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert _proc.stdout is not None
        for line in _proc.stdout:
            _append(line.rstrip())
        code = _proc.wait()
        if code != 0:
            _append(f"退出码 {code}")
            with _lock:
                _job["error"] = f"exit {code}"
        else:
            _append("=== 完成 ===")
    except Exception as e:
        _append(f"错误 {e}")
        with _lock:
            _job["error"] = str(e)
    finally:
        _proc = None
        with _lock:
            _job["running"] = False
            _job["done"] = True


@app.get("/api/log/stream")
def log_stream():
    def gen():
        i = 0
        while True:
            with _lock:
                lines = _job["log"]
                running = _job["running"]
                chunk = lines[i:]
                i = len(lines)
            for line in chunk:
                yield f"data: {line}\n\n"
            if not running and i > 0:
                yield "data: [END]\n\n"
                break
            time.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    url = f"http://{HOST}:{PORT}/"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Illustration Lettering 本机网页：{url}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
