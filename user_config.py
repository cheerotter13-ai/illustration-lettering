"""User-facing OpenAI-compat settings. Stored in the OS app-data dir, never in the install tree."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_ID = "illustration-lettering"

DEFAULTS = {
    "base_url": "",
    "api_key": "",
    "model": "",
    "lang": "en",
    "mode": "b",
    "src": "",
    "dst": "",
    "clean": "",
    "mit_root": "",
}


def resource_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def config_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / APP_ID / "config.json"


def load() -> dict:
    path = config_path()
    data = dict(DEFAULTS)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k in DEFAULTS:
                    if k in raw and raw[k] is not None:
                        data[k] = raw[k]
        except Exception:
            pass
    return data


def save(data: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(DEFAULTS)
    out.update({k: data.get(k, DEFAULTS[k]) for k in DEFAULTS})
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return path


def chat_completions_url(base: str) -> str:
    b = (base or "").strip().rstrip("/")
    if not b:
        return ""
    if b.endswith("/chat/completions"):
        return b
    if b.endswith("/v1"):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


def apply_to_env(data: dict | None = None) -> None:
    cfg = data if data is not None else load()
    base = (cfg.get("base_url") or "").strip()
    key = (cfg.get("api_key") or "").strip()
    model = (cfg.get("model") or "").strip()
    if base:
        os.environ["LETTER_OPENAI_BASE"] = base
    if key:
        os.environ["LETTER_OPENAI_KEY"] = key
    if model:
        os.environ["LETTER_OPENAI_MODEL"] = model
    mit = (cfg.get("mit_root") or "").strip()
    if mit:
        os.environ["MIT_ROOT"] = mit


def test_connection(base_url: str, api_key: str, model: str) -> dict:
    import json
    import urllib.error
    import urllib.request

    url = chat_completions_url(base_url)
    if not url or not (model or "").strip():
        return {"ok": False, "error": "需要 Base URL 和模型名"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    models = url.replace("/chat/completions", "/models")
    try:
        req = urllib.request.Request(models, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read(400)
        return {"ok": True, "via": "models", "url": models}
    except Exception as first:
        try:
            body = {
                "model": model.strip(),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
            }
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"), headers=headers
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read(80)
            return {"ok": True, "via": "chat", "url": url}
        except Exception as second:
            return {"ok": False, "error": str(second)[:400], "fallback_from": str(first)[:200]}


def user_endpoint() -> tuple[str, str, str]:
    """Return (chat_url, key, model) from env after apply_to_env."""
    base = (os.environ.get("LETTER_OPENAI_BASE") or "").strip()
    key = (os.environ.get("LETTER_OPENAI_KEY") or "").strip()
    model = (os.environ.get("LETTER_OPENAI_MODEL") or "").strip()
    return chat_completions_url(base), key, model
