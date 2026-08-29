"""Cloud VL locate: boxes + Chinese text. No image generation.

Provider order: Google AI Studio Gemini 3.7 Flash, then ZenMux same model.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ZENMUX = "https://zenmux.ai/api/v1/chat/completions"
GOOGLE_GEN = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.7-flash:generateContent"
)
ZENMUX_MODEL = os.environ.get("VISION_MODEL", "google/gemini-3.7-flash")
GOOGLE_MODEL = "gemini-3.7-flash"
HERE = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
except Exception:
    pass
KEY_FILE = HERE / ".gemini_ai_studio_key"

PROMPT = """You are locating every caption / overlay / SFX lettered onto an illustration or comic page.
Return ONLY a JSON array, no markdown. Each element:
{"text":"<exact text as written>","type":"bubble|overlay|sfx","box":[ymin,xmin,ymax,xmax]}
- box is normalized 0-1000: top, left, bottom, right of that text (not the whole page).
- Include Chinese overlays, short SFX (嗯？ 是...... 住手 呜呜 颤抖 哒哒), and already-English timestamps/titles lettered on the art (e.g. "2 hours later").
- Do not merge two speakers into one item.
- Do not invent text that is not visible.
- Skip in-world signage that is part of the scene (neon shop signs, posters on walls), not a caption.
"""

# Printed in batch logs; real route is google-ai-studio then zenmux.
MODEL = os.environ.get("VISION_MODEL", "google/gemini-3.7-flash")


def google_api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_STUDIO_API_KEY"):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return ""


def zenmux_api_key() -> str:
    return (os.environ.get("ZENMUX_API_KEY") or "").strip()


def _parse_list(raw: str) -> list | None:
    raw = (raw or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    m = re.search(r"\[.*\]", raw, flags=re.S)
    blob = m.group(0) if m else raw
    for loader in (json.loads, ast.literal_eval):
        try:
            out = loader(blob)
        except Exception:
            continue
        if isinstance(out, list):
            return out
    return None


def _google_text(payload: dict) -> str:
    fb = payload.get("promptFeedback") or {}
    if fb.get("blockReason"):
        raise RuntimeError("google blocked:" + str(fb.get("blockReason")))
    cands = payload.get("candidates") or []
    if not cands:
        raise RuntimeError("google empty candidates:" + json.dumps(payload)[:400])
    finish = str(cands[0].get("finishReason") or "")
    if finish in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"):
        raise RuntimeError("google finish:" + finish)
    parts = ((cands[0].get("content") or {}).get("parts")) or []
    bits = [str(p.get("text") or "") for p in parts if p.get("text")]
    text = "\n".join(bits).strip()
    if not text:
        raise RuntimeError("google empty text")
    return text


def _google_complete(system: str | None, user: str, image_b64: str | None, max_tokens: int) -> str:
    key = google_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing")
    parts: list[dict] = [{"text": user}]
    if image_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    body: dict = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    url = GOOGLE_GEN + "?key=" + key
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"google HTTP {e.code}: {err}") from e
    return _google_text(payload)


def _zenmux_complete(system: str | None, user: str, image_b64: str | None, max_tokens: int) -> str:
    key = zenmux_api_key()
    if not key:
        raise RuntimeError("ZENMUX_API_KEY missing")
    content: list[dict] = [{"type": "text", "text": user}]
    if image_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}}
        )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content if image_b64 else user})
    body = {
        "model": ZENMUX_MODEL,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": messages,
    }
    req = urllib.request.Request(
        ZENMUX,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"zenmux HTTP {e.code}: {err}") from e
    return (payload.get("choices") or [{}])[0].get("message", {}).get("content") or ""


_google_skip = False
_google_overload = 0


def cloud_complete(
    *,
    system: str | None,
    user: str,
    image_b64: str | None = None,
    max_tokens: int = 2500,
) -> tuple[str, str]:
    """Gemini 3.7 Flash: Google AI Studio first, ZenMux fallback. Returns (text, via)."""
    global _google_skip, _google_overload
    last = None
    if (not _google_skip) and google_api_key():
        try:
            text = _google_complete(system, user, image_b64, max_tokens)
            _google_overload = 0
            return text, "google-ai-studio"
        except Exception as e:
            last = e
            print(f"cloud_complete google fail: {e}", flush=True)
            msg = str(e)
            if any(s in msg for s in ("HTTP 503", "HTTP 429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand")):
                _google_overload += 1
                if _google_overload >= 2:
                    _google_skip = True
                    print("cloud_complete skip google-ai-studio for this process (overload)", flush=True)
    if zenmux_api_key():
        try:
            return _zenmux_complete(system, user, image_b64, max_tokens), "zenmux"
        except Exception as e:
            last = e
            print(f"cloud_complete zenmux fail: {e}", flush=True)
    raise RuntimeError(f"cloud_complete failed: {last}")


def _items_from_parsed(got: list, img_w: int, img_h: int) -> list[dict]:
    items = []
    for el in got:
        if not isinstance(el, dict):
            continue
        text = str(el.get("text") or "").strip()
        box = el.get("box") or el.get("bbox")
        if not text or not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        ymin, xmin, ymax, xmax = [float(v) for v in box[:4]]
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        scale = 1000.0 if max(ymin, xmin, ymax, xmax) > 1.5 else 1.0
        x1 = int(max(0, xmin / scale * img_w))
        y1 = int(max(0, ymin / scale * img_h))
        x2 = int(min(img_w, xmax / scale * img_w))
        y2 = int(min(img_h, ymax / scale * img_h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        items.append(
            {
                "text": text,
                "box": (x1, y1, x2 - x1, y2 - y1),
                "kind": str(el.get("type") or "overlay"),
                "fg": None,
                "bg": None,
            }
        )
    return items


def _locate_cache_path(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(
        f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    ).hexdigest()
    d = Path(
        os.environ.get("LETTER_LOCATE_CACHE")
        or str(Path(__file__).resolve().parent / "logs" / "locate_cache")
    )
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def locate(path: Path, img_w: int, img_h: int) -> list[dict]:
    cache = _locate_cache_path(path)
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        items = []
        for it in data.get("items") or []:
            box = it.get("box")
            if not box:
                continue
            items.append(
                {
                    "text": it.get("text") or "",
                    "box": tuple(int(v) for v in box[:4]),
                    "kind": str(it.get("kind") or "overlay"),
                    "fg": None,
                    "bg": None,
                }
            )
        print(f"vision_locate cached {path.name} n={len(items)}", flush=True)
        return items
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    raw, via = cloud_complete(system=None, user=PROMPT, image_b64=b64, max_tokens=2500)
    print(f"vision_locate model={GOOGLE_MODEL} via={via} {path.name}", flush=True)
    got = _parse_list(raw) or []
    items = _items_from_parsed(got, img_w, img_h)
    serial = [
        {"text": it["text"], "box": list(it["box"]), "kind": it["kind"]} for it in items
    ]
    cache.write_text(
        json.dumps({"file": path.name, "w": img_w, "h": img_h, "via": via, "items": serial}, ensure_ascii=False),
        encoding="utf-8",
    )
    return items
