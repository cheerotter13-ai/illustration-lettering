#!/usr/bin/env python3
"""Write OpenAI-compat settings for the bundled lettering engine. Never prints the key."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import engine_dir  # noqa: E402

sys.path.insert(0, str(engine_dir()))
import user_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Save illustration-lettering API settings")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--also-dotenv", action="store_true", help="Also write engine/.env")
    ns = ap.parse_args()
    data = user_config.load()
    data["base_url"] = ns.base_url.strip()
    data["api_key"] = ns.api_key.strip()
    data["model"] = ns.model.strip()
    data["mode"] = "b"
    path = user_config.save(data)
    print("wrote", path)
    if ns.also_dotenv:
        envp = engine_dir() / ".env"
        envp.write_text(
            f"LETTER_OPENAI_BASE={data['base_url']}\n"
            f"LETTER_OPENAI_KEY={data['api_key']}\n"
            f"LETTER_OPENAI_MODEL={data['model']}\n",
            encoding="utf-8",
        )
        print("wrote", envp)
    if ns.test:
        rec = user_config.test_connection(data["base_url"], data["api_key"], data["model"])
        print("test", "ok" if rec.get("ok") else "FAIL", rec.get("via") or rec.get("error", "")[:200])
        return 0 if rec.get("ok") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
