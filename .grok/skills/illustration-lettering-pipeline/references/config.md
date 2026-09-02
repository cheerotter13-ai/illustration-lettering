# Config (API key + base URL)

Do not put secrets in the skill tree, git, or chat logs.

## What to set

The engine talks to an OpenAI-compatible `/v1/chat/completions` that **accepts images**.

| Field | Example | Notes |
|---|---|---|
| base URL | `https://zenmux.ai/api/v1` or `http://127.0.0.1:8080/v1` | With or without `/chat/completions` |
| API key | user-supplied | Bearer |
| model | `google/gemini-3.7-flash` | Must be vision-capable |

Proven on the Odette batch: ZenMux + `google/gemini-3.7-flash` for **both** locate and translate. Local Ollama is optional; if it is down, cloud translate is used.

## Write config (recommended)

```text
python scripts/setup_config.py --base-url URL --api-key KEY --model MODEL [--test]
```

Writes `%APPDATA%/illustration-lettering/config.json` on Windows, `~/Library/Application Support/illustration-lettering/config.json` on macOS, `~/.config/illustration-lettering/config.json` on Linux. chmod 600 when possible.

`--test` hits `/v1/models` then a tiny chat ping. It prints ok/fail, never the key.

## Environment overrides (also fine)

```
LETTER_OPENAI_BASE
LETTER_OPENAI_KEY
LETTER_OPENAI_MODEL
```

Gemini-only (no OpenAI-compat):

```
GEMINI_API_KEY          # Google AI Studio
ZENMUX_API_KEY          # https://zenmux.ai
VISION_MODEL            # default google/gemini-3.7-flash
LETTER_GEMINI_MODEL     # default gemini-3.7-flash
```

Optional `engine/.env` (python-dotenv). Copy `env.example`. Never commit it.

## `--cloud` on run

| Value | Behavior |
|---|---|
| `auto` (default) | Use the configured OpenAI-compat endpoint if set; else Gemini/ZenMux |
| `endpoint` | Require LETTER_OPENAI_* / config.json |
| `gemini` | Ignore GUI/config OpenAI-compat (clears `LETTER_OPENAI_*` for that process). Use when the GUI saved grok-4.6 and it would hijack CLI |

Odette production used `--cloud gemini` because the GUI config pointed at grok-4.6.

## Fonts

Pillow typesets with OS fonts (Windows: `msyh.ttc` / `meiryo`; macOS: PingFang / Hiragino). No extra font install for CJK if those exist.
