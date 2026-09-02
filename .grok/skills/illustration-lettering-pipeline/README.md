# illustration-lettering-pipeline

Grok Build + Antigravity skill: Chinese illustration overlays → English/Japanese on clean plates (Mode B).

Agents: read `SKILL.md`. Humans: set a key, then ask the agent to run `/illustration-lettering-pipeline`.

## One-time

Python 3.10+ with pip.

```text
python -m pip install -r engine/requirements-b.txt
python scripts/setup_config.py --base-url URL --api-key KEY --model MODEL --test
```

Example model: `google/gemini-3.7-flash` at `https://zenmux.ai/api/v1`.

## Then

Give the agent `--src` (lettered), `--clean` (unlettered), `--dst-en` / `--dst-ja`. It will classify 图一 SFX vs 图二 overlays, recover renamed plates, run Mode B, and split double-pages.

Do not put keys in this folder if you will commit it. `setup_config.py` writes OS app-data.
