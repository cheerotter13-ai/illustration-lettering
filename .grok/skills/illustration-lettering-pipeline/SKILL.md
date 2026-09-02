---
name: illustration-lettering-pipeline
description: >
  Use when the user has Chinese-lettered illustrations plus unlettered clean plates and wants
  English/Japanese Mode B typesetting, overlay vs SFX classification, unpaired recovery,
  double-page split/stitch, Imagine erase of captioned halves, 图一拟声 / 图二叠字, 无字底图,
  奥黛塔/正篇-style batches, or /illustration-lettering-pipeline. Mode B only: no Mode A, no LaMa, no CUDA.
---

# illustration-lettering-pipeline

Batch Mode B: classify → recover pairs → typeset overlay captions onto clean plates → recover double-pages.

This skill ships the programs. Do not reach for Mode A / LaMa / manga-image-translator.

## Hard rules

- Never write into `--src` or `--clean`. Dest is a new folder.
- Never send 图一 SFX-only or identical-to-clean plates to the vision model.
- Mode B only. No CUDA, no LaMa, no Mode A.
- Pairing uses the **full filename** (`1 (1).png` stays `1 (1).png`). Do not split on spaces.
- Invoke Python directly. Do not pass Chinese paths through PowerShell command strings.
- Do not print API keys. Do not commit `.env` or `config.json`.
- Smoke one overlay file before the full batch.

## Layout

Skill root = this directory (`SKILL.md` lives here).

| Path | Role |
|---|---|
| `engine/` | `letter_b.py` + `letter.py` + `vision_locate.py` + `user_config.py` + `names.json` |
| `scripts/pipeline.py` | One CLI for the whole run |
| `references/config.md` | API key / base URL |
| `references/imagine-erase.md` | Missing-clean-half erase |
| `references/commands.md` | Flags and env |

## Inputs the user must give

1. `--src` lettered folder (Chinese overlays / SFX already on the art)
2. `--clean` unlettered plates (same art, captions stripped). Extra files (素材 / omake) are OK.
3. `--dst-en` and/or `--dst-ja`
4. Endpoint: OpenAI-compat **base URL + API key + vision model**, or Gemini/ZenMux keys. See [references/config.md](references/config.md).

Optional: character names (`engine/names.json` or `--add-name`).

## First-time setup (once per machine)

```text
python <SKILL>/scripts/setup_config.py --base-url URL --api-key KEY --model MODEL
python -m pip install -r <SKILL>/engine/requirements-b.txt
```

That writes OS app-data `illustration-lettering/config.json` (not the git tree). After this, default runs use that endpoint.

If the GUI config still points at a slow chat model (e.g. grok-4.6), pass `--cloud gemini` and set `ZENMUX_API_KEY` or `GEMINI_API_KEY`. Proven locate+translate model: `google/gemini-3.7-flash` via ZenMux.

## Agent procedure

Ask only for missing paths / langs / names. Then:

1. **Work dir** — create a new folder, never reuse dest as work.
2. **Names** — if the series has a proper name, `pipeline.py add-name --zh 奥黛塔 --en Odette --ja オデット`.
3. **Classify + recover + stage**

```text
python <SKILL>/scripts/pipeline.py prepare --src SRC --clean CLEAN --work WORK
```

Reads `WORK/manifest.json`. Classes: `overlay` (Gemini), `sfx` (copy lettered), `identical` (copy lettered), `double` (split later), `unpaired`.

4. **Copy passthrough** into dest (`sfx` + `identical` + unpaired with no captions).
5. **Smoke** one overlay:

```text
python <SKILL>/scripts/pipeline.py run --work WORK --dst DST --lang en --cloud auto --only FILENAME
```

Check dest ≠ lettered source and captions are EN/JA. If locate used the wrong model, stop and fix config.

6. **Full overlay run** — EN then JA (JA reuses locate cache):

```text
python <SKILL>/scripts/pipeline.py run --work WORK --dst-en EN --dst-ja JA --langs en,ja --cloud auto
```

`LETTER_SKIP_DONE=1` is on. Resume is safe. Redirect stdout to `WORK/run.log`.

7. **Doubles** — `pipeline.py split-doubles` then Mode B each half that has a clean plate. Halves with no clean: [references/imagine-erase.md](references/imagine-erase.md). Then `pipeline.py stitch`.
8. **Verify**

```text
python <SKILL>/scripts/pipeline.py verify --src SRC --work WORK --dst-en EN --dst-ja JA
```

Required: dest file count = src count; overlay dests are not byte-identical to lettered src; SUMMARY `failed=0`. Write a short report next to WORK.

## Imagine / image_edit

Only for captioned halves that have **no** clean plate. Grok: `image_edit`. Antigravity: Gemini image edit if available. Never LaMa. Upscale erased output to the half’s original pixels with LANCZOS before Mode B. Prompt in `references/imagine-erase.md`.

## Runtime notes

- Locate cache: `WORK/logs/locate_cache` (path+size+mtime). JA should not re-upload images.
- Expected throughput with Gemini 3.7 Flash: ~3–8s locate + translate per new overlay; JA much faster on cache. Hundreds of overlays take hours — run in background, do not poll every log line.
- Windows: `sys.executable` + script path. Chinese paths belong in Python argv, not in `pwsh -Command`.
