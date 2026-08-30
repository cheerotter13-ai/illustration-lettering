---
name: illustration-lettering-b
description: >
  Mode B illustration lettering: translate Chinese overlay/comic captions to English or Japanese
  onto a user-supplied unlettered clean plate. No CUDA, no LaMa. Works on Mac, Windows, and Linux.
  Use when the user has lettered images plus same-filename clean plates, wants platform-independent
  typesetting, says Mode B, clean plate, Mac lettering, or /illustration-lettering-b.
---

# illustration-lettering Mode B (portable)

Use **only Mode B**. Do not load manga-image-translator, ComicTextDetector, or CUDA.

Repo (this skill lives with the code): the `illustration-lettering` checkout that contains `letter.py`, `letter_b.py`, and `vision_locate.py`.

## Required inputs

- `--src` folder of **lettered** images (Chinese on the art)
- `--clean` folder of **unlettered plates**, **same filenames**, same pixel size
- `--dst` output folder
- `--lang en` or `ja`
- Cloud VL key: `GEMINI_API_KEY` / `GOOGLE_API_KEY` or `ZENMUX_API_KEY` in env or `.env` next to `letter.py`

Missing a clean plate → stop. Do not fall back to Mode A / LaMa.

## Run

```bash
python3 letter_b.py --src LETTERED --clean CLEAN --dst OUT --lang en --retranslate
```

`letter_b.py` is `letter.py --mode-b`. It never loads LaMa.

Translation default is local Ollama `qwen3.8:27b-uncensored`, then Gemini text if Ollama is down. Force with `--translate local` or `--translate gemini`.

Locate uses Gemini **API** (`GEMINI_API_KEY` / `ZENMUX_API_KEY`), not Gemini App chat and not Grok Heavy. Google AI Pro only raises AI Studio daily quota for that same account; it is not unlimited. Cache `logs/locate_cache/` (or `LETTER_LOCATE_CACHE`) by path+size+mtime — unchanged files are not re-uploaded.

## Agent steps

1. Confirm lettered + clean folders exist; spot-check one filename exists in both and dimensions match.
2. Confirm a Gemini/ZenMux key is set. Do not print secret values. Tell the user locate hits cache when files are unchanged; Heavy subscription does not pay xAI API.
3. `python3 -c "import cv2,numpy,PIL"` ; if missing, `pip install -r requirements-b.txt` (no PyTorch).
4. Smoke one file: `python3 letter_b.py --src ... --clean ... --dst ... --lang en --only one.jpg`
5. If that SUMMARY is ok, run the rest. Read `logs/qingge_<lang>.jsonl`. Report failed files. Do not invent missing plates.

## Do not

- Run Mode A (`letter.py` without `--clean` / `--mode-b`)
- Call cloud image-edit to erase NSFW
- Commit `.env`, API keys, or NSFW gold images
