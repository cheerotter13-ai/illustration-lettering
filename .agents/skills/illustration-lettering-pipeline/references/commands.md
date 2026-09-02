# pipeline.py commands

`python scripts/pipeline.py -h`

Engine dir is `../engine` relative to `scripts/`. Override with `LETTER_ENGINE`.

| Command | Purpose |
|---|---|
| `setup` | Write API base URL / key / model |
| `add-name` | Append a character to `engine/names.json` |
| `prepare` | Classify, recover pairs, stage overlay folders, copy passthrough |
| `run` | Mode B on staged overlay (`letter_b.py`) |
| `split-doubles` | Thin-gutter split + match halves |
| `fit-erase` | LANCZOS match erased halves to original size |
| `run-doubles` | Mode B on double-page halves |
| `stitch` | Paste halves back into dest |
| `verify` | Count / overlay-still-chinese / SUMMARY |

## prepare

```
--src --clean --work
[--omake DIR]   extra clean search (default: SRC/omake if present)
```

Writes `WORK/manifest.json`, `WORK/lettered_overlay/`, `WORK/clean_overlay/`.

Classes:

- `identical` — lettered ≈ clean (MAD low): copy lettered
- `sfx` — differs, but no horizontal dark/magenta/green caption band: copy lettered (图一)
- `overlay` — caption bands: Mode B (图二)
- `double` — size mismatch + dark gutter
- `unpaired` — no clean after fuzzy name + aHash

## run

```
--work --dst / --dst-en --dst-ja
--lang en|ja   or --langs en,ja
--cloud auto|endpoint|gemini
--only FILE [FILE ...]
--skip-done / --no-skip-done
```

Env set by the wrapper: `LETTER_SKIP_DONE`, `LETTER_LOG_DIR`, `LETTER_LOCATE_CACHE`, `PYTHONUNBUFFERED=1`. `--cloud gemini` also sets `LETTER_FORCE_CLOUD=gemini` and unsets `LETTER_OPENAI_*` in the child.

## verify

Fails (exit 2) if dest count ≠ src count, or any overlay dest is byte-identical to the lettered source.
