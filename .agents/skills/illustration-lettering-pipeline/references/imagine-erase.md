# Imagine erase (missing clean halves only)

Use this when a double-page half has captions and **no** matching clean plate.

Do **not** use LaMa, Mode A, or CUDA. Do **not** erase SFX-only halves that will be copied as-is. Do **not** send the whole double-page if only one half is missing.

## Export

`pipeline.py split-doubles` writes captioned halves that need erase to:

```
WORK/doubles/erase_in/<stem>__A.png
WORK/doubles/erase_in/<stem>__B.png
```

and lists them in `WORK/doubles/erase_manifest.json`.

## Erase

**Grok Build:** `image_edit` on each file in `erase_in`.

Prompt (keep this tight):

```
Remove every Chinese caption, overlay subtitle, and typeset dialogue from this illustration. Do not remove sound-effect graffiti that is drawn as part of the art unless it is a typeset caption. Do not change the character, pose, clothing, camera, or background. Fill the former text with matching background. No new text.
```

Save outputs under `WORK/doubles/erase_out/` with the **same filename**.

**Antigravity:** use the host's image-edit / Gemini image-edit tool the same way. If the host cannot edit images, stop and ask the user for a clean plate; do not invent LaMa.

NSFW may be refused (`imagine:content-moderated`). Do not retry with jailbreak wording. Copy the lettered half into dest (passthrough) and record it in the report.

## Size

Erased images often come back smaller. Before Mode B:

```text
python scripts/pipeline.py fit-erase --work WORK
```

LANCZOS-upscales each `erase_out` file to the pixel size recorded in `erase_manifest.json`, writes `WORK/doubles/panels/clean/`.

## Then Mode B + stitch

```text
python scripts/pipeline.py run-doubles --work WORK --dst-en EN --dst-ja JA --langs en,ja --cloud auto
python scripts/pipeline.py stitch --work WORK --dst-en EN --dst-ja JA
```

Stitch pastes each finished half into a canvas copied from the original lettered double-page and keeps the original gutter pixels. Unpack `gutter` as `(axis, g0, g1)` if present; `axis` is `"lr"` or `"tb"`, not a mode string mixed into the first slot.
