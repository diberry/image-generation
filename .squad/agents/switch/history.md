# Switch — History

## Project Context

- **Project:** image-generation — Python CLI tool using Stable Diffusion XL (SDXL) to generate blog post illustrations with a tropical magical-realism aesthetic.
- **Stack:** Python 3.10+, diffusers, transformers, torch, Pillow
- **Owner:** dfberry
- **Key files:** prompts/examples.md (master prompt library), generate.py (consumes prompts)
- **Joined:** 2026-03-26

## Learnings

### Prompt Library Audit (2026-03-26)

**Structure:** `prompts/examples.md` is a single flat file containing a 4-line style guide, 5 original abstract prompts (metaphorical, heavily-modified), and 5 vacation-theme prompts (concrete scenes, lighter modifiers). Two distinct prompt eras coexist with overlapping numbering (01-05 each).

**Key findings from first audit:**

1. **No negative prompts anywhere.** `generate.py` doesn't accept `--negative-prompt`, and no prompt includes negative guidance. SDXL benefits significantly from negatives like "blurry, text, watermark, deformed" to suppress artifacts.

2. **"no text" constraint inconsistency.** All 5 original prompts end with "no text" (prevents SDXL text artifacts). All 5 vacation prompts omit it — likely producing unwanted text in generated images.

3. **Style anchor drift.** Original prompts use "Latin American folk art style, magical realism illustration" (two anchors). Vacation prompts drop "magical realism" and alternate between "Latin American folk art illustration" and just "Folk art illustration". This inconsistency weakens aesthetic coherence.

4. **No prompt template/composition system.** Every prompt is bespoke. Style qualifiers, palette hints, and constraints are copy-pasted inline rather than factored into a composable template (e.g., `{scene} + {style_anchor} + {palette} + {constraints}`).

5. **Style guide is minimal (4 lines).** Missing: prompt structure guidance, negative prompt strategy, what to avoid, prompt-parameter interactions, seed strategy, troubleshooting, SDXL-specific tips.

6. **Refiner mismatch.** Parameter table says "blog quality" = 40 steps + refine. But all vacation prompts and `batch_generate()` hardcode `refine=False`. The original prompts' example uses `--refine`.

7. **Prompt duplication.** Vacation prompts exist in three places: `prompts/examples.md`, `generate_blog_images.sh`, and inline in `batch_generate()` defaults. No single source of truth.

**File paths:**
- `prompts/examples.md` — master prompt library (single file, ~123 lines)
- `generate.py` — CLI; `prompt` is positional arg, no `negative_prompt` support
- `generate_blog_images.sh` — batch script with duplicated prompts
- `generate.py:batch_generate()` — hardcodes steps=40, guidance=7.5, refine=False

**Architecture decisions to propose:**
- Add negative prompt support to pipeline and CLI
- Factor style system into composable template
- Add "no text" to all prompts or make it a default constraint
- Standardize style anchors across all prompts

### Negative Prompt Strategy Documentation (issue #3, 2026-03-26)

**What was done:** Added a comprehensive `## Negative Prompts` section to `prompts/examples.md` as part of issue #3 (negative prompt support). Also updated the Style Guide with a cross-reference to the new section.

**Section contents:**
1. **Default negative prompt** (11 terms): `blurry, bad quality, worst quality, low resolution, text, watermark, signature, deformed, ugly, duplicate, morbid`
2. **Term-by-term rationale table** — explains what each term prevents and why it matters for our tropical folk art aesthetic
3. **Customization guidelines** — when to add scene-specific negatives (people, architecture, fine patterns, photorealistic drift, dark mood) vs. using the default as-is
4. **SDXL-specific tips** — what works (quality pairs, style negation, double-blocking text) and what backfires (over-long prompts, negating colors, negating specific objects, contradicting positive tokens, SD 1.5 terms)
5. **Usage example** — shows the recommended `--negative-prompt` CLI invocation for when Trinity wires up the flag

**Key design choices:**
- Kept the default at 11 terms — SDXL's sweet spot before token blending kicks in
- `bad quality` + `worst quality` as a pair is more effective than either alone on SDXL (different training captions)
- `text` in negative + "no text" in positive = double-blocking, the most reliable text suppression for SDXL
- Explicitly warned against negating colors and specific objects — common SDXL pitfalls
- Warned against copying SD 1.5 negative prompts (different text encoder)
- `morbid` included specifically because our prompts use words like "ancient," "faded," "translucent forms" that can drift dark

**Dependency:** `generate.py` does not yet accept `--negative-prompt`. This documentation is ready for Trinity to reference when implementing the CLI flag.
