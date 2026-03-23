# Example Prompts

Curated prompts in the tropical magical-realism style for the dfberry blog post visuals.

## Style Guide
- **Palette:** deep magenta, teal, emerald green, warm gold, coral, amber
- **Aesthetic:** Latin American folk art, magical realism illustration
- **Mood:** Warmth, community, luminous energy
- **Resolution:** 1024×1024 (SDXL native)

## Prompt 01 — Friction Wall (Cloning without context)
```
A labyrinth of weathered colorful doors half-hidden by dense tangled vines and thorny tropical plants, each door painted in deep magenta, teal, and emerald green but locked and overgrown, narrow winding paths leading nowhere, dim amber light filtering through thick foliage, Latin American folk art style, magical realism illustration, sense of barrier and confusion, no text
```
**Alt text:** A labyrinth of colorful locked doors tangled in tropical vines, representing the friction a contributor faces when cloning a repo without team context.

```bash
python generate.py --prompt "A labyrinth of weathered colorful doors half-hidden by dense tangled vines..." --refine --seed 42
```

## Prompt 02 — The Squad Gift (What travels with the code)
```
A luminous gift box overflowing with glowing scrolls, colorful ceremonial banners, and glowing memory crystals, each item radiating deep magenta, teal, emerald, and gold light, arranged on a tropical wooden table surrounded by vibrant flowers, Latin American folk art aesthetic, magical realism illustration, warm amber lighting, sense of abundant knowledge and tools ready to use, no text
```
**Alt text:** A radiant gift box overflowing with scrolls, banners, and glowing crystals, symbolizing the committed Squad directory that travels with every cloned repo—conventions, routing, history, and skills all included.

## Prompt 03 — Inner Source Bridge (Standards travel with the code)
```
A sturdy bridge woven from golden circuit patterns and luminous tropical flowers spanning between two vibrant village squares, one village on each side connected by the glowing pathway, people crossing freely carrying colorful glowing lanterns, deep teal and emerald green palette with warm gold accents, Latin American folk art style, magical realism illustration, birds carrying light particles between the villages, sense of effortless knowledge transfer, no text
```
**Alt text:** A luminous bridge of tropical flowers and circuit patterns connecting two team villages, representing how Brady's standards and conventions automatically travel with cloned repos—contributors conform without friction.

## Prompt 04 — Contributor Experimentation (Cheap to try ideas)
```
A figure standing in a blooming tropical garden surrounded by floating translucent thought bubbles showing different colorful architectural diagrams, each bubble glowing with teal, magenta, or emerald light, the figure gesturing towards the bubbles as if testing ideas, lush foliage in warm coral and amber tones, Latin American folk art aesthetic, magical realism illustration, playful and experimental mood, light particles connecting the bubbles to a glowing decisions scroll in the center, no text
```
**Alt text:** A contributor in a blooming garden surrounded by glowing thought bubbles showing architectural ideas, representing how a committed Squad makes experimentation cheap—test approaches against decisions.md before building.

## Prompt 05 — Knowledge Persistence (What stays when contributors leave)
```
An ancient tropical tree with massive roots and glowing amber lanterns hanging from its branches, colorful glowing memory scrolls and skill badges embedded in the bark like living artifacts, deep emerald green foliage with teal and magenta blossoms, soft warm gold light illuminating the scene, Latin American folk art style, magical realism illustration, sense of permanence and living history, figures from different eras standing around the tree in faded translucent forms showing continuity across time, no text
```
**Alt text:** An ancient tropical tree with glowing memory scrolls and skill badges embedded in its bark, representing how Squad preserves institutional knowledge—when contributors leave, their learnings stay committed in agent histories and decisions.

## Parameter Recommendations

| Use case | --steps | --guidance | --refine |
|----------|---------|------------|----------|
| Quick draft | 20 | 7.5 | no |
| Blog quality | 40 | 7.5 | yes |
| Best quality | 50 | 8.0 | yes |

---

## Blog Post: squad-inner-source — Vacation/Travel Theme (2026-03-22)

**Theme:** Open source contribution as a resort vacation. Latin American folk art, magical realism illustration. Colors: deep magenta, teal, emerald green, warm gold, coral, amber. No text in images. People allowed.

### Image 01 — Seaplane at Tropical Dock (01-friction-wall.png)
**Concept:** First approaching an unfamiliar repo — exciting but uncertain

```
Latin American folk art illustration of a brightly painted seaplane gliding toward a colorful wooden dock, turquoise water below, coral and gold pennants waving from palm trees lining the pier, warm afternoon light
```
**Seed:** 42
**Alt text:** A brightly painted seaplane gliding toward a colorful tropical dock over turquoise water with coral pennants, representing the excitement and uncertainty of approaching an unfamiliar codebase

```bash
python generate.py --prompt "Latin American folk art illustration of a brightly painted seaplane gliding toward a colorful wooden dock, turquoise water below, coral and gold pennants waving from palm trees lining the pier, warm afternoon light" --output "outputs/01.png" --seed 42
```

### Image 02 — Welcome Basket at Resort Door (02-squad-gift.png)
**Concept:** What you get when you clone — the squad is already there waiting

```
Folk art illustration of a vibrant resort welcome hamper overflowing with maps, golden keys, and tropical fruit at a painted hotel door, magenta and teal ribbons, luminous warm light
```
**Seed:** 43
**Alt text:** A vibrant welcome hamper overflowing with maps and golden keys at a painted resort door with magenta ribbons, representing the Squad directory that greets every contributor at clone time

```bash
python generate.py --prompt "Folk art illustration of a vibrant resort welcome hamper overflowing with maps, golden keys, and tropical fruit at a painted hotel door, magenta and teal ribbons, luminous warm light" --output "outputs/02.png" --seed 43
```

### Image 03 — Arched Bridge Between Island Resorts (03-inner-source-bridge.png)
**Concept:** OSS standards and patterns travel across team boundaries

```
Latin American folk art illustration of an arched footbridge covered in painted flowers and folk patterns connecting two colorful resort islands over bright turquoise water, golden sunrise glow
```
**Seed:** 44
**Alt text:** A flower-covered arched footbridge connecting two colorful island resorts over bright turquoise water in golden morning light, representing how the repo owner's standards and patterns travel across team and org boundaries

```bash
python generate.py --prompt "Latin American folk art illustration of an arched footbridge covered in painted flowers and folk patterns connecting two colorful resort islands over bright turquoise water, golden sunrise glow" --output "outputs/03.png" --seed 44
```

### Image 04 — Guest in Hotel Lobby with Maps (04-contributor-success.png)
**Concept:** Validating your approach with the squad team before you build

```
Folk art illustration of a cheerful traveler leaning over a bright hotel lobby table covered in illustrated maps, tropical plants in terracotta pots, gold and teal tilework glowing in warm sunlight
```
**Seed:** 45
**Alt text:** A cheerful traveler leaning over maps spread across a bright hotel lobby table surrounded by tropical plants and gold tilework, representing how contributors can plan and validate their approach with the squad team before writing a line of code

```bash
python generate.py --prompt "Folk art illustration of a cheerful traveler leaning over a bright hotel lobby table covered in illustrated maps, tropical plants in terracotta pots, gold and teal tilework glowing in warm sunlight" --output "outputs/04.png" --seed 45
```

### Image 05 — Hotel Staff Passing Knowledge (05-ceremonies-circle.png)
**Concept:** Knowledge compounds over time; what stays when contributors leave

```
Latin American folk art illustration of three uniformed hotel staff in a sunlit lobby, one passing a glowing golden key and journal to a smiling newcomer, magenta and emerald uniforms, mosaic tile floor
```
**Seed:** 46
**Alt text:** Three hotel staff in a sunlit lobby passing a golden key and journal to a smiling newcomer in magenta and emerald uniforms, representing how institutional knowledge is preserved and passed on rather than lost when contributors move on

```bash
python generate.py --prompt "Latin American folk art illustration of three uniformed hotel staff in a sunlit lobby, one passing a glowing golden key and journal to a smiling newcomer, magenta and emerald uniforms, mosaic tile floor" --output "outputs/05.png" --seed 46
```
