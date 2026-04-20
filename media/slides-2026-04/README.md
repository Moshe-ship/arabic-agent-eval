# April 2026 results — four-slide deck

Four SVG slides for sharing the RESULTS.md summary without the full
document. Visual direction: white background, dark gray (#1a1a1a)
primary text, single accent gray (#666666), no gradients, no cards,
no glossy effects, no emojis.

System font stack — renders consistently on macOS, iOS, Windows
(falls back to Helvetica / Arial).

## Files

- `1-measured.svg` — what was measured (3 bundles, 7 models, schema-bound, git-pinned)
- `2-findings.svg` — ranking flips by surface, no universal winner
- `3-why.svg` — reproducibility framing
- `4-next.svg` — Hermes direct, linguistic adversarial, bigger agentic set

## Usage

Open any SVG in a browser and screenshot at 1200×675, or use
`rsvg-convert` / `inkscape` to render to PNG:

```bash
# 1200x675 PNGs for social posting
for f in *.svg; do
  rsvg-convert -w 1200 -h 675 "$f" -o "${f%.svg}.png"
done
```

## Copy

Text-only copy of the deck lives inline in each SVG — open any file
in a text editor to read or adapt.
