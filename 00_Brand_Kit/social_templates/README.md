# Blacksail social templates (static)

`bs_slides.py` holds the shared helpers for static posts, rendered with Python and Pillow at a 1200×1500 master size:
- `background()`: navy gradient with cyan and green glows, a dot grid and fine grain
- `chrome()`: kicker, page counter, logo and URL footer
- `logo_white()`: the official logo as a white PNG with transparency
- `glow_layer()`, `wrap()`, `export()`

Week scripts that use it (copies live in each post's `Source/` folder):
- carousel: `carousel_outgrown.py` (cover, 5 sign slides with a tick-box footer, "what good looks like", end card) → IG PNGs, LI PNGs and the LinkedIn PDF
- image: `image_blueprint.py` (6-stage blueprint in 4:5 and 1:1)
- story: `story.py` (checklist story, 1080×1920)

Needs: Python 3, numpy, pillow, img2pdf, the Inter fonts in `/usr/share/fonts/opentype/inter/`, and DejaVu Sans Mono. Set `BS_LOGO` if `video_toolkit/blacksail_logo.png` isn't at `../video_toolkit/`.

## Changelog
- 2026-10-12 (W42): v1, created.
