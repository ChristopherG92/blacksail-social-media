# Blacksail video toolkit

These are the Python scripts that render Blacksail's animated reels. The output is vertical video: 1080×1920, 30 fps, MP4.

## Setup (cloud workspace)

```
pip install --break-system-packages numpy opencv-python-headless pillow scipy global-land-mask
```

You also need `ffmpeg`, and the Inter and DejaVu fonts. `common.py` expects the Inter fonts in `/usr/share/fonts/opentype/inter/`. If they're missing, install them with `apt-get install fonts-inter` or point the `FD` variable at another folder.

Copy this whole folder into the workspace. Run the scripts from the examples folder, with the parent folder on `PYTHONPATH`:

```
cd video_toolkit/examples
PYTHONPATH=.. python3 payment_journey.py test 20,120,300   # test frames -> contact sheet
PYTHONPATH=.. python3 payment_journey.py render 0 435      # renders frames to an mp4 chunk
```

Each example has its own audio script. See `sfx.py`; it builds float32 audio, which you mux in with ffmpeg.

## Modules

| File | What it gives you |
|---|---|
| `common.py` | Canvas, brand colours, Inter fonts, text and kinetic captions, glow, grain, vignette, glitch, zoom-blur, writer |
| `logo.py` | `end_logo(t)`: the official Blacksail logo end card (wipe, glow, light sweep). Uses `blacksail_logo.png`. |
| `recon.py` | Reconciliation UI (bank vs ledger matching), `journal_ui` (auto journal entries), `close_ui` (month-end ring), `stats_ui`, `tilt()` 3D panel tilt |
| `globe.py` | 3D dot-matrix globe with payment arcs between cities |
| `card.py` | 3D business debit card (generic, unbranded) |
| `servers.py` | Server-room corridor fly-through |
| `sat.py` | Military-style satellite descent onto The Forum, Century City. Imagery is in `satellite/` and is © Esri, Maxar, Earthstar Geographics; keep the on-screen credit. |
| `sfx.py` | Original sound design: risers, impacts, beat, blips, chimes, whooshes |

## Example scenes (`examples/`)

| Script | Reel |
|---|---|
| `payment_journey.py` | Where does a payment actually go? |
| `chaos_to_order.py` | Particle swarm of unmatched transactions that snaps into a matched grid |
| `month_end_pipeline.py` | Close pipeline: sources → ingest → match → exceptions → journals → CLOSED |
| `forum_recon_hero.py` | Satellite → servers → recon → journals → close → logo (needs `sign_clip.mp4`) |
| `exception_journey.py` | Exception of the Week: unmatched payment → trace → detected, investigated, resolved, logged → logo |

## Brand rules

- Dark navy background, cyan `#2EDBFF`, green `#40FF8C`, amber for exceptions, red only for errors.
- Inter type.
- Hook in the first 1–2 seconds; captions burnt in.
- End every reel on the logo.
- No real bank or PSP brands, and no invented client results.

## Changelog
- 2026-10-12 (W42): added `examples/exception_journey.py`, the "Exception of the Week" reel. It has a tilted ledger with depth of field and a push-in, a camera-follow payment trail with a motion-blurred packet, an exception-workflow timeline and a payoff card. All text sits inside the 192–1536 px safe zone, and the audio function is built in (`python3 exception_journey.py audio out.f32`). Static templates now live in `00_Brand_Kit/social_templates/`.
