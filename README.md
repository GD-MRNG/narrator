# Narrator

Local page for typing a speech and generating numbered narrated MP3s with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).

## Quick start

```bash
brew install espeak-ng ffmpeg
uv run app.py
```

UV creates the virtual environment and installs Python dependencies on first run. The project is pinned to Python 3.13. Kokoro’s spaCy model fails on 3.14. Kokoro model weights and voices are loaded from the local Hugging Face cache (`~/.cache/huggingface/`). The first run downloads them once (~300 MB); later runs stay offline unless a file is missing.

`uv run app.py` opens a page at `http://127.0.0.1:8765`. Enter a title and the speech, press Generate, and the MP3 shows up below with a player. The title is saved as a snake_case filename under `outputs/` (for example `Minute 1 — Arriving` → `minute_1_arriving.mp3`). The page is served by the script on this machine.

## Adjust the voice and speed

Edit the variables at the top of `app.py`:

```python
VOICE_PRIMARY   = "af_nicole"  # base voice
VOICE_SECONDARY = "af_heart"   # blend voice
VOICE_MIX       = 0.8          # 80% primary, 20% secondary
SPEED           = 0.9          # 1.0 normal, higher is faster, lower is slower
```

`VOICE_MIX` is the weight of the primary voice. The secondary voice gets `1 - VOICE_MIX`. The two voice tensors are averaged once, then reused for the whole file.

`SPEED` is passed straight to Kokoro. `1.0` is the model's normal pace. `1.2` is faster. `0.8` is slower.

Available voices are listed at [hexgrad/Kokoro-82M/voices](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices). American English voices use the `af_` and `am_` prefixes and match `LANG_CODE = "a"`.
