# Narrator

Reads `speech.txt` and writes numbered narrated MP3s to `outputs/`, using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).

## Quick start

```bash
brew install espeak-ng ffmpeg
uv run narrate.py
```

UV creates the virtual environment and installs Python dependencies on first run. The project is pinned to Python 3.13. Kokoro’s spaCy model fails on 3.14. Kokoro model weights and voices download from Hugging Face on first use (~300 MB, cached in `~/.cache/huggingface/` after that).

The script stays open. When `speech.txt` has text, it asks whether to generate the audio. Each yes writes the next file: `outputs/speech_0001.mp3`, then `outputs/speech_0002.mp3`, and so on. `n` skips and checks the file again, so you can edit `speech.txt` without restarting. `q` quits. Kokoro stays loaded for the session.

## Adjust the voice and speed

Edit the variables at the top of `narrate.py`:

```python
VOICE_PRIMARY   = "af_bella"   # base voice
VOICE_SECONDARY = "am_adam"    # blend voice
VOICE_MIX       = 0.8          # 80% primary, 20% secondary
SPEED           = 1.0          # 1.0 normal, higher is faster, lower is slower
```

`VOICE_MIX` is the weight of the primary voice. The secondary voice gets `1 - VOICE_MIX`. The two voice tensors are averaged once, then reused for the whole file.

`SPEED` is passed straight to Kokoro. `1.0` is the model's normal pace. `1.2` is faster. `0.8` is slower.

Available voices are listed at [hexgrad/Kokoro-82M/voices](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices). American English voices use the `af_` and `am_` prefixes and match `LANG_CODE = "a"`.
