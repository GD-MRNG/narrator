"""Watch speech.txt and write numbered narrated MP3s.

Usage:
    uv run narrate.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from kokoro import KPipeline

# ── Config ────────────────────────────────────────────────────────────────────

KOKORO_REPO     = "hexgrad/Kokoro-82M"
VOICE_PRIMARY   = "af_bella"   # base voice (American female)
VOICE_SECONDARY = "am_adam"    # blend voice (American male)
VOICE_MIX       = 0.8          # weight of primary; secondary = 1 - VOICE_MIX
SPEED           = 0.9         # 1.0 normal, higher is faster, lower is slower
LANG_CODE       = "a"          # 'a' = American English
SAMPLE_RATE     = 24_000       # Hz — fixed by Kokoro
DEVICE          = "cpu"

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "speech.txt"
OUTPUT_DIR = ROOT / "outputs"
_SPEECH_FILE = re.compile(r"speech_(\d+)\.mp3$")

_pipeline: KPipeline | None = None
_voice: torch.Tensor | None = None

# Split only at blank lines so a heading stays with the paragraph under it.
_SPLIT_PATTERN = r"\n\n+"


def _clean(text: str) -> str:
    """Strip Markdown and HTML so Kokoro reads clean prose."""
    text = re.sub(r"<[^>]+>", " ", text)                         # HTML tags
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)      # **bold** *italic*
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)   # ## headings
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)         # [text](url)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)    # list bullets
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)    # 1. numbered lists
    text = re.sub(r"^---+$", ".", text, flags=re.MULTILINE)      # hr → brief pause
    text = re.sub(r"https?://\S+", "", text)                      # bare URLs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_voice() -> torch.Tensor:
    primary = torch.load(
        hf_hub_download(KOKORO_REPO, f"voices/{VOICE_PRIMARY}.pt"),
        weights_only=True,
    )
    secondary = torch.load(
        hf_hub_download(KOKORO_REPO, f"voices/{VOICE_SECONDARY}.pt"),
        weights_only=True,
    )
    return primary * VOICE_MIX + secondary * (1.0 - VOICE_MIX)


def _ensure_pipeline() -> tuple[KPipeline, torch.Tensor]:
    global _pipeline, _voice
    if _pipeline is not None and _voice is not None:
        return _pipeline, _voice
    print("Loading Kokoro…")
    _pipeline = KPipeline(lang_code=LANG_CODE, repo_id=KOKORO_REPO, device=DEVICE)
    _voice = _load_voice()
    return _pipeline, _voice


def _synthesize(text: str) -> np.ndarray:
    pipeline, voice = _ensure_pipeline()

    chunks: list[np.ndarray] = []
    print(f"Narrating at speed {SPEED}…")
    for _, _, audio in pipeline(text, voice=voice, speed=SPEED, split_pattern=_SPLIT_PATTERN):
        if audio is None:
            continue
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size:
            chunks.append(samples)

    if not chunks:
        raise SystemExit("Kokoro produced no audio.")
    return np.concatenate(chunks)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def _encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), str(mp3_path)],
        check=True,
    )


def _read_speech() -> str | None:
    """Return cleaned speech, an empty string, or None if the file is missing."""
    if not INPUT_PATH.is_file():
        return None
    return _clean(INPUT_PATH.read_text())


def _next_output_path() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    highest = 0
    for path in OUTPUT_DIR.glob("speech_*.mp3"):
        match = _SPEECH_FILE.fullmatch(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return OUTPUT_DIR / f"speech_{highest + 1:04d}.mp3"


def _write_mp3(text: str, output_path: Path) -> None:
    audio = _synthesize(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        _write_wav(wav_path, audio)
        _encode_mp3(wav_path, output_path)
    finally:
        wav_path.unlink(missing_ok=True)
    print(f"Wrote {output_path}")


def _ask(prompt: str, allowed: set[str]) -> str:
    labels = ", ".join(sorted(allowed))
    while True:
        try:
            raw = input(f"{prompt} ").strip().lower()
        except EOFError:
            print()
            return "q"
        if raw in allowed:
            return raw
        print(f"  Enter one of: {labels}")


def main() -> None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required. Install it with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)

    print("Narrator. Edit speech.txt between takes. q quits.\n")
    try:
        while True:
            text = _read_speech()
            if text is None:
                choice = _ask(f"{INPUT_PATH.name} is missing. [c]heck again, [q]uit?", {"c", "q"})
            elif not text:
                choice = _ask(f"{INPUT_PATH.name} is empty. [c]heck again, [q]uit?", {"c", "q"})
            else:
                print(f"\n{INPUT_PATH.name} has text.")
                choice = _ask("Generate the audio? [y]es, [n]o, [q]uit?", {"y", "n", "q"})
                if choice == "y":
                    _write_mp3(text, _next_output_path())
            if choice == "q":
                break
    except KeyboardInterrupt:
        print()

    print("Goodbye.")


if __name__ == "__main__":
    main()
