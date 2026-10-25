"""Local page for typing a speech and playing the numbered MP3s.

Usage:
    uv run web.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from kokoro import KPipeline

# ── Config ────────────────────────────────────────────────────────────────────

KOKORO_REPO     = "hexgrad/Kokoro-82M"
VOICE_PRIMARY   = "af_nicole"   # base voice (American female)
VOICE_SECONDARY = "af_heart"    # blend voice (American male)
VOICE_MIX       = 0.8          # weight of primary; secondary = 1 - VOICE_MIX
SPEED           = 0.9         # 1.0 normal, higher is faster, lower is slower
LANG_CODE       = "a"          # 'a' = American English
SAMPLE_RATE     = 24_000       # Hz — fixed by Kokoro
DEVICE          = "cpu"

HOST = "127.0.0.1"
PORT = 8765

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
_SPEECH_FILE = re.compile(r"speech_(\d+)\.mp3$")

_pipeline: KPipeline | None = None
_voice: torch.Tensor | None = None
_generate_lock = threading.Lock()

# Split only at blank lines so a heading stays with the paragraph under it.
_SPLIT_PATTERN = r"\n\n+"

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Narrator</title>
  <style>
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      display: flex;
      flex-direction: column;
      background: #f2f2f2;
      color: #111;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      flex: none;
      background: #fff;
      border-bottom: 1px solid #ddd;
    }
    .wrap {
      width: min(840px, calc(100% - 48px));
      margin: 0 auto;
    }
    header .wrap {
      padding: 16px 0;
      font-size: 15px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    main {
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .composer, .library {
      flex: 1;
      min-height: 0;
    }
    .composer {
      background: #fff;
      border-bottom: 1px solid #ddd;
    }
    .composer .wrap {
      height: 100%;
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 20px 0;
    }
    textarea {
      flex: 1;
      min-height: 0;
      width: 100%;
      resize: none;
      border: 1px solid #ccc;
      border-radius: 6px;
      padding: 14px 16px;
      background: #fafafa;
      color: #111;
      font: inherit;
      font-size: 16px;
      line-height: 1.5;
    }
    textarea:focus {
      outline: 2px solid #111;
      border-color: #111;
    }
    .actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 16px;
    }
    #status {
      margin: 0;
      color: #666;
      font-size: 14px;
    }
    button {
      background: #111;
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 10px 18px;
      font: inherit;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover:not(:disabled) { background: #333; }
    button:disabled { background: #9a9a9a; cursor: default; }
    .library { overflow: auto; }
    .library .wrap { padding: 20px 0 32px; }
    .clip {
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 12px 16px 14px;
      margin-bottom: 10px;
    }
    .clip h2 {
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 600;
      color: #444;
    }
    audio { width: 100%; height: 36px; }
    .empty { margin: 0; color: #777; }
  </style>
</head>
<body>
  <header><div class="wrap">Narrator</div></header>
  <main>
    <section class="composer">
      <div class="wrap">
        <textarea id="text" placeholder="Type the speech here."></textarea>
        <div class="actions">
          <p id="status" role="status"></p>
          <button id="generate" type="button" disabled>Generate</button>
        </div>
      </div>
    </section>
    <section class="library">
      <div class="wrap" id="list" aria-live="polite"></div>
    </section>
  </main>
  <script>
    const text = document.querySelector("#text");
    const button = document.querySelector("#generate");
    const status = document.querySelector("#status");
    const list = document.querySelector("#list");
    let busy = false;

    function syncButton() {
      button.disabled = busy || text.value.trim().length === 0;
      button.textContent = busy ? "Generating…" : "Generate";
    }

    function clipView(clip) {
      const item = document.createElement("article");
      item.className = "clip";
      const title = document.createElement("h2");
      title.textContent = clip.name;
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = clip.url;
      item.append(title, audio);
      return item;
    }

    async function loadClips() {
      const clips = await fetch("/api/clips").then((response) => response.json());
      list.replaceChildren();
      if (!clips.length) {
        const empty = document.createElement("p");
        empty.className = "empty";
        empty.textContent = "No audio yet.";
        list.append(empty);
        return;
      }
      for (const clip of clips) list.append(clipView(clip));
    }

    button.addEventListener("click", async () => {
      if (busy) return;
      busy = true;
      status.textContent = "";
      syncButton();
      try {
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text.value }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Could not generate audio.");
        await loadClips();
        list.parentElement.scrollTop = 0;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        busy = false;
        syncButton();
      }
    });

    text.addEventListener("input", syncButton);
    text.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !button.disabled) {
        button.click();
      }
    });

    loadClips().catch((error) => {
      status.textContent = error.message;
    });
  </script>
</body>
</html>
"""


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


def _clips() -> list[dict[str, str]]:
    if not OUTPUT_DIR.is_dir():
        return []
    paths = [
        path for path in OUTPUT_DIR.glob("speech_*.mp3")
        if _SPEECH_FILE.fullmatch(path.name)
    ]
    paths.sort(key=lambda path: path.name, reverse=True)
    return [{"name": path.name, "url": f"/audio/{path.name}"} for path in paths]


def _audio_path(name: str):
    if not _SPEECH_FILE.fullmatch(name):
        return None
    path = (OUTPUT_DIR / name).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.is_file():
        return None
    return path


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/clips":
            self._send_json(_clips())
            return
        if path.startswith("/audio/"):
            self._send_audio(unquote(path.removeprefix("/audio/")))
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._send_json({"error": "Not found."}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/generate":
            self._send_json({"error": "Not found."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Bad request."}, 400)
            return
        if length > 1_000_000:
            self._send_json({"error": "Text is too long."}, 413)
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = payload.get("text", "")
        except (json.JSONDecodeError, AttributeError):
            self._send_json({"error": "Bad request."}, 400)
            return
        if not isinstance(text, str):
            self._send_json({"error": "Bad request."}, 400)
            return

        speech = _clean(text)
        if not speech:
            self._send_json({"error": "Enter some text first."}, 400)
            return
        if not _generate_lock.acquire(blocking=False):
            self._send_json({"error": "Already generating. Wait for this take to finish."}, 409)
            return
        try:
            output = _next_output_path()
            _write_mp3(speech, output)
        except SystemExit as exc:
            self._send_json({"error": str(exc) or "Kokoro produced no audio."}, 500)
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc) or "Could not generate audio."}, 500)
            return
        finally:
            _generate_lock.release()
        self._send_json({"name": output.name, "url": f"/audio/{output.name}"})

    def _send_audio(self, name: str) -> None:
        path = _audio_path(name)
        if path is None:
            self._send_json({"error": "Not found."}, 404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            parsed = _parse_range(range_header, size)
            if parsed is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = parsed
            status = 206
        length = end - start + 1
        try:
            self.send_response(status)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as audio:
                audio.seek(start)
                remaining = length
                while remaining:
                    chunk = audio.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_json(self, payload: object, status: int = 200) -> None:
        self._send_bytes(json.dumps(payload).encode(), "application/json", status)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    if size <= 0 or not header.startswith("bytes="):
        return None
    spec = header.removeprefix("bytes=").split(",", 1)[0].strip()
    start_text, separator, end_text = spec.partition("-")
    if not separator:
        return None
    try:
        if start_text == "":
            suffix = int(end_text)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or start > end:
        return None
    return start, min(end, size - 1)


def main() -> None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required. Install it with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)

    server = ThreadingHTTPServer((HOST, PORT), _Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Narrator page at {url}", flush=True)
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGoodbye.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
