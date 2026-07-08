# Transcribe

A small Python CLI that turns long audio recordings into plain-text transcripts
using a local copy of OpenAI's Whisper model (via `faster-whisper`). Everything
runs on your machine — no API keys, no uploads.

## One-time setup

1. **Install Python 3.9+** if you don't already have it: <https://www.python.org/downloads/>.
2. **Install ffmpeg** (required to decode mp3/m4a/etc.).
   - Windows: download from <https://www.gyan.dev/ffmpeg/builds/> and add the `bin` folder to your PATH, or `winget install Gyan.FFmpeg`.
   - macOS: `brew install ffmpeg`.
   - Linux: `sudo apt install ffmpeg` (or your distro's equivalent).
3. **Install Python dependencies** from this folder:

   ```
   pip install -r requirements.txt
   ```

The first transcription run will also download the Whisper model itself
(a few hundred MB for `base`/`small`, ~3 GB for `large-v3`). After that it's
cached and offline.

## Two ways to use it

You can run this as a **command-line tool** or as a **local web app**. Same
underlying engine either way. Pick whichever feels more comfortable.

### Option A: web app (drag-and-drop in a browser)

From the project folder:

```
python app.py
```

Then open <http://localhost:5000> in any browser. Drop your audio file onto
the page, watch the progress bar, and download the `.txt` when it's done.
Ctrl+C in the terminal to stop the server.

Notes:
- Files are saved to a temporary `uploads/` folder and deleted after
  transcription finishes. Transcripts are kept in `transcripts/`.
- The web app uses the `small` model by default. To change, edit the
  `MODEL_SIZE` value near the top of `app.py`.
- Only one file at a time. Wait for it to finish before starting another.

### Option B: command line

```
python transcribe.py path/to/recording.mp3
```

The transcript is written next to the audio file with a `.txt` extension
(e.g. `recording.txt`). The script writes as it goes, so even if you hit
Ctrl-C partway through, the partial transcript is saved.

### Common options

```
python transcribe.py recording.mp3 --model small             # better accuracy
python transcribe.py recording.mp3 --output transcript.txt   # custom output path
python transcribe.py recording.mp3 --language en             # skip language detection
python transcribe.py recording.mp3 --device cuda             # force GPU (if you have one)
```

Run `python transcribe.py --help` for the full list.

### Picking a model

| Model       | Disk     | Speed (CPU) | Accuracy |
| ----------- | -------- | ----------- | -------- |
| `tiny`      | ~75 MB   | Very fast   | Lowest   |
| `base`      | ~150 MB  | Fast        | Decent   |
| `small`     | ~500 MB  | Medium      | Good     |
| `medium`    | ~1.5 GB  | Slow        | Better   |
| `large-v3`  | ~3 GB    | Very slow on CPU | Best |

For 2+ hour recordings on CPU, `small` is the sweet spot. If you have an
NVIDIA GPU, `medium` or `large-v3` with `--device cuda` will be much faster
than `small` on CPU.

### What to expect for long files

- A 2-hour recording on `small` / CPU takes roughly 30–90 minutes depending
  on your machine.
- The script logs progress every 10 seconds with current position, percent
  complete, processing rate (e.g. "3.2x realtime"), and ETA.
- Voice-activity detection is on by default, which skips long silences and
  speeds things up. Pass `--no-vad` if you want every pause preserved.
