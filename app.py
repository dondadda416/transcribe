"""
app.py - Local web UI for the transcription tool.

Run:
    python app.py

Then open http://localhost:5000 in your browser. Drag in an audio file,
watch the progress bar, download the transcript.

Everything runs on your machine. No files leave your computer.
"""

import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = PROJECT_DIR / "uploads"
OUTPUT_DIR = PROJECT_DIR / "transcripts"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Which model to use. "small" is a good balance for CPU; change to "medium"
# or "large-v3" if you have a GPU and want more accuracy.
MODEL_SIZE = "small"

# Max upload size: 5 GB (Flask default is 16 MB, way too small for long audio).
MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024

# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# job_id -> dict with progress info. Guarded by jobs_lock.
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# Cache the loaded WhisperModel so we don't reload it on every job.
_model = None
_model_lock = threading.Lock()


def get_model():
    """Load (once) and return the shared WhisperModel."""
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        return _model


# ---------------------------------------------------------------------------
# Background transcription
# ---------------------------------------------------------------------------

def _update(job_id: str, **fields):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(fields)


def transcribe_job(job_id: str, audio_path: Path, original_name: str) -> None:
    """Run in a background thread. Streams progress into jobs[job_id]."""
    try:
        _update(job_id, status="loading_model")
        model = get_model()

        _update(job_id, status="transcribing", started_at=time.time())

        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        duration = float(info.duration or 0.0)
        _update(
            job_id,
            duration=duration,
            language=info.language,
            language_probability=float(info.language_probability),
        )

        # Write transcript incrementally so a crash doesn't lose everything.
        transcript_path = OUTPUT_DIR / f"{job_id}_{Path(original_name).stem}.txt"
        with transcript_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                text = seg.text.strip()
                if text:
                    f.write(text + " ")
                    f.flush()
                _update(job_id, current_time=float(seg.end))

        _update(
            job_id,
            status="done",
            current_time=duration,
            transcript_path=str(transcript_path),
            finished_at=time.time(),
        )

    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="error", error=str(exc))
    finally:
        # Best-effort cleanup of the uploaded audio (transcript is what matters).
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", model=MODEL_SIZE)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("audio")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded."}), 400

    job_id = uuid.uuid4().hex[:10]
    original_name = file.filename
    # Save under job_id to avoid collisions if two files share a name.
    save_path = UPLOAD_DIR / f"{job_id}_{original_name}"
    file.save(save_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "filename": original_name,
            "current_time": 0.0,
            "duration": 0.0,
        }

    thread = threading.Thread(
        target=transcribe_job,
        args=(job_id, save_path, original_name),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job."}), 404
        resp = dict(job)

    duration = resp.get("duration", 0.0) or 0.0
    current = resp.get("current_time", 0.0) or 0.0
    started_at = resp.get("started_at")

    if duration > 0:
        resp["percent"] = min(100.0, current / duration * 100.0)
    else:
        resp["percent"] = 0.0

    if started_at and current > 0:
        elapsed = time.time() - started_at
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (duration - current) / rate if rate > 0 else 0.0
        resp["rate"] = rate
        resp["eta"] = max(0.0, eta)

    return jsonify(resp)


@app.route("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return "Transcript not ready.", 404

    transcript_path = Path(job["transcript_path"])
    if not transcript_path.exists():
        return "Transcript file is missing.", 404

    download_name = Path(job["filename"]).stem + ".txt"
    return send_file(transcript_path, as_attachment=True, download_name=download_name)


@app.route("/preview/<job_id>")
def preview(job_id):
    """Return the transcript body as plain text for in-page preview."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return "", 404
    return Path(job["transcript_path"]).read_text(encoding="utf-8"), 200, {
        "Content-Type": "text/plain; charset=utf-8"
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Transcription server starting on http://localhost:5000")
    print("Open that address in your browser. Press Ctrl+C to stop.")
    # threaded=True lets status polls come in while transcription runs.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
