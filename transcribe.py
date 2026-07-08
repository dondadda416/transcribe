#!/usr/bin/env python3
"""
transcribe.py - Transcribe long audio recordings using local Whisper.

Uses faster-whisper (a fast reimplementation of OpenAI's Whisper) running
entirely on your machine. No API key, no internet needed once the model
is downloaded.

Usage:
    python transcribe.py <audio_file> [options]

Examples:
    python transcribe.py meeting.mp3
    python transcribe.py lecture.m4a --model small
    python transcribe.py interview.wav --output notes.txt --language en

First run will download the model (a few hundred MB for 'base' or 'small',
~3 GB for 'large-v3'). Subsequent runs use the cached copy.
"""

import argparse
import sys
import time
from pathlib import Path


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "--:--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def pick_device_and_compute(device_arg: str, compute_arg: str):
    """Resolve 'auto' device/compute-type to concrete values."""
    device = device_arg
    if device == "auto":
        try:
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = compute_arg
    if compute_type == "auto":
        # int8 is the best CPU default; float16 is the best GPU default.
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe long audio files with local Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size. Larger = more accurate but slower and more RAM.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output .txt path. Defaults to the audio file's name with .txt.",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Force a language code (e.g. 'en', 'es'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device. 'auto' picks CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="Compute type (e.g. int8, int8_float16, float16, float32). "
             "'auto' picks int8 on CPU, float16 on GPU.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search width. Higher = slightly better, slower.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice-activity-detection filtering (keeps every silence/pause).",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else audio_path.with_suffix(".txt")
    )

    # Import here so --help works even if faster-whisper isn't installed yet.
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "Error: faster-whisper is not installed.\n"
            "Install it with:  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    device, compute_type = pick_device_and_compute(args.device, args.compute_type)

    print(f"Loading model '{args.model}' on {device} ({compute_type})...")
    print("(First run downloads the model; this may take a minute.)")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    print(f"Transcribing: {audio_path.name}")
    print(f"Output:       {output_path}")
    print("This can take a while for multi-hour files. Progress shown below.\n")

    start = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    duration = float(info.duration) if info.duration else 0.0
    print(f"Detected language: {info.language} "
          f"(confidence {info.language_probability:.2f})")
    print(f"Audio duration:    {format_time(duration)}\n")

    # Write segments as they stream out so a crash or Ctrl-C doesn't lose work.
    last_log = 0.0
    segment_count = 0
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for segment in segments:
                text = segment.text.strip()
                if text:
                    f.write(text + " ")
                    f.flush()
                segment_count += 1

                # Log progress every ~10 seconds of wall-clock time.
                now = time.time()
                if now - last_log >= 10:
                    elapsed = now - start
                    pct = (segment.end / duration * 100) if duration else 0
                    rate = (segment.end / elapsed) if elapsed > 0 else 0
                    eta = ((duration - segment.end) / rate) if rate > 0 else 0
                    print(
                        f"  [{format_time(segment.end)} / {format_time(duration)}]"
                        f"  {pct:5.1f}%   {rate:4.1f}x realtime   "
                        f"ETA {format_time(eta)}"
                    )
                    last_log = now
    except KeyboardInterrupt:
        elapsed = time.time() - start
        print(
            f"\nInterrupted after {format_time(elapsed)}. "
            f"Partial transcript saved to: {output_path}",
            file=sys.stderr,
        )
        return 130

    elapsed = time.time() - start
    print(f"\nDone in {format_time(elapsed)}  ({segment_count} segments).")
    print(f"Transcript saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
