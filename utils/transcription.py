"""
transcription.py — Stage 2: Audio transcription with timestamps

Upgraded from OpenAI Whisper (base) to Faster-Whisper with large-v3 model.
Faster-Whisper uses CTranslate2 with INT8 quantization:
  - 4-8x faster than standard Whisper
  - ~1.5 GB VRAM for large-v3 (vs. ~5 GB for standard large)
  - Segment-level timestamps (used for TTS duration targeting)

Falls back to the original openai-whisper if faster-whisper is not installed,
so the existing pipeline continues to work during the transition.
"""

import os
import datetime

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def generate_srt(segments: list, output_path: str):
    """Generate an SRT file from transcribed segments."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")


def transcribe_audio(audio_path: str, model_size: str = "large-v3"):
    """
    Transcribes audio and returns timestamped segments.

    Attempts to use Faster-Whisper first (recommended). Falls back to the
    original openai-whisper if not installed.

    Args:
        audio_path:  Path to input audio file (.wav, .mp3, etc.)
        model_size:  Whisper model size. Recommended: 'large-v3'.
                     For CPU-only / low VRAM: use 'base' or 'small'.

    Returns:
        list: Dicts with 'start' (float), 'end' (float), 'text' (str).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    # --- Attempt 1: Faster-Whisper (preferred) ---
    try:
        return _transcribe_faster_whisper(audio_path, model_size, device)
    except ImportError:
        print(
            "[Transcription] faster-whisper not installed. "
            "Falling back to openai-whisper (slower).\n"
            "  Install with: pip install faster-whisper"
        )
    except Exception as e:
        print(f"[Transcription] faster-whisper failed: {e}. Falling back to openai-whisper.")

    # --- Fallback: openai-whisper ---
    return _transcribe_openai_whisper(audio_path, model_size, device)


def _transcribe_faster_whisper(audio_path: str, model_size: str, device: str) -> list:
    """Transcribe using Faster-Whisper with INT8 quantization."""
    from faster_whisper import WhisperModel

    compute_type = "int8_float16" if device == "cuda" else "int8"
    print(f"[Transcription] Loading Faster-Whisper ({model_size}) on {device} ({compute_type})...")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"[Transcription] Transcribing: {audio_path}")

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",
        vad_filter=True,           # Skip silence — faster and cleaner segments
        vad_parameters=dict(
            min_silence_duration_ms=300,
        ),
    )

    print(f"[Transcription] Detected language: {info.language} (confidence: {info.language_probability:.2f})")

    formatted = []
    for seg in segments:
        if seg.text.strip():
            formatted.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "duration": round(seg.end - seg.start, 3),
            })

    # Explicit cleanup
    del model
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Transcription] Done. {len(formatted)} segments extracted.")
    return formatted


def _transcribe_openai_whisper(audio_path: str, model_size: str, device: str) -> list:
    """Fallback: Transcribe using original openai-whisper."""
    import whisper

    # openai-whisper doesn't have large-v3 in older versions; map it
    safe_size = model_size if model_size not in ("large-v3",) else "large"
    print(f"[Transcription] Loading Whisper ({safe_size}) on {device}...")

    model = whisper.load_model(safe_size, device=device)
    result = model.transcribe(audio_path, word_timestamps=False)

    formatted = []
    for seg in result.get("segments", []):
        formatted.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "duration": round(seg["end"] - seg["start"], 3),
        })

    del model
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Transcription] Done. {len(formatted)} segments extracted.")
    return formatted
