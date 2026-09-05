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


def transcribe_audio(audio_path: str, model_size: str = "large-v3", log_fn=None):
    """
    Transcribes audio and returns timestamped segments.

    Attempts to use Faster-Whisper first (recommended). Falls back to the
    original openai-whisper if faster-whisper is unavailable or fails.

    Args:
        audio_path:  Path to input audio file (.wav, .mp3, etc.)
        model_size:  Whisper model size. Recommended: 'large-v3'.
                     For CPU-only / low VRAM: use 'base' or 'small'.
        log_fn:      Optional callable(str) that receives progress messages, so
                     the caller (e.g. the Streamlit UI) can surface them. When
                     omitted, messages go to stdout. Backward-compatible: existing
                     callers that pass only (audio_path, model_size) are unaffected.

    Returns:
        list: Dicts with 'start' (float), 'end' (float), 'text' (str).

    Raises:
        RuntimeError: if BOTH backends are unavailable/fail. The message names the
            real faster-whisper failure and the fallback status — never a bare
            ``ModuleNotFoundError: No module named 'whisper'`` that hides the cause.
    """
    _log = log_fn if callable(log_fn) else print

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    # --- Attempt 1: Faster-Whisper (preferred) ---
    fw_reason = None
    try:
        return _transcribe_faster_whisper(audio_path, model_size, device, _log)
    except ImportError as e:
        fw_reason = f"not installed ({e})"
        _log(
            "[Transcription] faster-whisper is not installed — falling back to "
            "openai-whisper (slower). Install with: pip install faster-whisper"
        )
    except Exception as e:
        fw_reason = f"{type(e).__name__}: {e}"
        _log(
            f"[Transcription] faster-whisper failed ({fw_reason}) — "
            "falling back to openai-whisper."
        )

    # --- Fallback: openai-whisper ---
    try:
        return _transcribe_openai_whisper(audio_path, model_size, device, _log)
    except ImportError as e:
        raise RuntimeError(
            "Transcription failed — no usable Whisper backend.\n"
            f"  • Primary  (faster-whisper): {fw_reason}\n"
            f"  • Fallback (openai-whisper): not installed ({e})\n"
            "Fix: get faster-whisper working (it is the intended fast path), or "
            "`pip install openai-whisper` so the fallback can run."
        ) from e


def _transcribe_faster_whisper(audio_path: str, model_size: str, device: str, log=print) -> list:
    """Transcribe using Faster-Whisper with INT8 quantization."""
    from faster_whisper import WhisperModel

    compute_type = "int8_float16" if device == "cuda" else "int8"
    log(f"[Transcription] Loading Faster-Whisper ({model_size}) on {device} ({compute_type})...")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    log(f"[Transcription] Transcribing: {audio_path}")

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",
        vad_filter=True,           # Skip silence — faster and cleaner segments
        vad_parameters=dict(
            min_silence_duration_ms=300,
        ),
    )

    log(f"[Transcription] Detected language: {info.language} (confidence: {info.language_probability:.2f})")

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

    log(f"[Transcription] Done. {len(formatted)} segments extracted.")
    return formatted


def _transcribe_openai_whisper(audio_path: str, model_size: str, device: str, log=print) -> list:
    """Fallback: Transcribe using original openai-whisper."""
    import whisper

    # Map only if the requested size is not one this install actually ships.
    available = set(getattr(whisper, "available_models", lambda: [])())
    safe_size = model_size if (not available or model_size in available) else "large"
    log(f"[Transcription] Loading Whisper ({safe_size}) on {device}...")

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

    log(f"[Transcription] Done. {len(formatted)} segments extracted.")
    return formatted
