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


import re

ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.",
    "i.e.", "e.g.", "u.s.", "u.k.", "a.m.", "p.m.", "no."
}

def _is_sentence_terminal(word: str) -> bool:
    w = word.strip().lower()
    if w in ABBREVIATIONS:
        return False
    clean = re.sub(r'["\'\)\]\}]+$', '', w)
    return clean.endswith(('.', '!', '?'))

def _is_clause_boundary(word: str) -> bool:
    w = word.strip().lower()
    clean = re.sub(r'["\'\)\]\}]+$', '', w)
    return clean.endswith((',', ';', ':', '—', '-'))

def resegment_into_sentences(
    words: list,
    min_duration: float = 2.0,
    max_duration: float = 12.0,
    max_pause_sec: float = 0.6,
) -> list:
    """
    Group word-level tokens into syntactically complete sentences/clauses.

    Prevents VAD silence splits from severing verbs, postpositions, and clauses in the middle.
    """
    if not words:
        return []

    segments = []
    current_words = []

    def commit_segment():
        nonlocal current_words
        if not current_words:
            return
        text = "".join(w["word"] for w in current_words).strip()
        text = re.sub(r'\s+', ' ', text)
        if text:
            start_t = round(current_words[0]["start"], 3)
            end_t = round(current_words[-1]["end"], 3)
            segments.append({
                "start": start_t,
                "end": end_t,
                "text": text,
                "duration": round(end_t - start_t, 3),
            })
        current_words = []

    for i, w in enumerate(words):
        current_words.append(w)
        dur = w["end"] - current_words[0]["start"]
        word_text = w["word"].strip()

        next_w = words[i + 1] if i + 1 < len(words) else None
        pause = (next_w["start"] - w["end"]) if next_w else 0.0

        is_term = _is_sentence_terminal(word_text)
        is_clause = _is_clause_boundary(word_text)

        # 1: Terminal punctuation (. ? !) and duration >= min_duration
        if is_term and dur >= min_duration:
            commit_segment()
        # 2: Exceeding max_duration -> split at clause boundary or substantial pause
        elif dur >= max_duration:
            commit_segment()
        elif dur >= (max_duration * 0.75) and (is_clause or pause >= max_pause_sec):
            commit_segment()
        # 3: Very large silence gap between words (>= 1.2s)
        elif pause >= 1.2 and dur >= min_duration:
            commit_segment()

    commit_segment()
    return segments


def _transcribe_faster_whisper(audio_path: str, model_size: str, device: str, log=print) -> list:
    """Transcribe using Faster-Whisper with INT8 quantization and sentence-level resegmentation."""
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
        word_timestamps=True,
    )

    log(f"[Transcription] Detected language: {info.language} (confidence: {info.language_probability:.2f})")

    all_words = []
    raw_formatted = []
    for seg in segments:
        if seg.text.strip():
            raw_formatted.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "duration": round(seg.end - seg.start, 3),
            })
            for w in getattr(seg, "words", []) or []:
                all_words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                })

    # Resegment into complete sentences if word timestamps are available
    if all_words:
        formatted = resegment_into_sentences(all_words)
        log(f"[Transcription] Resegmented {len(raw_formatted)} raw pause chunks into {len(formatted)} complete sentences.")
    else:
        formatted = raw_formatted

    # Explicit cleanup
    del model
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"[Transcription] Done. {len(formatted)} segments extracted.")
    return formatted


def _transcribe_openai_whisper(audio_path: str, model_size: str, device: str, log=print) -> list:
    """Fallback: Transcribe using original openai-whisper with sentence-level resegmentation."""
    import whisper

    # Map only if the requested size is not one this install actually ships.
    available = set(getattr(whisper, "available_models", lambda: [])())
    safe_size = model_size if (not available or model_size in available) else "large"
    log(f"[Transcription] Loading Whisper ({safe_size}) on {device}...")

    model = whisper.load_model(safe_size, device=device)
    result = model.transcribe(audio_path, word_timestamps=True)

    all_words = []
    raw_formatted = []
    for seg in result.get("segments", []):
        if seg.get("text", "").strip():
            raw_formatted.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "duration": round(seg["end"] - seg["start"], 3),
            })
            for w in seg.get("words", []):
                all_words.append({
                    "start": w["start"],
                    "end": w["end"],
                    "word": w["word"],
                })

    if all_words:
        formatted = resegment_into_sentences(all_words)
        log(f"[Transcription] Resegmented {len(raw_formatted)} raw pause chunks into {len(formatted)} complete sentences.")
    else:
        formatted = raw_formatted

    del model
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"[Transcription] Done. {len(formatted)} segments extracted.")
    return formatted

