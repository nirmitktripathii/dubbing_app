"""
voice_manager.py — Voice profile extraction for zero-shot cloning

Extracts the cleanest N seconds from a vocal track to use as the reference
audio clip for IndicF5's zero-shot voice cloning.

Algorithm:
  1. Load the vocal track (output of Demucs source separation).
  2. Split into overlapping windows of WIN_SECONDS seconds.
  3. For each window, compute SNR (signal-to-noise ratio) as a proxy for
     speech clarity. In practice: RMS energy vs. silence threshold.
  4. Pick the window with the highest SNR that is also not too short.
  5. Export that window as a 24kHz mono WAV (IndicF5's native sample rate).

No ML model is used here — pure signal processing via soundfile + numpy.
Falls back to the first N seconds if no clean window is found.
"""

import os
import math
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# Target reference clip duration (seconds). IndicF5 works best with 10-15s.
REFERENCE_CLIP_SECONDS = 12
# Window size for SNR analysis
WIN_SECONDS = 12
# Hop between windows (overlap)
HOP_SECONDS = 4
# Silence threshold: RMS below this fraction of max RMS is considered silence
SILENCE_FRACTION = 0.05
# Native sample rate for IndicF5
INDICF5_SAMPLE_RATE = 24000


def _resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling. For production, use resampy or soxr."""
    if orig_sr == target_sr:
        return audio
    try:
        import resampy
        return resampy.resample(audio, orig_sr, target_sr)
    except ImportError:
        pass
    # Fallback: scipy resample
    try:
        from scipy.signal import resample as scipy_resample
        n_samples = int(len(audio) * target_sr / orig_sr)
        return scipy_resample(audio, n_samples).astype(np.float32)
    except ImportError:
        pass
    # Last resort: naive integer ratio resample (low quality but works)
    ratio = target_sr / orig_sr
    n = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _rms(chunk: np.ndarray) -> float:
    """Root-mean-square energy of an audio chunk."""
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def extract_reference_clip(
    vocals_path: str,
    output_path: str,
    clip_seconds: int = REFERENCE_CLIP_SECONDS,
    prefer_start_offset: float = 5.0,
    segments: list = None,
) -> tuple:
    """
    Extract the cleanest reference audio clip from a vocal stem.

    Args:
        vocals_path:         Path to the vocals WAV file (from Demucs).
        output_path:         Where to save the reference clip (.wav).
        clip_seconds:        Duration of the clip to extract (seconds).
        prefer_start_offset: Skip the first N seconds (intros are often noisy).
        segments:            List of transcribed segments (dicts with 'start', 'end', 'text') to match transcription.

    Returns:
        A tuple of (Path to the reference clip WAV file, Aligned reference text).
    """
    if not SOUNDFILE_AVAILABLE:
        raise RuntimeError(
            "soundfile is not installed. Run: pip install soundfile"
        )

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Vocals file not found: {vocals_path}")

    print(f"[VoiceManager] Extracting reference clip from: {vocals_path}")

    audio, sr = sf.read(vocals_path, dtype="float32", always_2d=False)

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    total_samples = len(audio)
    total_seconds = total_samples / sr
    clip_samples = int(clip_seconds * sr)
    hop_samples = int(HOP_SECONDS * sr)
    win_samples = int(WIN_SECONDS * sr)
    skip_samples = int(prefer_start_offset * sr)

    if total_seconds < clip_seconds:
        print(
            f"[VoiceManager] Audio ({total_seconds:.1f}s) shorter than clip_seconds "
            f"({clip_seconds}s). Using full audio."
        )
        clip = audio
        win_start_s = 0.0
        win_end_s = total_seconds
    else:
        # Slide a window and find the chunk with highest RMS (most speech content)
        best_rms = -1.0
        best_start = skip_samples  # default: skip intro

        start = skip_samples
        while start + win_samples <= total_samples:
            chunk = audio[start: start + win_samples]
            r = _rms(chunk)
            if r > best_rms:
                best_rms = r
                best_start = start
            start += hop_samples

        print(
            f"[VoiceManager] Best window at {best_start/sr:.1f}s "
            f"(RMS={best_rms:.4f})"
        )
        clip = audio[best_start: best_start + clip_samples]
        win_start_s = best_start / sr
        win_end_s = (best_start + clip_samples) / sr

    # Resample to IndicF5's native 24kHz
    if sr != INDICF5_SAMPLE_RATE:
        print(f"[VoiceManager] Resampling {sr}Hz -> {INDICF5_SAMPLE_RATE}Hz")
        clip = _resample_if_needed(clip, sr, INDICF5_SAMPLE_RATE)

    # Normalize to [-1, 1] to avoid clipping
    max_val = np.max(np.abs(clip))
    if max_val > 0:
        clip = clip / max_val * 0.95

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    sf.write(output_path, clip, INDICF5_SAMPLE_RATE, subtype="PCM_16")
    print(f"[VoiceManager] Reference clip saved: {output_path} ({len(clip)/INDICF5_SAMPLE_RATE:.1f}s)")

    # Find the aligned text corresponding to the extracted audio window
    ref_text = ""
    if segments:
        overlapping_texts = []
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            # Find segments that overlap with [win_start_s, win_end_s]
            if max(seg_start, win_start_s) < min(seg_end, win_end_s):
                text_part = seg.get("text", "").strip()
                if text_part:
                    overlapping_texts.append(text_part)
        ref_text = " ".join(overlapping_texts).strip()

    if not ref_text:
        ref_text = "Hello, this is a reference audio clip."

    print(f"[VoiceManager] Extracted aligned reference text: \"{ref_text}\"")
    return output_path, ref_text
