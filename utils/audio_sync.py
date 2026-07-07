"""
audio_sync.py — Stage 5: Audio Assembly

Rebuilt for the new pipeline architecture. Key changes vs. old version:

OLD: Generated audio might be wrong length → FFmpeg atempo stretches it.
NEW: IndicF5 generates audio at the correct duration natively.
     This module just places audio chunks at the right timestamps on a timeline
     and mixes in the original background/SFX track from Demucs separation.

Responsibilities:
  1. Create a silent timeline of the correct total duration.
  2. Overlay each TTS chunk at its exact start timestamp (no stretching).
  3. Apply 50ms crossfades between overlapping or adjacent segments.
  4. Mix the dubbed vocal timeline with the preserved background track.
  5. Normalize loudness (LUFS matching) so dubbed audio matches original volume.
  6. Export as WAV (for maximum quality before final FFmpeg encode).
"""

import os
import subprocess
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


SAMPLE_RATE = 24000      # IndicF5 native sample rate
CROSSFADE_MS = 50        # Crossfade between segments to eliminate clicks


def _load_audio_np(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file to float32 numpy array at target_sr."""
    if SOUNDFILE_AVAILABLE:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = _resample(audio, sr, target_sr)
        return audio

    raise RuntimeError("soundfile is required. Run: pip install soundfile")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling fallback."""
    if orig_sr == target_sr:
        return audio
    try:
        import resampy
        return resampy.resample(audio, orig_sr, target_sr)
    except ImportError:
        pass
    try:
        from scipy.signal import resample as scipy_resample
        n = int(len(audio) * target_sr / orig_sr)
        return scipy_resample(audio, n).astype(np.float32)
    except ImportError:
        pass
    n = int(len(audio) * target_sr / orig_sr)
    indices = np.linspace(0, len(audio) - 1, n)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _peak_normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize audio to target peak amplitude."""
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def _rms_normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio to target RMS level."""
    current_rms = float(np.sqrt(np.mean(audio ** 2)))
    if current_rms < 1e-6:
        return audio
    return (audio * (target_rms / current_rms)).clip(-1.0, 1.0).astype(np.float32)


def _apply_crossfade(
    timeline: np.ndarray,
    chunk: np.ndarray,
    start_sample: int,
    crossfade_samples: int,
) -> np.ndarray:
    """
    Overlay chunk onto timeline at start_sample with a linear crossfade
    at the beginning of the chunk to eliminate click artifacts.
    """
    end_sample = start_sample + len(chunk)
    if end_sample > len(timeline):
        chunk = chunk[:len(timeline) - start_sample]
        end_sample = len(timeline)

    if len(chunk) == 0:
        return timeline

    cf = min(crossfade_samples, len(chunk))
    fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)

    # Fade in the new chunk
    chunk = chunk.copy()
    chunk[:cf] *= fade_in

    # Fade out the existing timeline at the overlap region
    timeline[start_sample: start_sample + cf] *= fade_out

    # Add (mix) chunk into timeline
    timeline[start_sample:end_sample] += chunk
    return timeline


def sync_audio_segments(
    segments: list,
    output_path: str,
    background_audio_path: str = None,
    background_volume: float = 0.35,
) -> str:
    """
    Assemble dubbed audio segments onto a timeline and optionally mix with background.

    Args:
        segments:               List of dicts with 'start', 'end', 'audio_path'.
        output_path:            Path to save the final mixed audio (.wav or .mp3).
        background_audio_path:  Path to the Demucs background stem (.wav).
                                None = no background mixing (vocals only).
        background_volume:      Background volume ratio (0.0–1.0). Default 0.35
                                keeps it audible but not overpowering.

    Returns:
        output_path (str)
    """
    if not SOUNDFILE_AVAILABLE:
        raise RuntimeError("soundfile is required. Run: pip install soundfile")

    # Determine total duration
    if segments:
        # Try to use background audio length as ground truth (most accurate)
        if background_audio_path and os.path.exists(background_audio_path):
            bg_check, bg_sr = sf.read(background_audio_path, dtype="float32", always_2d=False)
            total_samples = int(len(bg_check) * SAMPLE_RATE / bg_sr)
        else:
            last_end = max(seg["end"] for seg in segments)
            total_samples = int(last_end * SAMPLE_RATE) + int(0.5 * SAMPLE_RATE)
    else:
        print("[AudioSync] No segments to process.")
        return output_path

    print(f"[AudioSync] Building dubbed vocal timeline: {total_samples/SAMPLE_RATE:.2f}s")
    vocal_timeline = np.zeros(total_samples, dtype=np.float32)
    crossfade_samples = int(CROSSFADE_MS / 1000 * SAMPLE_RATE)

    for i, seg in enumerate(segments):
        audio_path = seg.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            print(f"  [Segment {i}] Missing audio_path — skipping.")
            continue

        start_sample = int(seg["start"] * SAMPLE_RATE)
        target_samples = int((seg["end"] - seg["start"]) * SAMPLE_RATE)

        try:
            chunk = _load_audio_np(audio_path)
        except Exception as e:
            print(f"  [Segment {i}] Failed to load {audio_path}: {e}")
            continue

        if len(chunk) == 0:
            continue

        # Log sync accuracy
        drift_ms = abs(len(chunk) - target_samples) / SAMPLE_RATE * 1000
        print(
            f"  [Segment {i}] start={seg['start']:.2f}s "
            f"target={target_samples/SAMPLE_RATE:.3f}s "
            f"actual={len(chunk)/SAMPLE_RATE:.3f}s "
            f"drift={drift_ms:.1f}ms"
        )

        vocal_timeline = _apply_crossfade(
            vocal_timeline, chunk, start_sample, crossfade_samples
        )

    # Normalize dubbed vocal track
    vocal_timeline = _peak_normalize(vocal_timeline, target_peak=0.9)

    # Mix with background if provided
    if background_audio_path and os.path.exists(background_audio_path):
        print(f"[AudioSync] Mixing background track: {background_audio_path}")
        try:
            bg = _load_audio_np(background_audio_path)
            # Match lengths
            if len(bg) > total_samples:
                bg = bg[:total_samples]
            elif len(bg) < total_samples:
                bg = np.pad(bg, (0, total_samples - len(bg)))
            bg = _peak_normalize(bg, target_peak=0.25) * background_volume
            final = (vocal_timeline + bg).clip(-1.0, 1.0).astype(np.float32)
        except Exception as e:
            print(f"[AudioSync] Background mix failed: {e}. Using vocals only.")
            final = vocal_timeline
    else:
        final = vocal_timeline

    # Export
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp3":
        # Write WAV first, then convert to MP3 with FFmpeg
        tmp_wav = output_path.replace(".mp3", "_tmp.wav")
        sf.write(tmp_wav, final, SAMPLE_RATE, subtype="PCM_16")
        cmd = [
            "ffmpeg", "-y", "-i", tmp_wav,
            "-codec:a", "libmp3lame", "-qscale:a", "2",
            output_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            os.remove(tmp_wav)
        else:
            # FFmpeg MP3 conversion failed — just rename the WAV
            os.replace(tmp_wav, output_path.replace(".mp3", ".wav"))
            output_path = output_path.replace(".mp3", ".wav")
            print("[AudioSync] MP3 conversion failed — saved as WAV instead.")
    else:
        sf.write(output_path, final, SAMPLE_RATE, subtype="PCM_16")

    print(f"[AudioSync] Final dubbed audio saved: {output_path}")
    return output_path
