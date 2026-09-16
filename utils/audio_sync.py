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
  3. Crossfade over the ACTUAL overlap (blend to a single voice, trim gross overshoot)
     so a segment that runs past its slot never double-talks over the next one.
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
CROSSFADE_MS = 20        # Studio anti-click fade length (20ms preserves initial plosives/consonants)
MAX_CROSSFADE_MS = 250   # Max crossfade blend window for overlapping segments


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
    prev_end_sample: int,
    base_cf_samples: int,
    max_cf_samples: int,
) -> np.ndarray:
    """
    Overlay ``chunk`` at ``start_sample``, blending cleanly with whatever the PREVIOUS
    segment left on the timeline (``prev_end_sample`` = where that chunk actually ended).

    Two cases:

    * No overlap (previous segment ended at/before ``start_sample``): the timeline is
      silent here, so just fade the chunk in over ``base_cf_samples`` (~50ms) to kill the
      onset click, then add it.

    * Overlap (previous segment overshot past ``start_sample``): crossfade over the ACTUAL
      overlap, capped at ``max_cf_samples``. The previous tail is ramped 1->0 across the
      crossfade and its remainder is zeroed; the chunk is ramped 0->1. So exactly ONE voice
      plays after the crossfade instead of two summed — this is the fix for the audible
      "botch" heard at every segment join. Overshoot beyond the cap is trimmed, which is
      the isochrony-correct outcome (that tail was extra speech that did not fit the slot).

    The old version always faded a FIXED 50ms and then summed, so any overshoot beyond 50ms
    played as full-volume double-talk.
    """
    end_sample = start_sample + len(chunk)
    if end_sample > len(timeline):
        chunk = chunk[:len(timeline) - start_sample]
        end_sample = len(timeline)

    if len(chunk) == 0:
        return timeline

    chunk = chunk.copy()
    overlap = prev_end_sample - start_sample

    if overlap > 0:
        # Equal-power crossfade across the real overlap (capped).
        # cos^2 + sin^2 = 1.0 preserves perceived loudness and avoids the 3dB dip.
        cf = int(min(overlap, len(chunk), max_cf_samples))
        if cf > 0:
            t = np.linspace(0.0, np.pi / 2, cf, dtype=np.float32)
            timeline[start_sample:start_sample + cf] *= np.cos(t)
            chunk[:cf] *= np.sin(t)
        # Remove any previous-segment tail past the crossfade so it can't double-talk under
        # the new chunk. The cos ramp already reached 0 at start+cf, so zeroing here is click-free.
        clear_end = min(prev_end_sample, len(timeline))
        if clear_end > start_sample + cf:
            timeline[start_sample + cf:clear_end] = 0.0
    else:
        # No overlap: equal-power short fade-in from silent timeline avoids clicks without attenuating plosives
        cf = int(min(base_cf_samples, len(chunk)))
        if cf > 0:
            t = np.linspace(0.0, np.pi / 2, cf, dtype=np.float32)
            chunk[:cf] *= np.sin(t)

    # Studio anti-click fade-out on chunk tail prevents square-wave step clicks at segment ends
    tail_cf = int(min(base_cf_samples, len(chunk)))
    if tail_cf > 0:
        t_tail = np.linspace(np.pi / 2, 0.0, tail_cf, dtype=np.float32)
        chunk[-tail_cf:] *= np.sin(t_tail)

    timeline[start_sample:end_sample] += chunk
    return timeline


def sync_audio_segments(
    segments: list,
    output_path: str,
    background_audio_path: str = None,
    background_volume: float = 0.35,
    log_fn=None,
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
    def _emit(msg):
        print(msg)
        if log_fn:
            try:
                log_fn(f"    {msg}")
            except Exception:
                pass

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
        _emit("[AudioSync] No segments to process.")
        return output_path

    _emit(f"[AudioSync] Building dubbed vocal timeline: {total_samples/SAMPLE_RATE:.2f}s")
    vocal_timeline = np.zeros(total_samples, dtype=np.float32)
    crossfade_samples = int(CROSSFADE_MS / 1000 * SAMPLE_RATE)
    max_crossfade_samples = int(MAX_CROSSFADE_MS / 1000 * SAMPLE_RATE)

    # Where the previously-placed chunk actually ENDED on the timeline. Segments sit at
    # their fixed SRT starts, so a chunk longer than its slot runs past the next chunk's
    # start — an overlap _apply_crossfade blends into one voice instead of summing two.
    prev_end_sample = 0
    for i, seg in enumerate(segments):
        audio_path = seg.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            _emit(f"  [Segment {i}] Missing audio_path — skipping.")
            continue

        start_sample = int(seg["start"] * SAMPLE_RATE)
        target_samples = int((seg["end"] - seg["start"]) * SAMPLE_RATE)

        try:
            chunk = _load_audio_np(audio_path)
        except Exception as e:
            _emit(f"  [Segment {i}] Failed to load {audio_path}: {e}")
            continue

        if len(chunk) == 0:
            continue

        # Log sync accuracy AND how far this chunk overlaps the previous one. overlap_prev
        # is exactly the double-talk the crossfade now absorbs — the number to watch when
        # judging whether upstream duration control is tight enough.
        drift_ms = abs(len(chunk) - target_samples) / SAMPLE_RATE * 1000
        overlap_prev_ms = max(0, prev_end_sample - start_sample) / SAMPLE_RATE * 1000
        _emit(
            f"  [Segment {i}] start={seg['start']:.2f}s "
            f"target={target_samples/SAMPLE_RATE:.3f}s "
            f"actual={len(chunk)/SAMPLE_RATE:.3f}s "
            f"drift={drift_ms:.1f}ms overlap_prev={overlap_prev_ms:.0f}ms"
        )

        vocal_timeline = _apply_crossfade(
            vocal_timeline, chunk, start_sample, prev_end_sample,
            crossfade_samples, max_crossfade_samples,
        )
        prev_end_sample = min(start_sample + len(chunk), len(vocal_timeline))

    # Normalize dubbed vocal track
    vocal_timeline = _peak_normalize(vocal_timeline, target_peak=0.9)

    # Mix with background if provided
    if background_audio_path and os.path.exists(background_audio_path):
        _emit(f"[AudioSync] Mixing background track: {background_audio_path}")
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
            _emit(f"[AudioSync] Background mix failed: {e}. Using vocals only.")
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
            _emit("[AudioSync] MP3 conversion failed — saved as WAV instead.")
    else:
        sf.write(output_path, final, SAMPLE_RATE, subtype="PCM_16")

    _emit(f"[AudioSync] Final dubbed audio saved: {output_path}")
    return output_path
