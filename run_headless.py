#!/usr/bin/env python3
"""Headless batch driver for the Indic dubbing pipeline (Kaggle "Save & Run All").

Runs Steps 1-7 end-to-end with NO Streamlit UI and NO Cloudflare tunnel, so the whole
pipeline can be validated as a single Kaggle commit: source video in, dubbed video out,
everything written to an output directory the commit persists. This is the validation
vehicle — it exercises the SAME stage modules the Streamlit app uses, so a green batch run
means the real pipeline works end-to-end.

Step 6 (IndicF5 TTS) runs through pipeline.tts_supervisor.generate_tts_supervised — the
process-isolation freeze fix — so a wedged GPU op is SIGKILLed and relaunched (resuming
from disk) instead of hanging the whole run, and a poison segment degrades to silence
rather than aborting a long render.

Configuration — all via environment variables, with defaults:
    DUBBING_INPUT_VIDEO   path to the source video. If unset, the first video found under
                          /kaggle/input (recursively, sorted) is used.
    DUBBING_TARGET_LANG   target language display name (default "Hindi").
    DUBBING_WHISPER_MODEL Whisper model size (default "medium").
    DUBBING_USE_DEMUCS    "1"/"0" — run Demucs source separation (default "1").
    DUBBING_BG_VOLUME     background-music volume in the final mix (default "0.3").
    DUBBING_OUTPUT_DIR    where outputs go (default "/kaggle/working/dubbing_output").
    GEMINI_API_KEY        REQUIRED — Gemini key for isochrony-aware translation.
    (Step-6 stall budgets DUBBING_TTS_LOAD_STALL / DUBBING_TTS_SEG_STALL etc. are read by
     tts_supervisor itself.)

Exit codes: 0 success; 2 configuration error (no key / no input video); 1 a stage failed.
Step 6 degrades poison segments to silence internally rather than failing the run.
"""
import os
import sys
import json
import time
import traceback
from datetime import datetime

from utils.audio_extraction import extract_audio
from utils.transcription import transcribe_audio, generate_srt
from utils.video_merge import merge_video_audio_subs
from utils.audio_sync import sync_audio_segments
from pipeline.source_separation import separate_audio
from pipeline.isochrony_translation import translate_segments_isochrony
from pipeline.voice_manager import extract_reference_clip
from pipeline.tts_supervisor import generate_tts_supervised

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")


def _find_input_video():
    env = os.environ.get("DUBBING_INPUT_VIDEO", "").strip()
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"DUBBING_INPUT_VIDEO={env!r} does not exist.")
        return env
    for root in ("/kaggle/input",):
        if not os.path.isdir(root):
            continue
        hits = []
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(VIDEO_EXTS):
                    hits.append(os.path.join(dirpath, f))
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(
        "No input video found. Set DUBBING_INPUT_VIDEO, or attach a dataset with a video "
        "under /kaggle/input."
    )


def main():
    t0 = time.time()
    out_dir = os.environ.get("DUBBING_OUTPUT_DIR", "/kaggle/working/dubbing_output")
    temp_dir = os.path.join(out_dir, "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)

    logf = open(os.path.join(out_dir, "pipeline_log.txt"), "a", encoding="utf-8")

    def log(msg=""):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            logf.write(line + "\n")
            logf.flush()
        except Exception:
            pass

    target_lang = os.environ.get("DUBBING_TARGET_LANG", "Hindi").strip() or "Hindi"
    model_size = os.environ.get("DUBBING_WHISPER_MODEL", "medium").strip() or "medium"
    use_demucs = os.environ.get("DUBBING_USE_DEMUCS", "1").strip().lower() not in ("0", "false", "no", "")
    try:
        bg_volume = float(os.environ.get("DUBBING_BG_VOLUME", "0.3"))
    except ValueError:
        bg_volume = 0.3
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    log("=" * 72)
    log("Indic Dubbing — HEADLESS BATCH RUN")
    log(f"  target_lang={target_lang}  whisper={model_size}  demucs={use_demucs}  bg_vol={bg_volume}")
    log(f"  output_dir={out_dir}")
    if not api_key:
        log("FATAL: GEMINI_API_KEY is not set — isochrony translation cannot run.")
        logf.close()
        return 2
    try:
        video_path = _find_input_video()
    except Exception as e:
        log(f"FATAL: {e}")
        logf.close()
        return 2
    log(f"  input_video={video_path}")
    log("=" * 72)

    try:
        # ── Step 1: Audio extraction ──────────────────────────────────────
        log("Step 1/7: Extracting audio...")
        audio_path = os.path.join(temp_dir, "original_audio.wav")
        extract_audio(video_path, audio_path, log_fn=log)

        # ── Step 2: Source separation (optional Demucs) ───────────────────
        vocals_path = audio_path
        background_path = None
        if use_demucs:
            log("Step 2/7: Demucs source separation...")
            sep_dir = os.path.join(temp_dir, "separated")
            try:
                stems = separate_audio(audio_path, sep_dir, log_fn=log)
                vocals_path = stems["vocals"]
                background_path = stems["background"]
            except Exception as e:
                log(f"  WARNING: Demucs failed: {e}. Using full audio for transcription.")
        else:
            log("Step 2/7: Source separation skipped.")

        # ── Step 3: Transcription ─────────────────────────────────────────
        log(f"Step 3/7: Whisper transcription ({model_size})...")
        segments = transcribe_audio(vocals_path, model_size=model_size, log_fn=log)
        generate_srt(segments, os.path.join(out_dir, "english_subtitles.srt"))
        log(f"  {len(segments)} segments transcribed.")

        # ── Step 4: Isochrony-aware translation ───────────────────────────
        log(f"Step 4/7: Isochrony-aware translation -> {target_lang}...")
        translated_segments = translate_segments_isochrony(segments, target_lang, api_key, log_fn=log)
        avg_iso = sum(s.get("isochrony_score", 0) for s in translated_segments) / max(1, len(translated_segments))
        log(f"  avg isochrony score: {avg_iso:.3f}")
        translated_srt_path = os.path.join(out_dir, f"{target_lang}_subtitles.srt")
        generate_srt(translated_segments, translated_srt_path)

        # ── Step 5: Voice reference extraction ────────────────────────────
        ref_audio_path = None
        ref_text = None
        # DUBBING_VOICE_CLONE=0 forces Basic mode: duration_tts uses a native Hindi reference
        # instead of cloning the source (English) speaker. This is the in-language control that
        # isolates the cross-lingual onset-babble cause — IndicF5 is Indic-trained, so an English
        # ref (audio+text) destabilizes its onset when generating Hindi. Basic mode both proves
        # that cause (if the babble vanishes) and ships a clean, generic-voice dub. Default "1"
        # keeps voice cloning on (premium mode), unchanged.
        voice_clone = os.environ.get("DUBBING_VOICE_CLONE", "1").strip().lower() not in ("0", "false", "no", "")
        if not voice_clone:
            log("Step 5/7: voice cloning DISABLED (DUBBING_VOICE_CLONE=0) — Basic mode (native Hindi reference).")
        elif vocals_path and os.path.exists(vocals_path):
            log("Step 5/7: Extracting reference voice clip...")
            ref_audio_path = os.path.join(temp_dir, "voice_reference.wav")
            try:
                ref_audio_path, ref_text = extract_reference_clip(
                    vocals_path, ref_audio_path, segments=segments, log_fn=log
                )
            except Exception as e:
                log(f"  WARNING: voice extraction failed: {e}. Falling back to basic voice.")
                ref_audio_path = None
                ref_text = None
        else:
            log("Step 5/7: skipped (no vocals track).")

        # ── Step 6: Duration-controlled TTS (SUPERVISED subprocess) ───────
        log(f"Step 6/7: Supervised IndicF5 TTS for {len(translated_segments)} segments...")
        tts_dir = os.path.join(out_dir, "tts_chunks")
        translated_segments = generate_tts_supervised(
            translated_segments,
            target_language=target_lang,
            output_dir=tts_dir,
            reference_audio_path=ref_audio_path,
            reference_text=ref_text,
            log_fn=log,
        )

        # ── Step 7: Assembly + video merge ────────────────────────────────
        log("Step 7/7: Assembling audio and merging video...")
        synced_audio_path = os.path.join(out_dir, "dubbed_audio.wav")
        sync_audio_segments(
            translated_segments,
            synced_audio_path,
            background_audio_path=background_path,
            background_volume=bg_volume,
            log_fn=log,
        )
        final_video_path = os.path.join(out_dir, f"dubbed_{target_lang.lower()}.mp4")
        merge_video_audio_subs(video_path, synced_audio_path, translated_srt_path, final_video_path, log_fn=log)

    except Exception as e:
        log(f"FATAL: pipeline stage failed: {e}")
        log(traceback.format_exc())
        logf.close()
        return 1

    dt = time.time() - t0
    log("=" * 72)
    log(f"DONE in {dt / 60:.1f} min. Final video: {final_video_path}")
    # Report any segments the supervisor had to degrade to silence.
    try:
        with open(os.path.join(tts_dir, "tts_manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        forced = sorted(int(k) for k, v in manifest.items() if isinstance(v, dict) and v.get("forced_silence"))
        total = len(translated_segments)
        if forced:
            log(f"DEGRADED: {len(forced)}/{total} segment(s) are silence fallbacks: {forced}")
        else:
            log(f"All {total} segment(s) synthesized as real audio (no silence fallbacks).")
    except Exception:
        pass
    log("=" * 72)
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
