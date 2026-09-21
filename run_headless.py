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
    DUBBING_BURN_SUBS     "1"/"0" — hard-burn captions (full re-encode) vs soft-mux them
                          and stream-copy the video (default "0" = soft-mux, no re-encode).
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
import queue
import threading
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
from pipeline.voice_conversion import convert_segments_timbre

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


def _voice_mode(raw):
    """Map DUBBING_VOICE_CLONE to a voice path: 'basic' | 'vc' | 'xlingual'.

    0/basic/off      -> native TTS voice, no cloning (clean, ships today; safe default).
    2/vc             -> PREMIUM: native TTS + Step-6.5 voice conversion (clone timbre).
    1/xlingual/clone -> DEPRECATED cross-lingual TTS conditioning (the onset-babble path).
    """
    v = (raw or "").strip().lower()
    if v in ("0", "basic", "off", "false", "no", ""):
        return "basic"
    if v in ("2", "vc"):
        return "vc"
    if v in ("1", "xlingual", "clone", "premium", "true", "yes"):
        return "xlingual"
    return "basic"


def _ordered_segment_paths(tts_dir, n):
    """Resolve the ordered per-segment WAV paths for VC, preferring the TTS manifest.

    Manifest paths may be absolute for the environment that wrote them (e.g. /kaggle/...);
    fall back to the tts_dir basename, then to the segment_NNNN.wav naming convention.
    """
    paths = []
    try:
        with open(os.path.join(tts_dir, "tts_manifest.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        for i in range(n):
            e = m.get(str(i))
            p = e.get("path") if isinstance(e, dict) else None
            if p and not os.path.exists(p):
                p = os.path.join(tts_dir, os.path.basename(p))
            if p and os.path.exists(p):
                paths.append(p)
    except Exception:
        pass
    if len(paths) != n:
        paths = [os.path.join(tts_dir, f"segment_{i:04d}.wav") for i in range(n)]
    return paths


def _json_default(o):
    """Make numpy scalars/arrays JSON-serializable. Segment dicts pick up np.bool_/int64/
    float64 from the isochrony gate math, and a bare json.dump raises "not serializable" on
    them (float64 happens to work, bool_/int64 do not) — a landmine that has crashed headless
    output before. Used by _save_state so the stage checkpoint round-trips cleanly."""
    try:
        import numpy as _np
    except Exception:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


_STATE_NAME = "pipeline_state.json"


def _save_state(out_dir, state):
    """Persist the between-stages checkpoint atomically (temp + os.replace)."""
    import tempfile
    path = os.path.join(out_dir, _STATE_NAME)
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)
    return path


def _load_state(out_dir):
    with open(os.path.join(out_dir, _STATE_NAME), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    t0 = time.time()
    out_dir = os.environ.get("DUBBING_OUTPUT_DIR", "/kaggle/working/dubbing_output")
    temp_dir = os.path.join(out_dir, "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)

    logf = open(os.path.join(out_dir, "pipeline_log.txt"), "a", encoding="utf-8")

    # ── Console sink: a NON-BLOCKING mirror of the durable log ──────────────────────────
    # The source of truth is pipeline_log.txt (written+flushed synchronously in log() below).
    # The live console is only a best-effort mirror. On Kaggle this process is launched by the
    # notebook cell via subprocess.run WITHOUT a stdout pipe of its own, so it inherits the
    # KERNEL's stdout — a ~64 KB OS pipe drained by Kaggle's output relay (IOPub, rate-limited).
    # If that relay stops draining (rate-limit trip, disconnected tab), the pipe fills and a
    # blocking print() wedges every thread that logs. That is exactly how a run whose TTS had
    # ALREADY finished all segments hung after "Segment 12/13" with no Step 7 and a log that
    # stopped mid-line: print() blocked BEFORE the file write, freezing the cell and the file
    # at the same point. So console writes go through a bounded queue drained by a daemon
    # thread; if the console stalls we DROP console lines (never block) while the file keeps
    # every line. Project rule 3: Kaggle only publishes the log at session end anyway — the
    # live channel is W&B / the on-disk file, not this mirror.
    _console_q: "queue.Queue" = queue.Queue(maxsize=2000)

    def _console_writer():
        while True:
            item = _console_q.get()
            if item is None:
                break
            try:
                sys.stdout.write(item)
                sys.stdout.flush()
            except Exception:
                pass

    threading.Thread(target=_console_writer, name="console-sink", daemon=True).start()

    def log(msg=""):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        # 1) Durable file FIRST — must never be lost to a stalled console; flush so a killed
        #    session still has the tail on disk.
        try:
            logf.write(line + "\n")
            logf.flush()
        except Exception:
            pass
        # 2) Best-effort live mirror — non-blocking; drop under backpressure, never wedge.
        try:
            _console_q.put_nowait(line + "\n")
        except queue.Full:
            pass

    target_lang = os.environ.get("DUBBING_TARGET_LANG", "Hindi").strip() or "Hindi"
    model_size = os.environ.get("DUBBING_WHISPER_MODEL", "medium").strip() or "medium"
    use_demucs = os.environ.get("DUBBING_USE_DEMUCS", "1").strip().lower() not in ("0", "false", "no", "")
    # Default OFF: soft-mux subtitles + stream-copy the video (no re-encode). Set "1" to hard-
    # burn captions into the pixels (forces a full libx264 re-encode). See video_merge.py / P2.
    burn_subs = os.environ.get("DUBBING_BURN_SUBS", "0").strip().lower() in ("1", "true", "yes")
    try:
        bg_volume = float(os.environ.get("DUBBING_BG_VOLUME", "0.3"))
    except ValueError:
        bg_volume = 0.3
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    # DUBBING_STAGES selects which slice of the pipeline this process runs, for the Modal
    # translation-off-GPU split. "all" (default) is the single-process Kaggle/serial flow,
    # unchanged. "prep" runs the GPU-bound Steps 1-3, writes a checkpoint and exits so the GPU
    # is freed before the CPU-only translation. "resume" loads that checkpoint and runs Step 4
    # onward. Kaggle never sets this, so its behaviour is byte-for-byte identical.
    stages = os.environ.get("DUBBING_STAGES", "all").strip().lower()
    if stages not in ("all", "prep", "resume"):
        stages = "all"

    log("=" * 72)
    log("Indic Dubbing — HEADLESS BATCH RUN")
    log(f"  target_lang={target_lang}  whisper={model_size}  demucs={use_demucs}  "
        f"bg_vol={bg_volume}  burn_subs={burn_subs}  stages={stages}")
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
        if stages in ("all", "prep"):
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

            # ── Stage checkpoint (Modal translation-off-GPU split) ────────────
            # In the split deployment the GPU-bound stages above run in a GPU container, which
            # then persists this checkpoint and EXITS so the GPU is released before the ~60% of
            # wall clock that Step 4 spends in CPU-only Gemini translation. A CPU orchestrator
            # re-invokes with DUBBING_STAGES=resume to run Step 4 onward. In "all" mode (Kaggle
            # / serial) this never fires — no persist, no exit — so that flow is unchanged.
            if stages == "prep":
                _save_state(out_dir, {
                    "segments": segments,
                    "vocals_path": vocals_path,
                    "background_path": background_path,
                    "video_path": video_path,
                })
                log(f"PREP complete ({len(segments)} segments): checkpoint saved to "
                    f"{_STATE_NAME}; GPU released. Resume Step 4+ with DUBBING_STAGES=resume.")
                logf.close()
                return 0
        else:
            # ── RESUME: load the prep checkpoint, skip the GPU-bound stages ────
            log(f"Resuming from {_STATE_NAME} (Steps 1-3 already done on GPU)...")
            state = _load_state(out_dir)
            segments = state["segments"]
            vocals_path = state.get("vocals_path")
            background_path = state.get("background_path")
            video_path = state.get("video_path") or video_path
            log(f"  loaded {len(segments)} segments; running Step 4 onward "
                "(translation + TTS + assembly).")

        # ── Step 4: Isochrony-aware translation ───────────────────────────
        log(f"Step 4/7: Isochrony-aware translation -> {target_lang}...")
        translated_segments = translate_segments_isochrony(segments, target_lang, api_key, log_fn=log)
        avg_iso = sum(s.get("isochrony_score", 0) for s in translated_segments) / max(1, len(translated_segments))
        log(f"  avg isochrony score: {avg_iso:.3f}")
        translated_srt_path = os.path.join(out_dir, f"{target_lang}_subtitles.srt")
        generate_srt(translated_segments, translated_srt_path)

        # ── Step 5: Voice reference extraction ────────────────────────────
        # Voice path is selected by DUBBING_VOICE_CLONE (see _voice_mode):
        #   basic    -> native TTS voice (reference=None). Clean, generic, ships today.
        #   vc       -> PREMIUM: native TTS (clean Hindi) + Step-6.5 voice conversion that
        #               transfers the SOURCE speaker's timbre onto the clean Hindi. No
        #               cross-lingual TTS conditioning, so no onset babble.
        #   xlingual -> DEPRECATED: English reference fed straight into IndicF5. This is the
        #               cross-lingual path proven to produce onset babble; kept only for A/B.
        # Default is Basic — the pipeline never silently emits the garbled cross-lingual voice.
        ref_audio_path = None      # passed to TTS (only in the deprecated xlingual path)
        ref_text = None
        vc_target_path = None      # speaker reference consumed by the Step-6.5 VC pass
        mode = _voice_mode(os.environ.get("DUBBING_VOICE_CLONE", "0"))
        if mode == "basic":
            log("Step 5/7: Basic mode (DUBBING_VOICE_CLONE=0) — native TTS voice, no cloning.")
        elif not (vocals_path and os.path.exists(vocals_path)):
            log(f"Step 5/7: {mode} mode requested but no vocals track — falling back to Basic voice.")
            mode = "basic"
        else:
            log(f"Step 5/7: Extracting reference voice clip ({mode} mode)...")
            _ref_path = os.path.join(temp_dir, "voice_reference.wav")
            try:
                _ref_path, _ref_text = extract_reference_clip(
                    vocals_path, _ref_path, segments=segments, log_fn=log
                )
                if mode == "xlingual":
                    ref_audio_path, ref_text = _ref_path, _ref_text   # -> straight into TTS
                    log("  WARNING: xlingual mode is DEPRECATED (cross-lingual onset babble).")
                else:  # vc: keep the clip as the VC target; TTS stays native (ref=None)
                    vc_target_path = _ref_path
            except Exception as e:
                log(f"  WARNING: voice extraction failed: {e}. Falling back to Basic voice.")
                mode = "basic"

        # ── Step 6: Duration-controlled TTS (SUPERVISED subprocess) ───────
        # DUBBING_TTS_BACKEND selects HOW Step 6 executes; it never changes WHAT is
        # synthesized. Both backends end with the same segment WAVs + tts_manifest.json in
        # tts_dir and the same returned segments, so Steps 6.5 and 7 are identical either
        # way — and a run started under one backend resumes under the other.
        #   "" / "supervised" (default) — the proven single-process supervisor path with the
        #       SIGKILL-and-relaunch freeze fix. This is what Kaggle uses; unchanged.
        #   "modal-fanout" — shard the segments across parallel Modal GPU workers
        #       (deploy/tts_fanout.py). Only meaningful inside the Modal deployment, where
        #       there are containers to fan out to.
        tts_dir = os.path.join(out_dir, "tts_chunks")
        tts_backend = os.environ.get("DUBBING_TTS_BACKEND", "").strip().lower()
        if tts_backend in ("modal-fanout", "fanout"):
            # Fan-out earns its extra per-container model loads only when there are enough
            # segments that parallel synthesis outweighs them. Below the threshold, cap to a
            # SINGLE shard: one TTSEngine container, one model load, segments synthesized
            # serially on it — cheaper than spinning several containers that each load IndicF5
            # for a handful of segments (each load costs more than a segment; see the wall-clock
            # model in tts_fanout.py). Tunable via DUBBING_TTS_FANOUT_MIN_SEGMENTS (default 12).
            min_fanout = int(os.environ.get("DUBBING_TTS_FANOUT_MIN_SEGMENTS", "12"))
            fanout_kw = {}
            if len(translated_segments) < min_fanout:
                fanout_kw["max_shards"] = 1
            log(f"Step 6/7: Fan-out IndicF5 TTS for {len(translated_segments)} segments "
                f"(backend={tts_backend}, "
                f"shards={'1 (below threshold ' + str(min_fanout) + ')' if fanout_kw else 'auto'})...")
            from deploy.tts_fanout import generate_tts_fanout
            translated_segments = generate_tts_fanout(
                translated_segments,
                target_language=target_lang,
                output_dir=tts_dir,
                reference_audio_path=ref_audio_path,
                reference_text=ref_text,
                log_fn=log,
                **fanout_kw,
            )
        else:
            log(f"Step 6/7: Supervised IndicF5 TTS for {len(translated_segments)} segments...")
            translated_segments = generate_tts_supervised(
                translated_segments,
                target_language=target_lang,
                output_dir=tts_dir,
                reference_audio_path=ref_audio_path,
                reference_text=ref_text,
                log_fn=log,
            )

        # ── Step 6.5: Voice conversion (PREMIUM cloning path) ──────────────
        # Clone the source speaker's timbre onto the clean native-TTS Hindi. VC is
        # duration-preserving, so the isochrony achieved in Step 6 survives to the sample.
        # Degrades to a visible pass-through (native voice) if the VC backend is unavailable —
        # it never crashes a render.
        if mode == "vc" and vc_target_path:
            log("Step 6.5/7: Voice conversion — cloning source speaker onto clean Hindi...")
            seg_paths = _ordered_segment_paths(tts_dir, len(translated_segments))
            vc_stats = convert_segments_timbre(seg_paths, vc_target_path, log_fn=log)
            n_ok = len(vc_stats.get("converted", []))
            n_pt = len(vc_stats.get("passthrough", []))
            n_fail = len(vc_stats.get("failed", []))
            if n_ok == 0:
                log(f"  WARNING: voice conversion converted 0 segment(s) "
                    f"(pass-through={n_pt}, failed={n_fail}); output is the native TTS voice.")
            else:
                log(f"  Voice conversion: {n_ok} converted, {n_pt} pass-through, {n_fail} failed "
                    f"(backend={vc_stats.get('backend')}).")

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
        merge_video_audio_subs(video_path, synced_audio_path, translated_srt_path, final_video_path,
                               log_fn=log, burn_subs=burn_subs)

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
