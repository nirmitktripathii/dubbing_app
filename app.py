import streamlit as st
import os
import json
import hashlib
import torch

from utils.audio_extraction import extract_audio
from utils.transcription import transcribe_audio, generate_srt
from utils.video_merge import merge_video_audio_subs
from utils.audio_sync import sync_audio_segments

# New pipeline modules
from pipeline.source_separation import separate_audio
from pipeline.isochrony_translation import (
    translate_segments_isochrony,
    DISPLAY_TO_INTERNAL,
)
from pipeline.duration_tts import (
    generate_tts_for_segments,
    unload_indicf5,
    MANIFEST_NAME,
)
from pipeline.tts_supervisor import generate_tts_supervised
from pipeline.voice_manager import extract_reference_clip

# ── Steps 1-5 resume cache ────────────────────────────────────────────────────
# Steps 1-5 (extract → separate → transcribe → translate → voice-ref) are both
# expensive AND not bit-reproducible: Whisper's temperature fallback and Gemini's
# decoding can yield a DIFFERENT segment set on a re-run. The Step-6 manifest keys
# each WAV to its (text, duration) signature, so it never glues stale audio onto
# changed text — but if 1-5 re-ran and drifted, "resume" would re-synthesize
# almost everything. So we persist the 1-5 result and, when the inputs are
# unchanged, let the user JUMP straight to Step 6 with the exact same segments —
# which makes reconciliation a non-issue (1-5 can't drift if they don't re-run).
TEMP_DIR = "temp_processing"
RUN_CACHE_NAME = "run_cache.json"


def _run_key(video_hash: str, target_lang: str, model_size: str, use_demucs: bool) -> str:
    """Identity of a Steps 1-5 result. Everything that changes 1-5 output belongs here:
    the video bytes (Steps 1,2,3,5), the target language (Step 4), the Whisper model
    (Step 3), and Demucs on/off (Steps 2,3,5). bg_volume/tier/nfe_step are deliberately
    NOT here — they only affect Steps 6-7, which never reuse cached 1-5 output."""
    payload = f"{video_hash}|{target_lang}|{model_size}|{int(bool(use_demucs))}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _json_default(o):
    """Coerce numpy scalars (e.g. an isochrony score) to plain JSON types on save."""
    import numpy as _np
    if isinstance(o, _np.generic):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _load_run_cache(temp_dir):
    try:
        with open(os.path.join(temp_dir, RUN_CACHE_NAME), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_run_cache(temp_dir, data) -> bool:
    """Persist the Steps 1-5 result atomically. Best-effort: a failure just means resume
    won't be offered next time — it must never abort the run about to enter Step 6."""
    try:
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, RUN_CACHE_NAME)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=_json_default)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _manifest_done_count(tts_dir) -> int:
    """How many Step-6 segments a prior run already finished as real audio ('ok')."""
    try:
        with open(os.path.join(tts_dir, MANIFEST_NAME), "r", encoding="utf-8") as f:
            m = json.load(f)
        return sum(1 for v in m.values() if isinstance(v, dict) and v.get("status") == "ok")
    except Exception:
        return 0


def _resume_available(temp_dir, run_key):
    """Return the cached Steps 1-5 payload IFF it matches the current inputs AND the saved
    video is still on disk; else None (so resume isn't offered). Optional artifacts
    (voice-ref, background) are validated later and degraded gracefully, not required here."""
    cache = _load_run_cache(temp_dir)
    if not cache or cache.get("run_key") != run_key:
        return None
    if not cache.get("translated_segments"):
        return None
    if not os.path.exists(os.path.join(temp_dir, "input_video.mp4")):
        return None
    return cache

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indic AI Dubbing",
    page_icon="🎙️",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Required for isochrony-aware translation.",
    )

    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        help="Required to download the gated IndicF5 model weights. Make sure you accept the agreement at https://huggingface.co/ai4bharat/IndicF5 first.",
    )

    st.markdown("---")
    st.markdown("### 🌐 Languages")
    source_lang = st.selectbox("Source Language", ["English"], disabled=True)
    target_lang = st.selectbox(
        "Target Language",
        list(DISPLAY_TO_INTERNAL.keys()),
        index=0,
        help="All 11 Indic languages supported. Hindi is the primary & most tested.",
    )

    st.markdown("---")
    st.markdown("### 🎚️ Tier")
    tier = st.radio(
        "Dubbing Tier",
        ["Basic — Generic voice ($0.02/min)", "Premium — Voice Cloning ($0.05/min)"],
        index=0,
        help="Premium extracts the original speaker's voice and clones it into the dubbed audio.",
    )
    is_premium = tier.startswith("Premium")

    st.markdown("---")
    st.markdown("### 🔊 Source Separation")
    use_demucs = st.checkbox(
        "Separate vocals/background (Demucs)",
        value=True,
        help="Preserves original background music/SFX. Recommended but adds ~60s.",
    )

    st.markdown("---")
    st.markdown("### 🤖 Transcription")
    model_size = st.selectbox(
        "Whisper model",
        ["large-v3", "medium", "small", "base"],
        index=0,
        help="large-v3 = best quality (needs ~1.5GB VRAM). base = fastest.",
    )

    st.markdown("---")
    st.markdown("### 📊 Background mix volume")
    bg_volume = st.slider(
        "Background track volume",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        disabled=not use_demucs,
        help="How loud the original music/SFX is relative to the dubbed voice.",
    )

# ── Main panel ───────────────────────────────────────────────────────────────
st.title("🎙️ Indic AI Dubbing Platform")
st.markdown(
    "Upload a video. Get a professionally dubbed Indic-language version "
    "with **isochrony-aware translation** and **duration-controlled TTS**."
)

st.info(
    "**How this works differently from other dubbing tools:**\n"
    "- Translation is constrained to match source phoneme count *(isochrony-aware)*\n"
    "- TTS generates audio at the exact target duration — no post-processing stretch\n"
    "- Background music/SFX preserved via source separation\n"
    "- Supports all 11 Indic languages via IndicF5"
)

uploaded_file = st.file_uploader(
    "Upload Video (MP4, MKV, MOV — up to 15 min)",
    type=["mp4", "mkv", "mov"],
)

run_key = None
resume_info = None
if uploaded_file:
    _size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)
    st.caption(f"✓ Received **{uploaded_file.name}** — {_size_mb:.1f} MB")
    st.video(uploaded_file)

    # Hash the video ONCE per upload (cached in session_state) so detecting a resumable
    # prior run doesn't re-hash the whole file on every Streamlit rerun (e.g. slider moves).
    _fid = getattr(uploaded_file, "file_id", None) or f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
    if st.session_state.get("_video_fid") != _fid:
        _h = hashlib.sha1()
        _h.update(uploaded_file.getbuffer())
        st.session_state["_video_hash"] = _h.hexdigest()
        st.session_state["_video_fid"] = _fid
    run_key = _run_key(st.session_state["_video_hash"], target_lang, model_size, use_demucs)
    resume_info = _resume_available(TEMP_DIR, run_key)

start_btn = False
resume_btn = False
if resume_info:
    _done = _manifest_done_count(os.path.join(TEMP_DIR, "tts_chunks"))
    _total = len(resume_info["translated_segments"])
    st.info(
        f"↩️ A previous run of **this exact video + language + settings** was found. "
        f"Steps 1-5 (transcription, translation, voice profile) are cached, and Step 6 "
        f"had finished **{_done}/{_total}** segments. You can continue from Step 6, or start over."
    )
    col_a, col_b, _ = st.columns([1.4, 1.4, 2.2])
    with col_a:
        resume_btn = st.button(
            f"▶️ Resume from Step 6 — {_done}/{_total} done",
            type="primary", use_container_width=True,
        )
    with col_b:
        start_btn = st.button(
            "🔄 Start over", use_container_width=True,
            help="Re-run every step: transcribe, translate, and re-synthesize all segments from scratch.",
        )
else:
    col_start, _ = st.columns([1, 4])
    with col_start:
        start_btn = st.button("🚀 Start Dubbing", type="primary", use_container_width=True)

if start_btn or resume_btn:
    resume_mode = bool(resume_btn)
    if not uploaded_file:
        st.error("Please upload a video file first.")
        st.stop()
    # Translation (Step 4) is skipped on resume, so the Gemini key isn't needed then.
    if not resume_mode and not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    if not hf_token:
        st.error("Please enter your Hugging Face Token in the sidebar. This is required to access the gated IndicF5 model.")
        st.stop()

    os.environ["HF_TOKEN"] = hf_token

    temp_dir = TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)

    video_path = os.path.join(temp_dir, "input_video.mp4")

    if resume_mode and os.path.exists(video_path):
        # The cache is keyed on the video hash, so the saved copy is byte-identical
        # to the current upload — no need to re-write it.
        st.caption("↩️ Resuming — reusing the already-saved upload.")
    else:
        # Save the uploaded file to disk in chunks so we can show a real progress
        # bar. NOTE: Streamlit's st.file_uploader already streams the browser→server
        # network upload with its own built-in progress indicator; by the time this
        # code runs the bytes are fully in memory. This bar therefore reports the
        # disk-write (buffer → temp file) phase, which is the part we control.
        buffer = uploaded_file.getbuffer()
        total_bytes = buffer.nbytes
        save_progress = st.progress(0.0)
        save_status = st.empty()
        chunk_size = 4 * 1024 * 1024  # 4 MB
        written = 0
        with open(video_path, "wb") as f:
            while written < total_bytes:
                chunk = buffer[written:written + chunk_size]
                f.write(chunk)
                written += len(chunk)
                frac = written / total_bytes if total_bytes else 1.0
                save_progress.progress(frac)
                save_status.caption(
                    f"Saving upload… {written / (1024*1024):.1f} / "
                    f"{total_bytes / (1024*1024):.1f} MB ({frac*100:.0f}%)"
                )
        save_status.caption(f"✓ Upload saved ({total_bytes / (1024*1024):.1f} MB)")

    progress = st.progress(0)
    status = st.empty()
    log_expander = st.expander("📋 Pipeline Log", expanded=True)
    log_area = log_expander.empty()
    logs = []

    def log(msg):
        # Keep the FULL history (no truncation) and render it in a fixed-height,
        # scrollable, monospace box so long-running steps (esp. Step 6 TTS) stay
        # visible and reviewable in the UI instead of scrolling off the top.
        logs.append(msg)
        import html as _html
        body = _html.escape("\n".join(logs))
        log_area.markdown(
            '<div id="pipeline-log-box" style="height:340px;overflow-y:auto;'
            'white-space:pre-wrap;word-break:break-word;font-family:monospace;'
            'font-size:12px;line-height:1.45;background:#0e1117;color:#d4d4d4;'
            'padding:10px 12px;border-radius:6px;border:1px solid #262730;">'
            f'{body}</div>'
            '<script>var _b=document.getElementById("pipeline-log-box");'
            'if(_b){_b.scrollTop=_b.scrollHeight;}</script>',
            unsafe_allow_html=True,
        )

    try:
        if resume_mode:
            # ── Resume: load the cached Steps 1-5 result and jump to Step 6 ────
            status.markdown("**Resuming** — loading cached Steps 1-5, jumping to Step 6...")
            log("▶️ Resume mode: loading cached transcription / translation / voice-ref (Steps 1-5).")
            cache = resume_info or _load_run_cache(temp_dir)
            if not cache or not cache.get("translated_segments"):
                raise RuntimeError("Resume cache is missing or empty — please Start over.")
            segments = cache.get("segments") or []
            translated_segments = cache["translated_segments"]
            ref_audio_path = cache.get("ref_audio_path")
            ref_text = cache.get("ref_text")
            background_path = cache.get("background_path")
            vocals_path = cache.get("vocals_path")

            # Optional artifacts may have been cleaned up between runs — degrade
            # gracefully rather than crashing, so the expensive Step-6 work still resumes.
            if ref_audio_path and not os.path.exists(ref_audio_path):
                log("  ⚠️ Cached voice reference file is gone; falling back to the basic voice.")
                ref_audio_path = None
            if background_path and not os.path.exists(background_path):
                log("  ⚠️ Cached background track is gone; the final mix will omit it.")
                background_path = None

            # The translated SRT is a deterministic function of the cached segments —
            # regenerate it (needed for the final merge + the subtitles download).
            translated_srt_path = os.path.join(temp_dir, f"{target_lang}_subtitles.srt")
            generate_srt(translated_segments, translated_srt_path)
            avg_iso = sum(s.get("isochrony_score", 0) for s in translated_segments) / max(1, len(translated_segments))
            st.metric("Avg Isochrony Score", f"{avg_iso:.3f}", help="from the cached translation")
            log(f"  ✓ Loaded {len(translated_segments)} cached segments. Resuming at Step 6.")
            progress.progress(58)
        else:
            # ── Step 1: Audio extraction ──────────────────────────────────────
            status.markdown("**Step 1/7** — Extracting audio from video...")
            log("Step 1: Extracting audio...")
            audio_path = os.path.join(temp_dir, "original_audio.wav")
            extract_audio(video_path, audio_path, log_fn=log)
            log(f"  ✓ Audio extracted: {audio_path}")
            progress.progress(5)

            # ── Step 2: Source separation (optional Demucs) ───────────────────
            vocals_path = audio_path
            background_path = None

            if use_demucs:
                status.markdown("**Step 2/7** — Separating vocals and background (Demucs)...")
                log("Step 2: Running Demucs source separation...")
                sep_dir = os.path.join(temp_dir, "separated")
                try:
                    stems = separate_audio(audio_path, sep_dir, log_fn=log)
                    vocals_path = stems["vocals"]
                    background_path = stems["background"]
                    log(f"  ✓ Vocals: {vocals_path}")
                    log(f"  ✓ Background: {background_path}")
                except Exception as e:
                    log(f"  ⚠️ Demucs failed: {e}. Using full audio for transcription.")
                    st.warning(f"Source separation failed ({e}). Proceeding without background preservation.")
            else:
                log("Step 2: Source separation skipped by user.")
            progress.progress(15)

            # ── Step 3: Transcription ─────────────────────────────────────────
            status.markdown(f"**Step 3/7** — Transcribing with Whisper {model_size}...")
            log(f"Step 3: Transcribing audio with Whisper ({model_size})...")
            segments = transcribe_audio(vocals_path, model_size=model_size, log_fn=log)
            english_srt_path = os.path.join(temp_dir, "english_subtitles.srt")
            generate_srt(segments, english_srt_path)
            log(f"  ✓ {len(segments)} segments transcribed.")
            progress.progress(30)

            # ── Step 4: Isochrony-aware translation ───────────────────────────
            status.markdown(f"**Step 4/7** — Isochrony-aware translation → {target_lang}...")
            log(f"Step 4: Translating to {target_lang} with phoneme-budget constraints...")
            translated_segments = translate_segments_isochrony(
                segments, target_lang, api_key, log_fn=log
            )
            avg_iso = sum(s.get("isochrony_score", 0) for s in translated_segments) / max(1, len(translated_segments))
            log(f"  ✓ Translation complete. Avg isochrony score: {avg_iso:.3f}")
            translated_srt_path = os.path.join(temp_dir, f"{target_lang}_subtitles.srt")
            generate_srt(translated_segments, translated_srt_path)

            st.metric("Avg Isochrony Score", f"{avg_iso:.3f}", help="≥0.85 = excellent timing compliance")
            progress.progress(50)

            # ── Step 5: Voice reference extraction (cloning reference) ─────────
            ref_audio_path = None
            ref_text = None

            # IndicF5 requires a valid reference audio clip to synthesize audio.
            # We always extract the speaker's reference voice from the vocals track.
            if vocals_path and os.path.exists(vocals_path):
                status.markdown("**Step 5/7** — Extracting voice profile for cloning...")
                log("Step 5: Extracting reference voice clip...")
                ref_audio_path = os.path.join(temp_dir, "voice_reference.wav")
                try:
                    ref_audio_path, ref_text = extract_reference_clip(vocals_path, ref_audio_path, segments=segments, log_fn=log)
                    log(f"  ✓ Voice reference saved: {ref_audio_path}")
                except Exception as e:
                    log(f"  ⚠️ Voice extraction failed: {e}. Synthesis may fail.")
                    ref_audio_path = None
                    ref_text = None
            else:
                log("Step 5: Voice reference skipped (no vocals track available).")

            # Persist the Steps 1-5 result so a Step-6 interruption can resume from HERE
            # with the exact same segments (no re-transcribe / re-translate drift).
            # Best-effort — a write failure must not abort the run entering Step 6.
            if _save_run_cache(temp_dir, {
                "run_key": run_key,
                "target_lang": target_lang,
                "model_size": model_size,
                "use_demucs": bool(use_demucs),
                "segments": segments,
                "translated_segments": translated_segments,
                "ref_audio_path": ref_audio_path,
                "ref_text": ref_text,
                "background_path": background_path,
                "vocals_path": vocals_path,
            }):
                log("  ✓ Cached Steps 1-5 — a Step-6 interruption can resume from here.")
            else:
                log("  ⚠️ Could not write the resume cache; a re-run would restart from Step 1.")
            progress.progress(58)

        # ── Step 6: Duration-controlled TTS ──────────────────────────────
        status.markdown(f"**Step 6/7** — Generating {target_lang} audio with IndicF5...")
        log(f"Step 6: Running IndicF5 TTS for {len(translated_segments)} segments...")
        tts_dir = os.path.join(temp_dir, "tts_chunks")
        if not resume_mode:
            # "Start over" must re-synthesize every segment — drop any manifest from a
            # previous run so nothing is skipped. Resume deliberately KEEPS it: that's how
            # it continues from the exact segment the last run stopped at.
            os.makedirs(tts_dir, exist_ok=True)
            try:
                os.remove(os.path.join(tts_dir, MANIFEST_NAME))
            except OSError:
                pass
        # Supervised subprocess synthesis: Step 6 runs in a separate OS process that this
        # parent SIGKILLs + relaunches if its heartbeat stalls (a wedged CUDA op can freeze
        # its own process but never this one). Poison segments degrade to silence; progress
        # is checkpointed to disk, so a killed worker resumes seamlessly. This is the
        # process-isolation cure for the Step-6 GPU freeze — see pipeline/tts_supervisor.py.
        translated_segments = generate_tts_supervised(
            translated_segments,
            target_language=target_lang,
            output_dir=tts_dir,
            reference_audio_path=ref_audio_path,
            reference_text=ref_text,
            log_fn=log,
        )
        log(f"  ✓ TTS complete: {len(translated_segments)} audio chunks generated.")

        # IndicF5 lived in the (now-exited) worker process, so its VRAM is already freed by
        # process teardown. This call is a harmless no-op in the parent (which never loaded
        # the model) — kept as defensive cleanup in case that ever changes.
        unload_indicf5()
        progress.progress(80)

        # ── Step 7: Audio assembly + video merge ──────────────────────────
        status.markdown("**Step 7/7** — Assembling final dubbed video...")
        log("Step 7: Assembling and mixing audio tracks...")
        synced_audio_path = os.path.join(temp_dir, "dubbed_audio.wav")
        sync_audio_segments(
            translated_segments,
            synced_audio_path,
            background_audio_path=background_path,
            background_volume=bg_volume,
            log_fn=log,
        )

        log("  Merging video, audio, and subtitles with FFmpeg...")
        final_video_path = os.path.join(temp_dir, "final_dubbed_video.mp4")
        # v2 UI keeps hard-burned captions (its prior behaviour); the production headless/Modal
        # path defaults to the cheaper soft-mux + stream-copy. See video_merge.py / AUDIT P2.
        merge_video_audio_subs(
            video_path, synced_audio_path, translated_srt_path, final_video_path,
            log_fn=log, burn_subs=True,
        )
        progress.progress(100)
        status.markdown("✅ **Dubbing complete!**")
        log(f"  ✓ Final video: {final_video_path}")

        # ── Results ───────────────────────────────────────────────────────
        st.success("🎉 Your dubbed video is ready!")
        st.video(final_video_path)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            with open(final_video_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Video",
                    f,
                    file_name=f"dubbed_{target_lang.lower()}.mp4",
                    mime="video/mp4",
                )
        with col_d2:
            with open(translated_srt_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Subtitles",
                    f,
                    file_name=f"{target_lang}.srt",
                    mime="text/plain",
                )
        with col_d3:
            with open(synced_audio_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Audio",
                    f,
                    file_name=f"dubbed_{target_lang.lower()}.wav",
                    mime="audio/wav",
                )

    except Exception as e:
        import traceback
        st.error(f"Pipeline error: {str(e)}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())
        log(f"ERROR: {e}")
