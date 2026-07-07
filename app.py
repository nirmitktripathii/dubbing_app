import streamlit as st
import os
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
)
from pipeline.voice_manager import extract_reference_clip

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

if uploaded_file:
    st.video(uploaded_file)

col_start, _ = st.columns([1, 4])
with col_start:
    start_btn = st.button("🚀 Start Dubbing", type="primary", use_container_width=True)

if start_btn:
    if not uploaded_file:
        st.error("Please upload a video file first.")
        st.stop()
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    if not hf_token:
        st.error("Please enter your Hugging Face Token in the sidebar. This is required to access the gated IndicF5 model.")
        st.stop()

    os.environ["HF_TOKEN"] = hf_token

    temp_dir = "temp_processing"
    os.makedirs(temp_dir, exist_ok=True)

    video_path = os.path.join(temp_dir, "input_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    progress = st.progress(0)
    status = st.empty()
    log_expander = st.expander("📋 Pipeline Log", expanded=False)
    log_area = log_expander.empty()
    logs = []

    def log(msg):
        logs.append(msg)
        log_area.text("\n".join(logs[-30:]))

    try:
        # ── Step 1: Audio extraction ──────────────────────────────────────
        status.markdown("**Step 1/7** — Extracting audio from video...")
        log("Step 1: Extracting audio...")
        audio_path = os.path.join(temp_dir, "original_audio.wav")
        extract_audio(video_path, audio_path)
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
                stems = separate_audio(audio_path, sep_dir)
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
        segments = transcribe_audio(vocals_path, model_size=model_size)
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
                ref_audio_path, ref_text = extract_reference_clip(vocals_path, ref_audio_path, segments=segments)
                log(f"  ✓ Voice reference saved: {ref_audio_path}")
            except Exception as e:
                log(f"  ⚠️ Voice extraction failed: {e}. Synthesis may fail.")
                ref_audio_path = None
                ref_text = None
        else:
            log("Step 5: Voice reference skipped (no vocals track available).")
        progress.progress(58)

        # ── Step 6: Duration-controlled TTS ──────────────────────────────
        status.markdown(f"**Step 6/7** — Generating {target_lang} audio with IndicF5...")
        log(f"Step 6: Running IndicF5 TTS for {len(translated_segments)} segments...")
        tts_dir = os.path.join(temp_dir, "tts_chunks")
        translated_segments = generate_tts_for_segments(
            translated_segments,
            target_language=target_lang,
            output_dir=tts_dir,
            reference_audio_path=ref_audio_path,
            reference_text=ref_text,
        )
        log(f"  ✓ TTS complete: {len(translated_segments)} audio chunks generated.")

        # Unload IndicF5 to free VRAM before audio assembly
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
        )

        log("  Merging video, audio, and subtitles with FFmpeg...")
        final_video_path = os.path.join(temp_dir, "final_dubbed_video.mp4")
        merge_video_audio_subs(
            video_path, synced_audio_path, translated_srt_path, final_video_path
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
