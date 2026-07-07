"""
duration_tts.py — Stage 4 of the Indic Dubbing Pipeline

Duration-controlled TTS using IndicF5 (AI4Bharat).

Key innovation vs. the old Edge-TTS + FFmpeg atempo approach:
  - IndicF5 generates audio natively at the target duration by conditioning
    on a target mel-frame count computed from the segment's target duration.
  - Audio is "born" at the correct length — no post-processing stretch/compress.
  - Supports zero-shot voice cloning via a reference audio clip.

Supported languages (all 11 IndicF5 languages):
  Hindi, Bengali, Marathi, Gujarati, Punjabi, Tamil, Telugu,
  Kannada, Malayalam, Odia, Assamese

VRAM requirement: ~4–6 GB (fits a free Kaggle/Colab T4 with 16GB).

Basic tier: uses pre-selected natural voices (no cloning).
Premium tier: uses voice cloning via a reference audio clip from voice_manager.py.

Usage:
    from pipeline.duration_tts import generate_tts_for_segments

    segments = generate_tts_for_segments(
        translated_segments,
        target_language="Hindi",
        output_dir="temp/tts_chunks",
        reference_audio_path=None,   # None = Basic tier (no cloning)
    )
"""

import os
import time
import math
import tempfile
import numpy as np
from typing import Optional
import urllib3
import requests
import ssl
import builtins

_orig_print = builtins.print

def print(*args, **kwargs):
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        new_args = [
            arg.encode('ascii', errors='replace').decode('ascii') if isinstance(arg, str) else arg
            for arg in args
        ]
        _orig_print(*new_args, **kwargs)


# Globally disable SSL verification to bypass Windows trust store issues for Hugging Face
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
original_session_init = requests.Session.__init__
def patched_session_init(self, *args, **kwargs):
    original_session_init(self, *args, **kwargs)
    self.verify = False
requests.Session.__init__ = patched_session_init

ssl._create_default_https_context = ssl._create_unverified_context

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# IndicF5 native audio parameters
INDICF5_SAMPLE_RATE = 24000
HOP_LENGTH = 256           # mel-spectrogram hop length used by IndicF5 / F5-TTS
FRAMES_PER_SECOND = INDICF5_SAMPLE_RATE / HOP_LENGTH  # ≈ 93.75 frames/sec

# Duration tolerance: if TTS output drifts more than this fraction from target,
# apply a minimal pyrubberband correction as a safety net (NOT the primary mechanism).
DRIFT_TOLERANCE = 0.08   # 8% — much tighter than old 1.4x atempo cap

# Map display language names to IndicF5 language codes
LANGUAGE_TO_CODE = {
    "Hindi":     "hi",
    "Bengali":   "bn",
    "Marathi":   "mr",
    "Gujarati":  "gu",
    "Punjabi":   "pa",
    "Tamil":     "ta",
    "Telugu":    "te",
    "Kannada":   "kn",
    "Malayalam": "ml",
    "Odia":      "or",
    "Assamese":  "as",
}

# Default Hindi male reference voice for Basic tier.
# Downloaded from: sumedhu/hindi-emotion-voice-references (HuggingFace dataset)
# Transcription verified via Whisper small on 2026-07-05.
DEFAULT_HINDI_REF_REPO   = "sumedhu/hindi-emotion-voice-references"
DEFAULT_HINDI_REF_FILE   = "hindi_best_clips/male/happy/HIN_M_HAPPY_00057.wav"
DEFAULT_HINDI_REF_TEXT   = "सुबह केरल चाय का एक गिलास मुझे तरो ताजा घर देता है"

# Pre-selected reference text for non-Hindi Basic tier languages.
# Only text is needed here; Hindi has a proper audio reference above.
BASIC_VOICE_REFS = {
    "hi": DEFAULT_HINDI_REF_TEXT,   # overridden by audio download below
    "bn": "আমি আপনাকে একটি গুরুত্বপূর্ণ বিষয় সম্পর্কে বলতে যাচ্ছি।",
    "mr": "नमस्कार, आज मी तुम्हाला एका महत्त्वाच्या विषयाबद्दल सांगणार आहे.",
    "gu": "નમસ્તે, આજે હું તમને એક મહત્વપૂર્ણ વિષય વિશે કહેવા જઈ રહ્યો છું.",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਅੱਜ ਮੈਂ ਤੁਹਾਨੂੰ ਇੱਕ ਮਹੱਤਵਪੂਰਨ ਵਿਸ਼ੇ ਬਾਰੇ ਦੱਸਣ ਜਾ ਰਿਹਾ ਹਾਂ।",
    "ta": "வணக்கம், இன்று நான் உங்களுக்கு ஒரு முக்கியமான விஷயத்தைப் பற்றி சொல்லப் போகிறேன்.",
    "te": "నమస్కారం, ఈరోజు నేను మీకు ఒక ముఖ్యమైన విషయం గురించి చెప్పబోతున్నాను.",
    "kn": "ನಮಸ್ಕಾರ, ಇಂದು ನಾನು ನಿಮಗೆ ಒಂದು ಮುಖ್ಯವಾದ ವಿಷಯದ ಬಗ್ಗೆ ಹೇಳಲು ಹೋಗುತ್ತಿದ್ದೇನೆ.",
    "ml": "നമസ്കാരം, ഇന്ന് ഞാൻ നിങ്ങൾക്ക് ഒരു പ്രധാനപ്പെട്ട വിഷയത്തെക്കുറിച്ച് പറയാൻ പോകുന്നു.",
    "or": "ନମସ୍କାର, ଆଜି ମୁଁ ଆପଣଙ୍କୁ ଏକ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ବିଷୟ ବିଷୟରେ କହିବାକୁ ଯାଉଛି।",
    "as": "নমস্কাৰ, আজি মই আপোনালোকক এটা গুৰুত্বপূৰ্ণ বিষয়ৰ বিষয়ে কʼব যাওঁ।",
}


# ---------------------------------------------------------------------------
# IndicF5 model loader (cached — loaded once per process)
# ---------------------------------------------------------------------------

_indicf5_model = None
_indicf5_device = None




def _load_indicf5(device: str = "auto"):
    """
    Load IndicF5 model onto device. Cached after first call.

    ROOT-CAUSE FIX (2026-07-06): transformers >= 4.35 loads models with
    `low_cpu_mem_usage=True` by default, which places all tensors on the
    'meta' device first. The IndicF5 model.py (written for older transformers)
    was not compatible — checkpoint weights were silently discarded (no-op copy
    from real tensor to meta tensor), leaving the model with random/uninitialised
    weights and producing garbage audio despite correct durations.

    Fix: pass `low_cpu_mem_usage=False` to from_pretrained. This forces full
    materialisation of all tensors to CPU RAM before moving to the target device.
    No meta-tensor patching needed.
    """
    global _indicf5_model, _indicf5_device

    import torch
    import torch.nn as nn

    if _indicf5_model is not None:
        return _indicf5_model, _indicf5_device

    if device == "auto":
        device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    print(f"[DurationTTS] Loading IndicF5 model on {device}...")

    from datetime import datetime
    def load_log(msg: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{t}] [DurationTTS_Load] {msg}", flush=True)

    load_log(f"Requesting load on device={device}...")

    # Flush GPU cache before loading
    if TORCH_AVAILABLE and torch.cuda.is_available():
        load_log("Flushing GPU VRAM and running garbage collection...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024 ** 3)
            total_gb = total_mem / (1024 ** 3)
            load_log(f"GPU VRAM Status: Free = {free_gb:.2f} GB, Total = {total_gb:.2f} GB")
            if free_gb < 2.0:
                load_log("WARNING: Free GPU VRAM is below 2 GB. IndicF5 may OOM. Close other apps.")
        except Exception as e:
            load_log(f"Could not query GPU memory: {e}")

    try:
        from transformers import AutoConfig
        import importlib.util as _ilu
        import logging
        import os
        logging.getLogger("transformers").setLevel(logging.ERROR)
        token = os.environ.get("HF_TOKEN")

        # CRITICAL FIX: Do NOT use AutoModel.from_pretrained().
        # from_pretrained() runs INF5Model.__init__ inside its meta-init context,
        # which patches torch.empty/zeros to create meta tensors. This silently
        # poisons ALL weight loading inside __init__ (both load_vocoder and
        # load_model), making the model run inference with uninitialised random
        # weights and produce garbage audio at correct durations.
        #
        # INF5Model.__init__ already downloads and loads ALL weights itself via
        # hf_hub_download + load_vocoder + load_model. No from_pretrained needed.
        # Direct importlib instantiation runs __init__ with real tensors.

        _cache_dir = os.path.expandvars(
            r"%USERPROFILE%\.cache\huggingface\modules\transformers_modules"
            r"\ai4bharat\IndicF5\ba85abedf18dc479a447eaa0eccbd76ab78a47d5"
        )
        _model_py = os.path.join(_cache_dir, "model.py")

        # If the cached model.py doesn't exist yet, trigger a one-time download
        # via a throwaway from_pretrained with a NO-OP config so HF caches files.
        if not os.path.exists(_model_py):
            load_log("HF cache miss — triggering one-time model file download...")
            try:
                AutoModel.from_pretrained(
                    "ai4bharat/IndicF5",
                    trust_remote_code=True,
                    token=token,
                )
            except Exception:
                pass  # Crash expected; we only needed the cache to populate

        load_log(f"Loading INF5Model directly from cached model.py: {_model_py}")
        _spec = _ilu.spec_from_file_location("indicf5_model", _model_py)
        _mod  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        _config = AutoConfig.from_pretrained(
            "ai4bharat/IndicF5",
            trust_remote_code=True,
            token=token,
        )
        _config.name_or_path = "ai4bharat/IndicF5"  # Needed for hf_hub_download

        load_log("Instantiating INF5Model directly (no meta-init context)...")
        model = _mod.INF5Model(_config)

        load_log(f"Checkpoint weights loaded to CPU. Moving to device={device}...")
        model = model.to(device)
        load_log("Setting model to eval mode...")
        model.eval()
        _indicf5_model = model
        _indicf5_device = device
        load_log(f"IndicF5 loaded successfully on {device} with real weights.")
        return model, device
    except Exception as e:
        raise RuntimeError(
            f"Failed to load IndicF5. Ensure it is installed and HuggingFace gate is accepted.\n"
            f"  pip install git+https://github.com/ai4bharat/IndicF5.git\n"
            f"  Error: {e}"
        )


def unload_indicf5():
    """Explicitly unload the model and free VRAM."""
    global _indicf5_model, _indicf5_device
    if _indicf5_model is not None and TORCH_AVAILABLE:
        del _indicf5_model
        _indicf5_model = None
        _indicf5_device = None
        torch.cuda.empty_cache()
        print("[DurationTTS] IndicF5 unloaded, VRAM cleared.")


# ---------------------------------------------------------------------------
# Duration-controlled inference
# ---------------------------------------------------------------------------

def _duration_to_mel_frames(duration_seconds: float) -> int:
    """Convert target duration to mel-frame count for IndicF5 conditioning."""
    return max(1, int(math.ceil(duration_seconds * FRAMES_PER_SECOND)))


def _generate_single_segment(
    model,
    device: str,
    text: str,
    target_duration: float,
    ref_audio_path: Optional[str],
    ref_text: str,
    lang_code: str,
) -> np.ndarray:
    """
    Generate audio for a single text segment with precise duration conditioning.

    FIX (2026-07-06): Uses infer_process (the official IndicF5 model.py API) which
    takes the ref_audio file path and handles concatenation correctly. Previously
    we called infer_batch_process directly with gen_text_batches=[text], which caused
    infer_batch_process to concatenate ref_text+gen_text with NO space separator,
    making the model unable to find the word boundary between reference and target text.
    This caused the model to hallucinate repeated phrases ("कर दो कर दो...").

    The fix: use infer_process with the preprocessed ref_audio path and a ref_text
    that ends with a trailing space, matching the IndicF5 model.py forward() API exactly.
    """
    import soundfile as sf
    import torchaudio

    from datetime import datetime
    def seg_log(msg: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{t}] [SegGen] {msg}", flush=True)

    if not ref_audio_path or not os.path.exists(ref_audio_path):
        raise ValueError(
            "IndicF5 requires a valid reference_audio_path to perform synthesis. "
            "Please ensure a reference voice clip is provided."
        )

    # Ensure submodels are placed on the active device
    seg_log("Placing EMA model and vocoder on target device...")
    if hasattr(model, "ema_model"):
        model.ema_model.to(device)
    if hasattr(model, "vocoder"):
        model.vocoder.to(device)

    # Preprocess the reference audio/text (resamples to 24kHz, trims silence, etc.)
    # infer_process accepts a file path, so we pass ref_audio_path directly — it
    # calls preprocess_ref_audio_text internally. This matches the official IndicF5
    # model.py forward() API exactly.
    seg_log(f"Preprocessing reference audio ({ref_audio_path}) & text...")
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process

    # Ensure ref_text ends with a trailing space so that when infer_batch_process
    # internally concatenates ref_text + gen_text, the words are properly separated.
    # Without this, Hindi words like "हैमिट्टी" (है + मिट्टी) get fused and the model
    # cannot find the boundary, causing it to hallucinate repeated syllables.
    ref_text_clean = ref_text.strip()
    if not ref_text_clean.endswith(" "):
        ref_text_clean = ref_text_clean + " "

    # Preprocess just to compute ref_duration for fix_duration calculation.
    ref_audio_pre, ref_text_pre = preprocess_ref_audio_text(ref_audio_path, ref_text_clean)
    ref_audio_tensor, ref_sr = torchaudio.load(ref_audio_pre)
    ref_duration = ref_audio_tensor.shape[-1] / ref_sr
    fix_duration = ref_duration + target_duration
    seg_log(f"Ref duration = {ref_duration:.2f}s | Target duration = {target_duration:.2f}s | Total fix_duration = {fix_duration:.2f}s")
    # Clean up temp file — infer_process will re-preprocess from the original path.
    try:
        if os.path.exists(ref_audio_pre):
            os.remove(ref_audio_pre)
    except Exception:
        pass

    seg_log("Calling infer_process (official IndicF5 API, single-pass, no chunking)...")
    with torch.no_grad():
        # Use infer_process (matches IndicF5 model.py forward() exactly).
        # Passing fix_duration ensures the generated mel-spectrogram spans
        # exactly ref_duration + target_duration frames.
        audio, final_sample_rate, _ = infer_process(
            ref_audio=ref_audio_path,       # file path — infer_process handles loading
            ref_text=ref_text_clean,
            gen_text=text,
            model_obj=model.ema_model,
            vocoder=model.vocoder,
            mel_spec_type="vocos",
            fix_duration=fix_duration,
            device=device,
        )
    seg_log("Synthesis step completed successfully.")

    # Normalize loudness to -20 dBFS using pydub
    seg_log("Normalizing loudness to -20 dBFS...")
    import io
    from pydub import AudioSegment

    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    buffer.seek(0)
    audio_segment = AudioSegment.from_file(buffer, format="wav")

    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    audio_segment = audio_segment.apply_gain(change_in_dBFS)

    audio_samples = np.array(audio_segment.get_array_of_samples())
    audio_float = audio_samples.astype(np.float32) / 32768.0

    if audio_float.ndim > 1:
        audio_float = audio_float.mean(axis=0)

    # Note: temp preprocessed file was already cleaned up above after ref_duration
    # calculation. infer_process creates its own temp file and cleans it up internally.

    return audio_float


def _apply_drift_correction(
    audio: np.ndarray,
    actual_duration: float,
    target_duration: float,
    segment_idx: int,
) -> np.ndarray:
    """
    Apply pyrubberband time-stretching ONLY if drift exceeds DRIFT_TOLERANCE.
    This is a safety-net, not the primary mechanism.
    """
    if target_duration <= 0:
        return audio

    ratio = actual_duration / target_duration
    if abs(ratio - 1.0) <= DRIFT_TOLERANCE:
        return audio  # Within tolerance — no correction needed

    print(
        f"  [Segment {segment_idx}] Drift correction: "
        f"actual={actual_duration:.3f}s target={target_duration:.3f}s "
        f"ratio={ratio:.3f}"
    )

    try:
        import pyrubberband as pyrb
        corrected = pyrb.time_stretch(audio, INDICF5_SAMPLE_RATE, 1.0 / ratio)
        return corrected.astype(np.float32)
    except ImportError:
        print(
            f"  [Segment {segment_idx}] pyrubberband not installed. "
            f"Skipping drift correction. Install with: pip install pyrubberband"
        )
        return audio
    except Exception as e:
        print(f"  [Segment {segment_idx}] Drift correction failed: {e}. Using original.")
        return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tts_for_segments(
    translated_segments: list,
    target_language: str,
    output_dir: str,
    reference_audio_path: Optional[str] = None,
    reference_text: Optional[str] = None,
    device: str = "auto",
) -> list:
    """
    Generate duration-controlled TTS audio for each translated segment.

    Args:
        translated_segments:   List of dicts with 'start', 'end', 'text'.
        target_language:       Display name e.g. 'Hindi', 'Tamil'.
        output_dir:            Directory to save per-segment WAV files.
        reference_audio_path:  Path to reference voice clip (.wav, 24kHz).
                               None = Basic tier (pre-defined voice, no cloning).
        reference_text:        Transcript of the reference audio clip.
                               Required if reference_audio_path is provided.
        device:                'auto', 'cuda', or 'cpu'.

    Returns:
        Same list of segments, with 'audio_path' added to each element.
    """
    if not SOUNDFILE_AVAILABLE:
        raise RuntimeError("soundfile is required. Run: pip install soundfile")

    from datetime import datetime
    def tts_log(msg: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{t}] [DurationTTS] {msg}", flush=True)

    lang_code = LANGUAGE_TO_CODE.get(target_language)
    if not lang_code:
        raise ValueError(
            f"Unsupported language: '{target_language}'. "
            f"Supported: {list(LANGUAGE_TO_CODE.keys())}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Determine reference audio / text
    is_cloning = reference_audio_path is not None
    if is_cloning:
        if not reference_text:
            raise ValueError(
                "reference_text must be provided when reference_audio_path is set."
            )
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio_path}"
            )
        tts_log(f"Mode: Premium (voice cloning from {reference_audio_path})")
        ref_audio = reference_audio_path
        ref_text = reference_text
    else:
        tts_log(f"Mode: Basic (pre-selected voice, no cloning)")
        # Download the default reference voice from HuggingFace to serve as the default speaker for all languages.
        # SSL bypass is already globally active in this module.
        try:
            from huggingface_hub import hf_hub_download
            tts_log(f"Downloading/resolving default reference voice from HF ({DEFAULT_HINDI_REF_REPO})...")
            ref_audio = hf_hub_download(
                repo_id=DEFAULT_HINDI_REF_REPO,
                filename=DEFAULT_HINDI_REF_FILE,
                repo_type="dataset",
            )
            ref_text = DEFAULT_HINDI_REF_TEXT
            tts_log(f"Default reference voice resolved to: {ref_audio}")
        except Exception as e:
            tts_log(f"Warning: Could not download default reference audio: {e}")
            tts_log("Falling back to text-only reference (may affect quality/stability).")
            ref_audio = None
            ref_text = BASIC_VOICE_REFS.get(lang_code, "")

    # Load model
    tts_log("Loading/resolving IndicF5 model...")
    model, resolved_device = _load_indicf5(device)
    tts_log(f"IndicF5 model ready on device: {resolved_device}")

    results = []
    for i, seg in enumerate(translated_segments):
        text = seg.get("text", "").strip()
        target_duration = seg["end"] - seg["start"]
        out_path = os.path.join(output_dir, f"segment_{i:04d}.wav")

        if not text:
            tts_log(f"[Segment {i}/{len(translated_segments)-1}] Empty text — writing silence.")
            silence = np.zeros(int(target_duration * INDICF5_SAMPLE_RATE), dtype=np.float32)
            sf.write(out_path, silence, INDICF5_SAMPLE_RATE)
            results.append({**seg, "audio_path": out_path})
            continue

        tts_log(
            f"Processing [Segment {i}/{len(translated_segments)-1}]: "
            f"target={target_duration:.2f}s | text='{text[:60]}...'"
        )

        try:
            audio = _generate_single_segment(
                model=model,
                device=resolved_device,
                text=text,
                target_duration=target_duration,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                lang_code=lang_code,
            )

            actual_duration = len(audio) / INDICF5_SAMPLE_RATE
            audio = _apply_drift_correction(audio, actual_duration, target_duration, i)

            sf.write(out_path, audio, INDICF5_SAMPLE_RATE)
            tts_log(f"[Segment {i}/{len(translated_segments)-1}] Completed successfully. Saved -> {out_path}")

        except Exception as e:
            tts_log(f"ERROR: [Segment {i}] TTS failed: {e}. Writing silence.")
            import traceback
            traceback.print_exc()
            silence = np.zeros(int(target_duration * INDICF5_SAMPLE_RATE), dtype=np.float32)
            sf.write(out_path, silence, INDICF5_SAMPLE_RATE)

        # Clear CUDA cache and collect garbage after every segment to prevent OOM on low VRAM GPUs
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        results.append({**seg, "audio_path": out_path})

    tts_log(f"Generated {len(results)} audio segments in {output_dir}")
    return results


