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
import json
import hashlib
import threading
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

# Hard cap on how far the drift safety-net will time-stretch, in EITHER direction.
# A correction beyond this factor would be audibly rushed (compression) or dragged
# (expansion) — smeared plosives/aspiration, flattened cadence. A required stretch this
# large is not a small drift to paper over; it means the segment grossly over/undershot
# its slot at synthesis time (an upstream fault — reference-clip length, translation
# budget). We keep the UNCORRECTED audio and flag the segment for regeneration instead
# of shipping a chipmunked ~2x compression. See references/failure-modes.md ("Cap the
# last resort and watch it").
MAX_DRIFT_STRETCH = 1.3

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




def _load_indicf5(device: str = "auto", log_fn=None):
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
        line = f"[{t}] [DurationTTS_Load] {msg}"
        print(line, flush=True)
        if log_fn is not None:
            try:
                log_fn(f"    {msg}")
            except Exception:
                pass

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

        # Locate the cached IndicF5 model.py in a platform-independent way.
        # The remote-code module lands under the HF cache with a commit-hash
        # subdirectory, so glob the hash rather than hardcoding a path (the old
        # Windows-only %USERPROFILE% literal never resolved on Kaggle/Linux).
        import glob as _glob
        _patterns = _glob.glob(os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py"
        ))
        _model_py = _patterns[0] if _patterns else None

        # If the cached model.py doesn't exist yet, trigger a one-time download
        # via a throwaway from_pretrained so HF caches the remote-code files.
        if not _model_py or not os.path.exists(_model_py):
            load_log("HF cache miss - triggering one-time model file download...")
            try:
                from transformers import AutoModel
                AutoModel.from_pretrained(
                    "ai4bharat/IndicF5",
                    trust_remote_code=True,
                    token=token,
                )
            except Exception:
                pass  # Crash expected; we only needed the cache to populate
            _patterns = _glob.glob(os.path.expanduser(
                "~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py"
            ))
            _model_py = _patterns[0] if _patterns else None

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


def _move_indicf5_to(device: str, log_fn=None):
    """Move the already-cached IndicF5 model onto ``device`` in place. For Modal snapshots.

    A Modal CPU memory snapshot is captured with the model materialised in CPU RAM — loaded by
    ``_load_indicf5("cpu")`` inside a ``@modal.enter(snap=True)`` phase, before any GPU is
    attached, so the ~1.3 GB weight materialisation and the heavy imports are baked into the
    snapshot. After restore the GPU is present, and this moves the SAME cached object onto it,
    reaching the exact device state a direct ``_load_indicf5("cuda")`` would have produced — no
    reload, no re-download. If nothing is loaded yet it falls back to a normal load so callers
    stay correct off Modal.

    Additive: the Kaggle/serial/supervised paths never call this, so their behaviour is
    unchanged. The generate path re-places ema_model+vocoder per call anyway (see
    generate_tts_for_segments); moving them here just makes the whole model coherently resident
    on ``device`` immediately after restore.
    """
    global _indicf5_model, _indicf5_device
    if _indicf5_model is None:
        return _load_indicf5(device, log_fn=log_fn)
    if _indicf5_device == device:
        return _indicf5_model, _indicf5_device
    model = _indicf5_model.to(device)
    for attr in ("ema_model", "vocoder"):
        sub = getattr(model, attr, None)
        if sub is not None:
            try:
                sub.to(device)
            except Exception:
                pass
    _indicf5_model = model
    _indicf5_device = device
    msg = f"[DurationTTS] moved cached IndicF5 to {device}"
    print(msg, flush=True)
    if log_fn is not None:
        try:
            log_fn(f"    {msg}")
        except Exception:
            pass
    return model, device


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


# ── Cross-lingual onset-babble mitigation: Hindi "primer" (opt-in) ─────────────────────
# IndicF5 is Indic-trained; conditioning it on an ENGLISH voice reference (audio + text) and
# asking for HINDI makes its first diffusion steps resolve the language jump as ~0.5-1.5s of
# garbled pseudo-speech at the START of every segment. Confirmed on the raw tts_chunks: the
# artifact is on the onset, and chunk onsets vary more from each other (mean pairwise spectral
# cos 0.905) than the clean Hindi bodies do (0.977) — a generation-time onset INSTABILITY, not
# an assembly artifact (assembly is provably clean: overlap 0, no clicks, natural decay).
#
# Fix: prepend a short, throwaway Hindi utterance so the instability lands on DISCARDABLE
# priming audio, then slice the primer back off. Because the existing drift-correction re-times
# whatever we return to the exact target_duration, the primer's RENDERED length need not be
# exact — we only need to cut near the primer->target boundary. The primer ends in a danda so a
# clear pause marks that boundary. Opt-in (DUBBING_TTS_PRIMER=1) until validated on Kaggle, so
# the default path stays byte-for-byte unchanged. See non-negotiable #1: this is validated by
# instrumentation on a run, not assumed from "audio came out".
TTS_PRIMER_TEXT_DEFAULT = "नमस्ते।"          # short, neutral, ends in a danda (=> a pause to cut at)
TTS_PRIMER_BUDGET_S_DEFAULT = 1.0            # fix_duration seconds allotted to the primer
TTS_PRIMER_SEARCH_LO_S = 0.30                # earliest cut point (never clip a very short primer)
TTS_PRIMER_SEARCH_HI_MARGIN_S = 0.60         # search up to (budget + this) for the pause


def _tts_primer_enabled() -> bool:
    return os.environ.get("DUBBING_TTS_PRIMER", "").strip().lower() in ("1", "true", "yes", "on")


def _tts_primer_text() -> str:
    return os.environ.get("DUBBING_TTS_PRIMER_TEXT", "").strip() or TTS_PRIMER_TEXT_DEFAULT


def _tts_primer_budget_s() -> float:
    try:
        return float(os.environ.get("DUBBING_TTS_PRIMER_BUDGET", "").strip() or TTS_PRIMER_BUDGET_S_DEFAULT)
    except ValueError:
        return TTS_PRIMER_BUDGET_S_DEFAULT


def _slice_off_primer(audio, sr: int, primer_budget_s: float, seg_log) -> np.ndarray:
    """Remove the primer prefix from a primer+target generation by cutting at the pause
    (danda) between them. Search a 20 ms-RMS envelope for the quietest point in
    [SEARCH_LO, budget + HI_MARGIN]; cut there if it is a real trough (well below the region
    median), else fall back to the fixed budget. Safety: if the remaining target would be
    implausibly short (<0.3 s), keep the full audio — a primer artifact is recoverable, a lost
    segment is not (degrade, don't crash). Operates on the last axis so 1-D or (ch, N) both work."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim > 1:
        a = a.mean(axis=0)
    n = a.shape[-1]
    full = np.asarray(audio)
    lo = int(TTS_PRIMER_SEARCH_LO_S * sr)
    hi = min(n, int((primer_budget_s + TTS_PRIMER_SEARCH_HI_MARGIN_S) * sr))
    step = int(0.02 * sr)
    if hi - lo < step or step <= 0:
        cut = min(n, int(primer_budget_s * sr))
        seg_log(f"[primer] search window too small (n={n/sr:.2f}s) — slicing at budget {primer_budget_s:.2f}s")
        return full[..., cut:]
    env = np.array([np.sqrt(np.mean(a[k:k + step] ** 2)) for k in range(lo, hi - step, step)])
    if env.size == 0:
        cut = min(n, int(primer_budget_s * sr))
        return full[..., cut:]
    med = float(np.median(env))
    j = int(np.argmin(env))
    trough_rms = float(env[j])
    if trough_rms < 0.15 * med + 1e-9:
        cut_idx = lo + j * step + step // 2
        seg_log(f"[primer] pause at {cut_idx / sr:.2f}s (trough_rms={trough_rms:.4f} vs med={med:.4f}) — cutting there")
    else:
        cut_idx = min(n, int(primer_budget_s * sr))
        seg_log(f"[primer] no clear pause (min_rms={trough_rms:.4f} vs med={med:.4f}) — cutting at budget {primer_budget_s:.2f}s")
    if n - cut_idx < int(0.30 * sr):
        seg_log(f"[primer] WARNING: only {(n - cut_idx) / sr:.2f}s would remain after the cut — keeping full audio")
        return full
    return full[..., cut_idx:]


def _generate_single_segment(
    model,
    device: str,
    text: str,
    target_duration: float,
    ref_audio_path: Optional[str],
    ref_text: str,
    lang_code: str,
    log_fn=None,
    nfe_step: int = 32,
) -> np.ndarray:
    """
    Generate audio for a single text segment with precise duration conditioning.

    ISOCHRONY FIX (2026-09-11): Generate the whole segment in ONE batch by calling
    infer_batch_process directly with gen_text_batches=[text]. The convenience wrapper
    infer_process() splits gen_text into chunks sized from the reference length and then
    gives EACH chunk the FULL fix_duration — verified in the installed f5-tts source
    (utils_infer.infer_process -> chunk_text, then infer_batch_process sets
    duration=int(fix_duration*sr/hop) per chunk). So an N-chunk segment is generated ~N×
    its target length: the confirmed cause of segments running up to 2× their SRT slot and
    botching every join (fix_duration was passed and honored, but PER CHUNK, so it silently
    failed to control the total). A single batch makes the mel span exactly ref+target
    frames — i.e. target after IndicF5's internal ref slice.

    HALLUCINATION GUARD: infer_batch_process only auto-inserts a ref/gen separator space
    when ref_text's last character is a single UTF-8 byte, which is never true for
    Devanagari. Without a boundary space, ref_text+gen_text fuse and the model hallucinates
    repeats ("कर दो कर दो..."). We therefore force a trailing space on ref_text ourselves.
    That missing space — NOT the direct call — is why an earlier infer_batch_process attempt
    failed, so calling it directly is safe once the separator space is guaranteed.
    """
    import soundfile as sf
    import torchaudio

    from datetime import datetime
    def seg_log(msg: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{t}] [SegGen] {msg}", flush=True)
        if log_fn is not None:
            try:
                log_fn(f"      {msg}")
            except Exception:
                pass

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
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_batch_process

    # Ensure ref_text ends with a trailing space so that when infer_batch_process
    # internally concatenates ref_text + gen_text, the words are properly separated.
    # Without this, Hindi words like "हैमिट्टी" (है + मिट्टी) get fused and the model
    # cannot find the boundary, causing it to hallucinate repeated syllables.
    ref_text_clean = ref_text.strip()
    if not ref_text_clean.endswith(" "):
        ref_text_clean = ref_text_clean + " "

    # Preprocess the reference ONCE and reuse the identical clip for BOTH the fix_duration
    # math and the actual inference. infer_process() would instead re-load the raw path, so
    # its internal ref-slice length (ref_audio_len) would not match the ref_duration we
    # measured from the trimmed clip — a latent undershoot. Measuring and generating from
    # the same audio makes the arithmetic exact: kept frames = fix_duration - ref_audio_len
    # = target.
    ref_audio_pre, ref_text_pre = preprocess_ref_audio_text(ref_audio_path, ref_text_clean)
    ref_audio_tuple = torchaudio.load(ref_audio_pre)                 # (tensor, sr)
    ref_duration = ref_audio_tuple[0].shape[-1] / ref_audio_tuple[1]

    # Optional cross-lingual onset-babble primer (see _slice_off_primer above). When enabled we
    # prepend a throwaway Hindi utterance to gen_text and give it its own slice of fix_duration;
    # the babble lands on the primer, which we cut off after synthesis. Default OFF => gen_text
    # and fix_duration are exactly as before (ref + target), so this is a no-op unless opted in.
    primer_on = _tts_primer_enabled()
    primer_budget = _tts_primer_budget_s() if primer_on else 0.0
    if primer_on:
        primer_text = _tts_primer_text()
        gen_text_final = primer_text + " " + text
        seg_log(f"[primer] ENABLED text='{primer_text}' budget={primer_budget:.2f}s — prepended to gen_text")
    else:
        gen_text_final = text

    fix_duration = ref_duration + primer_budget + target_duration
    seg_log(f"Ref duration = {ref_duration:.2f}s | Primer budget = {primer_budget:.2f}s | Target duration = {target_duration:.2f}s | Total fix_duration = {fix_duration:.2f}s")

    # preprocess may strip the boundary space; re-assert it (see HALLUCINATION GUARD above).
    ref_text_infer = ref_text_pre if ref_text_pre.endswith(" ") else ref_text_pre + " "

    seg_log(f"Calling infer_batch_process (SINGLE batch, no chunking, nfe_step={nfe_step})...")
    with torch.no_grad():
        # CRITICAL isochrony fix: pass a SINGLE-element gen_text_batches so the whole
        # segment is generated in ONE pass with ONE fix_duration. infer_process() splits
        # gen_text into N chunks and gives EACH the full fix_duration => ~N× overshoot (the
        # confirmed cause of the every-join botch). One batch => mel spans exactly
        # ref+target frames => target after IndicF5's internal ref slice.
        #
        # nfe_step is the number of diffusion (denoising) steps. Synthesis time is
        # ~linear in it: the default 32 is the quality baseline; lowering it (e.g. 16)
        # roughly halves per-segment latency at some quality cost. It is a caller-tunable
        # knob (DUBBING_NFE_STEP env / nfe_step arg), NOT changed silently here.
        audio, final_sample_rate, _ = infer_batch_process(
            ref_audio_tuple,
            ref_text_infer,
            [gen_text_final],               # single batch => no length multiplication (primer-prepended when opted in)
            model.ema_model,
            model.vocoder,
            mel_spec_type="vocos",
            fix_duration=fix_duration,
            nfe_step=nfe_step,
            device=device,
        )
    seg_log("Synthesis step completed successfully.")

    # Clean up the preprocessed temp clip now that inference is done.
    try:
        if os.path.exists(ref_audio_pre):
            os.remove(ref_audio_pre)
    except Exception:
        pass

    # Slice the primer prefix back off (opt-in). The model rendered [primer | target] across
    # (primer_budget + target) post-ref seconds; cut at the primer->target pause so only the
    # clean target remains. The caller's drift-correction then re-times it to exact target.
    if primer_on:
        _pre_s = int(np.asarray(audio).shape[-1]) / float(final_sample_rate)
        audio = _slice_off_primer(audio, final_sample_rate, primer_budget, seg_log)
        _post_s = int(np.asarray(audio).shape[-1]) / float(final_sample_rate)
        seg_log(f"[primer] sliced primer: {_pre_s:.2f}s -> {_post_s:.2f}s (target {target_duration:.2f}s)")

    # Prove the control signal actually worked (do NOT infer it from "audio came out"):
    # the generated length must be ~target, never a multiple of it. Logging actual vs
    # target makes a future chunking/duration regression visible instead of silent.
    try:
        _actual_s = int(np.asarray(audio).shape[-1]) / float(final_sample_rate)
        _ratio = _actual_s / target_duration if target_duration > 0 else 0.0
        seg_log(f"Output duration = {_actual_s:.2f}s vs target {target_duration:.2f}s (ratio {_ratio:.2f})")
        if _ratio > 1.5:
            seg_log(f"WARNING: output {_ratio:.2f}x target — duration control may have regressed (chunking?).")
    except Exception:
        pass

    # Normalize loudness to -20 dBFS.
    #
    # Pure-numpy RMS normalization — deliberately NOT via pydub. pydub's
    # AudioSegment.from_file() spawns an ffmpeg subprocess for every segment, and
    # an ffmpeg child that deadlocks on its pipe buffers hangs the whole TTS loop
    # forever with no error and no timeout (observed: a 30-minute freeze mid-batch
    # on Kaggle, right after synthesis). This computes the exact same dBFS gain
    # with a single scalar multiply, so it has no subprocess and cannot hang.
    seg_log("Normalizing loudness to -20 dBFS...")
    audio_float = np.asarray(audio, dtype=np.float32)
    if audio_float.ndim > 1:
        audio_float = audio_float.mean(axis=0).astype(np.float32)

    target_dBFS = -20.0
    if audio_float.size:
        rms = float(np.sqrt(np.mean(audio_float.astype(np.float64) ** 2)))
    else:
        rms = 0.0
    if rms > 1e-9:
        # dBFS measured against full scale (1.0), matching pydub's 16-bit dBFS.
        current_dBFS = 20.0 * np.log10(rms)
        gain = float(10.0 ** ((target_dBFS - current_dBFS) / 20.0))
        audio_float = (audio_float * gain).astype(np.float32)
    else:
        seg_log("  Segment is effectively silent; skipping loudness normalization.")

    # Clip to [-1, 1] to match pydub's int16 clipping and keep the WAV in range.
    audio_float = np.clip(audio_float, -1.0, 1.0).astype(np.float32)

    # Note: the preprocessed reference temp clip was cleaned up above, right after
    # inference finished using it.

    return audio_float


# Hard ceiling (seconds) for the drift-correction rubberband subprocess. pyrubberband
# shells out to the `rubberband` CLI, and on Kaggle that child has been observed to
# deadlock intermittently (the same failure class as the old pydub→ffmpeg hang). Time-
# stretching a few seconds of audio takes well under a second, so 30s is a generous
# ceiling: any run past it is a wedged child, not slow work. Override: DUBBING_DRIFT_TIMEOUT.
DRIFT_TIMEOUT_DEFAULT = 30.0

# Hard ceiling (seconds) for a single WAV / manifest write to the Kaggle working dir.
# /kaggle/working is an overlay/network-backed filesystem that intermittently stalls a
# single write — the confirmed cause of the Step-6 freeze (heartbeats pinned it to
# sf.write, with the freeze landing on a random segment each run). The audio is already
# a finished in-memory numpy array by then, so a wedged write holds NO GPU/model state
# and is safe to abandon + retry to a fresh path. Writing ~0.5 MB of audio takes well
# under a second, so 20s is a generous ceiling. Override: DUBBING_DISK_WRITE_TIMEOUT.
DISK_WRITE_TIMEOUT_DEFAULT = 20.0
# How many times to retry a stalled write (each attempt to a NEW temp path) before
# giving up and marking the segment for retry-on-resume. Override: DUBBING_DISK_WRITE_ATTEMPTS.
DISK_WRITE_ATTEMPTS_DEFAULT = 3

# Wall-clock ceiling for one segment's heavy work (GPU synthesis + drift + disk write).
# Normal segments finish in seconds; this is a generous multiple, so the watchdog only
# fires on a genuine hang. Its real job is to bound the ONE otherwise-unbounded piece —
# GPU synthesis (drift and the disk write already have their own finer internal timeouts).
# A wedged CUDA op canNOT be retried in-process (the abandoned worker thread still holds
# the shared model), so when this fires we record the segment for retry and HALT Step 6
# gracefully; the next run auto-resumes from exactly that segment. Override: DUBBING_SEGMENT_TIMEOUT.
SEGMENT_TIMEOUT_DEFAULT = 180.0


def _disk_write_timeout() -> float:
    try:
        return float(os.environ.get("DUBBING_DISK_WRITE_TIMEOUT", "").strip() or DISK_WRITE_TIMEOUT_DEFAULT)
    except ValueError:
        return DISK_WRITE_TIMEOUT_DEFAULT


def _disk_write_attempts() -> int:
    try:
        return max(1, int(os.environ.get("DUBBING_DISK_WRITE_ATTEMPTS", "").strip() or DISK_WRITE_ATTEMPTS_DEFAULT))
    except ValueError:
        return DISK_WRITE_ATTEMPTS_DEFAULT


def _segment_timeout() -> float:
    try:
        return float(os.environ.get("DUBBING_SEGMENT_TIMEOUT", "").strip() or SEGMENT_TIMEOUT_DEFAULT)
    except ValueError:
        return SEGMENT_TIMEOUT_DEFAULT


def _run_with_timeout(fn, timeout: float, *args, **kwargs):
    """
    Run ``fn(*args, **kwargs)`` on a daemon thread and wait at most ``timeout`` seconds.

    Returns (True, result) on completion, (False, None) on timeout. On timeout the worker
    thread is ABANDONED (Python cannot force-kill a thread). This is used only around the
    rubberband subprocess, where the abandoned worker is a CPU-bound external `rubberband`
    process that finishes or dies on its own and holds no GPU/model state — so leaking it is
    benign. Do NOT use this to abandon GPU synthesis (a wedged CUDA op would leak into the
    shared model); resume-from-disk handles that case instead.
    """
    box = {}

    def _target():
        try:
            box["result"] = fn(*args, **kwargs)
            box["ok"] = True
        except Exception as e:  # noqa: BLE001 — surfaced to caller below
            box["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, None
    if "error" in box:
        raise box["error"]
    return True, box.get("result")


def _write_wav_with_timeout(out_path, audio, sample_rate, segment_idx, log_fn=None) -> bool:
    """Write ``audio`` to ``out_path`` as a WAV, bounded by a per-attempt wall-clock timeout.

    Guards the confirmed Step-6 freeze: on Kaggle, ``sf.write`` to /kaggle/working can wedge
    on a transient filesystem stall. The audio is a finished in-memory array here (synthesis
    already returned), so a wedged write holds no GPU/model state and is safe to abandon.

    Each attempt writes to a UNIQUE temp path then atomically ``os.replace``-s it into place.
    Uniqueness matters: if a write is abandoned, that daemon thread may still hold its temp
    file open, so a retry must never reuse the same temp name (that would risk two writers on
    one file). An abandoned writer's eventual write lands on the now-unlinked old temp inode,
    never on the live output — so the final file can't be corrupted by a straggler.

    Returns True if the file was written and renamed into place; False if every attempt
    stalled or errored (caller then records the segment for retry-on-resume and continues).
    """
    import uuid

    def _emit(msg: str):
        print(f"  {msg}", flush=True)
        if log_fn is not None:
            try:
                log_fn(msg)
            except Exception:
                pass

    timeout = _disk_write_timeout()
    attempts = _disk_write_attempts()
    for attempt in range(1, attempts + 1):
        # Temp name carries a .wav before the unique suffix so libsndfile can still infer
        # the container from the extension; we also pass format="WAV" so the write never
        # depends on the extension. Default subtype (PCM_16) matches the pre-guard write.
        tmp = f"{out_path}.{uuid.uuid4().hex[:8]}.wav.tmp"

        def _do():
            sf.write(tmp, audio, sample_rate, format="WAV")
            os.replace(tmp, out_path)

        reason = None
        try:
            ok, _ = _run_with_timeout(_do, timeout)
            if not ok:
                reason = f"stalled >{timeout:g}s (disk wedge)"
        except Exception as e:  # noqa: BLE001 — surfaced, then retried/failed-over below
            ok = False
            reason = f"errored ({e})"

        if ok:
            if attempt > 1:
                _emit(f"[Segment {segment_idx}] WAV write succeeded on attempt {attempt}/{attempts}.")
            return True

        # Do NOT touch the abandoned temp path here — a stat/remove on the same wedged
        # filesystem could itself block and re-freeze the loop. Orphan .tmp files are
        # harmless; the next attempt uses a fresh name.
        if attempt < attempts:
            _emit(
                f"[Segment {segment_idx}] WAV write attempt {attempt}/{attempts} {reason} "
                f"— retrying to a fresh path..."
            )
        else:
            _emit(f"[Segment {segment_idx}] WAV write attempt {attempt}/{attempts} {reason}.")
    _emit(
        f"[Segment {segment_idx}] WAV write FAILED after {attempts} attempt(s). "
        f"Segment marked for retry on next run."
    )
    return False


def _apply_drift_correction(
    audio: np.ndarray,
    actual_duration: float,
    target_duration: float,
    segment_idx: int,
) -> np.ndarray:
    """
    Apply pyrubberband time-stretching ONLY if drift exceeds DRIFT_TOLERANCE.
    This is a safety-net, not the primary mechanism.

    Correction runs only when the required stretch stays within MAX_DRIFT_STRETCH in
    either direction. A larger stretch means the segment grossly over/undershot at
    synthesis time — an upstream fault (reference-clip length, translation budget) that
    time-stretch cannot repair without audible damage — so the UNCORRECTED audio is kept
    and the segment is flagged (a loud log line) for regeneration rather than shipped
    rushed or dragged.

    The rubberband call is bounded by a hard timeout (DUBBING_DRIFT_TIMEOUT, default 30s):
    if the underlying `rubberband` subprocess wedges — the intermittent Kaggle hang this
    guards against — the segment falls back to the UNCORRECTED audio (residual drift is
    bounded near DRIFT_TOLERANCE) and the run continues instead of freezing. All fallbacks
    are logged with flush so they are visible, never silent.
    """
    if target_duration <= 0:
        return audio

    ratio = actual_duration / target_duration
    if abs(ratio - 1.0) <= DRIFT_TOLERANCE:
        return audio  # Within tolerance — no correction needed

    # pyrubberband.time_stretch(y, sr, rate): out_duration = in_duration / rate, so
    # rate>1 SHORTENS and rate<1 LENGTHENS. ratio = actual/target IS exactly that rate —
    # a too-long segment (ratio>1) shortens back to target, a too-short one (ratio<1)
    # lengthens. (Passing 1.0/ratio here was Failure Mode #3, the inverted stretch: it
    # pushed a too-long segment even LONGER, turning a ~2x overshoot into a ~4x one.)
    # stretch_factor is how extreme the correction is, in either direction (always >=1).
    stretch_factor = ratio if ratio >= 1.0 else 1.0 / ratio
    if stretch_factor > MAX_DRIFT_STRETCH:
        print(
            f"  [Segment {segment_idx}] DEGRADED: drift ratio={ratio:.3f} "
            f"(actual={actual_duration:.3f}s target={target_duration:.3f}s) exceeds the "
            f"{MAX_DRIFT_STRETCH:g}x safe stretch cap — a correction this large would be "
            f"audibly {'rushed' if ratio > 1.0 else 'dragged'}. Keeping UNCORRECTED audio "
            f"and flagging this segment for regeneration (shorten the voice reference or "
            f"re-budget the line upstream).",
            flush=True,
        )
        return audio

    print(
        f"  [Segment {segment_idx}] Drift correction: "
        f"actual={actual_duration:.3f}s target={target_duration:.3f}s "
        f"ratio={ratio:.3f}",
        flush=True,
    )

    try:
        _timeout = float(os.environ.get("DUBBING_DRIFT_TIMEOUT", "").strip() or DRIFT_TIMEOUT_DEFAULT)
    except ValueError:
        _timeout = DRIFT_TIMEOUT_DEFAULT

    def _stretch():
        import pyrubberband as pyrb
        return pyrb.time_stretch(audio, INDICF5_SAMPLE_RATE, ratio)

    try:
        ok, corrected = _run_with_timeout(_stretch, _timeout)
        if not ok:
            print(
                f"  [Segment {segment_idx}] Drift correction TIMED OUT after {_timeout:g}s "
                f"(rubberband subprocess wedged). Using UNCORRECTED audio and continuing.",
                flush=True,
            )
            return audio
        return corrected.astype(np.float32)
    except ImportError:
        print(
            f"  [Segment {segment_idx}] pyrubberband not installed. "
            f"Skipping drift correction. Install with: pip install pyrubberband",
            flush=True,
        )
        return audio
    except Exception as e:
        print(f"  [Segment {segment_idx}] Drift correction failed: {e}. Using original.", flush=True)
        return audio


# ---------------------------------------------------------------------------
# Resume / checkpoint support
# ---------------------------------------------------------------------------
# Step 6 writes a small JSON manifest next to the segment WAVs, updated atomically
# after every segment. On a re-run (e.g. after killing a wedged kernel) segments that
# already completed as real audio are skipped, so only the unfinished tail is redone.
# temp_dir in app.py is a fixed path that is never wiped, so the WAVs + manifest persist
# across runs. Segments that fell back to silence (timeout/error) are recorded as
# "placeholder", NOT "ok", so they are retried on the next run rather than skipped.

MANIFEST_NAME = "tts_manifest.json"


def resolve_nfe_step(nfe_step: Optional[int] = None, log_fn=None) -> int:
    """Resolve the diffusion step count: explicit arg > DUBBING_NFE_STEP env > 32.

    32 is the IndicF5 quality baseline; the knob exists so latency can be traded against
    quality deliberately (never lowered silently). This is the ONE resolver — the fan-out
    orchestrator (deploy/tts_fanout.py) calls it too, because nfe_step is part of the
    segment signature: if the two paths resolved it differently they would compute
    different signatures for identical work and silently re-synthesize everything.
    """
    def _say(m):
        if log_fn is not None:
            try:
                log_fn(m)
            except Exception:
                pass

    if nfe_step is None:
        _env_nfe = os.environ.get("DUBBING_NFE_STEP", "").strip()
        if _env_nfe:
            try:
                nfe_step = int(_env_nfe)
            except ValueError:
                _say(f"WARNING: DUBBING_NFE_STEP='{_env_nfe}' is not an integer; using default 32.")
                nfe_step = 32
        else:
            nfe_step = 32
    if nfe_step < 1:
        _say(f"WARNING: nfe_step={nfe_step} is invalid; clamping to 1.")
        nfe_step = 1
    return nfe_step


def _segment_signature(text: str, target_duration: float, lang_code: str, nfe_step: int) -> str:
    """Stable hash of everything that determines a segment's audio. If any of these change
    between runs (e.g. translation re-ran and produced different text), the signature
    changes and the cached WAV is treated as stale and re-synthesized — never silently reused."""
    payload = f"{lang_code}|{nfe_step}|{target_duration:.4f}|{text}"
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


def _load_manifest(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _json_default(o):
    """Coerce a numpy scalar (bool_ / int64 / float64 / …) to its native Python type so
    json.dump can serialize a manifest entry that carries one — segment metadata can pick up
    a numpy.bool_/numpy.int64 from an upstream gate comparison, and those are NOT
    JSON-serializable (unlike numpy.float64, which subclasses float). Applied as the
    ``default=`` hook, it fires ONLY for values the stdlib encoder rejects. Mirrors the
    identical helpers in app.py and tts_supervisor.py."""
    import numpy as _np
    if isinstance(o, _np.generic):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _save_manifest_atomic(path: str, manifest: dict) -> None:
    """Write the manifest to a temp file then os.replace() it, so a kill mid-write can
    never leave a half-written (unparseable) manifest that would defeat resume."""
    # Unique temp name so that if a manifest write is ever abandoned by a timeout
    # guard (same overlay-FS stall class as the segment WAV writes), the abandoned
    # writer can't collide with the next attempt's temp file.
    import uuid
    try:
        tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=0, default=_json_default)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  [Manifest] WARNING: could not persist resume manifest: {e}", flush=True)


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
    log_fn=None,
    nfe_step: Optional[int] = None,
    watchdog_mode: str = "thread",
    heartbeat_path: Optional[str] = None,
    only_indices: Optional[set] = None,
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
        log_fn:                Optional callable(str) that receives every progress
                               line so a UI (e.g. the Streamlit Pipeline Log) can
                               show download / model-load / per-segment progress.
                               When None, output goes to stdout only (prior behaviour).
        nfe_step:              Diffusion (denoising) steps per segment. Synthesis time
                               is ~linear in it. None (default) resolves to the
                               DUBBING_NFE_STEP env var, else 32 (quality baseline).
                               Lowering to ~16 roughly halves Step 6 latency at some
                               quality cost.
        watchdog_mode:         How a hung GPU synthesis is bounded.
                               'thread' (default): an in-process daemon-thread timer
                               (_run_with_timeout) bounds each segment and HALTS Step 6
                               on stall so a re-run resumes from disk. This is the
                               standalone/legacy behaviour; it CANNOT kill a wedged CUDA
                               op — it only stops issuing more GPU work.
                               'none': run synthesis straight-through with NO in-process
                               timer, and (if heartbeat_path is set) touch a heartbeat
                               file as each segment settles. Use this ONLY under an
                               external supervisor (pipeline.tts_supervisor) that owns
                               liveness: it SIGKILLs and relaunches a fresh child on a
                               stalled heartbeat, and on-disk resume makes the
                               continuation seamless. A wedged CUDA op is unkillable from
                               inside its own process — killing the whole process is the
                               only reliable cure, so 'none' + supervisor is the real fix
                               for the Step-6 freeze; 'thread' is the in-process
                               approximation.
        heartbeat_path:        When watchdog_mode='none', a file whose mtime/JSON the
                               supervisor polls for liveness. Written best-effort (never
                               raises, never materially blocks): boot -> loading -> synth
                               with the current segment index. Ignored when None.
        only_indices:          Restrict synthesis to this set of GLOBAL segment indices
                               (fan-out sharding). The full segment list is still passed
                               in, so indices, filenames and manifest keys are identical
                               to a serial run and shards merge without renumbering; the
                               segments outside the set are returned untouched and not
                               synthesized. None (default) = synthesize everything, the
                               serial behaviour. See deploy/tts_fanout.py.

    Returns:
        Same list of segments, with 'audio_path' added to each element.
    """
    if not SOUNDFILE_AVAILABLE:
        raise RuntimeError("soundfile is required. Run: pip install soundfile")

    from datetime import datetime
    def tts_log(msg: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{t}] [DurationTTS] {msg}", flush=True)
        if log_fn is not None:
            try:
                log_fn(f"  {msg}")
            except Exception:
                pass

    lang_code = LANGUAGE_TO_CODE.get(target_language)
    if not lang_code:
        raise ValueError(
            f"Unsupported language: '{target_language}'. "
            f"Supported: {list(LANGUAGE_TO_CODE.keys())}"
        )

    os.makedirs(output_dir, exist_ok=True)

    if watchdog_mode not in ("thread", "none"):
        tts_log(f"WARNING: unknown watchdog_mode='{watchdog_mode}'; falling back to 'thread'.")
        watchdog_mode = "thread"

    def _beat(phase: str, seg: int = -1, done: int = 0):
        """Best-effort liveness ping for an external supervisor (watchdog_mode='none').

        Atomically writes {phase, seg, done, ts} so the supervisor can (a) tell loading
        from synthesis and (b) see which segment we're on. A wedged CUDA op simply stops
        updating this file — that stalled mtime IS the signal the supervisor kills on, so
        this MUST be touched only by the thread doing the work (never a background timer),
        and only as real progress is made. Never raises, never blocks materially."""
        if not heartbeat_path:
            return
        try:
            tmp = f"{heartbeat_path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"phase": phase, "seg": seg, "done": done, "ts": time.time()}, fh,
                          default=_json_default)
            os.replace(tmp, heartbeat_path)
        except Exception:
            pass

    _beat("boot")

    nfe_step = resolve_nfe_step(nfe_step, log_fn=tts_log)
    if nfe_step != 32:
        tts_log(f"Diffusion steps: nfe_step={nfe_step} (default is 32 — latency/quality trade-off active).")
    else:
        tts_log("Diffusion steps: nfe_step=32 (quality baseline).")

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
    _beat("loading")  # supervisor grants a longer stall budget while phase == loading
    model, resolved_device = _load_indicf5(device, log_fn=log_fn)
    tts_log(f"IndicF5 model ready on device: {resolved_device}")
    _beat("synth")  # loaded — from here the supervisor switches to the per-segment budget

    # Load the resume manifest (if a prior run left one) so already-finished segments
    # can be skipped instead of re-synthesized. See the "Resume / checkpoint" section above.
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    manifest = _load_manifest(manifest_path)
    n_total = len(translated_segments)
    resumed_count = 0

    def _record(idx: int, status: str, signature: str, path: str,
                synth_seconds: Optional[float] = None):
        """Update + atomically persist the manifest after a segment settles.

        The manifest write hits the same /kaggle/working overlay FS as the segment WAVs,
        so it is bounded by the same wall-clock timeout: a stalled manifest write is
        abandoned (in-memory ``manifest`` is untouched) and the loop keeps going. Missing
        a manifest update only means the segment is re-synthesized on resume — safe and
        idempotent — which is far better than freezing the whole run on a JSON write."""
        entry = {"status": status, "sig": signature, "path": path}
        if synth_seconds is not None:
            # Additive latency field only — NOT part of the resume identity (status/sig/
            # path), so the resume-skip check is unaffected. Absent on empty-text/
            # placeholder/watchdog records; present only on a real "ok" synthesis.
            entry["synth_seconds"] = round(float(synth_seconds), 4)
        manifest[str(idx)] = entry
        ok, _ = _run_with_timeout(
            _save_manifest_atomic, _disk_write_timeout(), manifest_path, manifest
        )
        if not ok:
            tts_log(
                f"[Segment {idx}/{n_total-1}] Manifest write stalled >{_disk_write_timeout():g}s "
                f"(disk wedge) — continuing; segment will be re-checked on resume."
            )

    results = []
    for i, seg in enumerate(translated_segments):
        # ── Shard filter (fan-out) ────────────────────────────────────────────
        # `only_indices` lets a worker synthesize just its slice of a run while still
        # seeing the WHOLE segment list, so `i` stays the GLOBAL index: filenames
        # (segment_%04d.wav) and manifest keys remain identical to a serial run, and the
        # shards' outputs merge without renumbering. Skipped segments are still appended
        # so the returned list stays 1:1 with the input; they are simply not synthesized
        # and their manifest entries are left untouched for the owning shard to write.
        if only_indices is not None and i not in only_indices:
            results.append({**seg, "audio_path": os.path.join(output_dir, f"segment_{i:04d}.wav")})
            continue

        # Advance the heartbeat as we move to each segment. If synthesis of THIS segment
        # wedges, no further beat fires and the supervisor kills+relaunches after the
        # per-segment stall budget. (Covers every path below, incl. the resume-skip and
        # empty-text continues — the next iteration's beat only fires if we didn't hang.)
        _beat("synth", seg=i, done=len(results))
        text = seg.get("text", "").strip()
        target_duration = seg["end"] - seg["start"]
        out_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
        signature = _segment_signature(text, target_duration, lang_code, nfe_step)

        # ── Resume: skip segments a prior run already finished as real audio ──
        # Only an "ok" segment whose file still exists (non-empty) AND whose signature
        # matches the current text/duration/lang/nfe is trusted. "placeholder" (silence
        # fallback) entries and signature mismatches are deliberately re-synthesized.
        prior = manifest.get(str(i))
        if (
            prior
            and prior.get("status") == "ok"
            and prior.get("sig") == signature
            and os.path.exists(out_path)
            and os.path.getsize(out_path) > 44  # bigger than a bare WAV header
        ):
            resumed_count += 1
            tts_log(f"[Segment {i}/{n_total-1}] Resuming from disk — already complete, skipping.")
            results.append({**seg, "audio_path": out_path})
            continue

        if not text:
            tts_log(f"[Segment {i}/{n_total-1}] Empty text — writing silence.")
            silence = np.zeros(int(target_duration * INDICF5_SAMPLE_RATE), dtype=np.float32)
            # Even this tiny write goes through the bounded writer — the overlay-FS stall
            # that wedged the main writes doesn't care how small the buffer is.
            if _write_wav_with_timeout(out_path, silence, INDICF5_SAMPLE_RATE, i, log_fn):
                # Empty-text silence is the correct, final output for this segment
                # (deterministic), so mark it "ok" — a future run should skip it, not retry it.
                _record(i, "ok", signature, out_path)
            else:
                tts_log(f"[Segment {i}/{n_total-1}] Could not write empty-text silence; marked for retry.")
                _record(i, "placeholder", signature, out_path)
            results.append({**seg, "audio_path": out_path})
            continue

        tts_log(
            f"Processing [Segment {i}/{n_total-1}]: "
            f"target={target_duration:.2f}s | text='{text[:60]}...'"
        )

        # ── Per-segment watchdog ─────────────────────────────────────────────
        # The whole heavy body (GPU synthesis → drift → disk write) runs under one
        # wall-clock ceiling. Drift and the disk write have their own finer internal
        # timeouts that fire first and self-recover; this outer watchdog exists to bound
        # the ONE otherwise-unbounded piece: GPU synthesis. Closure over the loop vars is
        # safe — we run it synchronously in THIS iteration, before i/text/etc. change.
        def _process_segment():
            # Time ONLY the GPU synthesis — the part nfe_step governs, and the clean
            # per-segment latency the nfe sweep grades. Drift correction and the disk
            # write are nfe-independent overhead (and the write has its own stall/retry
            # path), so they are deliberately outside the timer.
            _synth_t0 = time.perf_counter()
            audio = _generate_single_segment(
                model=model,
                device=resolved_device,
                text=text,
                target_duration=target_duration,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                lang_code=lang_code,
                log_fn=log_fn,
                nfe_step=nfe_step,
            )
            synth_seconds = time.perf_counter() - _synth_t0

            actual_duration = len(audio) / INDICF5_SAMPLE_RATE

            # Heartbeats (2026-09-08): these flushed markers originally pinned the freeze
            # to the disk write (now bounded below). They stay — one cheap print/segment
            # that still names the exact call if any disk op ever wedges again.
            tts_log(f"[Segment {i}/{n_total-1}] Synthesis returned (actual={actual_duration:.2f}s) — checking drift...")
            audio = _apply_drift_correction(audio, actual_duration, target_duration, i)

            # Acoustic silence cushion: Prevents onset fade attenuation of initial plosives/consonants
            # and protects trailing consonants from crossfade truncation.
            cushion_lead = int(0.015 * INDICF5_SAMPLE_RATE)   # 15ms lead cushion
            cushion_trail = int(0.035 * INDICF5_SAMPLE_RATE)  # 35ms trail cushion
            audio = np.pad(audio, (cushion_lead, cushion_trail), mode="constant", constant_values=0.0)

            tts_log(f"[Segment {i}/{n_total-1}] Drift check done — writing WAV to disk ({out_path})...")
            wrote = _write_wav_with_timeout(out_path, audio, INDICF5_SAMPLE_RATE, i, log_fn)
            if not wrote:
                # Every write attempt stalled on the FS (disk, not GPU). Record a
                # placeholder so resume retries just this segment; the run continues.
                tts_log(
                    f"[Segment {i}/{n_total-1}] WARNING: could not persist WAV (disk wedged); "
                    f"marked for retry on next run. Continuing."
                )
                _record(i, "placeholder", signature, out_path)
                return

            tts_log(f"[Segment {i}/{n_total-1}] WAV written — updating resume manifest...")
            _record(i, "ok", signature, out_path, synth_seconds=synth_seconds)
            tts_log(f"[Segment {i}/{n_total-1}] Completed successfully. Saved -> {out_path}")

        def _fallback_to_silence(exc):
            # A RAISED exception means the call returned control (not a hang), so the GPU
            # is not wedged — fall back to silence and continue to the next segment.
            # Silence is a FALLBACK, not intended output — marked "placeholder" for retry.
            tts_log(f"ERROR: [Segment {i}] TTS failed: {exc}. Writing silence.")
            import traceback
            traceback.print_exc()
            silence = np.zeros(int(target_duration * INDICF5_SAMPLE_RATE), dtype=np.float32)
            if not _write_wav_with_timeout(out_path, silence, INDICF5_SAMPLE_RATE, i, log_fn):
                tts_log(f"[Segment {i}/{n_total-1}] Could not write the silence fallback either; marked for retry.")
            _record(i, "placeholder", signature, out_path)

        if watchdog_mode == "none":
            # ── External-supervisor mode ─────────────────────────────────────
            # No in-process timer: a wedged CUDA op is unkillable from inside its own
            # process, so we don't pretend otherwise. pipeline.tts_supervisor watches the
            # heartbeat this loop touches and, on a stall, SIGKILLs the whole child process
            # group and relaunches a fresh one that resumes from the on-disk manifest. That
            # is the real cure for the Step-6 freeze. A segment that merely ERRORS (returns
            # control) is still degraded to silence here so one bad line never aborts the run.
            try:
                _process_segment()
            except Exception as e:
                _fallback_to_silence(e)
        else:
            # ── In-process thread watchdog (default / standalone) ────────────
            segment_timeout = _segment_timeout()
            finished = False
            try:
                finished, _ = _run_with_timeout(_process_segment, segment_timeout)
            except Exception as e:
                _fallback_to_silence(e)
                finished = True  # error handled; safe to move to the next segment

            if not finished:
                # WATCHDOG FIRED: the segment exceeded its wall-clock ceiling — almost
                # certainly a wedged GPU synthesis (drift + disk have their own shorter
                # timeouts and self-recover). The abandoned worker thread may still be inside
                # a CUDA op on the shared model, so issuing ANY more GPU work this run risks
                # corrupting it. Persist a retry marker and HALT Step 6 gracefully: progress
                # is on disk, and the next run auto-resumes from exactly this segment.
                done_ok = len([k for k, v in manifest.items() if v.get("status") == "ok"])
                tts_log(
                    f"[Segment {i}/{n_total-1}] WATCHDOG: exceeded {segment_timeout:g}s "
                    f"(likely a wedged GPU synthesis) — abandoning this segment."
                )
                _record(i, "placeholder", signature, out_path)
                results.append({**seg, "audio_path": out_path})
                tts_log(
                    f"Step 6 halted after segment {i} to protect the shared TTS model from a "
                    f"possibly-wedged GPU. Progress saved ({done_ok} segment(s) done, "
                    f"{resumed_count} reused). Re-run Step 6 — it resumes automatically from segment {i}."
                )
                raise RuntimeError(
                    f"Step 6 halted: segment {i} hung >{segment_timeout:g}s and was abandoned to "
                    f"protect the shared TTS model from a possibly-wedged GPU. Progress is saved; "
                    f"re-run Step 6 to resume automatically from segment {i}."
                )

        # Periodically clear the CUDA cache + run GC to guard against OOM on low-VRAM
        # GPUs (e.g. the T4). Deliberately NOT every segment: torch.cuda.empty_cache()
        # releases the caching allocator's blocks back to the driver, so the next
        # segment pays a slow cudaMalloc — doing that 64× serially both adds direct
        # overhead and defeats the allocator. Every 8th segment (and the last) keeps
        # the OOM protection while cutting that overhead ~8x. IndicF5's per-segment
        # VRAM is small and roughly constant, so this is safe.
        is_last = (i == len(translated_segments) - 1)
        if TORCH_AVAILABLE and (i % 8 == 0 or is_last):
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        results.append({**seg, "audio_path": out_path})

    placeholders = [k for k, v in manifest.items() if v.get("status") == "placeholder"]
    if resumed_count:
        tts_log(f"Resume: {resumed_count}/{n_total} segment(s) reused from a previous run.")
    if placeholders:
        tts_log(
            f"WARNING: {len(placeholders)} segment(s) fell back to silence "
            f"(indices {sorted(int(k) for k in placeholders)}). Re-run Step 6 to retry just those."
        )
    tts_log(f"Generated {len(results)} audio segments in {output_dir}")
    return results


