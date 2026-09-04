# -*- coding: utf-8 -*-
"""
build_notebook.py
=================
Run this script on your local machine ONCE to produce:
    indicai_dubbing_kaggle.ipynb

Then upload that .ipynb to Kaggle as a new notebook.

Usage:
    python build_notebook.py
"""

import json, textwrap, os, sys

# ──────────────────────────────────────────────────────────────────────────────
# Helper – dedent + strip leading blank line
# ──────────────────────────────────────────────────────────────────────────────
def src(*lines):
    return "\n".join(lines)

def code_cell(source_str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_str,
    }

def md_cell(source_str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_str,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CELL CONTENTS
# ══════════════════════════════════════════════════════════════════════════════

CELL_MD_HEADER = r"""# 🎙️ Indic AI Dubbing Platform — Kaggle GPU Edition
### Powered by IndicF5 • Demucs • Gemini • Cloudflare Tunnel

This notebook runs the **complete** Hindi dubbing pipeline on Kaggle's free GPU
and exposes the Streamlit UI through a **Cloudflare Quick Tunnel** (no account needed).

---

## ⚙️ One-time Kaggle Setup (do this before running cells)

### 1. Enable GPU & Internet
Go to **Notebook Settings** (right sidebar) → set:
- **Accelerator** → `GPU T4 x1` (or P100)
- **Internet** → `On`

### 2. Add Secrets
Go to **Add-ons → Secrets** and add:

| Secret Name | Value |
|---|---|
| `GEMINI_API_KEY` | Your Google AI Studio key (free at [aistudio.google.com](https://aistudio.google.com)) |
| `HF_TOKEN` | Your HuggingFace token — **must have accepted [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) gate** |

### 3. Run All Cells in Order
`Run All` → Wait ~15 min for deps + model download → open the tunnel URL printed at the bottom.

---
> ⚠️ Kaggle free GPU sessions last up to **12 hours** / **30 hrs per week**. The Cloudflare URL resets each session.
"""

# ──────────────────────────────────────────────────────────────────────────────
CELL_GPU_CHECK = r"""import torch, subprocess, sys, os

print("=" * 60)
print("  INDIC AI DUBBING PLATFORM - KAGGLE GPU EDITION")
print("=" * 60)
print(f"\nPython      : {sys.version.split()[0]}")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM        : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
else:
    print("WARNING: No GPU found! Go to Notebook Settings → Accelerator → GPU T4 x1")
print(f"Working dir : {os.getcwd()}")
"""

# ──────────────────────────────────────────────────────────────────────────────
CELL_SYSTEM_DEPS = r"""import subprocess, sys

print("Installing system packages (ffmpeg, espeak-ng, fonts)...")
subprocess.run(
    ["apt-get", "install", "-y", "-q",
     "ffmpeg", "rubberband-cli", "espeak-ng",
     "fonts-noto", "libsndfile1"],
    check=True, capture_output=True
)
print("  ✓ System packages installed")

# Verify ffmpeg
result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
ver = result.stdout.split("\n")[0] if result.returncode == 0 else "NOT FOUND"
print(f"  ffmpeg: {ver[:60]}")
"""

# ──────────────────────────────────────────────────────────────────────────────
CELL_PYTHON_DEPS_1 = r"""import subprocess, sys

# ── Batch 1: Core audio + utility packages ───────────────────────────────────
pkgs = [
    "soundfile",
    "pydub",
    "scipy==1.13.1",         # last scipy built for the numpy 1.x ABI — MUST pair with numpy 1.26.4
                             # (unpinned pulls a numpy-2.x wheel -> scipy.special sph_legendre_p ABI crash)
    "resampy",
    "numpy==1.26.4",         # pin <2 — demucs/numba + torch 2.5 expect the NumPy 1.x ABI
    "huggingface_hub",
    "safetensors",
    "faster-whisper",
    "openai-whisper",        # transcription fallback if faster-whisper fails at runtime
    "demucs",
    "phonemizer",
    "indic-nlp-library",     # orthographic normalization for the real phoneme counter
    "sentence-transformers", # IndicSBERT cross-lingual semantic gate
    "pyrubberband",
]

print("Installing core packages...")
for pkg in pkgs:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                       capture_output=True, text=True)
    status = "✓" if r.returncode == 0 else "✗"
    print(f"  {status} {pkg}")

# ── Batch 2: Google Gemini SDK ────────────────────────────────────────────────
print("\nInstalling google-genai...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai"], check=True)
print("  ✓ google-genai")

# ── Batch 3: vocos (vocoder used by F5-TTS) ───────────────────────────────────
print("\nInstalling vocos...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vocos"], check=True)
print("  ✓ vocos")

print("\n✅ Batch 1 complete")
"""

# ──────────────────────────────────────────────────────────────────────────────
CELL_PYTHON_DEPS_2 = r"""import subprocess, sys

# F5-TTS must be installed BEFORE IndicF5 (IndicF5 depends on it)
print("Installing f5-tts (this may take ~2 min)...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "f5-tts"],
    capture_output=True, text=True
)
if r.returncode == 0:
    print("  ✓ f5-tts installed")
else:
    print(f"  ✗ f5-tts install failed:\n{r.stderr[-500:]}")

# IndicF5 – AI4Bharat's Indic TTS model
print("\nInstalling IndicF5 from GitHub (this may take ~3 min)...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "git+https://github.com/ai4bharat/IndicF5.git"],
    capture_output=True, text=True
)
if r.returncode == 0:
    print("  ✓ IndicF5 installed")
else:
    print(f"  ✗ IndicF5 install failed:\n{r.stderr[-500:]}")

# Streamlit
print("\nInstalling streamlit...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit"], check=True)
print("  ✓ streamlit")

# ── Re-assert transformers < 5.0.0 AFTER f5-tts/IndicF5 ───────────────────────
# f5-tts / IndicF5 can resolve transformers to a 5.x release. On 5.x the IndicF5
# DiT/vocoder load hits a meta-tensor error (the low_cpu_mem_usage default path
# changed), crashing Step 6 (TTS). Force it back into the tested 4.x range LAST
# so the model actually loads. This runs BEFORE the numpy re-assert on purpose —
# a transformers (re)install can itself drag in numpy 2.x, so numpy stays last.
print("\nRe-asserting transformers<5.0.0 (guards the IndicF5 meta-tensor crash)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers<5.0.0"],
    check=True,
)
_tv = subprocess.run(
    [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
    capture_output=True, text=True,
)
if _tv.returncode == 0:
    _tvs = _tv.stdout.strip()
    print(f"  ✓ transformers now {_tvs}")
    if _tvs.split(".")[0].isdigit() and int(_tvs.split(".")[0]) >= 5:
        print(f"  ⚠ transformers is {_tvs} (>=5) — IndicF5 will likely hit a "
              "meta-tensor error at Step 6. Restart the kernel and Run All.")
else:
    print("  ⚠ could not verify transformers version:",
          _tv.stderr.strip().splitlines()[-1] if _tv.stderr.strip() else "(no stderr)")

# ── Re-assert numpy 1.x AFTER f5-tts/IndicF5/vocos ────────────────────────────
# Those installs can silently drag in a wheel built against numpy 2.x, leaving
# the env with a 2.x-built extension over a 1.26 runtime -> "numpy.dtype size
# changed (Expected 96 ... got 88)" ABI errors when transformers/torch import
# during the IndicF5 load. Force numpy back to the pinned 1.x last so the final
# environment is coherent when the model actually loads at dubbing time.
print("\nRe-asserting numpy==1.26.4 (guards against a 2.x ABI mismatch)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
     "--no-deps", "numpy==1.26.4"],
    check=True,
)

# Verify coherence the way it actually matters. The dubbing pipeline runs inside
# the `streamlit run app.py` SUBPROCESS, which imports a FRESH numpy/scipy from
# disk — so what counts is on-disk coherence, NOT this kernel's already-loaded
# numpy. importlib.reload() cannot hot-swap a loaded C-extension, so it only ever
# printed a misleading "green". Instead we probe in a throwaway subprocess that
# mirrors the app: import numpy + scipy.special (where the sph_legendre_p ufunc
# lives) and exercise a compiled scipy path. If scipy was built against numpy 2.x
# over a 1.26 runtime, this fails LOUDLY here instead of mid-dub.
_probe = (
    "import numpy, scipy, scipy.special\n"
    "from scipy.spatial.distance import cosine\n"
    "cosine([1.0, 0.0], [0.0, 1.0])\n"
    "print(numpy.__version__ + '|' + scipy.__version__)\n"
)
_r = subprocess.run([sys.executable, "-c", _probe], capture_output=True, text=True)
if _r.returncode == 0:
    _nv, _sv = _r.stdout.strip().split("|")
    print(f"  ✓ Fresh-process import OK — numpy {_nv}, scipy {_sv} (ABI coherent)")
    if not _nv.startswith("1.26"):
        print(f"  ⚠ numpy resolved to {_nv}, not 1.26.x — the app subprocess may "
              "hit an ABI error. Restart the kernel and Run All before launching.")
else:
    _last = _r.stderr.strip().splitlines()[-1] if _r.stderr.strip() else "(no stderr)"
    print("  ✗ Fresh-process numpy/scipy import FAILED — this is the ABI mismatch "
          "that would crash the dubbing run:")
    print("   ", _last)
    print("    Fix: ensure scipy==1.13.1 (the numpy-1.x ABI pair) installed above, "
          "then Restart Kernel and Run All.")

print("\n✅ AI/TTS packages installed!")
"""

# ──────────────────────────────────────────────────────────────────────────────
CELL_API_KEYS = r"""import os

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()

    try:
        GEMINI_API_KEY = secrets.get_secret("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        print(f"✓ GEMINI_API_KEY loaded (preview: {GEMINI_API_KEY[:8]}...)")
    except Exception as e:
        print(f"⚠ GEMINI_API_KEY not in secrets: {e}")
        GEMINI_API_KEY = ""

    try:
        HF_TOKEN = secrets.get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = HF_TOKEN
        print(f"✓ HF_TOKEN loaded (preview: {HF_TOKEN[:8]}...)")
    except Exception as e:
        print(f"⚠ HF_TOKEN not in secrets: {e}")
        HF_TOKEN = ""

except ImportError:
    print("Not running on Kaggle — reading from environment variables")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not GEMINI_API_KEY:
    print("\n⛔ GEMINI_API_KEY missing! Add it via Add-ons → Secrets before running Step 4+.")
if not HF_TOKEN:
    print("\n⛔ HF_TOKEN missing! You need it to download the gated IndicF5 model weights.")
"""

# ──────────────────────────────────────────────────────────────────────────────
# CELL: Write all pipeline files
# This is the big cell that creates the complete pipeline on Kaggle's filesystem
# ──────────────────────────────────────────────────────────────────────────────

# We read each source file and embed it as a Python string literal
def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def _make_write_cell():
    """Build Python code that writes all pipeline files to /kaggle/working/"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(script_dir)  # E:/Dubbing app

    # Files to embed: (source_path, dest_path_on_kaggle)
    file_map = [
        (os.path.join(base, "app.py"),                          "/kaggle/working/app.py"),
        (os.path.join(base, "pipeline", "phoneme_counter.py"), "/kaggle/working/pipeline/phoneme_counter.py"),
        (os.path.join(base, "pipeline", "semantic_similarity.py"), "/kaggle/working/pipeline/semantic_similarity.py"),
        (os.path.join(base, "pipeline", "source_separation.py"),"/kaggle/working/pipeline/source_separation.py"),
        (os.path.join(base, "pipeline", "voice_manager.py"),    "/kaggle/working/pipeline/voice_manager.py"),
        (os.path.join(base, "pipeline", "translation_cache.py"), "/kaggle/working/pipeline/translation_cache.py"),
        (os.path.join(base, "pipeline", "isochrony_translation.py"), "/kaggle/working/pipeline/isochrony_translation.py"),
        (os.path.join(base, "utils", "audio_extraction.py"),    "/kaggle/working/utils/audio_extraction.py"),
        (os.path.join(base, "utils", "transcription.py"),       "/kaggle/working/utils/transcription.py"),
        (os.path.join(base, "utils", "audio_sync.py"),          "/kaggle/working/utils/audio_sync.py"),
    ]

    lines = [
        "import os",
        "",
        "# Create directory structure",
        "for d in ['/kaggle/working/pipeline', '/kaggle/working/utils', '/kaggle/working/temp_processing']:",
        "    os.makedirs(d, exist_ok=True)",
        "",
        "# Write empty __init__.py files",
        "for init in ['/kaggle/working/pipeline/__init__.py', '/kaggle/working/utils/__init__.py']:",
        "    open(init, 'w').close()",
        "",
    ]

    for src_path, dst_path in file_map:
        if not os.path.exists(src_path):
            print(f"WARNING: {src_path} not found — skipping")
            continue
        content = _read(src_path)
        # Use repr() to get a valid Python string literal with all escaping done
        repr_content = repr(content)
        lines.append(f"# ── {os.path.basename(dst_path)} ──")
        lines.append(f"with open({repr(dst_path)}, 'w', encoding='utf-8') as _f:")
        lines.append(f"    _f.write({repr_content})")
        lines.append("")

    # Kaggle-adapted files (written inline, not from Windows source)
    lines.append(_make_kaggle_video_merge())
    lines.append(_make_kaggle_duration_tts())
    lines.append(_make_kaggle_model_patch())

    lines.append("")
    lines.append("print('✅ All pipeline files written to /kaggle/working/')")
    lines.append("import subprocess")
    lines.append("r = subprocess.run(['python', '-c', 'import pipeline.phoneme_counter; print(\"Import OK\")'],")
    lines.append("                    capture_output=True, text=True, cwd='/kaggle/working')")
    lines.append("print('Import test:', r.stdout.strip() or r.stderr.strip())")

    return "\n".join(lines)


def _make_kaggle_video_merge():
    """Return Python code string that writes the Linux-adapted video_merge.py"""
    content = r'''import os
import subprocess

def merge_video_audio_subs(video_path: str, audio_path: str, srt_path: str, output_path: str):
    """
    Merges the original video, the new dubbed audio, and the subtitle file using FFmpeg.
    Linux-compatible (no Windows path escaping needed).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vf", "subtitles='" + srt_path + "':force_style='FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1'",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    print(f"Running FFmpeg: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print("Video merging complete.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error:\n{e.stderr}")
        raise RuntimeError(f"FFmpeg failed:\n{e.stderr}")
    return output_path
'''
    return (
        "# ── utils/video_merge.py (Linux-adapted) ──\n"
        f"with open('/kaggle/working/utils/video_merge.py', 'w', encoding='utf-8') as _f:\n"
        f"    _f.write({repr(content)})\n"
    )


def _make_kaggle_duration_tts():
    """Return Python code that writes the Kaggle-adapted duration_tts.py"""
    # Read the original
    orig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "pipeline", "duration_tts.py")
    with open(orig_path, encoding="utf-8") as f:
        orig = f.read()

    # Replace the Windows-specific cache path logic with Linux glob-based approach
    WIN_BLOCK = '''        _cache_dir = os.path.expandvars(
            r"%USERPROFILE%\\.cache\\\\huggingface\\\\modules\\\\transformers_modules"
            r"\\\\ai4bharat\\\\IndicF5\\\\ba85abedf18dc479a447eaa0eccbd76ab78a47d5"
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
                pass  # Crash expected; we only needed the cache to populate'''

    LINUX_BLOCK = '''        # Find cached model.py using glob — works across commit hashes on Linux
        import glob as _glob
        _patterns = _glob.glob(os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py"
        ))
        _model_py = _patterns[0] if _patterns else None

        if not _model_py or not os.path.exists(_model_py):
            load_log("HF cache miss — triggering one-time model file download...")
            try:
                from transformers import AutoModel
                AutoModel.from_pretrained(
                    "ai4bharat/IndicF5",
                    trust_remote_code=True,
                    token=token,
                )
            except Exception:
                pass
            _patterns = _glob.glob(os.path.expanduser(
                "~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py"
            ))
            _model_py = _patterns[0] if _patterns else None'''

    adapted = orig
    # The exact Windows block might not match due to escaping, so find a unique substring
    # Use a simpler marker approach
    marker_start = '_cache_dir = os.path.expandvars('
    marker_end = 'pass  # Crash expected; we only needed the cache to populate'

    if marker_start in adapted and marker_end in adapted:
        idx_start = adapted.index(marker_start)
        idx_end = adapted.index(marker_end) + len(marker_end)
        # Find the 8-space indent block start
        line_start = adapted.rfind('\n', 0, idx_start) + 1
        adapted = adapted[:line_start] + LINUX_BLOCK + adapted[idx_end:]
    else:
        # Fallback: append a note
        adapted += "\n# NOTE: Run on Kaggle — uses glob-based cache path\n"

    # Remove Windows SSL bypass (not needed on Kaggle, causes import issues)
    # Keep it in (it's harmless on Linux and prevents SSL errors)

    return (
        "# ── pipeline/duration_tts.py (Kaggle-adapted) ──\n"
        f"with open('/kaggle/working/pipeline/duration_tts.py', 'w', encoding='utf-8') as _f:\n"
        f"    _f.write({repr(adapted)})\n"
    )


def _make_kaggle_model_patch():
    """Return Python code that writes the patched model.py (no Windows paths)"""
    content = r'''import sys
import os
from datetime import datetime


def debug_log(msg: str):
    """Log to stdout (Kaggle-compatible, no Windows file paths)."""
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{t}] [HF_Model_Loader] {msg}", flush=True)


debug_log("Starting cached model.py initialization...")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

debug_log("Importing transformers and PyTorch...")
try:
    from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
    import torch
    import numpy as np
    debug_log("Core imports successful.")
except Exception as e:
    debug_log(f"ERROR: Failed during core imports: {e}")
    raise

debug_log("Importing f5_tts utilities...")
try:
    from f5_tts.infer.utils_infer import (
        infer_process,
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
    )
    from f5_tts.model import DiT
    debug_log("f5_tts imports successful.")
except Exception as e:
    debug_log(f"ERROR: Failed during f5_tts imports: {e}")
    raise

debug_log("Importing soundfile, pydub, and hub utilities...")
try:
    import soundfile as sf
    import io
    from pydub import AudioSegment, silence
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    debug_log("Sound and helper imports successful.")
except Exception as e:
    debug_log(f"ERROR: Failed during sound imports: {e}")
    raise


class INF5Config(PretrainedConfig):
    model_type = "inf5"

    def __init__(self, ckpt_path: str = "checkpoints/model_best.pt",
                 vocab_path: str = "checkpoints/vocab.txt",
                 speed: float = 1.0, remove_sil: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.ckpt_path = ckpt_path
        self.vocab_path = vocab_path
        self.speed = speed
        self.remove_sil = remove_sil


class INF5Model(PreTrainedModel):
    config_class = INF5Config

    def load_state_dict(self, state_dict, strict=False):
        debug_log("Custom load_state_dict: stripping _orig_mod keys...")
        new_sd = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        return super().load_state_dict(new_sd, strict=False)

    def __init__(self, config):
        debug_log("INF5Model.__init__ triggered.")
        super().__init__(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug_log(f"Selected device: {device}")

        # Step 1: Load Vocoder to CPU first (avoid meta-tensor error with transformers >= 4.35)
        debug_log("Step 1: Loading Vocoder on cpu first...")
        try:
            vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=torch.device("cpu"))
            if str(device) != "cpu":
                vocoder = vocoder.to(device)
            self.__dict__["vocoder"] = vocoder
            debug_log("Step 1 SUCCESS: Vocoder loaded.")
        except Exception as e:
            debug_log(f"ERROR loading vocoder: {e}")
            raise

        # Step 2: Download vocab.txt
        debug_log("Step 2: Downloading vocab.txt from HuggingFace Hub...")
        try:
            vocab_path = hf_hub_download(config.name_or_path, filename="checkpoints/vocab.txt")
            debug_log(f"Step 2 SUCCESS: vocab at {vocab_path}")
        except Exception as e:
            debug_log(f"ERROR downloading vocab: {e}")
            raise

        # Step 3: Load DiT architecture (no weights yet)
        debug_log("Step 3: Loading DiT model architecture...")
        try:
            self.ema_model = load_model(
                DiT,
                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device
            )
            debug_log("Step 3 SUCCESS: DiT architecture loaded.")
        except Exception as e:
            debug_log(f"ERROR loading DiT architecture: {e}")
            raise

        # Step 3b: Load weights from model.safetensors
        # (load_checkpoint is commented out in the installed f5_tts version)
        debug_log("Step 3b: Loading weights from model.safetensors...")
        try:
            ckpt_path = hf_hub_download(config.name_or_path, filename="model.safetensors")
            debug_log(f"Step 3b: checkpoint at {ckpt_path}")
            state_dict = load_file(ckpt_path, device=str(device))
            debug_log(f"Step 3b: {len(state_dict)} total keys in checkpoint")

            # Key mapping: 'ema_model._orig_mod.X' -> 'X'
            ema_state = {}
            for k, v in state_dict.items():
                if k.startswith("ema_model."):
                    nk = k[len("ema_model."):]
                    if nk.startswith("_orig_mod."):
                        nk = nk[len("_orig_mod."):]
                    nk = nk.replace("._orig_mod.", ".")
                    ema_state[nk] = v

            debug_log(f"Step 3b: {len(ema_state)} ema_model keys extracted")
            missing, unexpected = self.ema_model.load_state_dict(ema_state, strict=False)
            debug_log(f"Step 3b SUCCESS: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                debug_log(f"  First 3 missing: {missing[:3]}")
        except Exception as e:
            debug_log(f"ERROR loading checkpoint: {e}")
            raise

        debug_log("INF5Model.__init__ completed successfully!")

    def forward(self, text: str, ref_audio_path: str, ref_text: str):
        """Generate speech given a reference audio & text input."""
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        self.ema_model.to(self.device)
        self.vocoder.to(self.device)

        audio, final_sample_rate, _ = infer_process(
            ref_audio, ref_text, text,
            self.ema_model, self.vocoder,
            mel_spec_type="vocos",
            speed=self.config.speed,
            device=self.device,
        )

        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=24000, format="WAV")
        buffer.seek(0)
        audio_seg = AudioSegment.from_file(buffer, format="wav")

        if self.config.remove_sil:
            parts = silence.split_on_silence(
                audio_seg, min_silence_len=1000,
                silence_thresh=-50, keep_silence=500, seek_step=10,
            )
            audio_seg = sum(parts, AudioSegment.silent(duration=0))

        target_dBFS = -20.0
        audio_seg = audio_seg.apply_gain(target_dBFS - audio_seg.dBFS)
        return np.array(audio_seg.get_array_of_samples())
'''
    return (
        "# ── pipeline/indicf5_model_patched.py (Kaggle-compatible model.py) ──\n"
        f"with open('/kaggle/working/pipeline/indicf5_model_patched.py', 'w', encoding='utf-8') as _f:\n"
        f"    _f.write({repr(content)})\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
CELL_PATCH_INDICF5 = r"""import os, glob, subprocess, sys
from huggingface_hub import snapshot_download
from transformers import AutoConfig

hf_token = os.environ.get("HF_TOKEN")

print("Step 1: Caching IndicF5 model files (vocab + config, skipping large safetensors)...")
try:
    snapshot_download(
        "ai4bharat/IndicF5",
        ignore_patterns=["*.safetensors", "*.bin"],
        token=hf_token,
    )
    print("  ✓ Config/vocab files cached")
except Exception as e:
    print(f"  ⚠ snapshot_download issue: {e}")

# Trigger transformers_modules caching (creates the model.py in HF cache)
print("\nStep 2: Caching transformers_modules (model.py + config.py)...")
try:
    AutoConfig.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, token=hf_token)
    print("  ✓ transformers_modules cached")
except Exception as e:
    print(f"  ⚠ Config cache: {e}")

# Find cached model.py
patterns = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py"
))
print(f"\nStep 3: Found {len(patterns)} cached model.py file(s)")

if patterns:
    model_py_path = patterns[0]
    print(f"  Path: {model_py_path}")

    # Apply our patch
    patched_src = "/kaggle/working/pipeline/indicf5_model_patched.py"
    if os.path.exists(patched_src):
        with open(patched_src, "r", encoding="utf-8") as f:
            patch_content = f.read()
        with open(model_py_path, "w", encoding="utf-8") as f:
            f.write(patch_content)
        print("  ✓ model.py patched with Kaggle-compatible version (no Windows paths, CPU-first vocoder, real DiT weights)")
    else:
        print(f"  ⚠ Patched source not found at {patched_src}")
else:
    print("  ⚠ Could not find cached model.py — it will be patched on first model load")
    print("  This is OK — the pipeline will still work, just with a longer first-load time")

print("\n✅ IndicF5 model prep complete!")
print("Note: The actual 1.3 GB model weights (model.safetensors) will be downloaded")
print("automatically on the first dubbing run inside the Streamlit app.")
"""

# ──────────────────────────────────────────────────────────────────────────────
CELL_CLOUDFLARED = r"""import subprocess, os

print("Downloading cloudflared binary...")
r = subprocess.run([
    "wget", "-q", "-O", "/usr/local/bin/cloudflared",
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
], capture_output=True, text=True)

if r.returncode == 0:
    os.chmod("/usr/local/bin/cloudflared", 0o755)
    ver = subprocess.check_output(["/usr/local/bin/cloudflared", "--version"]).decode().strip()
    print(f"  ✓ cloudflared installed: {ver}")
else:
    print(f"  ✗ Download failed: {r.stderr}")
"""

# ──────────────────────────────────────────────────────────────────────────────
CELL_LAUNCH = r"""import subprocess, time, re, os, threading

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

if not GEMINI_KEY:
    print("⚠ WARNING: GEMINI_API_KEY is not set. Translation will fail.")
if not HF_TOKEN:
    print("⚠ WARNING: HF_TOKEN is not set. Model download will fail.")

env = os.environ.copy()
env.update({"GEMINI_API_KEY": GEMINI_KEY, "HF_TOKEN": HF_TOKEN, "PYTHONIOENCODING": "utf-8"})

# ── Launch Streamlit ──────────────────────────────────────────────────────────
print("🚀 Starting Streamlit app...")
streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "/kaggle/working/app.py",
     "--server.port", "8501",
     "--server.headless", "true",
     "--browser.gatherUsageStats", "false",
     "--server.maxUploadSize", "500"],
    env=env,
    cwd="/kaggle/working",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Wait for Streamlit to be ready
print("  Waiting for Streamlit to start", end="", flush=True)
for _ in range(20):
    time.sleep(1)
    print(".", end="", flush=True)
    if streamlit_proc.poll() is not None:
        out, _ = streamlit_proc.communicate()
        print(f"\n  ✗ Streamlit exited early:\n{out.decode()[-500:]}")
        break
print()
print(f"  ✓ Streamlit running (PID: {streamlit_proc.pid})")

# ── Launch Cloudflare tunnel ──────────────────────────────────────────────────
print("\n🌐 Starting Cloudflare tunnel (waiting for URL)...")
tunnel_proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

url_found = None
start = time.time()
for line in iter(tunnel_proc.stdout.readline, b""):
    decoded = line.decode("utf-8", errors="replace").strip()
    if decoded:
        print(f"  [cloudflared] {decoded}")
    match = re.search(r"https://[^\s]+\.trycloudflare\.com", decoded)
    if match:
        url_found = match.group()
        break
    if time.time() - start > 60:
        print("  ⚠ Timed out waiting for tunnel URL (60s)")
        break

print()
if url_found:
    print("=" * 65)
    print(f"  ✅  YOUR DUBBING APP URL:")
    print(f"  👉  {url_found}")
    print("=" * 65)
    print("\nOpen the URL above in any browser.")
    print("Enter your Gemini API Key and HF Token in the sidebar, then upload a video!")
    print("\nThe URL changes each time you restart this cell.")
    print("Keep this cell running to keep the app alive.")
else:
    print("⚠ Could not extract URL. Cloudflare may still be connecting.")
    print("Check the [cloudflared] lines above for the URL.")
    print("Or wait a few seconds and re-run this cell.")

# ── Keep alive ────────────────────────────────────────────────────────────────
print("\nApp is running. This cell will block until you stop it.")
print("Use Kernel → Interrupt to stop the session.")
try:
    streamlit_proc.wait()
except KeyboardInterrupt:
    print("\nStopping...")
    streamlit_proc.terminate()
    tunnel_proc.terminate()
"""

# ══════════════════════════════════════════════════════════════════════════════
# Assemble and write notebook
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Building Kaggle notebook...")

    file_write_source = _make_write_cell()

    cells = [
        md_cell(CELL_MD_HEADER),
        code_cell(CELL_GPU_CHECK),
        code_cell(CELL_SYSTEM_DEPS),
        code_cell(CELL_PYTHON_DEPS_1),
        code_cell(CELL_PYTHON_DEPS_2),
        code_cell(CELL_API_KEYS),
        code_cell(file_write_source),
        code_cell(CELL_PATCH_INDICF5),
        code_cell(CELL_CLOUDFLARED),
        code_cell(CELL_LAUNCH),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indicai_dubbing_kaggle.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nDONE: Notebook written to: {out_path}")
    print(f"   Size: {size_kb:.0f} KB")

    # ── Also generate kaggle_generated_files.py for the CLI deploy approach ──
    _generate_large_files_module()

    print(f"\n== WHAT TO DO NEXT (two options) ==")
    print(f"\nOPTION A — Notebook UI (easier, no CLI):")
    print(f"  1. kaggle.com/code -> New Notebook -> File -> Import Notebook")
    print(f"  2. Upload: {out_path}")
    print(f"  3. Set GPU T4 + Internet On + add Secrets -> Run All")
    print(f"\nOPTION B — kaggle CLI push (ShadowGPU pattern, recommended):")
    print(f"  1. Edit 'kaggle/deploy_dubbing_cloudflare.py':")
    print(f"       Set NTFY_CHANNEL = 'your_unique_channel_name'")
    print(f"  2. Edit 'kernel-metadata.json':")
    print(f"       Replace YOUR_KAGGLE_USERNAME with your actual username")
    print(f"  3. Add Kaggle Secrets: GEMINI_API_KEY + HF_TOKEN")
    print(f"  4. Run: kaggle kernels push -p . --accelerator NvidiaTeslaT4")
    print(f"  5. Watch: curl -s ntfy.sh/your_channel/raw")
    print(f"  6. Kill:  curl -d 'SHUTDOWN_DUBBING' ntfy.sh/your_channel")


def _generate_large_files_module():
    """Generate kaggle_generated_files.py — imported by the Kaggle deploy script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(script_dir)
    WORK_DIR = "/kaggle/working"

    large_files = [
        (os.path.join(base, "app.py"),                               f"{WORK_DIR}/app.py"),
        (os.path.join(base, "pipeline", "phoneme_counter.py"),       f"{WORK_DIR}/pipeline/phoneme_counter.py"),
        (os.path.join(base, "pipeline", "semantic_similarity.py"),   f"{WORK_DIR}/pipeline/semantic_similarity.py"),
        (os.path.join(base, "pipeline", "source_separation.py"),     f"{WORK_DIR}/pipeline/source_separation.py"),
        (os.path.join(base, "pipeline", "voice_manager.py"),         f"{WORK_DIR}/pipeline/voice_manager.py"),
        (os.path.join(base, "pipeline", "translation_cache.py"),     f"{WORK_DIR}/pipeline/translation_cache.py"),
        (os.path.join(base, "pipeline", "isochrony_translation.py"), f"{WORK_DIR}/pipeline/isochrony_translation.py"),
        (os.path.join(base, "utils", "transcription.py"),            f"{WORK_DIR}/utils/transcription.py"),
        (os.path.join(base, "utils", "audio_sync.py"),               f"{WORK_DIR}/utils/audio_sync.py"),
    ]

    # duration_tts.py needs the Linux glob-path adaptation
    duration_tts_path = os.path.join(base, "pipeline", "duration_tts.py")
    if os.path.exists(duration_tts_path):
        with open(duration_tts_path, encoding="utf-8") as f:
            orig = f.read()
        # Apply the same Linux path adaptation as in the notebook builder
        marker_start = '_cache_dir = os.path.expandvars('
        LINUX_BLOCK = '''        import glob as _glob
        _patterns = _glob.glob(os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py"
        ))
        _model_py = _patterns[0] if _patterns else None

        if not _model_py or not os.path.exists(_model_py):
            load_log("HF cache miss - triggering one-time model file download...")
            try:
                from transformers import AutoModel
                AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, token=token)
            except Exception:
                pass
            _patterns = _glob.glob(os.path.expanduser(
                "~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py"
            ))
            _model_py = _patterns[0] if _patterns else None'''
        marker_end = 'pass  # Crash expected; we only needed the cache to populate'
        if marker_start in orig and marker_end in orig:
            idx_s = orig.index(marker_start)
            idx_e = orig.index(marker_end) + len(marker_end)
            line_s = orig.rfind('\n', 0, idx_s) + 1
            adapted = orig[:line_s] + LINUX_BLOCK + orig[idx_e:]
        else:
            adapted = orig
    else:
        adapted = "# duration_tts.py not found\n"

    lines = [
        "# -*- coding: utf-8 -*-",
        "# AUTO-GENERATED by build_notebook.py — do not edit by hand",
        "# This module is imported by deploy_dubbing_cloudflare.py running on Kaggle",
        "import os as _os",
        "",
        "def _write(path, content):",
        "    _os.makedirs(_os.path.dirname(path), exist_ok=True)",
        "    with open(path, 'w', encoding='utf-8') as f:",
        "        f.write(content)",
        "",
    ]

    for src_path, dst_path in large_files:
        if not os.path.exists(src_path):
            print(f"  WARNING: {src_path} not found — skipping")
            continue
        with open(src_path, encoding="utf-8") as f:
            content = f.read()
        lines.append(f"_write({repr(dst_path)}, {repr(content)})")
        print(f"  Embedded: {os.path.basename(src_path)} ({len(content)//1024}KB)")

    # Add the adapted duration_tts
    lines.append(f"_write({repr(WORK_DIR + '/pipeline/duration_tts.py')}, {repr(adapted)})")
    print(f"  Embedded: duration_tts.py (Kaggle-adapted, {len(adapted)//1024}KB)")

    lines.append("")
    lines.append("print('[kaggle_generated_files] All large pipeline files written.')")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_generated_files.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nDONE: Large files module written to: {out_path} ({size_kb:.0f} KB)")
    print(f"  Upload this as a Kaggle Dataset named 'indicai-dubbing-files'")
    print(f"  OR include it in the kernel push directory (place alongside deploy script)")


if __name__ == "__main__":
    main()
