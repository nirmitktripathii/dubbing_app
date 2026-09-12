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

# The single monolithic file-write cell was split into one labeled cell per role
# (Step 6a–6h). Every file cell shares this small logging helper so each write is
# reported with its path + size — a short/failed write is then obvious in the log
# instead of surfacing as a mysterious ImportError mid-dub.

def _make_write_helper():
    """Return the shared `_write_file(path, content)` helper injected at the top of
    every file-writing cell. It logs each write (path + KB) so nothing fails silently."""
    return (
        "import os\n"
        "\n"
        "def _write_file(path, content):\n"
        "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
        "    with open(path, 'w', encoding='utf-8') as _f:\n"
        "        _f.write(content)\n"
        "    _kb = len(content.encode('utf-8')) / 1024\n"
        "    print(f'  ✓ {path}  ({_kb:.1f} KB)')\n"
    )


# Files copied verbatim from the local tree (no Kaggle adaptation needed).
_PIPELINE_FILES = [
    "phoneme_counter.py",
    "semantic_similarity.py",
    "source_separation.py",
    "voice_manager.py",
    "translation_cache.py",
    "isochrony_translation.py",
    # Step-6 process-isolation freeze fix. Cross-platform (they branch on os.name
    # internally), so no Kaggle glob-path adaptation is needed like duration_tts.py.
    "tts_worker.py",
    "tts_supervisor.py",
]
_UTILS_FILES = [
    "audio_extraction.py",
    "transcription.py",
    "audio_sync.py",
]


def _make_mkdirs_cell():
    """Step 6a — create the /kaggle/working package structure + __init__.py files."""
    return (
        "import os\n"
        "\n"
        "print('Step 6a — Creating /kaggle/working package structure...')\n"
        "for d in ['/kaggle/working/pipeline', '/kaggle/working/utils', '/kaggle/working/temp_processing']:\n"
        "    os.makedirs(d, exist_ok=True)\n"
        "    print(f'  ✓ {d}/')\n"
        "\n"
        "for _init in ['/kaggle/working/pipeline/__init__.py', '/kaggle/working/utils/__init__.py']:\n"
        "    open(_init, 'w').close()\n"
        "    print(f'  ✓ {_init}')\n"
        "\n"
        "print('Step 6a complete — package dirs ready.')\n"
    )


def _make_app_cell(base):
    """Step 6b — write the Streamlit UI (app.py)."""
    content = _read(os.path.join(base, "app.py"))
    return (
        _make_write_helper()
        + "\nprint('Step 6b — Writing the Streamlit app (app.py)...')\n"
        + f"_write_file('/kaggle/working/app.py', {repr(content)})\n"
        + "print('Step 6b complete — app.py written.')\n"
    )


def _make_headless_runner_cell(base):
    """Batch mode — write run_headless.py (the no-UI batch driver) to /kaggle/working."""
    content = _read(os.path.join(base, "run_headless.py"))
    return (
        _make_write_helper()
        + "\nprint('Writing headless batch driver (run_headless.py)...')\n"
        + f"_write_file('/kaggle/working/run_headless.py', {repr(content)})\n"
        + "print('run_headless.py written.')\n"
    )


def _make_pipeline_cell(base):
    """Step 6c — write the unmodified pipeline modules."""
    lines = [_make_write_helper(),
             "print('Step 6c — Writing pipeline modules...')"]
    written = 0
    for name in _PIPELINE_FILES:
        src_path = os.path.join(base, "pipeline", name)
        if not os.path.exists(src_path):
            print(f"WARNING: {src_path} not found — skipping")
            continue
        content = _read(src_path)
        lines.append(f"_write_file('/kaggle/working/pipeline/{name}', {repr(content)})")
        written += 1
    lines.append(f"print('Step 6c complete — {written} pipeline modules written.')")
    return "\n".join(lines)


def _make_utils_cell(base):
    """Step 6d — write the unmodified utils modules."""
    lines = [_make_write_helper(),
             "print('Step 6d — Writing utils modules...')"]
    written = 0
    for name in _UTILS_FILES:
        src_path = os.path.join(base, "utils", name)
        if not os.path.exists(src_path):
            print(f"WARNING: {src_path} not found — skipping")
            continue
        content = _read(src_path)
        lines.append(f"_write_file('/kaggle/working/utils/{name}', {repr(content)})")
        written += 1
    lines.append(f"print('Step 6d complete — {written} utils modules written.')")
    return "\n".join(lines)


def _make_video_merge_cell():
    """Step 6e — write the Linux-adapted utils/video_merge.py, then sanity-check it."""
    content = _kaggle_video_merge_content()
    return (
        _make_write_helper()
        + "\nprint('Step 6e — Writing utils/video_merge.py (Linux-adapted)...')\n"
        + f"_write_file('/kaggle/working/utils/video_merge.py', {repr(content)})\n"
        + "_src = open('/kaggle/working/utils/video_merge.py', encoding='utf-8').read()\n"
        + "assert 'def merge_video_audio_subs' in _src and 'log_fn' in _src, \\\n"
        + "    'video_merge.py is missing merge_video_audio_subs(log_fn=...)'\n"
        + "print('  sanity ✓ merge_video_audio_subs(log_fn=...) present')\n"
        + "print('Step 6e complete.')\n"
    )


def _make_duration_tts_cell(base):
    """Step 6f — write the Kaggle-adapted pipeline/duration_tts.py, then sanity-check it."""
    content = _kaggle_duration_tts_content(base)
    return (
        _make_write_helper()
        + "\nprint('Step 6f — Writing pipeline/duration_tts.py (Kaggle-adapted)...')\n"
        + f"_write_file('/kaggle/working/pipeline/duration_tts.py', {repr(content)})\n"
        + "_src = open('/kaggle/working/pipeline/duration_tts.py', encoding='utf-8').read()\n"
        + "assert 'transformers_modules/ai4bharat/IndicF5' in _src \\\n"
        + "    and '_cache_dir = os.path.expandvars(' not in _src, \\\n"
        + "    'duration_tts.py HF cache path was not adapted to the Linux glob'\n"
        + "print('  sanity ✓ HF cache path uses the Linux glob (no Windows expandvars)')\n"
        + "print('Step 6f complete.')\n"
    )


def _make_model_patch_cell():
    """Step 6g — write pipeline/indicf5_model_patched.py, then sanity-check it."""
    content = _kaggle_model_patch_content()
    return (
        _make_write_helper()
        + "\nprint('Step 6g — Writing pipeline/indicf5_model_patched.py (Kaggle model.py)...')\n"
        + f"_write_file('/kaggle/working/pipeline/indicf5_model_patched.py', {repr(content)})\n"
        + "_src = open('/kaggle/working/pipeline/indicf5_model_patched.py', encoding='utf-8').read()\n"
        + "assert 'class INF5Model' in _src and 'load_vocoder' in _src, \\\n"
        + "    'model patch is missing INF5Model / load_vocoder'\n"
        + "print('  sanity ✓ INF5Model + CPU-first vocoder load present')\n"
        + "print('Step 6g complete — applied to the HF cache in Step 7.')\n"
    )


def _kaggle_video_merge_content():
    """Return the raw source of the Linux-adapted utils/video_merge.py."""
    content = r'''import os
import subprocess

def merge_video_audio_subs(video_path: str, audio_path: str, srt_path: str, output_path: str, log_fn=None):
    """
    Merges the original video, the new dubbed audio, and the subtitle file using FFmpeg.
    Linux-compatible (no Windows path escaping needed).
    """
    def _emit(msg):
        print(msg)
        if log_fn:
            try:
                log_fn(f"    {msg}")
            except Exception:
                pass

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
    _emit("Encoding final video with FFmpeg (burning subtitles, this can take a minute)...")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        _emit("Video merging complete.")
    except subprocess.CalledProcessError as e:
        _emit(f"FFmpeg failed:\n{e.stderr[-1500:]}")
        raise RuntimeError(f"FFmpeg failed:\n{e.stderr}")
    return output_path
'''
    return content


def _kaggle_duration_tts_content(base):
    """Return the raw source of the Kaggle-adapted pipeline/duration_tts.py.

    The local source may already be Linux-adapted (the on-disk file uses the glob
    cache path), so this validates the RESULT positively rather than assuming a
    Windows source to rewrite — it works whether or not the marker block is present."""
    # Read the original
    orig_path = os.path.join(base, "pipeline", "duration_tts.py")
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
    # If the marker block was absent, the source is already Linux-adapted — leave it.

    # Validate the RESULT is Linux-correct regardless of which branch ran above.
    # (Discriminate on the Windows-only expandvars call, NOT on the string
    # "%USERPROFILE%" — that also appears in an explanatory comment on Linux.)
    assert "transformers_modules/ai4bharat/IndicF5" in adapted, (
        "duration_tts.py adaptation failed: Linux glob cache path missing"
    )
    assert "_cache_dir = os.path.expandvars(" not in adapted, (
        "duration_tts.py adaptation failed: Windows expandvars cache block still present"
    )

    return adapted


def _kaggle_model_patch_content():
    """Return the raw source of the Kaggle model.py (pipeline/indicf5_model_patched.py)."""
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
    return content


# ──────────────────────────────────────────────────────────────────────────────
CELL_VERIFY_IMPORTS = r"""import subprocess, sys

print('Step 6h — Verifying every written module imports cleanly...')
print('(runs in a throwaway subprocess so a bad file cannot corrupt this kernel)')

# Import in a child process with cwd=/kaggle/working so the package layout resolves.
# EXCLUDES app.py (Streamlit entry point) and indicf5_model_patched.py (applied to
# the HF cache in Step 7, not imported here). The listed modules all load lazily —
# importing them triggers NO model downloads.
_probe = r'''
import importlib, sys
mods = [
    "pipeline.phoneme_counter",
    "pipeline.semantic_similarity",
    "pipeline.source_separation",
    "pipeline.voice_manager",
    "pipeline.translation_cache",
    "pipeline.isochrony_translation",
    "pipeline.duration_tts",
    "pipeline.tts_worker",
    "pipeline.tts_supervisor",
    "utils.audio_extraction",
    "utils.transcription",
    "utils.audio_sync",
    "utils.video_merge",
]
ok = 0
for m in mods:
    try:
        importlib.import_module(m)
        print("  ✓ " + m)
        ok += 1
    except Exception as e:
        print("  ✗ " + m + ": " + type(e).__name__ + ": " + str(e))
sys.exit(0 if ok == len(mods) else 1)
'''

r = subprocess.run([sys.executable, "-c", _probe], cwd="/kaggle/working",
                   capture_output=True, text=True)
print(r.stdout, end="")
if r.stderr.strip():
    print(r.stderr[-2000:])
if r.returncode != 0:
    raise RuntimeError("Step 6h FAILED — a module did not import. Fix it before launching (Step 9) so you don't burn GPU quota on a broken build.")
print("Step 6h complete — all 11 modules import cleanly.")
"""


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

# Trigger transformers_modules caching (creates the model.py in HF cache).
# Run in a FRESH SUBPROCESS: importing IndicF5's remote code pulls in numpy-backed
# libs, and THIS kernel still holds Kaggle's preloaded numpy 2.x (kaggle-environments
# needs >=2) while Step 4 pinned numpy 1.26.4 ON DISK — so an in-kernel AutoConfig hits
# "numpy.dtype size changed (Expected 96 ... got 88)". A subprocess imports the on-disk
# numpy 1.26.4 cleanly and caches model.py reliably (Step 3 below then overwrites it with
# the patched version anyway, so we only need the file to exist).
print("\nStep 2: Caching transformers_modules (model.py + config.py)...")
_cache_code = (
    "import os\n"
    "from transformers import AutoConfig\n"
    "AutoConfig.from_pretrained('ai4bharat/IndicF5', trust_remote_code=True,\n"
    "                           token=os.environ.get('HF_TOKEN'))\n"
    "print('cached-ok')\n"
)
_r = subprocess.run([sys.executable, "-c", _cache_code],
                    capture_output=True, text=True, env=os.environ)
if _r.returncode == 0:
    print("  ✓ transformers_modules cached (fresh subprocess, coherent numpy)")
else:
    _tail = _r.stderr.strip().splitlines()[-1] if _r.stderr.strip() else "(no stderr)"
    print(f"  ⚠ Config cache subprocess failed: {_tail}")

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

# ═════════════════════════════════════════════════════════════════════════════
# ⚙️  SPEED vs QUALITY — Step 6 (TTS) diffusion steps
# ─────────────────────────────────────────────────────────────────────────────
#   32  →  quality baseline (default, slower)
#   16  →  ~2x faster synthesis, slight quality cost
#
# Set the value you want, then run this cell to launch the app.
# Only 32 or 16 are supported; any other value falls back to 32.
NFE_STEP = 32          # ← change to 16 for the fast run
# ═════════════════════════════════════════════════════════════════════════════

if NFE_STEP not in (16, 32):
    print(f"⚠ NFE_STEP={NFE_STEP!r} is not a supported option (32 or 16). Falling back to 32.")
    NFE_STEP = 32
print(
    f"⚙  Step 6 diffusion steps: NFE_STEP={NFE_STEP}"
    + (" — FAST mode (~2x quicker, slight quality cost)" if NFE_STEP == 16
       else " — quality baseline")
)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

if not GEMINI_KEY:
    print("⚠ WARNING: GEMINI_API_KEY is not set. Translation will fail.")
if not HF_TOKEN:
    print("⚠ WARNING: HF_TOKEN is not set. Model download will fail.")

env = os.environ.copy()
# UTF-8 everywhere in the app subprocess. Kaggle spawns this notebook kernel
# under a C/ASCII locale, so WITHOUT this any implicit str->bytes encode inside
# the pipeline (a library doing s.encode() that falls back to
# locale.getpreferredencoding(), a text-mode open(), httpx) defaults to ASCII and
# dies on the first non-ASCII char. The Indic pipeline is saturated with
# non-ASCII (Devanagari/Tamil output, plus — → Δ ✓ in prompts and logs), so this
# surfaces as e.g. "'ascii' codec can't encode character '✓'" and every
# translation call fails in 0.00s.
#   • PYTHONIOENCODING only fixes stdout/stderr/stdin.
#   • PYTHONUTF8=1 forces CPython UTF-8 Mode: it ALSO flips the default open()
#     encoding AND locale.getpreferredencoding() to UTF-8 — this is the switch
#     that actually stops the crash.
#   • LANG/LC_ALL back it up for any C-extension that reads the locale directly.
env.update({
    "GEMINI_API_KEY": GEMINI_KEY,
    "HF_TOKEN": HF_TOKEN,
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    # Step 6 diffusion-step count, read by pipeline/duration_tts.py. Chosen above.
    "DUBBING_NFE_STEP": str(NFE_STEP),
})

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

# ── Batch (headless) mode cells — replace the Streamlit+tunnel launch (Steps 8-9) ──
CELL_HEADLESS_CONFIG = r"""import os

# ── Batch-run configuration ────────────────────────────────────────────────────
# Edit to control the run. Leave DUBBING_INPUT_VIDEO empty to auto-detect the first
# video under /kaggle/input (attach your source video as a Kaggle Dataset).
os.environ.setdefault("DUBBING_TARGET_LANG", "Hindi")
os.environ.setdefault("DUBBING_WHISPER_MODEL", "medium")
os.environ.setdefault("DUBBING_USE_DEMUCS", "1")
os.environ.setdefault("DUBBING_BG_VOLUME", "0.3")
os.environ.setdefault("DUBBING_OUTPUT_DIR", "/kaggle/working/dubbing_output")
# os.environ["DUBBING_INPUT_VIDEO"] = "/kaggle/input/<your-dataset>/<video>.mp4"

# ── RUN SELECTOR — pick ONE. This is the ONLY line you change between runs. ──────
# The residual "~1-2s garbled audio at the start of every segment" is IndicF5 onset
# instability from CROSS-LINGUAL cloning (English reference -> Hindi target). Modes:
#   "premium" — clone the English speaker (original behaviour; THIS is what babbles).
#   "basic"   — RUN 1 control: native HINDI reference, English voice NOT cloned. If the
#               babble vanishes, the cross-lingual cause is proven. (Generic Hindi voice.)
#   "primer"  — RUN 2 fix: clone the English speaker BUT prepend a throwaway Hindi primer
#               that absorbs the onset babble, then slice it off. Keeps the speaker's voice.
RUN = "basic"     # <── change to "basic" / "primer" / "premium"

# Explicit assignment (NOT setdefault) so the value ALWAYS takes, even if the config cell
# was already run once this kernel session. setdefault silently no-ops on a re-run and is
# how a "basic" edit can still execute as premium.
_MODES = {"premium": ("1", "0"), "basic": ("0", "0"), "primer": ("1", "1")}
if RUN not in _MODES:
    raise SystemExit(f"RUN={RUN!r} invalid — pick one of {list(_MODES)}")
os.environ["DUBBING_VOICE_CLONE"], os.environ["DUBBING_TTS_PRIMER"] = _MODES[RUN]
print("=" * 64)
print(f"  RUN MODE = {RUN.upper()}   "
      f"(DUBBING_VOICE_CLONE={os.environ['DUBBING_VOICE_CLONE']}, "
      f"DUBBING_TTS_PRIMER={os.environ['DUBBING_TTS_PRIMER']})")
if RUN == "basic":
    print("  -> native HINDI reference; English voice NOT cloned.")
    print("  -> CONFIRM in the log within ~1 min: 'voice cloning DISABLED' then")
    print("     'Mode: Basic' then 'Default reference voice resolved to: ...HIN_M_HAPPY...'")
    print("  -> If you instead see 'Extracting reference voice clip' / 'Mode: Premium',")
    print("     STOP — it is cloning English again.")
elif RUN == "primer":
    print("  -> clone English speaker + Hindi primer absorbs the onset babble.")
    print("  -> CONFIRM in the log: 'Mode: Premium' AND '[primer] ... sliced primer'.")
else:
    print("  -> clones the English speaker; this is the babbling baseline.")
    print("  -> CONFIRM in the log: 'Mode: Premium'.")
print("=" * 64)
# Optional primer tuning (defaults are fine for the first run):
# os.environ["DUBBING_TTS_PRIMER_TEXT"]   = "नमस्ते।"   # throwaway Hindi utterance (ends in danda)
# os.environ["DUBBING_TTS_PRIMER_BUDGET"] = "1.0"      # seconds of fix_duration given to the primer

# ── Faster CPU run (optional) ───────────────────────────────────────────────────
# For the Basic-mode CPU run these cut wall-time a lot; leave them off for the GPU run.
# os.environ["DUBBING_NFE_STEP"]      = "16"     # ~2x faster TTS, slight quality cost (default 32)
# os.environ["DUBBING_USE_DEMUCS"]    = "0"      # skip source separation (not needed to judge the voice)
# os.environ["DUBBING_WHISPER_MODEL"] = "small"  # faster ASR; segmentation quality barely matters here

# ── GPU-aware Step-6 timeouts ───────────────────────────────────────────────────
# IndicF5 on CPU is ~10-50x slower per segment than on a T4, so the GPU-tuned Step-6
# watchdogs would fire on healthy-but-slow CPU segments — turning a good segment into
# silence (inner DUBBING_SEGMENT_TIMEOUT, default 180s) or SIGKILL-thrash (supervisor
# DUBBING_TTS_SEG_STALL, default 240s). Raise them ONLY when no usable GPU is present;
# on GPU the fast defaults stay intact so a genuinely-wedged CUDA op is still caught fast.
import shutil as _shutil, subprocess as _subprocess
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
_gpu_hidden = _cvd is not None and _cvd.strip() == ""   # torch sees no GPU when this is ""
_has_gpu = False
if not _gpu_hidden and _shutil.which("nvidia-smi"):
    try:
        _has_gpu = _subprocess.run(
            ["nvidia-smi"], stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL, timeout=10
        ).returncode == 0
    except Exception:
        _has_gpu = False
if _has_gpu:
    print("GPU detected -> keeping fast Step-6 timeouts (segment 180s / seg_stall 240s / load 900s).")
else:
    # Keep inner < supervisor so the graceful per-segment timeout wins before the
    # supervisor SIGKILLs the worker.
    os.environ.setdefault("DUBBING_SEGMENT_TIMEOUT", "900")    # inner per-segment ceiling (default 180)
    os.environ.setdefault("DUBBING_TTS_SEG_STALL",  "1200")    # supervisor no-progress kill (default 240)
    os.environ.setdefault("DUBBING_TTS_LOAD_STALL", "1800")    # CPU model load is slower  (default 900)
    print("No GPU detected -> CPU-safe Step-6 timeouts (segment 900s / seg_stall 1200s / load 1800s).")

# Locate the input video now so a missing dataset fails HERE (fast), not mid-run.
_iv = os.environ.get("DUBBING_INPUT_VIDEO", "").strip()
if not _iv:
    _cands = []
    for _dp, _dd, _ff in os.walk("/kaggle/input"):
        for _f in _ff:
            if _f.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")):
                _cands.append(os.path.join(_dp, _f))
    _cands.sort()
    if _cands:
        print(f"Auto-detected input video: {_cands[0]}")
        if len(_cands) > 1:
            print(f"  ({len(_cands)} videos found; using the first. Set DUBBING_INPUT_VIDEO to choose.)")
    else:
        print("No video under /kaggle/input. Attach a dataset with your source video,")
        print("   or set os.environ['DUBBING_INPUT_VIDEO'] above before running the next cell.")
else:
    print(f"Input video (from DUBBING_INPUT_VIDEO): {_iv}")

for _k in ("DUBBING_TARGET_LANG", "DUBBING_WHISPER_MODEL", "DUBBING_USE_DEMUCS",
           "DUBBING_BG_VOLUME", "DUBBING_OUTPUT_DIR", "DUBBING_VOICE_CLONE", "DUBBING_TTS_PRIMER"):
    print(f"  {_k} = {os.environ.get(_k)}")
"""

CELL_HEADLESS_RUN = r"""import os, sys, subprocess

# Run the batch driver in a FRESH SUBPROCESS — NOT in this kernel. Kaggle's kernel loads
# the base image's numpy 2.x at startup (kaggle-environments requires numpy>=2), while
# Step 4 pins numpy 1.26.4 ON DISK. In-kernel, a lazily-imported numpy-1.x-built extension
# (e.g. numba, dragged in by the openai-whisper fallback) then crashes with
# "numpy.dtype size changed (Expected 96 ... got 88)" — a 1.x-built .so meeting the 2.x
# core still live in memory. A subprocess imports the on-disk numpy 1.26.4 fresh, so the
# whole environment is coherent — the SAME reason the Streamlit app runs the pipeline in a
# subprocess. Step 6 (IndicF5) still runs in ITS own supervised subprocess (the freeze fix)
# nested under this one.
os.chdir("/kaggle/working")

# Carry Step 5's API keys + Step 9's DUBBING_* config to the child, and force UTF-8 so the
# non-ASCII Indic pipeline (Devanagari/Tamil output, ✓/→/Δ in logs) doesn't die on the C/
# ASCII locale Kaggle spawns kernels under — the app subprocess sets exactly these.
_env = dict(os.environ)
_env.update({
    "PYTHONUNBUFFERED": "1",   # stream the child's stdout/stderr live into this cell
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
})

_proc = subprocess.run([sys.executable, "-u", "run_headless.py"],
                       cwd="/kaggle/working", env=_env)
rc = _proc.returncode

print(f"\n=== HEADLESS RUN EXIT CODE: {rc} ===")
_out = os.environ.get("DUBBING_OUTPUT_DIR", "/kaggle/working/dubbing_output")
if os.path.isdir(_out):
    print("Outputs (persisted by the commit) under:", _out)
    for _f in sorted(os.listdir(_out)):
        _p = os.path.join(_out, _f)
        if os.path.isfile(_p):
            print(f"  {_f}  ({os.path.getsize(_p)/1024:.1f} KB)")
# A non-zero exit raises so a "Save & Run All" commit is marked FAILED (not silently green).
if rc != 0:
    raise SystemExit(f"Headless run failed with exit code {rc} — see the log above.")
"""


def main(mode="app"):
    print(f"Building Kaggle notebook (mode={mode})...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(script_dir)  # repo root (E:/Dubbing app/pipeline_v3)

    def step_md(title, body=""):
        """A markdown header cell placed before a code cell, so the notebook reads
        as clean numbered steps in the Kaggle UI."""
        text = "## " + title
        if body:
            text += "\n\n" + body
        return md_cell(text)

    cells = [
        # Title / overview
        md_cell(CELL_MD_HEADER),

        # Step 1 — GPU check
        step_md("Step 1 — Confirm the GPU is attached",
                "Verifies a T4 is visible. If this prints *no GPU*, set "
                "**Settings → Accelerator → GPU T4 x2** and re-run."),
        code_cell(CELL_GPU_CHECK),

        # Step 2 — system packages
        step_md("Step 2 — Install system packages (ffmpeg, etc.)"),
        code_cell(CELL_SYSTEM_DEPS),

        # Step 3 — python deps (part 1)
        step_md("Step 3 — Install Python dependencies (core)"),
        code_cell(CELL_PYTHON_DEPS_1),

        # Step 4 — python deps (part 2)
        step_md("Step 4 — Install Python dependencies (TTS stack) + pin ABIs",
                "Installs f5-tts / IndicF5 / vocos, then re-pins numpy/scipy and "
                "`transformers<5` so the ABI stays consistent."),
        code_cell(CELL_PYTHON_DEPS_2),

        # Step 5 — API keys
        step_md("Step 5 — Load API keys from Kaggle Secrets",
                "Reads `GEMINI_API_KEY` and `HF_TOKEN`. Secrets are **per-notebook** — "
                "a freshly-imported copy carries none, so add them under "
                "**Add-ons → Secrets** if this step reports them missing."),
        code_cell(CELL_API_KEYS),

        # Step 6 — write the application into /kaggle/working (split into labeled sub-steps)
        step_md("Step 6 — Write the dubbing application to `/kaggle/working`",
                "Each sub-step writes one role of files and logs every write with its "
                "size, so a short or failed write is obvious here — long before Step 9 "
                "burns GPU time. Step 6h then import-checks the whole build."),
        step_md("Step 6a — Create the package directory structure"),
        code_cell(_make_mkdirs_cell()),
        step_md("Step 6b — Write the Streamlit app (`app.py`)"),
        code_cell(_make_app_cell(base)),
        step_md("Step 6c — Write the pipeline modules"),
        code_cell(_make_pipeline_cell(base)),
        step_md("Step 6d — Write the utils modules"),
        code_cell(_make_utils_cell(base)),
        step_md("Step 6e — Write `utils/video_merge.py` (Linux-adapted)"),
        code_cell(_make_video_merge_cell()),
        step_md("Step 6f — Write `pipeline/duration_tts.py` (Kaggle-adapted)"),
        code_cell(_make_duration_tts_cell(base)),
        step_md("Step 6g — Write `pipeline/indicf5_model_patched.py` (Kaggle model.py)"),
        code_cell(_make_model_patch_cell()),
        step_md("Step 6h — Verify every module imports cleanly",
                "A throwaway subprocess imports all 13 modules. If any fails, this cell "
                "**stops the run** so you fix it before spending GPU quota."),
        code_cell(CELL_VERIFY_IMPORTS),

        # Step 7 — patch IndicF5 in the HF cache
        step_md("Step 7 — Cache IndicF5 and apply the patched model.py"),
        code_cell(CELL_PATCH_INDICF5),
    ]

    if mode == "batch":
        # Headless batch: write the no-UI driver, configure the run, execute it. No
        # Streamlit, no Cloudflare tunnel — outputs land in /kaggle/working and a
        # "Save & Run All" commit persists them. This is the validation vehicle.
        cells += [
            step_md("Step 8 — Write the headless batch driver (`run_headless.py`)"),
            code_cell(_make_headless_runner_cell(base)),
            step_md("Step 9 — Configure the batch run",
                    "Sets target language / Whisper model / Demucs and locates the input "
                    "video under `/kaggle/input`. Edit the `os.environ` lines to taste."),
            code_cell(CELL_HEADLESS_CONFIG),
            step_md("Step 10 — Run the pipeline end-to-end (headless)",
                    "Runs Steps 1-7 in this kernel; Step 6 (IndicF5) runs in a supervised "
                    "subprocess that is killed + relaunched if it wedges. A non-zero exit "
                    "fails the commit, so a broken run is never a silent green."),
            code_cell(CELL_HEADLESS_RUN),
        ]
    else:
        # Interactive app: Streamlit + Cloudflare tunnel (the original live-UI flow).
        cells += [
            step_md("Step 8 — Install the Cloudflare tunnel binary"),
            code_cell(CELL_CLOUDFLARED),
            step_md("Step 9 — Launch the app and open the public URL",
                    "Starts Streamlit + the Cloudflare tunnel and prints a "
                    "`*.trycloudflare.com` link. Keep this cell running to keep the app alive."),
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

    nb_name = "indicai_dubbing_kaggle_batch.ipynb" if mode == "batch" else "indicai_dubbing_kaggle.ipynb"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), nb_name)
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
        (os.path.join(base, "run_headless.py"),                       f"{WORK_DIR}/run_headless.py"),
        (os.path.join(base, "pipeline", "phoneme_counter.py"),       f"{WORK_DIR}/pipeline/phoneme_counter.py"),
        (os.path.join(base, "pipeline", "semantic_similarity.py"),   f"{WORK_DIR}/pipeline/semantic_similarity.py"),
        (os.path.join(base, "pipeline", "source_separation.py"),     f"{WORK_DIR}/pipeline/source_separation.py"),
        (os.path.join(base, "pipeline", "voice_manager.py"),         f"{WORK_DIR}/pipeline/voice_manager.py"),
        (os.path.join(base, "pipeline", "translation_cache.py"),     f"{WORK_DIR}/pipeline/translation_cache.py"),
        (os.path.join(base, "pipeline", "isochrony_translation.py"), f"{WORK_DIR}/pipeline/isochrony_translation.py"),
        # Step-6 process-isolation freeze fix (cross-platform, no adaptation needed).
        (os.path.join(base, "pipeline", "tts_worker.py"),            f"{WORK_DIR}/pipeline/tts_worker.py"),
        (os.path.join(base, "pipeline", "tts_supervisor.py"),        f"{WORK_DIR}/pipeline/tts_supervisor.py"),
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
    import sys as _sys
    _mode = "app"
    if "--batch" in _sys.argv[1:] or os.environ.get("DUBBING_BUILD_MODE", "").lower() == "batch":
        _mode = "batch"
    main(_mode)
