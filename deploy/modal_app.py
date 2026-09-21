#!/usr/bin/env python3
"""Modal deployment for the Indic dubbing pipeline — serverless GPU + async API.

What this is
------------
Wraps the PROVEN `run_headless` flow (all seven stages + the supervisor freeze-fix + the
Step-6.5 voice-conversion cloning path) in a Modal GPU function, fronted by an async FastAPI
gateway. Scale-to-zero between jobs; a warm pool keeps one GPU container hot so the model
isn't cold-loaded on the hot path.

Status: DEPLOYED. A basic-mode run has completed end-to-end on Modal. Places that still need
your account/secret/decision are marked `# SETUP:`. Iterate with `modal serve
deploy/modal_app.py`, then `modal deploy deploy/modal_app.py` (fan-out needs a DEPLOYED app —
`Cls.from_name` cannot see an ephemeral `modal serve` session).

Cost/latency architecture
-------------------------
This is an ASYNC job API (submit -> poll/webhook), not real-time. Three levers, all live here:
  1. TTS fan-out — Step 6 shards across TTSEngine GPU replicas (`TTSEngine.synth_shard.map`),
     turning serial per-segment synthesis into parallel work. Below
     DUBBING_TTS_FANOUT_MIN_SEGMENTS the run uses a single shard (one load) instead.
  2. Memory snapshots — TTSEngine bakes its imports + CPU-materialised model into a snapshot,
     so a cold worker RESTORES instead of re-importing/re-materialising (see the class).
  3. Translation off-GPU split — `dub_video` is a CPU orchestrator: a GPU container runs Steps
     1-3, then translation (~60% of wall clock, Gemini/CPU-bound) + assembly run on CPU with no
     GPU billed, and TTS fans out to TTSEngine. Kill-switch: DUBBING_MODAL_SPLIT=0. vc mode
     stays on the single-container GPU path until the vc pipeline is validated end-to-end.

Run:
    pip install modal && modal setup
    modal serve deploy/modal_app.py     # live-reload dev
    modal deploy deploy/modal_app.py    # production
"""
from __future__ import annotations

import os
import re
import time
import uuid

import modal

APP_NAME = "indic-dubbing"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")   # SETUP: L4 fits Whisper+Demucs+IndicF5+knn-vc; A10G for headroom.
# The repo is added to the image at /root/app and put on sys.path exactly like the Kaggle
# notebook does (sys.path.insert(0, ...)), so `from run_headless import main` resolves.
REPO_MOUNT = "/root/app"

# The semantic-gate model (IndicSBERT) ships ONLY pytorch_model.bin, which torch 2.5.1 refuses
# to load (transformers' CVE-2025-32434 guard wants torch>=2.6). prewarm_models converts it to
# safetensors ONCE and writes a clean sentence-transformers dir here on the hf_cache Volume; the
# pipeline reads it via DUBBING_SBERT_MODEL (baked into the image env below). The path MUST sit
# under the CACHE_DIR mount ("/cache/hf") so it lands on the Volume. See _prep_indicsbert().
SBERT_LOCAL_DIR = "/cache/hf/models/indic-sbert-st"

# ── Image: mirrors the PROVEN Kaggle environment ──────────────────────────────────────────
# The Kaggle notebook (kaggle/build_notebook.py) is the reference for what a working IndicF5
# environment needs; this reproduces its install sequence so the image doesn't build clean
# and then die at Step 6. The fragile parts (why each step is here) are called out inline.
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same system packages the Kaggle deps cell installs. rubberband-cli is NOT optional:
    # pyrubberband (drift correction) shells out to the `rubberband` binary. fonts-noto gives
    # libass real glyphs for Indic subtitle burn-in; libsndfile1 backs soundfile.
    .apt_install("ffmpeg", "rubberband-cli", "espeak-ng", "fonts-noto", "libsndfile1", "git")
    # Pinned core: numpy 1.26.4 / scipy 1.13.1 / transformers 4.57.6 — the fragile ABI set.
    .pip_install_from_requirements("requirements.txt")
    # API/UI deps (FastAPI + Streamlit). Version bounds live in deploy/requirements-deploy.txt
    # and are conservative guards only — the /v1/dub OpenAPI fix is structural, in api.py.
    .pip_install_from_requirements("deploy/requirements-deploy.txt")
    # torch/torchaudio are platform-specific and not pinned in requirements; install the CUDA
    # build explicitly for Modal's GPUs. knn-vc (Step 6.5) pulls WavLM+HiFiGAN from torch.hub.
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", index_url="https://download.pytorch.org/whl/cu121")
    # IndicF5 is NOT on PyPI — it installs from source, and f5-tts MUST precede it (IndicF5
    # depends on it). vocos is F5-TTS's vocoder. Mirrors the Kaggle deps cell.
    .pip_install("f5-tts", "vocos", "safetensors")
    .pip_install("git+https://github.com/ai4bharat/IndicF5.git")
    # Re-assert the two pins LAST. f5-tts / IndicF5 can drag transformers to 5.x (the IndicF5
    # meta-tensor crash) and pull a numpy-2.x wheel (dtype-size ABI error at model load).
    # Order matters: a transformers (re)install can itself pull numpy 2.x, so numpy is forced
    # back last, exactly as the Kaggle notebook does it.
    .pip_install("transformers<5.0.0")
    .run_commands("python -m pip install --force-reinstall --no-deps numpy==1.26.4")
    # f5-tts pulls gradio, which floats fastapi/starlette/pydantic forward on every rebuild.
    # That is harmless and NOT pinned here: we never run gradio (the TTS path imports only
    # f5_tts.infer.utils_infer), and api.py parses the /v1/dub multipart form off the raw
    # Request, so FastAPI never synthesizes the Body_* model whose OpenAPI generation used to
    # 500. The resolved web-stack version is therefore not load-bearing.
    .add_local_dir(".", REPO_MOUNT, copy=True,
                   ignore=["dubbing_output*", "*.zip", "*.mp4", "*.mkv", "*.mov", "*.wav",
                           ".git", "graphify-out", "**/__pycache__", ".claude"])
)

# Per-deploy runtime knobs: forward selected DUBBING_* vars from the DEPLOY shell into the
# container env, so `_pipeline_env` (which does os.environ.copy()) hands them to run_headless
# with no code change. Lets one deploy pick the bulk-translation model or Gemini concurrency,
# e.g.  DUBBING_GEMINI_BULK_MODEL=gemini-3.5-flash-lite modal deploy deploy/modal_app.py
# Only vars actually set at deploy time are baked in; nothing is forwarded otherwise, so a
# plain deploy behaves exactly as before.
_FORWARD_ENV = {
    k: os.environ[k]
    for k in ("DUBBING_GEMINI_BULK_MODEL", "DUBBING_GEMINI_CONCURRENCY", "DUBBING_MODAL_SPLIT",
              "DUBBING_TTS_FANOUT", "DUBBING_TTS_FANOUT_MIN_SEGMENTS", "DUBBING_SBERT_MODEL")
    if os.environ.get(k)
}
# The semantic gate (pipeline/semantic_similarity.py) reads DUBBING_SBERT_MODEL. Default it, for
# EVERY function, to the converted-safetensors dir on the Volume that prewarm_models writes — so
# the gate loads under the pinned torch 2.5.1 with no per-deploy flag. A deploy-shell
# DUBBING_SBERT_MODEL still wins: it is applied AFTER via _FORWARD_ENV (later .env override), e.g.
# to point the gate at a different model. Before prewarm has run, this path does not exist yet, so
# the gate degrades exactly as it does today (phoneme-only) — no regression, just a latent default.
image = image.env({"DUBBING_SBERT_MODEL": SBERT_LOCAL_DIR})
if _FORWARD_ENV:
    image = image.env(_FORWARD_ENV)

app = modal.App(APP_NAME, image=image)

# Model weights (HuggingFace cache) live on a Volume so they download ONCE, not per cold start.
hf_cache = modal.Volume.from_name("indic-dubbing-hf-cache", create_if_missing=True)
# Job artifacts (input + output video) — swap for S3/R2 in production (see api.py).
jobs_vol = modal.Volume.from_name("indic-dubbing-jobs", create_if_missing=True)
# Lightweight job status store (job_id -> {status, ...}). Fine for a scaffold; use Postgres/Neon at scale.
job_status = modal.Dict.from_name("indic-dubbing-status", create_if_missing=True)

# SETUP: create these secrets: `modal secret create dubbing-secrets GEMINI_API_KEY=... HF_TOKEN=... RAPIDAPI_PROXY_SECRET=...`
secrets = [modal.Secret.from_name("dubbing-secrets")]

CACHE_DIR = "/cache/hf"
JOBS_DIR = "/jobs"
VOLUMES = {CACHE_DIR: hf_cache, JOBS_DIR: jobs_vol}


def _env_for_hf():
    # duration_tts hard-globs ~/.cache/huggingface/modules/... for the IndicF5 remote-code
    # model.py, and knn-vc (Step 6.5) pulls WavLM/HiFiGAN via torch.hub's ~/.cache/torch —
    # BOTH keyed off $HOME. So point HOME at the persistent Volume: every weight, remote-code
    # module and hub download then lands under /cache/hf, survives cold starts, AND resolves
    # at the exact path the glob expects. This is the layout Kaggle relies on (HOME=/root by
    # default). Setting HF_HOME instead would put transformers_modules under $HF_HOME/modules,
    # which duration_tts's hardcoded ~/.cache glob would never find.
    return {"HOME": CACHE_DIR}


def _prep_indicf5():
    """Make the cached IndicF5 remote-code model.py OUR patched version — at runtime.

    duration_tts._load_indicf5 loads INF5Model *directly* from the transformers_modules
    model.py that HF caches on disk, and the proven Kaggle path OVERWRITES that cached file
    with a patched copy (CPU-first vocoder load, no Windows paths, real DiT weights). We reuse
    the exact same patch text (kaggle/build_notebook.py::_kaggle_model_patch_content, loaded
    by FILE PATH so no pip-installed `kaggle` client can shadow the local module) so there is
    ONE source of truth for the patch.

    Runs at RUNTIME, not image build, because the HF cache lives on a Volume not mounted at
    build time. Idempotent and safe under concurrent TTSEngine containers: a sentinel first
    line skips re-patching and the write is atomic (temp + os.replace).
    """
    import glob, importlib.util, tempfile
    os.environ.update(_env_for_hf())         # HOME -> the Volume, so ~/.cache resolves there
    token = os.environ.get("HF_TOKEN")
    marker = "# [modal-indicf5-patch-applied]\n"

    try:
        hf_cache.reload()                    # pick up a cache another container already wrote
    except Exception:
        pass

    # 1) Cache config/vocab + the remote model.py (NOT the 1.3 GB safetensors — duration_tts
    #    pulls those on first load). Mirrors Steps 1-2 of the Kaggle patch cell.
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("ai4bharat/IndicF5",
                          ignore_patterns=["*.safetensors", "*.bin"], token=token)
    except Exception as e:
        print(f"[indicf5-prep] snapshot_download skipped: {e}", flush=True)
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, token=token)
    except Exception as e:
        print(f"[indicf5-prep] AutoConfig cache-trigger: {e}", flush=True)

    # 2) Overwrite each cached model.py with the patched text, unless already patched. Same
    #    glob duration_tts uses, so we patch exactly the file it will load.
    pattern = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py")
    hits = glob.glob(pattern)
    if not hits:
        print(f"[indicf5-prep] no cached model.py at {pattern}; duration_tts will trigger and "
              "load it UNPATCHED on first use — check HF_TOKEN/gate acceptance", flush=True)
        return

    patched = None
    for mp in hits:
        try:
            if open(mp, encoding="utf-8").read().startswith(marker):
                continue
            if patched is None:
                bn = os.path.join(REPO_MOUNT, "kaggle", "build_notebook.py")
                spec = importlib.util.spec_from_file_location("_kaggle_bn", bn)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                patched = marker + mod._kaggle_model_patch_content()
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(mp), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(patched)
            os.replace(tmp, mp)              # atomic; concurrent containers write identical bytes
            print(f"[indicf5-prep] patched {mp}", flush=True)
        except Exception as e:
            print(f"[indicf5-prep] could not patch {mp}: {e}", flush=True)

    try:
        hf_cache.commit()                    # persist the patched model.py for later cold starts
    except Exception:
        pass


def _prep_indicsbert():
    """Materialise IndicSBERT as a safetensors sentence-transformers dir on the Volume so the
    semantic gate loads under the pinned torch 2.5.1. Returns the local dir.

    WHY: l3cube-pune/indic-sentence-similarity-sbert ships ONLY pytorch_model.bin, and
    transformers 4.57.6 refuses torch.load of a .bin unless torch>=2.6 (CVE-2025-32434). We pin
    torch 2.5.1 for the IndicF5/f5-tts ABI, so the gate never loads and Stage-4 silently drops to
    phoneme-only (confirmed on a real run via semantic_similarity.unavailable_reason()). Bumping
    torch is high ABI risk to IndicF5; instead convert ONCE: torch.load the .bin DIRECTLY (the
    guard is transformers', not torch's) and re-save as safetensors, leaving a clean ST dir with
    ONLY safetensors at SBERT_LOCAL_DIR. SentenceTransformer(dir) then loads via the safetensors
    path and the guard never runs. Idempotent + committed to the Volume, so every later cold start
    just loads it. Raises on a genuine access/convert failure — loud HERE, not silent per-dub.

    ASSERT THE PROPERTY (CLAUDE rule 5): converting is not the same as loading. We do NOT return
    on 'a safetensors file exists'; prewarm_models loads the dir and encodes through the SAME
    semantic_similarity path the gate uses, so a broken conversion fails prewarm, not a dub.
    """
    import shutil
    from pipeline import semantic_similarity
    os.environ.update(_env_for_hf())
    token = os.environ.get("HF_TOKEN")
    src_repo = semantic_similarity.DEFAULT_MODEL      # the canonical source repo id, not the env override
    target = SBERT_LOCAL_DIR
    st_file = os.path.join(target, "model.safetensors")

    try:
        hf_cache.reload()                             # pick up a conversion another container wrote
    except Exception:
        pass
    if os.path.exists(st_file):
        return target                                 # already converted (verified by the caller)

    from huggingface_hub import snapshot_download
    src_dir = snapshot_download(src_repo, token=token)   # real files (symlinks resolved on copy)

    # Copy the whole ST structure (modules.json, 1_Pooling, tokenizer, config) to the Volume, then
    # swap the transformer-root weights .bin -> safetensors so the dir is self-contained & offline.
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copytree(src_dir, target)                  # symlinks=False -> resolves to real files

    import torch
    from safetensors.torch import save_file
    bin_path = os.path.join(target, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        if os.path.exists(st_file):
            return target                             # some ST repos already ship safetensors
        raise FileNotFoundError(
            f"{src_repo}: neither pytorch_model.bin nor model.safetensors at the transformer root")
    # weights_only=True: a pure BERT state_dict is all tensors; the guard we are dodging is
    # transformers', so torch.load direct is allowed. clone().contiguous() breaks any shared
    # storage (tied weights) so save_file won't raise on shared memory.
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    tensors = {k: v.clone().contiguous() for k, v in state.items() if isinstance(v, torch.Tensor)}
    save_file(tensors, st_file, metadata={"format": "pt"})
    os.remove(bin_path)                               # leave NO .bin for transformers to reject
    print(f"[indicsbert-prep] converted {src_repo} .bin -> safetensors at {target} "
          f"({len(tensors)} tensors)", flush=True)
    try:
        hf_cache.commit()
    except Exception:
        pass
    return target


# ── Shared pipeline plumbing (used by the split orchestrator and the un-split GPU path) ─────
def _job_paths(job_id: str, input_name: str):
    job_dir = os.path.join(JOBS_DIR, job_id)
    out_dir = os.path.join(job_dir, "dubbing_output")
    in_path = os.path.join(job_dir, input_name)
    os.makedirs(out_dir, exist_ok=True)
    return job_dir, out_dir, in_path


def _make_set_status(job_id: str):
    def set_status(**kw):
        cur = job_status.get(job_id, {})
        cur.update(kw)
        job_status[job_id] = cur
    return set_status


def _pipeline_env(in_path: str, out_dir: str, target_lang: str, mode: str,
                  stages: str, fanout: bool):
    """Build the run_headless environment for one phase. `stages` is DUBBING_STAGES
    ("all" | "prep" | "resume"); `fanout` selects the Modal TTS backend."""
    mode_to_clone = {"basic": "0", "xlingual": "1", "vc": "2"}
    env = os.environ.copy()
    env.update(_env_for_hf())
    env.update({
        "DUBBING_INPUT_VIDEO": in_path,
        "DUBBING_OUTPUT_DIR": out_dir,
        "DUBBING_TARGET_LANG": target_lang,
        "DUBBING_VOICE_CLONE": mode_to_clone.get(mode, "0"),
        # Persist the Stage-4 translation candidate pool on the HF cache Volume. Its default
        # location is os.getcwd()/.dubbing_cache, which on Modal is the EPHEMERAL image layer —
        # so the pool that is supposed to accumulate across runs (and let a re-run select its
        # lines with ZERO Gemini calls) is silently thrown away on every container. Pointing it
        # at the Volume is what makes it cross-run. See PRODUCTION_OPTIMIZATION_AUDIT.md P10.
        "DUBBING_CACHE_DIR": os.path.join(CACHE_DIR, "dubbing_cache"),
        # Fan Step 6 out across TTSEngine containers instead of synthesizing serially.
        "DUBBING_TTS_BACKEND": ("modal-fanout" if fanout else ""),
        "DUBBING_STAGES": stages,
        # GEMINI_API_KEY / HF_TOKEN come from the Modal secret.
        "PYTHONUNBUFFERED": "1",
    })
    return env


def _run_headless_streamed(env, set_status):
    """Run run_headless.py as a subprocess, STREAMING its stdout into job status. Returns
    (rc, log_tail).

    Reuses the proven headless driver verbatim (supervisor freeze-fix + VC included). Running
    it as a subprocess keeps its own process-isolation model intact and makes a hung CUDA op
    the SUBPROCESS's problem, not the container's. We stream rather than capture_output=True so
    (1) a customer polling GET /v1/dub/{id} sees the live stage, not an opaque "running" for
    minutes, and (2) the reader loop drains the pipe continuously — an undrained pipe is what
    wedged the Kaggle run.
    """
    import sys, subprocess, collections
    proc = subprocess.Popen(
        [sys.executable, "-u", "run_headless.py"],
        cwd=REPO_MOUNT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    tail = collections.deque(maxlen=200)   # ring buffer: keep the tail, not the whole log
    last_push = 0.0
    for line in proc.stdout:
        tail.append(line.rstrip("\n"))
        stage = _parse_stage(line)
        now = time.time()   # throttle Dict writes: on a stage change, else at most every 5 s
        if stage or (now - last_push) > 5.0:
            last_push = now
            set_status(**({"stage": stage} if stage else {}),
                       log="\n".join(tail)[-4000:], heartbeat=now)
    rc = proc.wait()
    return rc, "\n".join(tail)[-4000:]


def _finalize_job(job_id, out_dir, target_lang, rc, log_tail, t0, set_status):
    """Detect the output, meter on its MEASURED duration, and set the terminal status. Shared
    by the split orchestrator and the un-split GPU path so metering is identical either way."""
    import sys
    sys.path.insert(0, REPO_MOUNT)
    from deploy.api import probe_duration   # one duration impl for the gate and the meter
    if rc != 0:
        set_status(status="failed", rc=rc, log=log_tail,
                   error=f"run_headless exited {rc}; see log tail.")
        try:
            jobs_vol.commit()
        except Exception:
            pass
        return {"job_id": job_id, "status": "failed", "rc": rc}

    final = os.path.join(out_dir, f"dubbed_{target_lang.lower()}.mp4")
    if not os.path.exists(final):
        mp4s = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]   # any produced mp4
        final = os.path.join(out_dir, mp4s[0]) if mp4s else ""
    # Meter on the MEASURED output duration; fall back to the gated input duration only if the
    # probe cannot read the result (never silently meter zero).
    measured = probe_duration(final) if final else None
    billed = measured if measured is not None else job_status.get(job_id, {}).get("input_seconds", 0.0)
    set_status(status="done", stage="complete", output=final,
               video_seconds=billed, metered_from=("output" if measured is not None else "input"),
               elapsed_s=round(time.time() - t0, 1), log=log_tail)
    try:
        jobs_vol.commit()
    except Exception:
        pass
    return {"job_id": job_id, "status": "done", "output": final, "video_seconds": billed}


@app.function(volumes=VOLUMES, secrets=secrets, timeout=60 * 20)
def prewarm_models():
    """One-time cache warm: put every model a run pulls onto the hf_cache Volume so NOTHING
    downloads on the request path. Run once after deploy (the models then persist on the Volume
    across future deploys):

        modal run deploy/modal_app.py::prewarm_models

    Why a warm function and not a build step: a Modal Volume is not writable by `pip_install`
    during image build — IndicF5 itself only reaches the Volume via a RUNTIME warm
    (`_prep_indicf5`) + commit, and this mirrors that exactly. Baking the download into the
    image build (`image.run_function(..., volumes=...)`) is possible but would hard-FAIL the
    whole image on any HF hiccup; this keeps the pipeline's graceful degradation and surfaces a
    genuine access problem HERE, loudly, instead of silently degrading every dub. Covers:
      • IndicF5 remote-code model.py + config/vocab (1.3 GB weights still load lazily in TTS).
      • IndicSBERT (l3cube-pune/indic-sentence-similarity-sbert) — the semantic gate. The repo
        ships only pytorch_model.bin, which torch 2.5.1 refuses; _prep_indicsbert converts it to
        safetensors on the Volume so it loads under the pinned torch. Once converted here,
        semantic_similarity.available() is True on every later run (DUBBING_SBERT_MODEL points at
        the converted dir, baked into the image env).
    Idempotent — re-running just re-verifies. Raises if IndicSBERT cannot be fetched, converted,
    or loaded (a broken conversion fails HERE, not silently on every dub)."""
    import sys
    sys.path.insert(0, REPO_MOUNT)
    os.chdir(REPO_MOUNT)
    os.environ.update(_env_for_hf())          # HOME -> the Volume, so the HF cache lands there

    # 1) IndicF5 remote code + config/vocab (same warm the GPU path runs).
    _prep_indicf5()

    # 2) IndicSBERT — convert the .bin weights to safetensors on the Volume (the repo ships no
    #    safetensors, and torch 2.5.1 refuses the .bin), then PROVE it loads + scores through the
    #    SAME semantic_similarity path the gate uses. Converting is not loading (CLAUDE rule 5):
    #    if the conversion is broken this fails HERE, not silently on every dub.
    target = _prep_indicsbert()
    os.environ["DUBBING_SBERT_MODEL"] = target        # so the gate below loads the converted dir
    from pipeline import semantic_similarity
    if not semantic_similarity.available():
        raise RuntimeError(
            f"IndicSBERT still unavailable after convert to {target}: "
            f"{semantic_similarity.unavailable_reason()}")
    sanity = semantic_similarity.score("hello world", "नमस्ते दुनिया")
    if sanity is None:
        raise RuntimeError(f"IndicSBERT loaded from {target} but scored None on the sanity pair")
    print(f"[prewarm] IndicSBERT ready (safetensors) at {target}; sanity score={sanity}", flush=True)

    try:
        hf_cache.commit()                     # persist IndicF5 remote code + converted IndicSBERT
    except Exception as e:
        print(f"[prewarm] hf_cache.commit warning: {e}", flush=True)
    return {"ok": True, "indicsbert": target, "sanity_score": sanity}


@app.function(
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=secrets,
    timeout=60 * 30,
    min_containers=int(os.environ.get("MODAL_WARM", "0")),  # SETUP: 1 keeps a GPU warm for the prep phase
    # SETUP: the GPU is done the instant Steps 1-3 return; without this the container lingers
    # the Modal default (~60 s) billing an idle L4 while the CPU orchestrator does translation.
    # 5 s scales it to zero right after it returns. Raise only if you want warm reuse across
    # back-to-back jobs (prefer min_containers for that — don't pay for idle to buy latency).
    scaledown_window=int(os.environ.get("MODAL_GPU_SCALEDOWN", "5")),
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def gpu_transcribe(job_id: str, input_name: str, target_lang: str = "Hindi", mode: str = "basic"):
    """SPLIT phase 1 (GPU): run Steps 1-3 (extract / Demucs / Whisper), persist the checkpoint
    and RELEASE the GPU. This container's GPU is held only for Demucs+Whisper — not for the
    ~60% of wall clock the CPU orchestrator then spends in Gemini translation. No IndicF5 here,
    so no _prep_indicf5 and no model load."""
    import sys
    sys.path.insert(0, REPO_MOUNT)
    os.chdir(REPO_MOUNT)
    set_status = _make_set_status(job_id)
    _, out_dir, in_path = _job_paths(job_id, input_name)
    env = _pipeline_env(in_path, out_dir, target_lang, mode, stages="prep", fanout=False)
    set_status(status="running", stage="transcribing", started_at=time.time())
    rc, log_tail = _run_headless_streamed(env, set_status)
    try:
        jobs_vol.commit()   # publish pipeline_state.json + separated audio to the orchestrator
        hf_cache.commit()
    except Exception:
        pass
    if rc != 0:
        set_status(status="failed", rc=rc, log=log_tail, error=f"transcribe (prep) exited {rc}")
    return {"rc": rc, "log": log_tail}


@app.function(
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=secrets,
    timeout=60 * 30,
    min_containers=int(os.environ.get("MODAL_WARM", "0")),
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def gpu_full_run(job_id: str, input_name: str, target_lang: str = "Hindi", mode: str = "basic"):
    """The un-split path: run ALL seven stages in ONE GPU container (the proven behaviour).
    Used for `mode="vc"` (Step 6.5 knn-vc needs a co-resident GPU), for the kill-switch
    (DUBBING_MODAL_SPLIT=0), and as the orchestrator's auto-fallback. TTS still fans Step 6 out
    to TTSEngine when DUBBING_TTS_FANOUT != 0."""
    import sys
    sys.path.insert(0, REPO_MOUNT)
    os.chdir(REPO_MOUNT)
    # Patch the cached IndicF5 model.py before anything loads it — needed for the serial TTS
    # path (DUBBING_TTS_FANOUT=0), and it warms + commits the shared cache so TTSEngine
    # containers reload a patched model.py instead of each racing to write it.
    _prep_indicf5()
    set_status = _make_set_status(job_id)
    _, out_dir, in_path = _job_paths(job_id, input_name)
    fanout = os.environ.get("DUBBING_TTS_FANOUT", "1") != "0"
    env = _pipeline_env(in_path, out_dir, target_lang, mode, stages="all", fanout=fanout)
    set_status(status="running", stage="starting", started_at=time.time())
    t0 = time.time()
    rc, log_tail = _run_headless_streamed(env, set_status)
    try:
        hf_cache.commit()   # persist Stage-4 cache adds even if a later stage failed
    except Exception:
        pass
    return _finalize_job(job_id, out_dir, target_lang, rc, log_tail, t0, set_status)


@app.function(
    volumes=VOLUMES,
    secrets=secrets,
    timeout=60 * 40,
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def dub_video(job_id: str, input_name: str, target_lang: str = "Hindi", mode: str = "basic"):
    """Orchestrator (CPU — no GPU billed while it runs).

    For `mode="basic"` it SPLITS the run: a GPU container (`gpu_transcribe`) does Steps 1-3,
    then translation + assembly run HERE on cheap CPU — releasing the GPU for the ~60% of wall
    clock Step 4 spends in Gemini translation — with Step 6 TTS fanned out to TTSEngine GPUs.
    `mode="vc"` and the kill-switch DUBBING_MODAL_SPLIT=0 route to the single-container GPU path
    (`gpu_full_run`), because Step 6.5 knn-vc needs a co-resident GPU. Any orchestration error
    (not a stage failure) falls back to that proven path, so a job degrades rather than fails.

    `mode`: "basic" (native voice) | "vc" (premium: native TTS + voice-conversion clone) |
            "xlingual" (deprecated cross-lingual — do not use in production).
    """
    import sys, traceback
    sys.path.insert(0, REPO_MOUNT)
    os.chdir(REPO_MOUNT)
    set_status = _make_set_status(job_id)
    _, out_dir, in_path = _job_paths(job_id, input_name)

    split_on = os.environ.get("DUBBING_MODAL_SPLIT", "1") != "0"
    if mode == "vc" or not split_on:
        # vc needs a co-resident GPU for Step 6.5; the kill-switch forces the proven path.
        return gpu_full_run.remote(job_id, input_name, target_lang, mode)

    t0 = time.time()
    try:
        set_status(status="running", stage="starting", started_at=t0)
        # Phase 1 on GPU (blocks until Steps 1-3 finish + the checkpoint is committed).
        r = gpu_transcribe.remote(job_id, input_name, target_lang, mode)
        if isinstance(r, dict) and r.get("rc", 1) != 0:
            return {"job_id": job_id, "status": "failed", "rc": r.get("rc")}  # already marked failed
        try:
            jobs_vol.reload()   # pick up pipeline_state.json + separated audio from phase 1
        except Exception:
            pass
        # Phase 2 on CPU: Step 4 (Gemini translation) + Step 5/7 (CPU) here; Step 6 TTS fans
        # out to TTSEngine GPUs. No local GPU held for any of it in basic mode.
        fanout = os.environ.get("DUBBING_TTS_FANOUT", "1") != "0"
        env = _pipeline_env(in_path, out_dir, target_lang, mode, stages="resume", fanout=fanout)
        rc, log_tail = _run_headless_streamed(env, set_status)
        try:
            hf_cache.commit()
        except Exception:
            pass
        return _finalize_job(job_id, out_dir, target_lang, rc, log_tail, t0, set_status)
    except Exception as e:
        # The orchestration itself broke (not a stage failure) — degrade to the proven single-
        # container path rather than losing the job. Idempotent: it re-runs from Step 1.
        set_status(status="running", stage="fallback: single-container GPU run",
                   split_error=str(e), trace=traceback.format_exc()[-1500:])
        return gpu_full_run.remote(job_id, input_name, target_lang, mode)


# run_headless logs stage headings as "Step N/7: ..." (N may be 6.5) and the TTS supervisor
# logs per-segment progress as "[Segment i/n]". Parsed here so job status reflects real
# emitted strings rather than invented ones.
_STEP_RE = re.compile(r"Step\s+(\d+(?:\.\d+)?)/7:\s*(.+?)\s*$")
_SEG_RE = re.compile(r"Segment\s+(\d+)\s*/\s*(\d+)")


def _parse_stage(line: str):
    m = _STEP_RE.search(line)
    if m:
        return f"step {m.group(1)}/7 — {m.group(2)[:80]}"
    m = _SEG_RE.search(line)
    if m:
        return f"step 6/7 — TTS segment {m.group(1)}/{m.group(2)}"
    return None


# ── Step-6 TTS fan-out worker ─────────────────────────────────────────────────────────────
# TTS is the pipeline's dominant cost (~235 s of a ~7.8 min run for an 80 s video: 13
# segments serialized at ~18 s each). Segments are independent, so they parallelize. One
# instance of this class = one container = ONE IndicF5 load serving a whole shard, which is
# why work is mapped over shards rather than over individual segments (a per-segment map
# would pay the model load per segment). See deploy/tts_fanout.py for the sharding model.
TTS_MAX_CONTAINERS = int(os.environ.get("MODAL_TTS_MAX_CONTAINERS", "8"))


@app.cls(
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=secrets,
    timeout=60 * 20,
    # NOTE: these are the STATIC ceiling/floor for any direct call. The fan-out driver
    # (deploy/tts_fanout.py::_modal_shard_runner) OVERRIDES max_containers and scaledown_window
    # PER-RUN via Cls.with_options — pinning containers to exactly the planned shard count for
    # that video — so these values only govern the non-fanned paths.
    max_containers=TTS_MAX_CONTAINERS,
    # SETUP: min_containers=1+ keeps engines warm, which removes the model-load term from
    # the wall-clock floor (the difference between ~60 s and ~25 s for a short video) at the
    # cost of paying for idle GPU. Leave 0 for bursty traffic; raise it for a latency SLA.
    min_containers=int(os.environ.get("MODAL_TTS_WARM", "0")),
    # SETUP: THE dominant cost lever. The fan-out maps once over N shards; they all finish
    # within a second of each other, the map drains, and every shard is then idle. Without this
    # each L4 container lingers the Modal default (~60 s) billing pure idle — ~65% of a short
    # dub's GPU bill. 5 s scales them to zero right after the drain. The result is already muxed
    # by then, so this cannot slow any dub. Deliberately a small CONSTANT, not scaled with video
    # length: the idle bill is workers x window and is length-independent, so inflating it for
    # long videos only wastes money. Snapshot restore (~seconds) makes the next cold start cheap;
    # for warm reuse under a latency SLA use MODAL_TTS_WARM, not a long window here.
    scaledown_window=int(os.environ.get("MODAL_TTS_SCALEDOWN", "5")),
    # Memory snapshot (GA, CPU): bake the heavy imports + the CPU-materialised 1.3 GB model
    # into a snapshot so every future cold start RESTORES it instead of re-importing and
    # re-materialising. The measured cold TTS load is ~28 s (f5_tts import alone ~16 s + weight
    # materialisation); restoring a snapshot skips almost all of it. See the two @modal.enter
    # phases below for why this is a plain CPU snapshot, not the Alpha GPU one.
    enable_memory_snapshot=True,
)
class TTSEngine:
    @modal.enter(snap=True)
    def load_snapshot(self):
        """Runs BEFORE the memory snapshot is captured, with NO GPU attached (the GA CPU-
        snapshot path — we deliberately do not use the Alpha GPU snapshot). Everything that is
        expensive AND device-independent happens here so it lands in the snapshot and is
        skipped on every future cold start:
          - the heavy imports (torch / transformers / f5_tts — f5_tts alone measured ~16 s);
          - patching the cached IndicF5 remote-code model.py;
          - materialising the ~1.3 GB checkpoint into CPU RAM.
        The model is loaded to CPU because no GPU exists during the snap phase; the post-restore
        hook below moves it onto CUDA. generate_tts_for_segments already re-places
        ema_model+vocoder per call, so a CPU-loaded-then-moved model reaches the identical state
        the old cuda-direct load produced.
        """
        import sys
        sys.path.insert(0, REPO_MOUNT)
        os.chdir(REPO_MOUNT)
        # Patch the cached model.py, THEN load — _load_indicf5 reads the file we just wrote.
        _prep_indicf5()
        from pipeline.duration_tts import _load_indicf5
        _load_indicf5("cpu")
        # Warm the synth-time imports into the snapshot too: generate_tts_for_segments imports
        # these lazily on the hot path, so pull them in now while we are building the snapshot.
        try:
            from f5_tts.infer.utils_infer import (  # noqa: F401
                preprocess_ref_audio_text, infer_batch_process)
        except Exception as e:
            print(f"[TTSEngine.snap] f5_tts warm-import skipped: {e}", flush=True)
        try:
            hf_cache.commit()   # persist the 1.3 GB weights this load downloaded on a cold cache
        except Exception:
            pass

    @modal.enter()
    def to_gpu(self):
        """Runs AFTER restore (and after load_snapshot on the very first, pre-snapshot start),
        with the GPU attached. Moves the CPU-resident cached model onto CUDA so synth_shard
        sees a fully-on-device model — the same state the old single-phase cuda load left."""
        from pipeline.duration_tts import _move_indicf5_to
        _move_indicf5_to("cuda")

    @modal.method()
    def synth_shard(self, spec: dict) -> dict:
        """Synthesize this shard's segments and return their WAV bytes + manifest entries.

        Calls the SAME generate_tts_for_segments the serial and Kaggle paths call, with the
        full segment list and only_indices for this shard, so `i` stays the global index and
        every resolution rule (language, nfe, reference voice, signature, drift, degradation)
        is literally the same code. Writes into a PRIVATE directory — workers never share a
        manifest — and the caller merges as the single writer.
        """
        import sys, json, tempfile, traceback
        sys.path.insert(0, REPO_MOUNT)
        from pipeline.duration_tts import generate_tts_for_segments, MANIFEST_NAME

        lines: list[str] = []
        work = tempfile.mkdtemp(prefix="ttsshard_")
        only = set(spec.get("only_indices") or [])

        ref_path = None
        if spec.get("reference_bytes"):
            ref_path = os.path.join(work, spec.get("reference_name") or "reference.wav")
            with open(ref_path, "wb") as fh:
                fh.write(spec["reference_bytes"])

        try:
            generate_tts_for_segments(
                spec["segments"],
                target_language=spec["target_language"],
                output_dir=work,
                reference_audio_path=ref_path,
                reference_text=spec.get("reference_text"),
                device="cuda",
                log_fn=lines.append,
                watchdog_mode="thread",   # no supervisor here; Modal's timeout+retry is the outer bound
                only_indices=only,
            )
        except Exception as e:
            lines.append(f"shard {sorted(only)[:3]}... FAILED: {e}")
            lines.append(traceback.format_exc()[-1500:])

        # Collect only what this shard actually produced.
        entries, wavs = {}, {}
        try:
            with open(os.path.join(work, MANIFEST_NAME), encoding="utf-8") as fh:
                entries = {k: v for k, v in json.load(fh).items() if int(k) in only}
        except Exception:
            pass
        for i in sorted(only):
            p = os.path.join(work, f"segment_{i:04d}.wav")
            if os.path.exists(p) and os.path.getsize(p) > 44:
                with open(p, "rb") as fh:
                    wavs[str(i)] = fh.read()

        return {"entries": entries, "wavs": wavs, "log": lines[-80:]}


# ── FastAPI gateway, served by Modal ──────────────────────────────────────────────────────
@app.function(volumes=VOLUMES, secrets=secrets, min_containers=1)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, REPO_MOUNT)
    # api.py builds the FastAPI app and wires it to dub_video / job_status / jobs_vol.
    from deploy.api import build_api
    return build_api(dub_video=dub_video, job_status=job_status, jobs_vol=jobs_vol, jobs_dir=JOBS_DIR)


# ── Streamlit UI, served by Modal ─────────────────────────────────────────────────────────
# SETUP: run the Streamlit UI either here (modal) or on Streamlit Community Cloud pointing at
# the FastAPI URL. See deploy/streamlit_app.py and deploy/README.md.

# ── REMAINING LEVERS ───────────────────────────────────────────────────────────────────────
# - DONE: the per-segment Gemini calls in Step 4 now run with bounded concurrency
#   (isochrony_translation._run_batches_concurrent, DUBBING_GEMINI_CONCURRENCY, default 4), so a
#   long clip's batches overlap their response waits; _throttle still holds each model's RPM.
# - Split vc mode too: delegate Step 6.5 knn-vc to a GPU function so vc also runs off-GPU for
#   translation. Deferred until the vc pipeline is validated end-to-end (owner's call).

if __name__ == "__main__":
    # Local smoke: `python deploy/modal_app.py` just prints the plan; real runs use modal CLI.
    print(f"Modal app '{APP_NAME}' defined. GPU={GPU_TYPE}. "
          f"Deploy with: modal deploy deploy/modal_app.py")
