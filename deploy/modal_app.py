#!/usr/bin/env python3
"""Modal deployment for the Indic dubbing pipeline — serverless GPU + async API.

What this is
------------
Wraps the PROVEN `run_headless` flow (all seven stages + the supervisor freeze-fix + the
Step-6.5 voice-conversion cloning path) in a Modal GPU function, fronted by an async FastAPI
gateway. Scale-to-zero between jobs; a warm pool keeps one GPU container hot so the model
isn't cold-loaded on the hot path.

Status: SCAFFOLD. It expresses the architecture with correct Modal idioms and reuses the
pipeline as the single source of truth, but it has NOT been `modal deploy`-run from here
(no Modal creds in the build env). Every place that needs your account/secret/decision is
marked `# SETUP:`. Validate with `modal serve deploy/modal_app.py` then `modal deploy ...`.

Latency note
------------
This is an ASYNC job API (submit -> poll/webhook), not real-time — the pipeline is a
multi-stage GPU job (~8 min today for 80 s of video, TTS-bound). The first optimization
lever after this scaffold is fanning TTS out across GPU replicas (see FAN-OUT NOTE below);
that is what turns the ~235 s serial TTS into ~30 s.

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

# ── Image: system ffmpeg + espeak-ng, pinned python deps, model weights on a Volume ───────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "espeak-ng", "git")
    .pip_install_from_requirements("requirements.txt")
    # API/UI deps + a PINNED fastapi/pydantic combo (see deploy/requirements-deploy.txt):
    # fastapi<0.129 + pydantic 2.12 breaks multipart UploadFile parsing.
    .pip_install_from_requirements("deploy/requirements-deploy.txt")
    # torch/torchaudio are platform-specific and not pinned in requirements; install the CUDA
    # build explicitly for Modal's GPUs. knn-vc (Step 6.5) pulls WavLM+HiFiGAN from torch.hub.
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", index_url="https://download.pytorch.org/whl/cu121")
    .add_local_dir(".", REPO_MOUNT, copy=True,
                   ignore=["dubbing_output*", "*.zip", "*.mp4", "*.wav", ".git", "graphify-out",
                           "**/__pycache__", ".claude"])
)

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
    # Point every HF/torch cache at the Volume so weights persist across cold starts.
    return {
        "HF_HOME": CACHE_DIR,
        "HUGGINGFACE_HUB_CACHE": f"{CACHE_DIR}/hub",
        "TORCH_HOME": f"{CACHE_DIR}/torch",
        "TRANSFORMERS_CACHE": f"{CACHE_DIR}/transformers",
    }


@app.function(
    gpu=GPU_TYPE,
    volumes=VOLUMES,
    secrets=secrets,
    timeout=60 * 30,          # a long video can take a while; async so the caller isn't blocked
    min_containers=int(os.environ.get("MODAL_WARM", "0")),  # SETUP: set 1 to keep a GPU warm (kills cold-start; costs idle GPU)
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0),
)
def dub_video(job_id: str, input_name: str, target_lang: str = "Hindi", mode: str = "basic"):
    """Run the full dubbing pipeline for one job. Reads input + writes output on the jobs Volume.

    `mode`: "basic" (native voice) | "vc" (premium: native TTS + voice-conversion clone) |
            "xlingual" (deprecated cross-lingual — do not use in production).
    """
    import sys, collections, subprocess, traceback

    sys.path.insert(0, REPO_MOUNT)      # mirror the Kaggle notebook's import root
    os.chdir(REPO_MOUNT)
    # One duration implementation for both the gate and the meter (see deploy/api.py).
    from deploy.api import probe_duration

    job_dir = os.path.join(JOBS_DIR, job_id)
    out_dir = os.path.join(job_dir, "dubbing_output")
    in_path = os.path.join(job_dir, input_name)
    os.makedirs(out_dir, exist_ok=True)

    def set_status(**kw):
        cur = job_status.get(job_id, {})
        cur.update(kw)
        job_status[job_id] = cur

    mode_to_clone = {"basic": "0", "xlingual": "1", "vc": "2"}
    env = os.environ.copy()
    env.update(_env_for_hf())
    env.update({
        "DUBBING_INPUT_VIDEO": in_path,
        "DUBBING_OUTPUT_DIR": out_dir,
        "DUBBING_TARGET_LANG": target_lang,
        "DUBBING_VOICE_CLONE": mode_to_clone.get(mode, "0"),
        # GEMINI_API_KEY / HF_TOKEN come from the Modal secret.
        "PYTHONUNBUFFERED": "1",
    })

    set_status(status="running", stage="starting", started_at=time.time())
    t0 = time.time()
    try:
        # Reuse the proven headless driver verbatim (supervisor freeze-fix + VC included).
        # Running it as a subprocess keeps its own process-isolation model intact and makes a
        # hung CUDA op the SUBPROCESS's problem, not the container's.
        #
        # We STREAM its stdout line-by-line rather than capture_output=True. Two reasons:
        # (1) a dub takes minutes, and a customer polling GET /v1/dub/{id} should see which
        #     stage it is in, not an opaque "running" for eight minutes; buffered capture
        #     only yields the log after the process has already exited.
        # (2) the reader loop drains the pipe continuously. run_headless now survives a
        #     stalled console by design, but the cheapest way to keep that true is to never
        #     stop reading — an undrained pipe is what wedged the Kaggle run.
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
            # Throttle Dict writes: update on a stage change, else at most every 5 s.
            now = time.time()
            if stage or (now - last_push) > 5.0:
                last_push = now
                set_status(**({"stage": stage} if stage else {}),
                           log="\n".join(tail)[-4000:], heartbeat=now)
        rc = proc.wait()
        log_tail = "\n".join(tail)[-4000:]

        if rc != 0:
            set_status(status="failed", rc=rc, log=log_tail,
                       error=f"run_headless exited {rc}; see log tail.")
            jobs_vol.commit()
            return {"job_id": job_id, "status": "failed", "rc": rc}

        final = os.path.join(out_dir, f"dubbed_{target_lang.lower()}.mp4")
        if not os.path.exists(final):
            # find any produced mp4 as a fallback
            mp4s = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
            final = os.path.join(out_dir, mp4s[0]) if mp4s else ""

        # Meter on the MEASURED output duration; fall back to the gated input duration only
        # if the probe cannot read the result (never silently meter zero).
        measured = probe_duration(final) if final else None
        billed = measured if measured is not None else job_status.get(job_id, {}).get("input_seconds", 0.0)
        set_status(status="done", stage="complete", output=final,
                   video_seconds=billed, metered_from=("output" if measured is not None else "input"),
                   elapsed_s=round(time.time() - t0, 1), log=log_tail)
        jobs_vol.commit()
        return {"job_id": job_id, "status": "done", "output": final, "video_seconds": billed}
    except Exception as e:
        set_status(status="failed", error=str(e), trace=traceback.format_exc()[-2000:])
        jobs_vol.commit()
        return {"job_id": job_id, "status": "failed", "error": str(e)}


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

# ── FAN-OUT NOTE (next optimization, not in this scaffold) ─────────────────────────────────
# To cut the serial ~235 s TTS: replace run_headless's in-container TTS with a Modal Cls that
# loads IndicF5 in @modal.enter() (warm) and exposes synth_one(segment); the orchestrator then
# calls `TTSEngine().synth_one.map(segments)` so Modal fans the segments across GPU replicas
# (set max_containers, min_containers for the warm pool). knn-vc (Step 6.5) fans out the same
# way. Everything else (translation, demucs, whisper, assembly) stays as-is.

if __name__ == "__main__":
    # Local smoke: `python deploy/modal_app.py` just prints the plan; real runs use modal CLI.
    print(f"Modal app '{APP_NAME}' defined. GPU={GPU_TYPE}. "
          f"Deploy with: modal deploy deploy/modal_app.py")
