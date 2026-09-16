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
        # Fan Step 6 out across TTSEngine containers instead of synthesizing serially.
        # Set DUBBING_TTS_FANOUT=0 to fall back to the in-container supervised path.
        "DUBBING_TTS_BACKEND": ("modal-fanout"
                                if os.environ.get("DUBBING_TTS_FANOUT", "1") != "0" else ""),
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
    max_containers=TTS_MAX_CONTAINERS,
    # SETUP: min_containers=1+ keeps engines warm, which removes the model-load term from
    # the wall-clock floor (the difference between ~60 s and ~25 s for a short video) at the
    # cost of paying for idle GPU. Leave 0 for bursty traffic; raise it for a latency SLA.
    min_containers=int(os.environ.get("MODAL_TTS_WARM", "0")),
)
class TTSEngine:
    @modal.enter()
    def load(self):
        """Load IndicF5 ONCE per container, before any shard runs.

        duration_tts caches the model in module globals, so the generate_tts_for_segments
        call inside synth_shard reuses what we load here instead of loading per shard.
        """
        import sys
        sys.path.insert(0, REPO_MOUNT)
        os.chdir(REPO_MOUNT)
        os.environ.update(_env_for_hf())
        from pipeline.duration_tts import _load_indicf5
        _load_indicf5("cuda")

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
