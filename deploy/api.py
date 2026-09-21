#!/usr/bin/env python3
"""FastAPI gateway for the dubbing service — async job API, RapidAPI-fronted.

Endpoints
---------
POST /v1/dub            multipart: file=<video>, target_lang, mode  -> {job_id, status}
GET  /v1/dub/{job_id}   -> {status, video_seconds, output?, error?}
GET  /v1/dub/{job_id}/download -> the dubbed mp4 (when status=done)
GET  /healthz           -> ok

Design
------
- ASYNC: POST validates + stores the input + `dub_video.spawn(...)` (non-blocking) and returns
  a job_id immediately. The pipeline is a multi-minute GPU job; clients poll GET or use a
  webhook. This is the honest shape — there is no synchronous "instant" dub.
- AUTH: RapidAPI proxies every call and injects `X-RapidAPI-Proxy-Secret`. We verify it so the
  endpoint can't be hit around RapidAPI. RapidAPI itself issues keys, enforces quotas, and
  bills — we don't build billing. `X-RapidAPI-User` identifies the caller for metering/logs.
- LIMITS: reject oversized / too-long inputs BEFORE any GPU spins up (the API twin of the
  CPU-preflight rule). Metering is on ACTUAL processed duration (ffprobe), set by the worker.
- IDEMPOTENCY: an `Idempotency-Key` header dedupes retries so a client re-send doesn't
  double-run / double-bill.

`build_api(...)` is called from modal_app.fastapi_app with the Modal handles injected, so this
module has no hard Modal dependency and can be unit-tested with fakes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid

# Imported at MODULE level on purpose: this file uses `from __future__ import annotations`
# (PEP 563), so endpoint annotations like `request: Request` are stored as strings and FastAPI
# resolves them against this module's globals at OpenAPI-generation time. If these names were
# imported inside build_api() instead, they would not be in module globals and generation would
# fail with "... is not fully defined" (PydanticUserError) on GET /openapi.json.
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

MAX_UPLOAD_MB = int(os.environ.get("DUB_MAX_UPLOAD_MB", "500"))
VALID_MODES = {"basic", "vc", "xlingual"}
# Per-plan input ceilings (seconds of video). RapidAPI enforces call quotas; this guards GPU.
PLAN_MAX_SECONDS = {"BASIC": 600, "PRO": 3600, "ULTRA": 7200, "": 120}  # "" = unauth/free probe


def probe_duration(path: str) -> float | None:
    """Media duration in seconds via ffprobe, or None if it could not be determined.

    None means UNKNOWN, never "zero" — callers must treat it as a failed check, not a
    passed one (CLAUDE rule 1: a check that cannot run is not a check that passed). The
    worker uses the same function on the OUTPUT for metering, so the number a customer is
    billed on and the number they were gated on come from one implementation.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            return None
        return round(float(out.stdout.strip()), 2)
    except Exception:
        # Missing ffprobe, unparseable output, not a media file, timeout -> UNKNOWN.
        return None


class ApiError(Exception):
    """Framework-independent error so the job-prep logic can be unit-tested without FastAPI."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def prepare_job(*, job_status, jobs_vol, jobs_dir, dub_video, data: bytes, filename: str,
                target_lang: str, mode: str, plan: str = "", user: str | None = None,
                idempotency_key: str | None = None, max_upload_mb: int = MAX_UPLOAD_MB):
    """Validate + persist input + spawn the GPU job. Pure (no FastAPI); raises ApiError.

    This is the POST /v1/dub business logic, extracted so auth/limits/idempotency/spawn are
    testable on CPU with fakes (the FastAPI multipart layer is env-version-fragile).
    """
    mode = (mode or "basic").lower()
    if mode not in VALID_MODES:
        raise ApiError(400, f"mode must be one of {sorted(VALID_MODES)}")
    if mode == "xlingual":
        raise ApiError(400, "mode 'xlingual' is deprecated (garbled cross-lingual output); use 'vc'.")

    # Idempotency: a repeated key returns the existing job without re-running / re-billing.
    if idempotency_key:
        prior = job_status.get(f"idem:{idempotency_key}")
        if prior:
            return {"job_id": prior,
                    "status": job_status.get(prior, {}).get("status", "running"),
                    "idempotent": True}

    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_upload_mb:
        raise ApiError(413, f"file {size_mb:.0f} MB exceeds {max_upload_mb} MB limit")

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(jobs_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    input_name = os.path.basename(filename or "input.mp4")
    in_path = os.path.join(job_dir, input_name)
    with open(in_path, "wb") as fh:
        fh.write(data)

    # ── The pre-GPU duration gate ────────────────────────────────────────────────────────
    # File size in MB is NOT a proxy for GPU cost: a heavily-compressed 3-hour video clears
    # a 500 MB limit and would then occupy a GPU for hours on a plan whose ceiling is 120 s.
    # Cost here is set by DURATION, so gate on duration, before anything spins up — the API
    # twin of the CPU-preflight rule. An UNDETERMINABLE duration is rejected, not waved
    # through: a check that cannot run has not passed, and an unprobeable upload is also not
    # a media file we can dub.
    plan = (plan or "").upper()
    max_seconds = PLAN_MAX_SECONDS.get(plan, PLAN_MAX_SECONDS[""])
    duration = probe_duration(in_path)
    if duration is None:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise ApiError(400, "could not determine media duration (not a readable video/audio file)")
    if duration > max_seconds:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise ApiError(413, f"input is {duration:.0f}s; plan {plan or 'FREE'} allows "
                            f"{max_seconds}s. Upgrade the plan or send a shorter clip.")

    try:
        jobs_vol.commit()
    except Exception:
        pass

    job_status[job_id] = {
        "status": "queued", "mode": mode, "target_lang": target_lang, "user": user,
        "plan": plan, "max_seconds": max_seconds, "input_seconds": duration,
        "created_at": time.time(),
    }
    if idempotency_key:
        job_status[f"idem:{idempotency_key}"] = job_id

    dub_video.spawn(job_id, input_name, target_lang, mode)   # fire-and-forget GPU job
    return {"job_id": job_id, "status": "queued", "poll": f"/v1/dub/{job_id}"}


def build_api(dub_video, job_status, jobs_vol, jobs_dir):
    api = FastAPI(title="Indic AI Dubbing API", version="1.0")

    expected_secret = os.environ.get("RAPIDAPI_PROXY_SECRET", "")

    def _auth(proxy_secret: str | None):
        # If a secret is configured, require it. (Unset => open, for local `modal serve` dev.)
        if expected_secret and proxy_secret != expected_secret:
            raise HTTPException(status_code=403, detail="Invalid or missing RapidAPI proxy secret.")

    @api.get("/healthz")
    def healthz():
        return {"ok": True, "ts": time.time()}

    @api.post(
        "/v1/dub",
        # We parse the multipart form off the raw Request below instead of declaring
        # UploadFile/Form params, so FastAPI never synthesizes a Body_<op> model. On some
        # fastapi/pydantic combos it fails to rebuild that model's TypeAdapter, and OpenAPI
        # generation then 500s with "Body_create_dub_v1_dub_post is not fully defined" — a
        # structural issue, not a version one (it reproduces on pydantic 2.10 too). The
        # resolver floats fastapi anyway because f5-tts pulls gradio, so pinning couldn't fix
        # it. openapi_extra restores the request schema that the params would have documented.
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string", "format": "binary",
                                         "description": "Source video/audio to dub."},
                                "target_lang": {"type": "string", "default": "Hindi",
                                                "description": "Target language name, e.g. Hindi, Tamil."},
                                "mode": {"type": "string", "enum": ["basic", "vc"], "default": "basic",
                                         "description": "basic = native voice; vc = premium voice clone."},
                            },
                        }
                    }
                },
            }
        },
    )
    async def create_dub(request: Request):
        _auth(request.headers.get("x-rapidapi-proxy-secret"))
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(422, "multipart field 'file' (a file upload) is required")
        data = await upload.read()
        filename = getattr(upload, "filename", None) or "input.mp4"
        target_lang = str(form.get("target_lang") or "Hindi")
        mode = str(form.get("mode") or "basic")
        try:
            return prepare_job(
                job_status=job_status, jobs_vol=jobs_vol, jobs_dir=jobs_dir, dub_video=dub_video,
                data=data, filename=filename, target_lang=target_lang,
                mode=mode, plan=request.headers.get("x-rapidapi-subscription") or "",
                user=request.headers.get("x-rapidapi-user"),
                idempotency_key=request.headers.get("idempotency-key"),
            )
        except ApiError as e:
            raise HTTPException(e.status_code, e.detail)

    @api.get("/v1/dub/{job_id}")
    def get_dub(job_id: str, x_rapidapi_proxy_secret: str | None = Header(None)):
        _auth(x_rapidapi_proxy_secret)
        st = job_status.get(job_id)
        if not st:
            raise HTTPException(404, "unknown job_id")
        # Don't leak the internal log tail unless failed (useful for support).
        public = {k: v for k, v in st.items() if k not in ("log",)}
        if st.get("status") == "failed":
            public["log"] = st.get("log", "")
        return public

    @api.get("/v1/dub/{job_id}/download")
    def download_dub(job_id: str, x_rapidapi_proxy_secret: str | None = Header(None)):
        _auth(x_rapidapi_proxy_secret)
        st = job_status.get(job_id)
        if not st:
            raise HTTPException(404, "unknown job_id")
        if st.get("status") != "done":
            raise HTTPException(409, f"job not done (status={st.get('status')})")
        out = st.get("output", "")
        try:
            jobs_vol.reload()
        except Exception:
            pass
        if not out or not os.path.exists(out):
            raise HTTPException(410, "output no longer available")
        return FileResponse(out, media_type="video/mp4", filename=f"dubbed_{job_id}.mp4")

    return api
