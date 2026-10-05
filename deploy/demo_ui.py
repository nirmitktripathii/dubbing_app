#!/usr/bin/env python3
"""Web demo UI for the dubbing service — one page, served by the same FastAPI gateway.

    GET  /                          the page (deploy/web/index.html)
    GET  /ui/config                 languages, limits, whether the demo is enabled
    POST /ui/dub                    multipart: file, target_lang, mode  (X-Demo-Code header)
    POST /ui/dub/youtube            form/JSON: url, target_lang, mode — a PUBLIC YouTube video
                                    within the duration limit, fetched server-side (X-Demo-Code)
    GET  /ui/dub/{job_id}           progress %, description, stage, error (X-Demo-Code)
    GET  /ui/dub/{job_id}/video?t=  the dubbed mp4 — a SHAREABLE link, gated by a per-job token
    GET  /ui/dub/{job_id}/transcript/{which}.{fmt}?t=
                                    which = source (English) | target (dubbed language),
                                    fmt = srt | txt | vtt — same per-job token as the video
    GET  /ui/dub/{job_id}/source?t= the input video as the pipeline received it (same token)

It submits through the SAME `prepare_job` + `dub_video.spawn` path as POST /v1/dub, so the page
exercises exactly the deployed pipeline — duration gate, idempotency, split orchestrator,
TTS fan-out — not a parallel copy of it. A YouTube link is fetched by a CPU function
(deploy/youtube_fetch.py) and then admitted through api.admit_job, the same duration gate an
upload passes; the limit is checked on YouTube's metadata before any download, and again by
ffprobe on the file that arrives.

ACCESS. The page is public and every submit spends GPU money, so /ui/dub requires a code from
the Modal secret `dubbing-demo`: DEMO_ACCESS_CODE (the one you hand out) or, optionally,
DEMO_OWNER_CODE (the owner's own; jobs record which one was used, as user demo-ui / demo-owner).
Both get the same DEMO limits. DEMO_ACCESS_CODE unset => the demo refuses (503): a check that
cannot run has not passed. Never hand out a code that equals RAPIDAPI_PROXY_SECRET — it would
let a third party call /v1 around RapidAPI; that is why the shared code is separate.

WHAT THE PAGE NEVER SEES. The pipeline log and the compute cost are owner-only: /ui/dub/{id}
returns neither (the log names internal models, paths and providers; the cost is the owner's
business). The owner reads both from the job record in the `indic-dubbing-status` Dict or via
/v1/jobs; estimate_cost() is kept for that owner-side use.

COST. estimate_cost() is an ESTIMATE: container-seconds we timed inside each Modal
function x Modal's per-second list price (the constants in tools/cost_model.py, so there is one
price table in the repo). It includes the configured scale-down idle tail of every container we
start. It EXCLUDES what no timer inside a function can see — image boot / snapshot restore
before the function body runs, the always-on API container, and Gemini API usage. The invoice
(`modal billing`) is the ground truth; not this estimate.
"""
from __future__ import annotations

import hmac
import os
import re
import secrets
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from deploy.api import ApiError, MAX_UPLOAD_MB, PLAN_MAX_SECONDS, check_mode, prepare_job
from deploy.youtube_fetch import FetchRejected, access_configured, canonical_url, parse_youtube_url

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

LANGUAGES = ["Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Marathi",
             "Bengali", "Gujarati", "Punjabi", "Odia", "Assamese"]
DEMO_SOURCE = "demo-ui"

# Scale-down idle tails billed after each function body returns. These mirror modal_app.py:
# gpu_transcribe and TTSEngine set MODAL_GPU_SCALEDOWN / MODAL_TTS_SCALEDOWN (default 5 s);
# gpu_full_run and the CPU orchestrator set none, so Modal's default (60 s) applies.
MODAL_DEFAULT_SCALEDOWN_S = 60.0


def _tail(env: str, default: float) -> float:
    try:
        return float(os.environ.get(env, default))
    except ValueError:
        return default


# ── Progress ───────────────────────────────────────────────────────────────────────────────
# Stage strings are written by modal_app._parse_stage from run_headless's real log lines
# ("step N/7 — ...", "step 6/7 — TTS segment i/n"). Each step owns a band of the bar sized to
# its share of a measured run's wall clock — Step 4 (Gemini translation) is ~60% of it — so the
# bar does not race to 60% and then sit there.
STEP_BANDS = {
    "1": (3, 6, "Extracting audio"),
    "2": (6, 14, "Separating the voice from background sound"),
    "3": (14, 24, "Transcribing the speech"),
    "4": (24, 62, "Translating with timing constraints"),
    "5": (62, 65, "Preparing the target voice"),
    "6": (65, 90, "Synthesising the dubbed speech"),
    "6.5": (90, 94, "Cloning the original speaker's voice"),
    "7": (94, 99, "Mixing audio and merging the video"),
}
_STEP_RE = re.compile(r"step\s+(\d+(?:\.\d+)?)/7")
_SEG_RE = re.compile(r"segment\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def progress_for(st: dict) -> tuple[int, str]:
    """(percent, human description) for a job-status record."""
    status = st.get("status", "queued")
    stage = str(st.get("stage") or "")
    if status == "done":
        return 100, "Done — your dubbed video is ready"
    if status == "queued":
        return 1, "Queued — waiting for a GPU"
    if stage.startswith("fetch"):
        if status == "failed":
            return 1, "Couldn't fetch the YouTube video"
        return 1, "Fetching the YouTube video"
    m = _STEP_RE.search(stage)
    if m and m.group(1) in STEP_BANDS:
        lo, hi, desc = STEP_BANDS[m.group(1)]
        seg = _SEG_RE.search(stage)
        if seg and int(seg.group(2)) > 0:
            frac = min(1.0, int(seg.group(1)) / int(seg.group(2)))
            return int(lo + (hi - lo) * frac), f"{desc} (segment {seg.group(1)}/{seg.group(2)})"
        pct, text = lo, desc
    elif stage.startswith("fallback"):
        pct, text = 3, "Retrying on the single-GPU path"
    elif stage == "transcribing":
        pct, text = 3, "Starting the GPU for transcription"
    else:
        pct, text = 2, "Starting up"
    if status == "failed":
        return pct, "Failed — see the log below"
    return pct, text


# ── Cost ───────────────────────────────────────────────────────────────────────────────────
def estimate_cost(st: dict) -> dict | None:
    """Modal compute cost for one job, from the meters modal_app writes. None before any meter.

    meter.gpu_transcribe_s  L4 body time of gpu_transcribe (Steps 1-3), summed over attempts
    meter.gpu_full_s        L4 body time of gpu_full_run (vc mode / kill-switch / fallback)
    meter.cpu_orchestrator_s  wall time of the CPU dub_video orchestrator
    meter.cpu_fetch_s       wall time of the CPU fetch_youtube function (YouTube-link jobs)
    tts_meter.workers       per TTSEngine shard: container task id + seconds since it came up
    """
    from tools.cost_model import cpu_s, gpu_s   # the repo's single price table

    m = st.get("meter") or {}
    tts = st.get("tts_meter") or {}
    if not m and not tts:
        return None
    lines = []

    def add(label, seconds, usd, note=""):
        lines.append({"label": label, "seconds": round(seconds, 1), "usd": round(usd, 5),
                      "note": note})

    gpu_tail = _tail("MODAL_GPU_SCALEDOWN", 5.0)
    if m.get("gpu_transcribe_s"):
        s = m["gpu_transcribe_s"] + gpu_tail * max(1, m.get("gpu_transcribe_n", 1))
        add("L4 GPU — separation + transcription (steps 1–3)", s, gpu_s(s),
            f"incl. {gpu_tail:.0f} s idle tail")
    if m.get("gpu_full_s"):
        s = m["gpu_full_s"] + MODAL_DEFAULT_SCALEDOWN_S * max(1, m.get("gpu_full_n", 1))
        add("L4 GPU — full single-container run", s, gpu_s(s),
            f"incl. {MODAL_DEFAULT_SCALEDOWN_S:.0f} s idle tail")
    workers = tts.get("workers") or []
    if workers:
        per_task: dict[str, float] = {}
        for i, w in enumerate(workers):
            key = w.get("task") or f"shard-{i}"
            per_task[key] = max(per_task.get(key, 0.0), float(w.get("up_s") or 0.0))
        tail = float(tts.get("scaledown_s", _tail("MODAL_TTS_SCALEDOWN", 5.0)))
        s = sum(per_task.values()) + tail * len(per_task)
        add(f"L4 GPU — speech synthesis, {len(per_task)} worker(s)", s, gpu_s(s),
            f"incl. {tail:.0f} s idle tail each")
    if m.get("cpu_fetch_s"):
        s = m["cpu_fetch_s"] + MODAL_DEFAULT_SCALEDOWN_S
        add("CPU — YouTube fetch", s, cpu_s(s), f"incl. {MODAL_DEFAULT_SCALEDOWN_S:.0f} s idle tail")
    if m.get("cpu_orchestrator_s"):
        s = m["cpu_orchestrator_s"] + MODAL_DEFAULT_SCALEDOWN_S
        add("CPU — orchestrator (translation + assembly)", s, cpu_s(s),
            f"incl. {MODAL_DEFAULT_SCALEDOWN_S:.0f} s idle tail")

    total = sum(l["usd"] for l in lines)
    secs = st.get("input_seconds") or st.get("video_seconds")
    return {
        "total_usd": round(total, 4),
        "input_seconds": secs,
        "usd_per_video_minute": round(total / secs * 60, 4) if secs else None,
        "lines": lines,
        # False until the orchestrator has written its own meter (it closes last).
        "final": bool(st.get("metered")) and st.get("status") in ("done", "failed"),
        "method": ("Estimate: container-seconds timed inside each Modal function × Modal list "
                   "price (L4 $0.000222/s + CPU/RAM, tools/cost_model.py), plus each "
                   "container's scale-down idle tail."),
        "excludes": ["container boot / snapshot restore before the function starts",
                     "the always-on API container", "Gemini API usage",
                     "the Modal invoice (`modal billing`) is the ground truth"],
    }


# ── Transcripts ────────────────────────────────────────────────────────────────────────────
# run_headless writes both next to the output video: english_subtitles.srt (Step 3) and
# <Language>_subtitles.srt (Step 4). They are served as written (srt), as plain text (txt), and
# as WebVTT (vtt) — the only caption format a browser <track> element accepts.
TRANSCRIPT_FORMATS = {"srt": "application/x-subrip", "txt": "text/plain", "vtt": "text/vtt"}
_SRT_TIME = re.compile(r"(\d{2}:\d{2}:\d{2}),(\d{3})")


def transcript_path(out_video: str, which: str, target_lang: str) -> str:
    name = "english_subtitles.srt" if which == "source" else f"{target_lang}_subtitles.srt"
    return os.path.join(os.path.dirname(out_video), name)


def _srt_blocks(srt: str) -> list[tuple[str, str]]:
    """[(timing line, text)] from an SRT, skipping index lines and malformed blocks."""
    blocks = []
    for raw in re.split(r"\r?\n\s*\r?\n", srt.strip().lstrip("﻿")):
        lines = [l for l in raw.splitlines() if l.strip()]
        timing = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if timing is None:
            continue
        text = "\n".join(lines[timing + 1:]).strip()
        if text:
            blocks.append((lines[timing].strip(), text))
    return blocks


def srt_to(srt: str, fmt: str) -> str:
    if fmt == "srt":
        return srt
    blocks = _srt_blocks(srt)
    if fmt == "txt":
        return "\n".join(text.replace("\n", " ") for _, text in blocks) + "\n"
    # vtt: header, '.' as the millisecond separator, no numeric cue ids needed
    cues = [_SRT_TIME.sub(r"\1.\2", timing) + "\n" + text for timing, text in blocks]
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


# ── Routes ─────────────────────────────────────────────────────────────────────────────────
_REDACT = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,})")


def _public_error(st: dict) -> str | None:
    """The one free-text field the page shows. It comes from an exception message, so redact
    anything key-shaped before it leaves the server."""
    err = st.get("error")
    return _REDACT.sub("[redacted]", str(err)) if err else None


def voice_coverage() -> dict:
    """{display language: "native" | "pinned" | "fallback"} — which Basic-mode reference voice
    each language gets (pipeline/voice_refs.py). {} if it cannot be determined; the page then
    simply shows no voice hint rather than a wrong one."""
    try:
        from pipeline import voice_refs
        cov = voice_refs.coverage()
        return {voice_refs.CODE_TO_LANGUAGE[c]: s for c, s in cov.items()}
    except Exception:
        return {}


def add_demo_routes(api: FastAPI, *, dub_video, job_status, jobs_vol, jobs_dir,
                    fetch_youtube=None) -> None:
    def _code() -> str:
        return os.environ.get("DEMO_ACCESS_CODE", "")

    def _check(request: Request) -> str:
        """Returns who is calling ("demo-ui" or "demo-owner"); raises 503/403 otherwise."""
        code = _code()
        if not code:
            raise HTTPException(503, "Demo UI is disabled: DEMO_ACCESS_CODE is not set "
                                     "(Modal secret 'dubbing-demo').")
        got = (request.headers.get("x-demo-code") or "").encode()
        owner = os.environ.get("DEMO_OWNER_CODE", "")
        # Compare against both unconditionally so timing does not reveal which one matched.
        is_shared = hmac.compare_digest(got, code.encode())
        is_owner = bool(owner) & hmac.compare_digest(got, owner.encode())
        if is_owner:
            return "demo-owner"
        if is_shared:
            return "demo-ui"
        raise HTTPException(403, "Wrong access code.")

    def _demo_job(job_id: str) -> dict:
        st = job_status.get(job_id)
        if not st or st.get("source") != DEMO_SOURCE:
            raise HTTPException(404, "unknown job")
        return st

    @api.get("/", include_in_schema=False)
    def index():
        with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as fh:
            return HTMLResponse(fh.read())

    @api.get("/ui/config", include_in_schema=False)
    def config():
        return {"enabled": bool(_code()), "languages": LANGUAGES, "voices": voice_coverage(),
                "max_seconds": PLAN_MAX_SECONDS["DEMO"], "max_upload_mb": MAX_UPLOAD_MB,
                "youtube": fetch_youtube is not None and access_configured(),
                "simulated": os.environ.get("DUB_DEMO_SIMULATED") == "1"}

    @api.post("/ui/dub", include_in_schema=False)
    async def ui_create(request: Request):
        who = _check(request)
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(422, "choose a video file")
        target_lang = str(form.get("target_lang") or "Hindi")
        if target_lang not in LANGUAGES:
            raise HTTPException(400, f"target_lang must be one of {LANGUAGES}")
        mode = str(form.get("mode") or "basic")
        data = await upload.read()
        try:
            r = prepare_job(
                job_status=job_status, jobs_vol=jobs_vol, jobs_dir=jobs_dir,
                dub_video=dub_video, data=data,
                filename=getattr(upload, "filename", None) or "input.mp4",
                target_lang=target_lang, mode=mode, plan="DEMO", user=who,
                idempotency_key=request.headers.get("idempotency-key"),
                extra={"source": DEMO_SOURCE, "dl_token": secrets.token_urlsafe(18)},
            )
        except ApiError as e:
            raise HTTPException(e.status_code, e.detail)
        return {"job_id": r["job_id"], "status": r["status"]}

    @api.post("/ui/dub/youtube", include_in_schema=False)
    async def ui_create_youtube(request: Request):
        """Accept a YouTube link; the fetch (and its duration gate) runs in a CPU function, so
        this returns a job id at once and the page follows the fetch like any other stage."""
        who = _check(request)
        if fetch_youtube is None or not access_configured():
            raise HTTPException(503, "YouTube links are not enabled on this deployment "
                                     "(no YouTube access is configured). Upload the file instead.")
        if "json" in (request.headers.get("content-type") or ""):
            form = await request.json()
        else:
            form = await request.form()
        target_lang = str(form.get("target_lang") or "Hindi")
        if target_lang not in LANGUAGES:
            raise HTTPException(400, f"target_lang must be one of {LANGUAGES}")
        try:
            mode = check_mode(str(form.get("mode") or "basic"))
            vid = parse_youtube_url(str(form.get("url") or ""))
        except ApiError as e:
            raise HTTPException(e.status_code, e.detail)
        except FetchRejected as e:
            raise HTTPException(400, str(e))
        idem = request.headers.get("idempotency-key")
        if idem:
            prior = job_status.get(f"idem:{idem}")
            if prior:
                return {"job_id": prior, "status": job_status.get(prior, {}).get("status")}
        job_id = uuid.uuid4().hex
        extra = {"source": DEMO_SOURCE, "dl_token": secrets.token_urlsafe(18)}
        job_status[job_id] = {"status": "fetching", "stage": "fetching", "mode": mode,
                              "target_lang": target_lang, "user": who, "plan": "DEMO",
                              "created_at": time.time(), "youtube": {"id": vid,
                              "url": canonical_url(vid)}, **extra}
        if idem:
            job_status[f"idem:{idem}"] = job_id
        fetch_youtube.spawn(job_id, canonical_url(vid), target_lang, mode, "DEMO", who, extra)
        return {"job_id": job_id, "status": "fetching"}

    @api.get("/ui/dub/{job_id}", include_in_schema=False)
    def ui_status(job_id: str, request: Request):
        _check(request)
        st = _demo_job(job_id)
        pct, desc = progress_for(st)
        created = st.get("created_at")
        if st.get("status") in ("done", "failed"):
            end = st.get("finished_at") or st.get("heartbeat") or time.time()
        else:
            end = time.time()
        out = {
            "job_id": job_id, "status": st.get("status"), "stage": st.get("stage"),
            "percent": pct, "description": desc,
            "target_lang": st.get("target_lang"), "mode": st.get("mode"),
            "input_seconds": st.get("input_seconds"),
            # Upload-accepted -> finished: what the person waiting actually experienced.
            "elapsed_s": round(end - created, 1) if created else None,
            "error": _public_error(st),
            "youtube": st.get("youtube"),
        }
        if st.get("status") == "done":
            tok = st.get("dl_token", "")
            base = f"/ui/dub/{job_id}/video?t={tok}"
            out["video_url"] = base
            out["download_url"] = base + "&download=1"
            out["video_seconds"] = st.get("video_seconds")
            out["source_url"] = f"/ui/dub/{job_id}/source?t={tok}"
            # URLs only; the page fetches each one and hides any that is missing (404), so a
            # status poll never has to reload the volume to check the files exist.
            tr = f"/ui/dub/{job_id}/transcript"
            out["transcripts"] = {
                which: {"label": label,
                        **{fmt: f"{tr}/{which}.{fmt}?t={tok}" for fmt in TRANSCRIPT_FORMATS}}
                for which, label in (("source", "English"),
                                     ("target", st.get("target_lang") or "Dubbed"))}
        return out

    def _done_output(job_id: str, t: str) -> tuple[dict, str]:
        """(status, output video path) for a finished demo job whose link token matches."""
        st = _demo_job(job_id)
        tok = st.get("dl_token") or ""
        if not tok or not hmac.compare_digest(t.encode(), tok.encode()):
            raise HTTPException(403, "invalid link")
        if st.get("status") != "done":
            raise HTTPException(409, f"not ready (status={st.get('status')})")
        try:
            jobs_vol.reload()
        except Exception:
            pass
        return st, st.get("output", "")

    @api.get("/ui/dub/{job_id}/video", include_in_schema=False)
    def ui_video(job_id: str, t: str = "", download: int = 0):
        st, out = _done_output(job_id, t)
        if not out or not os.path.exists(out):
            raise HTTPException(410, "output no longer available")
        name = f"dubbed_{(st.get('target_lang') or 'video').lower()}_{job_id[:8]}.mp4"
        return FileResponse(out, media_type="video/mp4", filename=name,
                            content_disposition_type="attachment" if download else "inline")

    @api.get("/ui/dub/{job_id}/transcript/{which}.{fmt}", include_in_schema=False)
    def ui_transcript(job_id: str, which: str, fmt: str, t: str = "", download: int = 0):
        if which not in ("source", "target") or fmt not in TRANSCRIPT_FORMATS:
            raise HTTPException(404, "unknown transcript")
        st, out = _done_output(job_id, t)
        lang = st.get("target_lang") or ""
        path = transcript_path(out, which, lang) if out else ""
        if not path or not os.path.exists(path):
            raise HTTPException(404, "transcript not available for this job")
        with open(path, encoding="utf-8-sig", newline="") as fh:   # srt served as written
            body = srt_to(fh.read(), fmt)
        tag = "english" if which == "source" else (lang or "dubbed").lower()
        name = f"transcript_{tag}_{job_id[:8]}.{fmt}"
        disp = "attachment" if download else "inline"
        return Response(body, media_type=f"{TRANSCRIPT_FORMATS[fmt]}; charset=utf-8",
                        headers={"Content-Disposition": f'{disp}; filename="{name}"'})

    @api.get("/ui/dub/{job_id}/source", include_in_schema=False)
    def ui_source(job_id: str, t: str = ""):
        """The input as the pipeline received it — for a YouTube job, the page's "Original"."""
        st, _ = _done_output(job_id, t)
        name = os.path.basename(st.get("input_name") or "")
        path = os.path.join(jobs_dir, job_id, name) if name else ""
        if not path or not os.path.isfile(path):
            raise HTTPException(404, "source not available for this job")
        return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")
