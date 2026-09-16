# Deploy — Modal + FastAPI + Streamlit + RapidAPI

Serverless-GPU deployment of the Indic dubbing pipeline as an **async job API**, plus a
Streamlit UI and a RapidAPI monetization front. Full rationale, latency budget, and unit
economics are in [`../../PRODUCTION_INFRA_PLAN.md`](../../PRODUCTION_INFRA_PLAN.md) at the
repo root (also `E:\Dubbing app\PRODUCTION_INFRA_PLAN.md`).

> Status: scaffold. The code reuses the proven `run_headless` flow (7 stages + supervisor
> freeze-fix + the Step-6.5 voice-conversion clone path) as the single source of truth. It
> has not been `modal deploy`-run from the build machine — every account/secret/decision is
> marked `# SETUP:` in the source. Validate with `modal serve` before `modal deploy`.

## Files
| file | role |
|---|---|
| `modal_app.py` | Modal app: GPU image, weight-cache Volume, secrets, `dub_video` worker, ASGI mount |
| `api.py` | FastAPI gateway: `POST /v1/dub`, `GET /v1/dub/{id}`, `/download`, RapidAPI auth, limits, idempotency |
| `streamlit_app.py` | Human UI + sales demo; talks to the API |

## One-time setup
```bash
pip install modal && modal setup
# secrets the pipeline needs (Gemini for translation, HF for gated weights, RapidAPI verify):
modal secret create dubbing-secrets \
  GEMINI_API_KEY=xxx HF_TOKEN=xxx RAPIDAPI_PROXY_SECRET=$(openssl rand -hex 16)
```
Volumes (`indic-dubbing-hf-cache`, `indic-dubbing-jobs`) and the `indic-dubbing-status` Dict
are auto-created on first deploy.

## Deploy
```bash
modal serve deploy/modal_app.py      # dev, live-reload; prints the FastAPI URL
modal deploy deploy/modal_app.py     # production
```
- Model weights download to the cache Volume on the **first** job, then persist (no re-download).
- Set `MODAL_WARM=1` to keep one GPU container warm (removes cold-start; costs idle GPU time).
- GPU defaults to `L4` (fits Whisper + Demucs + IndicF5 + knn-vc). Override with `MODAL_GPU`.

## Streamlit UI
```bash
DUB_API_URL=https://<your-modal-fastapi-url> DUB_API_KEY=<proxy-secret> \
  streamlit run deploy/streamlit_app.py
```
Or deploy on Streamlit Community Cloud with those two as secrets.

## RapidAPI listing (monetization)
1. Deploy the API; note the FastAPI base URL from Modal.
2. On RapidAPI Hub → **Add New API** → point the base URL at your Modal URL.
3. Add a **transform/header** that injects `X-RapidAPI-Proxy-Secret` = your secret so the
   endpoint only accepts calls that came through RapidAPI (`api.py::_auth` enforces it).
4. Define plans (meter = processed video-minute; ceilings in `api.py::PLAN_MAX_SECONDS`):
   - **Free** — 5 min/mo, Basic, watermark (funnel)
   - **Basic** — ~$0.20/video-min, native voice
   - **Pro** — ~$0.75–1.50/video-min, voice cloning (`mode=vc`) + `nfe 48`
   - **Enterprise** — committed volume + SLA
5. RapidAPI issues keys, enforces quotas, and bills. You don't build billing.

## API shape (async)
```bash
# submit
curl -X POST $URL/v1/dub -H "X-RapidAPI-Proxy-Secret: $S" \
  -F file=@clip.mp4 -F target_lang=Hindi -F mode=vc
# -> {"job_id":"...","status":"queued","poll":"/v1/dub/..."}

# poll
curl $URL/v1/dub/$JOB -H "X-RapidAPI-Proxy-Secret: $S"
# -> {"status":"running"|"done"|"failed", "video_seconds":..., "output":...}

# download when done
curl -L $URL/v1/dub/$JOB/download -H "X-RapidAPI-Proxy-Secret: $S" -o dubbed.mp4
```
`mode`: `basic` (native voice) | `vc` (premium clone). `xlingual` is rejected (garbled).
Send `Idempotency-Key` to make retries safe.

### Input limits — gated before any GPU starts
GPU cost is set by **duration**, not file size: a well-compressed 3-hour video clears a
500 MB upload limit and would then hold a GPU for hours. So `POST /v1/dub` ffprobes the
upload and rejects it *before* `dub_video.spawn()`:

| response | when |
|---|---|
| `413` | duration exceeds the caller's plan ceiling (`api.py::PLAN_MAX_SECONDS`) |
| `400` | duration cannot be determined — an unreadable/non-media upload |

An undeterminable duration is **rejected, not waived**: a check that cannot run has not
passed. The same `probe_duration()` gates the input and meters the output, so the number a
customer is billed on and the number they were gated on come from one implementation.

```bash
python deploy/test_api_gate.py   # CPU test of the gate; needs ffmpeg, no GPU/Modal
```

### Job progress
`GET /v1/dub/{id}` reports a live `stage` (`step 4/7 — Isochrony-aware translation…`,
`step 6/7 — TTS segment 9/13`) parsed from the worker's streamed stdout, so a multi-minute
job isn't an opaque `running`.

## Cost (approx — verify current Modal pricing)
L4 ≈ $0.80/hr ⇒ ~$0.013/GPU-min. All-in COGS ≈ **$0.03–0.05 / video-minute**. Basic at
$0.20 and Pro at $0.75–1.50 leave 80–90%+ gross margin (RapidAPI takes ~20%).

## Known next steps
- **TTS fan-out**: swap in-container serial TTS for a Modal `Cls` + `.map()` across GPU
  replicas to take the ~235 s TTS stage to ~30 s (see FAN-OUT NOTE in `modal_app.py`).
- **Get off the free Gemini key** before charging: it rate-limits under concurrent load and
  reselling its output is almost certainly against its ToS. See `PRODUCTION_INFRA_PLAN.md` §4.4
  (self-hosted IndicTrans2, gated on a real per-language quality comparison).
- **Licensing**: confirm IndicF5 *and* knn-vc permit commercial resale of generated audio
  before any paid listing (§4.5). No GPU needed; it can invalidate the business model.
- **Artifacts on S3/R2** instead of the jobs Volume for multi-region download + lifecycle.
- **Studio mastering**: EBU R128 loudness, sidechain ducking, `nfe 48` for the paid tier.
