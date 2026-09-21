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

**1. Account.** Sign up at [modal.com](https://modal.com) (GitHub/Google SSO). The free
tier includes monthly credit — enough to validate this pipeline — and GPUs are billed per
second while a container runs, so an idle deployment costs nothing.

**2. CLI + auth.** From the repo root:
```bash
pip install modal
modal setup
```
`modal setup` opens a browser and writes a token to `~/.modal.toml`. On a headless box use
`modal token new` and paste the token instead. Verify with `modal profile current`.

**3. Secrets.** One Modal secret holds everything the pipeline reads from the environment.
Unlike Kaggle, a Modal secret is created once and referenced by every function — there is
no per-notebook attach step:
```bash
modal secret create dubbing-secrets \
  GEMINI_API_KEY=... \
  HF_TOKEN=... \
  RAPIDAPI_PROXY_SECRET=$(openssl rand -hex 16)
```
| key | used by | required? |
|---|---|---|
| `GEMINI_API_KEY` | Step 4 isochrony translation | **yes** — the run aborts without it |
| `HF_TOKEN` | IndicF5 + the default reference voice from HF | yes for gated repos |
| `RAPIDAPI_PROXY_SECRET` | `api.py::_auth`, so the endpoint can't be called around RapidAPI | before listing; unset = open (dev only) |

> The `GEMINI_API_KEY` here should **not** stay a free-tier key once you charge for this —
> it rate-limits under concurrent load and reselling its output is almost certainly against
> its ToS. See `PRODUCTION_INFRA_PLAN.md` §4.4.

**4. Storage.** Nothing to create by hand — the two Volumes (`indic-dubbing-hf-cache` for
model weights, `indic-dubbing-jobs` for job artifacts) and the `indic-dubbing-status` Dict
are auto-created on first deploy. Model weights download to the cache Volume on the **first
job only** and persist after that, so the first run is slow and later ones are not.

**5. First run.** `modal serve` gives a live-reloading dev deployment with a public URL:
```bash
modal serve deploy/modal_app.py            # dev; prints the FastAPI URL, Ctrl-C to stop
curl $URL/healthz                          # sanity check, no GPU used
modal deploy deploy/modal_app.py           # production
```
Useful while iterating: `modal app list`, `modal app logs indic-dubbing`,
`modal app stop indic-dubbing`, `modal volume ls indic-dubbing-jobs`.

**Cost control.** GPU billing is per second of container runtime. `MODAL_TTS_WARM` and
`MODAL_WARM` keep containers alive between jobs — they remove cold-start latency but bill
idle GPU time, so leave them at `0` until latency actually matters. Set a spend limit in
the Modal dashboard before pointing paid traffic at it.

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

## Step-6 TTS fan-out
TTS dominates the run: a measured ~235 s of a ~7.8 min job for an 80 s video, because 13
segments are synthesized serially at ~18 s each. Segments are independent (each target
comes from its own SRT slot; drift correction compares a segment against its *own* target,
with no accumulated state), so this parallelizes.

`deploy/tts_fanout.py` shards the segments across `TTSEngine` containers (`modal_app.py`).
A worker calls the **same** `duration_tts.generate_tts_for_segments` as the serial and
Kaggle paths, passing the full segment list plus `only_indices` for its shard — so language
/nfe/reference resolution, the segment signature, drift correction and silence degradation
are all literally the same code, `i` stays the **global** index, and a run started on one
backend resumes on the other. Workers never share a manifest: each writes to a private dir
and returns WAV bytes + entries, which the orchestrator merges as the single writer.

Work maps over **shards, not segments** — a worker's cost is dominated by loading IndicF5,
so a per-segment map would pay that load per segment.

### What the speedup actually is
```
wall ≈ model_load + ceil(n_segments / n_shards) × per_segment
```
The floor is `model_load + per_segment`, not zero. For the measured 13-segment clip at
~18 s/segment with a ~40 s cold model load:

| shards | cold (`MODAL_TTS_WARM=0`) | warm (`MODAL_TTS_WARM≥1`) |
|---|---|---|
| 1 (serial today) | ~275 s | ~235 s |
| 4 | ~112 s | ~72 s |
| 8 | ~76 s | ~36 s |
| 13 (one each) | ~58 s | ~18 s |

So the plan's "~30 s" is reachable only with a **warm pool** — cold, the model load sets a
~58 s floor no matter how wide you fan out. Past ~8 shards the curve flattens while the
number of model loads keeps rising. These are projections from the measured 18 s/segment,
**not** measured fan-out numbers; the first real deploy should replace this table.

Fan-out is on by default in the Modal worker. Dials:
| env | default | effect |
|---|---|---|
| `DUBBING_TTS_FANOUT` | `1` | `0` reverts to the in-container supervised path |
| `MODAL_TTS_MAX_CONTAINERS` | `8` | ceiling on parallel TTS containers |
| `MODAL_TTS_WARM` | `0` | warm engines; removes the model-load term (costs idle GPU) |
| `DUBBING_TTS_MAX_SHARDS` | `8` | shards a run is split into |
| `DUBBING_NFE_STEP` | `32` | diffusion steps; latency ~linear (a per-tier cost dial) |

```bash
python deploy/test_tts_fanout.py   # CPU tests: sharding, merge, resume/staleness
```

## Known next steps
- **Get off the free Gemini key** before charging: it rate-limits under concurrent load and
  reselling its output is almost certainly against its ToS. See `PRODUCTION_INFRA_PLAN.md` §4.4
  (self-hosted IndicTrans2, gated on a real per-language quality comparison).
- **Licensing**: confirm IndicF5 *and* knn-vc permit commercial resale of generated audio
  before any paid listing (§4.5). No GPU needed; it can invalidate the business model.
- **Artifacts on S3/R2** instead of the jobs Volume for multi-region download + lifecycle.
- **Studio mastering**: EBU R128 loudness, sidechain ducking, `nfe 48` for the paid tier.
