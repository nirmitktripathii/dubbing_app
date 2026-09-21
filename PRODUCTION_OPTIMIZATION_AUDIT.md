# IndicAI Dubbing — Production Optimization Audit

Status: draft for review · 2026-09-18 · branch `feat/modal-deploy`
Companion to [`PRODUCTION_INFRA_PLAN.md`](PRODUCTION_INFRA_PLAN.md) (architecture intent).
This document is the **measured** audit: what the code on `feat/modal-deploy` actually does,
where the time and GPU-dollars actually go, and the smallest changes that move the
business metric — **cost per successfully dubbed source-minute at acceptable quality and
p95 latency** — not any single benchmark.

> **Landed so far (CPU-verified, no GPU):** **P2** (subtitle soft-mux + video stream-copy by
> default) and **P10** (persist the translation cache on the Modal Volume). Both shipped with a
> property test — `tools/test_merge_and_cache.py`, summarized under each item. The rest of the
> backlog is still proposal.

## How to read this

Every number here is one of two kinds, always labelled:

- **[measured]** — parsed from a real run log that already exists in the repo, via
  `tools/profile_log.py` (a log→stage-table parser; CPU only, no GPU, no re-run). The
  source log is cited. These are facts about runs that already happened.
- **[projection]** — arithmetic on a measured number (e.g. "nfe is ~linear, so 32→16 halves
  the 205 s"). A projection is a hypothesis with a source, **not** a result. Each one names
  the benchmark that would confirm it. Nothing in this document is an invented benchmark
  table (CLAUDE rule 8: no metric without a traceable source).

---

## 1. The measured baseline

Full fresh run, 80 s source video, Demucs ON, Basic-ish path, `nfe=32`.
**[measured]** — `dubbing_output_20260915_163400/dubbing_output/pipeline_log.txt`:

| stage | code | sec | % wall | on GPU? | is the GPU working? |
|---|---|---:|---:|:--:|---|
| 1 audio extract | `utils/audio_extraction.py` | 1 | 0% | yes | **no** — FFmpeg, CPU |
| 2 Demucs | `pipeline/source_separation.py` | 12 | 3% | yes | yes |
| 3 Whisper medium | `utils/transcription.py` | 16 | 3% | yes | yes |
| 4 isochrony translation | `pipeline/isochrony_translation.py` | **152** | **32%** | yes | **no** — Gemini network call |
| 5 voice-ref extract | `pipeline/voice_manager.py` | 2 | 0% | yes | **no** — CPU/ffmpeg |
| 6 IndicF5 TTS | `pipeline/duration_tts.py` | **240** | **51%** | yes | partly — 28 s load + 205 s synth |
| 7 assembly + merge | `utils/audio_sync.py`, `utils/video_merge.py` | 45 | 10% | yes | **no** — FFmpeg re-encode, CPU |
| | **TOTAL wall** | **468** | 100% | | |

Step 6 decomposition **[measured]**: IndicF5 cold-load inside the subprocess **28 s**, then
13 segments synthesized serially, mean **15.8 s** (min 12, max 22) at nfe=32.

The single most important column is the last one. Add up the time the L4 is **billed but not
doing neural work**: extract 1 + translation 152 + ref-extract 2 + assembly/merge 45 =
**200 s of 468 s — 43% of the billed GPU wall is non-GPU work.** That is the headline.

### Translation latency is not a number, it is a distribution

Step 4 across four real runs **[measured]**:

| run log | Step 4 |
|---|---:|
| `dubbing_output_20260915_163400/.../pipeline_log.txt` | 152 s |
| `dubbing_output_2/pipeline_log.txt` | 51 s |
| `dubbing_output_hi_ref/pipeline_log.txt` (fresh) | 40 s |
| `dubbing_output_hi_ref/pipeline_log.txt` (resume, cache warm) | 16 s |

**40–152 s for the same class of clip.** The spread is 503/429 retry-and-backoff on the free
Gemini tier plus the IndicSBERT cold load, both visible in `_call_gemini`
(`pipeline/isochrony_translation.py:492`). For a paid API this variance is a p99 latency and
a reliability problem, not just a mean-latency one — and it sits on the critical path (§3).

---

## 2. Three reframes that reorder the whole priority list

The infra plan's sequencing (fan out TTS first) is right for a **demo**: it makes the wall
clock small and impressive. But for a **priced API**, three facts from the measured data
change what to do first.

### R1 — The orchestrator rents a GPU to make phone calls and run FFmpeg

`dub_video` is a **GPU** function (`deploy/modal_app.py:190`, `gpu=GPU_TYPE`) that runs the
entire pipeline as one subprocess (`deploy/modal_app.py:249`). Only TTS is offloaded. So the
L4 is held — and billed per second — through the 152 s Gemini call and the 45 s FFmpeg
re-encode, during which GPU utilization is ~0.

- **[measured]** 200 s of the 468 s run is non-GPU work on a GPU container (43%).
- **[projection]** At an *illustrative* L4 rate of $0.0133/GPU-min (plan §4.2, flagged
  illustrative), that idle time is **~$0.044 per 80 s job** — roughly **half** the plan's
  own all-in COGS estimate of $0.04–0.06/video-min. Confirm with the live Modal invoice; the
  ratio (43% idle) is measured regardless of the dollar figure.

The infra plan's own component map (§2.2) already draws the fix: a **CPU orchestrator** that
calls GPU functions for the GPU stages. The scaffold has not built it yet. This is the
largest COGS lever in the system and it is invisible to a wall-clock benchmark — the run is
not slower for being wasteful.

### R2 — TTS fan-out spends money to buy latency; it does not save money

Fan-out shards segments across containers, each of which loads IndicF5 in `@modal.enter`
(`deploy/modal_app.py:333`). The wall model in `deploy/tts_fanout.py` is honest:
`wall ≈ model_load + ceil(n/shards)·per_segment`. But the **GPU-second** model is the one
that sets COGS, and fan-out makes it *worse*:

- **[measured]** serial TTS GPU-seconds = 28 load + 205 synth = **233**.
- **[projection]** cold 8-shard = 8×28 + 205 = **429 GPU-s (+84%)**; cold 4-shard = 4×28 +
  205 = **317 (+36%)**. Each shard re-pays the model load. (Source: measured 28 s load, 205 s
  synth; the tts_fanout wall model.)

So `MODAL_TTS_MAX_CONTAINERS=8` on a 13-segment clip nearly **doubles** TTS GPU cost to cut
wall clock. That is a fine trade for a live demo or an SLA tier, and a bad default for a
margin-sensitive Basic tier. A warm pool (`MODAL_TTS_WARM≥1`) amortizes the load term away
but then bills idle GPU between jobs. **Fan-out is a latency dial with a cost, not a cost
optimization** (prompt §14, §50). The genuine TTS *cost* levers are nfe (§5, item 4) and
batching (§5, item 5), because they cut GPU-seconds, not just wall-clock.

### R3 — After fan-out, translation is the critical path — and it is the one stage that should never touch a GPU

Once TTS is parallelized, wall clock is roughly `max(stage_i)` for the serial chain.
**[projection]** at 4 warm shards TTS wall ≈ 28 + ceil(13/4)·15.8 ≈ 92 s (cold) / ~64 s
(warm); Step 4 translation is 152 s in the worst measured case. **Translation becomes the
dominant wall-clock stage**, and it is a network call that is holding a GPU (R1). Optimizing
TTS harder past ~4 shards buys almost no end-to-end latency while still inflating cost.
The correct next lever after the first fan-out is **translation**, not more TTS shards.

---

## 3. Bottleneck map (ranked, sourced)

**By wall-clock latency (today, serial):** TTS 240 s > translation 152 s > merge 45 s >
Whisper 16 s ≈ Demucs 12 s. **[measured, 163400]**

**By GPU-COGS (billed GPU-seconds today):** idle-orchestrator 200 s ≈ TTS synth 205 s >>
TTS load 28 s ≈ Whisper 16 s ≈ Demucs 12 s. The idle term (R1) is co-largest and is pure
waste; the synth term is irreducible work (but see nfe, §5 item 4).

**By reliability / p99:** translation retry-storm variance (40–152 s, §1) > whole-job retry
re-running Steps 1–5 (§5 item 8) > VC serial loop on the premium tier (§5 item 6).

**By scalability ceiling:** multipart upload buffered in gateway RAM (§5 item 9) > in-memory
`modal.Dict` job store (`deploy/modal_app.py:88`, plan already flags Postgres/Neon) > no
content-hash idempotency / cross-job cache (§5 items 7, 10).

---

## 4. Optimization backlog (prioritized; smallest safe change first)

Each item: **current → problem → change → expected effect → how to benchmark → rollback.**
Effects are latency / GPU-COGS / reliability / quality. Do them roughly in this order.

### P0 — Split the orchestrator off the GPU (R1)
- **Current:** `dub_video` is one GPU function running all stages as a subprocess
  (`deploy/modal_app.py:190,249`).
- **Problem:** 43% of billed GPU time is CPU/network work **[measured]**.
- **Change:** make the orchestrator a **CPU** `@app.function` (no `gpu=`). It runs extract,
  translation, assembly, merge locally, and calls GPU functions for the GPU stages: Demucs,
  Whisper, TTS (`TTSEngine`, exists), VC. This is exactly the plan's §2.2 diagram. Reuse the
  existing stage modules unchanged; only the *placement* moves. Keep `DUBBING_TTS_FANOUT=0`
  falling back to today's in-container path so the change is reversible per-env.
- **Effect:** GPU-COGS −40%ish on a typical job (removes the 200 s idle term); latency
  roughly unchanged; reliability up (a hung FFmpeg no longer holds a GPU). Quality
  unchanged (same code).
- **Benchmark:** one before/after run through `profile_log.py`; compare summed GPU-seconds
  (Modal per-function billing), not wall clock.
- **Rollback:** revert to the single `dub_video` GPU function; it is untouched behind the
  env flag.
- **Caveat to measure, not assume:** Demucs (12 s) and Whisper (16 s) as *separate* GPU
  functions each pay a cold start + model load. For a short clip that overhead can exceed the
  idle-GPU saving. The unambiguous wins are translation (152 s) and merge (45 s) off the GPU;
  Demucs/Whisper placement is a measured decision (§7), and a single combined "source-side
  GPU function" (extract→Demucs→Whisper→ref in one container) is likely the sweet spot.

### P1 — Stop the free-Gemini retry storm on the critical path
- **Current:** `_call_gemini` walks a model chain with 503/429 backoff
  (`pipeline/isochrony_translation.py:492`); free tier rate-limits under load.
- **Problem:** Step 4 is 40–152 s, variance-dominated **[measured]**; it is (post-fan-out)
  the critical path (R3) and a p99 risk.
- **Change (two, independent):** (a) persist the translation candidate cache across jobs —
  see item 10, a one-line env fix that alone turns repeat/similar content into ~0-call
  selection; (b) execute the plan's §4.4: self-host IndicTrans2 as a GPU function that
  batches all segments in one forward pass — no RPM ceiling, no network variance, and it
  runs on the same L4. **Gate the switch on the per-language isochrony+semantic comparison**
  the plan requires (a real gate, not a formality).
- **Effect:** latency −60 to −130 s worst case and variance collapse; external-API cost →
  ~0; reliability up. Quality: **must be measured per language** before cutover (CLAUDE
  rules 5, 7).
- **Benchmark:** `pipeline/tts_benchmark.py`-style harness but for MT: IndicTrans2 vs Gemini
  on isochrony score + IndicSBERT semantic, per language, on the elastic corpus.
- **Rollback:** keep Gemini as the fallback head of the chain (the chain structure already
  supports this).

### P2 — Make subtitle burn-in optional; stream-copy the video by default — ✅ LANDED
- **Current (before):** `utils/video_merge.py` always burned subtitles with
  `-vf subtitles=... -c:v libx264` — a **full re-encode of the video stream**.
- **Problem:** forced ~45 s **[measured]** of CPU encode (on a GPU container today) for
  every job, even when the customer wants no burned-in captions.
- **Change (done):** default is now **stream copy** — `-c:v copy -c:a aac` + the SRT
  soft-muxed as `-c:s mov_text`. `burn_subs=True` (env `DUBBING_BURN_SUBS=1`) restores hard
  burn-in, now with `-preset veryfast`. The v2 Streamlit UI (`app.py`) pins `burn_subs=True`
  to keep its prior look; the headless/Modal path takes the cheap default.
- **Effect [measured]:** on a 25 s 720×1280 real clip, **copy 1.28 s vs burn 9.99 s (7.8×)**;
  the copy output is **byte-for-byte identical video** (per-frame md5 match — proof of no
  re-encode) with a toggleable caption track. Extrapolated to the 80 s job the ~45 s Step-7
  encode becomes ~2–3 s. Quality identical (lossless copy).
  Source: `tools/test_merge_and_cache.py`.
- **Gotcha caught in test:** a soft-sub track ends at its last cue, so `-shortest` would
  truncate the whole video to the last subtitle. The copy path therefore omits `-shortest`
  (the test asserts full frame count is preserved). The burn path keeps `-shortest` — it has
  no subtitle stream.
- **Rollback:** `DUBBING_BURN_SUBS=1` (or `burn_subs=True`) returns to the old behaviour.
- **Still open (GPU):** for the burn tier, benchmark NVENC (`-c:v h264_nvenc`) on the L4.

### P3 — nfe as the real TTS cost dial (plan §4.6, made concrete)
- **Current:** `nfe_step=32` fixed default (`pipeline/duration_tts.py:388`,
  `resolve_nfe_step`).
- **Problem:** synthesis time is ~linear in nfe; 205 GPU-s of synth **[measured]** is the
  largest irreducible GPU term and it is set by a knob.
- **Change:** sweep nfe {32,24,16,8} on `pipeline/tts_benchmark.py` (it exists), grade
  quality on the intelligibility/semantic gates, set Free/Basic to the lowest nfe that holds
  quality, keep Pro at 32–48.
- **Effect [projection]:** nfe 32→16 ≈ 205→~103 GPU-s (−50%) — a real COGS cut *and* a
  wall-clock cut, unlike fan-out (R2). Quality: the sweep decides; this is the whole point of
  measuring.
- **Benchmark:** the sweep, per language; each nfe a row.
- **Rollback:** it is one env var (`DUBBING_NFE_STEP`); revert instantly per tier.

### P4 — Right-size TTS fan-out to COGS, not to core count (R2)
- **Current:** `MODAL_TTS_MAX_CONTAINERS=8`, `DUBBING_TTS_MAX_SHARDS=8`
  (`deploy/modal_app.py:318`, `deploy/tts_fanout.py:59`).
- **Problem:** 8 shards on a 13-segment clip ≈ +84% TTS GPU-seconds **[projection]** for a
  wall-clock win that R3 says is off the critical path anyway.
- **Change:** default shards to **2–4** for Basic; reserve wide fan-out + warm pool for the
  SLA/Pro tier. Make shard count a function of segment count and tier, not a fixed 8.
- **Effect:** GPU-COGS down on Basic; latency slightly up but still below the translation
  floor (R3), so end-to-end unchanged.
- **Benchmark:** GPU-seconds and p95 at shards ∈ {1,2,4,8}, warm and cold, on one clip.
- **Rollback:** env vars; no code change needed to revert.

### P5 — Multi-language: compute source-side once, fan out per language
- **Current:** `dub_video(job_id, target_lang, mode)` runs the *whole* pipeline per language;
  the API has no multi-language job.
- **Problem:** extract + Demucs + Whisper + ref-extract are **language-independent** (31 s,
  of which Demucs+Whisper = 28 GPU-s **[measured]**) yet re-run for every requested language.
  A 3-language request pays 3× source-side for zero added value.
- **Change:** add `target_langs: [..]`. Run Steps 1–3+5 once → shared source artifacts
  (transcript, segments, voice-ref); then fan Steps 4,6,6.5,7 per language. Meter per
  target-language-minute (the marginal cost is translation+TTS+merge only).
- **Effect [projection]:** for k languages, GPU-COGS ≈ source-side + k·(per-language) instead
  of k·(everything) — saves (k−1)·28 GPU-s + (k−1) Whisper/Demucs cold starts. Larger the
  more languages. This is also the natural pricing unit (plan §4.3).
- **Benchmark:** 1-lang vs 3-lang job GPU-seconds.
- **Rollback:** single-language endpoint stays; multi is additive.

### P6 — Fan out (or at least warm) the VC stage for the premium tier
- **Current:** `convert_segments_timbre` is a serial per-segment GPU loop
  (`pipeline/voice_conversion.py:228`), called in the orchestrator subprocess
  (`run_headless.py:304`). The fan-out note (`deploy/modal_app.py:428`) claims "knn-vc fans
  out the same way" — **not implemented**.
- **Problem:** the highest-margin tier (`mode=vc`) adds a second serial GPU loop after TTS;
  knn-vc also reloads WavLM+HiFiGAN per job (`pipeline/voice_conversion.py:112`).
- **Change:** give VC the same treatment as TTS — a warm `@modal.enter` model load in a Cls,
  and shard segments. Or fold VC into the TTSEngine container so the reference-matching set
  and models stay resident.
- **Effect:** premium-tier latency and COGS both improve; no quality change (same knn-vc).
- **Benchmark:** `mode=vc` run GPU-seconds before/after.
- **Rollback:** the serial path is the fallback (it already degrades to pass-through safely).

### P7 — Stage-level checkpoint so a retry does not re-run Steps 1–5
- **Current:** `dub_video` has `retries=Retries(max_retries=1)`
  (`deploy/modal_app.py:188`); only TTS is resumable (manifest). A failure at Step 6 that
  retries re-runs extract+Demucs+Whisper+translation from scratch.
- **Problem:** a retry pays the full Demucs+Whisper+translation GPU/latency again — expensive
  failure economics (prompt §36).
- **Change:** persist each stage's output artifact to the jobs Volume keyed by job_id
  (transcript.json, translated.json, source stems), and skip a stage whose artifact exists
  and whose input hash matches. The manifest pattern already used for TTS, generalized.
- **Effect:** a retry resumes at the failed stage; reliability + failure COGS improve.
  Quality unchanged.
- **Benchmark:** kill a job at Step 6, retry, confirm Steps 1–5 are skipped.
- **Rollback:** delete the artifacts / ignore them; behavior reverts to full re-run.

### P8 — Signed-URL upload; keep large media out of the gateway
- **Current:** `create_dub` does `await upload.read()` — the whole file into the ASGI
  container's RAM (`deploy/api.py:199`) — then writes it to the jobs Volume.
- **Problem:** a 500 MB upload is 500 MB of gateway RAM; the gateway is `min_containers=1`
  always-on (`deploy/modal_app.py:409`). This is the scalability ceiling the plan's §2.2 and
  prompt §19 both call out.
- **Change:** `POST /v1/dub` returns a signed R2/S3 PUT URL + job_id; client uploads directly
  to object storage; the worker reads from there and writes the output back for a signed GET.
  The ffprobe duration gate moves to a tiny "register" call (probe by range/head or a quick
  server-side probe after upload) so it still gates before GPU.
- **Effect:** gateway memory O(1) per request; unbounded input size; multi-region download.
  Latency: upload no longer double-hops through the gateway.
- **Benchmark:** concurrent large uploads — gateway RAM flat vs today's linear.
- **Rollback:** keep the multipart path for small (<N MB) inputs; it is a good fast path for
  short audio.

### P9 — Content-hash idempotency + cross-job artifact cache
- **Current:** idempotency is keyed only on a client-supplied `Idempotency-Key` header
  (`deploy/api.py:94`). No dedupe if the client omits it.
- **Problem:** the same file re-uploaded without a key runs and bills twice (prompt §17).
- **Change:** derive a job identity from `sha256(normalized_input) + target_lang + mode +
  model_version + nfe`. Same identity → return the existing job/result. This also becomes the
  key for a cross-job **result cache** (the same clip dubbed to the same language is served
  from storage, zero GPU).
- **Effect:** eliminates duplicate GPU runs; big win for retry-happy SDKs and popular content.
  Quality unchanged (identity includes everything that changes output).
- **Benchmark:** submit the same file twice; second is a cache hit, 0 GPU-s.
- **Rollback:** fall back to header-only idempotency.

### P10 — Persist the translation cache on Modal — ✅ LANDED
- **Current (before):** `translation_cache._cache_dir()` uses `os.getcwd()` /
  `./.dubbing_cache` (`pipeline/translation_cache.py:45`); `DUBBING_CACHE_DIR` was **not set**
  in the Modal env, and `getcwd()` on Modal is the *ephemeral* `REPO_MOUNT` (image layer,
  `copy=True`).
- **Problem:** the candidate cache — designed to "accumulate across runs" so re-runs cost 0
  API calls — was written to a throwaway filesystem and **lost on every container**. Its whole
  benefit was silently disabled on Modal.
- **Change (done):** `dub_video` now sets `DUBBING_CACHE_DIR=$CACHE_DIR/dubbing_cache`
  (`/cache/hf/dubbing_cache`, on the persistent HF Volume) and calls `hf_cache.commit()` right
  after the run so whatever Stage 4 added survives — even if a later stage failed.
- **Effect [measured, mechanism]:** with `DUBBING_CACHE_DIR` set, candidates written by one
  `TranslationCache` are read back by a fresh instance from the configured dir, and nothing
  leaks to `cwd/.dubbing_cache` (`tools/test_merge_and_cache.py`). On Modal this makes
  repeat/similar content translate from cache (0 API calls, no 503 variance), directly
  shrinking the P1 problem. Zero quality change (same candidates).
- **Benchmark (to run on Modal):** run the same clip twice; second run's Step-4 API-call
  count → 0.
- **Rollback:** unset the env var.

---

## 5. Modal configuration recommendations (deliverable D)

Grounded in the measured profile; treat as starting points to benchmark, not fixed values.

| knob | today | recommend | why |
|---|---|---|---|
| orchestrator GPU | GPU function | **CPU function** (P0) | 43% of GPU time is non-GPU work |
| source-side (Demucs+Whisper) | in GPU orchestrator | one **combined GPU fn** | amortize cold start; avoid per-stage container churn |
| `TTSEngine` gpu | L4 | L4 | fits IndicF5 (~1.3 GB) + headroom (plan §2.3) |
| `MODAL_TTS_MAX_CONTAINERS` | 8 | **2–4 Basic / 8 Pro** | R2: shards inflate GPU-seconds |
| `MODAL_TTS_WARM` | 0 | 0 until traffic justifies (§6) | warm = idle GPU billed |
| `MODAL_WARM` (orchestrator) | 0 | 0 (it is CPU after P0 — cheap to warm if needed) | |
| `fastapi_app min_containers` | 1 | 1 (CPU, cheap) — fine | keeps the API responsive |
| `dub_video timeout` | 30 min | keep | long videos |
| `retries` | 1 whole-job | 1 **after** stage-checkpoint (P7) | don't re-run Steps 1–5 |
| job store | `modal.Dict` | Postgres/Neon at scale (plan agrees) | Dict is fine for the scaffold |
| GPU type | L4 everywhere | L4; benchmark L4 vs A10G for TTS throughput/$ | prompt §33 — decide on throughput-per-dollar, measured |

**Warm-pool break-even (§6):** keep a warm TTS engine only when
`requests_per_hour × cold_penalty_saved > 3600 × warm_idle_fraction`. With the measured
28 s load and illustrative $0.0133/GPU-min, one always-warm L4 costs ~$0.80/hr; it pays for
itself only above roughly **~100 cold starts/hr** of saved 28 s load. Below that, scale to
zero. Recompute on the live Modal rate.

---

## 6. Cost model (deliverable H) — equations, not invented dollars

Let (all per job):
- `S` = source seconds; `L` = number of target languages; `n` = segments; `t_s` = mean
  synth s/segment (measured 15.8 at nfe32, ~linear in nfe); `sh` = shards; `load` = model
  load s (measured 28); `g` = GPU $/s (from the live Modal invoice — **do not guess**).

```
GPU-seconds (today, per job)      = idle_orch + demucs + whisper + load + n·t_s
                                    where idle_orch = extract + translate + refext + merge   ← P0 removes this
GPU-seconds (after P0, per lang)  = demucs + whisper + load + n·t_s                          (source-side shared across L → /L amortized: P5)
GPU-seconds (with fan-out)        = sh·load + n·t_s                                          ← R2: sh·load is the fan-out tax
GPU cost / job                    = g · GPU-seconds
External MT cost / job            = calls · price_per_call         → ~0 after P1/P10
Storage+egress / job              ≈ output_bytes · egress_rate
COGS / job                        = GPU cost + MT cost + storage/egress
Contribution / job                = price(S, L, tier) · (1 − rapidapi_fee) − COGS/job
```

Report **contribution / source-minute** and **GPU-seconds / source-minute**, per tier, from
real invoices. The plan's illustrative $0.04–0.06/video-min is a placeholder until `g` is
read from Modal; the structural result — P0 removes `idle_orch`, which is ~43% of today's
GPU-seconds — holds independent of `g`.

Pricing framework (fill from measured `g`): `price/target-lang-min ≥ COGS/target-lang-min /
(1 − target_margin) / (1 − rapidapi_fee)`. Meter on **target-language-minutes** (P5), because
that is the marginal cost unit.

---

## 7. Measurement harness & benchmark matrix (deliverable K)

**In the repo:** `tools/profile_log.py` turns any `pipeline_log.txt` into the §1 stage
table and TTS decomposition — CPU only, works on runs that already happened (CLAUDE rule 2:
every run leaves an artifact the next consumes). `tools/test_merge_and_cache.py` is the
property test behind the P2/P10 numbers — self-contained (synthesizes its own clip if no
sample is present), so the measured claims here regenerate on any machine with ffmpeg.
Point the profiler at a run:

```bash
python tools/profile_log.py dubbing_output_*/**/pipeline_log.txt
```

**To get the numbers this audit marks [projection], run these — smallest/cheapest first:**

1. **No GPU:** P2 FFmpeg copy-vs-burn timing; P10 cache-hit API-call count on a re-run.
   Both answerable on your Windows box now.
2. **One warm GPU container:** the nfe sweep (P3) via `pipeline/tts_benchmark.py` — the
   single highest-value GPU benchmark; it sets the Basic tier's cost floor.
3. **Modal, per-function billing:** P0 before/after GPU-seconds; P4 shard sweep {1,2,4,8}
   warm+cold; P5 1-lang vs 3-lang.
4. **Quality gates alongside every latency/cost number** (hard constraints, §8): isochrony
   score, IndicSBERT semantic, intelligibility CER, VC pass-through rate. An optimization
   that moves a quality gate is rejected regardless of its speed (CLAUDE rules 5, 7; skill
   `dubbing-model-training` Gate 2).

The benchmark matrix worth filling (rows = configs, cols = wall / GPU-seconds / p95 /
quality-gates): {nfe 8..32} × {shards 1,2,4,8} × {warm,cold} × {Basic,vc} × {Demucs on/off}
× {1,3 langs}. Start with the nfe row at shards=1 — it dominates COGS.

---

## 8. Quality guardrails (hard constraints — an optimization that breaks one is rejected)

Pre-registered, from the existing gates and the `dubbing-model-training` doctrine:

- isochrony score ≥ current (measured avg 0.937–0.944 **[measured]**) — no length win paid
  out of meaning (CLAUDE rule 7);
- IndicSBERT semantic ≥ the per-language gate;
- intelligibility: back-transcription CER not worse than the ground-truth floor;
- VC: pass-through/failed rate not up (premium tier);
- A/V sync within tolerance after P2's stream-copy path.

Read **per language, first** — the failure modes here are silent (skill doctrine, CLAUDE
rule 5). None of P0–P10 changes *what* is synthesized except P1 (MT swap) and P3 (nfe down),
which are the two gated on a measured per-language quality comparison.

---

## 9. What NOT to do (anti-over-engineering, prompt §44)

- No Kubernetes / Kafka / Temporal / Redis cluster / service mesh. Modal + a job DB +
  object storage is the whole system. The measured bottlenecks are GPU placement and one
  network call, not orchestration throughput.
- No model swap before the nfe sweep — nfe down may buy most of the TTS speedup for free
  (P3), more cheaply than XTTS/CosyVoice integration.
- No wide fan-out as a default — it is a cost, not a saving (R2).
- No warm pool until the break-even traffic exists (§5).
- Do not optimize Demucs/Whisper by moving them to separate containers without measuring the
  cold-start tax (P0 caveat) — the combined source-side function is the likely answer.

---

## 10. Suggested sequence

1. **Today, no GPU:** P10 (cache env — one line), P2 (FFmpeg copy default), and move
   `profile_log.py` into `tools/`. Cheap, safe, immediate.
2. **First GPU session:** P3 nfe sweep (sets the cost floor) + P0 CPU-orchestrator split
   (removes the 43% idle). These two are the bulk of the COGS win.
3. **Then:** P4 shard right-sizing, P7 stage checkpoints, P5 multi-language, P1 IndicTrans2
   (gated on quality), P6 VC fan-out.
4. **Before paid traffic:** P8 signed-URL upload, P9 content idempotency, and the plan's
   §4.5 licensing checks (IndicF5 + knn-vc commercial resale) — a legal blocker no
   optimization can fix.
```
