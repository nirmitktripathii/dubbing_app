# IndicAI Dubbing — Production Infrastructure & Monetization Plan

Status: draft for review · 2026-09-16
Scope: (1) why the primer run is garbled and how to actually fix voice cloning,
(2) an optimized low-latency production architecture, (3) the Modal + Streamlit +
RapidAPI API-as-a-Service and monetization plan.

---

## Part 1 — Why the primer run is garbled (and the primer is the wrong fix)

### 1.1 The controlled experiment already isolated the cause

Two runs, same video, same pipeline, same `nfe_step=32`. The **only** difference is the
TTS reference, and it is visible in the two job specs:

| | Basic run (`dubbing_output_hi_ref`) | Primer run (`dubbing_output_20260915_163400`) |
|---|---|---|
| `reference_audio_path` | `null` | `…/voice_reference.wav` (the **English** speaker) |
| `reference_text` | `null` | `"So we cut the top off of three plastic bottles…"` (**English**) |
| primer | off | `"नमस्ते।"`, budget 1.0 s, sliced after synth |
| result | **clean** | **garbled / babbling** |

Basic mode passes `reference=None`, so IndicF5 synthesizes Hindi on its own built-in
**native Hindi** prompt — in-language, and clean. The primer run conditions IndicF5 on
**English audio + English text** and asks it to generate Hindi. IndicF5 is an Indic-only
F5-TTS; a cross-lingual reference destabilizes its flow-matching alignment across the whole
utterance. **That cross-lingual conditioning is the cause of the babble — not timing, not
the primer slicing, not the overlay FS.** Timing is provably fine (all segment drift
<500 ms; isochrony avg 0.94).

### 1.2 The acoustic fingerprint (measured, not guessed)

Onset analysis of all 13 segments in both runs (`scratchpad/onset_analyze.py`):

- **Basic run:** every segment opens with **175–450 ms of leading silence**, `onset_rms /
  median ≈ 0.0–0.5` — silence → smooth ramp into voiced speech. A natural attack.
- **Primer run:** leading silence ≈ **0 ms** on most segments, `onset_rms / median ≈
  1.0–1.5` — audio starts at **full energy from frame 0**. No silence, no ramp.

That 0-ms/full-energy onset is the signature of the primer being sliced **mid-stream**: the
RMS-trough slicer cannot reliably find the pause after `नमस्ते।`, so it either leaves primer
residue or amputates the first real phone. The log shows the slicer's cut points scattered
across **0.31 s → 1.57 s**, with several "no clear pause → cut at budget" and two cuts at
`trough_rms=0.0000` (it found leading silence, not the post-primer pause).

### 1.3 Why the primer approach cannot be rescued

The primer assumed onset babble is a *bounded warm-up transient at t=0* that a throwaway
word absorbs. It is not. F5-TTS generates the entire mel-spectrogram jointly conditioned on
(ref_audio, ref_text, gen_text, fix_duration). With an English reference the duration/phone
alignment is mismatched **everywhere**, so the instability recurs throughout the utterance,
not just at the start. Prepending a Hindi word gives no "clean runway," and even the part it
does absorb can't be cut cleanly (§1.2). **Retire the primer.**

### 1.4 The real fix for voice cloning: decouple content from timbre

Keep the thing that already works (native-reference IndicF5 Hindi — clean and
isochrony-controlled) and add timbre transfer as a **separate** stage:

```
English video ─▶ … ─▶ IndicF5 (native Hindi ref, NO cross-lingual) ─▶ clean Hindi speech
                                                                         │
                                             Voice Conversion (speaker = original English) ─▶ cloned Hindi
```

- **Stage A — TTS:** IndicF5 with `reference=None` (or a native Hindi seed voice). Proven clean.
- **Stage B — Voice conversion (VC):** convert timbre of the clean Hindi audio to the source
  speaker, using the English `voice_reference.wav` as the *speaker target*. VC transfers
  identity while preserving the (already-correct Hindi) content, so there is no cross-lingual
  phonetic conditioning to destabilize.

This is how cross-lingual dubbing is done in practice (say-it-right in the target language,
then move the timbre). Candidate VC models, cheapest-integration first:

1. **seed-vc** (zero-shot, one reference clip, good cross-lingual timbre) — recommended first try.
2. **knn-vc / FreeVC** (lightweight, fast, permissive licenses).
3. **RVC** (excellent quality but needs a short per-speaker train; better for "saved voices").
4. **OpenVoice v2** (tone-color conversion, designed exactly for this decoupling).

Fallback if VC quality is insufficient: swap the TTS to a model built for cross-lingual
cloning (**XTTS-v2**, **CosyVoice 2**, or **Sarvam Bulbul**) — but IndicF5's *native* Hindi
quality is a high bar, so native-IndicF5 + VC is the first bet.

**Action:** add `DUBBING_VOICE_CLONE=2` = "native TTS + VC" as the premium path; keep
`0` (Basic) shippable today; delete the `1` (cross-lingual primer) path once VC lands.

---

## Part 2 — Production architecture

### 2.1 The honest latency picture

Current run: **7.8 min wall for an 80 s video**. Breakdown from the log:

| stage | time | notes |
|---|---|---|
| audio extract | ~1 s | |
| Demucs separation | ~12 s | GPU |
| Whisper medium | ~16 s | GPU; cold-loads the model |
| Gemini isochrony translation | **~150 s** | 3 refine iters + audit; heavy 503 retry waste; +30 s IndicSBERT cold load |
| voice ref extract | ~2 s | |
| **IndicF5 TTS** | **~235 s** | 13 segments **serialized**, ~18 s each |
| assembly + FFmpeg burn | ~45 s | |

"Little to no latency" for a synchronous HTTP call is **not physically achievable** — this
is a multi-stage GPU pipeline. The correct product shape is an **async job API** (submit →
poll or webhook). What we *can* do is cut wall-clock hard and keep it predictable:

- **Parallelize TTS** (the big one): `.map()` segments across warm containers → **235 s → 25–40 s**.
- **Warm model pool** (no cold loads): keep 1 GPU container warm → save ~30–60 s/run.
- **Kill Gemini retry waste:** the log shows minutes lost to `503` on `gemini-3.1-flash-lite`;
  pin a higher-availability tier + cut candidates 3→2 → save 60–90 s.
- **Overlap stages:** Demucs ∥ nothing (first), but translation of the transcript can start
  while voice-ref extraction runs; TTS can begin on segment 0 as soon as its translation gates.

Realistic target after optimization: **~60–90 s for a 1-minute clip**, models warm, fully
async. That is the number to advertise — not "real-time."

### 2.2 Component map

```
                         ┌──────────────────────────────────────────────┐
   Direct users ──────▶  │  Streamlit UI  (upload, lang, Basic/Clone,    │
                         │  preview, download)   — Modal-hosted           │
                         └───────────────┬──────────────────────────────┘
                                         │ same REST API
   API customers ─▶ RapidAPI Hub ─▶ ┌────▼─────────────────────────────────┐
   (keys, quotas, billing)          │  FastAPI gateway  (Modal @asgi_app)   │
                                     │  POST /dub  GET /dub/{id}  webhook    │
                                     └────┬──────────────────────────────────┘
                                          │ enqueue job_id, upload source ▶ R2/S3
                                     ┌────▼──────────────────────────────────┐
                                     │  Modal orchestrator  (CPU fn)         │
                                     │  fans out over GPU stage functions:   │
                                     │   extract→demucs→whisper→translate    │
                                     │   →voiceref→TTS(.map fan-out)→VC→mux   │
                                     └────┬──────────────────────────────────┘
                                          │ writes result ▶ R2/S3, updates job row (DB)
                                     ┌────▼───────────────┐
                                     │  Job store (SQLite  │  status, cost, result URL
                                     │  on Modal Volume,   │
                                     │  or Postgres/Neon)  │
                                     └────────────────────┘
```

### 2.3 Why Modal fits

- **Serverless GPU, scale-to-zero, per-second billing** — pay only while dubbing.
- **Warm pools** (`min_containers=1`) — kill cold-start on the hot path.
- **`.map()` fan-out** — the free win on serialized TTS.
- **Weights baked into the image / on a Volume** — no per-run model download.
- **`@modal.asgi_app()`** hosts the FastAPI gateway *and* the Streamlit UI in the same project.

IndicF5 is tiny (used ~1.3 GB VRAM on a T4 per the log). Whisper-medium + Demucs + IndicF5 +
VC all fit comfortably on **one L4 (24 GB)** — no A100 needed. That keeps COGS low (§4.2).

Alternative if you want a persistent box (steady load, no cold-start ever): RunPod/Lambda
GPU pod running the same containers. Modal is the recommended default for bursty API traffic.

---

## Part 3 — Studio-quality output

Timing and translation are already studio-grade. The gaps are all post-TTS mastering:

1. **Voice cloning done right** — Part 1.4 (native TTS + VC). Non-negotiable for "premium."
2. **Higher fidelity render for final** — `nfe_step` 32 (baseline) → **48–64** for the paid
   tier; measurably smoother, ~1.5–2× TTS time (absorbed by fan-out).
3. **Loudness mastering** — replace flat `-20 dBFS` peak norm with **EBU R128** (−16 LUFS
   integrated for web) and a true-peak limiter at −1 dBTP.
4. **Background ducking** — replace the static `bg_vol=0.3` with **sidechain compression**
   (duck the music under vocals), so speech stays intelligible and the bed still breathes.
5. **TTS cleanup** — light denoise + de-ess on synthesized speech before the mux.
6. **Lip-sync (stretch goal)** — the pipeline is isochronous, not phoneme-lip-aligned; a
   Wav2Lip/LatentSync pass is the eventual "studio" ceiling. Defer until VC lands.

---

## Part 4 — API-as-a-Service & monetization

> Provenance / reconciliation: the earlier "Modal + Streamlit + RapidAPI" plan **has now
> been located** — it lives in [`docs/roadmap-tts-and-live-api.md`](pipeline_v3/.claude/worktrees/dubbing-phoneme-counter-v2-709124/docs/roadmap-tts-and-live-api.md)
> (created 2026-09-08/09, session `93f64cd0`) and was refined in session `888bbb87`
> (2026-09-14). This reconstruction matched it on **every architectural decision** (async job
> API, RapidAPI-as-storefront + proxy-secret check, TTS `.map()` fan-out, weights baked into
> the image, right-sized GPU per stage, warm pools as a tier dial, stateless Volume storage +
> manifest-resume, meter on audio-minutes, per-tier length caps, and the
> `DUBBING_VOICE_CLONE`-branch tier ladder with **basic shipped first / cloning gated behind
> validation**). Three things the prior plan treated as first-class were **missing here and
> are now added below**: get-off-Gemini (§4.4), licensing + cloning consent (§4.5), and the
> `nfe`-down sweep as a COGS lever (§4.6). One deliberate difference: the prior plan withheld
> all dollar figures (doctrine #8 — no metric without a source); §4.2's numbers are now
> partly grounded in the 2026-09-15 run log but pricing/$-per-GPU-hr remain **illustrative
> estimates**, flagged as such.

### 4.1 The stack and the split of responsibilities

- **RapidAPI Hub** = the storefront + metering. It issues API keys, enforces per-plan
  quotas/rate-limits, handles billing and payouts, and injects `X-RapidAPI-Proxy-Secret`
  and `X-RapidAPI-User` on every proxied call. You verify the secret in FastAPI so the
  endpoint can't be called around RapidAPI. **You do not build billing.**
- **FastAPI (on Modal)** = your real API: `POST /dub`, `GET /dub/{id}`, optional webhook.
  Async: returns `job_id` immediately, does the work in the Modal orchestrator, meters
  **video-minutes processed**.
- **Streamlit (on Modal)** = the human product for non-developers (and your own demo/sales
  tool). Same backend.
- **Modal** = the compute (Part 2).

### 4.2 Unit economics (compute term measured; prices illustrative)

**Measured (2026-09-15 run, doctrine #8):** 80 s video → **~235 s IndicF5 on one GPU**
across 13 segments (~18 s/segment); TTS is ~50% of the ~7.8 min wall and the dominant
compute term. That is ~2.9 GPU-min of TTS per video-minute *serialized*; fan-out cuts
wall-clock, not GPU-seconds, so the COGS term is set by total GPU-seconds regardless.

**Illustrative (verify against live Modal pricing — these $ are estimates, not measured):**
one L4 ≈ **$0.80/hr** ⇒ **$0.0133 / GPU-minute**. At ~2–3 GPU-min of compute per
video-minute (TTS-dominated, before `nfe`-down tuning in §4.6) ⇒ **~$0.03–0.04 GPU COGS /
video-min**. Add Gemini (~$0.001–0.005, → ~$0 once self-hosted per §4.4) and storage/egress
(~$0.005) ⇒ **all-in COGS ≈ $0.04–0.06 / video-minute** at today's `nfe=32`; the `nfe` sweep
(§4.6) is the lever that pulls this down.

Suggested pricing (meter = processed video-minute; **price points are placeholders** pending
a competitor scan and the measured COGS above):

| plan | who | voice | price | gross margin |
|---|---|---|---|---|
| **Free** | trial | Basic (native) | 5 min/mo, watermarked | — (funnel) |
| **Basic** | creators | Basic (native) | ~$0.20 / video-min | ~80–85% |
| **Pro** | studios | **Cloning + nfe 48** | ~$0.75–1.50 / video-min | ~90%+ |
| **Enterprise** | volume | Cloning + SLA | custom / committed | highest |

RapidAPI takes ~20% of revenue. Even at Basic, COGS $0.05 vs price $0.20 → healthy. The
cloning tier is where the margin and the moat are — which is exactly why Part 1.4 (making
cloning actually work) is the gating item for monetization.

### 4.3 Metering, safety, idempotency

- Meter on **actual processed duration** from ffprobe, not the requested value.
- Enforce a **max input length** per plan (protects GPU quota; mirrors CLAUDE rule 4 thinking).
- **Idempotency key** on `POST /dub` so a client retry doesn't double-bill / double-run.
- Reject on cost ceiling before any GPU loads (the API twin of the CPU-preflight rule).

### 4.4 Get off the free Gemini key before you charge for translation

Step 4 (isochrony translation) currently calls a **free-tier Gemini key**. For a *paid*
product this is a blocker on two axes, and the earlier roadmap flagged both:

- **Scaling:** a free key rate-limits under real concurrent load — the exact 503 storm the
  2026-09-15 log already shows on a *single* run (§2.1) becomes constant under many callers.
- **Terms of service:** reselling output generated from a free Gemini key is almost certainly
  against its ToS. You would be building revenue on a dependency you are not licensed to
  resell.

The production path is **self-hosted IndicTrans2 on the same Modal GPU** — no per-call cost,
no RPM ceiling, no external dependency, and it fans out and batches exactly like TTS. Keep
Gemini as a bootstrap/fallback, but **gate the switch on a real quality comparison**
(IndicTrans2 vs Gemini on the isochrony + semantic axes, per language) — that is a genuine
gate, not a formality. This lands with the premium-tier work, not the Basic MVP.

### 4.5 Licensing and consent — the cheapest checks that de-risk the most

Two legal items must clear **before** money changes hands, especially on the cloning tier:

1. **IndicF5 commercial license.** Confirm AI4Bharat's IndicF5 (and any successor TTS, e.g.
   the VC models in Part 1.4) permit commercial resale of generated audio. This costs an hour
   and can invalidate the whole business if it comes back "research-only" — do it first, in
   parallel with the build (no GPU needed). Same check for the chosen VC model (seed-vc /
   knn-vc / OpenVoice licenses vary).
2. **Voice-cloning consent.** The Pro/cloning tier reproduces a real person's voice. Require
   the caller to attest they have rights/consent to the source voice (ToS click-through +
   an API flag), and keep it out of scope for anonymous free-tier use. This is both a legal
   and a brand-safety guardrail.

### 4.6 `nfe_step` is a cost/latency dial, not just a quality knob

Part 3 raises `nfe_step` 32→48–64 for the paid tier (quality **up**). The inverse is a live
**COGS lever** the earlier roadmap called out: IndicF5 latency ≈ linear in `nfe_step`, so
sweeping it **down** (32 → 24 → 16 → 8) and measuring the quality/latency curve on the
benchmark harness may buy most of the TTS speedup at near-zero cost — possibly more cheaply
than a model swap. Run the sweep (each `nfe` a row on `pipeline/tts_benchmark.py`) and let it
set the Free/Basic tier's `nfe`, while Pro keeps the high-fidelity setting. `nfe` becomes a
per-tier price/latency/quality dial.

---

## Part 5 — Recommended build sequence

1. **Fix cloning (Part 1.4).** Add native-TTS + VC path; validate on the same 80 s clip
   against the Basic run. *This unblocks the paid tier — do it first.*
2. **Modal-ify the pipeline.** Port the stage modules into a Modal app; bake weights into the
   image; add the TTS `.map()` fan-out and a warm pool. Target ~60–90 s/min, async.
3. **FastAPI gateway** on Modal (`/dub`, `/dub/{id}`, webhook) + job store + R2/S3.
4. **Streamlit UI** on Modal for direct users and demos.
5. **List on RapidAPI**, wire the proxy-secret check, ship Free + Basic; gate Pro on VC quality.
6. **Master for studio** (Part 3: R128, ducking, nfe 48, cleanup).

Parallelizable: the Basic-only path can go live on Modal (steps 2–5) **now**, while step 1
(cloning via VC) proceeds — Basic is already clean and shippable.

---

## Open decisions (need your call)

- **Cloning approach:** native-IndicF5 + VC (recommended) vs. swap to XTTS/CosyVoice/Sarvam
  vs. ship Basic-only for launch and add cloning later.
- **First thing to build:** fix the audio (VC) first, or scaffold the Modal/FastAPI/Streamlit
  infra first on the Basic path.
- **Compute host:** Modal (recommended) vs. a persistent RunPod/Lambda pod.
