# Roadmap — TTS alternatives & the live-API endgame

Forward-looking scope. Two independent tracks the user asked to scope while the Kaggle
batch validation runs:

- **A. Alternative open-source Indic TTS** — a lower-latency / better-performing swap for
  IndicF5 at Step 6, subject to a hard *compatibility* filter (must cover the 11 Indic
  languages) and the isochrony requirement.
- **B. The live-API endgame** — a monetizable hosted API doing GPU Whisper (Step 2/3) +
  GPU Indic TTS (Step 6), CPU frontend, RapidAPI-style distribution.

> Ground rule for this whole document (dubbing doctrine #8): **no metric appears here
> without a traceable source.** Latency/quality claims below are either cited or explicitly
> marked *to-be-measured on the validation harness* — the supervised batch run we just
> built is exactly that harness. Do not paste any of these into a README or pitch until a
> repo script regenerates them.

---

## A. Alternative open-source Indic TTS

### The three-way tension

You cannot pick a TTS on one axis. For dubbing, three pull against each other:

1. **Isochrony / duration control** — can the model be *told* to produce N milliseconds,
   and does it obey? (Dubbing non-negotiable #1: prove the control signal reaches the
   model; never infer it from "the audio sounded fine.")
2. **Latency** — wall-clock per segment on a free T4. Architecture dominates here:
   - *Non-autoregressive + explicit duration* (FastPitch family) → one forward pass, lowest
     latency, duration is a first-class input.
   - *Flow-matching / diffusion* (IndicF5) → an iterative denoise loop; latency ≈ linear in
     NFE steps (our `nfe_step`, default 32). This loop **is** the latency cost.
   - *Autoregressive* (Parler, IndexTTS, CosyVoice) → token-by-token; latency grows with
     output length, though modern AR models add streaming/caching to hide it.
3. **Compatibility** — does it actually cover the 11 languages (Assamese…Telugu)? This is
   the filter that eliminates most 2026 "best TTS" leaderboard winners, which are
   Chinese/English-first.

### Candidate table

| Model | Indic coverage | Duration-control mechanism | Arch / latency profile | Verdict for us |
|---|---|---|---|---|
| **IndicF5** (current) | 11/11 (native) | Implicit: target mel-frame count from duration | Flow-matching, NFE=32 loop | Baseline. Best coverage; latency = the NFE loop; freeze-prone (now supervised) |
| **AI4Bharat Indic-TTS** (Coqui `ForwardTTS` = FastPitch + HiFi-GAN) | 13 langs, MIT | **Global scalar only**: `length_scale` (a *model attribute*) scales all phoneme durations; hit a target-ms via a **two-pass measure-then-rescale**. No per-phoneme / per-call override in stock `inference()` | Non-AR, single forward pass — **lowest latency**, least hang-prone | Best latency + Indic-native, but duration control is **coarser than IndicF5's `fix_duration`** (see below). Benchmark decides |
| **Indic Parler-TTS** (AI4Bharat) | 20–21 langs | **Coarse only**: "speaking rate" via natural-language prompt, not a target-ms signal | Autoregressive | Great coverage & expressivity; weak duration lever. Fails non-negotiable #1 unless a real duration path is found |
| **IndexTTS-2** | **Not official** (zh/en/ja/es/ar in 2.5); Hindi only a community fork | **Best explicit control**: specify #tokens → duration, <0.03% token error (cited) | AR, "industrial/efficient" | Gold-standard duration control, but the *compatibility* filter fails today. Watch for official Indic support |
| **CosyVoice2-0.5B** | Chinese/English-first | Streaming rate control | AR streaming, ultra-low latency | Latency leader, but Indic coverage unproven — likely disqualified |
| **A2TTS** (arXiv 2507.15272) | Low-resource Indian langs | Paper claims duration handling | Emerging | Track it; too new to depend on |

### The honest read

- **No single model wins all three axes today.** IndexTTS-2 has the control we want but not
  the languages; Indic Parler / IndicF5 have the languages but weaker or costlier control;
  AI4Bharat Indic-TTS (FastPitch) is the best *architecture* for our exact problem
  (non-AR + explicit duration) and deserves a real head-to-head.
- The biggest **latency lever we already own** is IndicF5's `nfe_step`. Before swapping
  models, sweep it (32 → 24 → 16 → 8) and measure the quality/latency curve — a swap that
  buys 2× speed is uninteresting if `nfe_step=16` buys the same at near-zero cost.
- **Correction after reading the source (non-negotiable #1 — prove the control signal, do
  not assume it):** AI4Bharat FastPitch's duration control is *not* the clean per-segment
  target-ms knob the earlier draft implied. Coqui `ForwardTTS.inference(x, aux_input=…)`
  takes no duration argument; duration is scaled by a **global model attribute**
  `self.length_scale` inside `format_durations`
  (`o_dr = (exp(dr_log) − 1) · length_scale; o_dr[o_dr<1]=1; round`). To hit a target you
  must (a) synth once at `length_scale=1` to read the natural frame count, (b) set
  `model.length_scale = target_frames / natural_frames`, (c) re-synth — a **two-pass** call,
  with per-phoneme rounding + a ≥1-frame floor adding quantization on short segments. Real,
  provable Stage-2 control, but **coarser and slower-to-target than IndicF5's one-pass
  `fix_duration`** (which sets the total generated length directly). Net: the benchmark is
  really choosing between *IndicF5 — clean duration control, latency = the NFE loop we can
  now tune 32→8* and *FastPitch — fastest architecture, coarser two-pass duration control.*
  Do not pre-judge it; that is what the scorecard is for (non-negotiable #6).

### Validation protocol (use the harness we just built)

For each candidate, on the same fixed clip, before believing anything:

1. **Prove the control signal (non-negotiable #1).** Feed a 2.0 s target; assert output ≈
   2.0 s, not 4.0 s (direction test, non-negotiable #5). Inspect the call signature / attach
   a forward hook — do not trust that the kwarg was honored.
2. **Length adherence + slope** (non-negotiables in the eval doctrine): regress produced vs.
   requested duration over 0.6–1.4×. Slope→1.0 = obeys; →0 = ignores. IndexTTS-2's <0.03%
   token error is the bar to beat.
3. **Latency per segment on a T4**, at matched quality — report seconds/segment with the
   checkpoint and clip named.
4. **Intelligibility**: back-transcribe (Whisper) → CER against the ground-truth-audio
   floor. Per-language, read first (Dravidian ≠ Indo-Aryan schedules).
5. **Degradation behavior**: does it hang/OOM on a T4? (Our supervisor makes any candidate
   safe to trial — a wedge is killed + relaunched, not a frozen run.)

**Recommended next action for Track A:** once the batch validation is green, run the
`nfe_step` sweep on IndicF5 first (free, no new model), then benchmark AI4Bharat Indic-TTS
(FastPitch) as the primary alternative — it is the only candidate that is both Indic-native
and architecturally low-latency-with-explicit-duration.

### The scorecard harness (what runs the protocol above)

Two harnesses, not one — worth keeping the names straight:

- **Freeze-fix supervised batch** (already built): `pipeline/tts_supervisor` + `tts_worker`
  + `run_headless.py`. Validates the *pipeline end-to-end* and makes any TTS backend
  **safe to trial** — a wedge is SIGKILLed + relaunched-and-resumed, never a frozen run. So
  "make sure FastPitch doesn't hang like IndicF5" is already answered structurally: FastPitch
  runs inside the same worker and inherits the same protection, and being single-forward-pass
  non-AR it is far less wedge-prone than IndicF5's flow-matching loop to begin with.
- **Metrics scorecard** (`pipeline/tts_benchmark.py`, being built now): the model-agnostic
  math that turns on-disk synth artifacts into the four axes — **control-signal proof**
  (recorded by the backend at synth time, asserted here — never inferred), **length slope**
  (regress produced vs requested over 0.6–1.4×), **latency** (s/segment on the run device),
  **CER** (Whisper back-transcription vs input text). Every report is provenance-stamped
  (checkpoint, clip, `nfe_step`, n samples, data version) per non-negotiable #8. Pure/CPU
  except the Whisper step, so the math is unit-tested without a GPU.

Run order is the user's: **nfe sweep on IndicF5 (32→24→16→8) first**, each `nfe` a row on
the scorecard; **then** FastPitch (two-pass `length_scale`) as another row. The winner is
whatever the scorecard says — this is a benchmark, not a prior.

---

## B. The live-API endgame

**Goal (user's words):** a live hosted API that does Whisper audio transcription (Step 2/3)
and Indic TTS (Step 6) — the two GPU steps — producing isochrony-aware, semantically
faithful dubbed audio/video with low latency and high performance, later distributed on an
"API-as-a-service" marketplace (RapidAPI etc.).

**Chosen hosting direction (user's decision, recorded here):**
> Move **Step 6** to **Modal serverless GPU** (~50 free T4-hours/month, container isolation
> built in) with a **free always-on Streamlit Community Cloud** frontend. Best long-term
> always-on URL; the cost is a new platform to set up and wiring the Step-6 call across the
> network.

### Endgame hosting — DECIDED: B1 (both GPU steps on Modal)

**Confirmed by the user:** Whisper (Step 2/3) also goes on Modal as a **second GPU
function**, alongside Step 6. "Move only Step 6" was the phrasing; because Streamlit
Community Cloud is CPU-only, the Whisper GPU requirement puts Step 2/3 in the same home.
So: **two Modal GPU functions** (`transcribe`, `synthesize`) sharing the ~50 free T4-hrs,
with everything else on the free always-on CPU frontend. Rejected alternatives, recorded so
the choice is traceable:

| Rejected | Where Whisper would run | Why not |
|---|---|---|
| CPU `faster-whisper` on Streamlit Cloud | on the free frontend | `large-v3` on CPU is far too slow for a product |
| Third-party STT API | someone else's GPU | per-call cost + a dependency you don't control |

### Architecture (B1)

```
Streamlit Community Cloud (free, CPU, always-on URL)   ← the frontend + orchestrator
  ├─ Step 1  extract audio            (ffmpeg, CPU)
  ├─ Step 2/3 → Modal.transcribe()    (GPU: faster-whisper)      ── network call
  ├─ Step 4  isochrony translation    (Gemini API, no GPU)
  ├─ Step 5  voice-reference clip     (CPU)
  ├─ Step 6  → Modal.synthesize()     (GPU: IndicF5 / successor)  ── network call
  └─ Step 7  assemble + mux           (ffmpeg, CPU)
```

- **Modal gives us the freeze fix for free.** Each Modal GPU call runs in a fresh container
  with a per-call timeout; a wedged CUDA op → Modal kills the container → the retry gets a
  clean one. That is *exactly* the process-isolation model we just built for Kaggle — the
  supervisor's design (isolate → heartbeat/timeout → kill → relaunch-and-resume) maps 1:1
  onto Modal (`@app.function(timeout=…, retries=…)`, GPU per call). Keep per-segment resume
  (the manifest) so a container recycle mid-job doesn't redo finished segments.
- **The network boundary is the new work.** Step 6 currently returns local WAV paths; across
  the network it must return **bytes** (or a signed URL to Modal storage). Define a clean
  request/response contract: `{segments:[{start,end,text}], ref_audio (bytes), lang, nfe}`
  → `{wavs:[bytes], degraded:[idx]}`. The `forced_silence` degrade list already exists in
  the manifest — surface it in the response so callers know which segments were silence.
- **Cold starts**: Modal cold-start + model load is seconds-to-minutes on first hit. Use
  Modal's container idle-timeout / keep-warm for a paid tier; accept cold starts on free.

### Monetization path (RapidAPI-style)

1. Wrap the two Modal functions behind one thin HTTP API (FastAPI on Modal, or Modal web
   endpoints) with an auth key and per-call metering.
2. Tiers map naturally to the GPU cost: free (short clips, `nfe` low), paid (long clips,
   `nfe=32`, voice cloning). The `nfe_step` knob is literally a price/latency dial.
3. List on RapidAPI: they handle keys, quotas, billing; you expose the FastAPI schema.
4. **Never** send the user's own credentials/email through these services; keys live in
   Modal secrets / RapidAPI's vault.

### Phased plan

- **Phase 0 (now):** green the Kaggle batch validation — proves the pipeline + freeze fix
  end-to-end. *Gate: a real dubbed video out, degraded-segment count acceptable.*
- **Phase 1:** lift Step 6 into a Modal function behind the *existing* Streamlit app
  (local or Kaggle), calling Modal over the network. Validate the bytes contract + resume.
- **Phase 2:** lift Step 2/3 (Whisper) into a second Modal function (option B1).
- **Phase 3:** move the frontend/orchestrator to Streamlit Community Cloud (always-on URL).
- **Phase 4:** FastAPI wrapper + auth + metering → list on RapidAPI.
- **Track A folds in at Phase 1**: whichever TTS wins the benchmark is what Modal.synthesize
  loads — the Modal function is model-agnostic behind the bytes contract.

---

## Sources

- [SiliconFlow — Best Open-Source TTS Models 2026](https://www.siliconflow.com/articles/best-open-source-text-to-speech-models)
- [SiliconFlow — Best Open-Source AI Models for Dubbing 2026](https://www.siliconflow.com/articles/en/best-open-source-AI-models-for-dubbing)
- [IndexTTS2 paper (arXiv 2506.21619)](https://arxiv.org/abs/2506.21619) — explicit duration control, token-error figures
- [IndexTTS GitHub](https://github.com/index-tts/index-tts) — language support (zh/en/ja/es/ar in 2.5)
- [ai4bharat/indic-parler-tts (HF)](https://huggingface.co/ai4bharat/indic-parler-tts-pretrained) & [AI4Bharat TTS](https://ai4bharat.iitm.ac.in/areas/tts/) — Indic coverage, prompt-based rate control
- [AI4Bharat/Indic-TTS (GitHub)](https://github.com/AI4Bharat/Indic-TTS) — FastPitch + HiFi-GAN, 13 langs, MIT, built on Coqui TTS (`TTS.bin.synthesize`)
- [Coqui `ForwardTTS` source](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/models/forward_tts.py) — `inference()` signature + `format_durations` (`length_scale` is the only duration lever)
- [A2TTS: TTS for Low Resource Indian Languages (arXiv 2507.15272)](https://arxiv.org/pdf/2507.15272)
