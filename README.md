<div align="center">

# 🎬 Indic AI Dubbing Platform

### Duration-accurate, meaning-preserving video dubbing for **all 11 major Indic languages**

*Separate the voice · transcribe · translate to fit the mouth · clone the speaker · remix*

![Version](https://img.shields.io/badge/version-2.5-6c5ce7)
![Languages](https://img.shields.io/badge/languages-11%20Indic-00b894)
![Phoneme%20Ruler](https://img.shields.io/badge/phonemes-espeak--ng%20G2P-0984e3)
![Semantic%20Gate](https://img.shields.io/badge/semantics-IndicSBERT-e17055)
![Validated](https://img.shields.io/badge/live%20validated-44%2F44%20segments-2d3436)

Hindi · Bengali · Marathi · Gujarati · Punjabi · Tamil · Telugu · Kannada · Malayalam · Odia · Assamese

</div>

---

## ✨ What is this?

A full **video-to-video dubbing pipeline**. You give it an English video; it returns the same video speaking a target Indic language — with the **background audio preserved**, the **speaker's voice optionally cloned**, and, crucially, the translated speech **timed to fit the original mouth movements** instead of running long or short.

The hard problem in dubbing is not translation — it is **isochrony**: making the dubbed line take the *same amount of time* to say as the original, while still *meaning the same thing*. Those two goals fight each other. v2.5 is built around resolving that fight explicitly.

---

## 🚀 What changed from v2 → v2.5 (and why the dubbing is better)

v2 could already translate "close to" the right length. But it measured length with a **rough syllable heuristic**, and it had **no way to know whether a shorter rewrite still meant the same thing** — so it sometimes shipped a line that fit the timing but subtly drifted in meaning, or a line that was faithful but overran the shot. v2.5 fixes both, and makes every failure *visible* instead of silent.

| # | v2 behaviour | v2.5 behaviour | Effect on the dub |
|---|---|---|---|
| **1. Length measurement** | Counted **syllables** with a vowel-group heuristic (`~0.43×` off the truth, with a language-dependent bias). | Counts **real phonemes** with **espeak-ng G2P** (`phonemes:espeak-ng-1.50`) + IndicNLP orthographic normalization. | The timing budget is now grounded in how the words are *actually pronounced*, per language — so lines land on the shot instead of near it. |
| **2. Meaning check** | **None.** Any grammatical translation was accepted. | **IndicSBERT cross-lingual semantic gate** (cosine ≥ **0.70**) scores every candidate against the English source *before* length is even considered. | Short rewrites that quietly change the meaning are now **rejected**, not shipped. Faithfulness is enforced, not hoped for. |
| **3. How a line is chosen** | Single generation, pick what comes back. | **Iterative CoT loop**: generate **3 candidates/segment**, keep only those that clear the meaning gate, then pick the one whose **phoneme count is closest** to the duration budget; refine the ones that miss for up to **3 rounds**. | Two-stage selection (*mean it first, then fit it*) instead of one-shot luck. |
| **4. When it can't win** | Failed silently — you couldn't tell a good line from a compromised one. | **Degrades visibly**: keeps the best candidate ever seen, reports `gates_passed=False`, and stamps the active ruler (`espeak-ng` vs `heuristic-fallback`) so you always know how a line was measured. | You can trust a "pass" and *see* every compromise, which makes QA on a long video tractable. |
| **5. Non-Devanagari scripts** | Prior embedder (MiniLM) collapsed 8/11 languages to a meaningless ~0.99 similarity — effectively blind to Tamil, Telugu, Bengali, etc. | IndicSBERT gives **discriminating** scores across **all 11 scripts** (e.g. Tamil, Telugu, Odia land in a real 0.72–0.94 range, not a flat 0.99). | Meaning-preservation actually works for the Dravidian and eastern languages, not just Hindi/Marathi. |

**In one sentence:** v2.5 stops guessing at length and stops ignoring meaning — it measures pronunciation for real, enforces a meaning floor, and searches candidates to satisfy *both*, telling you honestly whenever it can't.

### The selection objective

When a candidate clears the semantic gate, the loop ranks it by a combined cost that trades meaning against fit, and tracks the global best:

```
cost = 0.60 · (1 − semantic_similarity)  +  0.40 · |phonemes − ideal| / ideal
        └── meaning (IndicSBERT) ──┘        └──── timing fit (espeak-ng) ────┘

gates passed  ⇔  semantic ≥ 0.70  AND  |phonemes − ideal| / ideal ≤ 0.15
```

### ✅ Live validation (v2.5)

Validated end-to-end on the real deployment target (Kaggle CPU), **all 11 languages × 4 sentences = 44 segments**:

| Metric | Result |
|---|---|
| Real espeak-ng phoneme ruler active | ✅ `phonemes:espeak-ng-1.50` (not the fallback) |
| IndicSBERT scored every segment | ✅ 44/44 — Devanagari **and** all 8 non-Devanagari scripts |
| Non-empty translations | ✅ 44/44 |
| Average semantic similarity | **0.838** |
| Average isochrony score | **0.934** |
| Both gates passed | **42/44** — the 2 misses honestly reported as *too-short at the loop's global minimum* |

---

## ⚙️ Pipeline architecture (v2.5)

```
Input Video
    │
    ▼
Stage 1 · Source Separation  ── Demucs (htdemucs)
    ├── vocals.wav      → transcription & TTS reference
    └── background.wav  → preserved for the final remix
    │
    ▼
Stage 2 · Transcription  ── Faster-Whisper large-v3 (INT8)
    └── timestamped English segments
    │
    ▼
Stage 3 · Isochrony-Aware Translation  ★ v2.5 core ★
    │   Gemma-bulk / Gemini-refine hybrid generates 3 candidates/segment
    │   (throttled + disk-cached to stay under free-tier rate limits)
    │      ├─ Gate A · MEANING   → IndicSBERT cross-lingual cosine ≥ 0.70
    │      └─ Gate B · TIMING    → espeak-ng phoneme count vs duration budget (±15%)
    │   Iterative CoT: refine misses up to 3 rounds, keep global best, report honestly
    └── Indic text that means the same AND fits the shot
    │
    ▼
Stage 4 · Voice Manager  ── SNR analysis (Premium)
    └── voice_reference.wav (12s clean speech clip)
    │
    ▼
Stage 5 · Duration-Controlled TTS  ── IndicF5 (AI4Bharat)
    ├── Basic:   pre-selected natural voices
    └── Premium: zero-shot voice cloning from reference clip
    │
    ▼
Stage 6 · Audio Assembly
    ├── Direct timeline overlay (no FFmpeg atempo stretching)
    ├── 50ms crossfades between segments
    └── Background + dubbed vocal mix (LUFS normalized)
    │
    ▼
Stage 7 · Video Composition  ── FFmpeg
    └── final_dubbed_video.mp4 with burned subtitles
```

---

## 🏁 Quick start

### 1. Activate the virtual environment
```powershell
.\dubbing_app\Scripts\activate
```

### 2. Install dependencies
```powershell
# Core pipeline
pip install faster-whisper demucs soundfile pyrubberband resampy

# v2.5 translation core
pip install phonemizer indic-nlp-library sentence-transformers google-genai
```
> **espeak-ng is a system package, not a pip wheel.** The real phoneme ruler needs it.
> Linux: `apt-get install -y espeak-ng` · macOS: `brew install espeak-ng` · Windows: install the espeak-ng MSI.
> Without it the counter degrades **visibly** to `chars:heuristic-fallback` — it never lies about which ruler it used.

```powershell
# IndicF5 (from source — required for duration-controlled TTS)
pip install git+https://github.com/ai4bharat/IndicF5.git
# Accept the gate at https://huggingface.co/ai4bharat/IndicF5, then:  huggingface-cli login
```

### 3. Set your Gemini key and run
```powershell
$env:GEMINI_API_KEY = "your-key-here"
streamlit run app.py
```

---

## 🔧 Configuration knobs

| Setting | Where | Default | Purpose |
|---|---|---|---|
| Semantic threshold | `SEMANTIC_THRESHOLD` | `0.70` | Meaning floor a candidate must clear. |
| Phoneme tolerance | `PHONEME_TOLERANCE` | `0.15` | Allowed relative gap from the ideal length. |
| Candidates/segment | `N_CANDIDATES` | `3` | Breadth of the per-round search. |
| Refinement rounds | `MAX_ITERATIONS` | `3` | How hard the loop tries before reporting the best. |
| Objective weights | `SEMANTIC_WEIGHT` / `PHONEME_WEIGHT` | `0.6` / `0.4` | Meaning-vs-timing trade-off. |
| Embedder override | `DUBBING_SBERT_MODEL` (env) | `l3cube-pune/indic-sentence-similarity-sbert` | Swap the IndicSBERT model. |

### 🚦 Rate-limit knobs (v2.5.1 — hybrid routing, throttle & cache)

`gemini-3.1-flash-lite` has a tight free-tier limit (per-minute *and* per-day). To avoid 429s and cut API calls, the translation stage now **routes the bulk iteration-0 batch through the more lenient Gemma models** and reserves Gemini for the smaller refinement rounds, paces every model with a client-side throttle, honours the server's `retryDelay` on a 429, and **caches candidate pools to disk** so a re-run of the same video selects its final lines with *zero* API calls (selection is local and free).

| Setting | Env var | Default | Purpose |
|---|---|---|---|
| Bulk (iteration-0) model | `DUBBING_GEMINI_BULK_MODEL` | `gemma-4-31b` → `gemma-4-26b` | High-volume first pass on the lenient Gemma limits. Falls through the Gemini ladder if Gemma is unavailable. |
| Refine model | `DUBBING_GEMINI_REFINE_MODEL` / `DUBBING_GEMINI_MODEL` | `gemini-3.1-flash-lite` | Used only for refinement rounds on hard segments. Falls back to 2.5/2.0/1.5-flash. |
| Requests-per-minute cap | `DUBBING_GEMINI_RPM` | `30` (Gemma) / `15` (Gemini) | Client-side throttle: min interval between calls = `60/RPM`. |
| Requests-per-day cap | `DUBBING_GEMINI_RPD` | *(unset)* | Optional hard per-model daily cap; a capped model is skipped so the job degrades with a clear error instead of hammering 429s. |
| Cache directory | `DUBBING_CACHE_DIR` | `./.dubbing_cache` | Where the candidate pool + daily-usage counter live. |
| Candidate cache path | `DUBBING_TRANSLATION_CACHE` | `<cache_dir>/translations.json` | Full override for the candidate-pool file. |
| Cache on/off | `use_cache` param | `True` | Pass `use_cache=False` to `translate_segments_isochrony` to bypass the disk cache for a run. |

> **Gemma on the Gemini API doesn't support structured output (`response_schema`).** The stage detects a Gemma model and parses JSON from plain text instead (fence-strip + balanced-bracket match), so the hybrid routing is transparent to the caller. The candidate cache **accumulates across runs** — quality only improves, and repeated phrases within a single video are served from cache after the first generation.

---

## ☁️ Deployment on Kaggle (free GPU)

Run the pipeline on Kaggle's free dual T4 (2×16 GB VRAM), expose it via Cloudflare Tunnel or ngrok, and drive it from your local machine.

1. Create a Kaggle notebook with the GPU accelerator enabled.
2. Install deps (`python kaggle/build_notebook.py` regenerates the bundle from the canonical pipeline source).
3. Run `cloudflared tunnel --url http://localhost:8501`.
4. Open the Streamlit UI at the tunnel URL.

> Kaggle gives 30 GPU-hours/week free; dual T4 = 32 GB VRAM total.
> The **semantic gate runs on CPU by design** — force `CUDA_VISIBLE_DEVICES=""` for IndicSBERT, since the GPU lottery can hand out a Tesla P100 (sm_60) that the installed torch build has no kernels for.

---

## 💰 Pricing reference

| Tier | Features | Price/min |
|---|---|---|
| Basic | Audio dubbing, generic voices | $0.02 |
| Premium | Audio dubbing + voice cloning | $0.05 |

Self-hosted COGS on T4: ~$0.004/min (87%+ gross margin at Basic price).

---

<div align="center">
<sub>Built for accurate, faithful, natural-sounding Indic dubbing — measure the pronunciation, protect the meaning, fit the shot.</sub>
</div>
