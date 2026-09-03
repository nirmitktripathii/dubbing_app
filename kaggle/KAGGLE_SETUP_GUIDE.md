# 🎙️ Indic AI Dubbing Platform — Kaggle Deployment Guide (v2.5)

This guide tells you exactly how to run the full **v2.5** dubbing pipeline on Kaggle's free
T4 GPU and access it via a Cloudflare tunnel in your browser.

> **What's new in v2.5 (and what it means for this deployment)**
> The translation core was rebuilt. Instead of a syllable heuristic, it now counts **real
> phonemes** with **espeak-ng G2P**, and it gates every candidate through an **IndicSBERT
> cross-lingual meaning check (cosine ≥ 0.70)** before fitting it to the audio duration —
> looping up to 3 refinement rounds and reporting honestly when it can't satisfy both.
> Practically, that adds two dependencies to the notebook (`indic-nlp-library`,
> `sentence-transformers`), makes the **`espeak-ng` system package load-bearing**, and means
> the semantic gate runs on **CPU by design**. All 11 Indic languages are supported. See the
> [v2.5 notes](#-v25-notes-read-before-your-first-run) below.

---

## Step 1: Generate the Notebook (Local Machine)

`build_notebook.py` produces a **self-contained** Kaggle notebook that embeds the pipeline
code, written from the **canonical pipeline source** — so `phoneme_counter.py` and
`semantic_similarity.py` have a single source of truth and never drift from the app. Run it
once from your machine:

```powershell
# From the repo root
python kaggle\build_notebook.py
```

This creates **`indicai_dubbing_kaggle.ipynb`** (~155 KB) in `kaggle/`. Re-run it only when
you change any pipeline file.

---

## Step 2: Create Kaggle Secrets

Before uploading the notebook, add your API keys to Kaggle:

1. Go to [kaggle.com](https://www.kaggle.com) → **Settings** → **Account**
2. Scroll to **API** → Create a Kaggle token (download `kaggle.json`)
3. Go to any notebook → **Add-ons** → **Secrets** → Add:

| Secret Name | Value | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Your Gemini key | [aistudio.google.com](https://aistudio.google.com) → API Keys → Free tier |
| `HF_TOKEN` | Your HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

> **Important:** You must **accept the IndicF5 model gate** at
> https://huggingface.co/ai4bharat/IndicF5
> before your HF_TOKEN will work for downloading the weights.

> **Secrets are per-notebook.** A secret attached to one notebook is not visible to another.
> If translation exits reporting the key is missing, confirm `GEMINI_API_KEY` is toggled **on**
> for *this* notebook under Add-ons → Secrets.

---

## Step 3: Upload the Notebook to Kaggle

1. Go to **[kaggle.com/code](https://www.kaggle.com/code)**
2. Click **New Notebook**
3. Click **File → Import Notebook**
4. Upload `kaggle/indicai_dubbing_kaggle.ipynb`

---

## Step 4: Configure the Kaggle Notebook Settings

In the right sidebar of the notebook editor:

| Setting | Value |
|---|---|
| **Accelerator** | `GPU T4 x1` (free, 16 GB VRAM) |
| **Internet** | `On` (required for model downloads + Cloudflare tunnel) |
| **Persistence** | `Files only` |

Enable the secrets you created in Step 2 using the toggle next to each one.

---

## Step 5: Run All Cells

Click **Run All** (or run cells 1–10 in order). Here's what each cell does:

| Cell | Name | Time | Description |
|---|---|---|---|
| 1 | README | — | Instructions (markdown) |
| 2 | GPU Check | 5s | Verifies T4 GPU + CUDA |
| 3 | System Deps | 1 min | `ffmpeg`, `rubberband-cli`, **`espeak-ng`**, `fonts-noto`, `libsndfile1` |
| 4 | Python Deps (Part 1) | 4 min | Core audio/util **+ v2.5 translation core**: `phonemizer`, `indic-nlp-library`, `sentence-transformers` |
| 5 | Python Deps (Part 2) | 5 min | `f5-tts` + IndicF5 from GitHub |
| 6 | API Keys | 5s | Reads secrets from Kaggle Secrets |
| 7 | Write Pipeline Files | 10s | Creates all pipeline `.py` files (incl. **`phoneme_counter.py`** + **`semantic_similarity.py`**) in `/kaggle/working/` |
| 8 | Patch IndicF5 | 2 min | Downloads config, applies `model.py` fix |
| 9 | Setup cloudflared | 10s | Downloads Cloudflare tunnel binary |
| 10 | Launch App | — | **Starts Streamlit + prints your public URL** |

> **`espeak-ng` (Cell 3) is load-bearing in v2.5.** It is the real phoneme ruler. If it fails
> to install, the phoneme counter **degrades visibly** to `chars:heuristic-fallback` (it never
> lies about which ruler it used) and timing accuracy drops. The IndicSBERT model
> (`sentence-transformers`, Cell 4) downloads on first use during Stage 3.

**Total setup time: ~13–16 minutes** (first run; model weights download separately on first dub)

---

## Step 6: Access the App

Cell 10 will print something like:

```
================================================================
  YOUR DUBBING APP URL:
  https://random-words-here.trycloudflare.com
================================================================
```

Open that URL in any browser. The Streamlit UI will appear with:
- **Sidebar**: Enter your Gemini API Key + HF Token, and pick the **target Indic language**
- **Main panel**: Upload your video → click "Start Dubbing"

---

## How the Pipeline Works (Inside Kaggle)

```
Video Upload (MP4)
    ↓
[Step 1] FFmpeg → Extract 16kHz mono WAV
    ↓
[Step 2] Demucs (htdemucs) → Vocals track + Background track
    ↓
[Step 3] Faster-Whisper (large-v3) → Timestamped English segments
    ↓
[Step 4] Isochrony-Aware Translation  ★ v2.5 core ★
         Hybrid routing → 3 candidates/segment:
           • Bulk iteration-0 batch on lenient Gemma (gemma-4-31b/26b)
           • Refinement rounds on gemini-3.1-flash-lite
         (client-side throttle + disk cache keep it under free-tier limits), then:
           • Gate A · MEANING  → IndicSBERT cross-lingual cosine ≥ 0.70   (runs on CPU)
           • Gate B · TIMING   → espeak-ng phoneme count vs duration budget (±15%)
         Iterative CoT: refine misses up to 3 rounds, keep the global best,
         report gates_passed honestly. Works for all 11 Indic languages.
    ↓
[Step 5] Voice extraction → Best 12s clip from vocals
    ↓
[Step 6] IndicF5 (T4 GPU) → Duration-controlled TTS in the target language
           - Real DiT weights from model.safetensors
           - fix_duration conditioning (no post-stretch)
    ↓
[Step 7] Audio assembly → Dubbed vocal timeline + Background mix
    ↓
[Step 8] FFmpeg → Final MP4 with burnt-in subtitles
    ↓
Download Button → dubbed_<language>.mp4
```

---

## 🆕 v2.5 notes (read before your first run)

**The semantic gate runs on CPU on purpose.** Kaggle's GPU lottery can hand out a **Tesla
P100 (compute capability sm_60)** that the installed torch build has no compiled kernels for
— on which an IndicSBERT *GPU* encode dies with `cudaErrorNoKernelImageForDevice` and every
semantic score silently comes back as `None`. The pipeline forces IndicSBERT onto CPU
(`CUDA_VISIBLE_DEVICES=""` for the embedder), which also matches the intended deployment. The
T4 GPU stays dedicated to Demucs / Whisper / IndicF5.

**A `gates_passed=False` segment is not a bug.** When the loop cannot land a line that both
means the same thing *and* fits the shot within tolerance, it keeps the best candidate it ever
saw and marks it honestly. On a long video you should expect a handful of these — read the
per-run summary line the loop prints: `... | Both gates passed: N/total | ruler: <ruler>`.

**Confirm the real ruler is active.** That summary line (and the app logs) should say
`ruler: phonemes:espeak-ng-1.50`. If it says `chars:heuristic-fallback`, Cell 3's `espeak-ng`
install didn't take — re-run Cell 3, then restart the kernel.

**Rate limits are handled for you.** `gemini-3.1-flash-lite` has a tight free-tier quota
(per-minute *and* per-day). v2.5.1 keeps you under it three ways, all automatic: the **bulk
iteration-0 batch runs on the more lenient Gemma models** (`gemma-4-31b` → `gemma-4-26b`) and
only refinement rounds touch Gemini; a **client-side throttle** paces each model; and a
**disk cache** (under `/kaggle/working/.dubbing_cache/` by default) pools candidates so a
re-run selects final lines with **zero** API calls. On a 429 the stage reads the server's
`retryDelay` and waits, rather than retrying blind. Optional env overrides you can set before
launching Streamlit (Cell 10) if you have paid quota or want to force a model:

| Env var | Default | Effect |
|---|---|---|
| `DUBBING_GEMINI_BULK_MODEL` | `gemma-4-31b` | Model for the high-volume first pass. |
| `DUBBING_GEMINI_REFINE_MODEL` / `DUBBING_GEMINI_MODEL` | `gemini-3.1-flash-lite` | Model for refinement rounds. |
| `DUBBING_GEMINI_RPM` | 30 Gemma / 15 Gemini | Requests-per-minute throttle. |
| `DUBBING_GEMINI_RPD` | *(unset)* | Optional per-model daily cap; a capped model is skipped with a clear error. |
| `DUBBING_CACHE_DIR` | `./.dubbing_cache` | Where the candidate pool + usage counter live. |

---

## Troubleshooting

### "GEMINI_API_KEY not found" / translation skipped
→ Secrets are **per-notebook**. Toggle `GEMINI_API_KEY` on for *this* notebook under Add-ons
   → Secrets, then re-run the API-keys cell.

### All semantic scores are `None`
→ You likely drew a **P100** in the GPU lottery. v2.5 already forces the embedder onto CPU; if
   you're on a customized notebook, ensure `CUDA_VISIBLE_DEVICES=""` is set before IndicSBERT
   loads, or simply restart to try for a T4.

### Ruler shows `chars:heuristic-fallback`
→ `espeak-ng` isn't installed. Re-run Cell 3 (System Deps), confirm it prints ✓, then restart
   the kernel. Timing is approximate until the real ruler is active.

### "HF_TOKEN not in secrets"
→ Add `HF_TOKEN` in Kaggle Add-ons → Secrets. Must be a **read** token.

### "IndicF5 gate not accepted"
→ Visit https://huggingface.co/ai4bharat/IndicF5, click **"Agree and access"** while logged in.

### "CUDA out of memory"
→ IndicF5 uses ~5 GB VRAM. If OOM occurs, it's usually because Demucs and Whisper weren't
   freed before TTS. The pipeline already calls `unload_indicf5()` and
   `torch.cuda.empty_cache()` between steps — if still OOM, restart the session.

### Cloudflare URL not appearing
→ Wait 30 more seconds. If still nothing, stop Cell 10 and re-run it. The URL always appears
   within 60 seconds.

### "ffmpeg: subtitles filter not found"
→ The `libavfilter` build on Kaggle may not include the subtitles filter. The video is still
   produced without burnt-in subtitles; the `.srt` file is always available separately.

---

## Tips

- **Session limit**: Free Kaggle GPU = 30 hrs/week. A single full dub takes ~30–60 min.
- **File persistence**: Dubbed output is saved to `/kaggle/working/temp_processing/`. Download
  via the **Files** panel in the Kaggle sidebar before the session ends.
- **Restarting**: If you restart the kernel, just re-run Cell 10 (launch) — deps are already
  installed and pipeline files already written.
- **Updating pipeline**: If you change any `.py` file locally, re-run `python kaggle/build_notebook.py`
  to regenerate the notebook from the canonical source, then re-upload.
