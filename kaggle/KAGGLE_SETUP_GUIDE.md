# 🎙️ Indic AI Dubbing Platform — Kaggle Deployment Guide

This guide tells you exactly how to run the full dubbing pipeline on Kaggle's free T4 GPU
and access it via a Cloudflare tunnel in your browser.

---

## Step 1: Generate the Notebook (Local Machine)

The `build_notebook.py` script produces a **self-contained** Kaggle notebook
that embeds all pipeline code. Run it once from your Windows machine:

```powershell
# In E:\Dubbing app\kaggle\
D:\Miniconda\python.exe build_notebook.py
```

This creates **`indicai_dubbing_kaggle.ipynb`** (~120 KB) in the same folder.
You only need to re-run this if you change any pipeline files.

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

---

## Step 3: Upload the Notebook to Kaggle

1. Go to **[kaggle.com/code](https://www.kaggle.com/code)**
2. Click **New Notebook**
3. Click **File → Import Notebook**
4. Upload `E:\Dubbing app\kaggle\indicai_dubbing_kaggle.ipynb`

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
| 3 | System Deps | 1 min | Installs ffmpeg, espeak-ng, fonts |
| 4 | Python Deps (Part 1) | 3 min | soundfile, faster-whisper, demucs, etc. |
| 5 | Python Deps (Part 2) | 5 min | f5-tts + IndicF5 from GitHub |
| 6 | API Keys | 5s | Reads secrets from Kaggle Secrets |
| 7 | Write Pipeline Files | 10s | Creates all pipeline .py files in /kaggle/working/ |
| 8 | Patch IndicF5 | 2 min | Downloads config, applies model.py fix |
| 9 | Setup cloudflared | 10s | Downloads Cloudflare tunnel binary |
| 10 | Launch App | — | **Starts Streamlit + prints your public URL** |

**Total setup time: ~12–15 minutes** (first run, model weights download separately on first dub)

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
- **Sidebar**: Enter your Gemini API Key + HF Token (even though they're in secrets, the UI asks for them too — just re-enter)
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
[Step 4] Gemini 2.5 Flash → Isochrony-aware Hindi translation
           (3 candidates/segment, phoneme-budget constrained)
    ↓
[Step 5] Voice extraction → Best 12s clip from vocals
    ↓
[Step 6] IndicF5 (T4 GPU) → Duration-controlled Hindi TTS
           - Real DiT weights from model.safetensors
           - fix_duration conditioning (no post-stretch)
           - 16 GB VRAM handles all 72 segments
    ↓
[Step 7] Audio assembly → Hindi vocal timeline + Background mix
    ↓
[Step 8] FFmpeg → Final MP4 with burnt-in Hindi subtitles
    ↓
Download Button → dubbed_hindi.mp4
```

---

## Troubleshooting

### "HF_TOKEN not in secrets"
→ Add `HF_TOKEN` in Kaggle Add-ons → Secrets. Must be a **read** token.

### "IndicF5 gate not accepted"
→ Visit https://huggingface.co/ai4bharat/IndicF5, click **"Agree and access"** while logged in.

### "CUDA out of memory"
→ The model uses ~5 GB VRAM. If OOM occurs, it's usually because Demucs and Whisper
   weren't freed before TTS. The pipeline already calls `unload_indicf5()` and
   `torch.cuda.empty_cache()` between steps — if still OOM, restart the session.

### Cloudflare URL not appearing
→ Wait 30 more seconds. If still nothing, stop Cell 10 and re-run it.
   The URL always appears within 60 seconds.

### "ffmpeg: subtitles filter not found"
→ The `libavfilter` build on Kaggle may not include the subtitles filter.
   The video will still be produced without burnt-in subtitles; the `.srt` file
   is always available for download separately.

---

## Tips

- **Session limit**: Free Kaggle GPU = 30 hrs/week. A single full dub takes ~30–60 min.
- **File persistence**: Dubbed output is saved to `/kaggle/working/temp_processing/`.
  Download via the **Files** panel in the Kaggle sidebar before the session ends.
- **Restarting**: If you restart the kernel, just re-run Cell 10 (launch) — 
  deps are already installed and pipeline files already written.
- **Updating pipeline**: If you change any `.py` file locally, re-run `build_notebook.py`
  to regenerate the notebook, then re-upload.
