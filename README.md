# Indic AI Dubbing Platform — Setup Guide

## Quick Start

### 1. Activate the existing virtual environment
```powershell
.\dubbing_app\Scripts\activate
```

### 2. Install new dependencies
```powershell
# Core new deps
pip install faster-whisper demucs soundfile pyrubberband resampy

# IndicF5 (from source — required for duration-controlled TTS)
pip install git+https://github.com/ai4bharat/IndicF5.git

# Accept the HuggingFace gate before running:
# Visit https://huggingface.co/ai4bharat/IndicF5 and accept terms
# Then login: huggingface-cli login
```

### 3. Run the app
```powershell
streamlit run app.py
```

---

## Pipeline Architecture (v2)

```
Input Video
    │
    ▼
Stage 1: Source Separation (Demucs htdemucs)
    ├── vocals.wav       → transcription & TTS reference
    └── background.wav   → preserved for final remix
    │
    ▼
Stage 2: Transcription (Faster-Whisper large-v3, INT8)
    └── timestamped English segments
    │
    ▼
Stage 3: Isochrony-Aware Translation (Gemini + phoneme budget CoT)
    └── Hindi/Indic text constrained to source phoneme count
    │
    ▼
Stage 4: Voice Manager (SNR analysis — Premium only)
    └── voice_reference.wav (12s clean speech clip)
    │
    ▼
Stage 5: Duration-Controlled TTS (IndicF5)
    ├── Basic: pre-selected natural voices
    └── Premium: zero-shot voice cloning from reference clip
    │
    ▼
Stage 6: Audio Assembly
    ├── Direct timeline overlay (no FFmpeg atempo stretching)
    ├── 50ms crossfades between segments
    └── Background + dubbed vocal mix (LUFS normalized)
    │
    ▼
Stage 7: Video Composition (FFmpeg)
    └── final_dubbed_video.mp4 with burned subtitles
```

## Key Differences from v1

| Feature | v1 (Old) | v2 (New) |
|---|---|---|
| Transcription | openai-whisper base | Faster-Whisper large-v3 (INT8) |
| Translation | Generic Gemini prompt | Isochrony-aware CoT + phoneme scoring |
| TTS | Edge-TTS (Microsoft) | IndicF5 (AI4Bharat) — duration-controlled |
| Audio sync | FFmpeg atempo stretch | Direct overlay (no stretching) |
| Voice cloning | None | Zero-shot via IndicF5 (Premium) |
| Languages | Hindi only | All 11 Indic languages |
| Background audio | Overwritten | Preserved and remixed (Demucs) |

## Deployment on Kaggle (Free GPU)

ShadowGPU-style pattern: run the pipeline on Kaggle's free dual T4 (2×16GB VRAM),
expose via Cloudflare Tunnel or ngrok, hit it from local machine.

1. Create a Kaggle notebook with GPU accelerator enabled
2. Install deps in the notebook
3. Run `cloudflared tunnel --url http://localhost:8501` 
4. Access the Streamlit UI at the tunnel URL

Kaggle provides 30 GPU hours/week free. Dual T4 = 32GB VRAM total.

## Pricing Reference

| Tier | Features | Price/min |
|---|---|---|
| Basic | Audio dubbing, generic voices | $0.02 |
| Premium | Audio dubbing + voice cloning | $0.05 |

Self-hosted COGS on T4: ~$0.004/min (87%+ gross margin at Basic price).
