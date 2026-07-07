# AI Video Dubbing & Transcription Application

## 📌 Overview
The AI Video Dubbing & Transcription Application is a web-based, fully automated platform designed to bridge language barriers in video content. The tool accepts English video inputs (up to 5-10 minutes) and automatically dubs the audio into various target languages (such as Hindi, Spanish, French, and German). In addition to generating the dubbed audio track, it produces accurately synchronized translated subtitles, which are burned back into the final video output.

## ✨ Core Features
- **One-Click Automation**: A complete pipeline that takes a single video upload and outputs a fully localized and captioned video.
- **Cost-Effective Translation**: Transitions from high-cost models to the highly capable and generous **Google Gemini 2.5 Flash API**, ensuring professional-grade contextual translations at minimal to zero cost.
- **Hardware-Accelerated Transcription**: Uses **OpenAI Whisper** with CUDA parallel computing support. Automatically leverages GPU processing (if available on the host machine) to rapidly infer speech-to-text, defaulting to universally efficient models for speed without sacrificing significant accuracy.
- **Dynamic Time-Stretching & Padding**: Solves the classic "audio overlapping" problem in AI dubbing. If a translated Hindi sentence takes longer to say than the English original, the software uses **Librosa** to dynamically speed up the generated speech without distorting the pitch, ensuring it fits perfectly within the visual speaking window.
- **Interactive Web UI**: Powered by **Streamlit**, offering progress bars, live status updates, direct video playback, and individual asset download links for audio, captions, and the final video.

---

## ⚙️ System Architecture & Workflow

The pipeline is split into isolated modular functions housed within the `utils/` directory. When a user uploads a video and presses "Start Processing", the following sequence executes:

### 1. Audio Extraction (`utils/audio_extraction.py`)
- The system reads the raw `.mp4` video buffer.
- Utilizing **MoviePy**, it strips the original audio track and exports it as a standalone `original_audio.wav` file.
- *Why?* Standalone audio is required by the Whisper transcription model.

### 2. Transcription & Subtitling (`utils/transcription.py`)
- The `wav` file is fed into the local **OpenAI Whisper** model. 
- The model detects the speech and splits it into discrete contextual segments, each stamped with an exact `start` and `end` time in seconds.
- The text is formatted and saved as an initial `english_subtitles.srt` file.

### 3. Translation (`utils/translation.py`)
- The text from the Whisper segments is passed to the **Google Gemini API** (`gemini-2.5-flash`).
- Instead of translating word-by-word (which loses context), it translates the entire array of sentences simultaneously. This ensures the model understands the conversational context of the video.
- The translated text strings securely replace the English text in the data structure while maintaining the identical `start` and `end` timestamps.
- A new translated subtitle file (e.g., `Hindi_subtitles.srt`) is generated.

### 4. Text-to-Speech (TTS) Voice Generation (`utils/tts.py`)
- The system iterates through the translated text segments.
- Using Microsoft's **Edge-TTS**, it assigns a neural voice profile corresponding to the selected target language (e.g., `hi-IN-MadhurNeural` for Hindi, `es-ES-AlvaroNeural` for Spanish).
- It asynchronously requests audio generation for each chunk of text, creating dozens of short `.mp3` files in the temporary directory.

### 5. Audio Synchronization (`utils/audio_sync.py`)
- This is the most crucial synchronization step. The system measures the *target duration* (the time the original speaker took to say the line) and compares it to the *actual duration* of the newly generated TTS audio chunk.
- **If TTS is too long:** It uses **Librosa** to apply pitch-preserving time-stretching, compressing the audio slightly until it precisely fits the required window constraint.
- **If TTS is too short:** It uses **Pydub** to append exact milliseconds of silent padding to the end of the clip.
- Finally, it stitches all the individual chunks together, inserting silence for sections where no one was speaking, resulting in a single perfectly-timed `synced_audio.mp3` track.

### 6. Final Composition (`utils/video_merge.py`)
- Using a raw **FFmpeg** subprocess command, the original video track is merged with the new `synced_audio.mp3`.
- The original English audio stream is explicitly dropped/muted (`-map 0:v:0 -map 1:a:0`).
- A video-filter (`-vf`) is applied to "hardcode/burn" the new translated `.srt` subtitles directly into the video frames.
- The final product, `final_dubbed_video.mp4`, is exported and displayed to the user.

---

## 🛠️ Tech Stack
- **Frontend / Core App framework:** Python, Streamlit
- **Media Processing / Composition:** FFmpeg, MoviePy
- **Transcription (Local AI):** OpenAI Whisper
- **Translation (Cloud AI):** Google Gemini SDK (`google-genai`), Gemini 2.5 Flash Model
- **Voice Synthesis:** Edge-TTS (Microsoft Neural Voices)
- **Audio Engineering:** Librosa, Pydub, Soundfile
