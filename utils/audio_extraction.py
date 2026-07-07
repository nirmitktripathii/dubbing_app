import os
import subprocess


def extract_audio(video_path, output_audio_path):
    """
    Extracts the audio track from a video file and saves it as a WAV file.
    Uses FFmpeg directly via subprocess — works on any system without
    depending on moviepy or any Python audio library.

    Args:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio file.
    """
    print(f"Extracting audio from {video_path}...")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                       # overwrite output if exists
        "-i", video_path,           # input file
        "-vn",                      # no video
        "-acodec", "pcm_s16le",     # PCM 16-bit WAV
        "-ar", "16000",             # 16kHz sample rate (Whisper-friendly)
        "-ac", "1",                 # mono channel
        output_audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed:\n{result.stderr}"
        )

    print(f"Audio extracted to: {output_audio_path}")
