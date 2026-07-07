"""
source_separation.py — Stage 1 of the Indic Dubbing Pipeline

Uses Demucs v4 (htdemucs model) to separate an audio file into:
  - vocals: the foreground speech track
  - no_vocals (bg): background music, SFX, ambient sound

The background track is preserved and remixed with the dubbed audio at the
end of the pipeline, ensuring original soundscapes are not destroyed.

VRAM: ~2GB. Falls back to CPU automatically if no GPU is available.
"""

import os
import subprocess
import sys
import shutil


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _check_demucs_installed() -> bool:
    """Check whether demucs is importable / on PATH."""
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def separate_audio(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs",
    device: str = "auto",
) -> dict:
    """
    Separate vocals from background audio using Demucs.

    Args:
        audio_path: Path to the input audio file (.wav, .mp3, etc.)
        output_dir:  Directory where separated stems will be written.
        model:       Demucs model name. 'htdemucs' is the best quality/speed
                     balance. 'htdemucs_ft' is fine-tuned but slower.
        device:      'auto' (uses CUDA if available, else CPU), 'cuda', or 'cpu'.

    Returns:
        dict with keys:
          'vocals'     -> absolute path to vocals stem (.wav)
          'background' -> absolute path to no_vocals stem (.wav)

    Raises:
        RuntimeError: if demucs is not installed or separation fails.
        FileNotFoundError: if audio_path does not exist.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    if not _check_demucs_installed():
        raise RuntimeError(
            "Demucs is not installed. Run: pip install demucs"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Resolve device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"[SourceSeparation] Model: {model} | Device: {device}")
    print(f"[SourceSeparation] Input: {audio_path}")

    # Build demucs CLI command.
    # demucs writes to: <output_dir>/<model>/<track_name>/{vocals,no_vocals,drums,bass,other}.wav
    # We use --two-stems=vocals to only produce vocals + no_vocals (faster, less VRAM).
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-d", device,
        "-o", output_dir,
        audio_path,
    ]

    print(f"[SourceSeparation] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(f"[SourceSeparation] stderr:\n{result.stderr}")
        raise RuntimeError(
            f"Demucs failed with exit code {result.returncode}.\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    # Locate the output stems.
    # Demucs places files at: <output_dir>/<model>/<input_stem>/{vocals,no_vocals}.wav
    input_stem = os.path.splitext(os.path.basename(audio_path))[0]
    stems_dir = os.path.join(output_dir, model, input_stem)

    vocals_path = os.path.join(stems_dir, "vocals.wav")
    bg_path = os.path.join(stems_dir, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise RuntimeError(
            f"Demucs completed but vocals stem not found at: {vocals_path}\n"
            f"Check output dir: {stems_dir}"
        )
    if not os.path.exists(bg_path):
        raise RuntimeError(
            f"Demucs completed but no_vocals stem not found at: {bg_path}"
        )

    # Copy stems to a flat, predictable location in output_dir for easy access.
    flat_vocals = os.path.join(output_dir, "vocals.wav")
    flat_bg = os.path.join(output_dir, "background.wav")
    shutil.copy2(vocals_path, flat_vocals)
    shutil.copy2(bg_path, flat_bg)

    print(f"[SourceSeparation] Vocals  -> {flat_vocals}")
    print(f"[SourceSeparation] Background -> {flat_bg}")

    return {
        "vocals": flat_vocals,
        "background": flat_bg,
    }
