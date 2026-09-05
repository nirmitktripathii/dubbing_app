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
    log_fn=None,
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
    def _emit(msg):
        print(msg)
        if log_fn:
            try:
                log_fn(f"    {msg}")
            except Exception:
                pass

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

    _emit(f"[SourceSeparation] Model: {model} | Device: {device}")
    _emit(f"[SourceSeparation] Input: {audio_path}")
    _emit("[SourceSeparation] Separating (GPU ~30-90s / CPU several min)...")

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

    _emit(f"[SourceSeparation] Running: {' '.join(cmd)}")

    # Stream Demucs output line-by-line so its progress is visible in the UI
    # instead of buffering silently until the whole (minute-plus) run finishes.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail = []
    last_pct = -10
    for raw in iter(proc.stdout.readline, ""):
        # Demucs draws a tqdm bar with carriage returns; split on them so each
        # progress update is its own line rather than one giant blob.
        for line in raw.replace("\r", "\n").split("\n"):
            line = line.strip()
            if not line:
                continue
            tail.append(line)
            tail[:] = tail[-40:]
            import re as _re
            m = _re.search(r"(\d+)%", line)
            if m:
                # Throttle progress lines to every ~10% to avoid flooding the log.
                pct = int(m.group(1))
                if pct >= last_pct + 10 or pct >= 100:
                    last_pct = pct
                    _emit(f"[Demucs] {line}")
            else:
                _emit(f"[Demucs] {line}")
    proc.stdout.close()
    returncode = proc.wait()

    if returncode != 0:
        _emit(f"[SourceSeparation] Demucs failed (exit {returncode}).")
        raise RuntimeError(
            f"Demucs failed with exit code {returncode}.\n"
            f"last output:\n" + "\n".join(tail[-30:])
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

    _emit(f"[SourceSeparation] Vocals  -> {flat_vocals}")
    _emit(f"[SourceSeparation] Background -> {flat_bg}")

    return {
        "vocals": flat_vocals,
        "background": flat_bg,
    }
