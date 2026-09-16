#!/usr/bin/env python3
"""Zip the dubbing pipeline's output folder into ONE archive under /kaggle/working and
expose it for download to your local machine.

Two ways to use it:
  * Paste this whole file into a NEW notebook cell (AFTER the pipeline has run) and run it.
  * Or run it as a script:   python zip_output.py                 (full output)
                             DUB_ZIP_LEAN=1 python zip_output.py  (deliverables only)

Honest note on "download to local": a Kaggle kernel cannot push a file to your disk on its
own. This makes the .zip a persisted Kaggle output; you then pull it via the click-link
(interactive session), the notebook's Output tab (Save & Run All), or the Kaggle CLI — all
three are printed at the end.

The archive is written to /kaggle/working (a sibling of the output dir, never inside it), so
it is part of what the commit persists and is safe to re-run.
"""
import os
import time
import zipfile


def zip_dubbing_output(output_dir=None, dest_dir="/kaggle/working", lean=False):
    """Zip ``output_dir`` (default: $DUBBING_OUTPUT_DIR or /kaggle/working/dubbing_output)
    into ``dest_dir``. When ``lean`` is True the bulky intermediates (per-segment TTS WAVs,
    Demucs stems, the extracted source audio) are skipped and only the final deliverables —
    dubbed video, dubbed audio, subtitles, log, manifest — are kept. Returns the .zip path
    (or None if the output dir is missing)."""
    output_dir = (output_dir or os.environ.get(
        "DUBBING_OUTPUT_DIR", "/kaggle/working/dubbing_output")).rstrip("/")
    if not os.path.isdir(output_dir):
        print(f"[X] Output dir not found: {output_dir!r}. Run the pipeline first.")
        return None

    # These dirs hold the heavy intermediates. 'separated' is nested under
    # 'temp_processing', but listing it too is harmless if the layout ever changes.
    exclude = {"temp_processing", "tts_chunks", "separated"} if lean else set()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(dest_dir, f"dubbing_output_{stamp}{'_lean' if lean else ''}.zip")
    top = os.path.basename(output_dir)  # top-level folder name inside the archive

    def human(n):
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.1f} {unit}"
            n /= 1024.0

    n_files = 0
    raw_bytes = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for root, dirs, files in os.walk(output_dir):
            # Prune excluded dirs in place so os.walk never descends into them.
            dirs[:] = [d for d in dirs if d not in exclude]
            for name in sorted(files):
                fpath = os.path.join(root, name)
                if os.path.abspath(fpath) == os.path.abspath(zip_path):
                    continue  # never zip the archive into itself
                arcname = os.path.join(top, os.path.relpath(fpath, output_dir))
                try:
                    raw_bytes += os.path.getsize(fpath)
                except OSError:
                    pass
                zf.write(fpath, arcname)
                n_files += 1

    if n_files == 0:
        print(f"[!] {output_dir} has no files to zip (is the run finished?).")
        return None

    zsize = os.path.getsize(zip_path)
    print(f"[OK] Zipped {n_files} file(s): {human(raw_bytes)} raw -> {human(zsize)} compressed")
    print(f"     {zip_path}")
    if lean:
        print(f"     (LEAN mode: skipped {sorted(exclude)})")

    # Clickable download link — renders in an INTERACTIVE Kaggle/Jupyter session.
    try:
        from IPython.display import FileLink, display
        display(FileLink(os.path.relpath(zip_path, os.getcwd())))
    except Exception:
        pass

    print(
        "\nGet it onto your machine:\n"
        "  * Interactive session: click the link above, or use the right-hand Output\n"
        "    panel -> the .zip -> the download icon.\n"
        "  * Save & Run All (batch commit): open the finished version's 'Output' tab and\n"
        f"    download {os.path.basename(zip_path)} (saved under /kaggle/working).\n"
        "  * Local terminal (Kaggle CLI, after the commit finishes):\n"
        "      kaggle kernels output <username>/<kernel-slug> -p ./dubbing_download\n"
        f"    then unzip ./dubbing_download/{os.path.basename(zip_path)}\n"
    )
    return zip_path


# __name__ is '__main__' both when run as a script AND when pasted into a notebook cell,
# so this fires in both cases but not on import.
if __name__ == "__main__":
    _lean = os.environ.get("DUB_ZIP_LEAN", "").strip().lower() in ("1", "true", "yes", "on")
    zip_dubbing_output(lean=_lean)
