import os
import subprocess

def escape_path_for_ffmpeg(path):
    """
    Properly escapes a Windows filepath for FFmpeg filters.
    C:\\path\\file.srt -> C\\:/path/file.srt
    """
    path = path.replace('\\', '/')
    path = path.replace(':', '\\:')
    return path

def merge_video_audio_subs(video_path: str, audio_path: str, srt_path: str, output_path: str,
                           log_fn=None, burn_subs: bool = False):
    """
    Merges the original video, the new dubbed audio, and the subtitle file using FFmpeg.

    Two paths, selected by `burn_subs`:

    * `burn_subs=False` (default) — **stream-copy** the video and **soft-mux** the subtitles
      as a `mov_text` track. The video stream is passed through untouched (`-c:v copy`), so
      there is NO re-encode: this finishes in ~real-file-write time instead of the minute-plus
      a full libx264 encode of the source costs, and the caption track stays toggleable by the
      viewer. Only the (short, new) dubbed audio is encoded.
    * `burn_subs=True` — the original behaviour: hard-burn the subtitles into the pixels via
      the `subtitles` filter, which forces a full `libx264` re-encode of the whole video.
      Kept for callers/tiers that need pixels-baked captions; uses `-preset veryfast` so the
      unavoidable re-encode is as cheap as it can be.

    Rationale + measurement in PRODUCTION_OPTIMIZATION_AUDIT.md (P2). The re-encode was the
    only reason Step 7 held the machine for ~45 s; the copy path removes it.
    """
    def _emit(msg):
        print(msg)
        if log_fn:
            try:
                log_fn(f"    {msg}")
            except Exception:
                pass

    have_srt = bool(srt_path) and os.path.exists(srt_path)

    if burn_subs:
        if not have_srt:
            _emit(f"burn_subs requested but no subtitle file at {srt_path!r}; "
                  "burning is skipped — re-encoding with no caption overlay.")
        escaped_srt = escape_path_for_ffmpeg(srt_path) if have_srt else None
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path]
        if have_srt:
            cmd += ["-vf", f"subtitles='{escaped_srt}':force_style='Fontname=Nirmala UI,"
                           "FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                           "BorderStyle=1'"]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac",
            "-map", "0:v:0",   # video from the first input
            "-map", "1:a:0",   # dubbed audio from the second input
            "-shortest",
            output_path,
        ]
        _emit("Encoding final video with FFmpeg (burning subtitles, this can take a minute)...")
    else:
        # Copy the video stream verbatim; soft-mux the SRT as a mov_text subtitle track.
        # NOTE: deliberately NO -shortest here. A soft subtitle track ends at its last cue,
        # which is usually BEFORE the video's final frame (trailing music/silence carries no
        # caption). -shortest would then truncate the whole output to the last subtitle and
        # silently drop the end of the video. The dubbed audio is already built on the video's
        # timeline, so letting the streams run to their natural ends keeps the full video.
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path]
        if have_srt:
            cmd += ["-i", srt_path]
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        if have_srt:
            cmd += ["-map", "2:s:0"]
        cmd += ["-c:v", "copy", "-c:a", "aac"]
        if have_srt:
            cmd += ["-c:s", "mov_text"]
        cmd += [output_path]
        _emit("Muxing final video with FFmpeg (stream-copy video + soft subtitles, no re-encode)...")

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        _emit("Video merging complete.")
    except subprocess.CalledProcessError as e:
        _emit(f"FFmpeg failed:\n{e.stderr[-1500:]}")
        raise RuntimeError(f"FFmpeg failed with error:\n{e.stderr}")

    return output_path
