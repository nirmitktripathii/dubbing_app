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

def merge_video_audio_subs(video_path: str, audio_path: str, srt_path: str, output_path: str):
    """
    Merges the original video, the new dubbed audio, and the subtitle file using FFmpeg.
    """
    escaped_srt = escape_path_for_ffmpeg(srt_path)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vf", f"subtitles='{escaped_srt}':force_style='Fontname=Nirmala UI,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1'",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-map", "0:v:0", # Use video from first input
        "-map", "1:a:0", # Use audio from second input
        "-shortest",     # Finish encoding when shortest stream ends
        output_path
    ]
    
    print(f"Running FFmpeg: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print("Video merging complete.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error output:\n{e.stderr}")
        raise RuntimeError(f"FFmpeg failed with error:\n{e.stderr}")
        
    return output_path
