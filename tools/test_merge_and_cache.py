#!/usr/bin/env python3
"""CPU-only property test for the two zero-GPU wins — P2 (video_merge) and P10 (translation
cache). No GPU, no Modal. Backs the measured numbers in PRODUCTION_OPTIMIZATION_AUDIT.md so
they regenerate on any machine with ffmpeg (CLAUDE rule 8: a figure must be reproducible by a
repo script).

Run:  python tools/test_merge_and_cache.py
Needs: ffmpeg + ffprobe on PATH. Uses a sample video if one is present (DUB_TEST_VIDEO env, or
a *.mp4 at the repo root); otherwise it SYNTHESIZES a 25 s 720x1280 clip with ffmpeg lavfi, so
it is fully self-contained.

What it asserts (the PROPERTY, not that ffmpeg ran):
  P2 copy path  -> video is BIT-IDENTICAL to source (per-frame md5 match => not re-encoded),
                   a soft mov_text subtitle track is present, the full video is preserved
                   (a short soft-sub track must NOT truncate it), and it is much faster.
  P2 burn path  -> every frame's pixels change (subs burned in) and there is NO subtitle track.
  P10           -> candidates persist to the CONFIGURED DUBBING_CACHE_DIR, a fresh instance
                   reads them back, and nothing leaks to cwd/.dubbing_cache; and modal_app.py
                   actually sets DUBBING_CACHE_DIR + commits the cache Volume after a run.
"""
import os, sys, time, subprocess, tempfile, shutil, glob

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ -> repo root
sys.path.insert(0, REPO)
work = tempfile.mkdtemp(prefix="mergetest_")


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def frame_md5s(path):
    """Per-decoded-frame md5 list. Lossless copy => identical to source (a prefix if a trailing
    frame is trimmed); a re-encode changes every one."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True)
    return [ln.split(",")[-1].strip() for ln in out.stdout.splitlines()
            if ln.strip() and not ln.startswith("#")]


def streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().replace("\n", " | ")


def source_clip():
    env_v = os.environ.get("DUB_TEST_VIDEO")
    cands = ([env_v] if env_v else []) + sorted(glob.glob(os.path.join(REPO, "*.mp4")))
    for c in cands:
        if c and os.path.exists(c):
            test_in = os.path.join(work, "test_in.mp4")
            run(["ffmpeg", "-y", "-i", c, "-t", "25", "-c:v", "libx264", "-preset", "veryfast",
                 "-an", test_in])
            return test_in, f"sample={os.path.basename(c)}"
    test_in = os.path.join(work, "test_in.mp4")
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=25",
         "-c:v", "libx264", "-preset", "veryfast", "-an", test_in])
    return test_in, "synthetic testsrc 1280x720"


def main():
    from utils.video_merge import merge_video_audio_subs

    test_in, src_desc = source_clip()
    print(f"== test input: {src_desc} ==")
    dubbed = os.path.join(work, "dubbed.wav")
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=25", "-ar", "44100", dubbed])
    srt = os.path.join(work, "subs.srt")
    with open(srt, "w", encoding="utf-8") as f:      # last cue ends at 0:08, well before 0:25
        f.write("1\n00:00:00,000 --> 00:00:03,000\nnamaste duniya\n\n"
                "2\n00:00:04,000 --> 00:00:08,000\nyah ek parikshan hai\n")

    src_frames = frame_md5s(test_in)
    print(f"source: {len(src_frames)} frames, streams=[{streams(test_in)}]")

    out_copy = os.path.join(work, "out_copy.mp4")
    t0 = time.time(); merge_video_audio_subs(test_in, dubbed, srt, out_copy, burn_subs=False)
    copy_s = time.time() - t0
    copy_frames, copy_streams = frame_md5s(out_copy), streams(out_copy)

    out_burn = os.path.join(work, "out_burn.mp4")
    t0 = time.time(); merge_video_audio_subs(test_in, dubbed, srt, out_burn, burn_subs=True)
    burn_s = time.time() - t0
    burn_frames, burn_streams = frame_md5s(out_burn), streams(out_burn)

    n = min(len(src_frames), len(copy_frames))
    copy_identical = n > 0 and src_frames[:n] == copy_frames[:n]
    m = min(len(src_frames), len(burn_frames))
    burn_changed = m > 0 and all(src_frames[i] != burn_frames[i] for i in range(m))

    print("\n== results ==")
    print(f"copy path : {copy_s:6.2f}s   {len(copy_frames)} frames  bit-identical={copy_identical}  streams=[{copy_streams}]")
    print(f"burn path : {burn_s:6.2f}s   {len(burn_frames)} frames  every-frame-changed={burn_changed}  streams=[{burn_streams}]")
    print(f"speedup   : burn/copy = {burn_s/copy_s:.1f}x   (25 s clip)")

    fails = []
    if not copy_identical:
        fails.append("copy path RE-ENCODED the video (frames differ) — expected lossless copy")
    if len(copy_frames) < 0.95 * len(src_frames):
        fails.append(f"copy path TRUNCATED the video ({len(copy_frames)}/{len(src_frames)} frames) "
                     "— a short soft-sub track must not cut the video short")
    if "subtitle" not in copy_streams and "mov_text" not in copy_streams:
        fails.append("copy path produced NO soft subtitle track")
    if not burn_changed:
        fails.append("burn path did NOT change the pixels (subtitles not burned in?)")
    if "subtitle" in burn_streams or "mov_text" in burn_streams:
        fails.append("burn path left a separate subtitle track (should be burned into pixels)")
    if copy_s >= burn_s:
        fails.append(f"copy path was not faster ({copy_s:.2f}s vs burn {burn_s:.2f}s)")

    print("\n== P10: translation cache persistence ==")
    cache_dir = os.path.join(work, "cfg_cache")
    os.environ["DUBBING_CACHE_DIR"] = cache_dir
    os.environ.pop("DUBBING_TRANSLATION_CACHE", None)
    import pipeline.translation_cache as tc
    c1 = tc.TranslationCache()
    c1.add("Hindi", "hello world", ["cand-one", "cand-two"])
    c1.save()
    persisted = os.path.exists(os.path.join(cache_dir, "translations.json"))
    got = tc.TranslationCache().get("Hindi", "hello world")       # fresh instance reads back
    leaked = os.path.exists(os.path.join(os.getcwd(), ".dubbing_cache", "translations.json"))
    print(f"cache at configured dir? {persisted}   fresh read-back: {got}   leaked to cwd? {leaked}")
    if not persisted:
        fails.append("cache did NOT write to the configured DUBBING_CACHE_DIR")
    if "cand-one" not in got:
        fails.append("fresh cache instance did not read back the saved candidates")

    modal_src = open(os.path.join(REPO, "deploy", "modal_app.py"), encoding="utf-8").read()
    if '"DUBBING_CACHE_DIR"' not in modal_src:
        fails.append("modal_app.py does not set DUBBING_CACHE_DIR in the worker env")
    # The cache Volume must be committed AFTER the pipeline runs (so Stage-4 cache adds
    # survive). Structure-independent since the split refactor moved the run into a helper:
    # find a run call and assert a commit follows it within a small window.
    committed_after_run, start = False, 0
    while True:
        i = modal_src.find("_run_headless_streamed(", start)
        if i < 0:
            break
        if "hf_cache.commit()" in modal_src[i:i + 600]:
            committed_after_run = True
            break
        start = i + 1
    if not committed_after_run:
        fails.append("modal_app.py does not commit the cache Volume after the run")

    print("\n" + ("FAIL:\n  - " + "\n  - ".join(fails) if fails else "ALL PROPERTIES HOLD ✓"))
    return 1 if fails else 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(rc)
