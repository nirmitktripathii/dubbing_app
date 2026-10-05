#!/usr/bin/env python3
"""Run the web demo UI locally against a SIMULATED pipeline — no Modal, no GPU, no API calls.

    python deploy/demo_dev_server.py            # then open http://localhost:8765
    (access code: dev)
    python deploy/demo_dev_server.py --voice-dir <dir with curated <code>.wav + manifest.json>

Builds the real FastAPI app (deploy/api.py + deploy/demo_ui.py) with in-memory fakes in place
of the Modal Dict / Volume / dub_video. The fake job walks the real stage strings
modal_app._parse_stage emits, writes the same meter fields modal_app writes, and "dubs" by
copying the upload — so the page, the progress mapping and the cost maths are exercised end to
end. The page shows a banner saying the figures are simulated.

With --voice-dir, the voice hint and the simulated log come from the real resolver
(pipeline/voice_refs.py) over that directory, and the "dubbed" video's audio track is replaced by
the reference clip a real Basic-mode run would clone — so a curated voice can be heard in the
page's player. It is the reference voice, not a dub.
"""
import os
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DEMO_ACCESS_CODE", "dev")
os.environ["DUB_DEMO_SIMULATED"] = "1"
os.environ.setdefault("DUB_YOUTUBE_DIRECT", "1")   # this machine's IP can reach YouTube

from deploy.api import build_api  # noqa: E402

JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".demo_jobs")
STATUS: dict = {}

# (stage string as modal_app._parse_stage writes it, log line, seconds to dwell)
SCRIPT = [
    ("transcribing", "Starting GPU container (L4)…", 1.5),
    ("step 1/7 — Extracting audio...", "Step 1/7: Extracting audio...", 1.0),
    ("step 2/7 — Demucs source separation...", "Step 2/7: Demucs source separation...", 2.0),
    ("step 3/7 — Whisper transcription (medium)...", "Step 3/7: Whisper transcription (medium)...", 2.0),
    ("step 4/7 — Isochrony-aware translation -> {lang}...",
     "Step 4/7: Isochrony-aware translation -> {lang}...", 4.0),
    ("step 5/7 — Basic mode (DUBBING_VOICE_CLONE=0) — native TTS voice, no cloning.",
     "Step 5/7: Basic mode (DUBBING_VOICE_CLONE=0) — native TTS voice, no cloning.", 1.0),
    ("step 6/7 — Fan-out IndicF5 TTS for 6 segments", "Step 6/7: Fan-out IndicF5 TTS for 6 segments", 1.0),
] + [(f"step 6/7 — TTS segment {i}/6", f"  [Segment {i}/6] ok", 0.6) for i in range(1, 7)] + [
    ("step 7/7 — Assembling audio and merging video...", "Step 7/7: Assembling audio and merging video...", 1.5),
]


class FakeVol:
    def commit(self): pass
    def reload(self): pass


def _resolve_voice(lang, log_fn):
    """The real Basic-mode resolver's decision for ``lang``. The pinned Hindi clip is not
    downloaded here, so Hindi and fallback languages report the decision without audio."""
    from pipeline import voice_refs
    code = {v: k for k, v in voice_refs.CODE_TO_LANGUAGE.items()}.get(lang)
    if code is None:
        return None
    ref = voice_refs.resolve_basic_reference(
        code, log_fn=lambda m: log_fn(f"  [voice] {m}"),
        download_pinned=lambda: None)
    log_fn(f"  [voice] resolved: {ref['source']} {ref['detail'] or ''}".rstrip())
    return ref


def _mux_reference(src, ref_wav, out, log_fn):
    """Replace ``src``'s audio with ``ref_wav`` (video looped/cut to the clip). True on success."""
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", src,
                        "-i", ref_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                        "-preset", "ultrafast", "-c:a", "aac", "-shortest", out],
                       check=True, capture_output=True, timeout=120)
    except Exception as e:
        log_fn(f"  [voice] could not attach the reference clip ({e}); copying the upload instead")
        return False
    log_fn("  [voice] SIMULATED output: its audio is the reference clip a real run would clone, "
           "not a dub")
    return True


def _write_fake_transcripts(out_dir, lang, seconds):
    """english_subtitles.srt + <Language>_subtitles.srt, named exactly as run_headless names
    them, so the page's transcript panes, downloads and captions run against real files. The
    text says it is simulated — nothing was transcribed or translated."""
    def ts(s):
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    n = max(2, min(6, int(seconds // 4)))
    step = seconds / n
    for name, text in (("english_subtitles.srt", "[simulated English line {i}]"),
                       (f"{lang}_subtitles.srt", f"[simulated {lang} line {{i}}]")):
        blocks = [f"{i}\n{ts((i - 1) * step + 0.2)} --> {ts(i * step - 0.2)}\n{text.format(i=i)}\n"
                  for i in range(1, n + 1)]
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(blocks))


class FakeDub:
    def spawn(self, job_id, input_name, target_lang, mode):
        threading.Thread(target=self._run, args=(job_id, input_name, target_lang), daemon=True).start()

    def _run(self, job_id, input_name, lang):
        def put(**kw):
            cur = dict(STATUS.get(job_id, {}))
            cur.update(kw)
            STATUS[job_id] = cur
        t0 = time.time()
        log, ref = [], None
        put(status="running", stage="starting", started_at=t0)
        for stage, line, dwell in SCRIPT:
            log.append(line.format(lang=lang))
            put(stage=stage.format(lang=lang), log="\n".join(log), heartbeat=time.time())
            time.sleep(dwell)
            if stage.startswith("step 5/7"):
                ref = _resolve_voice(lang, log.append)
                put(log="\n".join(log))
        src = os.path.join(JOBS_DIR, job_id, input_name)
        out = os.path.join(JOBS_DIR, job_id, f"dubbed_{lang.lower()}.mp4")
        if not (ref and ref.get("audio") and _mux_reference(src, ref["audio"], out, log.append)):
            shutil.copyfile(src, out)
        _write_fake_transcripts(os.path.dirname(out), lang, STATUS[job_id].get("input_seconds") or 12.0)
        # Fake meters in the exact shape modal_app writes (numbers are illustrative only).
        put(meter={"gpu_transcribe_s": 24.0, "gpu_transcribe_n": 1},
            tts_meter={"workers": [{"task": "ta-1", "up_s": 21.0, "n_segments": 3},
                                   {"task": "ta-2", "up_s": 19.5, "n_segments": 3}],
                       "n_shards": 2, "scaledown_s": 5})
        put(status="done", stage="complete", output=out, video_seconds=STATUS[job_id].get("input_seconds"),
            finished_at=time.time(), elapsed_s=round(time.time() - t0, 1),
            log="\n".join(log + ["Done."]))
        time.sleep(1.0)   # the orchestrator closes its meter just after, as on Modal (vc path)
        m = dict(STATUS[job_id]["meter"]); m["cpu_orchestrator_s"] = round(time.time() - t0, 1)
        put(meter=m, metered=True)


class FakeFetch:
    """The REAL deploy/youtube_fetch.run_fetch_job in a thread: real yt-dlp, real metadata and
    duration gates, real download — from this machine's IP, which YouTube does not block the
    way it blocks Modal's. Only the dub after it is simulated (FakeDub)."""
    def __init__(self, dub):
        self.dub = dub

    def spawn(self, job_id, url, target_lang, mode, plan, user, extra):
        from deploy.youtube_fetch import run_fetch_job
        threading.Thread(target=run_fetch_job, daemon=True, kwargs=dict(
            job_id=job_id, url=url, target_lang=target_lang, mode=mode, plan=plan, user=user,
            job_status=STATUS, jobs_vol=FakeVol(), jobs_dir=JOBS_DIR, dub_video=self.dub,
            extra=extra)).start()


def main():
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voice-dir", help="curated voice-reference directory (sets DUBBING_VOICE_REF_DIR)")
    args = ap.parse_args()
    if args.voice_dir:
        os.environ["DUBBING_VOICE_REF_DIR"] = os.path.abspath(args.voice_dir)
        from deploy.demo_ui import voice_coverage
        print(f"Voice references from {os.environ['DUBBING_VOICE_REF_DIR']}: {voice_coverage()}")
    os.makedirs(JOBS_DIR, exist_ok=True)
    dub = FakeDub()
    try:
        import yt_dlp  # noqa: F401
        fetch = FakeFetch(dub)
    except ImportError:
        fetch = None
        print("yt-dlp not installed: the YouTube-link tab is hidden (pip install yt-dlp).")
    app = build_api(dub_video=dub, job_status=STATUS, jobs_vol=FakeVol(), jobs_dir=JOBS_DIR,
                    fetch_youtube=fetch)
    port = int(os.environ.get("PORT", "8765"))
    print(f"Demo UI (SIMULATED) on http://localhost:{port}  — access code: "
          f"{os.environ['DEMO_ACCESS_CODE']}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
