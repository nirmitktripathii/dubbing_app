#!/usr/bin/env python3
"""Run the web demo UI locally against a SIMULATED pipeline — no Modal, no GPU, no API calls.

    python deploy/demo_dev_server.py            # then open http://localhost:8765
    (access code: dev)

Builds the real FastAPI app (deploy/api.py + deploy/demo_ui.py) with in-memory fakes in place
of the Modal Dict / Volume / dub_video. The fake job walks the real stage strings
modal_app._parse_stage emits, writes the same meter fields modal_app writes, and "dubs" by
copying the upload — so the page, the progress mapping and the cost maths are exercised end to
end. The page shows a banner saying the figures are simulated.
"""
import os
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DEMO_ACCESS_CODE", "dev")
os.environ["DUB_DEMO_SIMULATED"] = "1"

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


class FakeDub:
    def spawn(self, job_id, input_name, target_lang, mode):
        threading.Thread(target=self._run, args=(job_id, input_name, target_lang), daemon=True).start()

    def _run(self, job_id, input_name, lang):
        def put(**kw):
            cur = dict(STATUS.get(job_id, {}))
            cur.update(kw)
            STATUS[job_id] = cur
        t0 = time.time()
        log = []
        put(status="running", stage="starting", started_at=t0)
        for stage, line, dwell in SCRIPT:
            log.append(line.format(lang=lang))
            put(stage=stage.format(lang=lang), log="\n".join(log), heartbeat=time.time())
            time.sleep(dwell)
        src = os.path.join(JOBS_DIR, job_id, input_name)
        out = os.path.join(JOBS_DIR, job_id, f"dubbed_{lang.lower()}.mp4")
        shutil.copyfile(src, out)
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


def main():
    import uvicorn
    os.makedirs(JOBS_DIR, exist_ok=True)
    app = build_api(dub_video=FakeDub(), job_status=STATUS, jobs_vol=FakeVol(), jobs_dir=JOBS_DIR)
    port = int(os.environ.get("PORT", "8765"))
    print(f"Demo UI (SIMULATED) on http://localhost:{port}  — access code: "
          f"{os.environ['DEMO_ACCESS_CODE']}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
