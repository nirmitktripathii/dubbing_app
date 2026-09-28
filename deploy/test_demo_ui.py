#!/usr/bin/env python3
"""CPU tests for the web demo UI (deploy/demo_ui.py). No GPU, no Modal, no network.

Run:  python deploy/test_demo_ui.py      (needs ffmpeg/ffprobe on PATH, like test_api_gate.py)

Asserts properties, not "it ran" (rule 5):
  - the access code actually gates submission and status (unset -> 503, wrong -> 403);
  - the shareable video link is gated by ITS job's token, and serves the right bytes;
  - a RapidAPI /v1 job is invisible to /ui, and /v1 never leaks a demo job's token;
  - progress never goes backwards across the real stage sequence, and done == 100;
  - the cost estimate equals the hand-computed figure from tools/cost_model's prices,
    dedupes a TTS worker that served two shards, and is not "final" before the
    orchestrator's meter lands;
  - tts_fanout writes the meter file in the shape estimate_cost reads.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from deploy import demo_ui  # noqa: E402
from deploy.api import build_api  # noqa: E402
from tools.cost_model import CPU, L4, MEM  # noqa: E402

FAILS = []


def check(name, cond, got=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n        {got}" if got and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeVol:
    def commit(self): pass
    def reload(self): pass


class Spawner:
    def __init__(self): self.calls = []
    def spawn(self, *a): self.calls.append(a)


def make_video(path, seconds=3):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=d={seconds}:s=160x120:r=10", "-f", "lavfi", "-i", f"sine=d={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", path],
                   check=True)
    return path


def test_access_and_links():
    jobs = tempfile.mkdtemp()
    status, spawner = {}, Spawner()
    client = TestClient(build_api(dub_video=spawner, job_status=status, jobs_vol=FakeVol(),
                                  jobs_dir=jobs))
    os.environ.pop("DUBBING_VOICE_REF_DIR", None)
    voices = client.get("/ui/config").json().get("voices", {})
    check("config reports a voice for every offered language: Hindi pinned, uncurated -> fallback",
          set(voices) == set(demo_ui.LANGUAGES) and voices["Hindi"] == "pinned"
          and voices["Tamil"] == "fallback", voices)

    vid = make_video(os.path.join(tempfile.mkdtemp(), "clip.mp4"))
    data = open(vid, "rb").read()
    files = {"file": ("clip.mp4", data, "video/mp4")}

    os.environ.pop("DEMO_ACCESS_CODE", None)
    r = client.post("/ui/dub", files=files, data={"target_lang": "Hindi"})
    check("no DEMO_ACCESS_CODE -> 503, nothing spawned", r.status_code == 503 and not spawner.calls,
          f"{r.status_code} spawned={len(spawner.calls)}")

    os.environ["DEMO_ACCESS_CODE"] = "s3cret-code"
    r = client.post("/ui/dub", files=files, data={"target_lang": "Hindi"},
                    headers={"X-Demo-Code": "wrong"})
    check("wrong code -> 403, nothing spawned", r.status_code == 403 and not spawner.calls,
          f"{r.status_code}")

    r = client.post("/ui/dub", files=files, data={"target_lang": "Klingon"},
                    headers={"X-Demo-Code": "s3cret-code"})
    check("unknown language -> 400", r.status_code == 400, f"{r.status_code}")

    r = client.post("/ui/dub", files=files, data={"target_lang": "Tamil", "mode": "basic"},
                    headers={"X-Demo-Code": "s3cret-code"})
    ok = r.status_code == 200 and len(spawner.calls) == 1
    check("right code -> job queued, dub_video.spawn called once", ok, f"{r.status_code} {r.text}")
    job = r.json()["job_id"]
    check("spawned with the chosen language", spawner.calls[0][2] == "Tamil", str(spawner.calls[0]))
    check("demo job gated at the DEMO ceiling", status[job]["plan"] == "DEMO"
          and status[job]["max_seconds"] == demo_ui.PLAN_MAX_SECONDS["DEMO"], str(status[job]))

    r = client.get(f"/ui/dub/{job}", headers={"X-Demo-Code": "wrong"})
    check("status with wrong code -> 403", r.status_code == 403, f"{r.status_code}")
    r = client.get(f"/ui/dub/{job}", headers={"X-Demo-Code": "s3cret-code"})
    check("queued status -> 1%, no video link yet",
          r.status_code == 200 and r.json()["percent"] == 1 and "video_url" not in r.json(), r.text)

    # Finish the job the way modal_app._finalize_job does.
    out = os.path.join(jobs, job, "dubbed_tamil.mp4")
    with open(out, "wb") as fh:
        fh.write(b"DUBBED-BYTES")
    status[job] = {**status[job], "status": "done", "stage": "complete", "output": out,
                   "finished_at": time.time(), "log": "Step 7/7: done\nkey AIzaSyA1234567890abcdefghijklmnop"}
    st = client.get(f"/ui/dub/{job}", headers={"X-Demo-Code": "s3cret-code"}).json()
    check("done -> 100% with a video link", st["percent"] == 100 and st.get("video_url"), str(st))
    check("log tail redacts an API-key-shaped string",
          not any("AIza" in l for l in st["log"]) and any("[redacted]" in l for l in st["log"]),
          str(st["log"]))

    r = client.get(st["video_url"])
    check("shareable link serves the dubbed bytes, no code needed",
          r.status_code == 200 and r.content == b"DUBBED-BYTES", f"{r.status_code}")
    r = client.get(st["download_url"])
    check("download link is an attachment",
          "attachment" in r.headers.get("content-disposition", ""), str(r.headers))
    r = client.get(f"/ui/dub/{job}/video?t=forged")
    check("forged token -> 403", r.status_code == 403, f"{r.status_code}")

    # A RapidAPI job (no source/dl_token) is invisible to the demo endpoints.
    status["f" * 32] = {"status": "done", "output": out}
    r = client.get(f"/ui/dub/{'f' * 32}", headers={"X-Demo-Code": "s3cret-code"})
    check("/v1 job invisible to /ui (404)", r.status_code == 404, f"{r.status_code}")
    r = client.get(f"/ui/dub/{'f' * 32}/video?t=")
    check("/v1 job not downloadable via /ui (404)", r.status_code == 404, f"{r.status_code}")
    r = client.get(f"/v1/dub/{job}")
    check("/v1 status never leaks the demo token", "dl_token" not in r.json(), r.text)

    r = client.get("/")
    check("GET / serves the page", r.status_code == 200 and "IndicAI Dubbing" in r.text,
          f"{r.status_code}")
    r = client.get("/openapi.json")
    check("/openapi.json still generates (and hides /ui)",
          r.status_code == 200 and not any(p.startswith("/ui") for p in r.json()["paths"]),
          f"{r.status_code}")


def test_progress_monotonic():
    stages = (["queued"] + ["starting", "transcribing"]
              + [f"step {n}/7 — x" for n in ("1", "2", "3", "4", "5", "6")]
              + [f"step 6/7 — TTS segment {i}/6" for i in range(1, 7)]
              + ["step 6.5/7 — vc", "step 7/7 — merge"])
    pcts = []
    for s in stages:
        st = {"status": "queued"} if s == "queued" else {"status": "running", "stage": s}
        pcts.append(demo_ui.progress_for(st)[0])
    pcts.append(demo_ui.progress_for({"status": "done"})[0])
    check("progress never decreases across the real stage order, ends at 100",
          all(b >= a for a, b in zip(pcts, pcts[1:])) and pcts[-1] == 100, str(pcts))
    pct, desc = demo_ui.progress_for({"status": "failed", "stage": "step 4/7 — x"})
    check("failure keeps the stage's percent and says failed", pct == 24 and "Failed" in desc,
          f"{pct} {desc}")


def test_cost_math():
    gpu = L4 + 1.0 * CPU + 4.0 * MEM
    cpu = 0.5 * CPU + 2.0 * MEM
    st = {"status": "done", "input_seconds": 60.0, "metered": False,
          "meter": {"gpu_transcribe_s": 20.0, "gpu_transcribe_n": 1, "cpu_orchestrator_s": 100.0},
          # worker ta-1 served two shards: its age is the larger value, counted once.
          "tts_meter": {"scaledown_s": 5, "workers": [
              {"task": "ta-1", "up_s": 10.0}, {"task": "ta-1", "up_s": 30.0},
              {"task": "ta-2", "up_s": 25.0}]}}
    c = demo_ui.estimate_cost(st)
    expect = ((20 + 5) * gpu + (30 + 25 + 2 * 5) * gpu
              + (100 + demo_ui.MODAL_DEFAULT_SCALEDOWN_S) * cpu)
    check("total equals the hand-computed cost from cost_model prices",
          abs(c["total_usd"] - round(expect, 4)) < 1e-4, f"{c['total_usd']} vs {expect:.5f}")
    check("per-minute figure = total / clip length x 60",
          abs(c["usd_per_video_minute"] - round(c["total_usd"], 4)) < 2e-4, str(c))
    check("TTS worker serving 2 shards counted once",
          any("2 worker" in l["label"] for l in c["lines"]), str(c["lines"]))
    check("not final until the orchestrator's meter lands", c["final"] is False, str(c["final"]))
    check("final once metered and terminal",
          demo_ui.estimate_cost({**st, "metered": True})["final"] is True)
    check("no meters -> no cost (not $0)", demo_ui.estimate_cost({"status": "done"}) is None)


def test_fanout_meter_file():
    from deploy import tts_fanout
    d = tempfile.mkdtemp()
    tts_fanout._write_meter([{"meter": {"task": "a", "up_s": 12.5, "n_segments": 3}},
                             {"entries": {}, "wavs": {}}], d)
    with open(os.path.join(d, "tts_meter.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    c = demo_ui.estimate_cost({"status": "running", "tts_meter": m})
    check("tts_fanout meter file round-trips into the estimate",
          m["n_shards"] == 2 and len(m["workers"]) == 1 and c and c["lines"][0]["seconds"] ==
          12.5 + m["scaledown_s"], json.dumps(m))


if __name__ == "__main__":
    test_progress_monotonic()
    test_cost_math()
    try:
        test_fanout_meter_file()
    except ImportError as e:   # tts_fanout imports pipeline.duration_tts (torch); skip if absent
        print(f"[SKIP] tts_fanout meter file ({e})")
    test_access_and_links()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    raise SystemExit(1 if FAILS else 0)
