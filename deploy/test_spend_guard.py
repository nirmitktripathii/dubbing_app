#!/usr/bin/env python3
"""CPU test for the spend guards (deploy/spend_guard.py) and their wiring in deploy/api.py.

Run:  python deploy/test_spend_guard.py     (needs ffmpeg/ffprobe; no GPU, no Modal)

Measures the PROPERTY (rule 5): after a guard refuses a request, nothing was spawned, the upload
is gone from the volume, and no budget was consumed. Real media through the real admit_job; the
Modal Dict is a plain dict. Time is passed in where the day matters.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy import spend_guard as sg  # noqa: E402
from deploy.api import ApiError, build_api, prepare_job  # noqa: E402
from deploy.test_api_gate import FakeSpawner, FakeVol, make_video  # noqa: E402

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {detail}" if detail and not ok else ""))


def submit(status, spawner, video, *, caller, plan="BASIC", user=None):
    """One upload through prepare_job. Returns (job_id|None, ApiError|None, jobs_dir)."""
    jobs = tempfile.mkdtemp()
    try:
        r = prepare_job(job_status=status, jobs_vol=FakeVol(), jobs_dir=jobs, dub_video=spawner,
                        data=open(video, "rb").read(), filename="in.mp4", target_lang="Hindi",
                        mode="basic", plan=plan, user=user or caller, caller=caller)
        return r["job_id"], None, jobs
    except ApiError as e:
        return None, e, jobs


def set_env(**kw):
    for k, v in kw.items():
        os.environ[k] = str(v)


def clear_env():
    for k in ("DUB_DAILY_VIDEO_SECONDS", "DUB_CALLER_JOBS_PER_DAY", "DUB_CALLER_SECONDS_PER_DAY",
              "RAPIDAPI_PROXY_SECRET", "DUB_ALLOW_UNAUTH_API"):
        os.environ.pop(k, None)


def main():
    work = tempfile.mkdtemp()
    v5 = make_video(os.path.join(work, "v5.mp4"), 5)

    # ── kill switch ─────────────────────────────────────────────────────────────────────
    clear_env()
    st, sp = {}, FakeSpawner()
    sg.pause(st, "bot traffic")
    jid, err, jobs = submit(st, sp, v5, caller="a")
    check("paused -> 503, nothing spawned, upload removed, no budget used",
          err is not None and err.status_code == 503 and "paused" in err.detail
          and not sp.calls and os.listdir(jobs) == [] and not any(k.startswith("guard:day") for k in st),
          f"{err and (err.status_code, err.detail)} calls={len(sp.calls)}")
    sg.resume(st)
    jid, err, _ = submit(st, sp, v5, caller="a")
    check("resume -> the same request is accepted and spawned", err is None and len(sp.calls) == 1)

    # ── one in flight per caller ────────────────────────────────────────────────────────
    clear_env(); set_env(DUB_CALLER_JOBS_PER_DAY=99, DUB_CALLER_SECONDS_PER_DAY=9999)
    st, sp = {}, FakeSpawner()
    j1, e1, _ = submit(st, sp, v5, caller="a")
    j2, e2, _ = submit(st, sp, v5, caller="a")
    check("second job while the first is queued -> 429, not spawned",
          e1 is None and e2 is not None and e2.status_code == 429 and len(sp.calls) == 1,
          f"{e2 and (e2.status_code, e2.detail)}")
    j3, e3, _ = submit(st, sp, v5, caller="b")
    check("a different caller is not blocked by caller a", e3 is None and len(sp.calls) == 2)
    st[j1]["status"] = "done"
    j4, e4, _ = submit(st, sp, v5, caller="a")
    check("once the first is done the caller can submit again", e4 is None and len(sp.calls) == 3)
    st[j4]["created_at"] -= sg.STALE_AFTER_S + 5     # a crashed worker never marks it done
    j5, e5, _ = submit(st, sp, v5, caller="a")
    check("a stale 'running' record does not lock a caller out forever",
          e5 is None and len(sp.calls) == 4)

    # ── jobs/day and seconds/day per caller ─────────────────────────────────────────────
    clear_env(); set_env(DUB_CALLER_JOBS_PER_DAY=2, DUB_CALLER_SECONDS_PER_DAY=9999)
    st, sp = {}, FakeSpawner()
    for _ in range(2):
        j, e, _ = submit(st, sp, v5, caller="a", plan="DEMO")
        st[j]["status"] = "done"
    j, e, jobs = submit(st, sp, v5, caller="a", plan="DEMO")
    check("3rd job in a day with a limit of 2 -> 429 and cleaned up",
          e is not None and e.status_code == 429 and len(sp.calls) == 2 and os.listdir(jobs) == [],
          f"{e and e.detail}")
    j, e, _ = submit(st, sp, v5, caller="a", plan="PRO")
    check("a PRO plan gets 10x the caller limit", e is None, f"{e and e.detail}")

    clear_env(); set_env(DUB_CALLER_JOBS_PER_DAY=99, DUB_CALLER_SECONDS_PER_DAY=8)
    st, sp = {}, FakeSpawner()
    j, e, _ = submit(st, sp, v5, caller="a", plan="DEMO")
    st[j]["status"] = "done"
    j, e, _ = submit(st, sp, v5, caller="a", plan="DEMO")
    check("5s + 5s against 8s/day -> second refused (429), says how much is left",
          e is not None and e.status_code == 429 and "3s" in e.detail, f"{e and e.detail}")

    # ── daily budget across callers ─────────────────────────────────────────────────────
    clear_env(); set_env(DUB_DAILY_VIDEO_SECONDS=12, DUB_CALLER_JOBS_PER_DAY=99,
                         DUB_CALLER_SECONDS_PER_DAY=9999)
    st, sp = {}, FakeSpawner()
    ok = [submit(st, sp, v5, caller=c)[1] is None for c in ("a", "b")]
    before = dict(st[f"guard:day:{sg.utc_day()}"])
    j, e, jobs = submit(st, sp, v5, caller="c")
    after = st[f"guard:day:{sg.utc_day()}"]
    check("budget 12s: two 5s jobs fit, the third is refused 503 with no side effects",
          ok == [True, True] and e is not None and e.status_code == 503
          and len(sp.calls) == 2 and os.listdir(jobs) == [] and before == after,
          f"{e and e.detail} before={before} after={after}")
    check("a refusal leaves no per-caller record behind",
          not any(k.startswith("guard:caller") and k.endswith(sg.caller_id("c")) for k in st))

    # ── the owner skips caller limits, not the switch or the budget ─────────────────────
    clear_env(); set_env(DUB_DAILY_VIDEO_SECONDS=12, DUB_CALLER_JOBS_PER_DAY=1)
    st, sp = {}, FakeSpawner()
    r = [submit(st, sp, v5, caller="o", plan="DEMO", user="demo-owner")[1] for _ in range(2)]
    check("owner: two in a row despite 1 job/day and in-flight", r == [None, None])
    j, e, _ = submit(st, sp, v5, caller="o", plan="DEMO", user="demo-owner")
    check("owner is still bound by the daily budget", e is not None and e.status_code == 503)
    sg.pause(st)
    clear_env(); set_env(DUB_DAILY_VIDEO_SECONDS=999)
    j, e, _ = submit(st, sp, v5, caller="o", plan="DEMO", user="demo-owner")
    check("owner is still bound by the kill switch", e is not None and e.status_code == 503)

    # ── spawn failure gives the budget back ─────────────────────────────────────────────
    clear_env(); set_env(DUB_DAILY_VIDEO_SECONDS=12)

    class BoomSpawner:
        def spawn(self, *a): raise RuntimeError("modal down")

    st = {}
    try:
        submit(st, BoomSpawner(), v5, caller="a")
    except RuntimeError:
        pass
    d = st.get(f"guard:day:{sg.utc_day()}")
    check("spawn raises -> reservation released (0 jobs, 0 s)", d == {"jobs": 0, "seconds": 0.0}, f"{d}")
    j, e, _ = submit(st, FakeSpawner(), v5, caller="a")
    check("...and the caller is not stuck 'in flight' by the failed spawn", e is None, f"{e and e.detail}")

    # ── fail closed when the counter store breaks ───────────────────────────────────────
    class BrokenStore(dict):
        def get(self, key, *a):
            if str(key).startswith("guard:"):
                raise ConnectionError("dict unreachable")
            return super().get(key, *a)

    clear_env()
    sp = FakeSpawner()
    j, e, jobs = submit(BrokenStore(), sp, v5, caller="a")
    check("unreadable counters -> refused 503, not waved through",
          e is not None and e.status_code == 503 and not sp.calls, f"{e and e.detail}")

    # ── the day rolls over ──────────────────────────────────────────────────────────────
    clear_env(); set_env(DUB_DAILY_VIDEO_SECONDS=10, DUB_CALLER_JOBS_PER_DAY=1)
    st = {}
    t0 = 1_800_000_000.0
    sg.reserve(st, job_id="x", seconds=10, cid="c1", now=t0)
    try:
        sg.reserve(st, job_id="y", seconds=1, cid="c2", now=t0 + 60)
        same_day = "allowed"
    except sg.GuardRefused as e:
        same_day = e.status_code
    sg.reserve(st, job_id="z", seconds=10, cid="c1", now=t0 + 86400)
    check("budget is per UTC day: full today (503), fresh tomorrow",
          same_day == 503 and sg.snapshot(st, t0 + 86400)["video_seconds_today"] == 10)

    # ── caller ids never hold the raw key ───────────────────────────────────────────────
    check("caller_id hashes (no raw IP in the store)",
          "203.0.113.9" not in sg.caller_id("demo-ui:203.0.113.9") and len(sg.caller_id("x")) == 16)

    # ── /v1 auth is fail-closed ─────────────────────────────────────────────────────────
    from fastapi.testclient import TestClient

    def client(**env):
        clear_env(); set_env(**env)
        return TestClient(build_api(dub_video=FakeSpawner(), job_status={}, jobs_vol=FakeVol(),
                                    jobs_dir=tempfile.mkdtemp()))

    c = client()
    check("no RAPIDAPI_PROXY_SECRET -> /v1 is 503, not open",
          c.get("/v1/dub/abc").status_code == 503 and c.post("/v1/dub").status_code == 503)
    c = client(DUB_ALLOW_UNAUTH_API=1)
    check("dev escape DUB_ALLOW_UNAUTH_API=1 reaches the handler (404 unknown job)",
          c.get("/v1/dub/abc").status_code == 404)
    c = client(RAPIDAPI_PROXY_SECRET="s3cret")
    check("secret set: missing -> 403, wrong -> 403, right -> handler",
          c.get("/v1/dub/abc").status_code == 403
          and c.get("/v1/dub/abc", headers={"X-RapidAPI-Proxy-Secret": "nope"}).status_code == 403
          and c.get("/v1/dub/abc", headers={"X-RapidAPI-Proxy-Secret": "s3cret"}).status_code == 404)

    # ── end to end through /v1: per-RapidAPI-user limits ────────────────────────────────
    clear_env(); set_env(RAPIDAPI_PROXY_SECRET="s3cret")
    sp = FakeSpawner()
    c = TestClient(build_api(dub_video=sp, job_status={}, jobs_vol=FakeVol(), jobs_dir=tempfile.mkdtemp()))

    def post(user):
        with open(v5, "rb") as fh:
            return c.post("/v1/dub", files={"file": ("in.mp4", fh, "video/mp4")},
                          headers={"X-RapidAPI-Proxy-Secret": "s3cret", "X-RapidAPI-User": user,
                                   "X-RapidAPI-Subscription": "BASIC"})
    r1, r2, r3 = post("alice"), post("alice"), post("bob")
    check("/v1: alice ok, alice again 429 (one in flight), bob ok",
          (r1.status_code, r2.status_code, r3.status_code) == (200, 429, 200),
          f"{r1.status_code} {r2.status_code} {r3.status_code} {r2.text[:120]}")

    clear_env()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    if FAILED:
        print("FAILED:", *FAILED, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
