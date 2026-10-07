#!/usr/bin/env python3
"""Spend guards: the layers that stop a flood of dub requests from draining the Modal account.

Where this sits
---------------
`admit_job` (deploy/api.py) is the one admission path for every input (upload, YouTube).
After the duration gate has measured the input, it calls `reserve()` here, BEFORE
`dub_video.spawn`. Anything refused here never reaches a GPU.

  1. kill switch   -- `python -m deploy.spend_guard pause "reason"` stops new jobs at once,
                      no redeploy (a flag in the shared status Dict). Running jobs finish.
  2. caller limits -- per caller: one job in flight, N jobs/day, S input-seconds/day.
  3. daily budget  -- all callers together: input video-seconds per UTC day.

The container cap on the GPU functions (`MODAL_GPU_MAX_CONTAINERS`, deploy/modal_app.py) and the
Modal workspace budget sit behind these as the hard backstops: this module is a counter in a
`modal.Dict`, and a Dict has no atomic increment, so two simultaneous requests can both read the
same total and both pass. The overshoot is bounded by the number of API containers, never by the
number of requests.

Pre-registered limits (CLAUDE.md rule 9): the numbers below are written before launch, as env
overridable defaults, so tightening them is a deliberate edit and not a mid-incident guess. At
the measured ~$0.10 per dub, DAILY_VIDEO_SECONDS=3600 is roughly 12 five-minute dubs, i.e.
about $1.20 of GPU a day at the ceiling.

Fails closed: if the counter store cannot be read or written the request is refused (503), not
waved through (rule 1: a check that cannot run has not passed).

No Modal dependency at import: the store is any mapping (`job_status`), so the whole module is
unit-tested on CPU with a plain dict.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time

PAUSE_KEY = "guard:paused"

# In-flight job states and how long one may be believed to still be running. dub_video's timeout
# is 40 min, so a record older than this is stale (crashed worker) and must not lock a caller out.
ACTIVE_STATES = ("queued", "running", "fetching")
STALE_AFTER_S = 45 * 60

# Per-plan multiplier on the caller limits. "" / DEMO are the strictest. RapidAPI enforces its own
# call quotas on top; these only bound GPU-seconds per caller.
PLAN_CALLER_MULT = {"": 1, "DEMO": 1, "BASIC": 3, "PRO": 10, "ULTRA": 30}

_lock = threading.Lock()    # serialises read-modify-write inside ONE container only


class GuardRefused(Exception):
    """A request was refused by a spend guard. ``status_code`` is the HTTP code to return."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def limits() -> dict:
    """The active limits, read at call time so a test (or an env change) takes effect."""
    return {
        "daily_video_seconds": _int_env("DUB_DAILY_VIDEO_SECONDS", 3600),
        "caller_jobs_per_day": _int_env("DUB_CALLER_JOBS_PER_DAY", 3),
        "caller_seconds_per_day": _int_env("DUB_CALLER_SECONDS_PER_DAY", 600),
    }


def utc_day(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now if now is not None else time.time()))


def caller_id(raw: str | None) -> str:
    """A stable, non-reversible id for a caller key (invite code, RapidAPI user, client IP), so
    the store never holds an IP address or a code in the clear."""
    return hashlib.sha256((raw or "anonymous").encode()).hexdigest()[:16]


# ── kill switch ─────────────────────────────────────────────────────────────────────────────
def pause(store, reason: str = "") -> dict:
    rec = {"paused": True, "reason": reason, "at": time.time()}
    store[PAUSE_KEY] = rec
    return rec


def resume(store) -> None:
    store[PAUSE_KEY] = {"paused": False, "at": time.time()}


def paused(store) -> dict | None:
    rec = store.get(PAUSE_KEY)
    return rec if rec and rec.get("paused") else None


# ── admission ───────────────────────────────────────────────────────────────────────────────
def _inflight(store, cid: str, now: float) -> bool:
    job_id = store.get(f"guard:inflight:{cid}")
    if not job_id:
        return False
    st = store.get(job_id) or {}
    if st.get("status") not in ACTIVE_STATES:
        return False
    return (now - float(st.get("created_at") or 0)) < STALE_AFTER_S


def reserve(store, *, job_id: str, seconds: float, cid: str, plan: str = "",
            exempt_caller_limits: bool = False, now: float | None = None) -> None:
    """Refuse (GuardRefused) or record this job's seconds against the budgets.

    ``cid`` is a caller_id() hash, never a raw IP or code.

    Order matters: the cheapest, most global refusal first (kill switch), then the per-caller
    ones (so one noisy caller is told "slow down" rather than "service full"), then the shared
    daily budget. Nothing is written unless every check passed.
    """
    now = now if now is not None else time.time()
    try:
        _reserve(store, job_id, float(seconds), cid, (plan or "").upper(),
                 exempt_caller_limits, now)
    except GuardRefused:
        raise
    except Exception as e:
        # The counter store failed. A budget that cannot be read is not a budget that passed.
        raise GuardRefused(503, "Dubbing is temporarily unavailable (usage limits could not be "
                                f"checked: {type(e).__name__}). Please try again shortly.")


def _reserve(store, job_id, seconds, cid, plan, exempt, now) -> None:
    lim = limits()
    day = utc_day(now)
    mult = PLAN_CALLER_MULT.get(plan, 1)

    with _lock:
        p = paused(store)
        if p:
            why = f" ({p['reason']})" if p.get("reason") else ""
            raise GuardRefused(503, f"New dubs are paused right now{why}. Please check back later.")

        ckey = f"guard:caller:{day}:{cid}"
        crec = dict(store.get(ckey) or {"jobs": 0, "seconds": 0.0})
        if not exempt:
            if _inflight(store, cid, now):
                raise GuardRefused(429, "You already have a dub in progress. "
                                        "Wait for it to finish, then start the next one.")
            if crec["jobs"] >= lim["caller_jobs_per_day"] * mult:
                raise GuardRefused(429, f"Daily limit reached: {lim['caller_jobs_per_day'] * mult} "
                                        "dubs per day. It resets at 00:00 UTC.")
            if crec["seconds"] + seconds > lim["caller_seconds_per_day"] * mult:
                left = max(0, lim["caller_seconds_per_day"] * mult - crec["seconds"])
                raise GuardRefused(429, f"This clip is {seconds:.0f}s but you have {left:.0f}s of "
                                        "video left today. It resets at 00:00 UTC.")

        dkey = f"guard:day:{day}"
        drec = dict(store.get(dkey) or {"jobs": 0, "seconds": 0.0})
        if drec["seconds"] + seconds > lim["daily_video_seconds"]:
            raise GuardRefused(503, "Today's free dubbing capacity is used up. "
                                    "It resets at 00:00 UTC; please try again tomorrow.")

        # Every check passed: record. Written last so a refusal leaves no trace.
        drec["jobs"] += 1
        drec["seconds"] = round(drec["seconds"] + seconds, 2)
        store[dkey] = drec
        crec["jobs"] += 1
        crec["seconds"] = round(crec["seconds"] + seconds, 2)
        store[ckey] = crec
        store[f"guard:inflight:{cid}"] = job_id


def release(store, *, seconds: float, cid: str, now: float | None = None) -> None:
    """Undo a reservation (the spawn that followed it raised). Best effort."""
    try:
        now = now if now is not None else time.time()
        day = utc_day(now)
        with _lock:
            for key in (f"guard:day:{day}", f"guard:caller:{day}:{cid}"):
                rec = dict(store.get(key) or {})
                if rec:
                    rec["jobs"] = max(0, rec.get("jobs", 0) - 1)
                    rec["seconds"] = round(max(0.0, rec.get("seconds", 0.0) - seconds), 2)
                    store[key] = rec
            store[f"guard:inflight:{cid}"] = ""
    except Exception:
        pass


def snapshot(store, now: float | None = None) -> dict:
    """What an operator wants to see: the switch, today's totals against the limits."""
    now = now if now is not None else time.time()
    lim = limits()
    d = store.get(f"guard:day:{utc_day(now)}") or {"jobs": 0, "seconds": 0.0}
    return {"paused": paused(store), "day": utc_day(now), "jobs_today": d["jobs"],
            "video_seconds_today": d["seconds"], "limits": lim,
            "budget_left_s": max(0.0, lim["daily_video_seconds"] - d["seconds"])}


def main(argv: list[str] | None = None) -> int:
    """Operator CLI against the live status Dict:

        python -m deploy.spend_guard status
        python -m deploy.spend_guard pause "bot traffic"
        python -m deploy.spend_guard resume
    """
    import json
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("status", "pause", "resume"):
        print(main.__doc__)
        return 2
    import modal    # only the CLI needs Modal
    store = modal.Dict.from_name(os.environ.get("DUB_STATUS_DICT", "indic-dubbing-status"))
    if argv[0] == "pause":
        pause(store, " ".join(argv[1:]))
    elif argv[0] == "resume":
        resume(store)
    print(json.dumps(snapshot(store), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
