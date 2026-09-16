#!/usr/bin/env python3
"""CPU test for the pre-GPU duration gate in deploy/api.py::prepare_job.

Run:  python deploy/test_api_gate.py     (needs ffmpeg/ffprobe on PATH; no GPU, no Modal)

Measures the PROPERTY that matters (rule 5): does an input that exceeds the plan's
ceiling ever reach dub_video.spawn()? Uses REAL media made with ffmpeg, not mocks of
the probe, so it also exercises probe_duration itself.
"""
import os
import subprocess
import sys
import tempfile

# Repo root is deploy/'s parent, exactly as on Modal (sys.path.insert of the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.api import prepare_job, ApiError, probe_duration  # noqa: E402


class FakeVol:
    def commit(self): pass
    def reload(self): pass


class FakeSpawner:
    def __init__(self): self.calls = []
    def spawn(self, *a): self.calls.append(a)


def make_video(path, seconds):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=d={seconds}:s=160x120:r=10",
         "-f", "lavfi", "-i", f"sine=d={seconds}", "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-shortest", path],
        check=True,
    )
    return path


def run_case(name, *, data, filename, plan, expect_spawn, expect_code=None):
    jobs = tempfile.mkdtemp()
    status, spawner = {}, FakeSpawner()
    err = None
    try:
        prepare_job(job_status=status, jobs_vol=FakeVol(), jobs_dir=jobs, dub_video=spawner,
                    data=data, filename=filename, target_lang="Hindi", mode="basic", plan=plan)
    except ApiError as e:
        err = e

    spawned = len(spawner.calls) > 0
    leftover = [d for d in os.listdir(jobs)] if os.path.isdir(jobs) else []
    ok = (spawned == expect_spawn)
    if expect_code is not None:
        ok = ok and err is not None and err.status_code == expect_code
    if expect_spawn:
        ok = ok and err is None
    # A rejected job must not leave its upload behind on the volume.
    if not expect_spawn:
        ok = ok and leftover == []

    got = f"spawn={spawned}"
    if err:
        got += f", ApiError {err.status_code}: {err.detail[:70]}"
    if leftover:
        got += f", LEFTOVER {leftover}"
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {got}")
    return ok


def main():
    tmp = tempfile.mkdtemp()
    short = make_video(os.path.join(tmp, "short.mp4"), 5)
    long_ = make_video(os.path.join(tmp, "long.mp4"), 200)

    print(f"probe_duration(short) = {probe_duration(short)}s")
    print(f"probe_duration(long)  = {probe_duration(long_)}s")
    print(f"probe_duration(junk)  = {probe_duration(os.path.join(tmp,'nope.mp4'))}\n")

    sb, lb = open(short, "rb").read(), open(long_, "rb").read()
    results = [
        # FREE plan ceiling is 120 s.
        run_case("5s clip on FREE -> accepted", data=sb, filename="a.mp4", plan="",
                 expect_spawn=True),
        run_case("200s clip on FREE (cap 120s) -> REJECTED 413, no GPU", data=lb,
                 filename="b.mp4", plan="", expect_spawn=False, expect_code=413),
        # BASIC ceiling is 600 s, so the same file is allowed there.
        run_case("200s clip on BASIC (cap 600s) -> accepted", data=lb, filename="c.mp4",
                 plan="BASIC", expect_spawn=True),
        # Unprobeable upload: UNKNOWN must not be treated as passed.
        run_case("non-media bytes -> REJECTED 400, no GPU", data=b"x" * 5000,
                 filename="d.mp4", plan="PRO", expect_spawn=False, expect_code=400),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
