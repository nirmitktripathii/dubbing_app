#!/usr/bin/env python3
"""CPU tests for YouTube-link inputs (deploy/youtube_fetch.py + POST /ui/dub/youtube).
No GPU, no Modal, no network: yt-dlp is replaced by a fake that records what it was asked.

Run:  python deploy/test_youtube_fetch.py      (needs ffmpeg/ffprobe and yt-dlp installed)

Asserts properties, not "it ran" (rule 5):
  - only YouTube video links parse, and what reaches yt-dlp is ALWAYS the canonical
    youtube.com watch URL rebuilt from the id — never the user's string;
  - a video over the limit, private, unlisted, live, age-restricted or of UNKNOWN length is
    rejected from metadata alone: the fake records that no download was attempted;
  - a file that passes the metadata gate but is actually longer is still rejected by the
    shared ffprobe gate, and no dub is spawned;
  - YouTube's bot wall becomes an actionable message; cookies are written to a temp file
    for yt-dlp and deleted after, and never appear in the job's log or status;
  - the route needs the access code, refuses non-YouTube links, and spawns the fetch with the
    canonical URL — returning at once with a "Fetching" status.
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from yt_dlp.utils import DownloadError  # noqa: E402

from deploy import demo_ui, youtube_fetch as yf  # noqa: E402
from deploy.api import build_api  # noqa: E402

FAILS = []
VID = "jNQXAC9IVRw"


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


def make_video(path, seconds):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=d={seconds}:s=160x120:r=10", "-f", "lavfi", "-i", f"sine=d={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", path],
                   check=True)
    return path


def fake_ydl(info=None, file_seconds=3, meta_error=None, dl_error=None):
    """A YoutubeDL stand-in. Records every URL and whether a download was attempted."""
    rec = {"urls": [], "downloads": 0, "opts": [], "cookie_seen": None}

    class Fake:
        def __init__(self, opts):
            self.opts = opts
            rec["opts"].append(opts)
            cf = opts.get("cookiefile")
            if cf:
                rec["cookie_seen"] = open(cf, encoding="utf-8").read()
                rec["cookie_path"] = cf

        def __enter__(self): return self
        def __exit__(self, *a): return False

        def extract_info(self, url, download=False):
            rec["urls"].append(url)
            if meta_error:
                raise DownloadError(meta_error)
            return info

        def download(self, urls):
            rec["urls"].extend(urls)
            rec["downloads"] += 1
            if dl_error:
                raise DownloadError(dl_error)
            out = self.opts["outtmpl"].replace("%(id)s", VID).replace("%(ext)s", "mp4")
            make_video(out, file_seconds)
    return Fake, rec


GOOD = {"id": VID, "title": "Me at the zoo", "channel": "jawed", "duration": 19,
        "availability": "public", "live_status": "not_live", "age_limit": 0}


def test_parse():
    ok = {
        f"https://www.youtube.com/watch?v={VID}": VID,
        f"https://youtube.com/watch?v={VID}&t=42s&list=PL123": VID,
        f"youtu.be/{VID}?si=abc": VID,
        f"https://m.youtube.com/watch?v={VID}": VID,
        f"https://www.youtube.com/shorts/{VID}": VID,
        f"https://www.youtube.com/embed/{VID}": VID,
        f"https://www.youtube-nocookie.com/embed/{VID}": VID,
        f"https://music.youtube.com/watch?v={VID}": VID,
        f"http://www.youtube.com/live/{VID}": VID,
    }
    got = {u: _parse(u) for u in ok}
    check("every YouTube video-link form parses to its id", got == ok, str(got))
    bad = ["", "not a url", f"https://evil.com/watch?v={VID}",
           f"https://youtube.com.evil.com/watch?v={VID}", f"https://www.youtube.com@evil.com/{VID}",
           f"https://user:pw@www.youtube.com/watch?v={VID}", f"https://www.youtube.com:8080/watch?v={VID}",
           f"ftp://www.youtube.com/watch?v={VID}", "https://www.youtube.com/playlist?list=PL123",
           "https://www.youtube.com/@channel", "https://www.youtube.com/watch?v=short",
           "file:///etc/passwd", "http://169.254.169.254/latest/meta-data/"]
    leaked = [u for u in bad if _parse(u) is not None]
    check("non-YouTube, credentialed, odd-port, playlist-only and malformed links are refused",
          not leaked, str(leaked))


def _parse(u):
    try:
        return yf.parse_youtube_url(u)
    except yf.FetchRejected:
        return None


def test_metadata_gate_before_download():
    cases = {
        "over the limit": ({**GOOD, "duration": 301}, "limit is 5:00"),
        "private": ({**GOOD, "availability": "private"}, "private"),
        "unlisted": ({**GOOD, "availability": "unlisted"}, "unlisted"),
        "unknown visibility": ({**GOOD, "availability": None}, "Couldn't confirm"),
        "live": ({**GOOD, "live_status": "is_live", "is_live": True}, "Live streams"),
        "upcoming premiere": ({**GOOD, "live_status": "is_upcoming"}, "Live streams"),
        "age-restricted": ({**GOOD, "age_limit": 18}, "age-restricted"),
        "unknown duration": ({**GOOD, "duration": None}, "didn't report"),
        "playlist result": ({"_type": "playlist", "entries": []}, "playlist"),
    }
    for name, (info, expect) in cases.items():
        Fake, rec = fake_ydl(info)
        d = tempfile.mkdtemp()
        try:
            yf.fetch(f"https://youtu.be/{VID}", d, 300, log_fn=lambda m: None, ydl_cls=Fake)
            check(f"metadata gate rejects {name}", False, "no rejection")
        except yf.FetchRejected as e:
            check(f"metadata gate rejects {name} — and downloads nothing",
                  expect in str(e) and rec["downloads"] == 0 and not os.listdir(d),
                  f"{e} | downloads={rec['downloads']}")
    Fake, rec = fake_ydl({**GOOD, "duration": 300})
    path, meta = yf.fetch(f"https://www.youtube.com/watch?v={VID}&list=PLx&t=9", tempfile.mkdtemp(),
                          300, log_fn=lambda m: None, ydl_cls=Fake)
    check("exactly at the limit (300 s) is accepted", os.path.isfile(path) and meta["id"] == VID)
    check("yt-dlp only ever sees the canonical watch URL (playlist/timestamp dropped)",
          set(rec["urls"]) == {yf.canonical_url(VID)}, str(rec["urls"]))
    dl_opts = rec["opts"][-1]
    check("the download itself re-checks duration (match_filter) and refuses playlists",
          dl_opts.get("match_filter") is not None and dl_opts.get("noplaylist") is True)


def test_run_job_shared_gate_and_secrets():
    jobs = tempfile.mkdtemp()
    # Metadata says 19 s, but the file that arrives is 6 s over a 5 s plan limit -> rejected by
    # the SAME ffprobe gate uploads pass; no dub spawned.
    from deploy import api
    api.PLAN_MAX_SECONDS["TINY"] = 5
    Fake, rec = fake_ydl({**GOOD, "duration": 4}, file_seconds=6)
    status, spawner = {}, Spawner()
    ok = yf.run_fetch_job(job_id="j1", url=f"https://youtu.be/{VID}", target_lang="Tamil",
                          mode="basic", plan="TINY", user="demo-ui", job_status=status,
                          jobs_vol=FakeVol(), jobs_dir=jobs, dub_video=spawner, ydl_cls=Fake)
    st = status["j1"]
    check("a file longer than its metadata claimed is stopped by the shared ffprobe gate",
          ok is False and st["status"] == "failed" and "allows 5s" in st["error"]
          and not spawner.calls and not os.path.exists(os.path.join(jobs, "j1")), str(st.get("error")))

    # Happy path with cookies: spawned with the fetched file, status carries the youtube meta,
    # cookies reached yt-dlp via a temp file that is gone afterwards and never logged.
    secret = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tSEKRET123"
    os.environ["YTDLP_COOKIES"] = secret
    try:
        Fake, rec = fake_ydl(GOOD, file_seconds=3)
        status, spawner = {"j2": {"source": "demo-ui", "dl_token": "tok", "created_at": 1.0}}, Spawner()
        ok = yf.run_fetch_job(job_id="j2", url=yf.canonical_url(VID), target_lang="Tamil",
                              mode="basic", plan="DEMO", user="demo-ui", job_status=status,
                              jobs_vol=FakeVol(), jobs_dir=jobs, dub_video=spawner,
                              extra={"source": "demo-ui", "dl_token": "tok"}, ydl_cls=Fake)
    finally:
        os.environ.pop("YTDLP_COOKIES", None)
    st = status["j2"]
    check("an in-limit public video is fetched, admitted and spawned once",
          ok is True and st["status"] == "queued" and len(spawner.calls) == 1
          and spawner.calls[0][1] == f"youtube_{VID}.mp4" and 2.5 < st["input_seconds"] < 3.5,
          f"{st.get('status')} {spawner.calls}")
    check("admission keeps the fetch's record (token, created_at, youtube meta, log)",
          st["dl_token"] == "tok" and st["created_at"] == 1.0 and st["youtube"]["title"] == GOOD["title"]
          and "[youtube] OK" in st["log"])
    check("cookies reach yt-dlp through a temp file that is deleted afterwards",
          rec["cookie_seen"] == secret and not os.path.exists(rec["cookie_path"]))
    check("the cookie value never appears in the job's status or log",
          "SEKRET123" not in repr(st))

    # The bot wall -> an actionable message, failed on CPU, nothing spawned.
    Fake, _ = fake_ydl(meta_error="ERROR: [youtube] x: Sign in to confirm you’re not a bot. Use --cookies")
    status, spawner = {}, Spawner()
    yf.run_fetch_job(job_id="j3", url=yf.canonical_url(VID), target_lang="Hindi", mode="basic",
                     plan="DEMO", user="u", job_status=status, jobs_vol=FakeVol(), jobs_dir=jobs,
                     dub_video=spawner, ydl_cls=Fake)
    check("YouTube's bot wall fails the job with the YTDLP_COOKIES / YTDLP_PROXY instruction",
          status["j3"]["status"] == "failed" and "YTDLP_COOKIES" in status["j3"]["error"]
          and not spawner.calls, status["j3"].get("error"))
    Fake, rec = fake_ydl(GOOD, dl_error="ERROR: unable to download video data: HTTP Error 403: Forbidden")
    status4, spawner4 = {}, Spawner()
    yf.run_fetch_job(job_id="j4", url=yf.canonical_url(VID), target_lang="Hindi", mode="basic",
                     plan="DEMO", user="u", job_status=status4, jobs_vol=FakeVol(), jobs_dir=jobs,
                     dub_video=spawner4, ydl_cls=Fake)
    check("metadata allowed but media refused (403) gets the same access instruction",
          rec["downloads"] == 1 and "YTDLP_COOKIES" in status4["j4"]["error"] and not spawner4.calls,
          status4["j4"].get("error"))
    pct, desc = demo_ui.progress_for(status["j3"])
    check("a failed fetch reads as such on the page", "Couldn't fetch" in desc, desc)
    shutil.rmtree(jobs, ignore_errors=True)


def test_route():
    jobs = tempfile.mkdtemp()
    os.environ["DEMO_ACCESS_CODE"] = "letmein"
    H = {"X-Demo-Code": "letmein"}
    off = TestClient(build_api(dub_video=Spawner(), job_status={}, jobs_vol=FakeVol(), jobs_dir=jobs))
    check("config hides the YouTube tab when no fetcher is wired",
          off.get("/ui/config").json()["youtube"] is False)
    r = off.post("/ui/dub/youtube", headers=H, json={"url": yf.canonical_url(VID)})
    check("no fetcher -> 503, not a silent no-op", r.status_code == 503, r.text)

    os.environ.pop("DUB_YOUTUBE_DIRECT", None)
    hidden = TestClient(build_api(dub_video=Spawner(), job_status={}, jobs_vol=FakeVol(),
                                  jobs_dir=jobs, fetch_youtube=Spawner()))
    check("fetcher wired but no YouTube access configured -> tab hidden, submit 503",
          hidden.get("/ui/config").json()["youtube"] is False and hidden.post(
              "/ui/dub/youtube", headers=H, json={"url": yf.canonical_url(VID)}).status_code == 503)
    os.environ["DUB_YOUTUBE_DIRECT"] = "1"
    status, fetcher, dub = {}, Spawner(), Spawner()
    c = TestClient(build_api(dub_video=dub, job_status=status, jobs_vol=FakeVol(), jobs_dir=jobs,
                             fetch_youtube=fetcher))
    check("config shows the YouTube tab when a fetcher is wired", c.get("/ui/config").json()["youtube"])
    r = c.post("/ui/dub/youtube", json={"url": yf.canonical_url(VID)})
    check("no access code -> 403", r.status_code == 403, r.text)
    r = c.post("/ui/dub/youtube", headers=H, json={"url": "https://vimeo.com/123", "target_lang": "Tamil"})
    check("a non-YouTube link -> 400 with a reason", r.status_code == 400 and "YouTube" in r.text, r.text)
    r = c.post("/ui/dub/youtube", headers=H, json={"url": f"youtu.be/{VID}", "target_lang": "Klingon"})
    check("an unknown language -> 400", r.status_code == 400, r.text)
    r = c.post("/ui/dub/youtube", headers={**H, "Idempotency-Key": "k1"},
               data={"url": f"https://youtu.be/{VID}?t=3", "target_lang": "Tamil", "mode": "basic"})
    job = r.json().get("job_id")
    check("a valid link returns a job at once, fetch spawned with the canonical URL",
          r.status_code == 200 and len(fetcher.calls) == 1 and fetcher.calls[0][0] == job
          and fetcher.calls[0][1] == yf.canonical_url(VID) and fetcher.calls[0][4] == "DEMO"
          and not dub.calls, f"{r.text} {fetcher.calls}")
    r2 = c.post("/ui/dub/youtube", headers={**H, "Idempotency-Key": "k1"},
                json={"url": f"https://youtu.be/{VID}", "target_lang": "Tamil"})
    check("a retried submit with the same Idempotency-Key does not fetch twice",
          r2.json().get("job_id") == job and len(fetcher.calls) == 1)
    s = c.get(f"/ui/dub/{job}", headers=H).json()
    check("status while fetching: 'Fetching the YouTube video', with the canonical link",
          s["description"] == "Fetching the YouTube video" and s["youtube"]["url"] == yf.canonical_url(VID), str(s))

    # Once done, /source serves the fetched input under the job's token only.
    os.makedirs(os.path.join(jobs, job), exist_ok=True)
    src = make_video(os.path.join(jobs, job, f"youtube_{VID}.mp4"), 2)
    status[job].update(status="done", input_name=f"youtube_{VID}.mp4", output=src)
    s = c.get(f"/ui/dub/{job}", headers=H).json()
    tok = status[job]["dl_token"]
    r = c.get(s["source_url"])
    check("the original (fetched) video is served for the result page",
          r.status_code == 200 and r.content == open(src, "rb").read())
    check("…and refused with a forged token", c.get(f"/ui/dub/{job}/source?t=x{tok}").status_code == 403)
    shutil.rmtree(jobs, ignore_errors=True)


if __name__ == "__main__":
    test_parse()
    test_metadata_gate_before_download()
    test_run_job_shared_gate_and_secrets()
    test_route()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    raise SystemExit(1 if FAILS else 0)
