#!/usr/bin/env python3
"""Fetch a public YouTube video as a dub input — with the plan's duration limit enforced
BEFORE a single byte of video is downloaded, and again on the file that arrives.

    parse_youtube_url(url)   -> the 11-character video id, or FetchRejected
    check_metadata(info, s)  -> None, or FetchRejected (not public / live / too long / unknown)
    fetch(url, out_dir, s)   -> (path, meta): metadata gate, then a capped download
    run_fetch_job(...)       -> the whole job step: status + log, fetch, then the SAME
                                admission gate an upload goes through (deploy/api.admit_job)

THREE GATES, cheapest first. (1) The URL must parse to a YouTube video id; we never hand a
user-supplied URL to yt-dlp — we rebuild a canonical watch URL from the id, so the server can
only ever be pointed at youtube.com, and playlist/timestamp parameters are dropped. (2) The
metadata (no media download yet) must say public, not live, and a KNOWN duration within the
limit — an unknown duration is a failed check, not a pass. (3) The downloaded file goes
through admit_job's ffprobe gate, the one every upload passes, so the number we bill on and the
number we gated on come from one implementation. yt-dlp's own match_filter re-checks (2) at
download time as a belt-and-braces guard against the metadata and the stream disagreeing.

SERVER ACCESS. YouTube refuses requests from datacenter IPs ("Sign in to confirm you're not a
bot") — measured from a Modal container on 2026-10-05 with every yt-dlp player client. So the
deployed fetch needs one of, from the Modal secret `dubbing-secrets`:
    YTDLP_COOKIES  Netscape-format cookies.txt CONTENT for a YouTube session
    YTDLP_PROXY    a proxy URL (e.g. a residential proxy), passed to yt-dlp as `proxy`
Neither value is ever logged. Without them (or DUB_YOUTUBE_DIRECT=1) the page hides the
YouTube tab and the endpoint answers 503; a fetch that still meets the bot wall fails on CPU
with a message saying so — no GPU is started.
"""
from __future__ import annotations

import glob
import os
import re
import tempfile
import time
from urllib.parse import parse_qs, urlsplit

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
                 "youtu.be", "www.youtube-nocookie.com", "youtube-nocookie.com"}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
# Path forms that carry the id as their second segment: /shorts/<id>, /embed/<id>, ...
_PATH_ID_PREFIXES = ("shorts", "embed", "live", "v", "e")
_FORMAT_PART_RE = re.compile(r"\.f\d+\.[A-Za-z0-9]+$")
MAX_HEIGHT = 720          # the dub keeps the source picture; 720p is plenty and keeps it small


def access_configured() -> bool:
    """Whether this deployment can reach YouTube: cookies or a proxy are set, or the operator
    asserts direct access works (DUB_YOUTUBE_DIRECT=1 — e.g. the local simulator, whose
    residential IP YouTube does not block). The UI hides the YouTube tab otherwise, rather than
    offering a button that fails on every press."""
    return bool(os.environ.get("YTDLP_COOKIES") or os.environ.get("YTDLP_PROXY")
                or os.environ.get("DUB_YOUTUBE_DIRECT") == "1")


class FetchRejected(Exception):
    """A reason the video will not be fetched, worded for the person who pasted the link."""


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def parse_youtube_url(url: str) -> str:
    """The video id in a YouTube link. Raises FetchRejected for anything else.

    Accepts watch, youtu.be, shorts, embed and live links on YouTube's own hosts, over http(s),
    with no credentials or port. A playlist link is accepted only if it names a video (v=).
    """
    url = (url or "").strip()
    if not url:
        raise FetchRejected("Paste a YouTube link.")
    if len(url) > 2048:
        raise FetchRejected("That link is too long to be a YouTube video link.")
    if "://" not in url:
        url = "https://" + url
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise FetchRejected("That doesn't look like a valid link.")
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https") or host not in YOUTUBE_HOSTS \
            or parts.username or parts.password or port not in (None, 80, 443):
        raise FetchRejected("Only YouTube links are supported (youtube.com or youtu.be).")

    segs = [s for s in parts.path.split("/") if s]
    vid = None
    if host == "youtu.be":
        vid = segs[0] if segs else None
    elif segs[:1] == ["watch"] or (not segs and "v" in parse_qs(parts.query)) \
            or segs[:1] == ["playlist"]:
        vid = (parse_qs(parts.query).get("v") or [None])[0]
    elif len(segs) >= 2 and segs[0] in _PATH_ID_PREFIXES:
        vid = segs[1]
    if not vid or not _ID_RE.match(vid):
        raise FetchRejected("That YouTube link doesn't point to a single video.")
    return vid


def check_metadata(info: dict, max_seconds: float) -> None:
    """Gate on what YouTube reports BEFORE downloading. Raises FetchRejected."""
    if info.get("_type") in ("playlist", "multi_video"):
        raise FetchRejected("That link is a playlist; paste the link of one video.")
    live = info.get("live_status")
    if info.get("is_live") or live in ("is_live", "is_upcoming", "post_live"):
        raise FetchRejected("Live streams and premieres can't be dubbed; "
                            "use a finished, uploaded video.")
    availability = info.get("availability")
    if availability != "public":
        what = {"private": "is private", "unlisted": "is unlisted, not public",
                "premium_only": "is for YouTube Premium members only",
                "subscriber_only": "is for channel members only",
                "needs_auth": "needs a signed-in account to watch"}.get(availability)
        raise FetchRejected(f"This video {what}. Only public videos can be dubbed." if what else
                            "Couldn't confirm this video is public, so it wasn't fetched.")
    if (info.get("age_limit") or 0) > 0:
        raise FetchRejected("This video is age-restricted, so it can't be fetched.")
    duration = info.get("duration")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise FetchRejected("YouTube didn't report this video's length, so it wasn't fetched.")
    if duration > max_seconds:
        raise FetchRejected(f"This video is {_mmss(duration)} long; the limit is "
                            f"{_mmss(max_seconds)}. Choose a shorter video.")


def _mmss(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def explain_download_error(msg: str) -> str:
    """yt-dlp's error text -> a sentence for the page. The raw text goes to the log."""
    m = msg.lower()
    # The bot wall comes in two shapes from a datacenter IP: refused metadata ("Sign in to
    # confirm you're not a bot"), or metadata allowed and the media itself refused with a 403
    # (seen from a production container on 2026-10-05). Both need the same operator fix.
    if ("confirm you" in m and "not a bot" in m) or "sign in to confirm" in m             or ("http error 403" in m and "download" in m):
        return ("YouTube refused the server's request (its bot check). The operator needs to "
                "configure YouTube access (YTDLP_COOKIES or YTDLP_PROXY) — or upload the file instead.")
    if "private video" in m:
        return "This video is private. Only public videos can be dubbed."
    if "members-only" in m or "join this channel" in m:
        return "This video is for channel members only. Only public videos can be dubbed."
    if "age" in m and ("restricted" in m or "confirm your age" in m):
        return "This video is age-restricted, so it can't be fetched."
    if "unavailable" in m or "removed" in m or "terminated" in m:
        return "This video is unavailable (removed, blocked in the server's region, or never existed)."
    if "does not pass filter" in m:
        return "This video didn't pass the length/live check at download time."
    return "Couldn't download this video from YouTube. Try again, or upload the file instead."


def ydl_options(out_dir: str, max_seconds: float, *, cookiefile: str | None = None,
                proxy: str | None = None, max_filesize: int | None = None,
                logger=None) -> dict:
    from yt_dlp.utils import match_filter_func
    opts = {
        "quiet": True, "no_warnings": True, "noprogress": True,
        "noplaylist": True, "playlist_items": "1",
        # H.264 + AAC first so the mux is a stream copy; any <=720p stream otherwise.
        "format": (f"bv*[height<={MAX_HEIGHT}][vcodec^=avc1]+ba[ext=m4a]/"
                   f"b[height<={MAX_HEIGHT}][ext=mp4]/bv*[height<={MAX_HEIGHT}]+ba/"
                   f"b[height<={MAX_HEIGHT}]/b"),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(out_dir, "youtube_%(id)s.%(ext)s"),
        "restrictfilenames": True, "windowsfilenames": True,
        "match_filter": match_filter_func(f"duration <= {int(max_seconds)} & !is_live"),
        "socket_timeout": 30, "retries": 3, "fragment_retries": 3,
        "cachedir": False, "overwrites": True,
    }
    if max_filesize:
        opts["max_filesize"] = int(max_filesize)
    if cookiefile:
        opts["cookiefile"] = cookiefile
    if proxy:
        opts["proxy"] = proxy
    if logger is not None:
        opts["logger"] = logger
    return opts


class _Log:
    """yt-dlp logger -> our log function; debug chatter dropped."""
    def __init__(self, log_fn):
        self.log_fn = log_fn

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        self.log_fn(f"[youtube] warning: {msg}")

    def error(self, msg):
        self.log_fn(f"[youtube] {msg}")


def fetch(url: str, out_dir: str, max_seconds: float, *, log_fn=print, cookies_text=None,
          proxy=None, max_filesize=None, ydl_cls=None) -> tuple[str, dict]:
    """Metadata gate, then download into ``out_dir``. Returns (path, meta).

    Raises FetchRejected with a user-facing reason. ``ydl_cls`` lets tests inject a fake.
    """
    vid = parse_youtube_url(url)
    if ydl_cls is None:
        from yt_dlp import YoutubeDL as ydl_cls
    from yt_dlp.utils import DownloadError

    cookiefile = None
    try:
        if cookies_text:
            fd, cookiefile = tempfile.mkstemp(prefix="ytc_", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(cookies_text)
        opts = ydl_options(out_dir, max_seconds, cookiefile=cookiefile, proxy=proxy,
                           max_filesize=max_filesize, logger=_Log(log_fn))
        log_fn(f"[youtube] Checking video {vid} (limit {_mmss(max_seconds)}, public only)...")
        meta_opts = {k: v for k, v in opts.items() if k not in ("match_filter",)}
        try:
            with ydl_cls({**meta_opts, "skip_download": True}) as y:
                info = y.extract_info(canonical_url(vid), download=False)
        except DownloadError as e:
            raise FetchRejected(explain_download_error(str(e)))
        info = info or {}
        check_metadata(info, max_seconds)
        meta = {"id": vid, "title": info.get("title"), "channel": info.get("channel")
                or info.get("uploader"), "duration": info.get("duration"),
                "url": canonical_url(vid)}
        log_fn(f"[youtube] OK: \"{meta['title']}\" by {meta['channel']} — "
               f"{_mmss(meta['duration'])}, public. Downloading (<= {MAX_HEIGHT}p)...")
        t0 = time.time()
        try:
            with ydl_cls(opts) as y:
                y.download([canonical_url(vid)])
        except DownloadError as e:
            raise FetchRejected(explain_download_error(str(e)))
        # The merged output; skip partials and per-format intermediates (youtube_<id>.f137.mp4).
        got = [p for p in glob.glob(os.path.join(out_dir, f"youtube_{vid}.*"))
               if not p.endswith((".part", ".ytdl")) and not _FORMAT_PART_RE.search(p)]
        if not got:
            raise FetchRejected("The download finished but produced no video file "
                                "(possibly over the size limit).")
        path = max(got, key=os.path.getsize)
        log_fn(f"[youtube] Downloaded {os.path.getsize(path) / 1048576:.1f} MB in "
               f"{time.time() - t0:.0f}s.")
        return path, meta
    finally:
        if cookiefile:
            try:
                os.remove(cookiefile)
            except OSError:
                pass


def run_fetch_job(*, job_id: str, url: str, target_lang: str, mode: str, plan: str, user,
                  job_status, jobs_vol, jobs_dir: str, dub_video, extra: dict | None = None,
                  ydl_cls=None) -> bool:
    """Fetch -> the shared admission gate -> spawn the dub. Records progress and failure in
    ``job_status[job_id]``. Returns True iff the dub was spawned."""
    from deploy.api import ApiError, MAX_UPLOAD_MB, PLAN_MAX_SECONDS, admit_job

    lines: list[str] = []

    def update(**kw):
        cur = dict(job_status.get(job_id) or {})
        cur.update(kw)
        job_status[job_id] = cur

    def log_fn(msg):
        lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        update(log="\n".join(lines), heartbeat=time.time())

    def fail(reason):
        log_fn(f"[youtube] FAILED: {reason}")
        update(status="failed", stage="fetch failed", error=reason, finished_at=time.time())
        return False

    update(status="fetching", stage="fetching")
    max_seconds = PLAN_MAX_SECONDS.get((plan or "").upper(), PLAN_MAX_SECONDS[""])
    job_dir = os.path.join(jobs_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    try:
        path, meta = fetch(url, job_dir, max_seconds, log_fn=log_fn,
                           cookies_text=os.environ.get("YTDLP_COOKIES") or None,
                           proxy=os.environ.get("YTDLP_PROXY") or None,
                           max_filesize=MAX_UPLOAD_MB * 1024 * 1024, ydl_cls=ydl_cls)
    except FetchRejected as e:
        return fail(str(e))
    except Exception as e:      # an unexpected fetch error must still end the job visibly
        log_fn(f"[youtube] {type(e).__name__}: {str(e)[:300]}")
        return fail("Couldn't download this video from YouTube. Try again, or upload the file.")
    update(youtube=meta)
    log_fn("[youtube] Checking the downloaded file's length, then starting the dub...")
    try:
        admit_job(job_status=job_status, jobs_vol=jobs_vol, dub_video=dub_video, job_id=job_id,
                  job_dir=job_dir, input_name=os.path.basename(path), target_lang=target_lang,
                  mode=mode, plan=plan, user=user, extra=extra)
    except ApiError as e:
        return fail(e.detail)
    return True
