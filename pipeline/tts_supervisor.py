#!/usr/bin/env python3
"""Step-6 TTS supervisor — process-isolation cure for the IndicF5 GPU freeze.

The problem
-----------
IndicF5 synthesis occasionally wedges the GPU on a CUDA op that never returns. Because
that op can hold the GIL, the whole Python process (Streamlit included) freezes. The
in-process daemon-thread watchdog in ``duration_tts.py`` ('thread' mode) can stop issuing
NEW GPU work but cannot kill the thread already stuck in the wedged kernel, and a re-run in
the SAME process inherits both the wedged CUDA context and the cached wedged model — which
is exactly why "resume from Step 6" inside Streamlit never cured the freeze.

The cure
--------
Run synthesis in a SEPARATE OS process (``pipeline.tts_worker``) and have THIS parent
supervise it. A separate process has its own interpreter and its own CUDA context, so the
parent's event loop is never frozen by the child's wedge, and a SIGKILL of the child lets
the OS tear down the wedged context and reclaim the GPU. All synthesis progress lives on
disk (per-segment WAVs + ``tts_manifest.json``), so a killed child loses nothing: the next
child resumes from the manifest.

How supervision works
---------------------
1. Resolve ``nfe_step`` to a concrete int (so the signatures we write for forced-silence
   segments match exactly what the worker computes and its resume-skip will accept).
2. Write a job-spec JSON and launch the worker in its own process group.
3. Poll the worker's heartbeat file every ``poll_interval`` seconds. The worker's MAIN
   thread touches that file as real progress is made (boot -> loading -> per-segment). A
   wedged CUDA op simply STOPS the beat — a stalled heartbeat is the signal we kill on.
   Two phase-appropriate budgets: a generous ``load_stall`` while phase is boot/loading
   (covers model download + load), and a tight ``seg_stall`` once synthesis starts.
4. On a stall: SIGKILL the child's process group, let the GPU settle, and relaunch a fresh
   child that resumes from the manifest.
5. A segment that stalls (or crashes the worker) ``kill_budget`` times is declared poison:
   the supervisor writes a silence WAV for it AND a manifest entry
   ``{"status":"ok","sig":<matching-signature>,"forced_silence":true}``. The worker's
   EXISTING resume check (status ok + sig match + file present + size>44) then skips it —
   no change to the worker's resume logic is needed. This is "degrade, don't crash": one
   poison line never aborts a 40-minute render.
6. An overall ``max_relaunch`` cap bounds pathological loops. When the loop ends (worker
   finished, or the cap was hit), final assembly fills ANY still-missing segment with
   silence so every returned segment always has a valid ``audio_path``.

Public API
----------
``generate_tts_supervised(...)`` mirrors ``generate_tts_for_segments`` (same required
args, same return contract: the segment list with ``audio_path`` added to each element),
plus supervision knobs and a ``worker_cmd`` override so tests can substitute a fake worker
that needs no GPU.
"""
import os
import sys
import json
import time
import signal
import subprocess
import threading
from typing import Optional

# Pure, dependency-light helpers reused from the synthesis module. Importing duration_tts
# does NOT require torch/soundfile (both are optional imports there), so the supervisor —
# and the no-GPU tests that drive it — can import these without a GPU stack. Reusing the
# REAL _segment_signature (rather than reimplementing it) is what guarantees a supervisor
# forced-silence entry is byte-identical to what the worker's resume-skip expects.
from pipeline.duration_tts import (
    _segment_signature,
    _load_manifest,
    _save_manifest_atomic,
    LANGUAGE_TO_CODE,
    MANIFEST_NAME,
    INDICF5_SAMPLE_RATE,
    SEGMENT_TIMEOUT_DEFAULT,
)

_IS_WINDOWS = os.name == "nt"

WORKER_MODULE = "pipeline.tts_worker"
HEARTBEAT_NAME = "tts_heartbeat.json"
JOBSPEC_NAME = "tts_job.json"

# Defaults (each overridable by arg or environment).
DEFAULT_LOAD_STALL = 900.0   # phase boot/loading: model download + load can be slow/cold
DEFAULT_SEG_STALL = SEGMENT_TIMEOUT_DEFAULT + 60.0  # 240s: the 180s per-segment ceiling + margin
DEFAULT_KILL_BUDGET = 2      # stalls/crashes at one segment before it is forced to silence
DEFAULT_MAX_RELAUNCH = 20    # overall launch cap (first launch + relaunches)
DEFAULT_POLL_INTERVAL = 5.0  # heartbeat poll cadence
DEFAULT_KILL_SETTLE = 2.0    # pause after a kill so the driver can reclaim the GPU
# Consecutive pure-loading failures (never reached synthesis, zero segments done) after
# which we stop retrying and raise: that is an environment/model-load failure, not a poison
# segment, and silently returning all-silence would mask a real breakage during validation.
LOADING_FATAL_STRIKES = 3


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _json_default(o):
    """Coerce a numpy scalar (bool_ / int64 / float64 / …) to its native Python type so
    json.dump can serialize a segment dict that carries one. The upstream gates write
    fields like ``gates_passed`` from numpy comparisons (``sim >= threshold`` -> numpy.bool_),
    and numpy.bool_/numpy.int64 are NOT JSON-serializable (unlike numpy.float64, which
    subclasses float) — so without this the job-spec write crashes before the worker even
    launches. Applied as the ``default=`` hook, it fires ONLY for values the stdlib encoder
    rejects. Mirrors app.py's identical helper so the headless and Streamlit paths agree."""
    import numpy as _np
    if isinstance(o, _np.generic):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _resolve_nfe_step(nfe_step: Optional[int], emit) -> int:
    """Resolve nfe_step EXACTLY as generate_tts_for_segments does (arg > env > 32, clamp
    >=1). Must match, because we feed this same value both to the worker (in the job spec)
    and to _segment_signature when writing forced-silence entries — a mismatch would make
    the worker treat our silence as stale and re-synthesize the poison segment forever."""
    if nfe_step is None:
        env = os.environ.get("DUBBING_NFE_STEP", "").strip()
        if env:
            try:
                nfe_step = int(env)
            except ValueError:
                emit(f"[supervisor] WARNING: DUBBING_NFE_STEP={env!r} is not an int; using 32.")
                nfe_step = 32
        else:
            nfe_step = 32
    if nfe_step < 1:
        emit(f"[supervisor] WARNING: nfe_step={nfe_step} invalid; clamping to 1.")
        nfe_step = 1
    return nfe_step


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _write_silence_wav(path: str, duration_s: float, sample_rate: int = INDICF5_SAMPLE_RATE) -> None:
    """Write ``duration_s`` of mono 16-bit-PCM silence to ``path`` (atomic temp+replace).

    16-bit PCM is exactly the subtype soundfile writes by default in the worker, so a
    supervisor-written silence file is byte-format-identical to a worker-written one — Step
    7 assembly sees a single uniform WAV format. Uses the stdlib ``wave`` module so the
    supervisor needs neither numpy nor soundfile for its own writes. A floor of a few
    frames guarantees the file is larger than a bare 44-byte header, so the worker's
    resume-skip size check (>44) accepts it even for a near-zero-duration segment."""
    import wave
    n_frames = max(int(round(duration_s * sample_rate)), 8)
    tmp = f"{path}.{os.getpid()}.sil.tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)   # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_frames)
    os.replace(tmp, path)


def _mark_forced_silence(manifest_path: str, idx: int, sig: str, path: str) -> None:
    """Merge a forced-silence entry into the on-disk manifest, atomically.

    status="ok" + a MATCHING signature is deliberate: it makes the next worker's resume
    check skip this segment instead of retrying it. ``forced_silence: true`` marks it as a
    supervisor degrade (not genuine audio) for the end-of-run report. Called only while no
    child is running, so there is no concurrent manifest writer."""
    m = _load_manifest(manifest_path)
    m[str(idx)] = {"status": "ok", "sig": sig, "path": path, "forced_silence": True}
    _save_manifest_atomic(manifest_path, m)


def _ok_count(manifest_path: str) -> int:
    m = _load_manifest(manifest_path)
    return sum(1 for v in m.values() if isinstance(v, dict) and v.get("status") == "ok")


def _read_beat(path: str):
    """Return the heartbeat dict, or None if absent/unreadable. os.replace makes each write
    atomic, so we never read a partial file; a transient miss just yields None and the
    caller keeps the last good beat."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _popen_kwargs() -> dict:
    """Put the child in its own process group so we can kill it AND anything it spawned."""
    if _IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _hard_kill(proc: "subprocess.Popen", emit) -> None:
    """SIGKILL the child's whole process group (POSIX) / TerminateProcess the tree
    (Windows), then reap it. SIGKILL — not SIGTERM — because a process wedged in a CUDA
    kernel will not run a signal handler; only an uncatchable kill frees the GPU."""
    try:
        if _IS_WINDOWS:
            proc.kill()
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True, timeout=15,
                )
            except Exception:
                pass
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass  # already gone
    except Exception as e:
        emit(f"[supervisor] WARNING: kill failed: {e}")
    try:
        proc.wait(timeout=15)
    except Exception:
        pass


def _pump(stream, emit) -> None:
    """Forward every line the child prints to our emitter (which fans out to stdout + the
    UI log). Runs in a daemon thread; child lines already carry their own prefixes."""
    try:
        for line in iter(stream.readline, ""):
            if line == "":
                break
            emit(line.rstrip("\n"))
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def generate_tts_supervised(
    translated_segments: list,
    target_language: str,
    output_dir: str,
    reference_audio_path: Optional[str] = None,
    reference_text: Optional[str] = None,
    device: str = "auto",
    log_fn=None,
    nfe_step: Optional[int] = None,
    *,
    worker_cmd: Optional[list] = None,
    load_stall: Optional[float] = None,
    seg_stall: Optional[float] = None,
    kill_budget: int = DEFAULT_KILL_BUDGET,
    max_relaunch: int = DEFAULT_MAX_RELAUNCH,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    kill_settle: float = DEFAULT_KILL_SETTLE,
) -> list:
    """Drop-in supervised replacement for generate_tts_for_segments.

    Same required args and the same return contract (the segment list with ``audio_path``
    added to every element). Runs synthesis in a supervised subprocess that is SIGKILLed
    and relaunched on a stalled heartbeat, degrading poison segments to silence — the
    process-isolation cure for the Step-6 GPU freeze.

    Extra keyword-only args:
        worker_cmd:    argv prefix for the worker (the spec path is appended). Defaults to
                       [sys.executable, "-m", "pipeline.tts_worker"]. Override in tests to
                       point at a fake, no-GPU worker.
        load_stall:    seconds of heartbeat silence tolerated during boot/loading before a
                       kill. Arg > env DUBBING_TTS_LOAD_STALL > 900.
        seg_stall:     seconds tolerated during per-segment synthesis. Arg > env
                       DUBBING_TTS_SEG_STALL > SEGMENT_TIMEOUT_DEFAULT+60 (240).
        kill_budget:   stalls/crashes blamed on one segment before it is forced to silence.
        max_relaunch:  overall launch cap (first launch included) before giving up and
                       assembling whatever is on disk.
        poll_interval: heartbeat poll cadence, seconds.
        kill_settle:   pause after a kill, seconds, so the driver can reclaim the GPU.
    """
    from datetime import datetime

    def emit(msg: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if log_fn is not None:
            # log_fn owns BOTH the durable file and the (hardened, non-blocking) console
            # mirror. Do NOT also raw-print here: under a headless run the parent's stdout is
            # a rate-limited Kaggle pipe, and a second blocking print per line would double
            # that volume AND re-introduce the wedge log_fn was hardened to avoid (a full
            # stdout pipe freezing the whole run after synthesis already finished). Standalone
            # (no log_fn — e.g. the no-GPU tests) still prints so output is visible.
            try:
                log_fn(f"  {msg}")
            except Exception:
                pass
        else:
            print(f"[{t}] {msg}", flush=True)

    lang_code = LANGUAGE_TO_CODE.get(target_language)
    if not lang_code:
        raise ValueError(
            f"Unsupported language: '{target_language}'. Supported: {list(LANGUAGE_TO_CODE.keys())}"
        )

    os.makedirs(output_dir, exist_ok=True)
    n_total = len(translated_segments)
    if n_total == 0:
        emit("[supervisor] No segments to synthesize — nothing to do.")
        return []

    nfe_step = _resolve_nfe_step(nfe_step, emit)
    load_stall = load_stall if load_stall is not None else _env_float("DUBBING_TTS_LOAD_STALL", DEFAULT_LOAD_STALL)
    seg_stall = seg_stall if seg_stall is not None else _env_float("DUBBING_TTS_SEG_STALL", DEFAULT_SEG_STALL)

    if worker_cmd is None:
        worker_cmd = [sys.executable, "-m", WORKER_MODULE]

    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    heartbeat_path = os.path.join(output_dir, HEARTBEAT_NAME)
    spec_path = os.path.join(output_dir, JOBSPEC_NAME)

    # The child runs with cwd = the directory that CONTAINS the `pipeline` package, so
    # `python -m pipeline.tts_worker` resolves regardless of the parent's cwd (on Kaggle
    # that dir is /kaggle/working). PYTHONPATH gets the same dir defensively.
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = pkg_root + os.pathsep + child_env.get("PYTHONPATH", "")
    child_env.setdefault("PYTHONUNBUFFERED", "1")  # so the child's stdout reaches us live

    spec = {
        "segments": translated_segments,
        "target_language": target_language,
        "output_dir": output_dir,
        "reference_audio_path": reference_audio_path,
        "reference_text": reference_text,
        "device": device,
        "nfe_step": nfe_step,       # concrete int — see _resolve_nfe_step
        "heartbeat_path": heartbeat_path,
    }
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, default=_json_default)

    emit(
        f"[supervisor] Supervising {n_total} segment(s): lang={target_language} nfe={nfe_step} "
        f"load_stall={load_stall:g}s seg_stall={seg_stall:g}s kill_budget={kill_budget} "
        f"max_relaunch={max_relaunch}"
    )

    seg_strikes: dict = {}   # segment index -> consecutive stalls/crashes blamed on it
    forced: set = set()      # segment indices the supervisor forced to silence
    loading_strikes = 0      # consecutive failures that never reached synthesis
    launches = 0
    completed = False

    while True:
        if launches >= max_relaunch:
            emit(f"[supervisor] Relaunch cap ({max_relaunch}) reached — assembling from disk.")
            break

        # Remove the previous child's heartbeat so we never mistake its final (stale) beat
        # for the new child's, which would trip an instant false stall before startup.
        _rm(heartbeat_path)
        launch_time = time.time()
        last_beat = None
        ok_before = _ok_count(manifest_path)
        launches += 1

        emit(f"[supervisor] Launch #{launches}/{max_relaunch}: {' '.join(str(c) for c in worker_cmd)}")
        proc = subprocess.Popen(
            list(worker_cmd) + [spec_path],
            cwd=pkg_root,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **_popen_kwargs(),
        )
        pump = threading.Thread(target=_pump, args=(proc.stdout, emit), daemon=True)
        pump.start()

        we_killed = False
        stall_phase = None
        while True:
            rc = proc.poll()
            if rc is not None:
                break  # child ended on its own (clean finish or crash)

            beat = _read_beat(heartbeat_path)
            if beat is not None:
                last_beat = beat
            if last_beat and "ts" in last_beat:
                phase = last_beat.get("phase", "loading")
                age = time.time() - float(last_beat["ts"])
            else:
                phase = "loading"           # no beat yet -> still starting up / loading
                age = time.time() - launch_time
            budget = load_stall if phase in ("boot", "loading") else seg_stall

            if age > budget:
                stall_phase = phase
                blamed = last_beat.get("seg", -1) if (last_beat and phase == "synth") else -1
                emit(
                    f"[supervisor] STALL: phase={phase} seg={blamed} silent {age:.0f}s > "
                    f"{budget:.0f}s — SIGKILL worker (launch #{launches})."
                )
                _hard_kill(proc, emit)
                we_killed = True
                break

            time.sleep(poll_interval)

        pump.join(timeout=5)
        ok_after = _ok_count(manifest_path)
        progressed = ok_after > ok_before

        # ── Decide what just happened and which segment (if any) to blame ──────────
        blamed_seg = -1
        if we_killed:
            blamed_seg = last_beat.get("seg", -1) if (last_beat and stall_phase == "synth") else -1
        else:
            rc = proc.returncode
            if rc == 0:
                emit(f"[supervisor] Worker exited cleanly (rc=0). Done {ok_after}/{n_total}.")
                completed = True
                break
            if rc in (2, 3):
                # Bad job spec (2) or import/packaging failure (3): not transient, not a
                # poison segment. Relaunching cannot help — surface it loudly.
                raise RuntimeError(
                    f"tts_worker exited rc={rc} (non-retryable: bad job spec or failed import). "
                    f"See the worker log above."
                )
            # rc == 1 (or any other): the worker crashed. Blame the segment it was last on.
            emit(f"[supervisor] Worker crashed (rc={rc}); progress {ok_before}->{ok_after}.")
            blamed_seg = last_beat.get("seg", -1) if (last_beat and last_beat.get("phase") == "synth") else -1

        # Let the GPU driver reclaim the killed context before a fresh child grabs it.
        time.sleep(kill_settle)

        # ── Strike / degrade logic ────────────────────────────────────────────────
        if blamed_seg is not None and blamed_seg >= 0:
            loading_strikes = 0  # we reached synthesis; reset the loading-failure counter
            seg_strikes[blamed_seg] = seg_strikes.get(blamed_seg, 0) + 1
            emit(f"[supervisor] Segment {blamed_seg}: strike {seg_strikes[blamed_seg]}/{kill_budget}.")
            if seg_strikes[blamed_seg] >= kill_budget and blamed_seg not in forced:
                seg = translated_segments[blamed_seg]
                text = (seg.get("text") or "").strip()
                dur = float(seg["end"] - seg["start"])
                out_path = os.path.join(output_dir, f"segment_{blamed_seg:04d}.wav")
                _write_silence_wav(out_path, dur, INDICF5_SAMPLE_RATE)
                sig = _segment_signature(text, dur, lang_code, nfe_step)
                _mark_forced_silence(manifest_path, blamed_seg, sig, out_path)
                forced.add(blamed_seg)
                emit(
                    f"[supervisor] Segment {blamed_seg} hit kill budget ({kill_budget}) — "
                    f"FORCED SILENCE ({dur:.2f}s). The next worker will skip it."
                )
        elif progressed:
            loading_strikes = 0  # made segment progress overall; not a pure loading failure
        else:
            loading_strikes += 1
            emit(
                f"[supervisor] Failure before synthesis (loading strike "
                f"{loading_strikes}/{LOADING_FATAL_STRIKES})."
            )
            if loading_strikes >= LOADING_FATAL_STRIKES and ok_after == 0:
                raise RuntimeError(
                    f"tts_worker failed to reach synthesis {loading_strikes} times and produced "
                    f"zero segments — the model load is failing (not a poison segment). See the "
                    f"worker log above."
                )
        # loop back and relaunch; a fresh child resumes from the manifest.

    # ── Final assembly: guarantee every segment has a valid WAV + build the result ──
    manifest = _load_manifest(manifest_path)
    results = []
    gap_filled = []
    for i, seg in enumerate(translated_segments):
        out_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 44):
            # Never produced (e.g. we hit the relaunch cap before reaching it). Fill with
            # silence so downstream assembly has a file for every segment.
            text = (seg.get("text") or "").strip()
            dur = float(seg["end"] - seg["start"])
            _write_silence_wav(out_path, dur, INDICF5_SAMPLE_RATE)
            sig = _segment_signature(text, dur, lang_code, nfe_step)
            manifest[str(i)] = {"status": "ok", "sig": sig, "path": out_path, "forced_silence": True}
            forced.add(i)
            gap_filled.append(i)
        results.append({**seg, "audio_path": out_path})
    if gap_filled:
        _save_manifest_atomic(manifest_path, manifest)

    # ── Report ─────────────────────────────────────────────────────────────────────
    real = n_total - len(forced)
    if not completed:
        emit(
            f"[supervisor] Stopped WITHOUT a clean worker finish (launches={launches}). "
            f"Assembled from disk: {real}/{n_total} real, {len(forced)} silence."
        )
    if forced:
        emit(
            f"[supervisor] DEGRADED: {len(forced)} segment(s) are silence fallbacks "
            f"(indices {sorted(forced)})"
            + (f", incl. {sorted(gap_filled)} never reached." if gap_filled else "")
            + ". Re-run Step 6 (Start over) to retry them."
        )
    emit(f"[supervisor] Done: {len(results)} segment(s) in {output_dir} ({real} real, {len(forced)} silence).")
    return results
