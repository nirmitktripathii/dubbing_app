#!/usr/bin/env python3
"""Step-6 TTS worker — runs IndicF5 synthesis in an ISOLATED OS process.

Why this exists
---------------
IndicF5 synthesis occasionally wedges the GPU: a CUDA op that never returns. Python
cannot kill the thread stuck in that op, and if the wedged op holds the GIL the whole
process (Streamlit included) freezes. A re-run in the SAME process inherits the wedged
CUDA context and the cached wedged model, so it re-freezes — which is why the in-process
"resume from Step 6" never cured it. The only reliable cure is to run synthesis in a
SEPARATE process that a parent can SIGKILL and relaunch: a fresh process gets a fresh
interpreter and a fresh CUDA context, and SIGKILL lets the OS reclaim the wedged GPU.

This module is that separate process. ``pipeline.tts_supervisor`` is the parent that
launches it, watches its heartbeat, kills it on a stall, and relaunches a fresh child
that resumes from the on-disk manifest.

Contract
--------
``argv[1]`` is the path to a job-spec JSON. The worker calls
``generate_tts_for_segments(..., watchdog_mode="none", heartbeat_path=...)``, which writes
per-segment WAVs + ``tts_manifest.json`` into ``output_dir`` and touches the heartbeat as
it goes. It exits 0 on clean completion. All progress is on disk, so a SIGKILLed worker
loses nothing — the next worker resumes from the manifest. stdout/stderr are the worker's
log; the supervisor captures and forwards them to the UI.

Exit codes:
    0  completed (every segment settled to a WAV, some possibly silence fallbacks)
    1  fatal run error (bad spec, model-load blowup, OOM) — supervisor decides on relaunch
    2  usage / unreadable job spec
    3  could not import the pipeline (packaging problem)

Job-spec keys:
    segments             : list of {start, end, text}
    target_language      : display name, e.g. "Hindi"
    output_dir           : where WAVs + manifest go
    reference_audio_path : optional voice-clone reference (or null)
    reference_text       : transcript of the reference (or null)
    device               : "auto" | "cuda" | "cpu"
    nfe_step             : concrete int (supervisor resolves it; never null in practice)
    heartbeat_path       : liveness file the supervisor polls
"""
import os
import sys
import json


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("[tts_worker] ERROR: no job-spec path given (usage: tts_worker <spec.json>)", flush=True)
        return 2

    spec_path = argv[0]
    try:
        with open(spec_path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception as e:
        print(f"[tts_worker] ERROR: could not read job spec {spec_path!r}: {e}", flush=True)
        return 2

    # Import inside main (not at module top) so an import failure is reported on stdout
    # with a dedicated non-zero exit the supervisor can act on, rather than blowing up
    # when the supervisor merely imports this module to find the worker command.
    try:
        from pipeline.duration_tts import generate_tts_for_segments
    except Exception as e:
        print(f"[tts_worker] ERROR: could not import pipeline.duration_tts: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 3

    segments = spec.get("segments") or []
    print(
        f"[tts_worker] pid={os.getpid()} starting: {len(segments)} segment(s), "
        f"lang={spec.get('target_language')!r}, device={spec.get('device', 'auto')!r}, "
        f"nfe={spec.get('nfe_step')}, out={spec.get('output_dir')!r}",
        flush=True,
    )

    try:
        generate_tts_for_segments(
            segments,
            target_language=spec["target_language"],
            output_dir=spec["output_dir"],
            reference_audio_path=spec.get("reference_audio_path"),
            reference_text=spec.get("reference_text"),
            device=spec.get("device", "auto"),
            log_fn=None,  # stdout IS the log; the supervisor forwards it to the UI
            nfe_step=spec.get("nfe_step"),
            watchdog_mode="none",
            heartbeat_path=spec.get("heartbeat_path"),
        )
    except Exception as e:
        # In watchdog_mode="none", generate_tts_for_segments degrades per-segment errors
        # to silence and does NOT raise on a hang (the supervisor owns hangs). So a raise
        # here is an unexpected WHOLE-RUN failure — report it and exit non-zero; the
        # supervisor inspects on-disk progress and decides whether to relaunch.
        print(f"[tts_worker] FATAL: run failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

    print(f"[tts_worker] pid={os.getpid()} finished cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    # Flush, then HARD-exit on the clean path. A normal sys.exit() runs interpreter shutdown,
    # which joins non-daemon threads — and torch/CUDA/IndicF5 can leave a lingering non-daemon
    # thread that never returns, hanging the worker at exit AFTER every segment is already on
    # disk. That would strand the supervisor waiting on a done-but-not-exiting child until its
    # 240 s stall watchdog kills it (a needless 4-min stall) — or, if the parent is itself
    # wedged, forever. All progress is durably on disk (per-segment WAVs + manifest), so
    # skipping interpreter shutdown loses nothing. Error paths keep sys.exit so tracebacks and
    # atexit hooks still run for diagnosis.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if _rc == 0:
        os._exit(0)
    sys.exit(_rc)
