"""Runner that turns a probe set into TTS scorecards — the IndicF5 nfe sweep first, and
(later) the FastPitch candidate.

For each candidate config it synthesizes the probe segments, measures produced-vs-requested
duration (length slope + signed error) and latency, optionally CER, and emits one scorecard
row via pipeline.tts_benchmark. It computes NOTHING that pipeline.tts_benchmark already
computes — it only orchestrates + collects, so the numbers have a single source (#8).

Synthesis is INJECTED (`synth_fn`) so this whole runner is testable with a fake backend and
no GPU / torch. The default `synth_fn` drives pipeline.tts_supervisor.generate_tts_supervised
(the freeze-fix path), so any candidate that hangs is killed + resumed, never a frozen sweep.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional, Sequence

from pipeline import tts_benchmark as B


def _aggregate_latency(wall_seconds: Optional[float], n: int) -> dict:
    """Fallback latency when per-segment timings aren't available: mean = wall/n, clearly
    flagged as including model load + any relaunches, so it is never mistaken for clean
    per-segment synth time (a relative comparison across configs is still meaningful)."""
    mean = (wall_seconds / n) if (wall_seconds is not None and n) else None
    return {"mean": mean, "p50": None, "p90": None, "max": None, "n": n,
            "note": "aggregate wall/n (includes model load + relaunches)"}


def _indicf5_control_record() -> dict:
    """Duration control for IndicF5 is `fix_duration` passed straight to `infer_process`
    (pipeline/duration_tts.py: `infer_process(..., fix_duration=fix_duration, nfe_step=...)`).
    This is a STATIC-code proof, not a runtime hook — labelled as such so it is honest
    (non-negotiable #1). A runtime forward-hook proof is a later worker enhancement."""
    rec = B.control_record("indicf5", "fix_duration", "ref+target", "ref+target", True)
    rec["proof"] = "static: duration_tts.py passes fix_duration to infer_process"
    return rec


def _default_synth(segments, out_dir, *, nfe_step, target_language, reference_audio_path,
                   reference_text, log_fn) -> dict:
    """Drive the hang-safe supervised path at a given nfe_step. Returns wall time, the clean
    per-segment synth times the worker stamped into its resume manifest, and a control
    record. If the manifest carries no timings (older worker, or an all-placeholder run),
    per_seg_seconds is None and latency falls back to aggregate wall/n downstream."""
    from pipeline.tts_supervisor import generate_tts_supervised  # lazy: keeps runner GPU-free to import

    t0 = time.time()
    generate_tts_supervised(
        segments, target_language=target_language, output_dir=out_dir,
        reference_audio_path=reference_audio_path, reference_text=reference_text,
        log_fn=log_fn, nfe_step=nfe_step,
    )
    wall = time.time() - t0
    # Prefer the worker's real per-segment GPU-synth times (survive relaunch/resume on
    # disk); compact away segments with no timing. Empty -> None so run_sweep's `if per_seg`
    # guard falls back to the honestly-labelled aggregate wall/n instead.
    timings = B.read_manifest_timings(out_dir, len(segments))
    per_seg = [t for t in (timings or []) if t is not None] or None
    return {"wall_seconds": wall, "per_seg_seconds": per_seg,
            "control": _indicf5_control_record()}


def run_sweep(probe_segments: Sequence[dict], nfe_values: Sequence[int], out_dir: str, *,
              candidate_prefix: str = "indicf5", target_language: str = "Hindi",
              reference_audio_path: Optional[str] = None, reference_text: Optional[str] = None,
              clip: str = "probe", data_version: str = "probe_v1", device: str = "unknown",
              synth_fn: Optional[Callable] = None, cer_fn: Optional[Callable] = None,
              lang_code: Optional[str] = None, log_fn: Optional[Callable] = None) -> dict:
    """Run one candidate per nfe value and write a combined scorecard.

    `probe_segments` should come from tts_benchmark.build_probe_segments so each carries
    `natural`/`ratio` for the direction tests. `synth_fn(segments, out_dir, *, nfe_step,
    target_language, reference_audio_path, reference_text, log_fn) -> {wall_seconds,
    per_seg_seconds, control}`. `cer_fn(wav_paths, ref_texts, lang_code) -> cer dict` is
    optional (omit to skip the GPU Whisper axis).
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    synth = synth_fn or _default_synth
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for nfe in nfe_values:
        sub = os.path.join(out_dir, f"{candidate_prefix}_nfe{nfe}")
        os.makedirs(sub, exist_ok=True)
        _log(f"[sweep] candidate {candidate_prefix} nfe={nfe} -> {sub}")

        res = synth(probe_segments, sub, nfe_step=nfe, target_language=target_language,
                    reference_audio_path=reference_audio_path, reference_text=reference_text,
                    log_fn=log_fn) or {}

        pairs, dir_tests, wav_paths, ref_texts = [], [], [], []
        for i, seg in enumerate(probe_segments):
            wav = os.path.join(sub, f"segment_{i:04d}.wav")
            requested = float(seg["end"] - seg["start"])
            produced = B.wav_duration_seconds(wav)
            pairs.append((requested, produced))
            wav_paths.append(wav)
            ref_texts.append(seg.get("text", ""))
            nat = seg.get("natural")
            if nat is not None:
                ok = B.direction_ok(requested, produced, float(nat))
                if ok is not None:
                    dir_tests.append({"ratio": seg.get("ratio"), "ok": ok})

        slope = B.length_slope(pairs)
        per_seg = res.get("per_seg_seconds")
        latency = B.latency_stats(per_seg) if per_seg else _aggregate_latency(res.get("wall_seconds"), len(pairs))
        cer = (cer_fn(wav_paths, ref_texts, lang_code) if cer_fn
               else {"mean_cer": None, "available": False, "n": 0})

        rows.append(B.build_scorecard(
            candidate=f"{candidate_prefix}@nfe{nfe}",
            provenance={"candidate_source": candidate_prefix, "clip": clip, "nfe_step": nfe,
                        "n_samples": slope.get("used"), "data_version": data_version, "device": device},
            control=res.get("control"), slope=slope, latency=latency, cer=cer,
            direction_tests=dir_tests,
        ))

    paths = B.write_reports(rows, out_dir, name="tts_scorecard",
                            title=f"{candidate_prefix} nfe sweep")
    _log(f"[sweep] wrote {paths['markdown']}")
    return {"rows": rows, **paths}
