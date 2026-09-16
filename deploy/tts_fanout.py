#!/usr/bin/env python3
"""Step-6 TTS fan-out — synthesize a run's segments across parallel GPU workers.

Why
---
TTS is the pipeline's dominant cost: a measured ~235 s of the ~7.8 min wall for an 80 s
video, because 13 segments are synthesized SERIALLY on one GPU (~18 s each). The segments
are independent — each one's target duration comes from its own SRT slot, and
``_apply_drift_correction`` compares a segment against its OWN target with no accumulated
cross-segment state — so this is embarrassingly parallel. Fanning it out cuts wall-clock
(not GPU-seconds, so COGS is unchanged; see PRODUCTION_INFRA_PLAN.md §4.2).

The one-implementation rule
---------------------------
A worker does NOT reimplement synthesis. It calls the same
``duration_tts.generate_tts_for_segments`` the serial and Kaggle paths call, passing the
FULL segment list plus ``only_indices`` for its shard. So language-code resolution,
nfe resolution, reference-voice download, the segment signature, drift correction, the
bounded disk writer and the silence-degradation path are all literally the same code, and
``i`` stays the global index — filenames (``segment_%04d.wav``) and manifest keys are
identical to a serial run, so shards merge without renumbering and a fanned-out run is
resumable by the serial path and vice versa.

Sharding, not per-segment mapping
---------------------------------
Work is mapped over SHARDS, not individual segments, because the cost of a worker is
dominated by loading IndicF5 (tens of seconds) — a per-segment map would pay that load for
every segment. One shard = one container = one model load = many segments.

    wall ≈ model_load + ceil(n_segments / n_shards) * per_segment

so the floor is `model_load + per_segment`, NOT zero: past a point, adding shards buys
nothing and only multiplies the number of model loads. With a warm pool (`min_containers`)
the load term drops out and the floor becomes roughly one segment's synthesis.

Manifest concurrency
--------------------
Workers never share a manifest. Each writes into its own private directory and returns its
WAV bytes plus its manifest entries; this module (single writer) merges them into the real
``tts_manifest.json``. That keeps the atomic-write guarantee without cross-container locking.
"""
from __future__ import annotations

import math
import os
from typing import Callable, Iterable, Optional

# The same helpers the serial path uses — imported, never reimplemented, so a signature
# computed here is byte-identical to one computed inside a worker.
from pipeline.duration_tts import (  # noqa: E402
    LANGUAGE_TO_CODE,
    MANIFEST_NAME,
    _load_manifest,
    _save_manifest_atomic,
    _segment_signature,
    resolve_nfe_step,
)

DEFAULT_MAX_SHARDS = int(os.environ.get("DUBBING_TTS_MAX_SHARDS", "8"))
MIN_SEGMENTS_PER_SHARD = int(os.environ.get("DUBBING_TTS_MIN_PER_SHARD", "2"))


def completed_indices(segments: list, output_dir: str, target_language: str,
                      nfe_step: Optional[int] = None) -> set:
    """Indices already finished on disk, by the SAME rule the serial resume uses.

    An index counts as done only when the manifest says ``ok``, the signature still matches
    the current text/duration/lang/nfe, and the WAV exists and is bigger than a bare header.
    A signature mismatch (e.g. translation re-ran and changed the line) is deliberately NOT
    done, so stale audio is never glued onto changed text.
    """
    lang_code = LANGUAGE_TO_CODE.get(target_language)
    if not lang_code:
        return set()
    nfe = resolve_nfe_step(nfe_step)
    manifest = _load_manifest(os.path.join(output_dir, MANIFEST_NAME))
    done = set()
    for i, seg in enumerate(segments):
        entry = manifest.get(str(i))
        if not entry or entry.get("status") != "ok":
            continue
        sig = _segment_signature(seg.get("text", "").strip(),
                                 seg["end"] - seg["start"], lang_code, nfe)
        if entry.get("sig") != sig:
            continue
        path = os.path.join(output_dir, f"segment_{i:04d}.wav")
        if os.path.exists(path) and os.path.getsize(path) > 44:
            done.add(i)
    return done


def plan_shards(pending: Iterable[int], max_shards: int = DEFAULT_MAX_SHARDS,
                min_per_shard: int = MIN_SEGMENTS_PER_SHARD) -> list:
    """Split pending global indices into contiguous shards, one per worker.

    Contiguous (not round-robin) so neighbouring segments — which share reference state and
    cache locality — stay together. The shard count is capped so we never spin up more
    containers than there is work for: each model load costs more than a segment, so a
    shard holding a single segment is usually a net loss (see the wall-clock model above).
    """
    idx = sorted(set(pending))
    if not idx:
        return []
    max_shards = max(1, int(max_shards))
    min_per_shard = max(1, int(min_per_shard))
    n_shards = max(1, min(max_shards, math.ceil(len(idx) / min_per_shard)))
    per = math.ceil(len(idx) / n_shards)
    return [idx[s:s + per] for s in range(0, len(idx), per)]


def merge_shard_results(results: Iterable[dict], output_dir: str, log_fn=None) -> dict:
    """Write each shard's WAV bytes into output_dir and merge manifest entries (single writer).

    ``results`` is an iterable of ``{"entries": {idx_str: entry}, "wavs": {idx_str: bytes},
    "log": [...]}``. Entry ``path`` values are rewritten to this run's output_dir, since a
    worker produced them under its own private directory.
    """
    def say(m):
        if log_fn is not None:
            try:
                log_fn(m)
            except Exception:
                pass

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    manifest = _load_manifest(manifest_path)
    written, failed = 0, []

    for res in results:
        for line in res.get("log", []) or []:
            say(f"    {line}")
        wavs = res.get("wavs", {}) or {}
        for key, blob in wavs.items():
            path = os.path.join(output_dir, f"segment_{int(key):04d}.wav")
            try:
                with open(path, "wb") as fh:
                    fh.write(blob)
                written += 1
            except Exception as e:
                failed.append(int(key))
                say(f"    WARNING: could not write {os.path.basename(path)}: {e}")
        for key, entry in (res.get("entries", {}) or {}).items():
            entry = dict(entry)
            entry["path"] = os.path.join(output_dir, f"segment_{int(key):04d}.wav")
            manifest[str(int(key))] = entry

    _save_manifest_atomic(manifest_path, manifest)
    return {"written": written, "failed": failed, "manifest": manifest}


def generate_tts_fanout(translated_segments: list, target_language: str, output_dir: str,
                        reference_audio_path: Optional[str] = None,
                        reference_text: Optional[str] = None, log_fn=None,
                        shard_runner: Optional[Callable] = None,
                        max_shards: int = DEFAULT_MAX_SHARDS) -> list:
    """Drop-in replacement for ``generate_tts_supervised`` that fans Step 6 across workers.

    Same contract: writes ``segment_%04d.wav`` + ``tts_manifest.json`` into ``output_dir``
    and returns the segments with ``audio_path`` set, so Steps 6.5 and 7 are unchanged.

    ``shard_runner(shard_specs) -> iterable of result dicts`` is injected so the
    orchestration is testable on CPU without Modal; it defaults to the Modal TTSEngine.
    Any segment a worker failed to produce is left for the caller's degradation path —
    this never raises a render away.
    """
    def say(m=""):
        if log_fn is not None:
            try:
                log_fn(m)
            except Exception:
                pass

    os.makedirs(output_dir, exist_ok=True)
    n = len(translated_segments)
    done = completed_indices(translated_segments, output_dir, target_language)
    pending = [i for i in range(n) if i not in done]
    if done:
        say(f"  Fan-out: {len(done)}/{n} segment(s) already complete on disk — resuming.")

    shards = plan_shards(pending, max_shards=max_shards)
    if not shards:
        say(f"  Fan-out: nothing to synthesize; all {n} segment(s) present.")
        return [{**s, "audio_path": os.path.join(output_dir, f"segment_{i:04d}.wav")}
                for i, s in enumerate(translated_segments)]

    say(f"  Fan-out: {len(pending)} segment(s) across {len(shards)} worker(s) "
        f"(~{max(len(s) for s in shards)} each).")

    ref_bytes = None
    if reference_audio_path and os.path.exists(reference_audio_path):
        with open(reference_audio_path, "rb") as fh:
            ref_bytes = fh.read()

    specs = [{
        "segments": translated_segments,
        "only_indices": shard,
        "target_language": target_language,
        "reference_bytes": ref_bytes,
        "reference_name": os.path.basename(reference_audio_path) if reference_audio_path else None,
        "reference_text": reference_text,
    } for shard in shards]

    if shard_runner is None:
        shard_runner = _modal_shard_runner()

    results = list(shard_runner(specs))
    merged = merge_shard_results(results, output_dir, log_fn=log_fn)

    produced = {int(k) for k in merged["manifest"].keys()
                if merged["manifest"][k].get("status") == "ok"}
    missing = [i for i in range(n) if i not in produced]
    if missing:
        say(f"  WARNING: fan-out did not settle {len(missing)} segment(s): {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}. They will be silence/degraded downstream.")
    else:
        say(f"  Fan-out complete: all {n} segment(s) settled.")

    return [{**s, "audio_path": os.path.join(output_dir, f"segment_{i:04d}.wav")}
            for i, s in enumerate(translated_segments)]


def _modal_shard_runner():
    """Default runner: map the shards onto the Modal TTSEngine defined in modal_app.py."""
    def run(specs):
        import modal
        engine = modal.Cls.from_name("indic-dubbing", "TTSEngine")()
        return engine.synth_shard.map(specs)
    return run
