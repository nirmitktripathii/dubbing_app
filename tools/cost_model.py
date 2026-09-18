#!/usr/bin/env python3
"""Per-run Modal cost model: CURRENT (one L4 container holds the GPU for the whole run + fan-out)
vs OPTIMIZED (memory snapshots + translation-off-GPU split + fan-out threshold). Grounded in the
ONE measured production run and Modal's per-second prices, so the figures in any report regenerate
here (CLAUDE rule 8). Run:  python tools/cost_model.py

MEASURED anchor -- the 25 s clip that ran end-to-end on L4 (tools/profile_log.py on the prod log):
    total 275 s;  Demucs 13 s;  Whisper(medium) 7 s;  translation 170 s (Gemini/CPU, NO GPU);
    fan-out TTS 82 s (~3-4 containers, dominated by cold IndicF5 load);  merge 3 s;  5 segments.

CONFIRMED Modal prices (modal.com/pricing, per second):
    L4 GPU 0.000222 ;  CPU 0.0000131 / core ;  memory 0.00000222 / GiB

This is an ESTIMATE. The soft input is translation time (34 s/seg) -- extrapolated from a single
5-segment sample and sequential, so it dominates wall clock at long clips. But in OPTIMIZED that
time bills CPU, not L4, so the $/run estimate is robust even where the wall-clock guess is soft.
"""
import math
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

L4  = 0.000222      # $/s, one L4 GPU
CPU = 0.0000131     # $/s per core
MEM = 0.00000222    # $/s per GiB


def gpu_s(seconds, cores=1.0, gib=4.0):   # one L4 container-second (GPU also bills CPU+RAM)
    return seconds * (L4 + cores * CPU + gib * MEM)


def cpu_s(seconds, cores=0.5, gib=2.0):   # one CPU container-second
    return seconds * (cores * CPU + gib * MEM)


# ── Stage-time model (seconds), anchored to reproduce the 25 s / 5-segment run ──────────────
def n_segments(T):  return max(1, round(T / 5.0))     # 25 s -> 5  (checked below)
def t_extract(T):   return 2.0                        # ffmpeg, ~flat
def t_demucs(T):    return 5.0 + 0.32 * T             # 25 -> 13
def t_whisper(T):   return 3.0 + 0.16 * T             # 25 -> 7
def t_translate(n): return 34.0 * n                   # 5 -> 170  (sequential Gemini, CPU-bound)
def t_merge(T):     return 3.0                        # stream-copy (P2)

PER_SEG   = 3.0     # IndicF5 synth per segment on L4
LOAD_COLD = 28.0    # cold IndicF5 load: f5_tts import (~16 s) + 1.3 GB weight materialisation
LOAD_SNAP = 6.0     # snapshot RESTORE instead of re-import/re-materialise
THRESHOLD = 12      # DUBBING_TTS_FANOUT_MIN_SEGMENTS: below this, 1 shard (1 container, 1 load)


def shards_current(n):  return max(1, min(8, math.ceil(n / 2)))                 # no threshold
def shards_opt(n):      return 1 if n < THRESHOLD else max(1, min(8, math.ceil(n / 2)))
def tts_wall(n, shards, load):  return load + math.ceil(n / shards) * PER_SEG


def cost_current(T):
    """One L4 runs ALL stages (GPU billed the whole time, incl. the 62% translation) and also
    spins fan-out TTSEngine L4s during Step 6 (double-paid there)."""
    n = n_segments(T)
    k = shards_current(n)
    W = (t_extract(T) + t_demucs(T) + t_whisper(T) + t_translate(n)
         + tts_wall(n, k, LOAD_COLD) + t_merge(T))
    gpu = gpu_s(W) + gpu_s(k * LOAD_COLD + n * PER_SEG)
    return {"n": n, "wall": W, "cost": gpu, "gpu": gpu, "cpu": 0.0}


def cost_opt(T):
    """gpu_transcribe (L4) does Steps 1-3; translation+assembly run on a CPU orchestrator (no GPU
    billed) while TTS fans out to snapshot-warm TTSEngine L4s (1 shard below threshold)."""
    n = n_segments(T)
    k = shards_opt(n)
    transcribe_w = t_extract(T) + t_demucs(T) + t_whisper(T)
    orch_w = t_translate(n) + tts_wall(n, k, LOAD_SNAP) + t_merge(T)
    gpu = gpu_s(transcribe_w) + gpu_s(k * LOAD_SNAP + n * PER_SEG)
    cpu = cpu_s(orch_w)
    return {"n": n, "wall": transcribe_w + orch_w, "cost": gpu + cpu, "gpu": gpu, "cpu": cpu}


def _selfcheck():
    # The model must reproduce the measured 25 s / 5-seg anchor within tolerance.
    assert n_segments(25) == 5
    assert abs(t_demucs(25) - 13) < 0.5 and abs(t_whisper(25) - 7) < 0.5
    assert abs(t_translate(5) - 170) < 1 and abs(t_merge(25) - 3) < 0.5


CLIPS = [(30, "30 s"), (60, "60 s"), (120, "2 min"), (300, "5 min"), (600, "10 min")]

if __name__ == "__main__":
    _selfcheck()
    print(f"{'clip':>6} {'segs':>4} | {'wall':>7} {'L4-sec-eq':>9} {'$/run':>8} "
          f"| {'wall':>7} {'$/run':>8} {'CPU part':>9} | {'saving':>7}")
    print(f"{'':>6} {'':>4} | {'CURRENT':^26} | {'OPTIMIZED':^27} |")
    print("-" * 90)
    for T, label in CLIPS:
        c, o = cost_current(T), cost_opt(T)
        save = (1 - o["cost"] / c["cost"]) * 100
        print(f"{label:>6} {c['n']:>4} | {c['wall']:>6.0f}s {c['gpu']/L4:>9.0f} ${c['cost']:>6.3f} "
              f"| {o['wall']:>6.0f}s ${o['cost']:>6.3f} ${o['cpu']:>7.4f} | {save:>5.0f}%")
    print("-" * 90)
    print("L4-sec-eq = current L4-equivalent seconds billed. Wall clock barely moves because")
    print("sequential translation dominates it in BOTH; the split is a COST win (GPU freed during")
    print("translation), not a wall-clock one. Parallelising Gemini (deferred) is the latency lever.")
