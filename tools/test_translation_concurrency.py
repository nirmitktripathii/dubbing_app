#!/usr/bin/env python3
"""CPU-only property test for the parallel Gemini dispatch in isochrony_translation.py.

No GPU, no network, no google-genai/espeak import: it lifts ONLY `_run_batches_concurrent`
and `_gemini_concurrency` out of the module with `ast` and execs them, so the properties are
proven without loading the heavy translation stack (CLAUDE rule 8: a property must be
regenerable by a repo script; rule 5: assert the property, not that code ran).

Run:  python tools/test_translation_concurrency.py

Properties asserted:
  P1 PARALLELISM   — with concurrency C>1 over N independent batches, more than one batch is
                     genuinely in flight at once AND wall time is far below the serial sum.
  P2 CORRECTNESS   — the merged {seg_id: cands} is IDENTICAL to the sequential result, because
                     chunks are disjoint (no update can clobber another chunk's segments).
  P3 SERIAL PARITY — concurrency<=1, or a single chunk, runs strictly sequentially (max
                     in-flight == 1) and returns the same dict.
  P4 DETERMINISM   — `served` is aggregated in CHUNK order, not completion order, even when
                     later chunks finish first.
  P5 NO SWALLOW    — a run_one exception propagates (the helper never hides a failure).
  P6 ENV KNOB      — _gemini_concurrency() parses/clamps DUBBING_GEMINI_CONCURRENCY to [1,16].
"""
import ast
import os
import sys
import time
import threading
import concurrent.futures

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO, "pipeline", "isochrony_translation.py")


def _lift(names):
    """Exec just the named top-level functions from isochrony_translation.py, with the modules
    they reference injected — no import of the heavy module itself."""
    src = open(SRC_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    g = {"os": os, "concurrent": concurrent, "time": time, "threading": threading,
         "List": list, "Optional": object, "Callable": object}
    picked = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = compile(ast.Module(body=[node], type_ignores=[]), SRC_PATH, "exec")
            exec(code, g)
            picked += 1
    if picked != len(names):
        raise AssertionError(f"lifted {picked}/{len(names)} functions; names changed?")
    return g


G = _lift({"_run_batches_concurrent", "_gemini_concurrency"})
run_batches = G["_run_batches_concurrent"]
gemini_concurrency = G["_gemini_concurrency"]


def make_run_one(sleep_for, tracker):
    """A fake per-chunk worker. `numbered` is (bnum, seg_id). Records concurrency and returns
    a partial dict keyed by seg_id plus a per-call served list (like the real run_one)."""
    def run_one(numbered):
        bnum, seg_id = numbered
        with tracker["lock"]:
            tracker["cur"] += 1
            tracker["max"] = max(tracker["max"], tracker["cur"])
        try:
            time.sleep(sleep_for(seg_id))
            return {seg_id: [f"cand-{seg_id}"]}, [f"m{seg_id}"]
        finally:
            with tracker["lock"]:
                tracker["cur"] -= 1
    return run_one


def new_tracker():
    return {"cur": 0, "max": 0, "lock": threading.Lock()}


def chunks(n):
    # mimic list(enumerate(chunks, 1)) — the element handed to run_one is (bnum, payload)
    return [(i + 1, i) for i in range(n)]


fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


# ── P1 PARALLELISM + P2 CORRECTNESS ─────────────────────────────────────────────────────────
N, DELAY, C = 8, 0.20, 4
tr = new_tracker()
t0 = time.time()
merged_par, served_par = run_batches(chunks(N), make_run_one(lambda s: DELAY, tr), C)
wall_par = time.time() - t0
serial_sum = N * DELAY

check(tr["max"] >= 2, f"P1: no real concurrency observed (max in-flight={tr['max']})")
check(tr["max"] <= C, f"P1: concurrency exceeded the cap ({tr['max']} > {C})")
check(wall_par < serial_sum * 0.6,
      f"P1: not faster than serial (wall={wall_par:.2f}s vs serial≈{serial_sum:.2f}s)")
expected = {i: [f"cand-{i}"] for i in range(N)}
check(merged_par == expected, "P2: merged dict wrong under concurrency")
check(len(served_par) == N, f"P2: served count wrong ({len(served_par)} != {N})")
print(f"P1/P2: max_in_flight={tr['max']} cap={C}  wall={wall_par:.2f}s vs serial≈{serial_sum:.2f}s  "
      f"segments={len(merged_par)}  -> {'ok' if not fails else 'FAIL'}")

# ── P3 SERIAL PARITY ────────────────────────────────────────────────────────────────────────
tr1 = new_tracker()
merged_seq, served_seq = run_batches(chunks(N), make_run_one(lambda s: 0.02, tr1), 1)
check(tr1["max"] == 1, f"P3: concurrency=1 was not serial (max in-flight={tr1['max']})")
check(merged_seq == expected, "P3: concurrency=1 merged dict differs from expected")
check(merged_seq == merged_par, "P3: concurrency=1 and concurrency>1 disagree on the merge")

trS = new_tracker()
_m, _s = run_batches([(1, 0)], make_run_one(lambda s: 0.02, trS), 8)   # single chunk
check(trS["max"] == 1, f"P3: single chunk not run serially (max in-flight={trS['max']})")
print(f"P3: serial max_in_flight={tr1['max']}, single-chunk max_in_flight={trS['max']}  "
      f"-> {'ok' if not fails else 'FAIL'}")

# ── P4 DETERMINISM: served in chunk order, not completion order ──────────────────────────────
tr2 = new_tracker()
# invert the delay so the LAST chunk finishes FIRST; served must still be m0,m1,...,m7
merged_inv, served_inv = run_batches(
    chunks(N), make_run_one(lambda s: 0.02 * (N - s), tr2), C)
check(served_inv == [f"m{i}" for i in range(N)],
      f"P4: served not in chunk order under out-of-order completion: {served_inv}")
check(merged_inv == expected, "P4: merged dict wrong under out-of-order completion")
print(f"P4: served order={served_inv}  -> {'ok' if not fails else 'FAIL'}")

# ── P5 NO SWALLOW: a run_one exception propagates ───────────────────────────────────────────
def boom(numbered):
    raise RuntimeError("kaboom")


for c in (1, 4):
    raised = False
    try:
        run_batches(chunks(3), boom, c)
    except RuntimeError:
        raised = True
    check(raised, f"P5: run_one exception was swallowed at concurrency={c}")
print(f"P5: exception propagates at concurrency 1 and 4  -> {'ok' if not fails else 'FAIL'}")

# ── P6 ENV KNOB ─────────────────────────────────────────────────────────────────────────────
cases = {None: 4, "1": 1, "4": 4, "0": 1, "-3": 1, "999": 16, "abc": 4, "": 4}
for val, want in cases.items():
    if val is None:
        os.environ.pop("DUBBING_GEMINI_CONCURRENCY", None)
    else:
        os.environ["DUBBING_GEMINI_CONCURRENCY"] = val
    got = gemini_concurrency()
    check(got == want, f"P6: DUBBING_GEMINI_CONCURRENCY={val!r} -> {got}, expected {want}")
os.environ.pop("DUBBING_GEMINI_CONCURRENCY", None)
print(f"P6: env parsing/clamping over {len(cases)} cases  -> {'ok' if not fails else 'FAIL'}")

print("\n" + ("FAIL:\n  - " + "\n  - ".join(fails) if fails else "ALL PROPERTIES HOLD ✓"))
sys.exit(1 if fails else 0)
