#!/usr/bin/env python3
"""CPU-only property tests for the three Modal cost optimizations — memory snapshots, the
translation-off-GPU split, and the fan-out threshold. No GPU, no Modal, no torch. Backs the
cost/latency claims with checks that assert the PROPERTY (per CLAUDE rule 5), not that code ran.

Run:  python tools/test_split_and_snapshot.py

The pipeline stage functions in run_headless.py / duration_tts.py pull heavy deps (whisper,
demucs, torch) that are not installed on a CPU dev box, so we do NOT import those modules. We
lift the SPECIFIC small, self-contained functions under test out of the source with `ast` and
exec only those, injecting fakes for their globals. That keeps the test hermetic and fast while
still exercising the real code text that ships.

What it asserts:
  SNAPSHOT   -> _move_indicf5_to moves a cpu-cached model to cuda IN PLACE (model + ema_model +
                vocoder), is idempotent when already on device, and falls back to a load when
                nothing is cached; and modal_app's TTSEngine declares enable_memory_snapshot
                with a snap=True CPU-load phase and a post-restore cuda-move phase.
  SPLIT      -> run_headless round-trips the stage checkpoint through JSON EVEN WHEN segment
                dicts carry numpy scalars (np.bool_/int64 are the landmine a bare json.dump
                chokes on); and modal_app wires a CPU orchestrator (gpu_transcribe -> resume)
                with the vc/kill-switch routing to the single-container GPU path.
  THRESHOLD  -> plan_shards(max_shards=1) yields exactly ONE shard covering every segment (the
                below-threshold path = one container, one load), while the default fans out;
                and run_headless feeds max_shards=1 below DUBBING_TTS_FANOUT_MIN_SEGMENTS.
"""
import os, sys, ast, json, tempfile, shutil, re

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ -> repo root


def _extract(path, names, base_globals):
    """Exec ONLY the named top-level defs/assigns from `path` into a fresh namespace.

    Avoids importing the module (which would drag in torch/whisper/demucs). `names` is the set
    of top-level function/variable names to lift; `base_globals` seeds the exec namespace with
    whatever those bodies reference at module scope (os, json, math, injected stubs, ...).
    """
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & names:
                keep.append(node)
    ns = dict(base_globals)
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


class _FakeSub:
    def __init__(self): self.device = "cpu"
    def to(self, d): self.device = d; return self


class _FakeModel:
    """Stand-in for INF5Model: .to() records the device and returns self; has ema_model+vocoder."""
    def __init__(self):
        self.device = "cpu"
        self.ema_model = _FakeSub()
        self.vocoder = _FakeSub()
    def to(self, d): self.device = d; return self


def test_snapshot(fails):
    print("== SNAPSHOT: _move_indicf5_to device move ==")
    load_calls = []
    ns = _extract(
        os.path.join(REPO, "pipeline", "duration_tts.py"),
        {"_move_indicf5_to"},
        {
            "_indicf5_model": None,
            "_indicf5_device": None,
            # stub the loader so the "nothing cached yet" branch is observable without torch
            "_load_indicf5": lambda device="auto", log_fn=None: (load_calls.append(device) or ("LOADED", device)),
        },
    )
    move = ns["_move_indicf5_to"]

    # (a) nothing cached -> delegates to a normal load
    res = move("cuda")
    if not (load_calls == ["cuda"] and res == ("LOADED", "cuda")):
        fails.append(f"SNAPSHOT: no-model path did not fall back to _load_indicf5 (got {res}, calls {load_calls})")

    # (b) cpu-cached model -> moved in place to cuda, submodules follow, globals updated
    m = _FakeModel()
    ns["_indicf5_model"] = m
    ns["_indicf5_device"] = "cpu"
    model, dev = move("cuda")
    ok = (model is m and dev == "cuda" and m.device == "cuda"
          and m.ema_model.device == "cuda" and m.vocoder.device == "cuda"
          and ns["_indicf5_model"] is m and ns["_indicf5_device"] == "cuda")
    if not ok:
        fails.append(f"SNAPSHOT: cpu->cuda move incomplete (model={m.device}, ema={m.ema_model.device}, "
                     f"voc={m.vocoder.device}, dev={dev})")

    # (c) already on device -> idempotent no-op (does not call .to again)
    m.device = "SENTINEL_UNTOUCHED"
    model2, dev2 = move("cuda")
    if not (model2 is m and dev2 == "cuda" and m.device == "SENTINEL_UNTOUCHED"):
        fails.append("SNAPSHOT: move was not idempotent when already on the target device")

    print(f"   no-model fallback -> load({load_calls}); cpu->cuda in place; idempotent when on-device")

    # modal_app wiring (source-level): the class must actually opt into snapshots and split enter.
    ma = open(os.path.join(REPO, "deploy", "modal_app.py"), encoding="utf-8").read()
    checks = {
        "enable_memory_snapshot=True": "enable_memory_snapshot=True" in ma,
        "@modal.enter(snap=True)": "@modal.enter(snap=True)" in ma,
        'snap phase loads to CPU': '_load_indicf5("cpu")' in ma,
        'restore phase moves to cuda': '_move_indicf5_to("cuda")' in ma,
    }
    for label, ok in checks.items():
        if not ok:
            fails.append(f"SNAPSHOT: modal_app.py missing [{label}]")
    print(f"   modal_app snapshot wiring: {sum(checks.values())}/{len(checks)} present")


def test_split(fails):
    print("\n== SPLIT: stage checkpoint round-trips numpy scalars ==")
    import numpy as np
    ns = _extract(
        os.path.join(REPO, "run_headless.py"),
        {"_json_default", "_STATE_NAME", "_save_state", "_load_state"},
        {"os": os, "json": json},
    )
    save, load, jdefault = ns["_save_state"], ns["_load_state"], ns["_json_default"]

    # Segments as they exist AFTER the gate math: start/end/scores carry numpy types. np.bool_
    # and np.int64 are exactly what a bare json.dump raises on ("not serializable").
    segments = [
        {"text": "नमस्ते दुनिया", "start": np.float64(0.0), "end": np.float64(3.2),
         "isochrony_score": np.float64(0.91), "obeys": np.bool_(True), "n_phonemes": np.int64(11)},
        {"text": "यह एक परीक्षण है", "start": np.float64(3.2), "end": np.float64(8.0),
         "obeys": np.bool_(False), "n_phonemes": np.int64(17)},
    ]

    # The guard is load-bearing: prove a naive dump fails and the guarded one succeeds.
    raised = False
    try:
        json.dumps(segments)
    except TypeError:
        raised = True
    if not raised:
        fails.append("SPLIT: expected a bare json.dumps of numpy-bearing segments to raise (test premise wrong)")
    try:
        json.dumps(segments, default=jdefault)
    except Exception as e:
        fails.append(f"SPLIT: _json_default did not rescue the dump: {e}")

    work = tempfile.mkdtemp(prefix="splittest_")
    try:
        save(work, {"segments": segments, "vocals_path": "/j/vocals.wav",
                    "background_path": "/j/bg.wav", "video_path": "/j/in.mp4"})
        st = load(work)
        # Round-trip must preserve values AND be pure-python JSON (no numpy leaked through).
        ok_vals = (len(st["segments"]) == 2
                   and abs(st["segments"][0]["end"] - 3.2) < 1e-9
                   and st["segments"][0]["obeys"] is True
                   and st["segments"][1]["obeys"] is False
                   and st["segments"][0]["n_phonemes"] == 11
                   and st["vocals_path"] == "/j/vocals.wav"
                   and st["background_path"] == "/j/bg.wav")
        if not ok_vals:
            fails.append(f"SPLIT: checkpoint values not preserved across save/load: {st['segments'][0]}")
        for seg in st["segments"]:
            for k, v in seg.items():
                if type(v).__module__ == "numpy":
                    fails.append(f"SPLIT: numpy type leaked through JSON for {k}={v!r}")
        raw = open(os.path.join(work, ns["_STATE_NAME"]), encoding="utf-8").read()
        json.loads(raw)   # must be valid JSON on disk
        print(f"   naive dump raises={raised}; guarded round-trip preserved 2 segments, "
              "no numpy leaked, valid JSON on disk")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # modal_app wiring (source-level): CPU orchestrator + phase functions + routing.
    ma = open(os.path.join(REPO, "deploy", "modal_app.py"), encoding="utf-8").read()
    rh = open(os.path.join(REPO, "run_headless.py"), encoding="utf-8").read()
    checks = {
        "gpu_transcribe runs prep": bool(re.search(r"def gpu_transcribe\b", ma)) and 'stages="prep"' in ma,
        "gpu_full_run runs all": bool(re.search(r"def gpu_full_run\b", ma)) and 'stages="all"' in ma,
        "orchestrator resumes": 'stages="resume"' in ma,
        "reload before resume": "jobs_vol.reload()" in ma,
        "kill-switch DUBBING_MODAL_SPLIT": "DUBBING_MODAL_SPLIT" in ma,
        "vc -> single-container path": bool(re.search(r'mode == "vc".*gpu_full_run', ma, re.S)),
        "run_headless prep persists+exits": 'stages == "prep"' in rh and "_save_state(out_dir" in rh,
        "run_headless resume loads": 'RESUME' in rh and "_load_state(out_dir)" in rh,
    }
    for label, ok in checks.items():
        if not ok:
            fails.append(f"SPLIT: wiring missing [{label}]")
    print(f"   split wiring: {sum(checks.values())}/{len(checks)} present")

    # The CPU orchestrator's dub_video must NOT request a GPU (that is the whole point). Read the
    # decorator IMMEDIATELY above `def dub_video` (rfind, not a DOTALL regex that would grab an
    # earlier gpu= decorator).
    didx = ma.index("\ndef dub_video")
    deco = ma[ma.rfind("@app.function(", 0, didx):didx]
    if "gpu=" in deco:
        fails.append("SPLIT: dub_video is not a CPU function (its @app.function has gpu=)")
    else:
        print("   dub_video orchestrator is CPU (no gpu= on its decorator) ✓")


def test_threshold(fails):
    print("\n== THRESHOLD: below the cutoff, fan-out collapses to one shard ==")
    import typing
    ns = _extract(
        os.path.join(REPO, "deploy", "tts_fanout.py"),
        {"plan_shards", "DEFAULT_MAX_SHARDS", "MIN_SEGMENTS_PER_SHARD"},
        # tts_fanout uses `from __future__ import annotations`, which the ast-lift drops, so the
        # signature annotations (Iterable/Optional) evaluate eagerly — seed the typing names.
        {"os": os, "math": __import__("math"),
         "Iterable": typing.Iterable, "Optional": typing.Optional, "Callable": typing.Callable},
    )
    plan = ns["plan_shards"]

    # Below threshold -> orchestrator passes max_shards=1 -> exactly ONE shard with all segments.
    small = plan(list(range(5)), max_shards=1)
    if not (len(small) == 1 and small[0] == list(range(5))):
        fails.append(f"THRESHOLD: max_shards=1 did not yield one all-covering shard: {small}")

    # Above threshold -> real fan-out (default max_shards=8, min_per_shard=2 -> 8 shards for 30).
    big = plan(list(range(30)))
    covered = sorted(i for shard in big for i in shard)
    if not (len(big) > 1 and covered == list(range(30))):
        fails.append(f"THRESHOLD: default plan did not fan out / cover all: {len(big)} shards, covered={len(covered)}")
    # Shards must be contiguous and disjoint (the fan-out contract).
    flat = [i for shard in big for i in shard]
    if flat != sorted(flat) or len(set(flat)) != len(flat):
        fails.append("THRESHOLD: fan-out shards are not contiguous+disjoint")
    print(f"   max_shards=1 -> 1 shard covering {small[0]}; default(30) -> {len(big)} contiguous shards")

    # run_headless must actually gate on the threshold and pass max_shards=1 below it.
    rh = open(os.path.join(REPO, "run_headless.py"), encoding="utf-8").read()
    checks = {
        "reads DUBBING_TTS_FANOUT_MIN_SEGMENTS": "DUBBING_TTS_FANOUT_MIN_SEGMENTS" in rh,
        "caps to one shard below threshold": 'fanout_kw["max_shards"] = 1' in rh,
        "compares segment count to threshold": bool(re.search(r"len\(translated_segments\)\s*<\s*min_fanout", rh)),
    }
    for label, ok in checks.items():
        if not ok:
            fails.append(f"THRESHOLD: run_headless missing [{label}]")
    print(f"   run_headless threshold gate: {sum(checks.values())}/{len(checks)} present")


def main():
    fails = []
    test_snapshot(fails)
    test_split(fails)
    test_threshold(fails)
    print("\n" + ("FAIL:\n  - " + "\n  - ".join(fails) if fails else "ALL PROPERTIES HOLD ✓"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
