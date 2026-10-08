#!/usr/bin/env python3
"""CPU-only property test for the 503 cooldown policy in `_call_gemini`.

No network, no GPU, no google-genai import: it lifts `_call_gemini` and its small helpers out
of pipeline/isochrony_translation.py with `ast` and runs them against a fake client and a fake
clock, so the policy is proven without loading the translation stack.

Run:  python tools/test_gemini_cooldown.py

Properties asserted (policy: 3.5 first; if every model 503s, cool down, flip the order, repeat
for N rounds; only then raise so the caller's own fallback applies):
  P1 ORDER        — the default chain is 3.5-flash-lite then 3.1-flash-lite.
  P2 HEALTHY      — a healthy first model answers once, no sleep.
  P3 FAILOVER     — a 503 on 3.5 moves straight to 3.1 with NO sleep and NO same-model retry.
  P4 COOLDOWN     — both 503 in round 1 -> one sleep of the base cooldown, then round 2
                    starts on 3.1 (order flipped), then 3.5.
  P5 ROUNDS       — if every call 503s, each model is attempted exactly 3 times, there are
                    exactly 2 cooldowns that DOUBLE (base, 2*base), and the 503 is raised.
  P6 RECOVERY     — a model that recovers in round 3 returns its text.
  P7 TIMEOUT      — a client-side timeout is treated like a 503.
  P8 ROUND TOGGLE — odd rounds reverse only the Gemini tail; leading Gemma models keep place.
  P10 EXPONENTIAL — _cooldown_for_round doubles per round and is capped.
  P9 ENV KNOBS    — DUBBING_GEMINI_COOLDOWN_S / DUBBING_GEMINI_ROUNDS parse, clamp, default.
"""
import ast
import os
import random
import sys
import time
import typing
import types as _types

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO, "pipeline", "isochrony_translation.py")

FUNCS = {"_call_gemini", "_gemini_rounds", "_gemini_cooldown_seconds", "_round_chain", "_is_gemma",
         "_refine_models", "_cooldown_for_round"}
CONSTS = {"_GEMINI_FALLBACK", "_GEMINI_COOLDOWN_S_DEFAULT", "_GEMINI_COOLDOWN_CAP_S",
          "_GEMINI_ROUNDS_DEFAULT"}


class FakeClock:
    def __init__(self):
        self.sleeps = []

    def sleep(self, s):
        self.sleeps.append(s)

    time = staticmethod(time.time)


def _load():
    src = open(SRC_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    clock = FakeClock()
    g = {
        "os": os, "random": random, "time": clock, "List": list,
        "Optional": typing.Optional, "Callable": typing.Callable,
        "translation_cache": _types.SimpleNamespace(
            rpd_limit=lambda: None, count_today=lambda m: 0, record_request=lambda m: None),
        "types": _types.SimpleNamespace(GenerateContentConfig=lambda **kw: kw),
        "_throttle": lambda m: None,
        "_call_timeout_seconds": lambda m="": 1.0,
        "_call_heartbeat_seconds": lambda: 1.0,
        "_run_with_timeout": lambda fn, t, heartbeat_fn=None, heartbeat_interval=0: fn(),
        "_parse_retry_delay": lambda e: None,
    }
    picked = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            picked.add(node.name)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in CONSTS for t in node.targets):
            picked.add(node.targets[0].id)
        else:
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), SRC_PATH, "exec"), g)
    missing = (FUNCS | CONSTS) - picked
    if missing:
        raise AssertionError(f"could not lift {sorted(missing)}; names changed?")
    return g, clock


class FakeClient:
    """`script` maps model -> list of outcomes per call; an outcome is a str (reply text) or an
    Exception. The last outcome repeats once the list is exhausted."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.models = _types.SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append(model)
        seq = self.script[model]
        n = sum(1 for m in self.calls if m == model) - 1
        out = seq[min(n, len(seq) - 1)]
        if isinstance(out, Exception):
            raise out
        return _types.SimpleNamespace(text=out)


M35, M31 = "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"
E503 = RuntimeError("503 UNAVAILABLE. The model is overloaded.")


def run(script, env=None, models=None):
    for k in ("DUBBING_GEMINI_COOLDOWN_S", "DUBBING_GEMINI_ROUNDS",
              "DUBBING_GEMINI_REFINE_MODEL", "DUBBING_GEMINI_MODEL"):
        os.environ.pop(k, None)
    os.environ.update(env or {})
    g, clock = _load()
    client = FakeClient(script)
    err = None
    out = None
    try:
        out = g["_call_gemini"](client, "p", models=models, log_fn=lambda m: None)
    except Exception as e:  # noqa: BLE001 - the test inspects what escaped
        err = e
    return g, client, clock, out, err


def main():
    # P1
    g, _ = _load()
    assert g["_GEMINI_FALLBACK"] == [M35, M31], g["_GEMINI_FALLBACK"]

    # P2
    _, c, clock, out, err = run({M35: ["ok"], M31: ["no"]})
    assert (out, err, c.calls, clock.sleeps) == ("ok", None, [M35], []), (out, err, c.calls)

    # P3
    _, c, clock, out, err = run({M35: [E503], M31: ["from31"]}, models=[M35, M31])
    assert (out, err, c.calls, clock.sleeps) == ("from31", None, [M35, M31], []), (c.calls, clock.sleeps)

    # P4
    _, c, clock, out, err = run({M35: [E503, "r2"], M31: [E503]},
                                env={"DUBBING_GEMINI_COOLDOWN_S": "7"}, models=[M35, M31])
    assert c.calls == [M35, M31, M31, M35], c.calls          # round 2 starts on 3.1
    assert clock.sleeps == [7.0], clock.sleeps
    assert (out, err) == ("r2", None), (out, err)

    # P5
    _, c, clock, out, err = run({M35: [E503], M31: [E503]},
                                env={"DUBBING_GEMINI_COOLDOWN_S": "5"}, models=[M35, M31])
    assert err is not None and "503" in str(err), repr(err)
    assert c.calls.count(M35) == 3 and c.calls.count(M31) == 3, c.calls
    assert c.calls == [M35, M31, M31, M35, M35, M31], c.calls
    assert clock.sleeps == [5.0, 10.0], clock.sleeps          # exponential: base, 2*base

    # P6
    _, c, clock, out, err = run({M35: [E503, E503, "late"], M31: [E503]}, models=[M35, M31])
    assert (out, err) == ("late", None) and len(clock.sleeps) == 2, (out, err, clock.sleeps)

    # P7
    _, c, clock, out, err = run({M35: [TimeoutError("timed out after 90s")], M31: ["ok31"]},
                                models=[M35, M31])
    assert (out, c.calls) == ("ok31", [M35, M31]), (out, c.calls)

    # P8
    g, _ = _load()
    rc = g["_round_chain"]
    assert rc([M35, M31], 0) == [M35, M31] and rc([M35, M31], 1) == [M31, M35]
    assert rc(["gemma-a", "gemma-b", M35, M31], 1) == ["gemma-a", "gemma-b", M31, M35]
    assert rc([M35], 1) == [M35]

    # P9
    for k in ("DUBBING_GEMINI_COOLDOWN_S", "DUBBING_GEMINI_ROUNDS"):
        os.environ.pop(k, None)
    g, _ = _load()
    assert g["_gemini_cooldown_seconds"]() == 20.0 and g["_gemini_rounds"]() == 3
    os.environ["DUBBING_GEMINI_COOLDOWN_S"] = "-4"
    os.environ["DUBBING_GEMINI_ROUNDS"] = "0"
    assert g["_gemini_cooldown_seconds"]() == 0.0 and g["_gemini_rounds"]() == 1
    os.environ["DUBBING_GEMINI_COOLDOWN_S"] = "abc"
    os.environ["DUBBING_GEMINI_ROUNDS"] = "x"
    assert g["_gemini_cooldown_seconds"]() == 20.0 and g["_gemini_rounds"]() == 3
    for k in ("DUBBING_GEMINI_COOLDOWN_S", "DUBBING_GEMINI_ROUNDS"):
        os.environ.pop(k, None)

    # P10
    g, _ = _load()
    cd, cap = g["_cooldown_for_round"], g["_GEMINI_COOLDOWN_CAP_S"]
    assert [cd(20.0, n) for n in (1, 2, 3)] == [20.0, 40.0, 80.0]
    assert cd(20.0, 4) == cap == 120.0 and cd(20.0, 9) == cap
    assert cd(0.0, 3) == 0.0

    print("PASS  test_gemini_cooldown: P1-P10")


if __name__ == "__main__":
    main()
