"""CPU test: every translation call reaches ONLY gemini-3.1-flash-lite / gemini-3.5-flash-lite.

Asserts the property at the API boundary — a fake client records the model id of every
generate_content call — rather than inspecting the chain helpers alone. Covers the default
chains, a Gemma env override, a caller passing a Gemma chain explicitly, and a fall-through
after the head model errors.

Run:  python tools/test_translation_models.py      (no network, no API key)
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ["DUBBING_CACHE_DIR"] = tempfile.mkdtemp(prefix="tm_cache_")
os.environ["DUBBING_GEMINI_RPM"] = "100000"   # no client-side pacing sleeps in the test

from pipeline import isochrony_translation as it  # noqa: E402

ALLOWED = {"gemini-3.1-flash-lite", "gemini-3.5-flash-lite"}
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail and not ok else ''}")
    FAILS += 0 if ok else 1


class _Resp:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """Records every model id asked for; optionally fails the first N calls with a 500."""
    def __init__(self, fail_first=0):
        self.seen, self._fail = [], fail_first
        self.models = self

    def generate_content(self, model, contents, config):
        self.seen.append(model)
        if self._fail > 0:
            self._fail -= 1
            raise RuntimeError("500 INTERNAL. Internal error encountered.")
        return _Resp('["ok"]')


def run(models=None, fail_first=0):
    c = FakeClient(fail_first=fail_first)
    it._call_gemini(c, "translate", temperature=0.2, models=models, log_fn=lambda m: None)
    return c.seen


for k in ("DUBBING_GEMINI_BULK_MODEL", "DUBBING_GEMINI_REFINE_MODEL", "DUBBING_GEMINI_MODEL"):
    os.environ.pop(k, None)

check("bulk chain defaults to flash-lite 3.1 -> 3.5",
      it._bulk_models() == ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"], str(it._bulk_models()))
check("refine chain defaults to flash-lite 3.1 -> 3.5",
      it._refine_models() == ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"], str(it._refine_models()))

seen = run(models=it._bulk_models())
check("default bulk call hits gemini-3.1-flash-lite", seen == ["gemini-3.1-flash-lite"], str(seen))

os.environ["DUBBING_GEMINI_BULK_MODEL"] = "gemma-4-31b-it"
seen = run(models=it._bulk_models())
check("Gemma bulk env override is ignored", set(seen) <= ALLOWED and seen, str(seen))
os.environ["DUBBING_GEMINI_BULK_MODEL"] = "gemini-3.5-flash-lite"
check("allowlisted bulk env override leads the chain",
      it._bulk_models()[0] == "gemini-3.5-flash-lite", str(it._bulk_models()))
os.environ.pop("DUBBING_GEMINI_BULK_MODEL")

seen = run(models=["gemma-4-31b-it", "gemma-4-26b-a4b-it"])
check("an explicit Gemma chain never reaches the API", set(seen) <= ALLOWED and seen, str(seen))

seen = run(models=["gemma-4-31b-it"], fail_first=6)
check("fall-through after head errors stays on the allowlist",
      set(seen) <= ALLOWED and "gemini-3.5-flash-lite" in seen, str(seen))

os.environ["DUBBING_GEMINI_MODEL"] = "gemma-4-26b-a4b-it"
seen = run()
check("Gemma legacy refine env is ignored", set(seen) <= ALLOWED and seen, str(seen))
os.environ.pop("DUBBING_GEMINI_MODEL")

from pipeline import translation_cache  # noqa: E402
check("candidate cache uses the fresh flash-lite namespace",
      translation_cache._translation_cache_path().endswith("translations_gemini.json"),
      translation_cache._translation_cache_path())

print("\nALL PASSED" if not FAILS else f"\n{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
