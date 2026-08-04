"""
tests/test_phoneme_ruler.py
===========================
Guards the repair of the ruler bug. These tests run WITHOUT espeak-ng, torch, or a GPU —
deliberately, because the environment where the bug happened was one that lacked espeak-ng
and did not notice.

The most important test here is `test_preflight_rejects_passthrough`. Merely checking that
a phonemizer is importable would have passed in the session that mislabelled the corpus;
what was needed was a check that the returned symbols are *phonemes* rather than the
input's own characters. That is what the preflight asserts and what this test pins.

    python -m pytest tests/test_phoneme_ruler.py -q
    # or, without pytest:
    python tests/test_phoneme_ruler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import phonemes as P  # noqa: E402


# ---------------------------------------------------------------------------------------
# Normalisation: what counts as one phoneme
# ---------------------------------------------------------------------------------------

def test_stress_marks_are_not_phonemes():
    # ˈ and ˌ mark which syllable is emphasised; they are not sounds.
    assert P._normalise_tokens("ˈd uː dʰ") == ["d", "uː", "dʰ"]
    assert P._normalise_tokens("ˌk ə ˈl a") == ["k", "ə", "l", "a"]


def test_length_mark_stays_attached_to_its_vowel():
    # aː is one long vowel, not two segments. Duration is exactly what we are proxying,
    # so splitting it would double-count the thing we care most about.
    assert P._normalise_tokens("a aː") == ["a", "aː"]


def test_language_switch_tags_are_stripped():
    # espeak emits (en)/(hi) around foreign words — typically English brand names inside
    # Indic text. They are markup, not sound.
    assert P._normalise_tokens("d uː (en) m ɪ l k (hi) dʰ") == ["d", "uː", "m", "ɪ", "l", "k", "dʰ"]


def test_word_separator_does_not_produce_tokens():
    assert P._normalise_tokens("d uː | m eː") == ["d", "uː", "m", "eː"]


def test_token_of_only_stress_marks_collapses():
    assert P._normalise_tokens("ˈ ˌ a") == ["a"]


# ---------------------------------------------------------------------------------------
# The wrong ruler, named explicitly so audits can detect it
# ---------------------------------------------------------------------------------------

def test_count_chars_ignores_whitespace():
    assert P.count_chars("दूध में कैल्शियम") == len("दूधमेंकैल्शियम")
    assert P.count_chars("  a b\tc\n") == 3
    assert P.count_chars("") == 0
    assert P.count_chars(None) == 0


# ---------------------------------------------------------------------------------------
# Fail-fast: the behaviour the old code did not have
# ---------------------------------------------------------------------------------------

def test_missing_backend_raises_with_an_actionable_hint(monkeypatch=None):
    """A dead backend must raise, and the message must say how to fix it.

    The old implementation returned `[c for c in text if not c.isspace()]` here, which is
    why an entire corpus was labelled in characters without one visible symptom.
    """
    P._backend_version.cache_clear()
    real = P._backend_version

    def boom():
        raise P.G2PUnavailable("simulated missing espeak-ng\n" + P._INSTALL_HINT)

    P._backend_version = boom
    try:
        raised = False
        try:
            P.phonemize("दूध", "hi")
        except P.G2PUnavailable as e:
            raised = True
            assert "apt-get" in str(e), "the error must tell the user how to install espeak-ng"
        assert raised, "phonemize must RAISE on a dead backend, never fall back to characters"
    finally:
        P._backend_version = real
        P._backend_version.cache_clear()


def test_preflight_rejects_passthrough():
    """The check that would have caught the original bug on day one.

    Simulates a backend that returns the input's own characters — which is exactly what a
    character-split fallback looks like from the outside, and exactly what produces
    plausible, non-empty, completely wrong labels.
    """
    P._backend_version.cache_clear()
    real_version, real_phonemize = P._backend_version, P.phonemize
    P._backend_version = lambda: "1.51-fake"
    P.phonemize = lambda text, lang: [c for c in text if not c.isspace()]
    try:
        raised = False
        try:
            P.assert_g2p_available(["hi"], verbose=False)
        except P.G2PUnavailable as e:
            raised = True
            assert "passthrough" in str(e).lower() or "character-split" in str(e).lower()
        assert raised, "preflight must reject a backend returning the input's own characters"
    finally:
        P._backend_version, P.phonemize = real_version, real_phonemize
        P._backend_version.cache_clear()


def test_preflight_accepts_real_phonemes():
    """Control for the test above: genuine IPA output must pass."""
    P._backend_version.cache_clear()
    real_version, real_phonemize = P._backend_version, P.phonemize
    P._backend_version = lambda: "1.51-fake"
    P.phonemize = lambda text, lang: ["d", "uː", "dʰ", "m", "eː"]
    try:
        m = P.assert_g2p_available(["hi"], verbose=False)
        assert m["ruler"].startswith("phonemes:espeak-ng")
        assert m["languages"]["hi"]["passthrough"] is False
    finally:
        P._backend_version, P.phonemize = real_version, real_phonemize
        P._backend_version.cache_clear()


def test_every_language_has_a_canary():
    """A language without a canary cannot be preflighted, so the preflight would pass it
    by default — the failure mode this whole module exists to prevent."""
    from common.languages import LANGUAGES
    missing = sorted(set(LANGUAGES) - set(P._CANARIES))
    assert not missing, f"no preflight canary for: {missing}"


def test_canaries_are_in_the_right_script():
    """A canary written in the wrong script would phonemize fine and prove nothing."""
    import unicodedata
    from common.languages import LANGUAGES

    # Unicode still uses several scripts' older names in its character names, so the
    # language table's display name is not always the string to match on.
    UNICODE_SCRIPT_NAME = {"odia": "ORIYA"}

    for code, text in P._CANARIES.items():
        script = LANGUAGES[code].script.lower()
        needle = UNICODE_SCRIPT_NAME.get(script, script.split()[0].upper())
        names = [unicodedata.name(c, "") for c in text if not c.isspace()]
        hits = sum(1 for n in names if needle in n)
        assert hits > len(names) // 2, (
            f"{code} canary is not predominantly {script} "
            f"({hits}/{len(names)} chars matched {needle!r})")


# ---------------------------------------------------------------------------------------
# Orthographic normalisation
# ---------------------------------------------------------------------------------------

def test_normalization_canonicalizes_precomposed_and_zwj():
    """Skipped if indic-nlp-library is absent; asserted precisely when it is present.

    Written with explicit escapes because precomposed and decomposed Indic letters are
    visually IDENTICAL in source — a raw literal here would assert nothing a reader can
    check.
    """
    if P._normalizer("hi") is None:
        print("    (skipped: indic-nlp-library not installed)")
        return

    QA_PRE, QA_DEC = "\u0958", "\u0915\u093c"        # DEVANAGARI QA
    RRA_PRE, RRA_DEC = "\u09dc", "\u09a1\u09bc"      # BENGALI RRA
    ZWJ_IN = "\u0915\u094d\u200d\u0937"             # ka + virama + ZWJ + ssa
    ZWJ_OUT = "\u0915\u094d\u0937"

    # The pipeline normalises AND recomposes, so either encoding ends up precomposed —
    # which is the encoding espeak's rules are written against.
    assert P.normalize_indic(QA_PRE, "hi") == QA_PRE
    assert P.normalize_indic(QA_DEC, "hi") == QA_PRE, "decomposed input must be recomposed"
    assert P.normalize_indic(RRA_DEC, "bn") == RRA_PRE

    # ZWJ removal is IndicNLP's contribution and must survive recomposition.
    assert P.normalize_indic(ZWJ_IN, "hi") == ZWJ_OUT

    # No-op on already-canonical text, in every language.
    for code, canary in P._CANARIES.items():
        assert P.normalize_indic(canary, code) == canary, f"{code} canary is not canonical"

def test_recomposition_restores_precomposed_nukta_letters():
    """IndicNLP decomposes nukta letters; espeak's rules want them precomposed.

    Measured consequence of getting this wrong: normalisation rewrites 50.8% of Assamese
    rows (almost all YYA U+09DF -> U+09AF U+09BC) and Assamese phonemes-per-character
    jumps 1.147 -> 1.604, while Bengali — same script, same substitution on 36.3% of its
    rows — stays flat. That divergence is espeak's `as` voice failing to parse the
    decomposed sequence, not a better measurement.
    """
    assert P.recompose_indic("\u09af\u09bc") == "\u09df", "Bengali YA+NUKTA -> YYA"
    assert P.recompose_indic("\u0915\u093c") == "\u0958", "Devanagari KA+NUKTA -> QA"
    assert P.recompose_indic("\u0b21\u0b3c") == "\u0b5c", "Oriya DDA+NUKTA -> RRA"
    # Idempotent on already-precomposed input.
    assert P.recompose_indic("\u09df") == "\u09df"
    # Text with no nukta sequence passes through untouched.
    plain = "\u09a6\u09c1\u09a7\u09c7"              # BENGALI 'dudhe'
    assert P.recompose_indic(plain) == plain

def test_recomposition_map_is_derived_not_hardcoded():
    """Built from unicodedata, so it cannot drift from the standard."""
    m = P._recomposition_map()
    assert len(m) > 30, f"suspiciously small map: {len(m)}"
    for seq, ch in m.items():
        assert len(seq) >= 2 and len(ch) == 1


def test_normalization_then_recomposition_keeps_both_benefits():
    """The combination must strip ZWJ *and* leave nukta letters precomposed — getting one
    at the cost of the other is what makes this subtle."""
    if P._normalizer("hi") is None:
        print("    (skipped: indic-nlp-library not installed)")
        return
    assert P.normalize_indic("क्‍ष", "hi") == "क्ष", "ZWJ must still be stripped"
    assert P.normalize_indic("য়", "as") == "য়", "nukta must end up precomposed"


def test_normalization_is_absent_safely():
    """With the library missing, counting continues on un-normalised text and says so
    once — degraded, but visibly, not silently."""
    real = P._normalizer
    P._normalizer = lambda code: None
    P._NORMALIZER_WARNED.clear()
    try:
        assert P.normalize_indic("दूध", "hi") == "दूध"
    finally:
        P._normalizer = real
        P._NORMALIZER_WARNED.clear()


# ---------------------------------------------------------------------------------------
# Inventory validation — a count cannot tell you the symbols are phonemes; this can
# ---------------------------------------------------------------------------------------

def test_inventory_validation_catches_leaked_source_graphemes():
    """Regression case taken verbatim from epitran's real Marathi output.

    epitran renders `कॅल्शियम` as `kəॅlɕijmə`, passing U+0945 DEVANAGARI VOWEL SIGN
    CANDRA E through untranslated because its map has no entry for it. The phoneme *count*
    still looks entirely reasonable — which is the point. Only the inventory exposes it.
    """
    from collections import Counter
    inv = Counter({"k": 40, "ə": 35, "l": 22, "ॅ": 3})   # U+0945 is Devanagari
    problems = P.validate_inventory(inv, "mr")
    assert len(problems) == 1, problems
    assert "U+0945" in problems[0]
    assert "CANDRA E" in problems[0]


def test_inventory_validation_passes_clean_ipa():
    from collections import Counter
    assert P.validate_inventory(Counter({"d": 9, "uː": 4, "dʰ": 3, "ə": 7}), "hi") == []


def test_inventory_validation_uses_the_right_script_per_language():
    """A Bengali grapheme in Bengali output must be flagged; the same grapheme is not a
    Devanagari leak, and flagging it for Hindi would be a false positive."""
    from collections import Counter
    bengali_char = "়"      # BENGALI SIGN NUKTA
    assert P.validate_inventory(Counter({bengali_char: 2}), "bn") != []
    assert P.validate_inventory(Counter({bengali_char: 2}), "hi") == []


def test_odia_inventory_uses_the_unicode_script_alias():
    """Odia characters are named ORIYA in Unicode. Without the alias this check would
    silently never fire for `or` — the language with the worst measured CV."""
    from collections import Counter
    odia_char = "଼"          # ORIYA SIGN NUKTA
    assert P.validate_inventory(Counter({odia_char: 5}), "or") != []


# ---------------------------------------------------------------------------------------
# Statistics fixed in the eval harness
# ---------------------------------------------------------------------------------------

def test_summarize_uses_sample_stdev_and_a_true_median():
    from evaluation.phoneme_adherence_eval import summarize
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert abs(s["median"] - 2.5) < 1e-9, "even-length median must be the midpoint, not the upper value"
    assert abs(s["std"] - 1.2909944487358056) < 1e-9, "must be the sample (n-1) stdev"
    assert summarize([])["n"] == 0
    assert summarize([5.0])["std"] == 0.0


def test_stopping_rule_keeps_spending_when_slope_still_climbs():
    """The decision the doctrine exists to protect: CE flat is not permission to stop."""
    from evaluation.phoneme_adherence_eval import stopping_verdict
    traj = [
        {"checkpoint": "a", "step": 3200, "ce_mean": 0.5056, "length_slope_probe": 0.656},
        {"checkpoint": "b", "step": 3801, "ce_mean": 0.5075, "length_slope_probe": 0.687},
    ]
    v = stopping_verdict(traj)
    assert v["verdict"] == "CONTINUE", v


def test_stopping_rule_stops_on_a_genuine_plateau():
    from evaluation.phoneme_adherence_eval import stopping_verdict
    traj = [
        {"checkpoint": "a", "step": 3200, "ce_mean": 0.5056, "length_slope_probe": 0.684},
        {"checkpoint": "b", "step": 3801, "ce_mean": 0.5075, "length_slope_probe": 0.687},
    ]
    assert stopping_verdict(traj)["verdict"] == "STOP"


def test_stopping_rule_prefers_the_probe_over_the_population_slope():
    """The population slope is confounded by sentence length. If both are present the rule
    must read the probe, or it can be talked into stopping by a fluency measurement."""
    from evaluation.phoneme_adherence_eval import stopping_verdict
    traj = [
        {"step": 3200, "ce_mean": 0.505, "length_slope_probe": 0.60, "length_slope_population": 0.70},
        {"step": 3801, "ce_mean": 0.506, "length_slope_probe": 0.75, "length_slope_population": 0.70},
    ]
    assert stopping_verdict(traj)["verdict"] == "CONTINUE"


# ---------------------------------------------------------------------------------------
# Ruler audit maths
# ---------------------------------------------------------------------------------------

def test_fit_through_origin_recovers_a_known_slope():
    from tools.ruler_audit import _fit_through_origin
    xs = [10.0, 20.0, 30.0, 40.0]
    ys = [8.0, 16.0, 24.0, 32.0]        # exactly 0.8x
    k, r2 = _fit_through_origin(xs, ys)
    assert abs(k - 0.8) < 1e-9
    assert r2 > 0.999


def test_corpus_ruler_gate_rejects_a_mixed_corpus():
    from training.train_translation_llm import assert_corpus_ruler
    mixed = [{"ruler": "phonemes:espeak-ng-1.51"}, {"ruler": "MISSING"}]
    raised = False
    try:
        assert_corpus_ruler(mixed, "mixed.jsonl", strict=True)
    except RuntimeError as e:
        raised = True
        assert "relabel_dataset" in str(e), "the error must name the repair tool"
    assert raised, "a mixed-ruler corpus must be refused, not warned about"

    # A uniformly phoneme-ruled corpus is fine.
    assert_corpus_ruler([{"ruler": "phonemes:espeak-ng-1.51"}] * 3, "ok.jsonl", strict=True)


def test_unlabelled_corpus_is_refused():
    from training.train_translation_llm import assert_corpus_ruler
    legacy = [{"n_phonemes": 40}, {"n_phonemes": 51}]
    raised = False
    try:
        assert_corpus_ruler(legacy, "legacy.jsonl", strict=True)
    except RuntimeError:
        raised = True
    assert raised, "a corpus with no ruler tag predates the repair and must be refused"


def test_pooled_probe_slope_is_not_zero_referenced():
    """The scar: a model that emits its natural translation every time, ignoring the
    budget completely, scores 0.60-0.83 on the pooled probe — not 0. Reading that number
    against zero overstates every model, and it is why five languages looked like they
    were 'partially following' when they were at or below their own floor."""
    from evaluation.phoneme_adherence_eval import linfit_slope

    naturals = [20, 28, 35, 41, 47, 52, 58, 63]
    scales = [0.6, 0.8, 1.0, 1.2, 1.4]
    req = [float(max(1, round(n * s))) for n in naturals for s in scales]
    flat = [float(n) for n in naturals for _ in scales]          # ignores the budget

    pooled, _ = linfit_slope(req, flat)
    assert pooled > 0.5, ("if this ever drops to ~0 the pooled estimator has been fixed "
                          "and this test should become an equality check")

    # The within-sentence estimator on the same flat model. This is the one to read.
    norm, _ = linfit_slope([s for _ in naturals for s in scales],
                           [1.0 for _ in naturals for _ in scales])
    assert abs(norm) < 1e-9, f"normalised slope must be 0 for a flat model, got {norm}"


def test_response_diagnosis_separates_the_four_failure_shapes():
    """A slope alone cannot distinguish four defects that need four different fixes.
    These are the canonical shapes; if the classifier stops separating them, the
    pre-registered branch in PHASE02_RUNBOOK.md is being chosen by accident."""
    import random
    from evaluation.response_diagnosis import diagnose

    random.seed(7)
    scales = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0]
    shapes = {
        "obeys":    lambda s: s,
        "flat":     lambda s: 1.0,
        "noisy":    lambda s: 1.0 + random.gauss(0, 0.35),
        "saturate": lambda s: min(max(s, 0.82), 1.25),
        "asym":     lambda s: s if s >= 1.0 else 1.0,
    }
    expect = {"obeys": "OBEYS", "flat": "FLAT", "noisy": "NOISY",
              "saturate": "SATURATING", "asym": "ASYMMETRIC"}

    pts = []
    for name, fn in shapes.items():
        for si in range(25):
            nat = random.randint(25, 60)
            for s in scales:
                ratio = fn(s) + (0 if name == "noisy" else random.gauss(0, 0.04))
                pts.append({"language": name, "sentence_idx": si, "scale": s,
                            "natural_n": nat, "requested_n": round(nat * s),
                            "produced_n": max(1, round(nat * ratio))})

    got = {lg: v["diagnosis"] for lg, v in diagnose(pts)["per_language"].items()}
    assert got == expect, f"expected {expect}, got {got}"


def test_preflight_rejects_a_corpus_with_no_elastic_rows():
    """The defect that cost Phase 02 two weeks: every row's budget is its own completion's
    length, so ignoring the budget is a correct answer everywhere and nothing teaches the
    task. This must be caught on CPU, before a push."""
    from tools.preflight import check_elasticity, check_direction

    rigid = [{"language": "hi", "english": f"sentence {i}", "n_phonemes": 30 + i}
             for i in range(500)]
    assert check_elasticity(rigid, "t").state == "FAIL"
    assert check_direction(rigid, "t").state == "FAIL"

    # The same corpus with each input seen at three budgets.
    elastic = []
    for i in range(500):
        base = 30 + i % 20
        elastic.append({"language": "hi", "english": f"s{i}", "n_phonemes": base})
        elastic.append({"language": "hi", "english": f"s{i}", "n_phonemes": round(base * 0.7),
                        "augmentation": {"direction": "compress"}})
        elastic.append({"language": "hi", "english": f"s{i}", "n_phonemes": round(base * 1.4),
                        "augmentation": {"direction": "expand"}})
    assert check_elasticity(elastic, "t").state == "PASS"
    assert check_direction(elastic, "t").state == "PASS"


def test_required_wandb_aborts_instead_of_running_blind():
    """Session 02i burned ~12 GPU-hours with no live channel because a Kaggle secrets
    outage was only a warning. Under --wandb_required the same condition must abort."""
    import os
    from evaluation.phoneme_adherence_eval import wandb_init

    saved = os.environ.pop("WANDB_API_KEY", None)
    try:
        if any((Path.home() / n).exists() for n in (".netrc", "_netrc")):
            return  # a real credential is present; this machine cannot exercise the path
        raised = False
        try:
            wandb_init({}, "p", "e", "r", required=True)
        except SystemExit as e:
            raised = True
            assert "wandb_required" in str(e), "the error must say how to opt out"
        assert raised, "a missing credential must be fatal when W&B is required"

        # Without the flag the old behaviour stands: warn and keep the disk artifacts.
        assert wandb_init({}, "p", "e", "r", required=False) is None
    finally:
        if saved is not None:
            os.environ["WANDB_API_KEY"] = saved


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
