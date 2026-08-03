"""
common/phonemes.py
==================
THE canonical grapheme-to-phoneme counter for pipeline_v3. Every module that writes a
phoneme budget into a label, and every module that scores a generation against one, must
import `count_phonemes` from here and from nowhere else.

WHY THIS MODULE EXISTS (read this before changing anything in it)
-----------------------------------------------------------------
The previous implementation lived in `translation/duration_predictor.phonemize_text` and
ended like this::

    except Exception as e:
        logger.debug("Phonemization failed ...; falling back to characters.")
    return [c for c in text if not c.isspace()]

That fallback fired for the **entire** corpus generation run, because espeak-ng (a system
binary, installed separately from the `phonemizer` pip package) was not present in that
Kaggle session. It logged at DEBUG, so nothing surfaced. The result: every `n_phonemes`
label in the base training corpus is a **non-space character count**, verified at 100.0%
over 1,289 locally-held rows with a chars/n_phonemes ratio of exactly 1.000 (min = max).

It got worse. A later session (length augmentation, 2026-07-26) *did* have espeak-ng, so
those rows are labelled in **real phonemes** — 8.4% coincidental agreement with character
counts, ratio spread 0.198-1.500. The corpus therefore carries two mutually incompatible
rulers under one prompt token, `[Target Phonemes: N]`, teaching the model two
contradictory tasks. A model asked to fit N of one unit and N of another cannot reach
slope 1.0 on either; it can only split the difference.

So this module enforces three things the old one did not:

1. **No silent fallback, ever.** A phonemization failure raises `G2PUnavailable` or
   `PhonemizationError`. Label-writing code must never degrade quietly, because a
   degraded label is indistinguishable from a good one downstream. Inference code that
   legitimately needs to survive a bad string catches the exception *explicitly* and
   records the degradation (see "degrade, don't crash" — but degrade *visibly*).

2. **A preflight that proves the output is phonemes, not passthrough.** Checking that
   espeak-ng is importable is not sufficient — the failure mode we actually hit produces
   plausible-looking output. `assert_g2p_available()` phonemizes a canary string in each
   language and asserts the returned symbols are not simply the input's own characters.
   That is the check that would have caught this on day one.

3. **A ruler identifier stamped into every artifact.** `ruler_id()` returns a string like
   ``phonemes:espeak-ng-1.51``. Dataset rows, eval reports, and run manifests all carry
   it, so "which ruler produced this number" is a grep, not a forensic exercise.

WHAT COUNTS AS ONE PHONEME
---------------------------
espeak-ng's IPA output carries symbols that are not sounds. We normalise before counting:

- **Stress marks** (``ˈ`` primary, ``ˌ`` secondary) are suprasegmental — they mark which
  syllable is emphasised, not an additional sound. Stripped.
- **Length marks** (``ː``) modify the preceding vowel's duration and stay attached to it,
  so ``aː`` is one (long) phoneme, not two. Kept attached — which is correct for our
  purpose, since duration is exactly what we are proxying.
- **Language-switch tags** (``(en)``, ``(hi)``) are emitted when espeak detects a foreign
  word — typically English brand names inside Indic text. They are markup. Stripped.
- **Tie bars** (``͡``) join affricates into one segment. Kept attached.

These choices are asymmetric-safe: they can only ever be wrong by a constant per language,
and non-negotiable #3 (same function for labels and scores) means a constant offset
cancels. What must never happen is two *different* normalisations in the same project.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Iterable, Optional, Sequence

from common.languages import LANGUAGES, get_language

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------
# Ruler identifiers. These are written into datasets and reports; treat them as a stable
# public vocabulary, not as free text.
# ---------------------------------------------------------------------------------------

RULER_PHONEMES = "phonemes:espeak-ng"
RULER_CHARS = "chars:non-space"
RULER_UNKNOWN = "unknown"


class G2PUnavailable(RuntimeError):
    """espeak-ng and/or the `phonemizer` package is missing or non-functional.

    Raised by the preflight and by `phonemize()`. This is deliberately fatal: every
    caller in the label-writing path would otherwise produce a corpus that looks correct
    and is measured in the wrong unit.
    """


class PhonemizationError(RuntimeError):
    """espeak-ng is present and working, but this specific string could not be converted."""


_INSTALL_HINT = (
    "espeak-ng is a SYSTEM package and is NOT installed by `pip install phonemizer`.\n"
    "  Kaggle / Debian / Ubuntu:  apt-get -qq update && apt-get -qq install -y espeak-ng\n"
    "  macOS:                     brew install espeak-ng\n"
    "  Windows:                   winget install --id eSpeak-NG.eSpeak-NG   (then set\n"
    "                             PHONEMIZER_ESPEAK_LIBRARY to the installed libespeak-ng.dll)\n"
    "  Then:                      pip install phonemizer\n"
    "Verify with: python -c \"from common.phonemes import assert_g2p_available;"
    " assert_g2p_available()\""
)

# Suprasegmental / markup symbols that are not themselves sounds.
_STRESS_MARKS = "ˈˌ"          # ˈ ˌ
_LANG_SWITCH_RE = re.compile(r"\([a-z]{2,3}\)")   # (en), (hi), ...
_UNDERTIE = "‿"

# Unicode names several scripts by an older label than the one the language table uses.
_UNICODE_SCRIPT_NAME = {"odia": "ORIYA"}


def _script_token(language_iso_code: str) -> str:
    """The token that appears in unicodedata.name() for this language's script."""
    script = get_language(language_iso_code).script.lower()
    return _UNICODE_SCRIPT_NAME.get(script, script.split()[0].upper())


# ---------------------------------------------------------------------------------------
# Orthographic normalisation (AI4Bharat IndicNLP)
# ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _normalizer(language_iso_code: str):
    """Per-language IndicNLP normaliser, or None if the library is absent."""
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    except ImportError:
        return None
    return IndicNormalizerFactory().get_normalizer(language_iso_code)


# Set once, the first time normalisation is skipped, so the absence is stated rather than
# inferred from a slightly-off number three stages downstream.
_NORMALIZER_WARNED: set = set()


@functools.lru_cache(maxsize=1)
def _recomposition_map() -> dict:
    """decomposed-sequence -> precomposed-character, for Indic nukta letters.

    Built from `unicodedata` rather than hand-listed, so it cannot drift from the standard
    and needs no maintenance when a script is added.

    WHY IT IS NEEDED. IndicNLP canonicalises *toward the decomposed form*: `য়` U+09DF
    becomes U+09AF + U+09BC (YA + NUKTA). espeak-ng's rule files are evidently written
    against the *precomposed* letters. Measured on this corpus: normalisation rewrites
    50.8% of Assamese rows, almost all of them this substitution, and Assamese
    phonemes-per-character then jumps 1.147 -> 1.604 while Bengali — same script, same
    substitution on 36.3% of its rows — barely moves (1.032 -> 1.020). Assamese and Bengali
    are phonologically close and orthographically shared; a 57% divergence between them is
    not a better measurement, it is espeak's `as` voice failing to parse a sequence its
    `bn` voice handles.

    These characters cannot be recomposed by `unicodedata.normalize("NFC", ...)`, because
    Indic nukta letters are Unicode *composition exclusions* — NFC deliberately leaves them
    decomposed. Hence an explicit map.

    So: take IndicNLP's genuine cleanups (ZWJ/ZWNJ removal, punctuation canonicalisation,
    Malayalam chillu handling) and then put the nukta letters back into the encoding the
    G2P actually recognises.
    """
    import unicodedata
    mapping = {}
    # Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
    for cp in range(0x0900, 0x0E00):
        ch = chr(cp)
        decomp = unicodedata.decomposition(ch)
        if not decomp or decomp.startswith("<"):   # skip compatibility decompositions
            continue
        try:
            seq = "".join(chr(int(p, 16)) for p in decomp.split())
        except ValueError:
            continue
        mapping[seq] = ch
    return mapping


def recompose_indic(text: str) -> str:
    """Restores precomposed Indic nukta letters after IndicNLP normalisation."""
    for seq, ch in _recomposition_map().items():
        if seq in text:
            text = text.replace(seq, ch)
    return text


def normalize_indic(text: str, language_iso_code: str) -> str:
    """Canonicalises Indic text before G2P.

    Indic scripts encode the same grapheme several ways, and espeak-ng only has rules for
    one of them. Verified on this machine:

        क़  U+0958 (precomposed)      -> U+0915 U+093C  (KA + NUKTA)
        ড়  U+09DC (precomposed)      -> U+09A1 U+09BC  (DDA + NUKTA)
        क्‍ष  with U+200D ZWJ            -> ZWJ removed

    Fed the precomposed or ZWJ-bearing form, espeak either skips the codepoint or emits
    something arbitrary — silently, and only for the subset of rows that happen to use that
    encoding. That is a per-row error concentrated in exactly the words most likely to be
    loanwords and proper nouns, which is worse than a uniform bias because it cannot be
    calibrated away.

    Normalising is a strict improvement and costs nothing: on already-canonical text it is
    a no-op (verified across all 11 languages). It runs inside `phonemize`, so labels and
    scores are normalised identically — the same-function rule (non-negotiable #3) extends
    to preprocessing, not just to the G2P call.
    """
    n = _normalizer(language_iso_code)
    if n is None:
        if "warned" not in _NORMALIZER_WARNED:
            _NORMALIZER_WARNED.add("warned")
            logger.warning(
                "indic-nlp-library is not installed — phoneme counts will be taken on "
                "UN-normalised text. Precomposed nukta forms and ZWJ sequences will be "
                "mis-phonemized for a minority of rows. Install with: "
                "pip install indic-nlp-library"
            )
        return text
    return recompose_indic(n.normalize(text))


# ---------------------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _backend_version() -> str:
    """Returns the espeak-ng version string, or raises G2PUnavailable.

    Cached because it shells into the espeak library, and callers ask for it once per
    artifact written.
    """
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError as e:
        raise G2PUnavailable(f"`phonemizer` is not importable ({e}).\n{_INSTALL_HINT}") from e

    try:
        version = EspeakWrapper().version
    except Exception as e:  # noqa: BLE001 - any failure here means the shared library is unusable
        raise G2PUnavailable(
            f"`phonemizer` imported but the espeak-ng shared library is unusable ({e}).\n"
            f"{_INSTALL_HINT}"
        ) from e

    # Normalise: some phonemizer builds return a tuple like (1, 50) rather than "1.50".
    # `str()` on that yields "(1, 50)" — which embeds a COMMA and a space into a string
    # that gets stamped into every dataset row and every report header, and would corrupt
    # any CSV column it ever lands in. A provenance field that can break its own container
    # is not provenance.
    if isinstance(version, (tuple, list)):
        version = ".".join(str(p) for p in version)
    return re.sub(r"[\s,]+", "", str(version))


def ruler_id() -> str:
    """The identifier stamped into every dataset row and eval report this module touches.

    Raises G2PUnavailable rather than returning a placeholder — a report that cannot name
    its ruler must not be written at all.
    """
    return f"{RULER_PHONEMES}-{_backend_version()}"


# ---------------------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------------------

def _normalise_tokens(raw: str) -> list[str]:
    """Turns espeak's separated output into a list of countable phoneme symbols."""
    raw = _LANG_SWITCH_RE.sub(" ", raw)
    raw = raw.replace("|", " ").replace(_UNDERTIE, " ")
    tokens = []
    for tok in raw.split():
        tok = tok.strip(_STRESS_MARKS)
        # A token that was *only* stress marks collapses to empty and is not a sound.
        if tok:
            tokens.append(tok)
    return tokens


def phonemize(text: str, language_iso_code: str) -> list[str]:
    """Converts `text` to a list of IPA phoneme symbols. Never falls back.

    Raises:
        G2PUnavailable: espeak-ng / phonemizer is missing or broken.
        PhonemizationError: the backend works but produced nothing for this input.
    """
    text = (text or "").strip()
    if not text:
        return []

    _backend_version()  # raises G2PUnavailable with the install hint if the backend is dead

    from phonemizer import phonemize as _ph
    from phonemizer.separator import Separator

    lang = get_language(language_iso_code)
    text = normalize_indic(text, language_iso_code)
    try:
        out = _ph(
            text,
            language=lang.espeak_code,
            backend="espeak",
            separator=Separator(phone=" ", word=" | "),
            strip=True,
            njobs=1,
        )
    except Exception as e:  # noqa: BLE001
        raise PhonemizationError(
            f"espeak-ng failed on {language_iso_code} input {text[:60]!r}: {e}"
        ) from e

    tokens = _normalise_tokens(out)
    if not tokens:
        raise PhonemizationError(
            f"espeak-ng returned no phonemes for {language_iso_code} input {text[:60]!r}. "
            f"Raw backend output was {out[:80]!r}."
        )
    return tokens


def phonemize_many(texts: Sequence[str], language_iso_code: str) -> list[list[str]]:
    """Batched `phonemize`, one espeak call for the whole list.

    Roughly an order of magnitude faster than looping — which matters, because relabelling
    the 53,350-row corpus one string at a time is a multi-hour job and a batched pass is
    minutes. Empty inputs map to empty lists; a backend failure on the batch raises, so a
    partially-phonemized corpus is never written.
    """
    if not texts:
        return []

    _backend_version()

    from phonemizer import phonemize as _ph
    from phonemizer.separator import Separator

    lang = get_language(language_iso_code)
    cleaned = [normalize_indic((t or "").strip(), language_iso_code) for t in texts]
    try:
        out = _ph(
            cleaned,
            language=lang.espeak_code,
            backend="espeak",
            separator=Separator(phone=" ", word=" | "),
            strip=True,
            njobs=1,
        )
    except Exception as e:  # noqa: BLE001
        raise PhonemizationError(
            f"espeak-ng failed on a batch of {len(texts)} {language_iso_code} strings: {e}"
        ) from e

    if isinstance(out, str):  # phonemizer collapses a 1-element list to a bare string
        out = [out]
    return [_normalise_tokens(o) for o in out]


@functools.lru_cache(maxsize=200_000)
def count_phonemes(text: str, language_iso_code: str) -> int:
    """The one function that defines "how many phonemes is this". Labels and scores both
    call it, which is what makes their numbers comparable (non-negotiable #3)."""
    return len(phonemize(text, language_iso_code))


def phoneme_inventory(texts: Iterable[str], language_iso_code: str) -> "Counter":
    """The distribution of phoneme symbols a language's G2P actually produces.

    This is a *validation* instrument, not a pipeline metric — the pipeline only ever needs
    the per-sentence count. But a count cannot tell you whether the symbols being counted
    are phonemes at all, and an inventory can, in one glance.

    Worked example of what it catches: epitran's Marathi renders `कॅल्शियम` as `kəॅlɕijmə`,
    leaking U+0945 DEVANAGARI VOWEL SIGN CANDRA E — a *source script* character — straight
    into its own IPA output, because that codepoint has no entry in its map. The count still
    comes out looking reasonable. The inventory makes it obvious.
    """
    from collections import Counter
    inv = Counter()
    for toks in phonemize_many(list(texts), language_iso_code):
        inv.update(toks)
    return inv


def validate_inventory(inventory: "Counter", language_iso_code: str) -> list[str]:
    """Returns a list of problems found in a phoneme inventory; empty means clean.

    The check that matters: a symbol containing a character from the language's OWN script
    is not a phoneme — it is an unmapped source character that the G2P passed through
    untranslated. Its presence proves the converter has a hole, and tells you exactly which
    grapheme fell in.
    """
    import unicodedata

    token = _script_token(language_iso_code)
    problems = []
    total = sum(inventory.values()) or 1
    for sym, n in inventory.most_common():
        leaked = [c for c in sym if token in unicodedata.name(c, "")]
        if leaked:
            names = ", ".join(f"U+{ord(c):04X} {unicodedata.name(c, '?')}" for c in leaked)
            problems.append(
                f"{sym!r} ({n} occurrences, {100 * n / total:.2f}%) contains untranslated "
                f"source-script characters: {names}"
            )
    return problems


def count_chars(text: str) -> int:
    """Non-space character count — the *wrong* ruler, defined here explicitly so audit
    code can name it and detect it rather than reimplementing it three times."""
    return len([c for c in (text or "") if not c.isspace()])


# ---------------------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------------------

# Short canaries in each language's own script. Any real sentence works; these are kept
# tiny so the preflight costs milliseconds.
_CANARIES: dict[str, str] = {
    "hi": "दूध में कैल्शियम है",
    "bn": "দুধে ক্যালসিয়াম আছে",
    "mr": "दुधात कॅल्शियम आहे",
    "gu": "દૂધમાં કેલ્શિયમ છે",
    "pa": "ਦੁੱਧ ਵਿੱਚ ਕੈਲਸ਼ੀਅਮ ਹੈ",
    "ta": "பாலில் கால்சியம் உள்ளது",
    "te": "పాలలో కాల్షియం ఉంది",
    "kn": "ಹಾಲಿನಲ್ಲಿ ಕ್ಯಾಲ್ಸಿಯಂ ಇದೆ",
    "ml": "പാലിൽ കാൽസ്യം ഉണ്ട്",
    "or": "ଦୁଧରେ କ୍ୟାଲସିୟମ ଅଛି",
    "as": "গাখীৰত কেলচিয়াম আছে",
}


def assert_g2p_available(languages: Optional[Iterable[str]] = None, verbose: bool = True) -> dict:
    """Preflight. Call this at the top of every notebook that writes labels or scores them.

    Checks three things, in increasing order of strictness:

    1. `phonemizer` imports and the espeak-ng shared library loads.
    2. Every requested language produces non-empty output.
    3. **The output is actually phonemes, not the input's own characters.** This is the
       check that matters. A backend that silently passes text through, or a caller that
       silently substitutes a character split, both produce plausible non-empty output —
       and that is precisely the failure that mislabelled this project's corpus. We assert
       that at least one returned symbol does not appear in the source string, which is
       guaranteed true for any Indic script rendered to IPA and false for passthrough.

    Returns a manifest dict suitable for writing next to any artifact produced afterwards.

    Raises:
        G2PUnavailable: with the install hint, if any check fails.
    """
    codes = list(languages) if languages is not None else list(LANGUAGES.keys())
    version = _backend_version()   # raises with install hint

    report: dict[str, dict] = {}
    failures: list[str] = []

    for code in codes:
        canary = _CANARIES.get(code)
        if canary is None:
            failures.append(f"{code}: no canary string defined in common/phonemes.py")
            continue
        try:
            tokens = phonemize(canary, code)
        except (G2PUnavailable, PhonemizationError) as e:
            failures.append(f"{code}: {e}")
            continue

        source_chars = set(canary)
        novel = [t for t in tokens if not set(t) <= source_chars]
        looks_like_passthrough = not novel

        # Partial leaks: individual unmapped graphemes riding through into the IPA. The
        # passthrough test above only catches total failure; this catches the holes.
        from collections import Counter
        leaks = validate_inventory(Counter(tokens), code)

        report[code] = {
            "voice": get_language(code).espeak_code,
            "n_phonemes": len(tokens),
            "n_chars": count_chars(canary),
            "sample": " ".join(tokens[:12]),
            "passthrough": looks_like_passthrough,
            "normalized": _normalizer(code) is not None,
            "leaks": leaks,
        }
        if looks_like_passthrough:
            failures.append(
                f"{code}: espeak returned symbols drawn entirely from the input's own "
                f"characters ({' '.join(tokens[:10])}) — this is a passthrough or a "
                f"character-split fallback, NOT phonemization."
            )
        for lk in leaks:
            failures.append(f"{code}: untranslated grapheme in G2P output — {lk}")

    if failures:
        raise G2PUnavailable(
            "G2P preflight FAILED — do not generate labels or scores in this session.\n"
            + "\n".join(f"  - {f}" for f in failures)
            + "\n\n" + _INSTALL_HINT
        )

    manifest = {
        "ruler": ruler_id(),
        "espeak_version": version,
        "languages": report,
    }
    if verbose:
        logger.info("G2P preflight PASSED — ruler=%s", manifest["ruler"])
        for code, r in report.items():
            logger.info(
                "  %-3s voice=%-3s %3d phonemes / %3d chars (ratio %.3f)  %s",
                code, r["voice"], r["n_phonemes"], r["n_chars"],
                r["n_phonemes"] / max(r["n_chars"], 1), r["sample"],
            )
    return manifest


if __name__ == "__main__":  # `python -m common.phonemes` as a standalone preflight
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(json.dumps(assert_g2p_available(), ensure_ascii=False, indent=2))
    except G2PUnavailable as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
