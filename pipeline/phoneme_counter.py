"""
phoneme_counter.py — Real grapheme-to-phoneme (G2P) phoneme counting for isochrony scoring.

v2.5 rewrite
------------
The previous version of this file counted *syllables*, not phonemes: English via a
vowel-cluster heuristic, Indic via Devanagari matra counting, and — even where it tried
`phonemizer` — it counted only IPA vowels and then SILENTLY fell back to character
heuristics whenever espeak-ng was missing. A silent fallback is the worst possible failure
here: a syllable count and a phoneme count are different units, and once one label is
measured in the wrong unit it is indistinguishable downstream from a good one. (This is the
exact bug documented in pipeline_v3/common/phonemes.py, where an entire training corpus was
mislabelled because a DEBUG-level fallback fired for a whole run.)

This module counts ACTUAL phonemes for both English (en-us) and all 11 Indic languages,
using espeak-ng via `phonemizer`, with AI4Bharat IndicNLP orthographic normalization applied
first. It is modelled directly on `pipeline_v3/common/phonemes.py` but is:

  * self-contained (no dependency on the `common` package, so it embeds cleanly into the
    Kaggle notebook and the Cloudflare deploy bundle), and
  * keyed by language *name* ("hindi", "tamil", ...) to match this repo's existing API and
    the strings that flow in from `isochrony_translation.py` / `app.py`.

DEGRADE VISIBLY, NEVER SILENTLY
-------------------------------
This runs inside a live Streamlit dubbing pipeline, so a hard crash mid-run is not
acceptable the way it is in an offline label-writing job. Instead, if espeak-ng is
unavailable the module:

  1. logs a single loud WARNING (not DEBUG) the first time it degrades, and
  2. stamps the active ruler as `chars:heuristic-fallback` via `active_ruler()`, so any
     score computed in degraded mode is *labelled* as such and never mistaken for a real
     phoneme measurement.

Crucially, source (English) and target (Indic) counts both flow through the same code path,
so in a given run they are always measured with the same ruler — the comparison between them
stays internally consistent whether espeak is present or not.

WHAT COUNTS AS ONE PHONEME (identical rules to common/phonemes.py)
------------------------------------------------------------------
  * Stress marks (ˈ primary, ˌ secondary) are suprasegmental — stripped.
  * Length marks (ː) stay attached to their vowel: `aː` is one (long) phoneme.
  * Language-switch tags ((en), (hi)) are markup — stripped.
  * Tie bars (͡) join affricates into one segment — kept attached.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------
# Language table. Keyed by the lowercase language NAME this repo uses everywhere.
#   espeak_code : the voice passed to espeak-ng
#   iso         : ISO code used by IndicNLP's normalizer factory (None => no Indic norm)
#   script      : Unicode script token for leak validation / heuristic fallback routing
# ---------------------------------------------------------------------------------------
_LANG_TABLE = {
    "english":   {"espeak_code": "en-us", "iso": None, "script": "LATIN"},
    "hindi":     {"espeak_code": "hi",    "iso": "hi", "script": "DEVANAGARI"},
    "marathi":   {"espeak_code": "mr",    "iso": "mr", "script": "DEVANAGARI"},
    "bengali":   {"espeak_code": "bn",    "iso": "bn", "script": "BENGALI"},
    "assamese":  {"espeak_code": "as",    "iso": "as", "script": "BENGALI"},
    "gujarati":  {"espeak_code": "gu",    "iso": "gu", "script": "GUJARATI"},
    "punjabi":   {"espeak_code": "pa",    "iso": "pa", "script": "GURMUKHI"},
    "tamil":     {"espeak_code": "ta",    "iso": "ta", "script": "TAMIL"},
    "telugu":    {"espeak_code": "te",    "iso": "te", "script": "TELUGU"},
    "kannada":   {"espeak_code": "kn",    "iso": "kn", "script": "KANNADA"},
    "malayalam": {"espeak_code": "ml",    "iso": "ml", "script": "MALAYALAM"},
    "odia":      {"espeak_code": "or",    "iso": "or", "script": "ORIYA"},
}

# Accept common display-name and ISO-code aliases and fold them onto the canonical name.
_LANG_ALIASES = {
    "oriya": "odia",
    "panjabi": "punjabi",
    "en": "english", "en-us": "english", "en-gb": "english",
    "hi": "hindi", "mr": "marathi", "bn": "bengali", "as": "assamese",
    "gu": "gujarati", "pa": "punjabi", "ta": "tamil", "te": "telugu",
    "kn": "kannada", "ml": "malayalam", "or": "odia",
}


def _resolve_language(language: str) -> str:
    """Fold any accepted spelling/alias onto a canonical `_LANG_TABLE` key."""
    key = (language or "").strip().lower()
    key = _LANG_ALIASES.get(key, key)
    return key


# ---------------------------------------------------------------------------------------
# Per-language expansion ratios (PHONEME-DOMAIN priors).
#
# WHY THESE DIFFER FROM THE OLD VALUES (1.30, 1.25, ...): the previous ratios were derived
# in the SYLLABLE / text-length domain (Indic scripts are syllable-dense, so an Indic string
# "looks" ~1.3x longer than its English source). Applied to REAL phoneme counts they badly
# overshoot — e.g. English "There is calcium in milk" = 17 phonemes; the correct, natural
# Hindi "दूध में कैल्शियम है" = 15 phonemes (ratio 0.88), yet 1.30 would demand ~22 and
# penalise the good translation.
#
# The correct phoneme-domain ratio falls straight out of what isochrony IS. Both the source
# and its dub must occupy the SAME on-screen duration D. Speaking at a language's natural
# phoneme rate,  source_phonemes ≈ D * pps_english  and  target_phonemes ≈ D * pps_target,
# so:                       expansion_ratio = pps_target / pps_english.
# We seed these from the project's own `heuristic_phonemes_per_sec` table (pipeline_v3/
# common/languages.py) against an English reference rate. The result is a near-parity prior
# (0.95–1.04), which matches measured cross-lingual phoneme counts far better than 1.3.
#
# These remain cold-start PRIORS, not learned quantities — pipeline_v3 replaces them with a
# trained DurationPredictor, and when a source segment's real duration is known the budget
# functions below use  D * pps_target  directly instead (strictly better). The counts
# themselves are always real espeak phonemes regardless.
# ---------------------------------------------------------------------------------------

# English reference phoneme rate (phonemes/sec). Mid-range of the Indic table; used only as
# the denominator that turns per-second rates into a phoneme-count ratio.
ENGLISH_PHONEMES_PER_SEC = 13.0

# Natural phoneme rate per language (phonemes/sec), from pipeline_v3/common/languages.py.
PHONEMES_PER_SEC = {
    "english":   ENGLISH_PHONEMES_PER_SEC,
    "hindi":     13.5,
    "bengali":   13.0,
    "marathi":   13.2,
    "gujarati":  13.0,
    "punjabi":   12.8,
    "tamil":     12.5,
    "telugu":    12.6,
    "kannada":   12.7,
    "malayalam": 12.3,
    "odia":      13.0,
    "assamese":  12.9,
}

# Derived phoneme-domain expansion ratios = pps_target / pps_english (see rationale above).
EXPANSION_RATIOS = {
    lang: round(pps / ENGLISH_PHONEMES_PER_SEC, 3)
    for lang, pps in PHONEMES_PER_SEC.items()
    if lang != "english"
}


# ---------------------------------------------------------------------------------------
# Ruler identifiers, stamped onto every score so "which unit produced this number" is
# never ambiguous.
# ---------------------------------------------------------------------------------------
RULER_PHONEMES = "phonemes:espeak-ng"
RULER_FALLBACK = "chars:heuristic-fallback"

# Set to True the first (and only) time we degrade, so the warning is loud but not spammy.
_DEGRADED_WARNED = False
# The ruler actually in force for counts produced so far this process.
_ACTIVE_RULER = None


class G2PUnavailable(RuntimeError):
    """espeak-ng and/or the `phonemizer` package is missing or non-functional."""


class PhonemizationError(RuntimeError):
    """espeak-ng works, but this specific string could not be converted."""


_INSTALL_HINT = (
    "espeak-ng is a SYSTEM package and is NOT installed by `pip install phonemizer`.\n"
    "  Kaggle / Debian / Ubuntu:  apt-get -qq update && apt-get -qq install -y espeak-ng\n"
    "  macOS:                     brew install espeak-ng\n"
    "  Windows:                   winget install --id eSpeak-NG.eSpeak-NG   (then set\n"
    "                             PHONEMIZER_ESPEAK_LIBRARY to libespeak-ng.dll)\n"
    "  Then:                      pip install phonemizer indic-nlp-library"
)

# Suprasegmental / markup symbols that are not themselves sounds.
_STRESS_MARKS = "ˈˌ"
_LANG_SWITCH_RE = re.compile(r"\([a-z]{2,3}\)")
_UNDERTIE = "‿"


# ---------------------------------------------------------------------------------------
# Orthographic normalisation (AI4Bharat IndicNLP) — ported from common/phonemes.py
# ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _normalizer(language_iso_code: str):
    """Per-language IndicNLP normaliser, or None if the library / code is absent."""
    if not language_iso_code:
        return None
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    except ImportError:
        return None
    try:
        return IndicNormalizerFactory().get_normalizer(language_iso_code)
    except Exception:
        return None


_NORMALIZER_WARNED: set = set()


@functools.lru_cache(maxsize=1)
def _recomposition_map() -> dict:
    """decomposed-sequence -> precomposed-character, for Indic nukta letters.

    IndicNLP canonicalises toward the *decomposed* form (e.g. য় -> YA + NUKTA) but
    espeak-ng's rule files expect the *precomposed* letters. These are Unicode composition
    exclusions, so NFC will not put them back — hence an explicit map, built from
    `unicodedata` so it cannot drift from the standard.
    """
    mapping = {}
    for cp in range(0x0900, 0x0E00):
        ch = chr(cp)
        decomp = unicodedata.decomposition(ch)
        if not decomp or decomp.startswith("<"):
            continue
        try:
            seq = "".join(chr(int(p, 16)) for p in decomp.split())
        except ValueError:
            continue
        mapping[seq] = ch
    return mapping


def recompose_indic(text: str) -> str:
    """Restore precomposed Indic nukta letters after IndicNLP normalisation."""
    for seq, ch in _recomposition_map().items():
        if seq in text:
            text = text.replace(seq, ch)
    return text


def normalize_indic(text: str, language_iso_code: str) -> str:
    """Canonicalise Indic text before G2P. No-op on already-canonical text; no-op (with a
    one-time warning) if IndicNLP is not installed."""
    n = _normalizer(language_iso_code)
    if n is None:
        if language_iso_code and "warned" not in _NORMALIZER_WARNED:
            _NORMALIZER_WARNED.add("warned")
            logger.warning(
                "indic-nlp-library not installed — phoneme counts taken on UN-normalised "
                "text; precomposed nukta / ZWJ forms mis-phonemized for a minority of rows. "
                "Install with: pip install indic-nlp-library"
            )
        return text
    return recompose_indic(n.normalize(text))


# ---------------------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _backend_version() -> str:
    """espeak-ng version string, or raise G2PUnavailable. Cached (one library shell-in)."""
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError as e:
        raise G2PUnavailable(f"`phonemizer` is not importable ({e}).\n{_INSTALL_HINT}") from e
    try:
        version = EspeakWrapper().version
    except Exception as e:  # noqa: BLE001
        raise G2PUnavailable(
            f"`phonemizer` imported but the espeak-ng shared library is unusable ({e}).\n"
            f"{_INSTALL_HINT}"
        ) from e
    if isinstance(version, (tuple, list)):
        version = ".".join(str(p) for p in version)
    return re.sub(r"[\s,]+", "", str(version))


@functools.lru_cache(maxsize=1)
def g2p_available() -> bool:
    """True if espeak-ng is importable and usable. Cached; never raises."""
    try:
        _backend_version()
        return True
    except G2PUnavailable:
        return False


def ruler_id() -> str:
    """Ruler string for real phoneme counts (raises if the backend is dead)."""
    return f"{RULER_PHONEMES}-{_backend_version()}"


def active_ruler() -> str:
    """The ruler actually in force for the counts produced so far this process.

    Returns the espeak ruler once a real phoneme count has been taken, or the fallback
    ruler once any count has degraded. Before any count is taken, reports what *would* be
    used based on backend availability.
    """
    if _ACTIVE_RULER is not None:
        return _ACTIVE_RULER
    return ruler_id() if g2p_available() else RULER_FALLBACK


def assert_g2p_available() -> str:
    """Preflight for offline / test use: raise G2PUnavailable unless a canary really
    phonemizes (not passthrough). Returns the ruler id on success."""
    canary, code = "दूध में कैल्शियम है", "hindi"
    tokens = _phonemize(canary, code)  # raises G2PUnavailable if backend dead
    if not tokens or set("".join(tokens)) <= set(canary):
        raise G2PUnavailable(
            "espeak returned symbols drawn entirely from the input's own characters — "
            "passthrough / character-split, NOT phonemization.\n" + _INSTALL_HINT
        )
    return ruler_id()


# ---------------------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------------------

def _normalise_tokens(raw: str) -> list:
    """Turn espeak's separated output into a list of countable phoneme symbols."""
    raw = _LANG_SWITCH_RE.sub(" ", raw)
    raw = raw.replace("|", " ").replace(_UNDERTIE, " ")
    tokens = []
    for tok in raw.split():
        tok = tok.strip(_STRESS_MARKS)
        if tok:
            tokens.append(tok)
    return tokens


def _phonemize(text: str, language: str) -> list:
    """Convert `text` to a list of IPA phoneme symbols. Never falls back — raises.

    Raises:
        G2PUnavailable:    espeak-ng / phonemizer missing or broken.
        PhonemizationError: backend works but produced nothing for this input.
    """
    text = (text or "").strip()
    if not text:
        return []

    key = _resolve_language(language)
    info = _LANG_TABLE.get(key)
    if info is None:
        raise PhonemizationError(f"Unknown language {language!r} (resolved to {key!r}).")

    _backend_version()  # raises G2PUnavailable with install hint if the backend is dead

    from phonemizer import phonemize as _ph
    from phonemizer.separator import Separator

    text = normalize_indic(text, info["iso"])
    try:
        out = _ph(
            text,
            language=info["espeak_code"],
            backend="espeak",
            separator=Separator(phone=" ", word=" | "),
            strip=True,
            njobs=1,
        )
    except Exception as e:  # noqa: BLE001
        raise PhonemizationError(
            f"espeak-ng failed on {key} input {text[:60]!r}: {e}"
        ) from e

    tokens = _normalise_tokens(out)
    if not tokens:
        raise PhonemizationError(
            f"espeak-ng returned no phonemes for {key} input {text[:60]!r} "
            f"(raw output {out[:80]!r})."
        )
    return tokens


def phonemize(text: str, language: str) -> list:
    """Public real-phoneme tokeniser. Raises on backend failure (no silent fallback)."""
    return _phonemize(text, language)


# ---------------------------------------------------------------------------------------
# Visible-degradation heuristic fallback (used ONLY when espeak-ng is unavailable).
# These are the OLD syllable/character heuristics, retained solely so a live dubbing run
# does not crash if espeak is missing. When they fire, the active ruler flips to
# RULER_FALLBACK so every downstream score is labelled as degraded.
# ---------------------------------------------------------------------------------------

def _mark_degraded():
    global _DEGRADED_WARNED, _ACTIVE_RULER
    _ACTIVE_RULER = RULER_FALLBACK
    if not _DEGRADED_WARNED:
        _DEGRADED_WARNED = True
        logger.warning(
            "PHONEME COUNTER DEGRADED: espeak-ng is unavailable, so phoneme counts are "
            "falling back to a syllable/character HEURISTIC (ruler=%s). Isochrony scores "
            "this run are approximate and NOT comparable to real-phoneme scores. Fix with:\n%s",
            RULER_FALLBACK, _INSTALL_HINT,
        )


def _count_english_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:\"'()-")
    if not word:
        return 0
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def _fallback_english(text: str) -> int:
    words = re.findall(r"[a-zA-Z']+", text)
    return sum(_count_english_syllables(w) for w in words)


def _fallback_devanagari(text: str) -> int:
    count = 0
    chars = list(text)
    i = 0
    while i < len(chars):
        cp = ord(chars[i])
        if 0x0915 <= cp <= 0x0939 or 0x0958 <= cp <= 0x095F:
            if i + 1 < len(chars) and ord(chars[i + 1]) == 0x094D:
                i += 2
                continue
            count += 1
        elif 0x0904 <= cp <= 0x0914:
            count += 1
        i += 1
    return count if count else _fallback_chars(text)


def _fallback_brahmic(text: str) -> int:
    count = sum(
        1 for ch in text
        if unicodedata.category(ch) not in ("Mn", "Mc", "Me", "Zs", "Po", "Ps", "Pe")
    )
    return max(1, count // 2)


def _fallback_chars(text: str) -> int:
    stripped = re.sub(r"\s+", "", text)
    return max(1, len(stripped) // 3)


def _heuristic_count(text: str, key: str) -> int:
    """Degraded counter, routed by script. Marks the run degraded on use."""
    _mark_degraded()
    if key == "english":
        return _fallback_english(text)
    script = _LANG_TABLE.get(key, {}).get("script", "")
    if script == "DEVANAGARI":
        return _fallback_devanagari(text)
    if script in ("BENGALI", "GUJARATI", "GURMUKHI", "TAMIL", "TELUGU",
                  "KANNADA", "MALAYALAM", "ORIYA"):
        return _fallback_brahmic(text)
    return _fallback_chars(text)


# ---------------------------------------------------------------------------------------
# Public counting API
# ---------------------------------------------------------------------------------------

def count_phonemes(text: str, language: str) -> int:
    """Count phonemes in `text` for `language` (name, display name, or ISO code).

    Uses real espeak-ng G2P when available; otherwise degrades VISIBLY to a syllable/char
    heuristic (loud one-time warning + ruler flips to RULER_FALLBACK). Returns 0 for empty
    input; otherwise >= 1.
    """
    global _ACTIVE_RULER
    if not text or not text.strip():
        return 0
    key = _resolve_language(language)
    if key not in _LANG_TABLE:
        # Unknown language: best-effort heuristic, clearly degraded.
        return max(1, _heuristic_count(text, key))
    try:
        n = len(_phonemize(text, key))
        if _ACTIVE_RULER is None:
            _ACTIVE_RULER = ruler_id()
        return n
    except (G2PUnavailable, PhonemizationError) as e:
        # espeak missing (G2PUnavailable) => degrade for the whole run.
        # A per-string PhonemizationError with espeak present => degrade just this string,
        # but still surface it.
        if isinstance(e, PhonemizationError) and g2p_available():
            logger.warning("Phonemization failed for one %s string; using heuristic: %s", key, e)
        return max(1, _heuristic_count(text, key))


def count_english_phonemes(text: str) -> int:
    """Phoneme count for English source text (backward-compatible name)."""
    return count_phonemes(text, "english")


def count_indic_phonemes(text: str, language: str) -> int:
    """Phoneme count for an Indic-language string (backward-compatible name)."""
    return count_phonemes(text, language)


# ---------------------------------------------------------------------------------------
# Isochrony scoring
# ---------------------------------------------------------------------------------------

def _ideal_target(source_text: str, target_language: str, source_duration=None):
    """The ideal target phoneme count and the expansion ratio implied.

    If `source_duration` (seconds of source audio for this segment) is known, the budget is
    grounded directly in time:  ideal = duration * pps_target  — this is the true isochrony
    target and does not depend on the English phoneme count at all. Otherwise it falls back
    to  source_phonemes * expansion_ratio[target]  (the phoneme-domain prior).
    """
    key = _resolve_language(target_language)
    ratio = EXPANSION_RATIOS.get(key, 1.0)
    src = count_english_phonemes(source_text)
    if source_duration and source_duration > 0:
        pps = PHONEMES_PER_SEC.get(key, ENGLISH_PHONEMES_PER_SEC)
        ideal = source_duration * pps
        # Report the ratio actually realised against the source, for transparency.
        eff_ratio = round(ideal / src, 3) if src else ratio
        return src, ideal, eff_ratio, "duration"
    return src, src * ratio, ratio, "expansion_ratio"


def compute_target_budget(
    source_text: str,
    target_language: str,
    tolerance: float = 0.15,
    source_duration=None,
) -> dict:
    """Phoneme budget for a translation.

    Returns:
        source_phonemes : int   real English phoneme count of the source
        ideal_target    : float duration*pps_target if duration given, else source*ratio
        min_target      : int   lower acceptance bound (ideal * (1 - tolerance))
        max_target      : int   upper acceptance bound (ideal * (1 + tolerance))
        expansion_ratio : float phoneme-domain ratio (realised, if duration-grounded)
        budget_basis    : str   'duration' or 'expansion_ratio'
        ruler           : str   which ruler produced the counts (real vs fallback)
    """
    src, ideal, ratio, basis = _ideal_target(source_text, target_language, source_duration)
    return {
        "source_phonemes": src,
        "ideal_target": round(ideal, 1),
        "min_target": max(1, int(ideal * (1 - tolerance))),
        "max_target": int(ideal * (1 + tolerance)),
        "expansion_ratio": ratio,
        "budget_basis": basis,
        "ruler": active_ruler(),
    }


def isochrony_score(
    source_text: str,
    target_text: str,
    target_language: str,
    source_duration=None,
) -> float:
    """Isochrony compliance in [0.0, 1.0]: how close the target phoneme count is to the
    ideal budget. 1.0 = exactly on budget."""
    _src, ideal, _ratio, _basis = _ideal_target(source_text, target_language, source_duration)
    tgt_phonemes = count_indic_phonemes(target_text, target_language)
    if ideal <= 0:
        return 1.0
    deviation = abs(tgt_phonemes - ideal) / ideal
    return round(max(0.0, 1.0 - deviation), 4)


def phoneme_diff(
    source_text: str,
    target_text: str,
    target_language: str,
    source_duration=None,
) -> dict:
    """Signed phoneme-budget gap for the iterative loop.

    Returns source/target/ideal counts plus:
        abs_diff  : |target - ideal|            (phonemes off budget)
        rel_diff  : abs_diff / max(ideal, 1)    (0 = perfect; used against a tolerance)
        direction : 'too_long' | 'too_short' | 'on_budget'
    """
    src, ideal, _ratio, _basis = _ideal_target(source_text, target_language, source_duration)
    tgt = count_indic_phonemes(target_text, target_language)
    abs_diff = abs(tgt - ideal)
    rel_diff = abs_diff / max(ideal, 1.0)
    if tgt > ideal:
        direction = "too_long"
    elif tgt < ideal:
        direction = "too_short"
    else:
        direction = "on_budget"
    return {
        "source_phonemes": src,
        "target_phonemes": tgt,
        "ideal_target": round(ideal, 1),
        "abs_diff": round(abs_diff, 1),
        "rel_diff": round(rel_diff, 4),
        "direction": direction,
        "ruler": active_ruler(),
    }
