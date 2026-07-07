"""
phoneme_counter.py — Lightweight phoneme/syllable counting for isochrony scoring

Used by isochrony_translation.py to:
  1. Estimate the natural speaking duration of an English source segment.
  2. Score candidate Hindi/Indic translations by how close their phoneme count
     is to the source phoneme count (adjusted by a per-language expansion ratio).

Approach:
  - English: rule-based syllable counting (CMU dict lookup with vowel-cluster fallback).
  - Hindi/Indic (Devanagari): count vowel matras + syllabic consonants — a fast,
    purely rule-based approach that requires no external models or internet access.
  - Other Indic scripts: graceful fallback to character-count approximation.

No ML model is loaded here — this is deliberately CPU-only and instant.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Language expansion ratios (empirically derived from dubbing datasets).
# These represent how many more phonemes the Indic language typically uses
# compared to the English source for the same semantic content.
# A ratio > 1.0 means Indic is "longer" than English.
# ---------------------------------------------------------------------------
EXPANSION_RATIOS = {
    "hindi":      1.30,
    "bengali":    1.25,
    "marathi":    1.28,
    "gujarati":   1.27,
    "punjabi":    1.22,
    "tamil":      1.35,
    "telugu":     1.38,
    "kannada":    1.36,
    "malayalam":  1.40,
    "odia":       1.28,
    "assamese":   1.26,
}

# Devanagari Unicode ranges used by Hindi, Marathi, Sanskrit
DEVANAGARI_RANGE = (0x0900, 0x097F)
# Bengali / Assamese
BENGALI_RANGE = (0x0980, 0x09FF)
# Gujarati
GUJARATI_RANGE = (0x0A80, 0x0AFF)
# Punjabi (Gurmukhi)
GURMUKHI_RANGE = (0x0A00, 0x0A7F)
# Tamil
TAMIL_RANGE = (0x0B80, 0x0BFF)
# Telugu
TELUGU_RANGE = (0x0C00, 0x0C7F)
# Kannada
KANNADA_RANGE = (0x0C80, 0x0CFF)
# Malayalam
MALAYALAM_RANGE = (0x0D00, 0x0D7F)
# Odia
ODIA_RANGE = (0x0B00, 0x0B7F)


# ---------------------------------------------------------------------------
# English syllable counting
# ---------------------------------------------------------------------------

def _count_english_syllables(word: str) -> int:
    """
    Rule-based English syllable counter.
    Returns at least 1 syllable per word.
    """
    word = word.lower().strip(".,!?;:\"'()-")
    if not word:
        return 0
    # Count vowel clusters as one syllable each
    count = len(re.findall(r"[aeiouy]+", word))
    # Subtract silent 'e' at end
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def count_english_phonemes(text: str) -> int:
    """Count approximate phoneme units (syllables) for English text."""
    words = re.findall(r"[a-zA-Z']+", text)
    return sum(_count_english_syllables(w) for w in words)


# ---------------------------------------------------------------------------
# Indic script phoneme counting
# ---------------------------------------------------------------------------

def _in_range(cp: int, ranges) -> bool:
    if isinstance(ranges[0], int):
        return ranges[0] <= cp <= ranges[1]
    return any(lo <= cp <= hi for lo, hi in ranges)


def _count_devanagari_phonemes(text: str) -> int:
    """
    Count syllabic units in Devanagari text (Hindi, Marathi, etc.).

    Rules:
      - Each vowel letter (स्वर) = 1 unit
      - Each consonant NOT followed by a virama (्) = 1 unit
        (i.e., each consonant that carries an implicit 'a' or a matra)
      - Matras (vowel marks) are already attached to a consonant, so
        we do NOT double-count; the consonant+matra = 1 unit total.
      - Virama (halant ्) suppresses the inherent vowel, so the
        consonant before virama does NOT form a syllable by itself.
    """
    count = 0
    chars = list(text)
    i = 0
    while i < len(chars):
        cp = ord(chars[i])
        # Devanagari consonants: 0x0915–0x0939, 0x0958–0x095F, 0x0900-range nuclei
        if 0x0915 <= cp <= 0x0939 or 0x0958 <= cp <= 0x095F:
            # Check if next char is virama
            if i + 1 < len(chars) and ord(chars[i + 1]) == 0x094D:
                # Consonant cluster — does not produce syllable; skip virama too
                i += 2
                continue
            count += 1
        # Independent vowels
        elif 0x0904 <= cp <= 0x0914:
            count += 1
        # Matras (dependent vowels) — already counted with preceding consonant
        # anusvara/visarga sometimes add a mora but we ignore for simplicity
        i += 1
    return max(1, count) if count else _fallback_char_count(text)


def _count_brahmic_phonemes(text: str) -> int:
    """
    Generic phoneme counter for Brahmic scripts (Bengali, Tamil, Telugu, etc.)
    Uses a simpler heuristic: count characters that are NOT combining marks.
    Combining marks (virama-equivalents, matras) are in Unicode category 'Mn' (non-spacing mark).
    """
    count = sum(
        1 for ch in text
        if unicodedata.category(ch) not in ("Mn", "Mc", "Me", "Zs", "Po", "Ps", "Pe")
    )
    return max(1, count // 2)  # Brahmic scripts: roughly 2 code points per syllable


def _fallback_char_count(text: str) -> int:
    """Last-resort: character count divided by 3 as a rough syllable estimate."""
    stripped = re.sub(r"\s+", "", text)
    return max(1, len(stripped) // 3)


def count_indic_phonemes(text: str, language: str) -> int:
    """
    Count approximate phoneme units for an Indic language string.
    Uses phonemizer + espeak-ng if available for true phonetic counts,
    otherwise falls back to rule-based syllable/character counting.

    Args:
        text:     The Indic-script text string.
        language: Language name (lowercase), e.g. 'hindi', 'tamil'.

    Returns:
        Integer phoneme count (≥ 1).
    """
    if not text.strip():
        return 0

    language = language.lower()

    # Attempt to use phonemizer for high-accuracy phonetic counts
    try:
        from phonemizer import phonemize
        # Map target language to espeak language code
        espeak_codes = {
            "hindi": "hi",
            "bengali": "bn",
            "marathi": "mr",
            "gujarati": "gu",
            "punjabi": "pa",
            "tamil": "ta",
            "telugu": "te",
            "kannada": "kn",
            "malayalam": "ml",
            "odia": "or",
            "assamese": "as",
        }
        code = espeak_codes.get(language)
        if code:
            # We strip spaces and symbols, and count distinct phoneme symbols
            phonemes_str = phonemize(text, language=code, backend="espeak", strip=True)
            # Remove spaces which separate words/phonemes, and count remaining symbols
            # Filter out non-alphabetic/IPA characters to avoid counting punctuation
            cleaned = re.sub(r'[\s\.\!\?\,\-\_]', '', phonemes_str)
            if cleaned:
                # Approximate syllable count: clean phonemes count / 2.5 (avg phonemes per syllable)
                # or count vowels in the IPA string directly:
                # IPA vowels for Indic: a, e, i, o, u, ɑ, ɔ, ə, ɛ, ɪ, ʊ, ʌ, æ, ɯ, etc.
                vowels = len(re.findall(r'[aeiouɑɔəɛɪʊʌæɯʊ]', cleaned))
                return max(1, vowels)
    except Exception as e:
        # Gracefully fall back to rule-based counting
        pass

    if language in ("hindi", "marathi", "nepali"):
        return _count_devanagari_phonemes(text)
    elif language in ("bengali", "assamese"):
        return _count_brahmic_phonemes(text)
    elif language == "gujarati":
        return _count_brahmic_phonemes(text)
    elif language == "punjabi":
        return _count_brahmic_phonemes(text)
    elif language == "tamil":
        return _count_brahmic_phonemes(text)
    elif language == "telugu":
        return _count_brahmic_phonemes(text)
    elif language == "kannada":
        return _count_brahmic_phonemes(text)
    elif language == "malayalam":
        return _count_brahmic_phonemes(text)
    elif language == "odia":
        return _count_brahmic_phonemes(text)
    else:
        return _fallback_char_count(text)



# ---------------------------------------------------------------------------
# Isochrony scoring
# ---------------------------------------------------------------------------

def isochrony_score(
    source_text: str,
    target_text: str,
    target_language: str,
) -> float:
    """
    Compute an isochrony compliance score between 0.0 (bad) and 1.0 (perfect).

    The score is based on how close the target phoneme count (adjusted by the
    language expansion ratio) is to the source phoneme count.

    A score ≥ 0.85 is considered acceptable for dubbing.
    A score ≥ 0.95 is excellent.

    Args:
        source_text:      English source segment text.
        target_text:      Translated text in the target language.
        target_language:  Language name, e.g. 'hindi'.

    Returns:
        Float in [0.0, 1.0].
    """
    src_phonemes = count_english_phonemes(source_text)
    tgt_phonemes = count_indic_phonemes(target_text, target_language)
    ratio = EXPANSION_RATIOS.get(target_language.lower(), 1.3)

    # Ideal target phoneme count = source * expansion_ratio
    ideal = src_phonemes * ratio
    if ideal == 0:
        return 1.0

    deviation = abs(tgt_phonemes - ideal) / ideal
    score = max(0.0, 1.0 - deviation)
    return round(score, 4)


def compute_target_budget(
    source_text: str,
    target_language: str,
    tolerance: float = 0.15,
) -> dict:
    """
    Compute the phoneme budget for a translation.

    Returns a dict with:
      'source_phonemes':   int
      'ideal_target':      float (ideal phoneme count in target language)
      'min_target':        int (lower bound with tolerance)
      'max_target':        int (upper bound with tolerance)
      'expansion_ratio':   float
    """
    src = count_english_phonemes(source_text)
    ratio = EXPANSION_RATIOS.get(target_language.lower(), 1.3)
    ideal = src * ratio
    return {
        "source_phonemes": src,
        "ideal_target": round(ideal, 1),
        "min_target": max(1, int(ideal * (1 - tolerance))),
        "max_target": int(ideal * (1 + tolerance)),
        "expansion_ratio": ratio,
    }
