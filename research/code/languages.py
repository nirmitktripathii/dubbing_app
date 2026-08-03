"""
common/languages.py
--------------------
Single source of truth for language metadata used across pipeline_v3.

Every other module (dataset_generator, duration_predictor, isochrony_translation_v3,
train_translation_llm, data_augmentation, train_tts) imports from here instead of
hard-coding its own language table. This avoids the classic bug where "hi" is spelled
one way in one file and another way in another file.

The 11 languages below are the intersection of:
  - Samanantar's 11 Indic targets (as, bn, gu, hi, kn, ml, mr, or, pa, ta, te)
  - IndicF5's 11 supported languages (same set)
  - espeak-ng's Indic voice coverage (verified: all 11 present as of espeak-ng 1.51)

NOTE ON EXPANSION RATIOS: The `heuristic_expansion_ratio` and `heuristic_phonemes_per_sec`
values below are *cold-start defaults*, carried forward from/consistent with the existing
V2 pipeline's phoneme_counter.py approach (e.g. Hindi=1.30, Tamil=1.35). They are
deliberately approximate. The entire point of V3 is to replace these hard-coded numbers
with the learned DurationPredictor (see translation/duration_predictor.py) once you have
trained it on real forced-aligned speech. Treat this table as the fallback that is used
(a) before you've trained anything, and (b) as a sanity-check ceiling/floor on predicted
durations even after training.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageInfo:
    name: str                          # Human-readable display name
    iso_code: str                      # 2-letter ISO 639-1 code, used by Samanantar & IndicF5
    espeak_code: str                   # espeak-ng voice code (for phonemizer backend="espeak")
    indicf5_code: str                  # Code IndicF5 expects (matches iso_code for all 11)
    script: str                        # Unicode script name, for sanity checks / logging
    heuristic_expansion_ratio: float   # Indic-text-length / English-text-length, rough prior
    heuristic_phonemes_per_sec: float  # Cold-start speaking rate for DurationPredictor fallback


# fmt: off
LANGUAGES: dict[str, LanguageInfo] = {
    "hi": LanguageInfo("Hindi",      "hi", "hi", "hi", "Devanagari", 1.30, 13.5),
    "bn": LanguageInfo("Bengali",    "bn", "bn", "bn", "Bengali",    1.28, 13.0),
    "mr": LanguageInfo("Marathi",    "mr", "mr", "mr", "Devanagari", 1.27, 13.2),
    "gu": LanguageInfo("Gujarati",   "gu", "gu", "gu", "Gujarati",   1.25, 13.0),
    "pa": LanguageInfo("Punjabi",    "pa", "pa", "pa", "Gurmukhi",   1.22, 12.8),
    "ta": LanguageInfo("Tamil",      "ta", "ta", "ta", "Tamil",      1.35, 12.5),
    "te": LanguageInfo("Telugu",     "te", "te", "te", "Telugu",     1.33, 12.6),
    "kn": LanguageInfo("Kannada",    "kn", "kn", "kn", "Kannada",    1.30, 12.7),
    "ml": LanguageInfo("Malayalam",  "ml", "ml", "ml", "Malayalam",  1.38, 12.3),
    "or": LanguageInfo("Odia",       "or", "or", "or", "Odia",       1.26, 13.0),
    "as": LanguageInfo("Assamese",   "as", "as", "as", "Bengali",    1.27, 12.9),
}
# fmt: on

# Samanantar's HF dataset config names are lowercase 2-letter codes identical to iso_code,
# EXCEPT the source side, which is always English.
SAMANANTAR_SOURCE_LANG = "en"
SAMANANTAR_HF_PATH = "ai4bharat/samanantar"

# ai4bharat/Kathbath directory structure uses full language *names* (lowercase), not codes.
KATHBATH_DIR_NAMES: dict[str, str] = {
    "hi": "hindi", "bn": "bengali", "mr": "marathi", "gu": "gujarati",
    "pa": "punjabi", "ta": "tamil", "te": "telugu", "kn": "kannada",
    "ml": "malayalam", "or": "odia", "as": "assamese",
    # Kathbath additionally has Sanskrit/Urdu/Nepali/Chhattisgarhi splits not covered here
    # because they fall outside the 11-language Samanantar/IndicF5 intersection.
}
KATHBATH_HF_PATH = "ai4bharat/Kathbath"

# ai4bharat/Rasa is the (much closer to IndicF5's own training mix) TTS-quality speech
# dataset: studio recordings, 13 languages, ~500+ hours, MIT-tagged on HF. Prefer this over
# Kathbath (which is ASR-oriented, phone/field recordings) for anything TTS-related, and
# fall back to Kathbath only if you need more raw hours than Rasa provides for a language.
RASA_HF_PATH = "ai4bharat/Rasa"

# ai4bharat/BPCC (Bharat Parallel Corpus Collection) - the successor to Samanantar:
# ~230M pairs, 22 languages, and (unlike standalone Samanantar's ambiguous HF license
# tag) an EXPLICIT license table on its dataset card: all MINED corpora - including
# Samanantar (19.4M) and Samanantar++ (121.6M) - are CC0; the human-annotated seed
# subsets (BPCC-H-Wiki/Daily) are CC-BY-4.0. This resolves the Samanantar license
# ambiguity flagged in dataset_generator.py for commercial use. GATED: accept the
# conditions on huggingface.co/datasets/ai4bharat/BPCC with your HF account first.
#
# STRUCTURE (verified 2026-07-13 by browsing the gated repo directly, including
# AI4Bharat's own compile.py in the repo root, which generated these files):
# BPCC is a RAW-FILE repo, not a config-per-language dataset - the HF dataset viewer is
# disabled for it and it has no loading script, so load_dataset("ai4bharat/BPCC",
# <config>) does NOT work. The data lives in per-subset directories of per-language
# TSVs keyed by FLORES-200 code:
#     <subset>/<flores_code>.tsv    e.g.  samanantar_v2/hin_Deva.tsv  (6.9 GB)
# Each TSV is tab-separated WITH a header row and exactly these columns (from
# compile.py: df.to_csv(sep='\t', index=False)):
#     src_lang  tgt_lang  src  tgt      (src = English text, tgt = Indic text)
# All 11 of our languages are present in samanantar_v2/ (plus npi/urd, unused here).
# dataset_generator.py streams these via load_dataset("csv", data_files="hf://...").
# Subsets seen in the repo: samanantar_v2 (default - 28.9GB, the filtered mined set),
# samanantar_v0.3_filtered, nllb_filtered, nllb_seed, bpcc-seed-latest/v1/v2,
# comparable, daily, ilci, massive, wiki. Only samanantar_v2's internal layout was
# inspected; treat other subsets' layouts as unverified until probed.
BPCC_HF_PATH = "ai4bharat/BPCC"
BPCC_SOURCE_FLORES = "eng_Latn"
BPCC_DEFAULT_SUBSET = "samanantar_v2"
BPCC_FLORES_CODES: dict[str, str] = {
    "hi": "hin_Deva", "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr",
    "pa": "pan_Guru", "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda",
    "ml": "mal_Mlym", "or": "ory_Orya", "as": "asm_Beng",
}

# IndicF5 itself
INDICF5_HF_PATH = "ai4bharat/IndicF5"
INDICF5_SAMPLE_RATE = 24000

# ai4bharat/vits_rasa_13 - fixed-inventory multi-speaker VITS TTS (40.2M params,
# CC-BY-4.0, GATED). Everything below is transcribed 1:1 from the model card's own
# speaker/style tables (verified 2026-07-13 via authenticated access, not assumed).
# IMPORTANT COVERAGE FACT: the model supports 13 languages but NOT Hindi, Gujarati, or
# Odia - three of this pipeline's 11 targets. hi/gu/or must use the IndicF5 backend
# for multi-speaker dubbing (see tts/vits_rasa_tts.py and multi_speaker_dubbing.py).
VITS_RASA_HF_PATH = "ai4bharat/vits_rasa_13"
VITS_RASA_SPEAKERS: dict[str, int] = {
    "ASM_F": 0, "ASM_M": 1, "BEN_F": 2, "BEN_M": 3, "BRX_F": 4, "BRX_M": 5,
    "DOI_F": 6, "DOI_M": 7, "KAN_F": 8, "KAN_M": 9, "MAI_M": 10, "MAL_F": 11,
    "MAR_F": 12, "MAR_M": 13, "NEP_F": 14, "PAN_F": 15, "PAN_M": 16, "SAN_M": 17,
    "TAM_F": 18, "TEL_F": 19,
}
# Style/emotion IDs exactly as published (note the real gaps at 9/11/13 - the model
# card skips those IDs; do not "fix" this by renumbering).
VITS_RASA_STYLES: dict[str, int] = {
    "ALEXA": 0, "ANGER": 1, "BB": 2, "BOOK": 3, "CONV": 4, "DIGI": 5, "DISGUST": 6,
    "FEAR": 7, "HAPPY": 8, "NEWS": 10, "SAD": 12, "SURPRISE": 14, "UMANG": 15, "WIKI": 16,
}
# Per-target-language voice availability, keyed by this pipeline's ISO codes.
# None = that gender doesn't exist in the model (ml/ta/te are female-only).
VITS_RASA_VOICES_BY_LANG: dict[str, dict] = {
    "as": {"F": 0, "M": 1},
    "bn": {"F": 2, "M": 3},
    "kn": {"F": 8, "M": 9},
    "ml": {"F": 11, "M": None},
    "mr": {"F": 12, "M": 13},
    "pa": {"F": 15, "M": 16},
    "ta": {"F": 18, "M": None},
    "te": {"F": 19, "M": None},
}
VITS_RASA_UNSUPPORTED: frozenset = frozenset({"hi", "gu", "or"})

# pyannote speaker diarization (multi-speaker dubbing's segmentation stage). BOTH repos
# are gated on HF - accept terms on each model page with the same HF_TOKEN account:
#   huggingface.co/pyannote/speaker-diarization-3.1
#   huggingface.co/pyannote/segmentation-3.0  (pulled in by the pipeline above)
PYANNOTE_DIARIZATION_HF_PATH = "pyannote/speaker-diarization-3.1"


def get_language(code: str) -> LanguageInfo:
    code = code.lower().strip()
    if code not in LANGUAGES:
        raise ValueError(
            f"Unknown language code '{code}'. Supported: {sorted(LANGUAGES.keys())}"
        )
    return LANGUAGES[code]


def all_codes() -> list[str]:
    return list(LANGUAGES.keys())


def display_name(code: str) -> str:
    return get_language(code).name
