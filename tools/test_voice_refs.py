#!/usr/bin/env python3
"""CPU tests for per-language reference voices (pipeline/voice_refs.py). No GPU, no network.

Run:  python tools/test_voice_refs.py

Asserts properties, not "it ran" (rule 5):
  - the script check separates each language's own script from another's (on real text), and
    a row labelled as another language is rejected even when its script fits;
  - check_clip rejects each failure it claims to catch — silence, clipping, a file that is not
    the length the dataset says, wrong-script text, text that cannot fit the audio, too short;
  - selection from a Rasa-shaped parquet skips higher-ranked clips that fail the audio checks
    and picks the right one, deterministically;
  - the stored clip IS the checked clip (sha256), and a tampered file is refused;
  - the resolver ladder: native > pinned Hindi (hi) > Hindi fallback WITH a warning > text-only;
  - the native voice actually reaches the synthesis call in duration_tts (the control signal
    arrives), and its identity is in the segment signature, so a voice change re-synthesizes
    while every pre-existing (Hindi-voice) signature stays valid.
"""
import hashlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402

from pipeline import voice_refs as vr  # noqa: E402
from pipeline import duration_tts as dt  # noqa: E402

FAILS = []
SR = 16000
TA = dt.BASIC_VOICE_REFS["ta"]        # real Tamil sentence
TE = dt.BASIC_VOICE_REFS["te"]        # real Telugu sentence (wrong script for ta)


def check(name, cond, got=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n        {got}" if got and not cond else ""))
    if not cond:
        FAILS.append(name)


def speech_like(seconds, lead=0.0, trail=0.0, gain=0.1, seed=0):
    """Amplitude-modulated noise: loud enough, unclipped, syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = rng.standard_normal(t.size).astype(np.float32) * gain * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))
    return np.concatenate([np.zeros(int(lead * SR), np.float32), x.astype(np.float32),
                           np.zeros(int(trail * SR), np.float32)])


def wav_bytes(x):
    buf = io.BytesIO()
    sf.write(buf, x, SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def test_language_tables():
    check("voice_refs covers exactly the pipeline's languages",
          {v: k for k, v in vr.CODE_TO_LANGUAGE.items()} == dt.LANGUAGE_TO_CODE)
    check("every language has a script range", set(vr.SCRIPT_RANGES) == set(vr.CODE_TO_LANGUAGE))
    own = {c: vr.script_fraction(dt.BASIC_VOICE_REFS[c], c) for c in vr.CODE_TO_LANGUAGE}
    check("each language's own sentence passes the script check",
          all(f >= vr.MIN_SCRIPT_FRACTION for f in own.values()), own)
    check("a Telugu sentence scores 0 as Tamil", vr.script_fraction(TE, "ta") == 0.0)
    check("Latin-heavy text fails the script check",
          vr.script_fraction("नमस्ते this is mostly English words", "hi") < vr.MIN_SCRIPT_FRACTION)


def test_check_clip():
    ok, why, st, trimmed = vr.check_clip(speech_like(6.0, lead=1.0, trail=1.0), SR, TA, "ta", "8.0")
    check("a clean 6 s Tamil clip passes", ok, why)
    check("edge silence is trimmed (8 s file -> ~6.2 s)", 6.0 <= st["trimmed_seconds"] <= 6.3,
          st["trimmed_seconds"])

    cases = {
        "silent audio is rejected": (np.zeros(6 * SR, np.float32), TA, "6.0", "trimmed_duration"),
        "quiet audio is rejected": (speech_like(6.0, gain=0.001), TA, "6.0", "rms"),
        "clipped audio is rejected": (np.clip(speech_like(6.0) * 30, -1, 1), TA, "6.0", "clipped"),
        "file length != stated length is rejected": (speech_like(6.0), TA, "8.0", "file_vs_stated"),
        "unknown stated length is rejected (unchecked != passed)":
            (speech_like(6.0), TA, "n/a", "stated_duration_unknown"),
        "wrong-script transcript is rejected": (speech_like(6.0), TE, "6.0", "script_fraction"),
        "transcript too long for the audio is rejected":
            (speech_like(6.0), " ".join([TA] * 4), "6.0", "letters_per_second"),
        "too-short clip is rejected": (speech_like(2.0), TA[:30], "2.0", "trimmed_duration"),
    }
    for name, (x, text, stated, expect) in cases.items():
        ok, why, _, _ = vr.check_clip(x, SR, text, "ta", stated)
        check(name, (not ok) and any(r.startswith(expect) for r in why), why)


def write_shard(path, rows):
    table = pa.table({
        "filename": [r[0] for r in rows], "text": [r[5] for r in rows],
        "language": ["Tamil"] * len(rows), "gender": [r[1] for r in rows],
        "style": [r[2] for r in rows], "duration": [r[3] for r in rows],
        "wav_path": [f"{r[0]}.wav" for r in rows],
        "audio": [{"bytes": wav_bytes(r[4]), "path": f"{r[0]}.wav"} for r in rows],
    })
    pq.write_table(table, path, row_group_size=2)
    return path


def test_single_speaker_shards():
    """Real Rasa layout: one speaker per shard (Tamil test-00000 is 457/457 Female)."""
    d = tempfile.mkdtemp()
    fem = write_shard(os.path.join(d, "test-00000.parquet"),
                      [(f"TAM_F_BOOK_{i}", "Female", "BOOK", "6.0", speech_like(6.0, seed=i), TA)
                       for i in range(3)])
    male = write_shard(os.path.join(d, "test-00001.parquet"),
                       [(f"TAM_M_CONV_{i}", "Male", "CONV", "6.0", speech_like(6.0, seed=10 + i), TA)
                        for i in range(3)])
    res = vr.select_from_parquet(lambda p: open(p, "rb"), [fem, male], "ta", gender="male",
                                 log=lambda m: None)
    check("the male voice is found in a later shard, past an all-female one",
          res.get("ok") and res["row"]["filename"].startswith("TAM_M_") and res["shard"] == male,
          res.get("row", {}).get("filename"))
    check("the all-female shard's metadata was never read in full (gender probe only)",
          sum(res["gender_histogram"].values()) == 3 and "Female" not in res["gender_histogram"],
          res["gender_histogram"])
    said = []
    res = vr.select_from_parquet(lambda p: open(p, "rb"), [fem], "ta", gender="male",
                                 log=said.append)
    check("a single-speaker language still gets a voice, and the log says it is the other one",
          res.get("ok") and res["row"]["gender"] == "Female"
          and any("no 'male' speaker" in m for m in said), said)


def make_rasa_parquet(path):
    """Rasa-shaped shard: metadata columns + audio struct{bytes,path}; 2 rows per row group."""
    rows = [  # filename, gender, style, stated, audio, text
        ("ta_m_book_06", "Male", "BOOK", "6.0", speech_like(8.5, seed=6), TA),     # rank 1: file != stated
        ("ta_m_book_04", "Male", "BOOK", "6.1", np.clip(speech_like(6.1, seed=4) * 30, -1, 1), TA),  # clipped
        ("ta_m_book_03", "Male", "BOOK", "6.0", speech_like(6.0, seed=3), TE),     # wrong script (metadata)
        ("ta_m_book_02", "Male", "BOOK", "6.3", speech_like(6.3, seed=2), TA),     # EXPECTED
        ("ta_m_anger_05", "Male", "ANGER", "6.0", speech_like(6.0, seed=5), TA),   # worse style
        ("ta_f_book_01", "Female", "BOOK", "6.0", speech_like(6.0, seed=1), TA),   # wrong gender
    ]
    return write_shard(path, rows)


def test_selection_and_curation():
    d = tempfile.mkdtemp()
    shard = make_rasa_parquet(os.path.join(d, "test-00000-of-00001.parquet"))
    logs = []
    res = vr.select_from_parquet(lambda p: open(p, "rb"), [shard], "ta", gender="male",
                                 log=logs.append)
    check("selection picks the best clip that PASSES (not the top-ranked one)",
          res.get("ok") and res["row"]["filename"] == "ta_m_book_02",
          res.get("row", {}).get("filename") if res.get("ok") else res)
    rej = res.get("rejections", {})
    check("higher-ranked failures were rejected for the right reasons",
          rej.get("clipped", 0) >= 1 and rej.get("file_vs_stated", 0) >= 1
          and rej.get("script_fraction", 0) >= 1, rej)
    res_f = vr.select_from_parquet(lambda p: open(p, "rb"), [shard], "ta", gender="female",
                                   log=lambda m: None)
    check("gender preference is honoured", res_f.get("ok") and res_f["row"]["filename"] == "ta_f_book_01",
          res_f.get("row", {}).get("filename"))

    out = os.path.join(d, "refs")
    sel = lambda code: vr.select_from_parquet(lambda p: open(p, "rb"), [shard], code,  # noqa: E731
                                              gender="male", log=lambda m: None)
    man = vr.curate(out, languages=["ta", "te"], selector=sel, log=lambda m: None)
    e = man["languages"]["ta"]
    wav = os.path.join(out, e["file"])
    with open(wav, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    check("manifest sha256 is the stored file's hash", e["sha256"] == digest)
    info = sf.info(wav)
    check("stored clip is the trimmed, checked clip",
          abs(info.frames / info.samplerate - e["stats"]["trimmed_seconds"]) < 0.01,
          (info.frames / info.samplerate, e["stats"]["trimmed_seconds"]))
    check("manifest records provenance + licence + criteria",
          e["source_filename"] == "ta_m_book_02" and man["license"] == vr.RASA_LICENSE
          and man["criteria"]["min_script_fraction"] == vr.MIN_SCRIPT_FRACTION)
    te = man["languages"]["te"]
    check("a language with no passing clip is recorded as failed, with a reason",
          te["status"] == "failed" and te["reason"], te)
    check("rows labelled with another language are rejected even if the script fits",
          (te.get("rejections") or {}).get("language_mismatch", 0) == 6, te.get("rejections"))
    cov = vr.coverage(out)
    check("coverage: ta native, te fallback, hi pinned",
          cov["ta"] == "native" and cov["te"] == "fallback" and cov["hi"] == "pinned", cov)
    return out


def test_resolver(ref_dir):
    pinned = os.path.join(tempfile.mkdtemp(), "hin.wav")
    sf.write(pinned, speech_like(4.0), SR)

    said = []
    r = vr.resolve_basic_reference("ta", said.append, directory=ref_dir,
                                   download_pinned=lambda: pinned)
    check("Tamil resolves to the native clip + its transcript",
          r["source"] == "native" and r["audio"].endswith("ta.wav") and r["text"] == TA, r)
    check("native ref_id carries the clip hash", r["ref_id"].startswith("native:")
          and r["ref_id"] == vr.native_ref_id("ta", ref_dir), r["ref_id"])

    said = []
    r = vr.resolve_basic_reference("hi", said.append, directory=ref_dir, download_pinned=lambda: pinned)
    check("Hindi keeps the pinned clip and its verified transcript, ref_id ''",
          r["source"] == "pinned" and r["audio"] == pinned and r["text"] == vr.DEFAULT_HINDI_REF_TEXT
          and r["ref_id"] == "", r)

    said = []
    r = vr.resolve_basic_reference("te", said.append, directory=ref_dir, download_pinned=lambda: pinned)
    check("Telugu without a native clip falls back to Hindi AND says so",
          r["source"] == "fallback" and r["audio"] == pinned
          and any("WARNING" in m and "cross-lingually" in m for m in said), said)

    def boom():
        raise OSError("HF unreachable")
    r = vr.resolve_basic_reference("te", None, directory=ref_dir, download_pinned=boom,
                                   text_only_refs=dt.BASIC_VOICE_REFS)
    check("HF unreachable -> text-only (previous behaviour), never raises",
          r["source"] == "text-only" and r["audio"] is None and r["text"] == TE, r)

    # Tamper with the stored clip: it is no longer the clip that was checked.
    tampered = tempfile.mkdtemp()
    for f in os.listdir(ref_dir):
        with open(os.path.join(ref_dir, f), "rb") as a, open(os.path.join(tampered, f), "wb") as b:
            b.write(a.read())
    with open(os.path.join(tampered, "ta.wav"), "ab") as fh:
        fh.write(b"\0\0")
    said = []
    r = vr.resolve_basic_reference("ta", said.append, directory=tampered, download_pinned=lambda: pinned)
    check("a modified clip is refused (sha256) and the fallback is logged",
          r["source"] == "fallback" and any("sha256" in m for m in said), said)
    check("native_ref_id agrees with the resolver on a tampered clip",
          vr.native_ref_id("ta", tampered) == "")
    return pinned


def test_signature_compat():
    legacy = hashlib.sha1("ta|32|1.5000|வணக்கம்".encode("utf-8")).hexdigest()
    check("empty ref_id keeps the pre-existing signature (old manifests stay valid)",
          dt._segment_signature("வணக்கம்", 1.5, "ta", 32) == legacy
          and dt._segment_signature("வணக்கம்", 1.5, "ta", 32, "") == legacy)
    check("a native voice changes the signature",
          dt._segment_signature("வணக்கம்", 1.5, "ta", 32, "native:abc") != legacy)


def test_native_voice_reaches_synthesis(ref_dir, pinned):
    """generate_tts_for_segments with the model stubbed: which reference does synthesis get?"""
    seen = []

    def fake_generate(**kw):
        seen.append((kw["ref_audio_path"], kw["ref_text"]))
        return np.zeros(int(kw["target_duration"] * dt.INDICF5_SAMPLE_RATE), np.float32)

    saved = (dt._load_indicf5, dt._generate_single_segment, os.environ.get(vr.ENV_DIR))
    dt._load_indicf5 = lambda device, log_fn=None: (object(), "cpu")
    dt._generate_single_segment = fake_generate
    os.environ[vr.ENV_DIR] = ref_dir
    try:
        out = tempfile.mkdtemp()
        segs = [{"start": 0.0, "end": 1.5, "text": "வணக்கம்"}, {"start": 2.0, "end": 3.0, "text": "நன்றி"}]
        dt.generate_tts_for_segments(segs, "Tamil", out, log_fn=lambda m: None)
        native_wav = os.path.join(ref_dir, "ta.wav")
        check("synthesis receives the native Tamil clip + Tamil transcript for every segment",
              len(seen) == 2 and all(a == native_wav and t == TA for a, t in seen), seen)
        with open(os.path.join(out, dt.MANIFEST_NAME), encoding="utf-8") as fh:
            man = json.load(fh)
        rid = vr.native_ref_id("ta", ref_dir)
        want = dt._segment_signature("வணக்கம்", 1.5, "ta", dt.resolve_nfe_step(None), rid)
        check("the worker's manifest signature includes the native voice", man["0"]["sig"] == want,
              (man["0"]["sig"], want))

        from deploy.tts_fanout import completed_indices
        check("fan-out resume agrees with the worker (native voice => done)",
              completed_indices(segs, out, "Tamil") == {0, 1})
        check("...and the same audio is NOT reused for a cloning run (different voice)",
              completed_indices(segs, out, "Tamil", reference_audio_path=pinned) == set())

        seen.clear()
        out2 = tempfile.mkdtemp()
        dt._generate_single_segment = fake_generate
        dt.generate_tts_for_segments(segs[:1], "Telugu", out2, log_fn=lambda m: None,
                                     reference_audio_path=pinned, reference_text="x")
        check("cloning mode still uses the caller's clip, untouched by voice_refs",
              seen == [(pinned, "x")], seen)
    finally:
        dt._load_indicf5, dt._generate_single_segment = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop(vr.ENV_DIR, None)
        else:
            os.environ[vr.ENV_DIR] = saved[2]


def test_supervisor_imports():
    from pipeline import tts_supervisor
    check("tts_supervisor uses the same ref identity helper", tts_supervisor.native_ref_id is vr.native_ref_id)


def main():
    test_language_tables()
    test_check_clip()
    test_single_speaker_shards()
    ref_dir = test_selection_and_curation()
    pinned = test_resolver(ref_dir)
    test_signature_compat()
    test_native_voice_reaches_synthesis(ref_dir, pinned)
    test_supervisor_imports()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
