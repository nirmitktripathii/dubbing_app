#!/usr/bin/env python3
"""CPU tests for the Step-6 fan-out orchestration (deploy/tts_fanout.py).

Run:  python deploy/test_tts_fanout.py      (no GPU, no Modal, no IndicF5)

Covers the parts that decide CORRECTNESS rather than speed, with a fake shard runner
standing in for the GPU workers:
  * shards partition the pending work exactly — every segment once, none dropped/duplicated
  * shard count is capped so we never spin a container up per segment
  * global indices survive sharding, so filenames/manifest keys match a serial run
  * merge is a single writer producing one coherent manifest
  * resume: an already-ok segment is not re-synthesized, but a segment whose TEXT CHANGED
    (signature mismatch) IS re-synthesized rather than silently reused
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.tts_fanout import (  # noqa: E402
    completed_indices, generate_tts_fanout, merge_shard_results, plan_shards,
)
from pipeline.duration_tts import (  # noqa: E402
    LANGUAGE_TO_CODE, MANIFEST_NAME, _segment_signature, resolve_nfe_step,
)

LANG = "Hindi"
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n        {detail}" if detail else ""))


def segs(n, text="line"):
    return [{"start": float(i), "end": float(i) + 1.0, "text": f"{text} {i}"} for i in range(n)]


def fake_wav(nbytes=200):
    """A blob >44 bytes so it passes the 'bigger than a bare WAV header' test."""
    return b"RIFF" + struct.pack("<I", nbytes) + b"WAVEfmt " + b"\0" * nbytes


def fake_runner(segments, lang=LANG):
    """Stand-in for TTSEngine.synth_shard.map — produces what a real worker would return."""
    lang_code = LANGUAGE_TO_CODE.get(lang)
    nfe = resolve_nfe_step(None)
    seen = []

    def run(specs):
        out = []
        for spec in specs:
            only = sorted(spec["only_indices"])
            seen.append(only)
            entries, wavs = {}, {}
            for i in only:
                s = segments[i]
                entries[str(i)] = {
                    "status": "ok",
                    "sig": _segment_signature(s["text"].strip(), s["end"] - s["start"],
                                              lang_code, nfe),
                    "path": f"/worker/private/segment_{i:04d}.wav",
                }
                wavs[str(i)] = fake_wav()
            out.append({"entries": entries, "wavs": wavs, "log": [f"shard {only} ok"]})
        return out

    run.seen = seen
    return run


def test_plan_shards():
    sh = plan_shards(range(13), max_shards=4)
    flat = [i for s in sh for i in s]
    check("shards partition the work exactly (13 segs, 4 shards)",
          sorted(flat) == list(range(13)) and len(flat) == 13,
          f"{sh}")
    check("shard count respects the cap", len(sh) <= 4, f"{len(sh)} shards")

    # Cheap work must not spin up one container per segment: the model load costs more
    # than a segment, so 3 segments should not become 3 shards at min_per_shard=2.
    sh2 = plan_shards(range(3), max_shards=8, min_per_shard=2)
    check("small job does not over-shard", len(sh2) <= 2, f"3 segments -> {sh2}")

    check("no pending work yields no shards", plan_shards([]) == [])

    # Sharding a sparse (resumed) index set keeps the GLOBAL indices, never renumbering.
    sh3 = plan_shards([2, 5, 9], max_shards=8, min_per_shard=1)
    check("sparse indices keep global numbering",
          sorted(i for s in sh3 for i in s) == [2, 5, 9], f"{sh3}")


def test_merge_is_single_writer():
    out = tempfile.mkdtemp()
    s = segs(4)
    res = fake_runner(s)([{"only_indices": [0, 1]}, {"only_indices": [2, 3]}])
    merged = merge_shard_results(res, out)

    files = sorted(f for f in os.listdir(out) if f.endswith(".wav"))
    check("every shard's WAV lands under the run's global name",
          files == [f"segment_{i:04d}.wav" for i in range(4)], f"{files}")

    with open(os.path.join(out, MANIFEST_NAME), encoding="utf-8") as fh:
        man = json.load(fh)
    check("one merged manifest covers all shards", sorted(man, key=int) == ["0", "1", "2", "3"],
          f"keys={sorted(man, key=int)}")
    check("manifest paths were rewritten to the run dir",
          all(os.path.dirname(v["path"]) == out for v in man.values()),
          f"sample={man['0']['path']}")
    check("merge reports what it wrote", merged["written"] == 4 and not merged["failed"],
          f"written={merged['written']} failed={merged['failed']}")


def test_end_to_end_and_resume():
    out = tempfile.mkdtemp()
    s = segs(7)
    runner = fake_runner(s)
    got = generate_tts_fanout(s, LANG, out, shard_runner=runner, max_shards=3)

    check("returns one entry per input segment, in order",
          len(got) == 7 and all(g["audio_path"].endswith(f"segment_{i:04d}.wav")
                                for i, g in enumerate(got)), f"{len(got)} returned")
    check("first pass synthesizes every segment",
          sorted(i for sh in runner.seen for i in sh) == list(range(7)), f"{runner.seen}")

    # ---- resume: nothing changed, so nothing should be re-synthesized ----
    done = completed_indices(s, out, LANG)
    check("all 7 count as complete on disk after the first pass", done == set(range(7)),
          f"done={sorted(done)}")
    runner2 = fake_runner(s)
    generate_tts_fanout(s, LANG, out, shard_runner=runner2, max_shards=3)
    check("resume re-synthesizes NOTHING when inputs are unchanged",
          runner2.seen == [], f"re-ran {runner2.seen}")

    # ---- staleness: a changed line must NOT reuse the old audio ----
    s2 = [dict(x) for x in s]
    s2[3]["text"] = "this line was retranslated"
    done2 = completed_indices(s2, out, LANG)
    check("a segment whose text changed is NOT considered done",
          3 not in done2 and len(done2) == 6, f"done={sorted(done2)}")
    runner3 = fake_runner(s2)
    generate_tts_fanout(s2, LANG, out, shard_runner=runner3, max_shards=3)
    check("only the changed segment is re-synthesized",
          sorted(i for sh in runner3.seen for i in sh) == [3], f"{runner3.seen}")


def test_missing_segment_is_reported():
    out = tempfile.mkdtemp()
    s = segs(3)

    def lossy(specs):
        # A worker that silently drops segment 1 (the degradation path must notice).
        res = fake_runner(s)(specs)
        for r in res:
            r["entries"].pop("1", None)
            r["wavs"].pop("1", None)
        return res

    said = []
    generate_tts_fanout(s, LANG, out, shard_runner=lossy, log_fn=said.append, max_shards=1)
    check("an unsettled segment is surfaced, not silently dropped",
          any("did not settle" in m for m in said),
          [m.strip() for m in said if "settle" in m][:1])


def main():
    test_plan_shards()
    test_merge_is_single_writer()
    test_end_to_end_and_resume()
    test_missing_segment_is_reported()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
