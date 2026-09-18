#!/usr/bin/env python3
"""Turn an existing pipeline_log.txt into a per-stage wall-clock profile — CPU only, no GPU.

The headless driver already logs every stage boundary with an [HH:MM:SS] stamp
("Step N/7: ...", per-segment "[Segment i/n]", "DONE in ..."). This parses those
stamps into the stage table the optimization audit asks for (§6), from runs that
ALREADY happened. No re-run, no fabrication — every number is sourced to a log line.
"""
import re, sys, datetime as dt

STEP = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s*Step\s+([\d.]+)/7:\s*(.+?)\s*$")
STAMP = re.compile(r"^\[(\d\d:\d\d:\d\d)\]")
DONE = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s*DONE in")
SEG_CALL = re.compile(r"Calling infer_batch_process.*nfe_step=(\d+)")
SEG_PROC = re.compile(r"Processing \[Segment (\d+)/(\d+)\]")
SEG_RET = re.compile(r"\[Segment (\d+)/\d+\] Synthesis returned")

def parse(path):
    def t(s):  # HH:MM:SS -> seconds since midnight (handles a single midnight wrap)
        h, m, s = map(int, s.split(":"))
        return h * 3600 + m * 60 + s
    steps, last_stamp, done = [], None, None
    seg_proc, seg_ret = {}, {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP.match(line)
            if m:
                steps.append((t(m.group(1)), m.group(2), m.group(3)[:42]))
            if DONE.match(line):
                done = t(DONE.match(line).group(1))
            sm = STAMP.match(line)
            if sm:
                last_stamp = t(sm.group(1))
            mp = SEG_PROC.search(line)
            if mp and sm:
                seg_proc[int(mp.group(1))] = t(sm.group(1))
            mr = SEG_RET.search(line)
            if mr and sm:
                seg_ret[int(mr.group(1))] = t(sm.group(1))
    end = done if done is not None else last_stamp
    rows = []
    for i, (ts, num, name) in enumerate(steps):
        nxt = steps[i + 1][0] if i + 1 < len(steps) else end
        dur = (nxt - ts) % 86400
        rows.append((num, name, dur))
    total = ((end - steps[0][0]) % 86400) if steps else 0

    # TTS decomposition: model-load term (Step-6 stamp -> first "Processing Segment 0")
    # vs the per-segment synthesis times.
    load = None
    step6 = next((ts for ts, num, _ in steps if num.startswith("6")), None)
    if step6 is not None and 0 in seg_proc:
        load = (seg_proc[0] - step6) % 86400
    seg_times = []
    for i in sorted(seg_proc):
        if i in seg_ret:
            seg_times.append((seg_ret[i] - seg_proc[i]) % 86400)
    return rows, total, load, seg_times

def main(paths):
    for p in paths:
        try:
            rows, total, load, seg = parse(p)
        except Exception as e:
            print(f"{p}: parse error {e}"); continue
        print("=" * 66)
        print(p)
        print("=" * 66)
        print(f"{'stage':<40}{'sec':>7}{'% total':>10}")
        print("-" * 57)
        for num, name, dur in rows:
            pct = (100 * dur / total) if total else 0
            print(f"  Step {num:<4} {name:<30}{dur:>7d}{pct:>9.0f}%")
        print("-" * 57)
        print(f"{'TOTAL wall':<40}{total:>7d}{100:>9}%")
        if load is not None:
            print(f"\n  Step-6 IndicF5 cold load (in-subprocess): {load}s")
        if seg:
            print(f"  TTS segments: n={len(seg)}  sum={sum(seg)}s  "
                  f"mean={sum(seg)/len(seg):.1f}s  min={min(seg)}s  max={max(seg)}s")
        print()

if __name__ == "__main__":
    main(sys.argv[1:])
