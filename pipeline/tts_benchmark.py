"""Model-agnostic TTS benchmark scorecard for the dubbing pipeline.

Turns on-disk synthesis artifacts into the four candidate-scorecard axes the dubbing
doctrine requires, per candidate (IndicF5 at a given ``nfe_step``, or a FastPitch backend):

  1. control-signal proof — did the duration-control value actually reach the model?
                            Recorded BY THE BACKEND at synth time and asserted here; never
                            inferred from "the audio sounded right" (non-negotiable #1).
  2. length slope         — regress produced vs requested duration over 0.6-1.4x natural;
                            slope -> 1.0 obeys, -> 0.0 ignores (dubbing eval doctrine).
  3. latency              — seconds per segment on the run device.
  4. CER                  — Whisper back-transcription -> char error rate vs the input text
                            (catches hallucination / truncation / wrong script).

Design rules honoured here:
  * Everything is pure/CPU except :func:`back_transcribe_cer` (Whisper, guarded import), so
    the math is unit-testable with NO GPU and NO torch.
  * Every report is provenance-stamped (checkpoint, clip, nfe_step, n samples, data version)
    per non-negotiable #8 — no metric without a traceable source.
  * A ratio direction test is built in (non-negotiable #5): a target that is short relative
    to natural must produce short audio, never natural-length (control ignored) or blown-up
    (control inverted).

This module does NOT drive a GPU model — a separate benchmark worker (run under
pipeline.tts_supervisor for hang-safety) produces the artifacts this module scores.
"""
from __future__ import annotations

import io
import json
import math
import os
import wave
from datetime import datetime
from typing import Callable, Optional, Sequence

# Default probe ratios: target = natural_duration * ratio. Spans the dubbing eval band.
DEFAULT_PROBE_RATIOS = (0.6, 0.8, 1.0, 1.2, 1.4)

# A produced segment shorter than this many seconds is treated as empty/failed, not a
# real duration measurement (matches the >44-byte WAV sanity check elsewhere).
_MIN_VALID_SECONDS = 0.02

# Name of the worker's resume manifest (mirrors pipeline.duration_tts.MANIFEST_NAME).
# Duplicated as a literal so this module imports with no torch/GPU/duration_tts deps.
_MANIFEST_NAME = "tts_manifest.json"


# ─────────────────────────────────────────────────────────────────────────────
# Char error rate (pure Python; no external dependency)
# ─────────────────────────────────────────────────────────────────────────────
def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (iterative, O(len(a)*len(b)) time, O(len(b)) mem)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _normalize_text(s: str) -> str:
    """Whitespace-collapse + casefold. Deliberately script-agnostic: no transliteration,
    no punctuation stripping beyond edge whitespace, so Devanagari/Dravidian text is
    compared as-is (non-negotiable #2/#3 — do not silently transform the unit of measure)."""
    return " ".join((s or "").split()).casefold()


def compute_cer(ref: str, hyp: str, *, normalize: bool = True) -> float:
    """Character error rate = edit_distance(ref, hyp) / max(len(ref), 1).

    Returns 0.0 for two empty strings, and 1.0 when ref is empty but hyp is not (any
    output against no reference is fully wrong). Values can exceed 1.0 when hyp is much
    longer than ref (many insertions) — that is correct and a useful hallucination signal.
    """
    r = _normalize_text(ref) if normalize else (ref or "")
    h = _normalize_text(hyp) if normalize else (hyp or "")
    if not r and not h:
        return 0.0
    if not r:
        return 1.0
    return _levenshtein(r, h) / len(r)


# ─────────────────────────────────────────────────────────────────────────────
# Length-slope regression (least squares; no numpy/scipy)
# ─────────────────────────────────────────────────────────────────────────────
def linregress(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """Ordinary least-squares fit y = slope*x + intercept.

    Returns {"slope", "intercept", "r2", "n"}. slope/intercept/r2 are None when the fit is
    undefined (n < 2 or all x identical) — a benchmark must not invent a number it cannot
    compute (non-negotiable #8).
    """
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    n = len(pts)
    if n < 2:
        return {"slope": None, "intercept": None, "r2": None, "n": n}
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx == 0.0:
        return {"slope": None, "intercept": None, "r2": None, "n": n}
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else None
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n}


def length_slope(pairs: Sequence[tuple]) -> dict:
    """Given (requested_seconds, produced_seconds) pairs, regress produced ON requested.

    slope -> 1.0 means the backend obeys the duration budget; slope -> 0.0 means it ignores
    it and always speaks at its own pace. Also returns mean signed error
    (produced-requested)/requested — sign matters more than magnitude (dubbing eval
    doctrine): positive = the dub runs long (needs compression), negative = runs short
    (padding is the friendlier failure). Invalid/empty produced durations are dropped and
    counted in "dropped".
    """
    good = [(float(rq), float(pr)) for rq, pr in pairs
            if pr is not None and rq is not None and float(pr) >= _MIN_VALID_SECONDS and float(rq) > 0]
    dropped = len(pairs) - len(good)
    fit = linregress([g[0] for g in good], [g[1] for g in good])
    signed = [(pr - rq) / rq for rq, pr in good]
    mean_signed = sum(signed) / len(signed) if signed else None
    abs_err = [abs(s) for s in signed]
    mean_abs = sum(abs_err) / len(abs_err) if abs_err else None
    return {
        **fit,
        "mean_signed_error": mean_signed,
        "mean_abs_error": mean_abs,
        "used": len(good),
        "dropped": dropped,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Control-signal proof + ratio direction test (non-negotiables #1 and #5)
# ─────────────────────────────────────────────────────────────────────────────
def control_record(backend: str, param_name: str, requested_value, applied_value,
                   applied: bool) -> dict:
    """A backend fills this at synth time to PROVE the duration control reached the model.

    e.g. IndicF5 -> control_record("indicf5", "fix_duration", ref+target, passed_value, True)
         FastPitch -> control_record("fastpitch", "length_scale", target/natural, model.length_scale, True)
    `applied` must be set by instrumentation (inspecting the call / reading model state),
    NOT by trusting that a kwarg was accepted.
    """
    return {
        "backend": backend,
        "param": param_name,
        "requested": requested_value,
        "applied": applied_value,
        "reached_model": bool(applied),
    }


def assert_control_signal(rec: dict) -> None:
    """Raise if the backend could not confirm the control value reached the model."""
    if not rec or not rec.get("reached_model"):
        raise RuntimeError(
            f"control signal NOT confirmed at the model: {rec!r} — a duration knob the "
            f"library silently drops produces plausible, uncontrolled audio (non-negotiable #1)."
        )


def direction_ok(target_s: float, produced_s: float, natural_s: float,
                 *, min_gap_ratio: float = 0.15) -> Optional[bool]:
    """Ratio direction test: with an asymmetric target (target clearly != natural), the
    produced duration must land nearer the TARGET than the NATURAL length.

    Catches both failure modes an inverted or dropped ratio hides:
      * control ignored  -> produced ~ natural (far from target)
      * control inverted -> produced ~ natural^2/target (far from target the other way)

    Returns None when target and natural are too close to discriminate (no asymmetry to
    test), True/False otherwise.
    """
    if not (target_s and natural_s and produced_s is not None):
        return None
    if abs(target_s - natural_s) < min_gap_ratio * natural_s:
        return None  # not asymmetric enough to be a meaningful direction test
    return abs(produced_s - target_s) < abs(produced_s - natural_s)


# ─────────────────────────────────────────────────────────────────────────────
# Duration / latency helpers
# ─────────────────────────────────────────────────────────────────────────────
def wav_duration_seconds(path: str) -> Optional[float]:
    """Duration of a WAV in seconds, or None if unreadable/too small. Tries soundfile
    (any format) then stdlib wave (PCM). Both are import-guarded so this module loads with
    neither installed."""
    if not path or not os.path.exists(path) or os.path.getsize(path) <= 44:
        return None
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(path)
        if info.frames and info.samplerate:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        with wave.open(path, "rb") as w:
            fr = w.getframerate()
            n = w.getnframes()
            if fr:
                return n / float(fr)
    except Exception:
        pass
    return None


def latency_stats(seconds: Sequence[float]) -> dict:
    """Aggregate per-segment synth seconds into {mean, p50, p90, max, n}."""
    vals = sorted(float(s) for s in seconds if s is not None and float(s) >= 0)
    n = len(vals)
    if n == 0:
        return {"mean": None, "p50": None, "p90": None, "max": None, "n": 0}

    def _pct(p):
        if n == 1:
            return vals[0]
        idx = min(n - 1, max(0, int(math.ceil(p / 100.0 * n)) - 1))
        return vals[idx]

    return {"mean": sum(vals) / n, "p50": _pct(50), "p90": _pct(90), "max": vals[-1], "n": n}


def read_manifest_timings(out_dir: str, n_segments: int) -> Optional[list]:
    """Read the per-segment synth times the worker stamped into its resume manifest.

    The worker (pipeline.duration_tts) records ``synth_seconds`` on each real "ok" segment
    it synthesizes — the clean GPU-synthesis time, surviving relaunch/resume because it
    lives on disk. This returns a list of length ``n_segments`` aligned by segment index:
    a float where a timed "ok" record exists, else None (empty-text silence, a placeholder
    fallback, a not-yet-synthesized segment, or a manifest written before timing existed).

    Returns None if the manifest is missing or unreadable — the caller then falls back to
    aggregate wall/n latency rather than inventing per-segment numbers (non-negotiable #8).
    """
    path = os.path.join(out_dir, _MANIFEST_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None

    timings: list = []
    for i in range(n_segments):
        entry = manifest.get(str(i))
        secs = None
        if isinstance(entry, dict) and entry.get("status") == "ok":
            v = entry.get("synth_seconds")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
                secs = float(v)
        timings.append(secs)
    return timings


# ─────────────────────────────────────────────────────────────────────────────
# Probe construction
# ─────────────────────────────────────────────────────────────────────────────
def build_probe_segments(texts_with_natural: Sequence[tuple],
                         ratios: Sequence[float] = DEFAULT_PROBE_RATIOS) -> list:
    """Expand (text, natural_seconds) items into probe segments at each ratio.

    Each output segment is {"start","end","text","natural","ratio"} where
    end-start == natural*ratio, so the harness can request a spread of budgets around each
    line's natural length and later regress produced-vs-requested. `natural_seconds` is the
    line's unconstrained duration — measure it once (a length_scale=1 / no-fix_duration
    pass) rather than guessing.
    """
    segs = []
    t = 0.0
    for text, natural in texts_with_natural:
        natural = float(natural)
        for r in ratios:
            dur = max(natural * float(r), _MIN_VALID_SECONDS)
            segs.append({"start": t, "end": t + dur, "text": text,
                         "natural": natural, "ratio": float(r)})
            t += dur + 0.5  # non-overlapping timeline; the gap is irrelevant to synth
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# CER via Whisper back-transcription (guarded; the only GPU-touching path)
# ─────────────────────────────────────────────────────────────────────────────
def back_transcribe_cer(wav_paths: Sequence[str], ref_texts: Sequence[str],
                        lang_code: Optional[str] = None, *, model_size: str = "small",
                        transcribe_fn: Optional[Callable] = None,
                        log_fn: Optional[Callable] = None) -> dict:
    """Back-transcribe each produced WAV and CER it against the intended text.

    `transcribe_fn(path, lang_code, model_size) -> str` is injectable so tests can supply a
    fake recogniser (and so the production call site can wire whichever Whisper wrapper it
    prefers). If omitted, a lazy Whisper import is attempted; if that fails the CER axis is
    reported as unavailable rather than crashing the whole scorecard.
    """
    def _log(m):
        if log_fn:
            log_fn(m)

    fn = transcribe_fn
    if fn is None:
        try:
            import whisper  # type: ignore

            _model_cache = {}

            def _fn(path, lang, size):
                if size not in _model_cache:
                    _model_cache[size] = whisper.load_model(size)
                kw = {"language": lang} if lang else {}
                return (_model_cache[size].transcribe(path, **kw) or {}).get("text", "")

            fn = _fn
        except Exception as e:  # noqa: BLE401
            _log(f"[benchmark] CER unavailable (no whisper): {e}")
            return {"mean_cer": None, "per_segment": [], "available": False, "n": 0}

    per = []
    for path, ref in zip(wav_paths, ref_texts):
        if not path or not os.path.exists(path) or os.path.getsize(path) <= 44:
            per.append({"path": path, "cer": None, "hyp": None})
            continue
        try:
            hyp = fn(path, lang_code, model_size) or ""
        except Exception as e:  # noqa: BLE401
            _log(f"[benchmark] transcribe failed for {path}: {e}")
            per.append({"path": path, "cer": None, "hyp": None})
            continue
        per.append({"path": path, "cer": compute_cer(ref, hyp), "hyp": hyp})
    cers = [p["cer"] for p in per if p["cer"] is not None]
    mean = sum(cers) / len(cers) if cers else None
    return {"mean_cer": mean, "per_segment": per, "available": True, "n": len(cers)}


# ─────────────────────────────────────────────────────────────────────────────
# Scorecard assembly + report
# ─────────────────────────────────────────────────────────────────────────────
def build_scorecard(*, candidate: str, provenance: dict, control: Optional[dict],
                    slope: dict, latency: dict, cer: dict,
                    direction_tests: Optional[list] = None) -> dict:
    """Assemble one candidate's row. `provenance` MUST carry enough to regenerate the run:
    checkpoint/model id, clip, nfe_step (or length_scale), n_samples, data_version, device.
    """
    missing = [k for k in ("candidate_source", "clip", "n_samples", "data_version", "device")
               if k not in provenance]
    prov = dict(provenance)
    if missing:
        prov["_incomplete_provenance"] = missing  # surfaced, never silently dropped (#8)
    return {
        "candidate": candidate,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provenance": prov,
        "control_signal": control,
        "length": slope,
        "latency": latency,
        "cer": cer,
        "direction_tests": direction_tests or [],
    }


def _fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_markdown(scorecards: Sequence[dict], *, title: str = "TTS candidate scorecard") -> str:
    """Human-readable comparison table across candidates. Numbers only from the scorecards
    — this function computes nothing, so the report can never drift from the data (#8)."""
    lines = [f"# {title}", "", f"_generated {datetime.now().isoformat(timespec='seconds')}_", ""]
    lines += [
        "| candidate | slope | signed err | abs err | latency mean (s) | p90 | CER | control | dir-tests | n |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sc in scorecards:
        ln, lat, ce = sc.get("length", {}), sc.get("latency", {}), sc.get("cer", {})
        ctl = sc.get("control_signal")
        ctl_s = ("n/a" if ctl is None
                 else ("reached" if ctl.get("reached_model") else "NOT REACHED"))
        dts = sc.get("direction_tests", [])
        dt_pass = sum(1 for d in dts if d.get("ok") is True)
        dt_total = sum(1 for d in dts if d.get("ok") is not None)
        dt_s = f"{dt_pass}/{dt_total}" if dt_total else "n/a"
        lines.append(
            f"| {sc.get('candidate','?')} | {_fmt(ln.get('slope'))} | "
            f"{_fmt(ln.get('mean_signed_error'))} | {_fmt(ln.get('mean_abs_error'))} | "
            f"{_fmt(lat.get('mean'))} | {_fmt(lat.get('p90'))} | {_fmt(ce.get('mean_cer'))} | "
            f"{ctl_s} | {dt_s} | {ln.get('used','?')} |"
        )
    lines += ["", "## Provenance", ""]
    for sc in scorecards:
        p = sc.get("provenance", {})
        lines.append(f"- **{sc.get('candidate','?')}** — "
                     + ", ".join(f"{k}={v}" for k, v in p.items()))
        if p.get("_incomplete_provenance"):
            lines.append(f"  - ⚠️ incomplete provenance: missing {p['_incomplete_provenance']}")
    lines += [
        "",
        "## How to read this",
        "- **slope** → 1.0 = obeys the duration budget; → 0.0 = ignores it.",
        "- **signed err** > 0 = dub runs long (needs compression); < 0 = runs short (pad — friendlier).",
        "- **control** = the duration knob was proven to reach the model (non-negotiable #1); "
        "`NOT REACHED` invalidates every other number in the row.",
        "- **dir-tests** = asymmetric-target direction checks passed (non-negotiable #5).",
        "- **CER** from Whisper back-transcription; `n/a` = Whisper was unavailable, not a pass.",
        "",
    ]
    return "\n".join(lines)


def write_reports(scorecards: Sequence[dict], out_dir: str, *, name: str = "tts_scorecard",
                  title: str = "TTS candidate scorecard") -> dict:
    """Write <name>.json and <name>.md into out_dir. Returns the two paths."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{name}.json")
    md_path = os.path.join(out_dir, f"{name}.md")
    tmp = json_path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as fh:
        json.dump(list(scorecards), fh, ensure_ascii=False, indent=2)
    os.replace(tmp, json_path)
    with io.open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(scorecards, title=title))
    return {"json": json_path, "markdown": md_path}
