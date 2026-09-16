#!/usr/bin/env python3
"""Step 6.5 — Voice conversion (timbre transfer) for the premium cloning path.

Why this exists
---------------
IndicF5 is an Indic-only F5-TTS. Conditioning it on an ENGLISH reference (audio + text)
and asking for Hindi destabilizes its flow-matching alignment across the WHOLE utterance
— the "onset babble". This was proven by a controlled A/B: native-reference TTS
(``reference=None``) is clean; English-reference cloning is garbled. The primer mitigation
(prepend a throwaway Hindi word, slice it off) did NOT fix it, because the instability is
distributed, not a bounded onset transient, and the slice cannot be placed reliably.

The correct fix decouples *content* from *timbre*:

    1. Generate CLEAN Hindi with a native reference (IndicF5, reference=None).   [duration_tts]
    2. Convert the timbre of that clean Hindi to the source speaker.            [THIS MODULE]

Voice conversion transfers speaker identity while preserving the content and — critically —
the *duration* of each frame, so the isochrony the TTS stage already achieved is untouched.
There is no cross-lingual phonetic conditioning to destabilize: the speaker embedding is
extracted from the reference audio regardless of what language it is in.

Contract
--------
``convert_segments_timbre(segment_paths, target_speaker_wav, ...)`` converts each per-segment
WAV IN PLACE so it sounds like ``target_speaker_wav``, preserving sample rate and exact
sample length (so drift-corrected isochrony survives) and matching the input's loudness.

Robustness
----------
This stage NEVER raises into the pipeline. If the backend cannot be imported/loaded, or a
single segment fails, it logs a visible WARNING and leaves that segment UNCHANGED
(pass-through) — degrade visibly, never crash a long render (project rule 5). A run where VC
silently no-op'd is reported as such in the returned status, not hidden.

Backends (``DUBBING_VC_BACKEND``)
---------------------------------
- ``knn-vc`` (default): zero-shot kNN voice conversion over WavLM features
  (torch.hub ``bshall/knn-vc``). No per-speaker training; duration-preserving. Works from a
  single short reference, though a few minutes of target audio improves timbre fidelity.
- ``seed-vc``: stronger zero-shot / cross-lingual VC from a short reference. Heavier; must be
  installed separately (``pip install seed-vc`` / repo). Selected but not present -> visible
  warning + pass-through, so a misconfig never corrupts audio.
- ``none``: explicit pass-through (useful for A/B timing without touching audio).
"""
from __future__ import annotations

import os
import time

VC_BACKEND_DEFAULT = os.environ.get("DUBBING_VC_BACKEND", "knn-vc").strip().lower()
VC_TOPK_DEFAULT = int(os.environ.get("DUBBING_VC_TOPK", "4"))


def _resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _rms(x):
    import numpy as np
    x = np.asarray(x, dtype="float64")
    return float((x * x).mean() ** 0.5) if x.size else 0.0


def _match_loudness_and_length(out_wav, in_wav):
    """Scale `out_wav` to `in_wav`'s RMS and force it to `in_wav`'s exact sample length.

    Length is forced (pad with zeros / trim) so the drift-corrected isochrony from the TTS
    stage is preserved to the sample. Loudness is matched so the assembled mix is seamless.
    A true-peak guard prevents clipping after the gain.
    """
    import numpy as np
    out = np.asarray(out_wav, dtype="float32").reshape(-1)
    ref = np.asarray(in_wav, dtype="float32").reshape(-1)

    tgt_rms = _rms(ref)
    cur_rms = _rms(out)
    if cur_rms > 1e-8 and tgt_rms > 1e-8:
        out = out * (tgt_rms / cur_rms)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = out * (0.999 / peak)

    n = ref.shape[0]
    if out.shape[0] < n:
        out = np.pad(out, (0, n - out.shape[0]))
    elif out.shape[0] > n:
        out = out[:n]
    return out.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────
class _KNNVCBackend:
    """Zero-shot kNN-VC over WavLM features (bshall/knn-vc). Duration-preserving, 16 kHz."""

    name = "knn-vc"
    native_sr = 16000

    def __init__(self, device: str, topk: int, log):
        import torch
        self._torch = torch
        log(f"    [VC] Loading knn-vc (WavLM + HiFiGAN) on {device} via torch.hub...")
        # prematched=True uses the prematched HiFiGAN vocoder — best quality for kNN-VC.
        self.model = torch.hub.load(
            "bshall/knn-vc", "knn_vc",
            prematched=True, trust_repo=True, pretrained=True, device=device,
        )
        self.device = device
        self.topk = topk
        self._matching_cache = {}
        log("    [VC] knn-vc ready.")

    def _matching_set(self, target_speaker_wav: str):
        ms = self._matching_cache.get(target_speaker_wav)
        if ms is None:
            # A single reference clip is accepted; more target audio -> better timbre.
            ms = self.model.get_matching_set([target_speaker_wav])
            self._matching_cache[target_speaker_wav] = ms
        return ms

    def convert(self, src_wav_path: str, target_speaker_wav: str):
        """Return (audio_float32_mono, sample_rate)."""
        query = self.model.get_features(src_wav_path)
        ms = self._matching_set(target_speaker_wav)
        out = self.model.match(query, ms, topk=self.topk)  # 16 kHz tensor
        return out.detach().cpu().numpy().reshape(-1), self.native_sr


class _SeedVCBackend:
    name = "seed-vc"

    def __init__(self, device: str, topk: int, log):
        # Intentionally not implemented inline: seed-vc's API/weights are installed out of band.
        # Raising here makes the supervisor/stage fall back to visible pass-through rather than
        # guessing an interface. Wire this once seed-vc is chosen for validation.
        raise ImportError(
            "seed-vc backend selected but no adapter is wired. Install seed-vc and implement "
            "_SeedVCBackend.convert(), or set DUBBING_VC_BACKEND=knn-vc."
        )

    def convert(self, src_wav_path, target_speaker_wav):  # pragma: no cover
        raise NotImplementedError


def _load_backend(backend: str, device: str, topk: int, log):
    backend = (backend or VC_BACKEND_DEFAULT).lower()
    if backend in ("none", "off", ""):
        return None
    if backend == "knn-vc":
        return _KNNVCBackend(device, topk, log)
    if backend == "seed-vc":
        return _SeedVCBackend(device, topk, log)
    raise ValueError(f"Unknown DUBBING_VC_BACKEND={backend!r} (expected knn-vc | seed-vc | none)")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def convert_segments_timbre(
    segment_paths,
    target_speaker_wav,
    backend: str | None = None,
    device: str = "auto",
    topk: int | None = None,
    log_fn=None,
):
    """Convert each WAV in `segment_paths` to sound like `target_speaker_wav`, in place.

    Args:
        segment_paths: ordered list of per-segment WAV paths (written by the TTS stage).
        target_speaker_wav: reference clip of the speaker to clone (any language).
        backend: DUBBING_VC_BACKEND override ("knn-vc" | "seed-vc" | "none").
        device: "auto" | "cuda" | "cpu".
        topk: kNN neighbours (knn-vc only).
        log_fn: logger; falls back to print.

    Returns:
        dict {"backend", "converted": [paths], "passthrough": [paths], "failed": [paths]}.
        Always returns; never raises. Sample rate and exact length of each WAV are preserved.
    """
    log = log_fn or (lambda m="": print(m, flush=True))
    import numpy as np
    import soundfile as sf

    result = {"backend": None, "converted": [], "passthrough": [], "failed": []}
    paths = [p for p in (segment_paths or []) if p and os.path.exists(p)]
    if not paths:
        log("    [VC] No segment WAVs to convert.")
        return result

    if not target_speaker_wav or not os.path.exists(target_speaker_wav):
        log(f"    [VC] WARNING: target speaker reference missing ({target_speaker_wav!r}); "
            f"leaving {len(paths)} segment(s) unchanged (pass-through).")
        result["passthrough"] = list(paths)
        return result

    device = _resolve_device(device)
    topk = VC_TOPK_DEFAULT if topk is None else topk
    backend_name = (backend or VC_BACKEND_DEFAULT).lower()
    result["backend"] = backend_name

    if backend_name in ("none", "off", ""):
        log("    [VC] DUBBING_VC_BACKEND=none — pass-through (no timbre transfer).")
        result["passthrough"] = list(paths)
        return result

    # Load the backend once. A load failure degrades the WHOLE stage to pass-through
    # (visible), rather than aborting the render.
    try:
        eng = _load_backend(backend_name, device, topk, log)
    except Exception as e:
        log(f"    [VC] WARNING: could not load backend {backend_name!r}: {e}. "
            f"Leaving {len(paths)} segment(s) unchanged (pass-through).")
        result["passthrough"] = list(paths)
        return result

    log(f"    [VC] Converting {len(paths)} segment(s) -> speaker in "
        f"{os.path.basename(target_speaker_wav)} (backend={eng.name}, device={device}).")
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            in_wav, in_sr = sf.read(p, dtype="float32", always_2d=False)
            if getattr(in_wav, "ndim", 1) > 1:
                in_wav = in_wav.mean(axis=1)

            out_wav, out_sr = eng.convert(p, target_speaker_wav)

            # Resample VC output back to the segment's own sample rate (duration preserved).
            if out_sr != in_sr:
                import torch
                t = torch.from_numpy(np.asarray(out_wav, dtype="float32")).unsqueeze(0)
                import torchaudio
                t = torchaudio.functional.resample(t, out_sr, in_sr)
                out_wav = t.squeeze(0).cpu().numpy()

            out_wav = _match_loudness_and_length(out_wav, in_wav)
            sf.write(p, out_wav, in_sr)
            result["converted"].append(p)
            log(f"    [VC] [{i + 1}/{len(paths)}] {os.path.basename(p)} converted.")
        except Exception as e:
            log(f"    [VC] [{i + 1}/{len(paths)}] WARNING: {os.path.basename(p)} failed: {e}. "
                f"Left unchanged.")
            result["failed"].append(p)

    log(f"    [VC] Done in {time.time() - t0:.1f}s — "
        f"{len(result['converted'])} converted, {len(result['failed'])} failed, "
        f"{len(result['passthrough'])} pass-through.")
    return result
