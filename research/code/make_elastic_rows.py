"""
tools/make_elastic_rows.py
===========================
Manufactures the training signal that no parallel corpus can contain.

THE PROBLEM, IN ONE PARAGRAPH
------------------------------
Samanantar gives one Indic translation per English sentence. The phoneme budget written
into a training prompt is therefore that translation's own length — a deterministic
function of the input. On such a row, "ignore the number and translate naturally" is a
completely correct answer, so gradient descent never has a reason to prefer a model that
reads the budget. Measured on the corpus that trained checkpoint 3801: 0.2% of rows sit in
a group where one English sentence appears with more than one budget. Malayalam's budget
obedience moved 0.419 -> 0.418 across 3,801 steps.

The fix is to make the number load-bearing: the SAME English sentence, seen with SEVERAL
budgets and several correct answers of matching length.

WHY A PARAPHRASE AND NOT A RE-TRANSLATION
------------------------------------------
We ask the generator to rewrite the *existing reference translation* at a different length,
not to translate the English afresh. Two reasons. It is a far more constrained task, so it
succeeds more often. And it keeps the semantic anchor on human-quality text rather than on
the generator's own translation habits, which is what stops the corpus slowly turning into
an imitation of whatever model built it.

WHY WE DO NOT TRUST THE GENERATOR
----------------------------------
An LLM cannot count phonemes; asking for "37 phonemes" produces confident nonsense. So we
never ask for an exact count. We ask for a proportional change, take several candidates,
and then measure them ourselves with the exact counter. Two gates, both ours:

  LENGTH    the candidate actually landed near the target length
  MEANING   cosine similarity to the ORIGINAL reference is above 0.80, the same threshold
            and the same embedder the production gate uses

Anything failing either gate is discarded, not down-weighted. This is the rare case where
the objective is exactly checkable, and the whole design leans on it: the generator is a
proposal mechanism, and the verifier is ours.

CREDENTIALS
-----------
Reads `GEMINI_API_KEY` from the environment. Set it in your own shell; it is never passed
on the command line, never logged, and never written to the output.

    $env:GEMINI_API_KEY = "..."      # PowerShell
    export GEMINI_API_KEY="..."      # bash

USAGE
-----
    python -m tools.make_elastic_rows --corpus data/train.phonemes.jsonl \\
        --plan research/elastic_corpus_plan.json --out data/elastic_rows.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.languages import LANGUAGES, get_language  # noqa: E402
from common.phonemes import assert_g2p_available, count_phonemes, ruler_id  # noqa: E402

logger = logging.getLogger("make_elastic")

DEFAULT_MODEL = "gemini-3.1-flash-lite"   # what utils/translation.py already uses

# Asked for in one call, verified individually. Wider than the training sweep on purpose:
# the extremes are where the five stuck languages stopped responding.
TARGET_SCALES = [0.60, 0.75, 1.30]

MIN_SIMILARITY = 0.80        # production's gate, same embedder
SCALE_TOLERANCE = 0.15       # landed within 15% of the requested scale
MIN_LENGTH_CHANGE = 0.10     # and moved at least 10% from the original, in the right way

PROMPT_TEMPLATE = '[Translate to {language}] [Target Phonemes: {n}] "{english}"'

INSTRUCTION = """You rewrite sentences in {language} to be longer or shorter while keeping \
their meaning.

Original {language} sentence:
{target}

Its English source (for meaning only — do not translate it again):
{english}

Produce {k} rewrites of the ORIGINAL {language} SENTENCE, each at a different length:
{spec}

Rules:
- Stay in {language}. Same script. Never answer in English.
- Preserve the meaning. Do not drop facts, names, numbers or negation to hit a length.
- To shorten: prefer shorter synonyms, drop redundancy, tighten grammar. Do not truncate.
- To lengthen: add natural elaboration that is already implied. Do not invent new facts.
- Natural, fluent sentences a native speaker would say.

Return ONLY a JSON array of {k} objects, no prose, no code fence:
[{{"scale": <the requested scale>, "text": "<rewrite>"}}]"""


EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Gate:
    """The semantic half of the verification. Loaded once, shared across threads.

    Implemented on `transformers` rather than `sentence_transformers`, because importing
    the latter segfaults on this machine while torch and transformers both import cleanly.
    For this model that is not a compromise: a SentenceTransformer here is exactly an
    AutoModel followed by attention-masked mean pooling and L2 normalisation, which is what
    `_embed` does. Same weights, same arithmetic, one fewer dependency that can break.
    `sentence_transformers` is still preferred when it is importable, so the Kaggle path is
    unchanged.
    """

    def __init__(self, device: Optional[str] = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._st = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(EMBEDDER, device=self.device)
            logger.info("semantic gate: sentence-transformers on %s", self.device)
            return
        except Exception as e:  # noqa: BLE001
            logger.info("sentence-transformers unavailable (%s) — using transformers "
                        "directly", type(e).__name__)
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(EMBEDDER)
        self.model = AutoModel.from_pretrained(EMBEDDER).to(self.device).eval()
        logger.info("semantic gate: transformers on %s", self.device)

    def _embed(self, texts: list[str]):
        torch = self.torch
        batch = self.tok(texts, padding=True, truncation=True, max_length=256,
                         return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def similarity(self, a: str, b: str) -> float:
        with self._lock:
            if self._st is not None:
                emb = self._st.encode([a, b], convert_to_tensor=True,
                                      normalize_embeddings=True, show_progress_bar=False)
            else:
                emb = self._embed([a, b])
        return float((emb[0] * emb[1]).sum().item())


def make_client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY is not set.\n"
            "  PowerShell:  $env:GEMINI_API_KEY = '...'\n"
            "  bash:        export GEMINI_API_KEY='...'\n"
            "The key is read from the environment only — never pass it as an argument.")
    from google import genai
    # NOTE: utils/translation.py disables certificate verification to work around this
    # machine's antivirus TLS interception. That is not done here. `truststore` is injected
    # interpreter-wide (see D:/Miniconda/Lib/site-packages/zz_os_trust_store.pth), which
    # routes verification through the OS certificate store and keeps it ON.
    return genai.Client(api_key=key)


_JSON = re.compile(r"\[.*\]", re.S)


def call_model(client, model: str, prompt: str, attempts: int = 5) -> list[dict]:
    last = None
    for i in range(attempts):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            txt = (resp.text or "").strip()
            m = _JSON.search(txt)
            if not m:
                raise ValueError(f"no JSON array in response: {txt[:120]!r}")
            return json.loads(m.group(0))
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e)
            transient = any(c in msg for c in
                            ("429", "503", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                             "overloaded", "deadline"))
            if i + 1 < attempts:
                # Jittered backoff. Without the jitter a thread pool that hits a rate limit
                # retries in lockstep and hits it again.
                time.sleep((2 ** i) + random.random() * (2.0 if transient else 0.5))
            else:
                raise
    raise RuntimeError(str(last))


def build_group(client, model: str, gate: Optional[Gate], row: dict,
                scales: list[float]) -> list[dict]:
    """One English sentence -> verified rewrites at several lengths."""
    lang = row["language"]
    lang_name = get_language(lang).name
    ref = row.get("completion") or row.get("target") or ""
    eng = row.get("english") or ""
    if not ref or not eng:
        return []

    natural = count_phonemes(ref, lang)
    if not natural:
        return []

    spec = "\n".join(
        f"  - scale {s}: about {abs(1 - s) * 100:.0f}% "
        f"{'shorter' if s < 1 else 'longer'} than the original"
        for s in scales)
    prompt = INSTRUCTION.format(language=lang_name, target=ref, english=eng,
                                k=len(scales), spec=spec)

    try:
        cands = call_model(client, model, prompt)
    except Exception as e:  # noqa: BLE001
        logger.debug("%s: generation failed: %s", lang, e)
        return []

    out = []
    for c in cands:
        text = (c.get("text") or "").strip()
        try:
            scale = float(c.get("scale"))
        except (TypeError, ValueError):
            continue
        if not text or text == ref:
            continue

        # GATE 1 — did it actually land at the requested length? Measured by us.
        try:
            n = count_phonemes(text, lang)
        except Exception:  # noqa: BLE001
            continue
        if not n:
            continue
        want = max(1, round(natural * scale))
        if abs(n - want) / want > SCALE_TOLERANCE:
            continue
        change = (n - natural) / natural
        wanted_dir = -1 if scale < 1 else 1
        if wanted_dir * change < MIN_LENGTH_CHANGE:
            continue        # "shorter" that is barely shorter teaches nothing

        # GATE 2 — did it keep the meaning, or hit the length by deleting content?
        sim = None
        if gate is not None:
            sim = gate.similarity(ref, text)
            if sim < MIN_SIMILARITY:
                continue

        out.append({
            "language": lang, "english": eng, "target": text, "completion": text,
            "n_phonemes": n, "ruler": ruler_id(),
            "prompt": PROMPT_TEMPLATE.format(language=lang_name, n=n, english=eng),
            "augmentation": {
                "direction": "compress" if scale < 1 else "expand",
                "requested_scale": scale,
                "source_phonemes": natural,
                "relative_change": round(change, 4),
                "similarity": round(sim, 4) if sim is not None else None,
                "generator": model,
            },
        })
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="Base corpus to draw sentences from.")
    p.add_argument("--plan", default=None,
                   help="tools/plan_elastic_corpus.py --json output: how many per language.")
    p.add_argument("--groups_per_lang", type=int, default=None,
                   help="Override the plan with a flat number (useful for a dry run).")
    p.add_argument("--out", required=True, help="JSONL, appended to and resumable.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--scales", nargs="*", type=float, default=None)
    p.add_argument("--languages", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no_semantic_gate", action="store_true",
                   help="Skip gate 2. Only for a plumbing test — never for real data.")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    assert_g2p_available()
    logger.info("counter ready — ruler=%s", ruler_id())
    scales = args.scales or TARGET_SCALES

    # Resume: never regenerate a sentence we already have rows for.
    out_path = Path(args.out)
    done: set[tuple] = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r.get("language"), r.get("english")))
        logger.info("resuming — %d sentences already have rows", len(done))

    rows = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_lang_target: dict[str, int] = {}
    if args.groups_per_lang is not None:
        per_lang_target = {lg: args.groups_per_lang for lg in LANGUAGES}
    elif args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        per_lang_target = {lg: v["build_groups"]
                           for lg, v in plan["per_language"].items()}
    else:
        raise SystemExit("pass --plan or --groups_per_lang")

    wanted_langs = args.languages or sorted(LANGUAGES)
    by_lang: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        lg = r.get("language")
        if lg in wanted_langs and (lg, r.get("english")) not in done:
            by_lang[lg].append(r)

    rnd = random.Random(args.seed)
    todo: list[dict] = []
    for lg in wanted_langs:
        pool = by_lang.get(lg, [])
        need = per_lang_target.get(lg, 0)
        if need <= 0 or not pool:
            continue
        rnd.shuffle(pool)
        todo += pool[:need]
    if not todo:
        logger.info("nothing to do — the plan is already satisfied")
        return 0

    logger.info("%d sentences to expand, %d scales each, model=%s",
                len(todo), len(scales), args.model)

    client = make_client()
    gate = None if args.no_semantic_gate else Gate()
    if gate is None:
        logger.warning("semantic gate DISABLED — plumbing test only, do not train on this")

    written = 0
    stats = collections.Counter()
    lock = threading.Lock()
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_group, client, args.model, gate, r, scales): r
                   for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            src = futures[fut]
            try:
                got = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.error("%s: %s", src.get("language"), e)
                got = []
            with lock:
                for g in got:
                    sink.write(json.dumps(g, ensure_ascii=False) + "\n")
                    stats[(g["language"], g["augmentation"]["direction"])] += 1
                    written += 1
                if got:
                    stats[("groups", src["language"])] += 1
                sink.flush()
            if i % 50 == 0 or i == len(todo):
                rate = i / max(1e-6, time.time() - t0)
                logger.info("%d/%d sentences  %d rows  %.1f/s  eta %.0fs",
                            i, len(todo), written, rate, (len(todo) - i) / max(rate, 1e-6))

    print(f"\n{written} verified rows written to {out_path}")
    print(f"{'lang':<6}{'groups':>8}{'compress':>10}{'expand':>8}{'yield/group':>13}")
    print("-" * 45)
    for lg in wanted_langs:
        g = stats[("groups", lg)]
        c = stats[(lg, "compress")]
        e = stats[(lg, "expand")]
        if g or c or e:
            print(f"{lg:<6}{g:>8}{c:>10}{e:>8}{((c + e) / g if g else 0):>13.2f}")
    tot_c = sum(v for k, v in stats.items() if k[1] == "compress")
    tot_e = sum(v for k, v in stats.items() if k[1] == "expand")
    tot = tot_c + tot_e
    print("-" * 45)
    print(f"{'TOTAL':<6}{'':>8}{tot_c:>10}{tot_e:>8}")
    if tot:
        print(f"\ncompression share: {tot_c / tot:.0%}  (gate floor 40%)")
    print("Everything above passed BOTH gates: landed within "
          f"{SCALE_TOLERANCE:.0%} of the requested length, and stayed above "
          f"{MIN_SIMILARITY} similarity to the original reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
