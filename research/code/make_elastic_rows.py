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

# One call carries several SENTENCES, each needing several scales. This is not a
# micro-optimisation: the free tier is metered in requests per day, not tokens, so the
# number of calls is the binding constraint. 3,167 groups one-per-call is 3,167 requests —
# four days at flash-lite's ~1,000 RPD. Five sentences per call is ~634 requests, which
# finishes inside a single day's quota on any of the candidate models.
INSTRUCTION = """You rewrite sentences in {language} to be longer or shorter while keeping \
their meaning.

You will be given {n} numbered {language} sentences. For EACH one, produce {k} rewrites at \
these lengths relative to the original:
{spec}

Rules:
- Stay in {language}. Same script. Never answer in English.
- Preserve the meaning. Do not drop facts, names, numbers or negation to hit a length.
- To shorten: prefer shorter synonyms, drop redundancy, tighten grammar. Do NOT truncate
  the sentence or leave it unfinished.
- To lengthen: add natural elaboration that is already implied. Do not invent new facts.
- Natural, fluent sentences a native speaker would actually say.

Sentences:
{items}

Return ONLY a JSON array, no prose and no code fence. One object per rewrite, so
{n} x {k} = {total} objects:
[{{"id": <sentence number>, "scale": <the requested scale>, "text": "<rewrite>"}}]"""


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


def dominant_script(text: str) -> tuple[Optional[str], float]:
    """Which Unicode script is this text written in, and how purely?

    Derived from the text rather than a hardcoded range table: Unicode character names
    begin with their script ("DEVANAGARI LETTER KA"), so the reference translation defines
    what the rewrite is allowed to look like. Nothing to keep in sync per language.
    """
    import unicodedata
    counts: collections.Counter = collections.Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            counts[unicodedata.name(ch).split()[0]] += 1
        except ValueError:
            continue
    if not counts:
        return None, 0.0
    script, n = counts.most_common(1)[0]
    return script, n / sum(counts.values())


MIN_SCRIPT_PURITY = 0.90


def verify(lang: str, ref: str, eng: str, natural: int, text: str, scale: float,
           gate: Optional[Gate], model: str, thresholds: dict,
           reject: Optional[collections.Counter] = None) -> Optional[dict]:
    """The three gates. All ours; the generator's opinion does not enter here.

    `reject` accumulates WHY candidates were dropped. Without it, a low yield is just a
    number and the obvious response is to loosen a threshold — which is how a gate quietly
    stops gating. With it, "the model drifts out of script" and "the model cannot hit the
    length" are distinguishable, and they call for different fixes.
    """
    def no(reason: str):
        if reject is not None:
            reject[reason] += 1
        return None

    text = (text or "").strip()
    if not text:
        return no("empty")
    if text == ref:
        return no("identical to source")

    # GATE 0 — is it even in the right script?
    #
    # This is not paranoia. espeak will happily phonemize Latin text through an Assamese
    # voice and return a plausible count, so a model that drifts into English — or into
    # Hindi when asked for Odia, which is a real failure mode for low-resource targets —
    # can sail straight through the length gate. Cheapest gate, checked first.
    ref_script, _ = dominant_script(ref)
    cand_script, purity = dominant_script(text)
    if ref_script and cand_script != ref_script:
        return no(f"wrong script ({cand_script} for {ref_script})")
    if ref_script and purity < MIN_SCRIPT_PURITY:
        return no("script not pure enough (code-mixed)")

    # GATE 1 — did it land at the requested length? Measured with the exact counter.
    try:
        n = count_phonemes(text, lang)
    except Exception:  # noqa: BLE001
        return no("phonemization failed")
    if not n:
        return no("phonemization empty")
    want = max(1, round(natural * scale))
    err = (n - want) / want
    if abs(err) > SCALE_TOLERANCE:
        return no("missed length: too long" if err > 0 else "missed length: too short")
    change = (n - natural) / natural
    wanted_dir = -1 if scale < 1 else 1
    if wanted_dir * change < MIN_LENGTH_CHANGE:
        return no("moved too little from the source")

    # GATE 2 — or did it hit the length by deleting content? Threshold is PER LANGUAGE:
    # measured on Kaggle 2026-08-05, a true Tamil paraphrase scored 0.556 where the
    # equivalent Hindi pair scored 0.997, so one global 0.80 would have starved Tamil of
    # exactly the rows it most needs.
    sim = None
    if gate is not None:
        sim = gate.similarity(ref, text)
        if sim < thresholds.get(lang, MIN_SIMILARITY):
            return no("semantic below threshold")

    lang_name = get_language(lang).name
    return {
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
    }


def build_batch(client, model: str, gate: Optional[Gate], rows: list[dict],
                scales: list[float], thresholds: dict,
                reject: Optional[collections.Counter] = None) -> list[dict]:
    """Several sentences of ONE language -> verified rewrites at several lengths."""
    if not rows:
        return []
    lang = rows[0]["language"]
    lang_name = get_language(lang).name

    prepared = []
    for r in rows:
        ref = r.get("completion") or r.get("target") or ""
        eng = r.get("english") or ""
        if not ref or not eng:
            continue
        try:
            natural = count_phonemes(ref, lang)
        except Exception:  # noqa: BLE001
            continue
        if natural:
            prepared.append((eng, ref, natural))
    if not prepared:
        return []

    spec = "\n".join(
        f"  - scale {s}: about {abs(1 - s) * 100:.0f}% "
        f"{'shorter' if s < 1 else 'longer'} than the original"
        for s in scales)
    items = "\n".join(f"{i + 1}. {ref}" for i, (_, ref, _) in enumerate(prepared))
    prompt = INSTRUCTION.format(language=lang_name, n=len(prepared), k=len(scales),
                                total=len(prepared) * len(scales), spec=spec, items=items)

    try:
        cands = call_model(client, model, prompt)
    except Exception as e:  # noqa: BLE001
        logger.debug("%s: generation failed: %s", lang, e)
        return []

    out = []
    for c in cands:
        try:
            idx = int(c.get("id")) - 1
            scale = float(c.get("scale"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(prepared)):
            continue
        eng, ref, natural = prepared[idx]
        row = verify(lang, ref, eng, natural, c.get("text"), scale, gate, model,
                     thresholds, reject)
        if row:
            out.append(row)
    if reject is not None:
        # Rewrites the model simply did not return are a distinct failure from ones it
        # returned badly, and only shows up as a gap in the arithmetic.
        missing = len(prepared) * len(scales) - len(cands)
        if missing > 0:
            reject["not returned by the model"] += missing
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
    p.add_argument("--sentences_per_call", type=int, default=5,
                   help="The free tier meters REQUESTS per day, so this is the knob that "
                        "decides whether the job fits in one day's quota.")
    p.add_argument("--thresholds", default=None,
                   help="JSON {lang: min_similarity} from the calibration run. Without it "
                        "every language uses the global 0.80, which is known to be wrong "
                        "for Tamil.")
    p.add_argument("--list_models", action="store_true",
                   help="Ask the account which models it can actually use, and exit. "
                        "Ground truth beats a blog post about rate limits.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_semantic_gate", action="store_true",
                   help="Skip gate 2. Only for a plumbing test — never for real data.")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    if args.list_models:
        client = make_client()
        rows = []
        for m in client.models.list():
            methods = list(getattr(m, "supported_actions", None)
                           or getattr(m, "supported_generation_methods", None) or [])
            if "generateContent" not in methods:
                continue
            rows.append((m.name.replace("models/", ""),
                         getattr(m, "input_token_limit", "") or "",
                         getattr(m, "output_token_limit", "") or ""))
        print(f"\n{len(rows)} models on this account support generateContent:\n")
        print(f"{'model id (use as --model)':<48}{'in-tokens':>12}{'out-tokens':>12}")
        print("-" * 72)
        for name, i, o in sorted(rows):
            print(f"{name:<48}{i:>12}{o:>12}")
        print("\nRate limits are per-account and are not exposed by the API — check them at")
        print("  https://aistudio.google.com/app/rate-limit")
        return 0

    assert_g2p_available()
    logger.info("counter ready — ruler=%s", ruler_id())
    scales = args.scales or TARGET_SCALES
    thresholds = (json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
                  if args.thresholds else {})
    if thresholds:
        logger.info("per-language similarity thresholds: %s", thresholds)
    else:
        logger.warning("no --thresholds: every language uses the global %.2f. The Kaggle "
                       "calibration showed this is wrong for at least Tamil.",
                       MIN_SIMILARITY)

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

    # Batches never mix languages: one instruction, one script, one set of rules.
    rnd = random.Random(args.seed)
    batches: list[list[dict]] = []
    n_sentences = 0
    for lg in wanted_langs:
        pool = by_lang.get(lg, [])
        need = per_lang_target.get(lg, 0)
        if need <= 0 or not pool:
            continue
        rnd.shuffle(pool)
        chosen = pool[:need]
        n_sentences += len(chosen)
        step = max(1, args.sentences_per_call)
        batches += [chosen[i:i + step] for i in range(0, len(chosen), step)]
    if not batches:
        logger.info("nothing to do — the plan is already satisfied")
        return 0

    rnd.shuffle(batches)   # spread languages over time so a rate-limit stall is not all one
    logger.info("%d sentences in %d requests (%d per call), %d scales each, model=%s",
                n_sentences, len(batches), args.sentences_per_call, len(scales), args.model)
    logger.info("free-tier RPD is the binding constraint — this job needs %d requests",
                len(batches))

    client = make_client()
    gate = None if args.no_semantic_gate else Gate()
    if gate is None:
        logger.warning("semantic gate DISABLED — plumbing test only, do not train on this")

    written = 0
    stats = collections.Counter()
    reject = collections.Counter()
    lock = threading.Lock()
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_batch, client, args.model, gate, b, scales,
                               thresholds, reject): b for b in batches}
        for i, fut in enumerate(as_completed(futures), 1):
            src = futures[fut]
            lg = src[0].get("language") if src else "?"
            try:
                got = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.error("%s: %s", lg, e)
                got = []
            with lock:
                seen_english = set()
                for g in got:
                    sink.write(json.dumps(g, ensure_ascii=False) + "\n")
                    stats[(g["language"], g["augmentation"]["direction"])] += 1
                    seen_english.add(g["english"])
                    written += 1
                stats[("groups", lg)] += len(seen_english)
                stats[("attempted", lg)] += len(src)
                sink.flush()
            if i % 20 == 0 or i == len(batches):
                rate = i / max(1e-6, time.time() - t0)
                logger.info("%d/%d requests  %d rows  %.2f req/s  eta %.0fs",
                            i, len(batches), written, rate,
                            (len(batches) - i) / max(rate, 1e-6))

    print(f"\n{written} verified rows written to {out_path}")
    print(f"{'lang':<6}{'tried':>7}{'groups':>8}{'compress':>10}{'expand':>8}"
          f"{'rows/try':>10}")
    print("-" * 49)
    for lg in wanted_langs:
        a = stats[("attempted", lg)]
        g = stats[("groups", lg)]
        c = stats[(lg, "compress")]
        e = stats[(lg, "expand")]
        if a or g:
            # rows-per-attempted is the number to watch: a language whose yield collapses
            # is one where the gates are rejecting nearly everything, and that is a finding
            # about the language, not a throughput problem to tune away.
            print(f"{lg:<6}{a:>7}{g:>8}{c:>10}{e:>8}{((c + e) / a if a else 0):>10.2f}")
    tot_c = sum(v for k, v in stats.items() if k[1] == "compress")
    tot_e = sum(v for k, v in stats.items() if k[1] == "expand")
    tot = tot_c + tot_e
    print("-" * 45)
    print(f"{'TOTAL':<6}{'':>8}{tot_c:>10}{tot_e:>8}")
    if tot:
        print(f"\ncompression share: {tot_c / tot:.0%}  (gate floor 40%)")
    if reject:
        total_rej = sum(reject.values())
        print(f"\n{total_rej} candidates rejected. A low yield is only actionable if you "
              f"know which gate did the rejecting:")
        for reason, n in reject.most_common():
            print(f"    {n:>6}  {n / total_rej:>5.0%}  {reason}")
    print(f"\nEverything above passed every gate: right script, landed within "
          f"{SCALE_TOLERANCE:.0%} of the requested length, moved at least "
          f"{MIN_LENGTH_CHANGE:.0%} from the source"
          + ("." if gate is None else ", and stayed above the language's similarity "
                                      "threshold."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
