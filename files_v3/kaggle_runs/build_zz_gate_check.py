"""
Builds files_v3/kaggle_runs/zz_rough_gate_check.ipynb — a THROWAWAY notebook.

Purpose: verify that the semantic gate actually separates meaning, in Indic script, with
the 0.80 threshold falling between a true paraphrase and a meaning-changing edit. The local
machine segfaults on the embedder (exit 139) whether loaded through sentence-transformers
or transformers directly, so the check has to happen somewhere the stack works.

CPU only — the embedder is 118M parameters. No GPU quota is consumed. No secrets needed.
Delete the notebook from Kaggle once the gate is confirmed.

    python files_v3/kaggle_runs/build_zz_gate_check.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "files_v3" / "kaggle_runs" / "zz_rough_gate_check.ipynb"

CASES = r'''
# The cases. Each is a pair plus what the gate MUST say about it.
#
# This is deliberately not "does encode() return a vector". That kind of check is what let
# a broken phonemizer through for an entire corpus. The gate has one job — tell a
# length-preserving rewrite apart from a meaning-changing one — so that is what is tested,
# in the scripts it will actually see, with the threshold that will actually be used.
CASES = [
    ("hi  paraphrase, shorter",
     "मुझे यह फ़िल्म बहुत ज़्यादा पसंद आई थी।",
     "मुझे यह फ़िल्म बहुत पसंद आई।", True),

    ("hi  NEGATION FLIPPED",
     "मुझे यह फ़िल्म बहुत पसंद आई।",
     "मुझे यह फ़िल्म पसंद नहीं आई।", False),

    ("hi  unrelated sentence",
     "मुझे यह फ़िल्म बहुत पसंद आई।",
     "कल दिल्ली में बहुत बारिश हुई।", False),

    ("ta  paraphrase, shorter",
     "அவர் நேற்று மாலை வீட்டிற்குத் திரும்பி வந்தார்.",
     "அவர் நேற்று வீடு திரும்பினார்.", True),

    ("or  paraphrase, shorter",
     "ଏହା ପରେ ସେ ଘଟଣା ପୁରୁଣା ହୋଇଗଲା।",
     "ପରେ ଘଟଣା ପୁରୁଣା ହେଲା।", True),

    ("ml  truncated, point dropped",
     "അദ്ദേഹം ഇന്നലെ വൈകുന്നേരം വീട്ടിലേക്ക് മടങ്ങി വന്നു.",
     "അദ്ദേഹം ഇന്നലെ", False),

    ("te  paraphrase, longer",
     "అతను నిన్న ఇంటికి వచ్చాడు.",
     "అతను నిన్న సాయంత్రం తన ఇంటికి తిరిగి వచ్చాడు.", True),
]
'''

GATE = r'''
EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.80

import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(EMBEDDER)
mdl = AutoModel.from_pretrained(EMBEDDER).eval()

def embed(texts):
    """Exactly what a SentenceTransformer does for this model: attention-masked mean
    pooling over the last hidden state, then L2 normalisation."""
    b = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)

def similarity(a, b):
    e = embed([a, b])
    return float((e[0] * e[1]).sum())
'''

CHECK = r'''
print(f"\n{'case':<30}{'sim':>8}   gate says      expected")
print("-" * 72)
ok = True
rows = []
for label, a, b, expect_high in CASES:
    s = similarity(a, b)
    passes = s >= THRESHOLD
    good = (passes == expect_high)
    ok &= good
    rows.append({"case": label, "similarity": round(s, 4),
                 "gate_passes": passes, "expected": expect_high, "correct": good})
    print(f"{label:<30}{s:>8.3f}   "
          f"{'ACCEPT' if passes else 'REJECT':<14} "
          f"{'ACCEPT' if expect_high else 'REJECT':<8} {'' if good else '<-- WRONG'}")
print("-" * 72)

# Margin matters as much as direction: a gate that ranks correctly but puts every pair
# within 0.02 of the threshold will misclassify real data constantly.
acc = [r["similarity"] for r in rows if r["expected"]]
rej = [r["similarity"] for r in rows if not r["expected"]]
print(f"\nshould-accept: min {min(acc):.3f}   should-reject: max {max(rej):.3f}")
print(f"margin around the {THRESHOLD} threshold: {min(acc) - max(rej):+.3f}")

import json
json.dump({"ok": bool(ok), "threshold": THRESHOLD, "embedder": EMBEDDER,
           "margin": round(min(acc) - max(rej), 4), "cases": rows},
          open("/kaggle/working/GATE_CHECK.json", "w"), indent=2, ensure_ascii=False)

if ok and min(acc) - max(rej) > 0:
    print("\nGATE VERIFIED — it separates meaning, and 0.80 falls in the gap.")
else:
    print("\nGATE FAILED — do not build a corpus with it. Either the threshold is wrong "
          "for these languages or the embedder is not doing what we assume.")
    raise SystemExit(1)
'''


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def build() -> None:
    cells = [
        md("""# zz — rough gate check (THROWAWAY, delete after use)

Verifies that the semantic gate separates meaning in Indic scripts and that the 0.80
threshold falls between a true paraphrase and a meaning-changing edit.

This exists because the local machine segfaults (exit 139) on the embedder, through both
`sentence_transformers` and `transformers` directly, while torch and transformers import
fine on their own. The gate is the only thing standing between "compress this sentence"
and "delete words until the number matches", so it does not get taken on trust.

**CPU only.** The embedder is 118M parameters; no GPU, no quota, no secrets.
Delete this notebook once `GATE_CHECK.json` says VERIFIED."""),
        code("!pip -q install -U transformers sentence-transformers 2>&1 | tail -2"),
        code(CASES),
        code(GATE),
        code(CHECK),
        md("If this says VERIFIED, the same pooling code in "
           "`pipeline_v3/tools/make_elastic_rows.py::Gate` is sound and the elastic corpus "
           "can be gated with it. Then delete this notebook."),
    ]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith(("%%", "!")) or "\n!" in s:
            continue
        ast.parse(s)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ({len(cells)} cells, every code cell parses)")


if __name__ == "__main__":
    build()
