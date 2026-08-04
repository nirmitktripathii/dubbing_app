"""
Builds files_v3/kaggle_runs/02j_budget_sweep.ipynb.

Built by a script, not by hand, and the embedded modules are read from disk at build time.
Hand-copied module text into a notebook is how the version that ran and the version in the
repo silently diverge — this project has already lost a session to a notebook cell that no
longer matched its source.

    python files_v3/kaggle_runs/build_02j.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PV3 = REPO / "pipeline_v3"
OUT = REPO / "files_v3" / "kaggle_runs" / "02j_budget_sweep.ipynb"

SRC = REPO / "files_v3" / "kaggle_runs" / "02i_corrected_eval.ipynb"

# Modules written into /kaggle/working/pipeline_v3, in dependency order.
MODULES = [
    ("common/__init__.py", None),
    ("common/languages.py", PV3 / "common/languages.py"),
    ("common/phonemes.py", PV3 / "common/phonemes.py"),
    ("translation/__init__.py", None),
    ("translation/semantic_gate.py", PV3 / "translation/semantic_gate.py"),
    ("evaluation/__init__.py", None),
    ("evaluation/phoneme_adherence_eval.py", PV3 / "evaluation/phoneme_adherence_eval.py"),
    ("evaluation/response_diagnosis.py", PV3 / "evaluation/response_diagnosis.py"),
    ("evaluation/budget_sweep.py", PV3 / "evaluation/budget_sweep.py"),
    ("tools/__init__.py", None),
    ("tools/preflight.py", PV3 / "tools/preflight.py"),
]


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def writefile(dest: str, path) -> dict:
    body = "" if path is None else Path(path).read_text(encoding="utf-8")
    if path is not None:
        ast.parse(body)  # never ship a module that does not parse
    return code(f"%%writefile /kaggle/working/pipeline_v3/{dest}\n{body}")


def build() -> None:
    src_nb = json.loads(SRC.read_text(encoding="utf-8"))
    # Cells 2-5 are the credential, torch-record, install and CUDA preflight. They are
    # environment setup that took six attempts to get right; reuse them verbatim.
    setup = src_nb["cells"][2:6]

    cells: list[dict] = [md("""# 02j — Budget-response sweep: WHY five languages ignore the phoneme budget

At checkpoint 3801 six languages follow the budget and five do not. A slope alone cannot
say which of four different defects that is, and they need four different fixes.

This session sweeps the budget **wide** (0.4x to 2.0x, against the old 0.6x–1.4x) on a
fixed sentence and keeps **every raw point**, so the shape of the response is readable
rather than just its slope.

Two things are different from every previous session in this phase.

**It is zero-referenced.** The old probe pooled raw lengths across sentences, which let
sentence-length variance leak into the fit: a model that ignores the budget entirely scored
0.60–0.83 depending on the language, not 0. Six of eleven languages were at or below their
own floor. This run reports the within-sentence normalised slope, where 0 means ignored.

**It pays for itself.** Every generation here is exactly measurable — the phoneme counter
is exact, and the semantic gate is the one production uses. So any generation that
verifiably hit its budget without losing its meaning is a training row for the corrective
fine-tune, harvested from work that had to happen anyway. Decision-critical sweep first,
harvest second, hard wall-clock stop throughout.
"""),
                         md("## 1. Environment (reused verbatim from 02i — six attempts to get right)")]
    cells += setup

    cells.append(md("## 2. Embedded modules\n\nRead from `pipeline_v3/` at build time, "
                    "not hand-copied, so the notebook cannot drift from the repo."))
    cells.append(code(
        "import os, sys\n"
        "for d in ('common', 'translation', 'evaluation', 'tools'):\n"
        "    os.makedirs(f'/kaggle/working/pipeline_v3/{d}', exist_ok=True)\n"
        "sys.path.insert(0, '/kaggle/working/pipeline_v3')\n"
        "os.chdir('/kaggle/working/pipeline_v3')\n"
        "print(os.getcwd())\n"))
    for dest, path in MODULES:
        cells.append(writefile(dest, path))

    cells.append(md("## 3. Preflight — G2P, then the corpus, before a model is loaded\n\n"
                    "Every check here is answerable in seconds. Discovering any of them "
                    "after four hours of generation is what this phase has been doing "
                    "wrong."))
    cells.append(code(
        "import logging, subprocess, sys\n"
        "logging.basicConfig(level=logging.INFO, format='%(message)s')\n"
        "from common.phonemes import assert_g2p_available, ruler_id\n"
        "info = assert_g2p_available()\n"
        "print('G2P preflight PASSED — ruler =', ruler_id())\n"))

    cells.append(md("## 4. Locate inputs"))
    cells.append(code(
        "from pathlib import Path\n"
        "\n"
        "CAND = list(Path('/kaggle/input').rglob('val.phonemes.jsonl'))\n"
        "assert CAND, 'val.phonemes.jsonl not found in /kaggle/input — attach 02h output'\n"
        "VAL = CAND[0]\n"
        "print('val :', VAL)\n"
        "\n"
        "CK = sorted({p.parent for p in Path('/kaggle/input').rglob('adapter_model.safetensors')},\n"
        "            key=lambda p: int(''.join(c for c in p.name if c.isdigit()) or 0))\n"
        "assert CK, 'no checkpoints found in /kaggle/input'\n"
        "CKPT = str(CK[-1])\n"
        "print('checkpoint:', CKPT)\n"))
    cells.append(code(
        "# The corpus gate. Structural checks only — the training corpus is not attached\n"
        "# here, so this validates what the sweep will actually read.\n"
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-m', 'tools.preflight', '--stage', 'kaggle',\n"
        "                '--corpus', str(VAL), '--no_wandb_check',\n"
        "                '--require', 'ruler', 'labels', 'prompt'], check=True)\n"))

    cells.append(md("## 5. Smoke test — one generation, before committing an hour\n\n"
                    "A `!` cell swallows exit codes, so this is a real assertion. The "
                    "first attempt at 02i failed here and reported it four cells later as "
                    "an unrelated missing file."))
    cells.append(code(
        "import time, json\n"
        "from evaluation.phoneme_adherence_eval import load_model, generate_batch, "
        "true_phoneme_len, SemanticScorer\n"
        "\n"
        "t0 = time.time()\n"
        "rows = [json.loads(l) for l in open(VAL, encoding='utf-8')]\n"
        "row = next(r for r in rows if r['language'] == 'or')\n"
        "m, tk = load_model(CKPT, 'meta-llama/Llama-3.1-8B-Instruct', 512)\n"
        "\n"
        "# Ask for HALF the natural length — the regime the old sweep never entered.\n"
        "nat = int(row['n_phonemes'])\n"
        "want = max(1, round(nat * 0.5))\n"
        "p = f'[Translate to Odia] [Target Phonemes: {want}] \\\"' + row['english'] + '\\\"'\n"
        "gen = generate_batch(m, tk, [p], batch_size=1)[0]\n"
        "got = true_phoneme_len(gen, 'or')\n"
        "print(f'natural {nat}  requested {want}  produced {got}')\n"
        "print('generated:', gen[:120])\n"
        "assert gen and got, 'generation or phonemization failed'\n"
        "\n"
        "# Throughput, measured rather than assumed — it sets the sweep size below.\n"
        "t1 = time.time()\n"
        "batch = generate_batch(m, tk, [p] * 24, batch_size=24)\n"
        "rate = 24 / (time.time() - t1)\n"
        "print(f'SMOKE PASSED in {time.time()-t0:.0f}s — {rate:.1f} generations/sec at batch 24')\n"
        "\n"
        "del m\n"
        "import gc, torch\n"
        "gc.collect(); torch.cuda.empty_cache()\n"))

    cells.append(md("## 6. The sweep\n\n"
                    "11 languages x 40 sentences x 9 scales. Writes after every language, "
                    "so an interrupted session still answers the question. "
                    "`--time_budget_s` is a hard stop: on Kaggle's T4x2 every wall-clock "
                    "hour costs two GPU-hours, so 5400s here is ~3 GPU-hours."))
    cells.append(code(
        "import subprocess, sys\n"
        "\n"
        "cmd = [sys.executable, '-m', 'evaluation.budget_sweep',\n"
        "       '--val_jsonl', str(VAL),\n"
        "       '--checkpoint', CKPT,\n"
        "       '--output_dir', '/kaggle/working/sweep',\n"
        "       '--sentences_per_lang', '40',\n"
        "       '--batch_size', '24',\n"
        "       '--time_budget_s', '5400',\n"
        "       '--harvest', '--harvest_sentences', '60',\n"
        "       '--wandb', '--wandb_required',\n"
        "       '--wandb_project', 'indic-dubbing-v3',\n"
        "       '--wandb_entity', 'nktthegreat-soccernet',\n"
        "       '--wandb_run_name', '02j-budget-sweep']\n"
        "print(' '.join(cmd), flush=True)\n"
        "subprocess.run(cmd, check=True)\n"))

    cells.append(md("## 7. The diagnosis — read this, not the aggregate"))
    cells.append(code(
        "from pathlib import Path\n"
        "print(Path('/kaggle/working/sweep/DIAGNOSIS.json').read_text(encoding='utf-8')[:4000])\n"))
    cells.append(code(
        "import json\n"
        "from evaluation.response_diagnosis import format_report\n"
        "d = json.loads(Path('/kaggle/working/sweep/DIAGNOSIS.json').read_text(encoding='utf-8'))\n"
        "print(format_report(d))\n"
        "print()\n"
        "print(Path('/kaggle/working/sweep/SUMMARY.json').read_text(encoding='utf-8'))\n"))

    cells.append(md("## 8. What was harvested\n\n"
                    "Verified elastic rows: the generation hit its budget within 10% AND "
                    "stayed above the 0.80 semantic gate. These are training data for the "
                    "corrective fine-tune, produced by a run whose purpose was diagnosis."))
    cells.append(code(
        "import json, collections\n"
        "cands = [json.loads(l) for l in open('/kaggle/working/sweep/candidates.jsonl',\n"
        "                                     encoding='utf-8')]\n"
        "ok = [c for c in cands if c['harvestable']]\n"
        "print(f'{len(ok)} harvestable of {len(cands)} generations')\n"
        "per = collections.Counter(c['language'] for c in ok)\n"
        "print('per language:', dict(sorted(per.items())))\n"
        "comp = sum(1 for c in ok if c['scale'] < 1.0)\n"
        "print(f'compression rows: {comp} ({comp/max(1,len(ok)):.0%}) — dubbing needs these')\n"
        "for c in ok[:5]:\n"
        "    print(f\"\\n  {c['language']} @ {c['scale']}x  want {c['requested_n']} \"\n"
        "          f\"got {c['produced_n']}  sem {c['semantic']}\")\n"
        "    print('   ', c['generated'][:110])\n"))

    cells.append(md("Save Version. Step 2 consumes `sweep/candidates.jsonl` and is chosen "
                    "by `DIAGNOSIS.json`'s `dominant_failure` — the action for each "
                    "diagnosis was pre-registered in `evaluation/response_diagnosis.py` "
                    "before this run, so the decision does not get made under pressure."))

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith(("%%", "!")) or "\n!" in s:
            continue
        try:
            ast.parse(s)
        except SyntaxError as e:
            raise SystemExit(f"cell {i} does not parse: line {e.lineno}: {e.msg}\n{s[:400]}")

    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ({len(cells)} cells, every code cell parses)")


if __name__ == "__main__":
    build()
