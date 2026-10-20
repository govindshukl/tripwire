"""Credit-event benchmark on public, human-labelled headlines (needle, MIT licence).

Shreyas94/needle-event-classifier-eval holds 1,102 human-reviewed financial-news headlines in 13
event classes. Each unique headline is judged as a passage about "the company the headline is
about", and the example decision rule (Tripwire.adverse) decides whether it is adverse, without its
same-company test: every headline is about its company by construction, and asking whether "the
company" in a headline is the target is not a meaningful question (the model often answers "cannot
tell"). Reported:
  recall     defaults and downgrades read as adverse (and the same leaving out companies emerging
             from Chapter 11, which every model reads as positive news)
  false alarms   upgrades and refinancings read as adverse

Run:  python benchmark/needle_eval.py --adapter govindshukl/Tripwire-AdverseMedia-4B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripwire import Tripwire, load_questions  # noqa: E402

REPO, REVISION = "Shreyas94/needle-event-classifier-eval", "2c9781f197251fbf1179586df39899e3b935e24a"
EMERGED = ("emerges from chapter 11", "concludes chapter 11", "emerges from chapter")


def headlines() -> list[dict]:
    seen, out = set(), []
    for split in ("test", "validation"):
        path = hf_hub_download(REPO, f"{split}.jsonl", repo_type="dataset", revision=REVISION)
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r["headline"] not in seen:
                seen.add(r["headline"])
                out.append({"headline": r["headline"], "label": r["true_label"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="needle_results.jsonl")
    a = ap.parse_args(argv)
    tw = Tripwire(adapter=a.adapter)
    qs = load_questions(mention="the company")
    pos = neg = hit = hit_excl = pos_excl = fa = 0
    with open(a.out, "w") as f:
        for r in headlines():
            state = {"target": {"name": "the company the headline is about", "why_tracked": "Borrower of the bank"},
                     "article": {"headline": r["headline"], "passage": "(headline only)"}}
            d = tw.adverse(tw.decide(state, qs), same_min=0.0)   # the company is given (see above)
            f.write(json.dumps({**r, **d}) + "\n")
            if r["label"] in ("default", "downgrade"):
                pos += 1
                hit += d["adverse"]
                if not any(e in r["headline"].lower() for e in EMERGED):
                    pos_excl += 1
                    hit_excl += d["adverse"]
            elif r["label"] in ("upgrade", "refinancing"):
                neg += 1
                fa += d["adverse"]
    print(json.dumps({"defaults_downgrades": pos, "recall": round(hit / pos, 3),
                      "recall_excluding_chapter11_emergences": f"{hit_excl}/{pos_excl}",
                      "upgrades_refinancings": neg, "false_alarms": fa}, indent=1))


if __name__ == "__main__":
    main()
