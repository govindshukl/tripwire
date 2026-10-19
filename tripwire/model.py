"""Tripwire: typed decisions about news, from Qwen3.5-4B with the Tripwire-AdverseMedia-4B adapter.

    from tripwire import Tripwire, load_questions
    tw = Tripwire()                                         # downloads the base model and the adapter
    qs = load_questions(mention="Acme Steel")
    answers = tw.decide(state, qs)                          # one passage, every question
    tw.adverse(answers)                                     # the example decision rule

Each question is answered in one forward pass with no generated text: the answer is the probability
the model puts on each option's token at the answer position. Questions about one passage share
it: the passage is read once and its cache copied to one row per question, and the questions run as
one batch ("tree" reading).

Apache-2.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from . import prompt as P

BASE = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER = "govindshukl/Tripwire-AdverseMedia-4B"   # Hugging Face id of the adapter, or a local folder
LEAD = "<think>\n\n</think>\n\n"           # what Qwen3.5's chat template writes before a direct answer
QUESTIONS = Path(__file__).resolve().parents[1] / "questions" / "adverse_media.json"


def load_questions(mention: str, path: str | Path = QUESTIONS) -> dict:
    """The adverse-media question set, with `mention` (how the article names the company or person)."""
    text = Path(path).read_text().replace("{mention}", mention.replace('"', "'"))
    return json.loads(text)["questions"]


def _answer(q: dict, probs: list[float]) -> dict:
    if q["type"] == "noul":
        return {"type": "noul", "p_yes": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if q["type"] == "choice":
        names = list(q["criteria"])
        return {"type": "choice", "choice": names[best], "probabilities": dict(zip(names, probs))}
    return {"type": "score", "level": best, "expected": sum(i * p for i, p in enumerate(probs)),
            "probabilities": {str(i): p for i, p in enumerate(probs)}}


class Tripwire:
    def __init__(self, adapter: str = ADAPTER, base: str = BASE, base_revision: str | None = BASE_REVISION,
                 device: str | None = None, dtype: torch.dtype | None = None, merge: bool = True):
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype is None:   # bfloat16 where the hardware has it; float16 on older GPUs (e.g. T4); float32 on CPU
            if self.device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
            else:
                dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(base, revision=base_revision)
        model = AutoModelForImageTextToText.from_pretrained(base, revision=base_revision, dtype=dtype).to(self.device)
        model = PeftModel.from_pretrained(model, adapter)
        self.model = (model.merge_and_unload() if merge else model).eval()
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    @torch.inference_mode()
    def decide(self, state: Any, questions: dict, max_batch: int = 32) -> dict:
        """Answer every question about one state (a dict, like {"target": ..., "article": ...})."""
        inner = self.model.model
        names = list(questions)
        qs = [P.as_question(questions[n]) for n in names]
        trunk = self.tok.encode(P.prefix(self.tok, state, ""), add_special_tokens=False)
        out = {}
        for s in range(0, len(names), max_batch):
            part = list(range(s, min(s + max_batch, len(names))))
            branches = [self.tok.encode(P.suffix(self.tok, qs[i], LEAD), add_special_tokens=False) for i in part]
            n, p, width = len(branches), len(trunk), max(map(len, branches))
            cache = inner(input_ids=torch.tensor([trunk], device=self.device), use_cache=True).past_key_values
            cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=self.device))
            ids = torch.tensor([b + [self.pad] * (width - len(b)) for b in branches], device=self.device)
            mask = torch.tensor([[1] * (p + len(b)) + [0] * (width - len(b)) for b in branches], device=self.device)
            h = inner(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True).last_hidden_state
            last = torch.tensor([len(b) - 1 for b in branches], device=self.device)
            logz = torch.log_softmax(self.model.lm_head(h[torch.arange(n, device=self.device), last]).float(), -1)
            for k, i in enumerate(part):
                out[names[i]] = _answer(questions[names[i]], P.probabilities(self.tok, qs[i], logz[k]))
        return out

    @staticmethod
    def adverse(answers: dict, same_min: float = 0.5, negative_min: float = 0.3, severity_min: float = 0.5) -> dict:
        """The example decision rule the benchmark uses: the passage is about the target, the news is
        negative (or negative is at least `negative_min` likely), and expected severity reaches
        `severity_min` (scale 0-4). Thresholds are yours to set; this is not a credit decision."""
        p_same = answers["same_entity"]["probabilities"]["same"]
        d = answers["direction"]
        negative = d["choice"] == "negative" or d["probabilities"]["negative"] >= negative_min
        sev = answers["severity"]["expected"]
        cats = sorted(k[4:] for k, v in answers.items() if k.startswith("cat_") and v["p_yes"] >= 0.5)
        return {"adverse": p_same >= same_min and negative and sev >= severity_min,
                "p_same": round(p_same, 3), "severity": round(sev, 2), "categories": cats}
