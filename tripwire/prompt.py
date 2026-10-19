"""Tripwire prompt format: how a decision question is written for the model and how its answer is read.

Apache-2.0. Written for Tripwire-AdverseMedia-4B; replaces the third-party prompt module the
experiments used, producing the same text and the same answer tokens (checked by
experiments/release/check_prompt.py), so the trained adapter sees exactly the format it learned.

A decision is one forward pass. The prompt is the state (JSON), then one question, then the start of
the assistant's turn; the answer is read from the next-token probabilities of the option tokens only:
  yes / no question ("noul")   tokens yes / no; P(yes)
  choice                        each option gets a code (A, B, C ...); softmax over the codes
  score (2-10 ordered levels)   digits 0..n-1; softmax over the digits
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: Mapping[str, str | None]
    type: str = "choice"


@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: Mapping[str, Any] | None = None
    type: str = "noul"


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: Sequence[str]
    type: str = "score"


def as_question(q):
    """A question given as a dict ({"type", "instructions", "criteria"}) as one of the classes above."""
    if not isinstance(q, Mapping):
        return q
    kind = q.get("type", "choice")
    if kind == "noul":
        return Noul(q["instructions"], q.get("criteria"))
    if kind == "score":
        return Score(q["instructions"], list(q["criteria"]))
    return Choice(q["instructions"], q["criteria"])


# ---------------------------------------------------------------- option codes

def _codes(labels: Sequence[str]) -> list[str]:
    """Labels that are already single letters keep them; up to 26 options get A..Z; more get 00, 01 ..."""
    labs = [str(x).strip() for x in labels]
    if labs and all(len(x) == 1 and x.isalpha() for x in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


_SPARE = ([chr(c) for c in range(ord("A"), ord("Z") + 1)] + [f"{i:02d}" for i in range(100)]
          + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [f"#{i}" for i in range(200)]
          + [a + b for a in map(chr, range(65, 91)) for b in map(chr, range(65, 91))])


def option_codes(tokenizer, labels: Sequence[str]) -> list[tuple[str, int]]:
    """Each option's code and its token id. A code must be one token and not used twice; one that is
    not falls back to the next free single-token code from a spare list."""
    used, out = set(), []

    def take(code: str) -> bool:
        ids = tokenizer.encode(code, add_special_tokens=False)
        if len(ids) != 1 or ids[0] in used:
            return False
        used.add(ids[0])
        out.append((code, ids[0]))
        return True

    for code in _codes(labels):
        if not take(code) and not any(take(c) for c in _SPARE):
            raise RuntimeError(f"no single-token code left for {len(labels)} options")
    return out


def _single_ids(tokenizer, texts: Sequence[str]) -> list[int]:
    out = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def answer_token_ids(tokenizer, q) -> list[list[int]]:
    """For each option, the token ids that count as that answer (the highest of them is used)."""
    if isinstance(q, Noul):
        yes, no = _single_ids(tokenizer, ("yes", "Yes", "YES")), _single_ids(tokenizer, ("no", "No", "NO"))
        if not yes or not no:
            raise RuntimeError("the tokenizer has no single-token yes / no")
        return [yes, no]
    if isinstance(q, Score):
        groups = [_single_ids(tokenizer, [str(i)]) for i in range(len(q.criteria))]
        if any(not g for g in groups):
            raise RuntimeError("a score question needs 2 to 10 levels (single-token digits)")
        return groups
    groups = []
    for code, tid in option_codes(tokenizer, list(q.criteria)):
        groups.append([tid] + [i for i in _single_ids(tokenizer, [f" {code}"]) if i != tid])
    return groups


def probabilities(tokenizer, q, logprobs) -> list[float]:
    """Option probabilities from the log-probabilities at the answer position (indexable by token id)."""
    scores = [max(float(logprobs[i]) for i in g) for g in answer_token_ids(tokenizer, q)]
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    return [x / sum(e) for x in e]


# ---------------------------------------------------------------- prompt text

def state_text(state: Any) -> str:
    """The state as the model reads it: a string as is, anything else as indented JSON."""
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def question_text(tokenizer, q) -> str:
    if isinstance(q, Choice):
        codes = option_codes(tokenizer, list(q.criteria))
        lines = "\n".join(f"{codes[i][0]} {desc or label.replace('_', ' ')}"
                          for i, (label, desc) in enumerate(q.criteria.items()))
        return f"{q.instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if isinstance(q, Noul):
        extra = f"\nYes: {q.criteria.get('true')}\nNo: {q.criteria.get('false')}" if q.criteria else ""
        return f"{q.instructions}{extra}\n\nReply with yes or no only."
    if isinstance(q, Score):
        levels = "\n".join(f"{i} {name}" for i, name in enumerate(q.criteria))
        return f"{q.instructions}\n\n{levels}\n\nReply with a single digit 0-{len(q.criteria) - 1} only."
    raise TypeError(f"unknown question type {type(q)}")


def prefix(tokenizer, state: Any, bos: str = "") -> str:
    """Everything before the question; shared by every question about one state (tree reading)."""
    body = "" if state is None else f"{state_text(state)}\nQUESTION:\n"
    return f"{bos}{IM_START}user\n{body}"


def suffix(tokenizer, q, lead: str = "") -> str:
    """The question and the start of the assistant's turn, up to the answer position. `lead` is what the
    base model's chat template writes before a direct answer (Qwen3.5: an empty thinking block)."""
    return f"{question_text(tokenizer, q)}{IM_END}\n{IM_START}assistant\n{lead}"


def render(tokenizer, state: Any, q, bos: str = "", lead: str = "") -> str:
    return prefix(tokenizer, state, bos) + suffix(tokenizer, q, lead)
