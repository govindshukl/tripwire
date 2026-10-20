# Tripwire

On-prem decision models for adverse news about companies. The first model,
**Tripwire-AdverseMedia-4B**, is a LoRA adapter for Qwen3.5-4B that reads a news passage and answers
typed questions about a company you monitor (is it really about that company, is the news negative,
how severe, which kind of problem, is it current), each with a probability per option, in one forward
pass and without generating text.

- Model: [govindshukl/Tripwire-AdverseMedia-4B](https://huggingface.co/govindshukl/Tripwire-AdverseMedia-4B) on Hugging Face (model card, results, limits)
- How it was built: [ARTICLE.md](ARTICLE.md)
- Decision models across banking, finance and insurance: [articles/decision-models-in-financial-services.md](articles/decision-models-in-financial-services.md)

## Install
```bash
git clone https://github.com/govindshukl/tripwire.git && cd tripwire
pip install -e ".[fast]"     # or pip install -e . without the CUDA kernels
```
A GPU with 16 GB or more is comfortable (the model runs in bfloat16, about 9 GB); Apple Silicon
works, slowly.

## Use
```python
from tripwire import Tripwire, load_questions

tw = Tripwire(adapter="govindshukl/Tripwire-AdverseMedia-4B")
state = {"target": {"name": "Halvorsen Marine ASA", "type": "company",
                    "description": "Norwegian operator of offshore supply vessels.",
                    "why_tracked": "Monitored company"},
         "article": {"headline": "Halvorsen Marine misses bond coupon, hires restructuring adviser",
                     "passage": "Halvorsen Marine ASA did not pay the coupon due on its 2027 bonds ..."}}
answers = tw.decide(state, load_questions(mention="Halvorsen Marine"))
tw.adverse(answers)   # example rule: about the company, negative, expected severity >= 0.5
```
`examples/quickstart.py` runs this end to end.

**Your own questions:** any yes/no (`noul`), choice or score question in the same JSON shape works:
```python
{"type": "noul", "instructions": "Does `article` report a plant closure by `target`?"}
{"type": "choice", "instructions": "...", "criteria": {"option_a": "description", "option_b": "..."}}
{"type": "score", "instructions": "...", "criteria": ["level 0 description", "level 1", "..."]}
```
Questions close to the trained set are the most reliable.

## How it reads
`tripwire/prompt.py` writes the prompt (the state as JSON, `QUESTION:`, the question with lettered
options, the start of the assistant turn) and reads the answer from the probabilities of the option
tokens at the next position. `tripwire/model.py` reads a passage once and answers all its questions
as one batch: about 262 ms for 18 questions on an A100.

## Benchmark
```bash
python benchmark/needle_eval.py --adapter govindshukl/Tripwire-AdverseMedia-4B
```
Scores the public, human-labelled needle headlines (MIT): defaults and downgrades caught, upgrades
and refinancings wrongly flagged. Results for comparison models are in the model card.

## Responsible use
Outputs are leads for a person to review, not findings. Do not use it for automated decisions about
individuals. See the model card for limits.

## Licence
Apache-2.0 (`LICENSE`, `NOTICE`). Built on Qwen3.5-4B (Apache-2.0).
