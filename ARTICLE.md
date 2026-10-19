# Building an on-prem decision model for adverse news

*How Tripwire-AdverseMedia-4B was built: a 4B open model that reads news about companies and returns
typed, probability-weighted decisions in one forward pass.*

## The job
A credit or compliance team that monitors hundreds of companies sees thousands of news items a day.
Almost all are noise: product reviews, passing mentions, a namesake in another country, a look back at
a scandal from ten years ago. The few that matter (a missed coupon, a regulator's probe, a lost
customer) have to reach a person quickly. Most of that triage is a set of small, repeatable judgements:

- Is this article about the company we watch, or a namesake, or just its product?
- Is the news negative, and how severe for its ability to pay?
- Which kind of problem is it: liquidity, legal, operational, governance?
- Is it a current development, or commentary on old events?

A chat model can answer these, but slowly, expensively and in prose that then has to be parsed. A
**decision model** answers each one as a choice among fixed options with a probability for every
option. Thresholds, routing and audit trails become ordinary code.

## The design
**One question, one token.** Each question is written as a short prompt: the state (the monitored
company and the article) as JSON, the question, and lettered options. The model's answer is read from
the probabilities it puts on the option tokens (A, B, C, or yes/no, or digits for a severity scale)
at the next position. Nothing is generated. Each question costs a single forward pass.

**A small open base.** Qwen3.5-4B (Apache-2.0) is small enough to fine-tune on one GPU and to run on
premises, so news about the companies you watch never leaves your network.

**A LoRA adapter, not a new model.** Training changes about 0.7% of the parameters (rank-16 LoRA on
the attention, linear-attention and MLP projections). The adapter is 122 MB; one base can carry
several adapters for different use cases.

## Training data
About 16,000 examples, each a (passage, question, answer) triple:
- **Real news passages** about 30 large, news-heavy companies and related people, labelled by a
  commercial decision-model API acting as the teacher. Only answers the teacher was at least 70% sure
  of were kept.
- **Human-judged company-name pairs** (OpenSanctions) for the "is this the same company?" question.
- **About 1,000 fictional passages written to cover the hard cases** the real data rarely shows:
  namesakes, product mentions ("a fire in a Halden dishwasher" versus "a regulator investigates Halden
  dishwashers"), anniversary pieces, dismissed lawsuits, the company as victim rather than accused,
  rating actions, sovereign defaults, credit-derivatives rulings, distressed exchanges written in calm
  language, and routine financings that only sound alarming. Each was cross-checked by the teacher;
  where they disagreed, the written answer was kept.

Three lessons from building the mix:
1. **Balance the rare answers.** Categories are "yes" for under 1% of real passages. Sampling them
   50/50 made the first model flag categories everywhere; 10% "yes" fixed it.
2. **Feed the failure modes.** Early versions missed rating downgrades (reading "Fitch cuts X to BB+"
   as negative but not severe). A hundred fictional rating actions, written with the right severity,
   fixed most of it.
3. **Keep evaluation data out of training, including its style.** No evaluation headline was used or
   imitated; when two training passages turned out to be modelled on test headlines, they were
   removed.

## Speed: read the passage once
The first harness asked the 18 questions one after another, re-reading the passage each time: 1.5 s
per passage on an A100. Three changes, measured one by one:

| | ms per passage (18 questions, A100) | answers unchanged |
| --- | --- | --- |
| One question at a time | 1,524 | — |
| Batch of 8 prompts | 507 | 899 / 900 |
| **Tree: passage once, questions as one batch** | **262** | 899 / 900 |

In tree mode the passage is run once and its cache (attention keys and values, and the linear-attention
layers' recurrent state) is copied to one row per question; the 18 questions then run as one batch
continuing from it. On the full benchmark 13,763 of 13,770 answers were identical and no decision
changed. One lesson along the way: copying the cache and then running the questions one at a time
was slower than doing nothing; the gain comes from batching the questions.

## Results
On public, human-labelled data (full tables in the model card):
- **Credit events** (1,102 headlines): 171 of 173 defaults and downgrades read as adverse (the other
  two are companies emerging from Chapter 11, which every model reads as good news), and none of 177
  upgrades and refinancings.
- **Company names** (565 human-judged pairs): precision 0.878 at recall 0.957, with only 1.8% of pairs
  left unsure, against 24.6% for the untrained base model.

The base model is already good at clear credit-event headlines. What fine-tuning adds is precise name
matching, calibrated probabilities, and reliable answers to the narrower questions (categories,
timing, adverse findings).

## What it is not
It reads; it does not check facts. It can confuse namesakes. It is deliberately strict about opinion
pieces and passing product mentions, which suits a busy inbox but means you should route borderline
answers to a person if you need maximum sensitivity. It is not for automated decisions about people.

## Adapting it
The same recipe works for other text decisions: alert triage, complaint routing, document checks.
Write the questions, gather a few thousand examples with trustworthy answers (real outcomes or
expert labels beat any teacher model), train one adapter per use case on the same base, and measure
on data the model never saw.
