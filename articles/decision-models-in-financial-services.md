# Decision models in banking, finance and insurance: where they fit and how to build them

Most of the text that flows through a bank, an asset manager, a payment firm or an insurer is read by people who then make a small decision about it. Is this alert worth an investigator's hour? Is this claim simple enough to pay now? Does this news item change what we think about a borrower? Is this hit on the sanctions list the same person? The reading is expensive and the decision is narrow. That is the shape of work a decision model is built for.

This article explains what a decision model is, where it fits across the three sectors, how to build and run one, and what it cannot do. The worked example is Tripwire-AdverseMedia-4B, an open-source adapter for adverse news about companies ([code](https://github.com/govindshukl/tripwire), [model](https://huggingface.co/govindshukl/Tripwire-AdverseMedia-4B)). The method is general.

## What a decision model is

A decision model reads a piece of text and answers a fixed, typed question. The question has a small set of options, and the model returns a probability for every option. The options can be yes or no, one of several named choices, or a level on an ordered scale. The model does this in a single forward pass and generates no text.

The mechanism is simple. The prompt ends at the point where the answer would begin. At that position the model has a probability for every token in its vocabulary. The answer is read from the probabilities on the tokens that stand for the options (A, B, C; yes, no; the digits 0 to 4), renormalised so they sum to one. Nothing is parsed, nothing is sampled.

A typed question looks like this:

```json
{
  "id": "severity",
  "type": "score",
  "text": "How severe is this news for the company's credit standing?",
  "options": {
    "0": "no signal",
    "1": "early sign of difficulty",
    "2": "material pressure on earnings, cash or funding",
    "3": "acute stress: covenant breach, missed payment, forced sale",
    "4": "default or insolvency event"
  }
}
```

and an answer looks like this:

```json
{"severity": {"0": 0.02, "1": 0.06, "2": 0.71, "3": 0.19, "4": 0.02}}
```

Several things follow from this design.

- It is fast and cheap per decision. One pass, no generated tokens, and the same document can serve many questions.
- Ordinary code can act on it. A probability is a number. Thresholds, routing, sorting and review bands are a few lines of code, and the same code runs unchanged when the model is retrained.
- It is auditable. For every item you can store the question, the probabilities and the threshold that applied. A reviewer can see why an item went where it went.
- It can run inside the firm. A small model on one GPU handles a large volume. Claim notes, alert narratives and customer emails never leave the firm's network.

### How it compares

Chat and generative LLMs are flexible. You can describe a new task in the prompt and get a reasonable answer the same afternoon. The costs are speed (tokens are generated one at a time), output handling (the answer is text that must be parsed, and the parser is where silent failures live) and calibration (a model that writes the word "likely" is giving you a word, and the probability behind it is hard to recover and harder to trust). For a decision made ten thousand times a day with the same question each time, these costs dominate.

Classic classifiers and gradient-boosted models remain the right tool for tabular data. They are small, fast, well understood by model-risk teams and explainable feature by feature. Their weakness is open text. Bag-of-words features miss negation, timing and who did what to whom, and each new label set is a new model to build and validate.

Decision models sit between these. They read text with the comprehension of a language model and return the kind of output a classifier returns. They do not replace rules or tabular models. They turn text into decisions and into features that rules and tabular models consume. In the adverse-media example, the decision model says what the news means; code decides what to do about it, using exposure, relationship weights and policy thresholds that were never inside the model.

Hosted decision-model APIs exist as a category and can be a sensible way to start. The rest of this article is about building your own, which is what you need when the data cannot leave the firm, the labels are your own history, or the model must sit in your inventory under your control.

## Where decision models fit

The common pattern across sectors is a queue of text items, a human who reads them, and a small decision with a known cost of being wrong in each direction. The table for each sector lists the input, the typed questions, where labels come from, who consumes the output and the main risk. All company names are invented.

### Banking

| Use case | Input | Typed questions | Labels from | Consumer | Main risk |
| --- | --- | --- | --- | --- | --- |
| Adverse-media early warning (corporate credit) | News passage plus the monitored company's profile | Same entity? Direction? Severity 0 to 4? Credit relevant? Adverse finding? Current? Which category? | Expert labels with a rubric; later, credit events | Credit officers via a triage queue | Namesake confusion; stale news read as new |
| Sanctions and name-screening hit triage | Screening hit: list entry versus customer record | False positive / possible match / escalate | Past analyst dispositions | Screening analysts; compliance | A true match closed as a false positive |
| AML alert triage | Alert narrative, transaction summary, customer profile | Close / review / escalate; typology options | Past investigation outcomes and SAR decisions | Investigators | Model learns the old team's blind spots |
| Trade-finance document checks | Letter of credit terms and presented documents | Discrepancy present? Which field? Material? | Past document examiner findings | Document examiners | Subtle discrepancies in formats the model never saw |
| Covenant monitoring | Financial statements, borrower notes, compliance certificates | Covenant mentioned? Breach stated? Waiver requested? | Past covenant events and reviewer notes | Portfolio monitoring | Reading a forecast as a fact |
| Complaint and email routing | Customer message | Team (one of N); priority; conduct-risk indicator yes/no | Past routing and complaint outcomes | Operations; conduct risk | Vulnerable customers under-prioritised |
| KYC document and adverse-information checks | Onboarding documents, search results | Document type? Fields consistent? Adverse information about this person? | Past onboarding outcomes; expert labels | Onboarding analysts | Wrong-person matches on common names |

Consider Brennard Capital, an invented commercial lender with 1,200 corporate borrowers and a daily feed of several thousand news items about them and the companies linked to them. The decision-model version asks eighteen typed questions of every passage, and code uses the answers to decide which items reach a credit officer and in what order. Most items are nothing. The model's job is to say so with a probability that can be trusted, and to lift the few that matter.

The sanctions and AML cases are often the fastest to build, because years of dispositions exist and the output goes to a well-defined queue. They also carry the sharpest risk. A model trained on past dispositions learns the past team's habits, including the ones a regulator would not like. Audit the labels before training, not after.

### Capital markets, asset management and payments

| Use case | Input | Typed questions | Labels from | Consumer | Main risk |
| --- | --- | --- | --- | --- | --- |
| Event tagging for portfolio monitoring | News item or filing excerpt about an issuer | Event type (downgrade, default, guidance cut, M&A, none)? Which issuer? Confirmed or rumoured? | Past analyst tags; rating-agency actions | Portfolio managers, credit analysts | Rumour tagged as a confirmed event |
| ESG controversy screening | News passage about a holding | Controversy present? Topic? Severity? Company at fault or affected? | Expert labels with a rubric | Stewardship and ESG teams | Treating a victim as a perpetrator |
| Merchant onboarding | Merchant website text, application narrative | Business category? Prohibited activity indicator? Mismatch with stated category? | Past onboarding outcomes and chargeback history | Risk operations | Category fraud that reads as a plain business |
| Transaction-dispute triage | Cardholder and merchant statements, evidence | Dispute reason code? Evidence sufficient? Likely outcome? | Past dispute outcomes | Dispute teams | Cardholder with a weak statement but a valid claim |
| Surveillance alert triage | Alert plus communications or order context | Close / review / escalate; behaviour type | Past surveillance dispositions | Compliance surveillance | Training on dispositions that were themselves too lenient |

At Oakhill Payments, an invented acquirer, the model answers three questions per merchant application. The answers do not decide whether to onboard. They decide which of three onboarding paths the application takes, and the highest-risk path has a person on it.

### Insurance

| Use case | Input | Typed questions | Labels from | Consumer | Main risk |
| --- | --- | --- | --- | --- | --- |
| First notice of loss and claims triage | FNOL narrative, claim notes | Covered? Severity band? Complexity? Fast-track eligible? Injury involved? | Past claim outcomes and handler assignments | Claims operations | Fast-tracking a claim that needed investigation |
| Fraud-indicator flags | Claim narrative, correspondence | Indicator present (one of N)? Investigator referral warranted? | Past SIU referrals and their outcomes | Special investigations unit, as leads only | A flag that becomes a denial without a person |
| Underwriting submission checks | Broker submission, schedules, loss runs | Information missing (which)? Risk flag present? Outside appetite? | Underwriter notes and declinatures | Underwriters | Declining good risks on a wording quirk |
| Policy-wording and exclusion checks | Wording versus claim circumstance | Exclusion engaged? Which clause? Ambiguous? | Coverage counsel opinions; past coverage decisions | Claims and coverage teams | Over-confident reading of ambiguous wording |
| Subrogation detection | Claim file, incident narrative | Third party at fault? Recovery possible? | Past subrogation recoveries | Recovery teams | Low recall; missed money rather than wrong money |
| Complaint routing | Customer complaint | Team; priority; vulnerable customer indicator | Past routing and ombudsman outcomes | Customer relations | Under-prioritising a vulnerable customer |

Meridian Mutual, an invented personal-lines insurer, applies the claims-triage questions at first notice of loss. Claims the model reads as covered, low severity and low complexity, with high confidence, go to a fast-track team; everything else goes to a handler. The fraud-indicator question runs on the same pass, and its output is a lead for an investigator. It never changes the outcome of a claim on its own.

### Cross-industry: guardrails

Any firm that runs a generative model in front of customers or staff has a second problem, which is checking what the first model said. A decision model fits. The input is the generated output plus the context it was meant to answer. The questions are typed: supported by the source material (yes/no), contains a prohibited claim (yes/no), and an overall pass / revise / block. Labels come from reviewers who graded earlier outputs. Code lets the response through, sends it back for regeneration or holds it for a person. Because the guardrail is a separate, small, non-generative model, it is easy to validate on its own and hard to talk out of its answer.

## A build-and-run recipe

The steps below are in order. Each one has a lesson from building Tripwire attached where there is one.

### 1. Write the questions first

Before any data work, write the questions as if they were going into a rubric for a new hire. Keep each one small. Make the options mutually exclusive, and give each option a plain-language definition. Mix the three types where it helps. Yes/no is right for a single fact (is this an adverse finding against the company). A choice is right for a routing decision (same entity, product mention only, different entity, cannot tell). A score is right when the options are ordered and you will want an expected value (severity 0 to 4).

Two habits pay off. Include an option for the honest uncertain case where one exists, such as cannot tell, so the model is not forced into a wrong confident answer. And split compound questions. Is this bad news about our borrower is three questions (is it about the borrower, is it negative, how severe), and a model answers three clean questions better than one muddled one.

Tripwire's set has eighteen questions: same entity, direction, severity 0 to 4, credit relevance, adverse finding, current development, and twelve yes/no risk categories. The twelve are separate yes/no questions rather than one twelve-way choice because a passage can belong to two categories at once, and a yes/no per category is easier to threshold and audit.

### 2. Get labels you can trust

Label quality decides the model. In descending order of value:

1. Real outcomes and past decisions. Alert dispositions, claim outcomes, subrogation recoveries, credit events, document examiner findings. These are what the model is supposed to predict.
2. Expert labels against a written rubric, with two labellers and a measured agreement rate. Where the two disagree, the rubric is unclear, and fixing the rubric is part of the work.
3. Synthetic examples, as a supplement only, for rare hard cases that the real data under-represents. Every synthetic example should be read by a person before it enters training.

Outcome labels beat a teacher model's opinion for a plain reason. A teacher model is a guess about the answer, and a student trained on it learns the teacher's errors as faithfully as its successes. An outcome is the answer. Where you must use a teacher to bootstrap before real labels exist, keep only the answers it was confident about, audit a sample by hand, and plan to replace them with outcome labels as those accumulate.

Balance rare classes deliberately, then check the balance against reality. In this project's first training run (on an earlier, smaller base model), the twelve risk categories were sampled so that each was yes about half the time. In real news each category is yes well under 1% of the time. The model learned the training distribution and flagged categories everywhere. Resampling to 10% yes fixed it. A rare class needs enough examples to learn from, not an equal share.

Feed known failure modes with targeted examples: namesakes, product mentions (an article about a car is not an article about the carmaker's credit), old news retold, dismissed allegations, companies that are the victim rather than the perpetrator, and positive events written in negative-sounding language. Each is a class of error a reviewer will recognise, and each wants a few hundred examples.

Keep evaluation data and its style out of training. Not just the items, but the template. If your test set is headlines in a particular house style and your training set contains headlines in that style, your test score will flatter you.

### 3. Pick a base model and adapt it

A small open model is enough. Tripwire uses a 4B-parameter base under Apache-2.0 and adapts it with a LoRA adapter of rank 16. The adapter is 122 MB, about 0.7% of the base model's parameters. That size matters operationally. A dozen adapters for a dozen decisions fit beside one base model on one GPU, and each adapter is a file you can version, hash, diff and roll back.

Two training details are worth copying. First, the training loss is computed on the single answer token only, so the model is trained to put probability on the right option and nothing else. Second, the order of the options is shuffled per example, so the model reads the options rather than learning that the answer is usually B.

A smaller experiment on a 0.8B proxy model points at a third detail. Computing the loss over the answer options only, instead of over the whole vocabulary, removed a systematic misreading of one upgrade-headline template (false alarms fell from 16 to 1 out of 87). That is one seed on a small model, so treat it as an early finding to test on your own data rather than a rule.

Tripwire was trained for one epoch over about 16,000 examples.

### 4. Evaluate on data the model never saw

Evaluate on human-labelled or outcome-labelled data that is separate from training by item and by style. Report three kinds of numbers.

- Precision and recall on the rare, costly class. For adverse media that is the adverse item; for claims it is the claim that should not have been fast-tracked; for sanctions it is the true match.
- Calibration. The Brier score (lower is better) and a calibration error tell you whether a 0.8 means 80%. Thresholds only mean something if the probabilities do.
- An operational measure. How many items go to human review at the thresholds you intend to use. A model with excellent precision that sends 40% of items to a person has not solved the problem.

Use separate dev and test splits, and set thresholds on dev, never on test. Report confidence (an interval or at least the sample size) with every number. And prefer two training seeds before believing a difference between two configurations. A one-seed improvement of a few points is often noise.

Tripwire's published results show why this discipline is needed. On a public set of credit-event headlines (human-labelled, MIT licence), the model read 171 of 173 defaults and downgrades as adverse and wrongly flagged 0 of 177 upgrades and refinancings. The two it missed were companies emerging from Chapter 11, which every model we tested read as good news, a reasonable reading. The untrained base model scored the same on this set. A naive write-up would have claimed the fine-tune for those numbers. The gains of fine-tuning are elsewhere, on company-name matching: on 565 human-judged company-name pairs from OpenSanctions, precision rose from 0.668 to 0.878 at recall 0.957, the share of pairs the model left unsure fell from 24.6% to 1.8%, and the Brier score improved from 0.182 to 0.075. The calibration gain is the one that matters for routing.

### 5. Put thresholds and routing in code, with a review band

The model returns probabilities; policy lives in code. A typical routing rule for one yes/no question has three regions.

```python
p = answers["adverse_finding"]["p_yes"]
if p >= 0.80:
    route("escalate")
elif p >= 0.35:
    route("review")        # a person decides
else:
    route("auto_close")
```

The middle band is the important one. Items the model is unsure about go to a person instead of being silently dropped or silently escalated. The width of the band is an operational choice, set by review capacity and the cost of each kind of error, and it is written down where model risk can see it. The same structure works for a choice question (route on the top option only when it clears a margin over the second) and for a score (route on the expected value and on the probability mass above a level).

Keeping thresholds in code has a second benefit. Changing a threshold is a configuration change with a small audit trail. Changing the model is a model change with a validation cycle. Most policy adjustments turn out to be the first kind.

### 6. Serve it efficiently

The cost of a decision model is dominated by reading the document, not by answering the question. So read each document once and answer all its questions as one batch. Tripwire calls this tree reading: the passage (the trunk) runs through the model once, its state is copied to one row per question, and the questions (the branches) then run together as one batch.

On one A100 (40 GB), a passage with eighteen questions took 1,524 ms when each question re-read the passage, 507 ms with batches of eight, and 262 ms with tree reading. On 13,770 test answers, tree reading and the naive method gave the same answer 13,763 times. One negative result is instructive: copying the passage's cache and then asking the questions one at a time was slower than doing nothing at all. The gain comes from batching the questions, not from caching the passage.

Serve one base model with many adapters and select the adapter per request, so the sanctions adapter, the claims adapter and the complaints adapter share a GPU. Serving engines such as vLLM support many adapters on one base and add continuous batching across documents; check that the engine supports your base model's layer types before you rely on it.

Calibrate with temperature scaling on held-out labelled data from your own population. A single scalar fitted on a few thousand labelled items is usually enough to make the probabilities honest, and it is cheap to refit when the population drifts.

### 7. Govern it as a model

Each adapter is a model in the inventory. It is validated, versioned and rolled back on its own, with its own data lineage, evaluation report and owner. The base model is a shared dependency with its own entry. This is the SR 11-7 shape of model risk management, and the NIST AI RMF covers the same ground in broader language. A decision model needs no new governance framework; it needs the existing one applied to four things that are easy to skip.

Monitor drift and override rates. The share of items in each routing band, the share of reviewed items where the person disagreed with the model, and the score distribution over time are cheap to compute and are the earliest warning that the world has moved. A rising override rate in one category is a request for labels in that category.

Keep a person on consequential outcomes. The routing bands above are the mechanism. A decision model suggests, prioritises and sorts; a person decides where the decision has real consequences for a person or a firm.

Know where the regulatory lines are. Under the EU AI Act, creditworthiness assessment of natural persons and risk assessment and pricing for life and health insurance are listed high-risk uses, with the obligations that follow. GDPR limits decisions based solely on automated processing that have legal or similarly significant effects on individuals. Most use cases in this article concern companies, or route items to people rather than deciding, and that is deliberate. If you build towards natural persons, bring legal in at the question-writing stage.

Check data rights. Every data source has terms, and so does every model whose outputs you train on. News licences often restrict redistribution and sometimes model training. Open datasets carry attribution and sometimes non-commercial clauses (Tripwire's name-pair data is CC BY-NC, and the model card says so). If you bootstrap labels from another model, read its terms on training with its outputs. Write the lineage into the inventory entry; model risk and legal will both ask.

## Limits

A decision model reads what the text says. It does not check facts, and hostile or simply wrong text will mislead it. It can confuse similarly named companies and people, which is why the same-entity question exists and why its answer is a lead rather than a finding. Its probabilities are only as calibrated as your last temperature fit on your own data. Its answers degrade on questions far from the ones it was trained on, and a new question usually wants a few hundred labelled examples rather than a prompt change. A small model knows less about the world than a large one, so a passage that needs outside knowledge to interpret (what a regulator's letter implies, whether a counterparty is state-owned) may need a second step with a larger model or a lookup. And a model trained on past decisions inherits the past team's biases, which is the strongest argument for auditing labels before training and override rates after.

## How to start

Pick one decision. Choose a queue with high volume, a clear reader today, a known cost for each kind of error, and labels that already exist in some system (dispositions, outcomes, routing history). Sanctions-hit triage, claims triage and complaint routing are the usual candidates.

Write the questions. Small, typed, mutually exclusive options, each with a definition a new hire could apply. Review them with the people who do the work today.

Gather a few thousand trustworthy labels. Outcomes first, then expert labels with a rubric and two labellers, then a small number of audited synthetic examples for the rare hard cases. Set aside a test split before anyone trains anything.

Train one adapter on a small open base model, with the loss on the answer token and the option order shuffled.

Measure on unseen data. Precision and recall on the costly class, Brier score, and the share of items that would go to review. Two seeds before you believe a difference.

Add a review band. Put the thresholds in code, send the middle to a person, and record the model's probabilities with every item so the audit trail exists from day one.

Then register it, monitor overrides, and let the labels from the review band feed the next version.

The Tripwire code and adapter are at [github.com/govindshukl/tripwire](https://github.com/govindshukl/tripwire) and [huggingface.co/govindshukl/Tripwire-AdverseMedia-4B](https://huggingface.co/govindshukl/Tripwire-AdverseMedia-4B), under Apache-2.0. The question file, the prompt format, the tree-reading inference code and the needle benchmark script are there to copy for your own decision.
