# Protocol: experimental model benchmark v0.1

Updated 17 October 2026. This is a **development experiment**, not a completed accuracy study.

## Objective

Improve reproducibility and assess whether open-weight tweet sentiment models and
semantic stance models can handle historical vaccination tweets more reliably than
lexicon-only sentiment tools. **Emotion polarity and vaccination stance are different
targets**, and their predicted labels must never be conflated.

## Prespecified samples

1. Select up to 2,000 exact-text distinct records spanning observed calendar years
   2006-2019 using fixed SHA-256 ranking, with a minimum equal allocation per year
   where available, then filling shortages by global hash order. This sample is
   **year balanced, not prevalence-representative**.
2. Select up to 1,000 additional distinct records from the previously saved
   3,208-record enrichment pool by hash, excluding texts in sample (1).
   Enrichment records are deliberately non-probability; report results separately.
3. Never train, select cutoffs, or estimate corpus-level prevalence using the
   combined sample as though it were a random population sample.
4. Do not use saved model labels as independent ground-truth references.

## Stages and interpretation

A. **Language screening**: retain pre-existing source language values; when
   absent, run langid.py as an explicitly unvalidated automatic screen on the
   sampled texts, withholding labels on extremely short content. This adds no
   human-annotated reference data.
B. **English sentiment**: CardiffNLP Twitter-RoBERTa for English-screened
   records only. Automatically predicted English is not verified ground truth.
C. **Multilingual sentiment**: CardiffNLP Twitter XLM-R for the entire sample.
   English/multilingual comparisons are descriptive, not validity statistics.
D. **Exploratory generic vaccine stance**: DeBERTa-v3 zero-shot NLI with explicit
   candidate descriptions. Model scores and margins drive only **heuristic**
   abstention; they are not calibrated probabilities or accuracy.
E. **Semantic adjudication**: Qwen3-4B-Instruct-2507 for up to 300
   early-stage cases, favoring cross-model disagreement, ambiguities and a
   random agreement audit in the later experiment. The notebook uses a frozen 60% low-margin/abstention enrichment plus
   40% hash-ranked routine cases, with calendar-year coverage. Record-level
   evidence must be exact substring; outputs failing schema checks are marked
   needs_reassessment, never coerced to a valid label.

Different vaccine targets, authors vs quoted speakers, mandate policy, receipt vs
endorsement, alleged harm vs proven causality, and safety vs procedural fear are
distinct fields. No user-level political, demographic or geographic inferences.

## Accuracy evidence

No in-domain independent ground truth exists here. Do **not** publish an accuracy
percent from assistant annotations, evidence checks or model agreement. For future
accuracy work: identify legally accessible held-out gold for *each exact task*,
freeze predictions first, then report macro F1, class-wise recall/precision,
confusion matrix, model coverage and abstention rate. TweetEval sentiment and
published vaccine-stance sets are potential benchmarks, subject to task and text
compatibility. External benchmark performance does not estimate historical-corpus
error without a domain-shift study.

## Compute, provenance and privacy

Run only with authorized source data mounted in a private Kaggle dataset.
No raw tweets, passwords, .env files, identifiers, downloaded model weights or
inference outputs are added to the **public** GitHub repository.
Store JSONL with SHA-256 text digests, model hub commit hashes, selected device,
checkpointing and inference timings. Save per-source and per-year manifest.
Use GPU quota conservation: model stages run in separate processes to release
GPU memory between stages. Large-model inference is optional and capped by default.
A failed model load or unavailable GPU must be reported as a failure, not success.

## Decision gate

Before processing millions of records, confirm model load feasibility and stable
throughput, evaluate external validity where compatible, review abstention rates,
and justify the target-specific semantic classification strategy. No full-corpus
deployment or published estimate is authorized by a successful smoke test alone.
