# Full vaccination tweet corpus workflow

**Initiated:** 19 October 2026. User-authorized full-corpus processing, separate from the original 2,000-tweet development benchmark and Qwen model comparisons.

## Verified dataset scope

The two private Kaggle sources are master.csv and vaccination2.csv. The private CPU index completed with **2,285,081 raw tweet records**, **1,944,361 distinct exact-text tweets**, and **340,720 duplicate-text occurrences**. Nonblank unique texts were assigned to **64 deterministic SHA-256 shards**. No source tweet text or IDs are committed to GitHub.

Deduplication reduces inference cost; it does not redefine the original source-event population. A separate private CPU job reconstructs raw per-text occurrence weights by source and date, enabling both unique-text and original-event-weighted descriptive summaries. The event stage must complete before reporting raw-event-weighted label frequencies.

## Execution DAG

Private CPU source index → 1,000-text GPU capacity pilot → 64 private sentiment shard kernels → 64 private exploratory stance shard kernels → validated descriptive aggregation.

The [private corpus index](https://www.kaggle.com/code/keplaxo/vax-full-corpus-index-20261019) is COMPLETE. The [GPU capacity pilot](https://www.kaggle.com/code/keplaxo/vax-corpus-capacity-20261019) is COMPLETE with full coverage. The first [sentiment shard](https://www.kaggle.com/code/keplaxo/vax-corpus-sentiment-00-20261019) was submitted after the pilot passed. [GitHub Actions scheduler](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/workflows/vax-full-corpus-inference.yml) advances at most one new private GPU kernel per hourly check, if all prior output quality gates pass.

All unique texts receive the established CardiffNLP Twitter XLM-R sentiment model. English-screened texts also receive CardiffNLP Twitter RoBERTa sentiment and exploratory DeBERTa-v3-large six-category zero-shot vaccine stance. Qwen v1/v2/v3 are not being run on 1.94 million texts: their original disagreement and external intention-proxy results do not support broad deployment.

Language screening is heuristic and does not presume unscreened text is English. Missing or out-of-range parsed dates are included as unknown year, not dropped. Each inference shard verifies the complete exact hash set and private model output count, recording inference runtime, model revision and anonymous aggregate labels.

The controller uses a conservative 14,500-second estimated per-shard upper bound (relative to Kaggle's 18,000-second execution limit). If the pilot suggests that a shard is too large, or if Kaggle denies quota, loses input data or records a failed stage, progression stops for assessment. Scheduling is not a guarantee of prompt completion, and no paid compute is being purchased.

## Data handling and interpretation

Keep source texts, row hashes, private checkpoint predictions and model evidence only in private Kaggle outputs. Github contains scripts, documentation and safe aggregate summaries.

Always distinguish source-event denominators from unique-text denominators, include coverage by language and year, describe abstentions, and distinguish emotional sentiment from vaccine stance and personal vaccination intention. **No task-matched independent ground-truth accuracy has been established for vaccine stance.** Model-generated label rates must be described as unvalidated predictions, not confirmed vaccine-hesitancy prevalence or vaccine-related medical causality.

## Completion gates

1. CPU index: completed with both original CSVs.
2. CPU original-event weights: submitted, completion still to be checked.
3. GPU capacity and throughput: submitted and verified running, results pending.
4. All 64 sentiment and 64 stance shards: scheduled conditionally after capacity gating; none should be claimed complete until verified.
5. Private CPU research reporting is now implemented: 16 four-shard waves, each joining completed sentiment/stance predictions to original source-event weights, and one final synthesis that checks all 64 shards and year/source denominators. It is gated on all required inputs and is **not yet executed**. Resulting label shares are predictions, not independently validated vaccine-hesitancy prevalence.

Previously completed 2,000-tweet and Qwen v1–v3 experimental outputs remain intact. No full-corpus research conclusions have been produced yet.

## Recovery and resumed processing — 27 October 2026

The GitHub scheduler verified the CPU index with **2,285,081 raw rows**, **1,944,361 distinct exact texts**, **340,720 duplicate-text occurrences**, and **64 shards**. The Kaggle GPU capacity probe was confirmed complete. On 25 October, it verified sentiment shards 00–13 and submitted shard 14.

A read-only Kaggle diagnostic on 27 October checked shard 13, shard 14, shard 15 and the CPU original-event-weight backfill. Both **shards 13 and 14** were marked COMPLETE, their `corpus_shard_report.json` files were listed and successfully downloaded and parsed. Shard 15 was not registered. The **original-event-weight backfill** was also COMPLETE, with its `full_corpus_occurrence_manifest.json` listed and successfully parsed. These are file-presence and parse checks; the resumed controller is responsible for enforcing exact model coverage and hash-integrity requirements.

The previous controller stopped with `Completed private Kaggle summary unavailable: corpus_shard_report.json` despite the shard-14 report being retrievable by the diagnostic. The recovery patch now uses Kaggle's documented **regular-expression file-pattern** semantics for nested output paths, performs up to three bounded *metadata-read* attempts, and logs only filename/availability categories on failure. It does not automatically retry canceled/failed inference or expose tweet content. Synthetic regression tests for nested-path retrieval and approved report filenames passed CI.

The existing [gated full-corpus workflow](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/workflows/vax-full-corpus-inference.yml) was re-triggered on 27 October to revalidate prior shards and submit **at most one** additional GPU shard. Consult its run logs for submission and completion. Do not infer that all 64 sentiment or 64 exploratory stance shards have completed. The exploratory stance classifier still lacks task-matched independent accuracy validation.

Evidence: [read-only diagnostic](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/40474727433), [recovery run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/40475024682).
