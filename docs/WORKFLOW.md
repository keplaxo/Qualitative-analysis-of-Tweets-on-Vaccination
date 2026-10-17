# Project workflow, execution evidence, and release gates

**Last documentation update:** 17 October 2026. **Active branch:** feature/kaggle-model-benchmark-v1. **Base branch:** master. **PR:** [#1](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/pull/1), open at the last inspection. This document distinguishes implemented code, completed notebook inference, GitHub job submission, and research validation.

## 1. Research workflow

1. **Controlled data source.** The existing Kaggle dataset [keplaxo/twitter-vaccination-dataset](https://www.kaggle.com/datasets/keplaxo/twitter-vaccination-dataset) supplies master.csv and vaccination2.csv; source rows are not copied to GitHub. The private schema probe confirms both CSVs are readable. In master.csv, the relevant observed columns include tweet (text), date, and id. No language field was identified by the probe.
2. **Deterministic development sampling.** The prepare command reads IDs as strings when provided, derives the **canonical record_key from an exact-text SHA-256 digest**, deduplicates exact texts, and selects hash-ranked, calendar-year-covered observations from 2006–2019. The 2,000-case stage is a stratified development benchmark, not a representative prevalence survey. The planned enrichment extension is a separately flagged, purposive sample and is excluded from automated runs.
3. **Language screening.** Existing language labels are retained when present; otherwise an automatic langid screen is applied. This screening is not a gold language annotation. Do not assume all records are English.
4. **Sentiment inference.** CardiffNLP Twitter XLM-R runs on the whole selected sample. CardiffNLP Twitter-RoBERTa runs on English-screened records only. Predictions, revisions, timing, and coverage are checkpointed in private JSONL output.
5. **Exploratory stance inference.** DeBERTa-v3 zero-shot NLI runs on English-screened records and uses six author-stance candidates and heuristic abstention thresholds. It is a comparator, not a validated multilingual vaccine-stance classifier.
6. **Optional semantic adjudication.** Qwen3-4B-Instruct-2507 can analyze up to 300 selected records, emphasizing low-margin/abstention records (60%) plus hash-ranked comparison controls (40%), with calendar-year coverage. Outputs must pass structured schema and exact-evidence-substring checks or be marked needs_reassessment. **This stage was not enabled for the completed smoke run or the submitted 2,000-record run.** No manual annotation or label-approval gate is part of the agreed pipeline.
7. **Aggregate analysis.** The summarizer reports model-level coverage, label distributions, revision IDs, and prediction agreement, never relabeling agreement as accuracy. Save aggregate report plus private run artifacts and provenance before drawing research conclusions.

## 2. GitHub → Kaggle orchestration

- The implementation and Actions workflows currently reside on the **feature branch**. [Unit-test CI](../.github/workflows/test.yml) runs for pushes and PRs; passing tests verify deterministic code and synthetic fixtures only.
- [Dataset discovery](../.github/workflows/kaggle-dataset-discovery.yml) uses Kaggle credentials from GitHub Actions secrets and checks metadata without downloading source rows to GitHub.
- [Schema preflight](../.github/workflows/kaggle-schema-preflight.yml) submits a private CPU Kaggle notebook, attaches the dataset, and retrieves only schema metadata.
- [Benchmark submission](../.github/workflows/kaggle-benchmark.yml) uses an approved secret, attaches the hosted dataset, enables GPU and notebook Internet, and submits a **private** Kaggle kernel. The smoke configuration sets 100 main / 0 enrichment / no Qwen. The full configuration sets 2,000 main / 0 enrichment / no Qwen. Full submission requires a completed, sufficiently covered 100-record smoke summary first.
- For branch-based push launch, modifying kaggle/launch-smoke.trigger starts smoke. Modifying kaggle/launch-full.trigger with **[full-benchmark]** in the commit message selects full mode. Only an intentional launch-trigger change should initiate a submission. GitHub Actions workflow_dispatch may be available after the workflows are registered on the default branch; do not assume this is operational while the workflows exist only on the feature branch.
- [Status checker](../.github/workflows/kaggle-status.yml) queries three private notebook references and retrieves approved schema/summary JSON where available. [Diagnostics](../.github/workflows/kaggle-diagnostics.yml) categorizes failures without exporting tweet content. **Important:** GitHub's schedule events run from the repository's *default branch*; the status cron defined only on this feature branch does not establish active hourly monitoring until merged/installed on master.

A GitHub job marked success can mean only that the Kaggle kernel was submitted. A notebook state of QUEUED or RUNNING is not model completion. A Kaggle kernel marked COMPLETE must additionally have a readable summary.json and expected stage coverage before being treated as a valid checkpoint.

## 3. Evidence table (UTC, 17 October 2026)

| Checkpoint | Direct evidence | Interpretation |
| --- | --- | --- |
| Data preflight | Private schema probe COMPLETE; master.csv and vaccination2.csv readable | Authorized dataset mounted, source schema known |
| First smoke attempt | Kaggle diagnostic showed duplicate prediction key caused by scientific-notation ID | Initial attempt failed; do not count as successful inference |
| Corrected smoke | [GitHub status run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39046209397) at 15:42 UTC: smoke COMPLETE; summary.json retrievable | **100** sampled; **100** multilingual sentiment; **85** English sentiment; **85** zero-shot stance |
| Cross-model sentiment | Same smoke summary: **56/85** paired labels match (65.9%) | Agreement, **not** independently verified accuracy |
| First 2,000-tweet attempt | [Cancellation diagnostic](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39046808459) at 15:50 UTC: CANCEL_ACKNOWLEDGED, no summary.json | No completed full-stage output established |
| 2,000-tweet retry | [Submission log](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39047084057) at 15:54 UTC: submission SUCCESS; initial Kaggle state QUEUED | Job submitted; **no later Kaggle completion/summary verified by this audit** |
| Latest tested feature head before documentation | 36123bf1a524f15f91cd76c295216e2b57c35522; [CI succeeded](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39047086976) | Code-level test pass, not scientific validation |

Smoke summary label distributions and model revisions are retained in the [status-run log](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39046209397); they are model outputs for a nonrepresentative sample and should not be presented as population results.

## 4. Output contract and provenance

The Kaggle notebook writes under /kaggle/working/vax_outputs. Main artifacts include benchmark_sample.csv, its .manifest.json, benchmark_sample_screened.csv, xlmt_sentiment.jsonl, roberta_english_sentiment.jsonl, deberta_stance.jsonl, summary.json, and optionally qwen_triage.csv and qwen_semantics.jsonl. Per-record files contain private study content and must **not** be uploaded as public Actions artifacts. Status scripts are restricted to approved aggregate summaries. Preserve the Kaggle kernel version, Git commit/revision printed by the notebook, SHA-256 text digests, model commit hashes, date/time, device, sample seed, and stage completion counts.

JSONL checkpoint resumption works only when prior output files are present; Kaggle ephemeral working directories do not automatically survive new runs. A canceled/failed run is not resumed simply by pushing a fresh notebook version. Saved Kaggle version outputs must be explicitly restored in a private session before attempting genuine stage-level resume.

## 5. Recovery rules and next decisions

When errors occur, check Kaggle kernel state and sanitized diagnostics first. Confirm input mount and selected columns, library imports and model download capability, GPU memory/quota, unique canonical record keys, and existence of summary.json. The corrected ID collision is specifically handled by text-hash-based record keys; never merge per-stage predictions on rounded source IDs. Do not repeatedly resubmit while a notebook is queued/running, and do not infer success from submission logs.

**Next gate:** confirm the retried 2,000-record Kaggle kernel reached COMPLETE and retrieve summary.json with expected stage coverage. If it failed or was canceled, inspect safe diagnostic categories and repair only the demonstrated issue before resubmission. Then examine throughput, language coverage, abstention, label disagreement, and plausible external benchmark compatibility. Consider the enrichment/Qwen extension only after compute and quality review. Before any full-corpus application or manuscript accuracy claim, establish **independent task-matched validity evidence**; no in-domain accuracy was estimated here. No cost-incurring inference expansion is implicitly authorized by updating these documents.

## 6. Security and research boundaries

Only approved public code, documentation, and synthetic tests belong in GitHub. Keep Kaggle secrets in GitHub Actions secrets; never paste keys into notebooks or log them. Keep raw tweet texts, IDs, source archives, licensed evaluation data, and per-record prediction artifacts private. Preserve pre-existing historical analyses rather than overwriting them. Sentiment is not stance; opposition to a mandate, receipt of vaccination, reporting adverse events, and endorsement of a quoted statement must not be automatically conflated.
