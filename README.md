# Vaccination Tweet Sentiment & Stance Research

**Research author:** Dr William Kane Olwit  
**Documentation checkpoint:** 5 November 2026

I returned to an archived collection of vaccination-related tweets to investigate something that a simple positive/negative sentiment score could not capture: what, if anything, did the author actually say about vaccination? Negative language about an infection, opposition to a vaccine mandate, and an expressed decision not to be vaccinated are different things. This project develops a reproducible model-assisted pipeline that treats those distinctions seriously.

This public repository documents the research questions, data processing, model comparisons, failed attempts, workflow repairs and current full-corpus analysis. It does **not** publish source tweets, identifiers or individual predictions, and it does **not** claim independent accuracy for vaccination stance.

## Verified status as of 5 November 2026

| Component | Evidence-based status |
| --- | --- |
| Original source rows | 2,285,081 across two private Kaggle CSVs |
| Exact-text distinct records | 1,944,361; 340,720 duplicate occurrences |
| Processing partitions | 64 deterministic SHA-256 shards |
| Development stages | 100-record smoke, 2,000-record benchmark, Qwen 20-record pilot and 300-record review complete |
| Qwen vs DeBERTa stance agreement | 31/290 (10.7%) valid paired predictions, **not accuracy** |
| Qwen v2 controlled experiment | 57/60 structurally valid outputs |
| Published VACMA intention-proxy comparison | 12/61 (19.7%) strict alignment, macro-F1 0.337; **not six-class stance accuracy** |
| Full-corpus sentiment | **45/64 shards validated (indices 00–44)**; shard 45 **submitted**, not verified complete |
| Exploratory full-corpus stance | **0/64 verified** at the latest controller checkpoint |
| Final aggregation and full report | Not verified complete |

The latest [GitHub Actions evidence (5 November)](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41956954067) reported a validated sentiment prefix of 45, zero validated stance shards and submission of sentiment shard 45. A green GitHub workflow run can mean successful control and submission, **not completed Kaggle inference**. This table is a dated checkpoint, not a live counter.

## Objectives and analytical design

The four practical questions are how sentiment varies in historical vaccine-related discourse; whether English-specific and multilingual social-media models agree; whether author stance differs from emotional tone, policy discussion and individual vaccination intention; and how much uncertainty remains when models disagree.

The historical archive covers observed years **2006–2019**. The [private approved Kaggle dataset](https://www.kaggle.com/datasets/keplaxo/twitter-vaccination-dataset) supplies master.csv and vaccination2.csv. Preparation preserves date/source provenance, calculates canonical exact-text SHA-256 keys, deduplicates for inference, and separately reconstructs original-event weights so that repeated posts do not disappear from later reporting.

Three task types remain distinct:

- **Sentiment:** positive, neutral or negative. The pipeline uses [Twitter XLM-T](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) across eligible texts and [Twitter-RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) on English-screened records.
- **Author vaccine stance:** support, oppose, uncertain, mixed, no expressed stance or insufficient context. The [DeBERTa zero-shot NLI](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) component is **exploratory and not independently validated** for the historical vaccine-stance task.
- **Author vaccination intention:** accept, reject, undecided, not expressed or insufficient context. This is a separate v3 development axis, not a synonym for stance. [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) was used for capped semantic experiments, **not** mass inference on the full corpus.

No human-label approval pipeline is required by the project. This places particular importance on appropriately matched independent published benchmarks and explicit abstention. Cross-model agreement, JSON validity, and source-text evidence substrings are not ground-truth accuracy. Model-predicted stance rates must not be reported as true population vaccine-hesitancy prevalence.

## Repository architecture and documentation

The default [master branch](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/tree/master) holds registered GitHub Actions orchestration, the research record and public progress articles. The implementation, Kaggle scripts, tests and original detailed protocols are on the [feature/kaggle-model-benchmark-v1 branch](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/tree/feature/kaggle-model-benchmark-v1), associated with [draft pull request #1](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/pull/1). That PR has not been merged.

- [Research history and dated evidence](docs/PROJECT_HISTORY.md) reconstructs the process from the October restart through the latest run, including unsuccessful attempts.
- [Research methods and validation boundaries](docs/RESEARCH_METHODS.md) specifies source handling, denominators, model targets and interpretation.
- [Pipeline operations and recovery](docs/PIPELINE_OPERATIONS.md) explains dependencies, stages, failures and safe progression.
- [Scientific sources and primary execution references](docs/REFERENCES.md) provides model cards, studies, datasets, commits and Actions runs.
- [Progress articles](progress-articles/README.md) contains blog-ready first-person articles, with a consistent voice and linked evidence.
- The feature branch retains the [original workflow protocol](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/WORKFLOW.md), [full-corpus runbook](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/FULL_CORPUS_PIPELINE.md), [v2 ontology record](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/ONTOLOGY_V2_EXPERIMENT.md) and [v3 intention record](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/STANCE_INTENTION_V3_DEVELOPMENT.md).

The feature branch uses Python 3.11, synthetic unit tests and private Kaggle notebooks for actual inference. Passing code tests establishes implementation behavior, not scientific validity. Do not copy private tweet text, tweet IDs, dataset row hashes, model weights, credentials, per-record predictions or licensed reference datasets into this public repository.

**Next objective:** validate sentiment shard 45, finish the remaining sentiment shards, evaluate separately staged exploratory stance outputs, complete original-event-weighted aggregation, and resolve independent target-matched model validation before interpreting vaccine stance distributions.
