# Vaccination Tweet Sentiment & Stance Benchmark

Reproducible research pipeline for model-assisted analysis of historical vaccination-related tweets (2006–2019). The pipeline separates **sentiment polarity** (positive, neutral, negative) from **the author's stance toward vaccination** (support, oppose, uncertain, mixed, no expressed stance, insufficient context).

**Active development:** [feature/kaggle-model-benchmark-v1](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/tree/feature/kaggle-model-benchmark-v1) ([pull request #1](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/pull/1)). The default branch, master, has not yet received this implementation.

## Verified progress — 17 October 2026

| Component | Verified status |
| --- | --- |
| Research code and synthetic unit tests | Implemented; latest branch CI succeeded |
| Kaggle API / hosted data connection | Authenticated GitHub Actions integration; dataset found and attached |
| Private dataset schema probe | **Complete**; master.csv and vaccination2.csv readable |
| 100-tweet GPU smoke benchmark | **Complete**; aggregate summary retrieved |
| 2,000-tweet GPU benchmark | **Resubmitted** at 15:54 UTC after a canceled run; initially **QUEUED**; latest completion/result unverified |
| Qwen semantic adjudication | Implemented as an optional, capped stage; **disabled** for smoke and 2,000-tweet runs |
| Independent in-domain accuracy / complete-corpus inference | **Not performed** |

The completed 100-record smoke run produced 100 multilingual sentiment predictions, 85 English sentiment predictions, and 85 exploratory zero-shot stance predictions. The two sentiment models agreed on **56 of 85** overlapping records (65.9% agreement, **not accuracy**). No independent human-reference labels were generated or approved as part of this workflow, by design; model agreement is not a substitute for task-specific validity evidence.

The previous smoke error involved rounded/scientific-notation source IDs creating duplicate prediction keys. Canonical inference keys now use exact-text SHA-256 digests. The successful smoke run passed the subsequent summary gate. The initial 2,000-record notebook was canceled without a verified summary, and the retry was only confirmed queued in the last available submission log. See [workflow and execution status](docs/WORKFLOW.md) for the dated audit trail and next decision gates.

## Repository guide

- [Project workflow and current execution status](docs/WORKFLOW.md): stages, GitHub/Kaggle hand-off, verified outcomes, failure recovery, and next steps.
- [Kaggle setup and operations](docs/KAGGLE.md): data mount, secrets, safe status monitoring, and launch controls.
- [Methods / research protocol](docs/METHODS.md): deterministic sampling, target definitions, model comparison, limitations, and validation plan.
- [Candidate models](docs/MODELS.md): model IDs and interpretation safeguards.
- [Private GPU benchmark notebook](kaggle/vaccination_benchmark.ipynb) and [schema probe](kaggle/schema_probe.ipynb).
- [Inference CLI](src/vaxbench/cli.py), [language screen](src/vaxbench/langscreen.py), [triage](src/vaxbench/triage.py), and [aggregate reporting](src/vaxbench/summarize.py).
- [GitHub Actions workflows](.github/workflows) and [synthetic tests](tests).

## Execution design

The hosted Kaggle dataset supplies authorized input; GitHub Actions submits a **private** Kaggle notebook; the notebook selects a deterministic development sample, applies language screening, runs multilingual and eligible English-language model stages, and saves checkpointed predictions and aggregate summaries **inside private Kaggle storage**. The optional Qwen stage is off for current automated benchmark modes. A GitHub status workflow can retrieve only approved aggregate reports. A successful GitHub Actions *submission* or passing pytest run does not prove Kaggle inference completed.

Current automated modes are smoke (100 records) and full benchmark (2,000 records), each with **zero enrichment records and Qwen disabled**. The separate planned 1,000-record enrichment and up-to-300-record Qwen review are not part of these modes. Neither development sample represents corpus prevalence.

## Local code verification

Python 3.11 is used in CI. From the repository root:

~~~bash
python -m pip install -e . pytest
python -m pytest -q
~~~

These synthetic tests verify software behavior, not classification quality. See [Kaggle operations](docs/KAGGLE.md) before executing models on authorized data.

## Research and data policy

**Public repository: code and documentation only.** Never commit tweet text, archives, tweet/user identifiers, licensed third-party corpora, Kaggle credentials, model weights, or per-record inference outputs. Keep study inputs and detailed artifacts in controlled research storage. Do not treat machine-generated labels, evidence substrings, cross-model agreement, or sentiment about disease/policy as independent proof of vaccine stance, model accuracy, or vaccine-related causality. Historical research checkpoints remain separate and immutable.

## Automated next-stage execution (18 October 2026)

The private 2,000-record Kaggle benchmark is **COMPLETE** (2,000 multilingual, 1,666 English sentiment and 1,666 zero-shot stance predictions). The following pipeline has been installed and activated on the repository's **master branch** [hourly GitHub Actions scheduler](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/master/.github/workflows/vax-kaggle-scheduler.yml):

\`Qwen 20-case pilot → Qwen 300-case semantic review with pilot reuse → CPU aggregate model comparison and research report\`.

The first [orchestrator run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39081406763) successfully **submitted** the 20-case pilot. The hourly controller checks completion and structural quality before launching each dependent stage. It never re-runs a canceled or failed notebook automatically. Automatic full-corpus inference is not part of this chain. Original tweets and per-record outputs remain in private Kaggle notebook inputs and outputs, not this public repository. See [workflow gates](docs/WORKFLOW.md#7-automatically-scheduled-qwen-and-research-report-stages-18-october-2026).
