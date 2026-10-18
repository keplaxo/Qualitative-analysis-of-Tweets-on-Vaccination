# Vaccination Tweet Sentiment & Stance Benchmark

Reproducible research pipeline for model-assisted analysis of historical vaccination-related tweets (2006–2019). The pipeline separates **sentiment polarity** (positive, neutral, negative) from **the author's stance toward vaccination** (support, oppose, uncertain, mixed, no expressed stance, insufficient context).

**Active development:** [feature/kaggle-model-benchmark-v1](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/tree/feature/kaggle-model-benchmark-v1) ([pull request #1](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/pull/1)). The default branch, master, has not yet received this implementation.

## Active ontology v2 evaluation — 18 October 2026

- **Controlled Qwen3 v2 pilot submitted:** 12 deterministic, interleaved examples from the saved 300-case private comparison. Completion has **not** been confirmed; last checked Kaggle state was QUEUED.
- **Automated next steps on GitHub:** if at least 9/12 pilot results are structurally valid, run a versioned 60-case comparison reusing those 12 results. Require at least 45/60 valid outputs. The GitHub default-branch [controlled experiment workflow](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/workflows/vax-v2-controlled.yml) checks progression hourly and never automatically retries failed/queued kernels.
- **Public reference preflight completed:** [VaxxHesitancy V2](https://zenodo.org/records/7601328) has 431 held-out rows but only tweet IDs plus annotation fields, **not original tweet text**. It cannot directly evaluate an inference model without lawful tweet rehydration.
- **Secondary, task-qualified, independent reference prepared:** [VACMA-PUBLIC](https://github.com/peregilk/VACMA-PUBLIC) has 200 published annotated English texts, with 43 intention-to-accept and 18 intention-to-reject examples. A separate 61-case Qwen3 v2 **intention-proxy** evaluation is scheduled after successful v2 12/60 gates. Intent labels are not exact six-class vaccination-stance ground truth.
- **No results yet** for v2 re-inference or external intention-proxy accuracy. Original model predictions and the 290-pair disagreement audit remain intact, without any full historical corpus launch.

See the [frozen experimental design and reference caveats](docs/ONTOLOGY_V2_EXPERIMENT.md). Private per-tweet data never enters public GitHub artifacts or logs.

## Verified progress — 18 October 2026

| Component | Verified status |
| --- | --- |
| Private Kaggle dataset and schema preflight | **Complete**, both source CSVs readable |
| 100-record GPU smoke and 2,000-record base benchmark | **Complete**, preserved Kaggle summaries |
| Sentiment (XLM-T, Twitter-RoBERTa) and zero-shot DeBERTa stance | **Complete** for eligible development records |
| Qwen3 pilot (20) and targeted review (300) | **Complete**; 290/300 structurally valid outputs |
| Private CPU-only DeBERTa–Qwen disagreement audit | **Complete**, all 300 record hashes verified |
| Aggregate Qwen stance agreement | **31/290 (10.7%)**, agreement **not** model accuracy |
| Key diagnosis | Disagreement concentrated in `insufficient_context` versus `uncertain`, amplified by purposive uncertain-case enrichment |
| Full-corpus stance inference and population-level conclusions | **Held** pending ontology clarification and independent benchmark validation |

See [the verified disagreement audit](docs/STANCE_DISAGREEMENT_AUDIT.md), including a full label cross-tab, methodologic limitations, diagnostic subgroup results, and a controlled v2 evaluation plan. The audit used saved private outputs without rerunning either model or copying tweet texts into GitHub.

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
