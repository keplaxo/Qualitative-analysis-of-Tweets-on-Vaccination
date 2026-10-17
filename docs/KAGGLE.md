# Kaggle GPU execution and operational checks

**As of 17 October 2026:** the Kaggle API secret was verified via GitHub Actions. The hosted dataset is linked; the private schema probe and corrected 100-record GPU smoke benchmark completed. A 2,000-record Kaggle benchmark was **resubmitted** at 15:54 UTC after a cancellation and was initially QUEUED; completion of that retry is **not verified** in the last available GitHub logs. See [project workflow and audit evidence](WORKFLOW.md).

## Approved dataset and source schema

The dataset handle is [keplaxo/twitter-vaccination-dataset](https://www.kaggle.com/datasets/keplaxo/twitter-vaccination-dataset). The probe confirmed two readable files: master.csv and vaccination2.csv. For master.csv, the usable fields include tweet for text, date for timestamp, and id for source ID. The probe found no language column, so the pipeline uses automatic language screening. The Kaggle notebook discovers these headers rather than assuming an unrelated CSV schema.

Observed mount path:

~~~text
/kaggle/input/datasets/keplaxo/twitter-vaccination-dataset/master.csv
~~~

Alternative Kaggle mount layouts are supported by the notebook. The [schema probe notebook](../kaggle/schema_probe.ipynb) contains only safe schema extraction; never print raw tweet rows or identifiers to GitHub job logs.

## Authentication, privacy, and notebook lifecycle

The GitHub Actions secret KAGGLE_API_TOKEN is verified; legacy KAGGLE_USERNAME and KAGGLE_KEY are also supported by some scripts. Store credentials in **GitHub Actions secrets**, never in repository files, public notebook source, or output artifacts. Kaggle notebook metadata sets is_private=true and dataset_sources to the handle above. GPU and Internet are enabled for the benchmark; the schema probe uses CPU without Internet.

Automated submission uses [kaggle-benchmark.yml](../.github/workflows/kaggle-benchmark.yml) with [scripts/submit_kaggle_benchmark.py](../scripts/submit_kaggle_benchmark.py). The staged notebook is [vaccination_benchmark.ipynb](../kaggle/vaccination_benchmark.ipynb).

| Mode | Notebook | Historical records | Enrichment | Qwen |
| --- | --- | ---: | ---: | --- |
| smoke | [Private smoke kernel](https://www.kaggle.com/code/keplaxo/vax-sentiment-benchmark-smoke) | 100 | 0 | Off |
| full benchmark | [Private full kernel](https://www.kaggle.com/code/keplaxo/vax-sentiment-benchmark-full) | 2,000 | 0 | Off |

The full mode has a smoke-gate check: previous 100-record notebook COMPLETE, summary.json present, 100 multilingual outputs, and nonzero English and zero-shot coverage. Passing that gate allows **submission**, not a claim of later execution success. The additional 1,000-record enrichment pool and optional capped Qwen semantic stage are future **separate** experiment settings, not silently included in either automated mode.

Branch-based GitHub launch files are kaggle/launch-smoke.trigger and kaggle/launch-full.trigger. For the full trigger, the commit message must contain [full-benchmark] because the workflow derives mode from that literal marker. Avoid touching trigger files in ordinary documentation commits. GitHub manual dispatch and hourly schedules are affected by default-branch registration; while these workflows remain feature-only, do **not** rely on scheduled hourly status checks. After merge, verify action registration, least-privilege secrets, and schedules explicitly.

## Running and observing a private Kaggle version

The submitted notebook clones the feature branch, installs package/requirements, executes synthetic tests, prepares a deterministic sample, automatically screens languages, then runs XLM-R multilingual sentiment, English-screened RoBERTa sentiment, and English-screened DeBERTa NLI. Qwen is conditional, currently off. Output goes to /kaggle/working/vax_outputs, including JSONL checkpoints and summary.json. The notebook prints the cloned repository revision for provenance.

Use [kaggle-status.yml](../.github/workflows/kaggle-status.yml) and [scripts/check_kaggle_runs.py](../scripts/check_kaggle_runs.py) to query actual Kaggle kernel states and retrieve only safe aggregate summaries. For failures, [kaggle-diagnostics.yml](../.github/workflows/kaggle-diagnostics.yml) and [scripts/diagnose_kaggle.py](../scripts/diagnose_kaggle.py) expose limited categories and milestones, not full source text. Inspect private Kaggle notebook logs directly when redaction precludes detailed GitHub diagnosis.

State interpretation: QUEUED and RUNNING are incomplete; CANCEL_ACKNOWLEDGED is not a valid output; COMPLETE plus readable summary.json and expected stage coverage is a usable benchmark checkpoint. A green GitHub Actions submission check **alone** is not that evidence.

## Verified smoke checkpoint and known failure

At the 15:42 UTC status run on 17 October 2026, the corrected smoke kernel was COMPLETE with n_sampled=100, 100 multilingual predictions, 85 English predictions, and 85 exploratory zero-shot stance predictions. The initial smoke run had failed in aggregation because source IDs represented in scientific notation collided (for example, 2.44E+17). The prepare stage now uses exact-text SHA-256 as the inference record_key and deduplicates identical text. English/multilingual sentiment agreed on 56/85 common records (65.9% **agreement, not accuracy**).

The first full benchmark was subsequently canceled. At 15:50 UTC its kernel state was CANCEL_ACKNOWLEDGED and no summary.json was present. After resubmission, the last verified state was QUEUED at 15:54 UTC. **Do not report a completed 2,000-record benchmark without fresh Kaggle state and aggregate output evidence.**

## Checkpointing and troubleshooting

A JSONL stage can skip previously processed record_key values **only if its existing checkpoint file is available**. A new Kaggle version does not automatically inherit /kaggle/working contents of an old version; restore outputs securely before using resumption. Preserve the per-year manifest, model revisions, Git revision, runtime, and hardware metadata.

If a notebook fails, inspect in this order: Kaggle execution state and status timestamps; GPU availability/quota and model downloads; dataset_sources/mount path and chosen text/date columns; identifier/duplicate-key checks; dependency or memory failures; expected JSONL files; summary.json and coverage. Never classify an unfinished, canceled, or missing-summary run as successful. Keep all raw data and row-level outputs in controlled private project storage.
