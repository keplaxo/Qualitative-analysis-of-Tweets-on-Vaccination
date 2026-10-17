# Kaggle GPU execution guide

This repository contains a Kaggle-compatible Python notebook at
kaggle/vaccination_benchmark.ipynb. Use a **private** Kaggle dataset containing
the historical source CSV, and optionally a private enrichment-pool CSV.

1. Enable GPU accelerator and Internet for model downloads, subject to account
   restrictions and available quotas.
2. Add only the private data sources to the Kaggle notebook.
3. Set MASTER_CSV and optionally ENRICHMENT_CSV to real paths under
   /kaggle/input. Inspect the listed input CSV paths first.
4. Set TEXT_COLUMN / DATE_COLUMN / ID_COLUMN / LANGUAGE_COLUMN to the actual
   source schema. IDs must be read as strings.
5. Run the notebook. The outputs are under /kaggle/working/vax_outputs.
6. Save the Kaggle notebook version and its output artifacts. The JSONL stages
   can resume from their existing output files within the same or a restored
   session, but Kaggle ephemeral storage does not persist across unrelated runs
   unless outputs are explicitly saved/reintroduced.

Avoid putting access keys into notebook source, GitHub, or public Kaggle
datasets. The source corpus is not currently mirrored to the public repository.

**Current access:** GitHub write access is confirmed. Direct authenticated
Kaggle CLI submission is *not* configured in the present execution environment.
The notebook is prepared but must not be described as a completed GPU run.
