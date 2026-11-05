# Research methods, targets and validation boundaries

**Updated 5 November 2026.** This describes exploratory model-assisted research, not an independently validated vaccine-hesitancy measurement instrument.

## Source and research design

The project is a retrospective computational analysis of vaccination-related tweets in a historical archive with observed years from **2006 to 2019**. The approved private Kaggle source is [keplaxo/twitter-vaccination-dataset](https://www.kaggle.com/datasets/keplaxo/twitter-vaccination-dataset), using master.csv and vaccination2.csv. Private indexing identified **2,285,081 source rows** and **1,944,361 distinct exact texts**. These records do not represent all Twitter/X users, all citizens, individual people, vaccine uptake or any other population frame.

The original development experiments used a **100-record smoke test**, a **2,000-text fixed-hash-ranked year-covered benchmark**, a **20-record Qwen pilot**, and a **300-case deliberately uncertainty-enriched semantic comparison**. A separate 3,208-case enrichment pool existed, but it was purposively selected, and it is not interchangeable with a probability sample. The full-corpus pipeline instead indexes and analyzes every eligible distinct text from the two archived sources.

## Data handling and reproducibility

1. Inspect the private files and identify usable columns without assuming matching source schemas. Existing text/date fields are retained; unknown dates are explicitly reported.
2. Construct a canonical **SHA-256 digest of exact text** for inference row keys. This prevents scientific-notation rounding of original tweet IDs from colliding at the model-join stage.
3. Deduplicate exact texts for computational efficiency, while retaining the **340,720 excess source occurrences**. A separate private original-event reconstruction allows both unique-text counts and original source-event-weighted analysis. Neither equals the number of unique authors.
4. Split distinct texts deterministically into **64 fixed SHA-256 partitions**. Check for disjoint sets, expected totals and exact record-key integrity.
5. Use automatic language screening, because an authoritative language column was not available in the observed master schema. This is an **unvalidated heuristic language screen**; a predicted English label is not language ground truth.
6. Keep source text, original IDs, hash joins, model evidence spans and per-record prediction outputs in controlled Kaggle storage. The public GitHub repository contains safe code, synthetic tests and aggregate evidence only.

## Distinct outcome axes

| Axis | Classes | Critical boundary |
| --- | --- | --- |
| Sentiment | negative, neutral, positive | A negative tweet may be negative about disease rather than vaccines |
| Author vaccine stance | support, oppose, uncertain, mixed, no_expressed_stance, insufficient_context | Neutral information, mandate positions and insufficient attribution are not automatically anti/pro vaccine views |
| Personal vaccination intention (v3) | accept, reject, undecided, not_expressed, insufficient_context | Plans to receive/refuse a vaccine are not the same as general advocacy or opinion |

In the stance ontology, uncertain requires evidence of **author-expressed indecision**. No-expressed-stance applies to intelligible content without a discernible author's vaccine position. Insufficient-context applies to genuinely missing, truncated or unattributable information. Past vaccination alone is not a future intention; reporting an adverse event is not proof of vaccine harm, and a vaccine-mandate view does not automatically identify vaccine stance.

## Inference models

**Multilingual sentiment:** [CardiffNLP XLM-T](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment), used for all eligible selected texts. The rationale for multilingual Twitter modeling is described by [Barbieri et al.](https://aclanthology.org/2022.lrec-1.27/).

**English sentiment:** [CardiffNLP Twitter-RoBERTa latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), applied only to English-screened text. Its training period differs from much of the historical corpus, which requires temporal-domain-shift caution; see [TimeLMs](https://aclanthology.org/2022.acl-demo.25/).

**Exploratory stance:** [DeBERTa-v3-large zero-shot NLI](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0), with six author-stance candidate descriptions and heuristic score/margin abstention. Its scores are not calibrated probabilities, and it has not been independently validated on this exact historical task.

**Capped semantic development:** [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), used for selected v1/v2/v3 comparisons with JSON schema constraints and verbatim-evidence-substring checks. A valid JSON response is not proof of semantic correctness. Qwen is not being run across 1.94 million unique records.

Versions, prompts, labels, git revision, model revisions, seed, device and runtime belong in the private model provenance. Previously completed experiment versions should be preserved rather than overwritten by later ontology changes.

## Results that cannot be interpreted as accuracy

The 2,000-record benchmark produced 2,000 multilingual and 1,666 English-screened sentiment and stance predictions. On the 1,666 paired English records, two sentiment models agreed on **1,340 (80.4%)** labels. The Qwen experiment produced 290 valid structured outputs from 300 selected records, with DeBERTa/Qwen agreement **31/290 (10.7%)**. Both measures describe model agreement, not independently checked truth.

The v2 60-case follow-up had **57 valid outputs**. The separate published [VACMA-PUBLIC](https://github.com/peregilk/VACMA-PUBLIC) intention-proxy experiment included **61** cases with an explicit published accept/reject intention, and its strict Qwen v2 proxy score was **12/61 (19.7%)**, macro-F1 **0.337**, with binary prediction coverage **18/61**. These are not six-way author vaccine-stance accuracy measures. V3 was developed after those results were visible, so a repeat on the same cases is post-hoc, not an independent test.

The independent [VaxxHesitancy V2](https://zenodo.org/records/7601328) annotation set is related but differs in label semantics and era. The inspected public CSVs contained tweet IDs/annotations rather than readable tweet text; direct testing without lawful rehydration was unavailable.

## Requirements before stronger scientific claims

An acceptable validation study requires an **untouched, lawfully usable, text-bearing and independently labeled reference** for the same target. Freeze the exact prompt, candidate labels and thresholds before evaluation; report source provenance, class-wise precision/recall, macro-F1, abstention and invalid-output rates, and errors by year and language. A test of personal intention cannot be silently described as stance validation, and a COVID-era sample cannot establish generalization to a 2006–2019 historical corpus without examining domain shift.

The final planned descriptive analysis distinguishes **unique-text** from **original-event-weighted** denominators, missingness, year and source coverage, model coverage and uncertainty. Model-predicted stance counts remain *exploratory predictions*, not population vaccine hesitancy rates or causal/clinical findings.

See the [project history](PROJECT_HISTORY.md), [operational runbook](PIPELINE_OPERATIONS.md) and [reference register](REFERENCES.md).
