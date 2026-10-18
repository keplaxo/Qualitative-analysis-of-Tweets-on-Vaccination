# Controlled ontology v2 study and independent reference evaluation

**Frozen protocol:** 18 October 2026. This branch adds an opt-in Qwen3 v2 author-stance ontology and controlled, private Kaggle execution. This project does not generate human approval labels and never treats another LLM as a human gold reference.

## Why another experiment is necessary

The version-1 300-case Qwen sample produced 290 structurally valid predictions and exact DeBERTa–Qwen stance agreement of 31/290 (10.7%). A separate CPU-only audit confirmed all 300 SHA-256 record matches, but found severe differences in how models used `uncertain`, `no_expressed_stance`, and `insufficient_context`. The changed ontology is a hypothesis to test, **not evidence that v2 is more accurate**. See [the frozen disagreement audit](STANCE_DISAGREEMENT_AUDIT.md).

## Prespecified design

Sample **60** unique historical cases from the 300 privately annotated Qwen v1 cases using a fixed SHA-256 rank seed (`vax-ontology-v2-20261018`), with four mutually exclusive strata:

| Stratum | Target |
| --- | ---: |
| DeBERTa insufficient context versus Qwen v1 uncertain | 20 |
| DeBERTa support versus Qwen v1 uncertain | 15 |
| DeBERTa support versus Qwen oppose, or DeBERTa insufficient versus Qwen oppose | 10 |
| DeBERTa and Qwen v1 exact agreement controls | 15 |
| **Total** | **60** |

The first **12** are selected by round-robin across the strata (three per group). Run `v2_pilot` first and require at least **9/12 structurally valid outputs**. If the gate passes, run `v2_full` on the same 60 keys, restore the 12 pilot outputs, and infer only the remaining 48. The full stage requires at least **45/60 structurally valid outputs**. Pilot and full predictions retain original SHA-256 identifiers only within **private Kaggle notebooks**; the public repository contains code and aggregate metadata only.

Version-2 rules include explicit author hesitancy evidence for `uncertain`, distinct `no_expressed_stance` and `insufficient_context` interpretation bases, and an explicit author-vaccine position for substantive support/opposition. They preserve vaccine mandate stance, safety concerns, adverse-event discussion and procedure fear as independent concepts. Evaluate the v2 schema-validity rate, v1-to-v2 transitions, and paired agreements separately. **This experiment measures label stability and structural coherence, not accuracy.** The original Qwen v1 and DeBERTa files are immutable inputs.

## Computation and reliability

[GPU experiment script](../kaggle/v2_experiment.py), [fixed sampler](../src/vaxbench/v2_selection.py), [v2 ontology](../src/vaxbench/ontology_v2.py), and [hourly gated scheduler](../scripts/advance_v2.py). The versioned notebook references are:

- [Pilot](https://www.kaggle.com/code/keplaxo/vax-stance-v2-pilot-20261018)
- [60-case follow-up](https://www.kaggle.com/code/keplaxo/vax-stance-v2-full-20261018)
- [Independent dataset CPU-only preflight](https://www.kaggle.com/code/keplaxo/vax-external-reference-probe-20261018)

The scheduler submits **at most one** missing private notebook per pass. It will not automatically retry a queued, running, failed or canceled notebook, nor launch large-corpus inference. Every stage must pass downstream output count, model-version, record identity and schema checks. The preflight is **not** a reference-model accuracy evaluation.

## Independent external reference: VaxxHesitancy

The University of Sheffield ICWSM 2023 **VaxxHesitancy V2** dataset is available at [Zenodo DOI 10.5281/zenodo.7601328](https://zenodo.org/records/7601328), published under CC BY 4.0. Its >3,100 annotated English COVID-19 vaccine tweets have labels pro-vaxx, anti-vaxx, vaxx-hesitant and irrelevant, and the source explicitly publishes separate `VaccineHesitancy_train_v2.csv` and `VaccineHesitancy_test.csv` files.

The gated CPU preflight retrieves **schema metadata only** and does not publish tweet text or actual examples. Preserve the **test set as held out**. Upon preflight success, freeze an exact mapping before testing classifiers, accounting for differences in what the external publication means by *vaxx-hesitant* and our strict v2 *explicit author indecision*. Do not mechanically count *irrelevant* as *no expressed stance*. Report aligned-label precision/recall/F1, evaluation coverage, explicit abstentions, irreducible ontology mismatches and effect of 2020s COVID-19-to-2006–2019 domain shift. Never claim external validity transfers directly to historical tweets.

## End conditions

The scheduled pipeline stops after the 12- and 60-record comparisons and the external schema preflight. Running the independent held-out classifier evaluation requires interpreting its confirmed schema and prespecifying the crosswalk. **No full historical dataset classification or population-level inference is authorized.**
