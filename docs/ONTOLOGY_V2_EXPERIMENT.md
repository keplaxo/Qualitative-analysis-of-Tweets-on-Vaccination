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

The completed CPU preflight verified that **VaxxHesitancy contains tweet IDs and independent annotation fields but no text**: test = 431 records, training = 2,790 records. The test split has 176 pro, 76 anti, 28 hesitant and 151 irrelevant (first annotation field). Direct model inference is therefore **not possible without lawful text rehydration**. No claim of test-set accuracy is made and no rehydration was attempted. Upon preflight success, freeze an exact mapping before testing classifiers, accounting for differences in what the external publication means by *vaxx-hesitant* and our strict v2 *explicit author indecision*. Do not mechanically count *irrelevant* as *no expressed stance*. Report aligned-label precision/recall/F1, evaluation coverage, explicit abstentions, irreducible ontology mismatches and effect of 2020s COVID-19-to-2006–2019 domain shift. Never claim external validity transfers directly to historical tweets.

## Separate, text-accessible intention proxy

To obtain at least one independently human-labeled evaluation without generating new annotations, a **separate published English resource** was identified: [VACMA-PUBLIC, Kummervold et al.](https://github.com/peregilk/VACMA-PUBLIC) linked to [JMIR Medical Informatics 2021](https://medinform.jmir.org/2021/10/e29584/). A pinned `tweets_200_labeled.csv` source includes tweet text and eight binary annotation columns. Of 200 text-bearing cases, **43 have intent to accept COVID-19 vaccination, 18 have intent to reject, 0 have both**. The other 139 lack either intention label, so they cannot be used as negative examples for this narrow test.

A frozen [out-of-domain intention proxy evaluation](../kaggle/vacma_intent_evaluation.py) is queued **only after successful v2 12/60 quality gates**. It will run Qwen3 v2 on the 61 eligible publicly sourced texts, without modifying prompts using this reference. The prespecified binary comparison is:
- published `Intent to accept` -> expected Qwen v2 `support`;
- published `Intent to reject` -> expected Qwen v2 `oppose`;
- other Qwen v2 labels and invalid parses -> explicit abstention/error, counted in denominator.

Report a 2 × 3 reference-versus-prediction table, class-specific precision and recall, strict accuracy including abstentions, macro-F1 and coverage. These metrics apply **only to the narrow author-intention proxy and its domain**, not to six-class historical vaccination stance. Tweet text and per-row reference labels remain within the private Kaggle execution and never enter public repository data or logs. Retain original VACMA citation and do not redistribute its raw tweets.

## End conditions

The scheduled pipeline stops after the 12- and 60-record comparisons, VaxxHesitancy ID-only schema preflight, and the separately scoped 61-case VACMA intention proxy evaluation. An exact independent six-way author-vaccination-stance benchmark remains unavailable here because the directly located VaxxHesitancy references are ID-only and the VACMA endpoint is intention rather than stance. **No full historical dataset classification or population-level inference is authorized.**
