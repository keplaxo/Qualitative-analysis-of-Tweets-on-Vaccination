# Vaccination stance model disagreement audit

**Audit completed:** 18 October 2026, 07:42 UTC. **Status verified:** Kaggle COMPLETE with private \`stance_disagreement_audit.json\`, read through GitHub Actions at 07:49 UTC.

- [Private Kaggle CPU audit](https://www.kaggle.com/code/keplaxo/vaccination-stance-disagreement-audit)
- [GitHub safe aggregate audit status](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39109108830)
- [Reproducible CPU audit code](../kaggle/stance_disagreement_audit.py)
- [Prior Qwen 300 stage](https://www.kaggle.com/code/keplaxo/vax-qwen-300)
- [Prior 2,000 record model benchmark](https://www.kaggle.com/code/keplaxo/vax-sentiment-benchmark-full)

## 1. Purpose, study design, and provenance

Investigate the observed agreement of 31/290 (10.69%) between DeBERTa zero-shot author-vaccination stance and Qwen3 semantic author-stance predictions without conflating agreement with predictive accuracy. **No new sentiment, zero-shot, or Qwen model inference** was required. The CPU audit used saved predictions kept in two *private* Kaggle notebook outputs; the GitHub logs and committed scripts expose aggregate counts only.

The base benchmark contains 2,000 exact-text distinct year-covered historical records. Its English-screened DeBERTa arm has 1,666 predictions. Qwen evaluated 300 purposively selected records, approximately 60% ambiguous/abstained NLI cases and 40% hash-ranked comparison cases, with 290 structurally valid annotations (96.7%). These 300 are **not a representative sample**.

**Integrity checks:** 300/300 text hashes recomputed from saved source text matched the canonical record key; all triage, Qwen, DeBERTa, and sample keys are unique and matched. Of 300 Qwen predictions, 10 failed schema/parsing checks (recorded as ValueError); their substantive stance was not compared. There is no evidence of a key-join mismatch causing the 10.7% agreement.

## 2. Exact label agreement

| DeBERTa \ Qwen | Support | Oppose | Uncertain | Mixed | No expressed stance | Insufficient context | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Support | 7 | 6 | 30 | 0 | 6 | 0 | 49 |
| Oppose | 0 | 16 | 6 | 0 | 0 | 0 | 22 |
| Uncertain | 0 | 0 | 2 | 0 | 0 | 0 | 2 |
| Mixed | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| No expressed stance | 1 | 0 | 14 | 0 | 6 | 0 | 21 |
| Insufficient context | 6 | 34 | 130 | 0 | 26 | 0 | 196 |
| **Total** | **14** | **56** | **182** | **0** | **38** | **0** | **290** |

**Overall exact agreement:** 31/290 = 10.69%; discordant classifications: 259/290. The single largest direction, DeBERTa insufficient_context versus Qwen uncertain, accounts for 130/259 discordances (50.2%). Qwen **never assigned insufficient_context**, while DeBERTa assigned it 196 times; conversely Qwen assigned uncertain 182 times, DeBERTa just twice.

## 3. Operational drivers of disagreement

| Comparison set | Paired N | Exact matches | Exact agreement |
| --- | ---: | ---: | ---: |
| Full selected Qwen set with valid annotation | 290 | 31 | 10.7% |
| Low-margin/abstention enriched | 176 | 0 | 0% |
| Hash-ranked comparison | 114 | 31 | 27.2% |
| DeBERTa formally abstained | 177 | 0 | 0% |
| DeBERTa not formally abstained | 113 | 31 | 27.4% |
| Qwen-labeled human vaccine context | 241 | 28 | 11.6% |
| Qwen explicit interpretation basis | 15 | 8 | 53.3% |
| Both labels in substantive set (support, oppose, mixed, uncertain) | 67 | 25 | 37.3% |

Of 176 structurally valid enriched cases, DeBERTa assigned insufficient_context in every case, while Qwen assigned uncertain to 116, oppose to 28, no_expressed_stance to 26, and support to six. The overall rate is therefore sensitive to intentional uncertainty enrichment and the asymmetric use of an abstention label.

On 290 valid pairs, Qwen declared 241 relevant to human vaccination, 27 to animal vaccination, 11 unrelated and 11 unclear. Interpretation basis was explicit in 15, inferred in 154, unresolved in 96, and not_expressed in 25. Qwen returned nonempty source-substring evidence in 170/290; an empty evidence field in 120/290. This is a **format/evidence check, not a semantic correctness score**.

Mean DeBERTa NLI top-two candidate margin was 0.175 among discordant cases and 0.620 among matching cases. NLI scores are heuristic, not calibrated confidence estimates. The Qwen 10 invalid outputs were excluded rather than coerced into a label.

## 4. Diagnosis

1. **No detectable joining failure.** Full exact-text SHA-256 lineage and paired-key integrity passed.
2. **Asymmetric uncertainty ontology.** Qwen systematically used uncertain for content DeBERTa labeled insufficient_context, including 130 direct pairs. The prompt specifies both labels but does not enforce a distinct explicit *author-expressed vaccine hesitancy* test before uncertain.
3. **Abstention threshold plus targeted selection.** DeBERTa overrides low-top-score/low-margin predictions to insufficient_context. The purposive Qwen selection oversampled such cases. Its 0% agreement on this stratum is a predictable interaction with Qwen's zero use of insufficient_context.
4. **Substantive disagreement remains.** The comparison stratum agreed only 27.2%, and the substantive-label subset only 37.3%. Six DeBERTa support cases became Qwen oppose and 30 became Qwen uncertain. These cannot be resolved by a cosmetic label crosswalk.
5. **No independent reference.** Neither model is established as correct; retaining one model as a pseudo-gold source would create confirmation bias. Automatic schema validation (290/300) measures parseability, not accuracy.

Merging uncertain into insufficient_context would mechanically increase exact agreement by 130 cases to 161/290 (55.5%), but **this recoding has no substantive justification** without task-specific evidence, and it is **not a corrected agreement or accuracy estimate**. Those categories answer different questions.

## 5. Ontology for the next experiment (v2, to be prespecified)

| Label | Operational criterion | Differentiation |
| --- | --- | --- |
| support | Author expresses approval of vaccine(s), vaccination, or intention to vaccinate | Receiving a vaccine or repeating a headline is not endorsement by itself |
| oppose | Author expresses opposition to vaccination or vaccines as such | Opposing a mandate, reporting harm, or expressing concern is not automatically opposition |
| uncertain | The **author explicitly expresses indecision, hesitancy, or unresolved personal vaccine choice** | Do not use as a catchall when author intention cannot be inferred |
| mixed | Author expresses positive and negative vaccine positions within the same content | Ambivalence must be evidenced; not merely differing views quoted |
| no_expressed_stance | Tweet is interpretable and on topic but makes no ascertainable author stance claim (e.g., neutral information) | Do not infer the author's position from the topic |
| insufficient_context | The text, source attribution, ambiguity, or truncation prevents determining whether a relevant author stance is expressed | This is an abstention because evidence is insufficient, not the same as author hesitancy |

**Required inference procedure:** First decide human vaccination relevance; second decide whether the author expresses a stance *rather than quoting another person or mentioning a mandate*; third require evidence for any support/oppose/uncertain/mixed label; fourth use no_expressed_stance for comprehensible neutral statements and insufficient_context for underspecified attribution/context. Distinctly report mandate_stance, adverse_event_report, procedure_fear and safety_concern; these features must not determine author vaccine stance alone.

Enforce structured JSON validation separately from stance correctness. Proposed tests should flag missing evidence on purportedly explicit stance, unresolved basis paired with forced support/oppose, and uncertain without express author hesitancy for *review or re-inference*, rather than silently changing the class.

## 6. Controlled next experiments

**A. Prespecified v2 semantic re-inference pilot:** Freeze the present Qwen and DeBERTa output versions. Use the revised ontology on a small held-out/disagreement-enriched comparison set; include cases from the \`insufficient_context → uncertain\` cell, direct pro/anti conflicts, Qwen explicit-basis cases, and routine comparison examples. Save versioned outputs in a new private Kaggle notebook; never overwrite the 300 original classifications. Report structural validity, shifts between uncertainty categories, abstention, and within/between-model stability, not accuracy.

**B. Independent automatic benchmarking:** Find legally accessible, task-matched published vaccine-stance evaluation corpora with independent reference labels. No new human labels or manual approval stage are planned. Freeze prompting and candidate labels *before* benchmark evaluation. Report exact reference provenance, class-specific F1/recall, abstention, error categories and domain mismatch. Do not equate sentiment, mandate stance and vaccine stance benchmarks.

**C. Deployment gate:** Do not deploy either stance model across the full historical corpus or report vaccination stance prevalence until the label ontology is stable, off-domain accuracy evidence exists, the effects of purposive sampling are quantified, and the full-corpus sampling/inference design is agreed. Preserve the complete set of 2,000 and 300 private output artifacts.

All percentages in this document characterize agreement or output structure on a selected development sample. They are **not** population estimates, medical truth validation, or model accuracy.
