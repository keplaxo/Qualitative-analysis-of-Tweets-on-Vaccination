# Vaccine stance vs personal intention: post-hoc v3 development

Protocol: 18 October 2026. This is model development **after seeing the previous 61-case reference scores**, and cannot be called independent or confirmatory accuracy validation.

## Findings requiring follow-up

On 60 purposively selected historical tweets, 57 Qwen v2 classifications passed structural validation. Of those 57, 27 shifted from original Qwen v1 uncertain to v2 no_expressed_stance. V2/v1 matched on 22/57 and v2/DeBERTa on 11/57; these are agreement measures only.

In the published VACMA-PUBLIC 61-case intention proxy, v2 returned a binary support/oppose class for just 18/61 records. Among 43 reference intentions to accept vaccination, v2 called one support, six oppose and 36 abstained/other. Among 18 reference intentions to reject, v2 called 11 oppose and seven abstained/other. The 12/61 (19.7%) strict proxy score and 0.337 macro-F1 do NOT estimate six-class historical stance accuracy. A trivial majority-positive prediction would score 43/61 (70.5%).

## Proposed construct distinction

V2 attempts to infer the author's stance toward vaccines (opinion/advocacy), whereas the reference concerns their personal intention to get vaccinated. The statement "I intend to be vaccinated" conveys acceptance intention, but need not assert general support for vaccination. Bare past receipt of a vaccine is also neither future intention nor necessarily advocacy. Confusion between these endpoints is a plausible, *unverified* explanation of v2 failure.

The [CPU-only failure audit](../kaggle/v2_intent_failure_audit.py) checks saved private predictions against the 61 published references, stratified by reference intent, predicted label, interpretation basis, presence of quoted evidence and coarse linguistic features. It exports aggregate counts only and does no new model inference.

[Ontology v3](../src/vaxbench/ontology_v3.py) explicitly uses two independent outputs: (A) six-category author stance, unchanged from v2, and (B) five-category personal intention: accept, reject, undecided, not_expressed, insufficient_context. Each axis has separate interpretation and exact text evidence. A past vaccination statement alone must not be construed as planned uptake or advocacy. Synthetic cross-axis tests are in [test_ontology_v3.py](../tests/test_ontology_v3.py).

## Limited experiment

The [private Qwen3 script](../kaggle/v3_intention_development.py) selects 16 published VACMA cases as a balanced exploratory preflight, eight with published positive intention and eight with negative intention, ranked deterministically by a frozen hash seed. The minimum continuation gate is 12/16 structurally valid records. If passed, the full 61-case development sample reuses the original 16 and infers only the remaining 45, requiring 46/61 valid predictions. Detailed predictions stay in private Kaggle.

Published intent to accept corresponds to the v3 personal-intention label accept and intent to reject to reject. Other predictions and invalid outputs count as abstentions for the strict binary development score. Scores using these SAME previously evaluated 61 examples must be explicitly marked POST-HOC DEVELOPMENT; exposure to earlier errors precludes independent comparison and confirmed improvement claims. No per-case manual annotation or approval stage is added.

The [hourly gated controller](../scripts/advance_v3.py) runs the post-hoc CPU diagnosis and then a GPU pilot and 61-case follow-up. Never auto-retry an error or canceled notebook and never submit full historical corpus inference.

## Need for untouched external evidence

No additional independently annotated, text-accessible and task-matched vaccine intention evaluation set has yet been verified. [VaxxHesitancy](https://zenodo.org/records/7601328), [Lindelof et al.](https://github.com/GabrielLindelof/Vaccine-Discourse-on-Twitter-During-the-COVID-19-Pandemic), and [WebImmunization](https://huggingface.co/datasets/webimmunization/COVID-19-vaccine-attitude-tweets) provide tweet IDs rather than text; they cannot be evaluated without lawful authorized hydration. The remaining 139 VACMA items have neither positive nor negative intention labels and cannot be treated as negative references.

Obtain an unseen authorized text-bearing dataset with independent labels for the same outcome, freeze v3 and its label crosswalk, and report class-wise performance, abstention and domain transfer. Until then, describe v3 performance as developmental/feasibility only and do not deploy full historical stance inference or claim representativeness.
