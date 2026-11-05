# When two models could barely agree

*Research journal, Part 3 — 18 October 2026*

*The surprising result wasn't that the models made mistakes. It was how differently they interpreted uncertainty.*

The original plan included a limited comparison between a zero-shot stance classifier and a more contextual language model. DeBERTa would provide a structured first pass on English-screened texts. Qwen3 would examine a capped set of difficult tweets and return a richer, constrained interpretation.

I deliberately selected more low-confidence and abstained cases than a routine sample would contain. The goal was to stress-test the uncertain parts of the classifier, not to estimate the frequency of opinions in the dataset. The Qwen stage examined 300 texts, and 290 responses passed the required structural checks.

Then came a number I was not expecting: **the two models agreed on 31 of 290 labels, or 10.7%.**

It was tempting to interpret that immediately as one model doing a poor job. Instead, I checked whether we had made a more basic mistake. Were predictions being joined to the wrong tweet? Had the earlier ID collision returned in another form? A private CPU-only audit recomputed the hashes of all 300 source texts and verified the paired records. The exact matching passed.

That shifted attention from the pipeline to the ontology.

One category accounted for an extraordinary share of the problem. In **130 cases**, DeBERTa assigned insufficient context while Qwen assigned uncertain. There were **259 total mismatches**, so that single pairing represented about half of the disagreement. On the 176 structurally valid texts deliberately enriched from low-margin or abstained DeBERTa outputs, the models had **zero exact matches**. The more routine 114-case comparison group achieved **31 matches, 27.2%**.

Those differences matter because the sampled cases were never a random miniature of the entire corpus. I had selected them precisely where the first classifier struggled. Reporting 10.7% as the universal disagreement rate for all vaccine tweets would be misleading.

More importantly, insufficient context and uncertainty mean different things. A tweet can be perfectly understandable but contain no author opinion about vaccination. An author can explicitly say they are undecided about getting vaccinated. Or a fragment can be too truncated to determine what the author believes. These should not all be captured by whichever label a model finds most convenient.

The next experiment therefore changed the classification rules rather than simply merging the problematic categories until agreement looked better. Ontology v2 demanded evidence of **the author's own expressed indecision** before calling a tweet uncertain. It distinguished readable neutral content from genuinely insufficient context and asked the model not to mistake quoted statements or arguments about mandates for the author's vaccine stance.

In a controlled 60-case experiment, 57 outputs were structurally valid. Among them, 27 tweets that Qwen v1 had labeled uncertain became no-expressed-stance under v2. That was meaningful evidence that the wording of the task influenced classification. It still was not evidence that those new labels were objectively correct.

The broader lesson was uncomfortable but valuable. A sophisticated model can give a precise-looking answer to an imprecisely specified question. If two models are answering different versions of that question, their disagreement is not something to hide. It is a clue about where the research construct itself needs work.

### Research notes

The [frozen disagreement audit](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/STANCE_DISAGREEMENT_AUDIT.md) contains the full paired cross-tab and selection-stratum results; its [safe Actions status record](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39109108830) verifies the aggregate audit. The [v2 experiment protocol](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/ONTOLOGY_V2_EXPERIMENT.md) documents the revised classes. Neither cross-model agreement nor JSON validity is an accuracy measurement.
