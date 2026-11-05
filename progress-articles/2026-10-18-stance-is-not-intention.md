# Vaccine stance is not vaccination intention

*Research journal, Part 4 — the October 2026 ontology revision*

*One of the least convenient findings in the project was also one of the most useful: I had been comparing outcomes that were related, but not the same.*

After the disagreement audit, I wanted to find a proper independently published reference dataset. I had no interest in improving a model by having another model declare its outputs correct. A published reference was necessary if I wanted to move beyond agreement and begin saying anything about validity.

The search exposed an awkward practical problem. The [VaxxHesitancy dataset](https://zenodo.org/records/7601328) was relevant and independently annotated, with categories such as pro-vaxx, anti-vaxx and vaccine hesitant. But the public files I could inspect held tweet IDs and annotations without the original tweet text. Direct inference would require lawfully retrieving that text, and the labels would still need careful reconciliation with the historical task.

I also examined the [VACMA-PUBLIC research resource](https://github.com/peregilk/VACMA-PUBLIC), which included a smaller number of text-bearing tweets with independently published annotation fields. Of 200 cases, 43 carried an intention-to-accept label and 18 an intention-to-reject label. That created a 61-case benchmark for something important—but it was **personal vaccination intention**, not the same thing as **an expressed stance toward vaccination as a general subject**.

We ran the existing v2 stance classifier against that limited cross-task proxy. Its strict mapping matched **12 of 61 records (19.7%)**, with macro-F1 **0.337**, and it returned a binary support/oppose outcome for just **18 of 61**. Those results were poor for the forced proxy alignment, but describing them as the accuracy of a six-class vaccine-stance model would have made a methodological error out of an already difficult experiment.

The distinction becomes clearer with ordinary language. Someone may say they intend to receive a vaccine without making a broad statement about vaccines. Another person may support vaccination in principle but express no personal plan. Past vaccination is not the same as future intention. A bare appointment or logistical announcement should not automatically be counted as advocacy.

I changed the model specification to reflect this. Ontology v3 has two separate axes. The first asks for the author's stance toward human vaccination, retaining the six stance categories. The second asks for the author's personal intention—accept, reject, undecided, not expressed or insufficient context. Each axis requires its own evidence, and neither one is inferred automatically from the other.

That conceptual repair is a success in the design of the research, not proof that v3 works better. There is another limitation worth stating plainly: I had already seen the 61 VACMA outcomes before refining v3. Any subsequent comparison on those same records is **post-hoc model development**. It cannot be presented as a clean independent test or a confirmed performance improvement.

I would rather be precise about that limitation than publish a flattering number generated from a circular comparison.

The next valid step is an untouched, text-accessible reference that has exactly the outcome I want to classify. Until that exists, the historical model can still describe its predictions and where it abstains, but its label frequencies should not be promoted into measured vaccine hesitancy.

### Research notes

See the [frozen v2 experiment and external-reference preflight](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/ONTOLOGY_V2_EXPERIMENT.md), the [v3 dual-axis development protocol](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/STANCE_INTENTION_V3_DEVELOPMENT.md), [VaxxHesitancy V2](https://zenodo.org/records/7601328) and the [VACMA-PUBLIC source](https://github.com/peregilk/VACMA-PUBLIC). The 12/61 comparison evaluates a limited intention proxy, **not** historical six-class stance accuracy.
