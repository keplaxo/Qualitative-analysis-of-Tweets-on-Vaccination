# The first thing that broke wasn't the model

*Research journal, Part 2 — 17 October 2026*

*Before I could compare vaccine sentiment models, I had to find a way of identifying the tweets without accidentally merging different records.*

Once I had settled on the research questions, the next step was to make the pipeline reproducible. I wanted GitHub to hold the code and audit trail, Kaggle to handle the private data and model runs, and neither platform to turn a research experiment into a casual upload of raw tweet records.

The dataset preflight identified two source files, master.csv and vaccination2.csv. That might sound like an unremarkable result. In practice, it meant confirming that the files could be read, that the relevant text and date fields existed, and that the notebook would not make assumptions about an English-language column that the observed schema did not contain. A small problem in data preparation, repeated a million times, is no longer a small problem.

The first 100-record smoke experiment then failed for a reason I had not been worrying about. Some source tweet IDs had been represented in scientific notation. Used as numeric keys, they could collapse into the same value and create collisions when predictions were joined back to their source records.

There is nothing particularly glamorous about losing records to the way a CSV reader interprets a large number. But it is exactly the sort of failure that makes a final chart unreliable if nobody catches it.

The fix was to stop trusting numeric source IDs as the canonical inference key. We used a SHA-256 digest of the exact text instead. Identical texts share a prediction key, while their original source occurrences can be accounted for separately. That distinction is important: deduplicating inference inputs saves computation, but it must not quietly erase repeated posts from a later analysis of recorded events.

The corrected smoke run completed. All 100 selected texts received the multilingual sentiment model prediction; 85 were screened as English and received an English-specific sentiment prediction plus exploratory DeBERTa stance prediction. The two sentiment models agreed on **56 of those 85 overlapping cases, 65.9%**.

That number was worth recording, but it was not an accuracy score. Two models can agree and both be wrong, particularly when trained on related social-media tasks. We had not independently adjudicated those records.

The next step was the 2,000-record development benchmark. The initial larger run was canceled, and the attempted resubmission brought a second operational lesson: a GitHub Action can successfully send a notebook to Kaggle while Kaggle is still waiting to run it. A submission receipt is not a model result.

The full benchmark later completed with **2,000 multilingual** and **1,666 English-screened** sentiment and stance outputs. It demonstrated that the data flow and model execution were feasible on a manageable development sample. The sample was deliberately year-covered, however, and not designed to estimate historical population proportions.

I came away from this first stage with a more useful definition of progress. A pipeline is not working simply because it exits without an exception. It is working when I can reconstruct which inputs were processed, which predictions exist, which checks passed and which questions the outputs can legitimately answer.

### Research notes

See the [initial repository commit](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/1bddd9be504e4c5e7957771a6b844ebb44178cc4), the [completed 100-case status run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39046209397) and the [canceled 2,000-case diagnostic](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39046808459). The full experimental contract is recorded in the [feature-branch workflow document](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/WORKFLOW.md).
