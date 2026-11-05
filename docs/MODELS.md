# Candidate model cards (selection, not proven corpus performance)

**Current operations addendum — 5 November 2026:** The latest [full-corpus Actions run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41956954067) verified **45 of 64 sentiment shards (00–44)**, with **sentiment shard 45 submitted but not yet validated complete**; **0 of 64 exploratory stance shards** were verified by that controller checkpoint. The original index contains **2,285,081 source rows** and **1,944,361 distinct exact texts**. Full-corpus aggregation and a scientifically validated vaccine-stance prevalence estimate **are not complete**. The 3 November diagnostics found private-output API rate limiting and missing-accessible-summary failures in legacy Qwen/v2/v3 controllers; those retrieval failures do not by themselves erase earlier completed experiments. A green GitHub Action is not equivalent to a complete Kaggle model output.

**Maintained public documentation:** [README](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/master/README.md), [dated project history](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/master/docs/PROJECT_HISTORY.md), [research methods](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/master/docs/RESEARCH_METHODS.md), [operations and recovery](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/master/docs/PIPELINE_OPERATIONS.md), [references](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/master/docs/REFERENCES.md), and [blog-ready progress articles](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/tree/master/progress-articles). The material below retains the original stage-specific details and historical checkpoint language. Treat earlier statements of what was 'latest', 'pending', or 'not yet authorized' as dated research records.


- English sentiment: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
- Multilingual sentiment: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
- NLI stance comparator: https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0
- Contextual semantic adjudication: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

All are exploratory model candidates for these historical texts. The zero-shot
model is English-oriented and its stance predictions must not be interpreted
as validated target-specific author beliefs. Inspect rights, model-card
restrictions, published benchmarks and context limits before wider rollout.
The runtime records model commit hashes for each inference stage.

Verified execution (17 October 2026): a private Kaggle GPU smoke experiment has **completed** for 100 records. It produced 100 multilingual XLM-R sentiment predictions, 85 English RoBERTa sentiment predictions and 85 DeBERTa zero-shot stance predictions. RoBERTa/XLM-R agreed on 56 of 85 overlap records (65.9% agreement, **not accuracy**). Qwen was **not executed** in that smoke configuration. The 2,000-record full benchmark was resubmitted after cancellation; its latest submission log reported QUEUED, not completion. No independent in-domain model-accuracy evidence exists.

Model outputs are exploratory until tested against appropriately labeled, independent, task-matched data. English sentiment and generic vaccine stance run only for English-screened tweets; automatic language detection, NLI scores and evidence substrings are not ground truth. See [workflow status](WORKFLOW.md) and [methods](METHODS.md) for the execution/interpretation boundaries.
