# Candidate model cards (selection, not proven corpus performance)

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
