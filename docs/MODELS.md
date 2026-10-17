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

Current outcome: No actual GPU inference results have been generated in this
repository. Passing unit tests establishes code-level sanity, not NLP quality.
