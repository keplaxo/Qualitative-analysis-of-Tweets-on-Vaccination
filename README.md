# Vaccination Tweet Sentiment & Stance Benchmark

Research code for reproducible, model-assisted analysis of vaccination-related
tweets spanning 2006-2019.

**Data policy:** This is a public code repository. Raw tweet texts, source
archives, tweet/user identifiers, licensed external benchmark corpora, model
weights, credentials and project outputs must remain in **private** research
storage. Only code, protocol documentation, and synthetic tests are committed.

## Current stage

A Kaggle-compatible GPU benchmark pipeline has been developed. No Kaggle
GPU inference, in-domain accuracy evaluation, or full-corpus classification has
been completed by this repository. Passing CI tests is not a scientific model
accuracy result.

## Main artifacts

- [Kaggle notebook](kaggle/vaccination_benchmark.ipynb): staged 2,000 to
  3,000-record comparison (2,000 year-covered + up to 1,000 enrichment records).
- [Methods](docs/METHODS.md): prespecified selection and validation rules.
- [Kaggle setup](docs/KAGGLE.md): private dataset mounting and GPU instructions.
- [Model cards](docs/MODELS.md): chosen open-weight model candidates.
- src/vaxbench/cli.py: sampling, English sentiment, multilingual sentiment,
  exploratory NLI stance, and Qwen semantic inference.
- src/vaxbench/triage.py: prioritize ambiguous tweets and routine controls for
  capped Qwen review.
- src/vaxbench/summarize.py: coverage and prediction-agreement statistics
  explicitly distinguished from accuracy.

## Quick local tests

Use Python 3.11 or later. From this repository root:

    python -m pip install -e . pytest
    python -m pytest -q

For Kaggle, see docs/KAGGLE.md. No human-labeled data are required for
these tests, which use **synthetic** messages. Historical project checkpoints
remain immutable.

## Research interpretation

Sentiment about an illness or policy is not a position on vaccination.
Automated agreement and traceable evidence substrings do not prove correctness.
No corpus prevalence, accuracy, or causal vaccine-safety claims should be
reported without an appropriate design and independent evidence.
