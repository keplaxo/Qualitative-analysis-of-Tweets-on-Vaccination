# Two million records, without pretending to have infinite compute

*Research journal, Part 5 — 19 October 2026*

*Scaling the study turned out to be less about finding a bigger GPU and more about making the records, denominators and stages fit together.*

By 19 October, the experiments on a few thousand tweets had answered a useful question: the models could run on the historical data. They had also raised an uncomfortable second question: how much of what they produced could I trust?

I decided to expand the computational pipeline while preserving those limitations. That meant allowing the scalable sentiment models to process the full archive, keeping the exploratory stance work clearly labeled, and not launching millions of expensive Qwen calls after the disagreement findings.

The first step was to determine what “the full dataset” actually meant. The private CPU index read the two source CSVs and counted **2,285,081 source rows**. But only **1,944,361 exact tweet texts** were distinct. The difference—**340,720 duplicate occurrences**—was large enough to affect both compute cost and the interpretation of any later chart.

There are two sensible denominators here, and they answer different questions. A unique-text analysis asks how often different pieces of content receive a particular prediction. An original-event-weighted analysis asks how those predictions distribute across the observed source rows, including repeated text. Neither should be mistaken for a count of unique people, and neither makes the archive representative of an entire population.

I did not want to pay for classifying identical text more than once. The solution was to infer once per exact-text hash while reconstructing original occurrence weights in a separate private CPU process. That event-weight stage later completed, preserving the possibility of presenting both views honestly.

Next came the compute problem. Rather than put nearly two million texts into one Kaggle notebook and hope for the best, the index divided them into **64 deterministic hash shards**. We ran a 1,000-text GPU capacity probe to estimate model runtime. The first sentiment shard covered **30,365 unique texts**, producing 30,365 multilingual and 25,380 English-screened sentiment predictions. The wider schedule could then submit one job at a time, inspect its output, and advance only after the expected coverage and record identity checks passed.

This was deliberately conservative. The full schedule contains **64 sentiment jobs** followed by **64 separate exploratory stance jobs**. The intended report stage subsequently combines those outputs with the original-event weights through **16 CPU aggregation waves** and a final synthesis. Qwen was not included in the mass run. Its role had been to expose semantic ambiguity in the development experiments, not become a costly source of unvalidated labels for every tweet.

The scale is impressive on paper, but scale alone does not improve the validity of a classifier. If a model systematically mistakes neutral reporting for vaccine opposition, running it on another million tweets gives me a million more opportunities to make the same mistake.

So my target for this phase was narrower than a finished research conclusion. I wanted a complete, traceable model-generated corpus with known denominators, explicit failures, controlled compute use and the ability to resume without reprocessing what had already passed validation.

That is less dramatic than announcing an AI result from 2.3 million tweets. It is also a much more useful foundation for the actual analysis.

### Research notes

The [full-corpus index commit](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/1f112fecc02238332c29814cfd8f85e22e88a4a3), [shard scheduling commit](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/cb227e7bc42c644997ec40416b9a2368a66c405b) and [full-corpus protocol](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/feature/kaggle-model-benchmark-v1/docs/FULL_CORPUS_PIPELINE.md) describe indexing, model coverage, GPU stages and the separate original-event-weight stage. No independent target-specific vaccine-stance accuracy had been established by this scale-up.
