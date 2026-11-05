# A green check is not the finish line

*Research journal, Part 6 — 27 October to 5 November 2026*

*The pipeline started moving again once I learned to distinguish a failed model run from a failed attempt to read its report.*

There is a peculiar frustration in automated research workflows: a job can finish its actual computation, save the correct result, and still look broken to the next stage because the orchestration software cannot retrieve the output.

That was part of the problem in late October. The full-corpus sentiment pipeline had started moving through the 64 indexed partitions, but the controller stopped when Kaggle's private output API intermittently failed to return a summary. A missing response was being treated as if the result itself might be missing.

A read-only diagnostic showed a more nuanced picture. Sentiment shards 13 and 14 were complete, their private report files could be found and parsed, and the separate original-event-weight reconstruction was also complete. The problem was not simply that model inference had failed. It was how the controller requested the output files and how it handled transient errors.

The repair made the file matching aware that Kaggle applies a regular expression to output paths, sometimes including a nested directory. We also added bounded retries for private report reads and kept public diagnostics limited to safe categories. Crucially, the fix did **not** rerun already completed inference just because the status check had been unreliable.

On **27 October**, the controller verified **15 sentiment shards, numbers 00 through 14**, and submitted shard 15. That was good progress, but the language matters: submitting a shard did not mean it was finished.

Early November brought another set of complications. A Kaggle CLI startup issue required an explicit dependency fix, and a sanitized diagnostic subsequently identified intermittent **HTTP 429** rate-limit responses while reading private reports. Other research controllers—for the earlier Qwen and v2/v3 experiments—were sometimes blocked by inaccessible summary files, even though those historical model outputs had previously been completed. We reduced redundant polling, repaired the dependency chain and kept the errors visible rather than manufacturing green results.

The full-corpus sentiment pipeline continued. The **4 November** run verified 44 completed sentiment shards and submitted shard 44. In the latest **5 November** controller output, the verified prefix rose to **45 completed sentiment shards, indices 00–44**, and the next shard, **45**, was submitted. The same checkpoint recorded **zero verified stance shards**.

There is an appealing shortcut at this stage: point to a green GitHub check and say the pipeline is done. But that would undermine the exact reproducibility I am trying to build. A green GitHub check can mean the controller ran successfully. A successful Kaggle submission means a model job was accepted. A Kaggle COMPLETE state means a job ended. Only after the private output report passes the expected model coverage, record count and hash integrity checks does a shard become a validated research checkpoint.

Even that does not establish that the model's sentiment or vaccine-stance prediction was scientifically correct.

So this is where the project stands on 5 November: the full source index is established, the limited benchmark and semantic development studies are preserved, and sentiment analysis has made substantial progress through the 64 partitions. But **the entire full corpus is not finished**. The exploratory stance stages, the aggregation waves and the final research report have not been verified complete. The task-matched external validation question is still open.

The next steps are not mysterious. Verify shard 45, complete the remaining sentiment partitions, process the separately gated exploratory stance partitions, reconcile original-event counts and only then review a full descriptive report. If the data cannot support a particular scientific claim, the report needs to say so.

Not every interesting project milestone is a breakthrough. Sometimes the useful milestone is being able to tell exactly what is done, what failed, and what a number actually means.

### Research notes

The [27 October recovery run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/40475556667) established the first 15 validated sentiment shards. The [3 November connectivity diagnostic](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41641762607) recorded rate limiting; [CLI repair](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/ddf2c0416bff0cd20b8d1542faaf8a07cd2279a1) and [polling reduction](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/f5e97ca79f73bd4c442ddb0d57f5d2382a241dba) followed. Compare the [4 November controller](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41922232916) with the [5 November controller](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41956954067). The later run recorded 45 validated sentiment shards, zero stance shards and shard 45 **submitted**, not completed.
