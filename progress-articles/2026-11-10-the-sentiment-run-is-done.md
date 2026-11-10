# The sentiment run is done. The bottleneck isn't.

*10 November 2026 · Research journal · Dr William Kane Olwit*

There's an odd kind of satisfaction in seeing a process that has been running for weeks finally reach the end of a queue. This morning, that happened with the sentiment arm of my vaccination tweet project. The controller had checked all 64 partitions of the historical corpus and confirmed that the corresponding private model outputs met the expected coverage and integrity checks.

That is a useful milestone, although it does not mean the study is finished. The two source files contain 2,285,081 original rows, which resolve to 1,944,361 distinct exact texts after deduplication. We infer once on each unique text, but keep the original occurrence information so the final analyses can describe the source archive without quietly discarding repeated posts. The sentiment stage is complete across those 64 partitions. Vaccination stance is a different task, with a different model and a harder interpretation problem.

The first stance shard has now passed the same operational checks. The next shard has been submitted. That progress is documented in the [10 November accelerated GitHub Actions run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42771438209). For anyone following the repository, the distinction matters: a green workflow check may tell us that GitHub ran the controller successfully. It cannot, on its own, tell us that every Kaggle model prediction has finished.

## The cost of being too careful

I built the first controller to err on the side of caution. It accepted one new GPU job at a time, checked its output and waited for the next scheduled pass before moving on. That helped us recover from a couple of unglamorous problems, including bad identifier handling, ambiguous Kaggle output paths and intermittent report-download failures. Most importantly, it avoided silently replacing private notebooks that had already completed.

But once the sentiment pipeline had stabilized, the waiting itself became an obstacle. There was no scientific reason to leave the next independent hash partition idle while the preceding one was still running. And the final CPU aggregation did not need to wait for the *last* stance shard when the first groups of four were already independently validated.

I have therefore changed the orchestration rather than the science. The GitHub controller now checks three times per hour and can manage up to two outstanding GPU shards, while submitting no more than one new job per pass. A verified count only advances across a contiguous sequence of completed and validated shards; finishing shard 03 before shard 02 cannot magically make shard 02 complete. As groups of four stance shards are validated, the matching private CPU aggregation waves can start in parallel with the remaining inference.

These changes passed the [synthetic controller tests](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42771358795) and the [first live run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42771438209). I have not yet measured a sustained speedup. Kaggle's available concurrency and GPU quota may still decide how much time we actually save. That is a throughput limitation worth recording rather than explaining away.

## The question the GPUs won't answer

A separate issue remains unresolved: a convincing-looking stance label is not necessarily a correct one. In an earlier deliberately selected comparison, DeBERTa and Qwen gave the same stance label to just 31 of 290 valid paired examples. That disagreement pushed me to distinguish an author's attitude toward vaccination from their personal intention to receive a vaccine, and from simple discussion of vaccine policy. These are not interchangeable research outcomes.

For that reason, the forthcoming full-corpus stance counts remain **exploratory model predictions**. They cannot responsibly be advertised as independently measured vaccine-hesitancy prevalence. A task-matched, untouched, text-accessible reference is still needed for stronger validity claims.

The immediate work is more straightforward: finish the remaining stance shards, verify their exact-hash coverage, assemble the original-event-weighted summaries, and produce the first complete descriptive figures. I am pleased to have finished the sentiment run. I'm more interested in whether the workflow can now finish the rest efficiently without asking the data to say more than it actually can.

*Technical record: [project history](../docs/PROJECT_HISTORY.md), [pipeline operations](../docs/PIPELINE_OPERATIONS.md), [research methods](../docs/RESEARCH_METHODS.md), [references](../docs/REFERENCES.md).*
