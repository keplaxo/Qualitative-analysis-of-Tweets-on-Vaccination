# Pipeline operations and recovery record

**Audit date:** 10 November 2026. This document describes actual GitHub/Kaggle controller behavior, not an assurance that every scheduled GPU job finished.

## Repository and execution architecture

The [default master branch](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/tree/master) hosts registered GitHub Actions workflows and public documentation. The [feature/kaggle-model-benchmark-v1 branch](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/tree/feature/kaggle-model-benchmark-v1) hosts Python library modules, Kaggle scripts, orchestration controllers, notebooks and synthetic tests, under [draft PR #1](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/pull/1). It is **not merged**. The master workflow checks out the feature branch when executing, so the revision used in a run should be recorded before publication.

Pipeline dependencies are:

Private source CSVs → CPU index and 64 hash partitions → 1,000-text GPU capacity pilot → 64 sentiment GPU shards → 64 exploratory zero-shot stance GPU shards → private original-event join → 16 four-shard CPU aggregation waves → final research report.

The original-event join has a separate private backfill and was verified COMPLETE on 27 October. The pipeline's [full-corpus scheduler](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/blob/master/.github/workflows/vax-full-corpus-inference.yml) now uses three best-effort scheduled checks each hour, concurrency control, an explicit Kaggle secret, **one new GPU stage maximum per pass**, and **two outstanding GPU shards maximum**. Capacity predictions are compared with a conservative 14,500-second bound, below Kaggle's 18,000-second job allowance. A scheduled controller run is not a completed Kaggle notebook.

## Verification gates

**Action succeeded** means orchestration ran. **Kernel submitted** means a Kaggle job was accepted. **Kernel COMPLETE** means Kaggle ended the job. Only a **validated private checkpoint** additionally proves that the approved report is readable and matches stage, shard, expected input count, exact text-hash set, model predictions and privacy flags. Only after every required shard and event count is validated can a completed **full-corpus descriptive report** be asserted. Neither a green Action nor a valid JSON report establishes independent model accuracy.

The controller saves only verified contiguous shard counts and a source manifest fingerprint in the GitHub Actions cache, never tweet content. A manifest mismatch blocks reuse. Previously submitted notebooks are checked before submission to prevent replacement. Failed or canceled inference is not automatically rerun.

## Evidence by date

| Date | Direct evidence | Interpretation |
| --- | --- | --- |
| 17 October | [Corrected 100-case smoke](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39046209397) | Pilot completed after a text-ID repair |
| 17 October | [Canceled 2,000-case attempt](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/39046808459) | No valid checkpoint from that canceled version |
| 19 October | [Index and corpus setup](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/1f112fecc02238332c29814cfd8f85e22e88a4a3) | 2,285,081 raw rows; 1,944,361 distinct texts; 64 partitions |
| 27 October | [Read-only diagnostic](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/40474727433) | Private outputs found for sentiment 13/14 and event backfill |
| 27 October | [Controller recovery](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/40475556667) | Sentiment 00–14 validated; shard 15 **submitted** |
| 3 November | [Connectivity diagnostic](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41641762607) | Kaggle client operational, transient 429 errors seen on private report retrieval |
| 3 November | [v2 controller](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41642073524) | Blocked by inaccessible summary.json |
| 3 November | [v3 controller](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41642080886) | Blocked by inaccessible VACMA private report |
| 4 November | [Full-corpus controller](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41922232916) | 44 validated sentiment shards, 0 stance; 44 submitted |
| 5 November | [Controller checkpoint](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/41956954067) | 45 validated sentiment shards, 0 stance; shard 45 submitted |
| 10 November | [Original controller](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42745672807) | **64/64 sentiment shards verified; stance 00 submitted** |
| 10 November | [Accelerated controller](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42771438209) | **64/64 sentiment and 1/64 stance verified; stance 01 submitted** |
| 10 November | [Bounded overlap check](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42771840544) | Stance 01 was RUNNING when stance 02 was submitted; concurrent completed inference not yet measured |

## Failure analysis and repairs

**Scientific-notation IDs:** an initial read of source IDs could produce collisions. Exact-text SHA-256 keys replaced numeric identifiers in the inference join. This does not erase source occurrence weights.

**Canceled notebook:** the first 2,000-record attempt could not support a completed-output claim. Kaggle working directories are not automatically restored across versions; a resumed job requires actual private checkpoint recovery.

**Semantically incompatible categories:** DeBERTa and Qwen's 31/290 stance agreement was investigated with a frozen hash-join audit and ontologies v2/v3, rather than declaring either model correct. The limited Qwen studies remain preserved and separate from the mass sentiment pipeline.

**Nested output filenames:** a completed private Kaggle notebook report could be missed by an incorrect exact file-pattern query. Read-only diagnostics and regex-safe nested filename matching with bounded retries repaired controller progression ([27 October recovery](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/40475556667)).

**CLI startup and throttling:** early-November workflows required explicit installation of the Kaggle client dependency six ([repair commit](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/ddf2c0416bff0cd20b8d1542faaf8a07cd2279a1)). An authenticated diagnostic distinguished 429 rate-limited private report reads from compute failures; repeated polling of completed research stages was curtailed ([change](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/f5e97ca79f73bd4c442ddb0d57f5d2382a241dba)). A prior model result inaccessible through the API must not be re-described as a failed inference job.

## Throughput acceleration deployed 10 November

The principal operational delay was a sequential, hourly scheduler rather than a requirement of the immutable shard definitions. The active master workflow now targets minute 7, 27 and 47 each hour (best effort; GitHub may delay scheduled runs). The feature-branch controller supports at most **two active GPU shards**, but submits **at most one new GPU job per pass**. It never overwrites an existing private Kaggle notebook. Quota-related submission rejection is explicitly logged without treating it as a successful shard. Other failures and cancellations continue to block automatic replacement.

Validated shard counts are persisted only as a **contiguous prefix**. An out-of-order completed kernel can be checked without prematurely increasing the verified count. The source manifest fingerprint, exact SHA-256 joins, expected counts, stage/partition identity and approved privacy flags retain their existing gates. No source tweet text or individual predictions are placed in the public checkpoint.

The CPU reporting scheduler can now start a four-shard wave once its four stance shards are validated, even while later stance jobs run. This overlaps aggregation with inference but does not relax the final 16-wave integrity and original-event denominator checks. The first successful accelerated [GitHub run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42771438209) confirmed the first stance shard; a [second run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42771840544) observed stance 01 RUNNING and submitted stance 02 without reusing a completed kernel. **Actual throughput improvement, second-GPU concurrency and quota sustainability still require observation**. The [synthetic controller test run](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/runs/42771358795) passed.

## Safe recovery steps

Inspect the latest [full-corpus Action](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/actions/workflows/vax-full-corpus-inference.yml) and its controller markers FULL_CORPUS_CHECKPOINT, FULL_CORPUS_PROGRESS, FULL_CORPUS_WAIT_SHARD, FULL_CORPUS_GPU_SUBMITTED and FULL_CORPUS_CONTROLLER_BLOCKED. Check the private Kaggle notebook state and approved JSON report. Confirm manifest identity, exact SHA-256 counts and per-model coverage before advancing. Retry a transient *metadata read* within limits; never confuse that with retrying inference. Do not overwrite a registered private kernel or restart a canceled/failed notebook without a separate authorized recovery decision. Keep private stdout, source texts, row hashes and per-record outputs out of public logs.

## Immediate next gate

Verify private completion of **stance shard 01** and continue the 63 outstanding exploratory stance shards under quota gates. Run each four-shard CPU wave as soon as its inputs pass the existing validation gates, then complete all 16 waves and final synthesis. Independently address task-matched stance validation before publishing stance prevalence or health-behavior claims. The project is operationally advancing but not scientifically complete.

See [methods](RESEARCH_METHODS.md), [history](PROJECT_HISTORY.md) and [references](REFERENCES.md).
