2026-10-17: Initiate one private Kaggle GPU smoke benchmark (100 year-covered records, Qwen disabled). Dataset keplaxo/twitter-vaccination-dataset is mounted by Kaggle. This trigger does not contain credentials or tweet text.

2026-10-17 retry #2: Kaggle dataset is confirmed mounted at /kaggle/input/datasets/keplaxo/twitter-vaccination-dataset/master.csv. Fixed notebook to support both current and legacy layouts. Submit 100-tweet GPU smoke only.

2026-10-17 retry #3: fix scientifically notated duplicate source IDs (2.44E+17), use SHA-256 text fingerprint for inference records, regression tests passed. Resubmit capped 100-record smoke only.
