# Vaccination research journal — blog articles

These are first-person, publication-ready draft articles about the vaccination sentiment project. I have written them to be readable by clinical researchers, data scientists and curious non-specialists while keeping every outcome and limitation visible. Each article is independently readable, with direct research-note links for factual claims.

The article dates refer to **when the episode took place**, not a claim that the article was publicly published on that date. A final authorial and factual review is appropriate before publishing.

1. [Why I went back to old vaccine tweets](2026-10-06-why-revisit-vaccine-tweets.md) — project question, reasons for changing models, goals.
2. [The first thing that broke wasn't the model](2026-10-17-the-first-thing-that-broke.md) — public code, Kaggle preflight, ID collisions, first successes and cancellation.
3. [When two models could barely agree](2026-10-18-when-models-disagree.md) — the 31/290 result, sampling bias and uncertainty categories.
4. [Vaccine stance is not vaccination intention](2026-10-18-stance-is-not-intention.md) — ontology v2/v3, public reference data and honest validation.
5. [Two million records, without pretending to have infinite compute](2026-10-19-scaling-the-corpus.md) — deduplication, sharding, GPU constraints and original-event denominators.
6. [A green check is not the finish line](2026-11-05-a-green-check-is-not-finished.md) — late-October recovery, early-November throttling and the latest audited state.

## Publishing notes

The recurring style is first-person, clinically and scientifically literate, but relaxed enough to discuss things that broke and what I learned from them. Avoid inflated claims about artificial intelligence or a seamless research process. Keep operational milestones distinct from scientific validation. A model agreement percentage is not accuracy, and a structurally valid label is not a verified observation.

Never include private tweet text, usernames, source IDs, row-level data, original private model outputs or API credentials. Link the public evidence provided in each article. When reporting new progress, write a new date-stamped post and preserve earlier articles rather than silently rewriting the historical record.

Companion technical records: [project history](../docs/PROJECT_HISTORY.md), [methods](../docs/RESEARCH_METHODS.md), [operations](../docs/PIPELINE_OPERATIONS.md) and [reference register](../docs/REFERENCES.md).
