# Why I went back to old vaccine tweets

*Research journal, Part 1 — the project restart in October 2026*

*An old dataset, a more interesting question, and a decision not to confuse emotional tone with what people actually believe.*

I have had a vaccination tweet dataset sitting in the background for some time. It was collected during a very different period in online health discussion, before the COVID-19 pandemic transformed both vaccine conversations and the way those conversations are studied. Revisiting it in October 2026 seemed useful, but I was less interested in reproducing another sentiment plot than in asking whether we could interpret the discussions more carefully.

Sentiment analysis is an attractive starting point. Classify a sentence as positive, negative or neutral, put the results on a timeline, and you have something visually persuasive. The problem is that vaccination discourse is rarely that simple. Someone might be deeply alarmed by an infectious disease and strongly supportive of vaccines. Someone else might criticize a compulsory vaccination policy without opposing the vaccine. A news headline describing an adverse event might communicate neither endorsement nor rejection by the person sharing it.

A negative sentiment score cannot reliably tell these situations apart.

My original analytical direction had relied on more conventional lexicon-based sentiment methods, including VADER-style approaches. They remain useful for some purposes, but the question had changed. I wanted to examine the actual meaning of the text, with models more familiar with social-media language, and to see where those models disagreed rather than hide that disagreement inside a final label.

That led to three distinct targets. **Sentiment** would describe emotional tone. **Vaccination stance** would describe a position that the author actually expresses about vaccines or vaccination. **Personal vaccination intention** would describe the author's stated decision or plan to receive or refuse a vaccine. I would not assume that one of these could substitute for another.

I also made an early decision not to build a new human-labeling and approval operation around the project. It was partly practical, but it has a scientific consequence: an apparently convincing model-generated label is still a model-generated label. Without an appropriately matched independent reference dataset, I cannot call that label correct. The study would need to report uncertainty, abstentions and mismatches as part of the evidence, rather than as inconvenient noise.

The dataset itself would require a fresh look. The original files were historical, the time coverage was uneven, source IDs were not guaranteed to behave nicely, and English could not simply be assumed. The right first step was to inspect what was actually there, not announce a model architecture and work backwards.

I planned a staged experiment: establish a controlled data source, run a small reproducible benchmark, compare a multilingual and an English-specific sentiment model, add a deliberately cautious stance classifier, and use a more capable language model on a limited selection of difficult examples. Only then would I decide what was defensible at the scale of the full corpus.

I expected the hardest part to be the language models. In the end, some of the most important lessons came from data identity, the meaning of an uncertain label, and the awkward difference between a workflow saying “success” and a research stage actually being complete.

That is the point of documenting the process as it develops. The unglamorous parts are often where the work becomes more reliable.

### Research notes

The working [research protocol](../docs/RESEARCH_METHODS.md) specifies distinct task targets and explains why independent validation is still required. The historical data are held in the [approved Kaggle dataset](https://www.kaggle.com/datasets/keplaxo/twitter-vaccination-dataset). The current code implementation began with the [17 October 2026 initialization](https://github.com/keplaxo/Qualitative-analysis-of-Tweets-on-Vaccination/commit/1bddd9be504e4c5e7957771a6b844ebb44178cc4); this entry describes the preceding research decision, not the dataset's original collection date. For background, see [XLM-T](https://aclanthology.org/2022.lrec-1.27/) and [TimeLMs](https://aclanthology.org/2022.acl-demo.25/).
