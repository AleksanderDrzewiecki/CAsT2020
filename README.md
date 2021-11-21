# CAsT2020

## Evaluation

The results (.trec) files are located under eval/runs.

**Nomenclature:**

* ae/me - automatic evaluation or manual evaluation. For ae, we use raw utterances, and manual_rewritten_utterances for me.
* cq# - context query, the number of previous queries that get prepended to the query before query reformulation.
* cr# - context responses, same as above but for system responses.
* rr[T/F] - reranking, T(true) if reranking is used, else F(false).
* rs[T/F] - remove stopwords, same as above but for stopword removal
* base/d2q - type of index.baseis the index with no processing of passages, whereas d2q has doc2query performed on a fraction of MS MARCO.

## Using our implementations

A requirements file *requirements.txt* is located at root level. Take care in using this. PyGaggle and Transformers may not play nicely with each other. If torch has problems running, install torch==1.7.0 even though PyGaggle wants a newer version.

### Indexing

Under src, there are two notebooks for indexing, one with docT5query and on without.

### Baseline and advanced approach

Under src there are two python files. *baseline.py* and *adv.py* these files are self contained.
Some example of how to use the code is shown in the main sections.

