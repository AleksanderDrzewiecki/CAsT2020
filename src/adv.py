# T5 query expantion
import logging
import json
from typing import Dict, List, Optional
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# MonoT5 reranking
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5, DuoT5

# Stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
stop_words = set(stopwords.words("english"))


class CAsT:
    """Represents a session for the adv. approach

    An object of this class represents a session (conversation)
    with the adv. approach. This class can be used with the
    function "run_queries" to generate a testfile for trec_eval.

    Example usage:
    ae_cq0_cr0_rrF_base = CAsT(
        context_queries=0, context_responses=0, reranking=False, index_name="cast_base"
    )
    ae_cq0_cr0_rrF_base.query(q="What is the meaning of life?")

    Settings:
    * context_queries: the amount of previous queries to use for query reformulation
    * context_responses:  the amount of previous responses to use for query reformulation
    * reranking: bool to use MonoT5 for reranking or not
    * rerank_stage_two: bool to use DueT5 as a stage two reranker
        Note: only usable if reranking is set to True aswell.
    * remove_stopwords: bool to remove stopwords from query before further processing.
    * prev_query: the amount of previous queries to add to the end query for retrieval
        Note: Untested

    """

    def __init__(
        self,
        index_name: str = "cast_base",
        context_queries: int = 0,
        context_responses: int = 0,
        prev_query: int = 0,
        reranking: bool = False,
        rerank_stage_two: bool = False,
        remove_stopwords: bool = False,
    ) -> None:
        self.INDEX_NAME = index_name
        self.es = Elasticsearch()
        es_logger = logging.getLogger("elasticsearch")
        es_logger.setLevel(logging.WARNING)
        self.queries = []
        self.responses = []
        self.context_queries = context_queries
        self.context_responses = context_responses

        self.reranking = reranking
        self.reranker = MonoT5() if reranking else None
        self.rerank_stage_two = rerank_stage_two
        self.reranker_two = DuoT5() if rerank_stage_two else None
        self.remove_stopwords = remove_stopwords
        self.prev_query = prev_query
        self.tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")

    def clear_context(self, clear_queries: bool = True, clear_responses: bool = True):
        """Resets the context for the next topic"""
        if clear_queries:
            self.queries = []
        if clear_responses:
            self.responses = []

    def query(self, q: str) -> List[Dict]:
        """Returns up to 1000 responses for the query.

        Args:
        q: text query.

        returns: sorted list of responses. [{"_id": , "_score":, "passage"}]

        """

        if self.remove_stopwords:
            tokens = word_tokenize(q)
            q_list = []
            for w in tokens:
                if w not in stop_words:
                    q_list.append(w)
            q = " ".join(q_list)

        sep = " <sep>"
        qs = []
        if self.context_queries > 0 or self.context_responses > 0:
            for i in range(1, max(self.context_queries, self.context_responses) + 1):
                if i <= self.context_queries:
                    if len(self.queries) >= i:
                        qs.insert(0, self.queries[-i])

                if i <= self.context_responses:
                    if len(self.responses) >= i:
                        qs.insert(0, self.responses[-i])
        qs.append(q)

        input_ids = self.tokenizer(sep.join(qs), return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)

        query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.queries.append(query)  # * Adding reformated query to context

        if self.prev_query > 0:
            wanted = self.prev_query
            if wanted + 1 > len(self.queries):
                wanted = len(self.queries)
            query = "{} <sep> {}".format(
                " ".join([self.queries[-i] for i in range(1, wanted + 1)]), query
            )

        query = query.replace("[", "").replace("]", "")

        hits = (
            self.es.search(index=self.INDEX_NAME, q=query, _source=True, size=100)
            .get("hits", {})
            .get("hits")
        )

        hits_cleaned = [
            {
                "passage": hit.get("_source", {}).get("passage"),
                "_id": "MARCO_" + hit.get("_id")
                if hit.get("_source").get("origin") == "msmarco"
                else "CAR_" + hit.get("_id"),
                "_score": hit.get("_score", "FAILED"),
            }
            for hit in hits
        ]

        if self.reranking:
            texts = [
                Text(hit.get("passage"), {"_id": hit.get("_id", "FAILED")}, 0)
                for hit in hits_cleaned
            ]

            reranked = self.reranker.rerank(Query(query), texts)
            hits_cleaned = [
                {"passage": hit.text, "_id": hit.metadata["_id"], "_score": hit.score}
                for hit in reranked
            ]

            if self.rerank_stage_two:
                texts = [
                    Text(hit.get("passage"), {"docid": hit.get("_id", "FAILED")}, 0)
                    for hit in hits_cleaned[:50]
                ]
                reranked = self.reranker_two.rerank(Query(query), texts)
                hits_cleaned = [
                    {
                        "passage": hit.text,
                        "_id": hit.metadata["docid"],
                        "_score": hit.score,
                    }
                    for hit in reranked
                ]

        if len(hits) > 0:
            self.responses.append(hits_cleaned[0].get("passage"))
            return hits_cleaned[:1000]
        else:
            return []


def run_queries(query_file: str, key: str, CAsT: object, run_id: str) -> None:
    queries = json.load(open(query_file))
    if queries[0].get("turn", {})[0].get(key) is None:
        raise KeyError("Provided key: " + key + "is not a valid key for queryfile")
    total_num = len(queries)
    f = open(run_id + ".trec", "w")

    for i, topic in enumerate(queries):
        print("Topic: {}/{}".format(i + 1, total_num))
        CAsT.clear_context()
        topic_id = topic.get("number")
        for turn in topic.get("turn"):
            turn_id = turn.get("number")
            hits = CAsT.query(turn.get(key))
            for j, hit in enumerate(hits):
                f.write(
                    str(topic_id)
                    + "_"
                    + str(turn_id)
                    + "\t"
                    + "Q0"
                    + "\t"
                    + str(hit.get("_id"))
                    + "\t"
                    + str(j)
                    + "\t"
                    + str(hit.get("_score"))
                    + "\t"
                    + str(run_id)
                    + "\n"
                )
    f.close()


# Base
# Reranking FALSE
ae_cq0_cr0_rrF_base = CAsT(
    context_queries=0, context_responses=0, reranking=False, index_name="cast_base"
)
ae_cq3_cr0_rrF_base = CAsT(
    context_queries=3, context_responses=0, reranking=False, index_name="cast_base"
)
ae_cq0_cr3_rrF_base = CAsT(
    context_queries=0, context_responses=3, reranking=False, index_name="cast_base"
)
ae_cq3_cr3_rrF_base = CAsT(
    context_queries=3, context_responses=3, reranking=False, index_name="cast_base"
)
# Reranking TRUE
ae_cq0_cr0_rrT_base = CAsT(
    context_queries=0, context_responses=0, reranking=True, index_name="cast_base"
)
ae_cq3_cr0_rrT_base = CAsT(
    context_queries=3, context_responses=0, reranking=True, index_name="cast_base"
)
ae_cq0_cr3_rrT_base = CAsT(
    context_queries=0, context_responses=3, reranking=True, index_name="cast_base"
)
ae_cq3_cr3_rrT_base = CAsT(
    context_queries=3, context_responses=3, reranking=True, index_name="cast_base"
)


# d2q
# Reranking FALSE
ae_rc0_qr0_rrF_d2q = CAsT(
    context_queries=0, context_responses=0, reranking=False, index_name="cast_d2q"
)
ae_qc3_qr0_rrF_d2q = CAsT(
    context_queries=3, context_responses=0, reranking=False, index_name="cast_d2q"
)
ae_qc0_qr3_rrF_d2q = CAsT(
    context_queries=0, context_responses=3, reranking=False, index_name="cast_d2q"
)
ae_qc3_qr3_rrF_d2q = CAsT(
    context_queries=3, context_responses=3, reranking=False, index_name="cast_d2q"
)
# Reranking TRUE
ae_qc0_qr0_rrT_d2q = CAsT(
    context_queries=0, context_responses=0, reranking=True, index_name="cast_d2q"
)
ae_qc3_qr0_rrT_d2q = CAsT(
    context_queries=3, context_responses=0, reranking=True, index_name="cast_d2q"
)
ae_qc0_qr3_rrT_d2q = CAsT(
    context_queries=0, context_responses=3, reranking=True, index_name="cast_d2q"
)
ae_qc3_qr3_rrT_d2q = CAsT(
    context_queries=3, context_responses=3, reranking=True, index_name="cast_d2q"
)

# STOPWORD REMOVAL
ae_cq7_cr0_rrF_rsT_base = CAsT(
    context_queries=7,
    context_responses=0,
    remove_stopwords=True,
    reranking=False,
    index_name="cast_base",
)

# PREV_QUERY
ae_cq7_cr0_rrF_rsF_pq1_base = CAsT(
    context_queries=7,
    context_responses=0,
    prev_query=1,
    remove_stopwords=False,
    reranking=False,
    index_name="cast_base",
)


path = "../eval/2020_automatic_evaluation_topics_v1.0.json"
key = "raw_utterance"
run_queries(path, key=key, CAsT=ae_cq0_cr0_rrF_base, run_id="ae_cq0_cr0_rrF_base")
