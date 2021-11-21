from elasticsearch import Elasticsearch
from typing import Dict, List, Optional
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
stop_words = set(stopwords.words("english"))

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoBERT


class CAsT_base:
    """Represents a session for the baseline approach

    An object of this class represents a session (conversation)
    with the baseline approach. This class can be used with the
    function "run_queries" to generate a testfile for trec_eval.

    Example usage:
    cast = CAsT_base(reranking=True, remove_stopwords=True)
    cast.query(q="What is the meaning of life?")

    Settings:
    * reranking: bool to use MonoBERT for reranking or not
    * remove_stopwords: bool to remove stopwords from query before further processing.

    """

    def __init__(
        self,
        reranking: bool = False,
        remove_stopwords: bool = True,
    ) -> None:
        self.INDEX_NAME = "cast_base"
        self.es = Elasticsearch()
        self.queries = []
        self.responses = []
        self.reranking = reranking
        self.reranker = MonoBERT() if reranking else None
        self.remove_stopwords = remove_stopwords

    def clear_context(self, clear_queries: bool = True, clear_responses: bool = True):
        if clear_queries:
            self.queries = []
        if clear_responses:
            self.responses = []

    def listToString(self, s: List):
        # initialize an empty string
        str1 = " "

        # return string
        return str1.join(s)

    def query(self, q: str) -> str:
        """
        Preprocessing query and scoring using bm25
        """
        stop_words = set(stopwords.words("english"))

        if self.remove_stopwords:
            tokens = word_tokenize(q)
            q_list = []
            for w in tokens:
                if w not in stop_words:
                    q_list.append(w)
            q = self.listToString(q_list)

        hits = (
            self.es.search(index=self.INDEX_NAME, q=q, _source=True, size=100)
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
            print("RERANKING")
            texts = [
                Text(hit.get("passage"), {"_id": hit.get("_id", "FAILED")}, 0)
                for hit in hits_cleaned
            ]

            reranked = self.reranker.rerank(Query(q), texts)
            hits_cleaned = [
                {"passage": hit.text, "_id": hit.metadata["_id"], "_score": hit.score}
                for hit in reranked
            ]

        if len(hits) > 0:
            print("Query: " + q)
            return hits_cleaned[:1000]
        else:
            return []


def run_queries(query_file: str, key: str, CAsT: object, run_id: str):
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


if __name__ == "__main__":
    path = "../eval/2020_automatic_evaluation_topics_v1.0.json"
    key = "raw_utterance"

    cast = CAsT_base(reranking=True, remove_stopwords=True)
    run_queries(path, key=key, CAsT=cast, run_id="Test")
