"""BM25 over the corpus. Its top passages for a question are the context UNREAL reads before the question."""

import bm25s
import Stemmer

STEMMER = Stemmer.Stemmer("english")


def _tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=False)


class BM25:
    def __init__(self, retriever: bm25s.BM25):
        self.retriever = retriever

    @classmethod
    def build(cls, texts: list[str]) -> "BM25":
        retriever = bm25s.BM25(k1=1.5, b=0.75)
        retriever.index(_tokenize(texts), show_progress=False)
        return cls(retriever)

    def save(self, path: str) -> None:
        self.retriever.save(path)

    @classmethod
    def load(cls, path: str) -> "BM25":
        return cls(bm25s.BM25.load(path))

    def search(self, questions: list[str], k: int) -> list[list[int]]:
        """Row indices of the top-k chunks for each question, best first. Chunks that share no term are left out."""
        k = min(k, self.retriever.scores["num_docs"])
        ids, scores = self.retriever.retrieve(_tokenize(questions), k=k, show_progress=False)
        return [[int(i) for i, s in zip(row_ids, row_scores) if s > 0] for row_ids, row_scores in zip(ids, scores)]
