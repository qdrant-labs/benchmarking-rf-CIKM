"""
Relevance feedback using OpenAI text embeddings.

Scores each document by cosine similarity between the document embedding and the
query embedding, using text-embedding-3-large. All texts are embedded in a single
API call and the query vector is normalised before computing dot products.
"""

from typing import Any

import numpy as np
from openai import OpenAI
from qdrant_relevance_feedback.feedback import Feedback

class OpenAIFeedback(Feedback):
    _MODEL = "text-embedding-3-large"

    def __init__(self, api_key: str, model_name: str = _MODEL, **kwargs: Any):
        self._model_name = model_name
        self._client = OpenAI(api_key=api_key, timeout=60.0, **kwargs)

    def score(self, query: str, responses: list[str]) -> list[float]:
        texts = [query] + responses
        # Sort by .index — API does not guarantee response order matches input order.
        data = sorted(
            self._client.embeddings.create(model=self._model_name, input=texts).data,
            key=lambda e: e.index,
        )

        matrix = np.array([e.embedding for e in data], dtype=np.float32)
        matrix = self._normalize_rows(matrix)

        query_vec = matrix[0]          # shape (d,)
        doc_matrix = matrix[1:]        # shape (n_docs, d)
        return (doc_matrix @ query_vec).tolist()

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
