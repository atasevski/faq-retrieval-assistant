from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .embedder import Embedder


@dataclass
class RetrievalResult:
    id: str
    lang: str
    category: str
    question: str
    answer: str
    score: float


class Retriever:
    def __init__(self, faqs: List[Dict[str, Any]], faq_embeddings: np.ndarray):
        """
        faqs: list of dicts loaded from faqs.json
        faq_embeddings: normalized embeddings for faq questions, shape (N, D)
        """
        self.faqs = faqs
        self.embs = faq_embeddings

        if len(self.faqs) != self.embs.shape[0]:
            raise ValueError("Number of FAQs and number of embeddings must match.")

    def search(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 3,
        lang: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Returns top_k most similar FAQs to the query.
        If lang is provided (e.g. 'en' or 'mk'), restrict results to that language.
        """
        q = embedder.encode([query])[0]

        if lang is None:
            candidate_idx = np.arange(len(self.faqs))
        else:
            candidate_idx = np.array(
                [i for i, f in enumerate(self.faqs) if f.get("lang") == lang],
                dtype=int,
            )

        if candidate_idx.size == 0:
            return []

        cand_embs = self.embs[candidate_idx]
        scores = cand_embs @ q

        k = min(top_k, scores.shape[0])
        top_local = np.argsort(scores)[::-1][:k]
        top_global = candidate_idx[top_local]

        results: List[RetrievalResult] = []
        for i in top_global:
            f = self.faqs[i]
            results.append(
                RetrievalResult(
                    id=str(f.get("id", "")),
                    lang=str(f.get("lang", "")),
                    category=str(f.get("category", "")),
                    question=str(f.get("question", "")),
                    answer=str(f.get("answer", "")),
                    score=float((self.embs[i] @ q)),
                )
            )

        return results
