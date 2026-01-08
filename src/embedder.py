from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, eps, None)

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts)
        embs = np.asarray(embs, dtype=np.float32)

        embs = self._l2_normalize(embs)

        return embs
