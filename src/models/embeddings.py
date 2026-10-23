from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def cosine(self, a: str, b: str) -> float:
        va, vb = self.embed([a, b])
        return float(np.dot(va, vb))

    def cosine_matrix(self, a_texts: Sequence[str], b_texts: Sequence[str]) -> np.ndarray:
        return self.embed(a_texts) @ self.embed(b_texts).T