from functools import lru_cache
from typing import Any

from app.config import get_config
from app.rag.vector_store import get_vector_store


class MedicalRetriever:

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.store = get_vector_store()

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.top_k
        if not query.strip():
            return []
        return self.store.query(query, top_k=k)

    def is_ready(self) -> bool:
        return self.store.count() > 0


@lru_cache(maxsize=1)
def get_retriever() -> MedicalRetriever:
    cfg = get_config()
    return MedicalRetriever(top_k=cfg.rag.top_k)
