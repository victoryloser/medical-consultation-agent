from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_config


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    cfg = get_config()
    return SentenceTransformer(cfg.rag.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()
