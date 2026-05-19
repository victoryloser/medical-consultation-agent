from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings

from app.config import get_config
from app.rag.embedding import embed_texts


class MedicalVectorStore:

    def __init__(self, persist_dir: str, collection_name: str):
        self.chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: list[dict[str, Any]]):
        """docs: [{"id": str, "text": str, "metadata": dict}]"""
        if not docs:
            return
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        embeddings = embed_texts(texts)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        q_emb = embed_texts([query_text])[0]
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "content": doc,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "score": round(1 - dist, 4),
                    "knowledge_type": meta.get("knowledge_type", ""),
                }
            )
        return output

    def count(self) -> int:
        return self.collection.count()


@lru_cache(maxsize=1)
def get_vector_store() -> MedicalVectorStore:
    cfg = get_config()
    return MedicalVectorStore(cfg.rag.persist_dir, cfg.rag.collection_name)
