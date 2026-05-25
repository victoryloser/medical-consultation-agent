"""
急症向量检索器：用 embedding 相似度作为关键词规则引擎的语义兜底链路。

调用方 (emergency_check_node) 只需:
    from app.rules.emergency_embedding import get_emergency_embedding_retriever
    retriever = get_emergency_embedding_retriever()
    hit, category, risk_level, score = retriever.check(text)
"""
import json
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

from app.config import get_config
from app.rag.embedding import embed_texts

COLLECTION_NAME = "emergency_cases"


class EmergencyEmbeddingRetriever:

    def __init__(self, cases_path: str, persist_dir: str, threshold: float = 0.75):
        self.threshold = threshold
        self.chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() == 0:
            self._build_index(cases_path)

    def _build_index(self, cases_path: str):
        path = Path(cases_path)
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            categories = json.load(f)

        ids, texts, metadatas = [], [], []
        for cat in categories:
            category = cat["category"]
            risk_level = cat["risk_level"]
            for i, example in enumerate(cat["examples"]):
                ids.append(f"{category}_{i}")
                texts.append(example)
                metadatas.append({"category": category, "risk_level": risk_level})

        embeddings = embed_texts(texts)
        batch = 100
        for i in range(0, len(ids), batch):
            self.collection.upsert(
                ids=ids[i:i + batch],
                embeddings=embeddings[i:i + batch],
                documents=texts[i:i + batch],
                metadatas=metadatas[i:i + batch],
            )

    def check(self, text: str) -> tuple[bool, str, str, float]:
        """
        返回 (is_hit, category, risk_level, score)
        score = 余弦相似度，超过 threshold 视为命中。
        """
        if not text.strip() or self.collection.count() == 0:
            return False, "", "", 0.0

        q_emb = embed_texts([text])[0]
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=1,
            include=["metadatas", "distances"],
        )
        distance = results["distances"][0][0]
        score = round(1 - distance, 4)
        meta = results["metadatas"][0][0]

        if score >= self.threshold:
            return True, meta["category"], meta["risk_level"], score
        return False, "", "", score

    def rebuild(self):
        """重建索引（更新 emergency_cases.json 后调用）。"""
        self.chroma.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        cfg = get_config()
        self._build_index(cfg.rules.emergency_cases_path)


@lru_cache(maxsize=1)
def get_emergency_embedding_retriever() -> EmergencyEmbeddingRetriever:
    cfg = get_config()
    return EmergencyEmbeddingRetriever(
        cases_path=cfg.rules.emergency_cases_path,
        persist_dir=cfg.rag.persist_dir,
        threshold=cfg.rules.emergency_embedding_threshold,
    )
