from fastapi import APIRouter

from app.rag.retriever import get_retriever

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    retriever = get_retriever()
    return {
        "status": "ok",
        "rag_ready": retriever.is_ready(),
        "doc_count": retriever.store.count(),
    }
