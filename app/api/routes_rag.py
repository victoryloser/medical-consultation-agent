from fastapi import APIRouter
from pydantic import BaseModel

from app.rag.retriever import get_retriever

router = APIRouter(prefix="/api", tags=["rag"])


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/rag/search")
def rag_search(req: RAGSearchRequest):
    retriever = get_retriever()
    results = retriever.search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}
