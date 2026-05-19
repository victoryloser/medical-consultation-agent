from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_consultation import router as consultation_router
from app.api.routes_health import router as health_router
from app.api.routes_rag import router as rag_router
from app.config import get_config

cfg = get_config()

app = FastAPI(
    title=cfg.app.title,
    version=cfg.app.version,
    description="多 Agent 多模态医疗辅助问诊 Demo API（仅供学习演示，不替代医生诊断）",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(consultation_router)
app.include_router(rag_router)
app.include_router(health_router)


@app.on_event("startup")
async def startup():
    import logging
    logging.basicConfig(level=cfg.logging.level)
    logging.info("Medical Consultation Demo API started.")
