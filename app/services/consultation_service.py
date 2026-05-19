import uuid
from typing import Optional

from app.agents.graph import get_consultation_graph
from app.agents.state import ConsultationState
from app.schemas.consultation import ConsultationResponse, ReferenceDoc


class ConsultationService:

    async def run(
        self,
        text: str,
        image_path: Optional[str] = None,
        model_provider: str = "auto",
    ) -> ConsultationResponse:
        initial_state: ConsultationState = {
            "user_text": text,
            "image_path": image_path,
            "image_analysis": None,
            "symptoms": [],
            "emergency_flag": False,
            "emergency_reason": None,
            "retrieved_docs": [],
            "risk_level": "unknown",
            "department_suggestion": [],
            "advice": "",
            "safety_warnings": [],
            "final_report": "",
            "errors": [],
        }

        graph = get_consultation_graph()
        result: ConsultationState = await graph.ainvoke(initial_state)

        docs = result.get("retrieved_docs") or []
        refs = [
            ReferenceDoc(
                title=d.get("title", ""),
                source=d.get("source", ""),
                score=d.get("score"),
            )
            for d in docs
            if d.get("title")
        ]

        return ConsultationResponse(
            report_id=_generate_report_id(),
            risk_level=result.get("risk_level", "unknown"),
            symptoms=result.get("symptoms") or [],
            department_suggestion=result.get("department_suggestion") or [],
            emergency_flag=result.get("emergency_flag", False),
            advice=result.get("advice", ""),
            safety_warnings=result.get("safety_warnings") or [],
            references=refs,
            final_report=result.get("final_report", "未能生成报告"),
        )


def _generate_report_id() -> str:
    return uuid.uuid4().hex[:12].upper()
