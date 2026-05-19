from typing import Optional
from pydantic import BaseModel


class ReferenceDoc(BaseModel):
    title: str
    source: str
    score: Optional[float] = None


class ConsultationResponse(BaseModel):
    report_id: str
    risk_level: str
    symptoms: list[str]
    department_suggestion: list[str]
    emergency_flag: bool
    advice: str
    safety_warnings: list[str]
    references: list[ReferenceDoc]
    final_report: str
    disclaimer: str = "本结果仅供健康咨询参考，不能替代医生诊断。"
